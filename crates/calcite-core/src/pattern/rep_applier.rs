//! Descriptor-driven REP fast-forward applier.
//!
//! Phase 3b step 4 shipped [`apply_fill`] — the `BulkClass::Fill` arm of
//! the eventual replacement for `compile.rs::rep_fast_forward`'s hardcoded
//! per-opcode `match`. Step 5 adds [`apply_copy`] — the `BulkClass::Copy`
//! arm, MOVS-shaped: each iteration reads a byte (or a small block of
//! bytes) at a stepping source pointer and writes it at a stepping
//! destination pointer. Step 6 adds [`apply_read_only`] for LODS / CMPS /
//! SCAS — loops that walk pointers without mutating memory, with a
//! flag-conditioned early exit on CMPS/SCAS. Step 7 then rips out the
//! hardcoded path.
//!
//! ## What "descriptor-driven" means here
//!
//! The applier reads only structural metadata produced by the recogniser:
//!
//! - `descriptor.counter.property` — the slot whose decrement gates the
//!   loop; resolved to a slot index and read for the current iteration
//!   count.
//! - `descriptor.pointers[0]` — the destination pointer slot, its
//!   per-iteration `base_step`, and the direction-flag slot+bit. The
//!   applier never asks "is this STOSB or STOSW?"; the byte-vs-word width
//!   is whatever `base_step` says.
//! - `descriptor.writes[]` — one entry per byte the loop emits per
//!   iteration. The position of each entry in the vector is its offset
//!   *within* one iteration's destination slice. (Kiln emits paired
//!   addr/val slots in source order, sorted here by `assignment_index`;
//!   for a 2-byte word write the low byte is index 0, high byte is index
//!   1 — which is what the hardcoded path encodes by name as `lo` / `hi`.)
//! - `WriteEntry.addr_decomposition` — `Some((seg_property, ptr_property))`
//!   for the canonical `seg*16 + ptr` shape. The applier reads the seg
//!   slot once per call and the pointer slot once per call.
//! - `WriteEntry.val_expr` — the source of the byte. For `Fill` the
//!   classifier guarantees the value is independent of any pointer mirror;
//!   step 4 supports the bare `Var(name)` shape (resolves to a slot read)
//!   and panics on anything else, leaving the door open for future shapes
//!   without silently miscomputing.
//!
//! ## What it does NOT read
//!
//! - No property name as text. Strings only ever flow through
//!   `program.property_slots` and `state.state_var_index` lookups
//!   (whole-name equality, the same routing the hardcoded path uses).
//! - No opcode value. The dispatch in [`apply_fill`] is by structural
//!   role — counter slot, pointer slot, write entries — and the function
//!   would behave identically on a 6502 / brainfuck / non-emulator
//!   cabinet whose CSS happened to share this shape.
//! - No constant other than 16 (already absorbed by the recogniser into
//!   `addr_decomposition`) and the literal byte mask 0xFF (the structural
//!   "extract a byte from a wider integer slot" — same as the hardcoded
//!   path; not encoded as "this is x86").
//!
//! ## Where the byte mutations go
//!
//! Routes through the existing `compile::bulk_fill` /
//! `compile::bulk_store_byte` primitives. Those primitives own the
//! packed-cell + windowed-array + extended-map routing the hardcoded path
//! relies on; rewriting them here would be reinvention.

use crate::compile::{bulk_fill, bulk_store_byte, compute_sub_flags, ranges_overlap_virtual, CompiledProgram};
use crate::pattern::loop_descriptor::{BulkClass, LoopDescriptor};
use crate::state::State;
use crate::types::Expr;

/// Outcome of applying a descriptor to state. The harness consumes this to
/// decide whether to compare against the hardcoded path; future callers
/// (step 7) will simply ignore non-`Applied` outcomes and let the
/// hardcoded path keep running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ApplyOutcome {
    /// Applier ran successfully, mutated state, charged cycles.
    /// `iterations` is the number of iterations the applier collapsed.
    Applied { iterations: i32 },
    /// The cabinet's own outer guard predicate evaluates false on this
    /// tick — the CSS already produced the correct post-state and
    /// calcite has nothing to do. Dispatcher silently bails.
    ///
    /// No applier produces this in Phase 2; Phase 3 Task 3.4 (the
    /// precondition wart) wires it up.
    PreconditionNotMet,
    /// The applier understands the shape but bulk application cannot
    /// reproduce what per-tick evaluation would compute — e.g. a Copy
    /// whose SOURCE bytes exist only in compiled dispatch (ROM regions
    /// with no backing cells), which `state.read_mem` cannot resolve.
    /// Per-tick CSS evaluation IS correct for these; the dispatcher
    /// bails silently (with a diag counter) and the loop runs at one
    /// iteration per tick. Distinct from `Unsupported` (a recogniser/
    /// applier gap, which panics): here falling back is the *correct*
    /// semantics, and such loops are structurally rare and short (a
    /// boot loader copying a parameter table out of ROM, not a hot
    /// framebuffer path).
    PerIterFallback(&'static str),
    /// The recogniser produced a descriptor but the applier doesn't
    /// handle this shape (e.g., `BulkClass::PerIter`, or a Copy shape
    /// without indirect-read decomposition). The dispatcher PANICS
    /// with cabinet state; per-iter CSS fallback is not acceptable
    /// for REP loops because the 10-1000x slowdown makes carts
    /// unusable.
    Unsupported(&'static str),
}

/// Whether an applier should also commit the post-loop state vars
/// (DI/SI/CX/IP/cycleCount/flags), or only mutate memory.
///
/// `MemoryOnly` is what the dual-execute harness uses: the harness leaves
/// state-var commits to the hardcoded path (which runs on the real state
/// in parallel), and only diffs memory + extended after the applier runs
/// on the clone.
///
/// `Full` is what step 7's flag-flip uses: the applier becomes the only
/// source of truth for the post-tick state vars too, replacing every
/// observable side effect of the hardcoded path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommitMode {
    /// Only mutate `state.memory` / `state.extended`. Used by the dual
    /// harness — the hardcoded path concurrently commits state vars on
    /// the real (non-clone) state.
    MemoryOnly,
    /// Mutate memory AND commit DI/SI/CX/IP/cycleCount/flags exactly as
    /// the hardcoded path would. Used by `CALCITE_REP_GENERIC=1` callers
    /// that have replaced the hardcoded path entirely.
    Full,
}

// ---------------------------------------------------------------------------
// Per-iter cycle cost.
// ---------------------------------------------------------------------------
//
// The per-iteration cycle cost is **structurally derived** by the
// recogniser and stored on the descriptor as `per_iter_cycles:
// Option<CycleCharge>` — the cycle-counter slot's own name plus the
// per-opcode literal. See
// `pattern/loop_descriptor.rs::extract_per_iter_cycles` for the
// structural rule (the most-populated dispatch family member whose
// per-key bodies have shape `Calc(Add(Var(X), Literal(K)))` with X the
// same opaque slot reference across keys).
//
// When the recogniser couldn't extract a value, the applier returns
// `Unsupported("per_iter_cycles not extracted")` and the dispatcher
// panics — charging a fabricated cost would break cabinets whose
// progression is gated on cycle-derived timers.

// (Removed `read_prefix_len`: the IP extra-advance addend is now resolved
// from the descriptor's structurally-captured `ip_extra_advance_slot`.
// See `LoopDescriptor::ip_extra_advance_slot` and the resolution in
// `commit_ip_and_cycles` below.)

/// Look up a property name's current value, using the same routing the
/// compiler gives reads of that name (`compile.rs::resolve_property`):
///
/// 1. Buffer-copy names (`is_buffer_copy`) skip the slot table — the
///    engine compiles those reads as loads of the committed state var,
///    so we resolve them via the canonical bare name (`to_bare_name`
///    strips the `--` and `__N` buffer prefixes). This is what makes
///    the cabinet's prior-tick pointer mirrors (the `self_property` of
///    a PointerEntry) resolvable here.
/// 2. Every other name tries the compiled `property_slots` (current-tick
///    slot view) first, then the state-var table by canonical name.
///
/// `to_bare_name` (not `property_to_address`) is deliberate: the address
/// map is a thread-local populated on the loading thread, and the
/// debugger runs the engine on worker threads where it's empty.
/// Matching the compiler's routing is what keeps the applier's view of a
/// slot equal to what the cabinet's own compiled reads would see —
/// i.e. what Chrome computes. Returns `None` if no table has the name.
fn read_prop(
    program: &CompiledProgram,
    state: &State,
    slots: &[i32],
    name: &str,
) -> Option<i32> {
    if !crate::eval::is_buffer_copy(name) {
        if let Some(&s) = program.property_slots.get(name) {
            return Some(slots[s as usize]);
        }
    }
    state.get_var(crate::eval::to_bare_name(name))
}

/// Resolve a `val_expr` to its current i32 value, structurally.
///
/// Step 4 supports the bare `Var(name)` shape only — the recogniser's
/// `BulkClass::Fill` classification guarantees the value is independent
/// of any pointer mirror, and on doom8088 the cabinet emits STOS as a
/// bare slot read of AL/AX. Future shapes (e.g. structural byte-extract
/// of a wider register) extend this matcher; bailing with
/// `Unsupported` keeps unmatched shapes routed through the hardcoded
/// path until they're handled.
fn resolve_fill_value(
    val_expr: &Expr,
    program: &CompiledProgram,
    state: &State,
    slots: &[i32],
) -> Option<i32> {
    match val_expr {
        Expr::Var { name, fallback: None } => read_prop(program, state, slots, name),
        Expr::Var { name, fallback: Some(_) } => {
            // A var with a fallback expression — kiln does emit these
            // for some slot reads. The fallback only triggers if the
            // primary slot is missing; in a fully-loaded cabinet the
            // primary always resolves, so reading the slot is correct.
            // If it doesn't resolve, returning None forces the caller
            // to bail.
            read_prop(program, state, slots, name)
        }
        // Step 4 deliberately handles only the bare-Var shape. STOSW on
        // doom8088 emits two writes whose val_exprs are bare `Var(--AL)`
        // and `Var(--AH)` (kiln has dedicated AL/AH slots that mirror
        // AX). Cabinets that emit `(AX & 0xFF)` and `((AX >> 8) & 0xFF)`
        // calc shapes will need their own arm here — bail with
        // `Unsupported` until that arm lands.
        _ => None,
    }
}

/// Apply a `BulkClass::Fill` descriptor to state. The structural contract:
///
/// 1. Resolve counter, destination pointer, direction flag.
/// 2. For each `WriteEntry`, resolve its address decomposition's seg+ptr
///    slots and its `val_expr`. Position-in-vec is the byte offset within
///    one iteration.
/// 3. For each iteration `i ∈ [0, n)`, for each write entry `k`, write
///    `byte_k` at `seg*16 + pointer + signed_step*i + k`.
///
/// Returns the iteration count so the caller can charge cycles. Mirrors
/// what `compile::rep_fast_forward` does, but driven by descriptor
/// structure instead of a `match opcode { 0xAA => ..., 0xAB => ... }`.
///
/// This function is reentrant and pure of side effects beyond the state
/// mutations it performs — no env-var reads, no global state.
pub(crate) fn apply_fill(
    descriptor: &LoopDescriptor,
    program: &CompiledProgram,
    state: &mut State,
    slots: &[i32],
) -> ApplyOutcome {
    apply_fill_with_commit(descriptor, program, state, slots, CommitMode::MemoryOnly)
}

/// Variant of [`apply_fill`] that optionally commits post-loop state vars
/// (DI/CX/IP/cycleCount). When `commit == CommitMode::Full`, the applier
/// reproduces every observable side effect of the hardcoded `rep_fast_forward`
/// path for STOS-shape loops; when `MemoryOnly`, only memory is mutated
/// (existing dual-harness contract).
pub(crate) fn apply_fill_with_commit(
    descriptor: &LoopDescriptor,
    program: &CompiledProgram,
    state: &mut State,
    slots: &[i32],
    commit: CommitMode,
) -> ApplyOutcome {
    if descriptor.bulk_class != BulkClass::Fill {
        return ApplyOutcome::Unsupported("not Fill class");
    }
    if let Some(pre) = descriptor.precondition.as_ref() {
        if !crate::pattern::loop_descriptor::evaluate_precondition(pre, program, state, slots) {
            return ApplyOutcome::PreconditionNotMet;
        }
    }
    if descriptor.per_iter_cycles.is_none() {
        return ApplyOutcome::Unsupported("per_iter_cycles not extracted");
    }
    if commit == CommitMode::Full && descriptor.ip_extra_advance_slot.is_none() {
        return ApplyOutcome::Unsupported("ip_extra_advance_slot not captured");
    }
    // Counter — must be present, must resolve, must be > 0. CX<=0 is the
    // "no work to do" case the hardcoded path already filters before
    // calling us; the dual harness only reaches here after that filter,
    // so we treat it as Unsupported (caller bails to hardcoded path).
    let Some(counter) = descriptor.counter.as_ref() else {
        return ApplyOutcome::Unsupported("no counter");
    };
    let Some(n) = read_prop(program, state, slots, &counter.property) else {
        return ApplyOutcome::Unsupported("counter unresolved");
    };
    if n <= 0 {
        return ApplyOutcome::Applied { iterations: 0 };
    }

    // A Fill needs exactly one destination pointer (the byte sink). Two
    // pointers would be a Copy-shape (one source, one dest) — the
    // classifier wouldn't emit `Fill` in that case, but we double-check
    // structurally.
    if descriptor.pointers.len() != 1 {
        return ApplyOutcome::Unsupported("Fill expects exactly one pointer");
    }
    let ptr_entry = &descriptor.pointers[0];
    let Some(ptr_value) = read_prop(program, state, slots, &ptr_entry.self_property) else {
        return ApplyOutcome::Unsupported("pointer self_property unresolved");
    };
    let Some(flags) = read_prop(program, state, slots, &ptr_entry.flag_property) else {
        return ApplyOutcome::Unsupported("flag_property unresolved");
    };
    let df_active = ((flags >> ptr_entry.flag_bit) & 1) != 0;
    let step: i32 = ptr_entry.base_step;
    let signed_step: i32 = if df_active { -step } else { step };

    // Pre-compute (seg, val) for each write entry. Bail early on any
    // missing structural piece.
    if descriptor.writes.is_empty() {
        return ApplyOutcome::Unsupported("Fill expects at least one write");
    }
    if descriptor.writes.len() != step as usize {
        // Structural sanity: the per-iteration byte-count emitted as
        // writes should match the pointer's stride. If they don't, the
        // recogniser emitted an unexpected shape and the position-as-
        // offset assumption is unsafe. Bail.
        return ApplyOutcome::Unsupported(
            "writes.len() != base_step (per-iter byte count mismatch)",
        );
    }

    let mut prepared: Vec<(i64, u8)> = Vec::with_capacity(descriptor.writes.len());
    for w in &descriptor.writes {
        let Some((seg_prop, _ptr_prop)) = w.addr_decomposition.as_ref() else {
            return ApplyOutcome::Unsupported("write has no addr_decomposition");
        };
        let Some(seg) = read_prop(program, state, slots, seg_prop) else {
            return ApplyOutcome::Unsupported("seg_property unresolved");
        };
        let Some(byte_value) = resolve_fill_value(&w.val_expr, program, state, slots) else {
            return ApplyOutcome::Unsupported("val_expr shape not yet supported");
        };
        let seg_base = (seg as i64) * 16;
        prepared.push((seg_base, (byte_value & 0xFF) as u8));
    }

    // Pointer wrap check (16-bit segment offset). Each iteration writes
    // `step` bytes starting at `pointer + i*signed_step`. Forward (DF=0):
    //   range = [pointer, pointer + n*step)
    // Reverse (DF=1):
    //   range = [pointer - (n-1)*step, pointer + step)
    let n64 = n as i64;
    let step64 = step as i64;
    let (off_lo, off_hi) = if df_active {
        (ptr_value as i64 - (n64 - 1) * step64, ptr_value as i64 + step64)
    } else {
        (ptr_value as i64, ptr_value as i64 + n64 * step64)
    };
    if off_lo < 0 || off_hi > 0x10000 {
        return ApplyOutcome::Unsupported("pointer would wrap past 16-bit segment");
    }

    // Virtual-region overlap check: bulk writes through state.memory
    // can't substitute for cabinet-emitted dispatch over BIOS / ROM-disk
    // / packed-broadcast windows — but per-tick evaluation CAN: it runs
    // the cabinet's own write cascade, which is the reference semantic
    // for such writes (on a rom cabinet a store into the disk window
    // simply matches no write rule and drops, exactly as Chrome would
    // evaluate it — Windows 1.01's WRITE.EXE hits this via Corduroy's
    // INT 13h AH=03h path on a read-only floppy, 2026-07-13). Fall back
    // per-iter instead of refusing; the diag counter keeps it visible.
    let seg_for_range = prepared[0].0; // all writes share the same seg in Fill
    let dst_lo_linear = seg_for_range + off_lo;
    let dst_hi_linear = seg_for_range + off_hi;
    if ranges_overlap_virtual(state, dst_lo_linear, dst_hi_linear - dst_lo_linear) {
        return ApplyOutcome::PerIterFallback(
            "destination range overlaps virtual region",
        );
    }

    // Hot loop. Two structural shapes:
    //   - Single write per iter (step=1): collapse the entire loop into
    //     one bulk_fill call (the hardcoded STOSB path's optimisation).
    //   - Multiple writes per iter: walk per-iteration, per-write. The
    //     hardcoded STOSW path also does this; bulk_fill can't be used
    //     because the bytes within an iteration are *different*.
    let iterations = n;
    if descriptor.writes.len() == 1 {
        let (seg_base, byte) = prepared[0];
        let dst_start = seg_base + off_lo;
        let dst_count = (off_hi - off_lo) as usize;
        bulk_fill(state, dst_start, dst_count, byte);
    } else {
        // Multi-write per iter (e.g. STOSW: low byte then high byte).
        // For each iteration, write each prepared byte at the
        // structurally-implied offset (position-in-vec). Direction only
        // affects the iteration step — within a single iteration the
        // bytes are written in the order the recogniser sorted them
        // (low-to-high in source order, which matches the hardcoded
        // STOSW path's `lo` / `hi` ordering).
        for i in 0..n64 {
            let iter_off = i * (signed_step as i64);
            let iter_base = (ptr_value as i64) + iter_off;
            for (k, (seg_base, byte)) in prepared.iter().enumerate() {
                let addr = seg_base + iter_base + k as i64;
                bulk_store_byte(state, addr, *byte);
            }
        }
    }

    if commit == CommitMode::Full {
        // Pointer commit (single pointer, 16-bit wrap).
        let new_ptr = (ptr_value + n * signed_step) & 0xFFFF;
        commit_pointer(state, &ptr_entry.property, new_ptr);
        // Counter commit: drains to zero.
        commit_counter(state, &counter.property, 0);
        // IP advance and cycle charge.
        commit_ip_and_cycles(program, state, slots, descriptor, n);
    }

    ApplyOutcome::Applied { iterations }
}

/// Apply a `BulkClass::Copy` descriptor to state. The structural contract:
///
/// 1. Resolve counter, identify the destination and source pointers.
///    The destination pointer is the one named in every write entry's
///    `addr_decomposition.pointer_property`; the source pointer is the
///    one named in every write entry's `val_indirect_read.pointer_property`.
///    Both names must resolve to entries in `descriptor.pointers` so we
///    can find each pointer's `base_step`, `flag_property`, and `flag_bit`.
/// 2. Resolve segment slots: destination from `addr_decomposition.seg_property`,
///    source from `val_indirect_read.seg_property`. Both must be present
///    (`None` from the recogniser means the address shape was too messy
///    to decompose — bail to hardcoded path).
/// 3. For each iteration `i ∈ [0, n)`, for each write entry `k`:
///       byte = state.read_mem(src_seg*16 + src_ptr + signed_step*i + k)
///       bulk_store_byte(dst_seg*16 + dst_ptr + signed_step*i + k, byte)
///    The `signed_step` is the same for both pointers — MOVS-shape loops
///    advance both pointers by the same direction-flag-multiplied step,
///    and the descriptor's two pointer entries structurally describe
///    that (same flag_property, same flag_bit, same base_step). We
///    verify those structurally rather than assume.
///
/// Per-iter, per-write `state.read_mem` calls match the hardcoded path's
/// MOVS shape exactly — the fetch routes through packed cells, windowed
/// byte arrays, and the extended map without any of that knowledge
/// leaking up here. Overlap semantics are preserved by the
/// read-then-write iteration order: forward direction with overlapping
/// ranges (e.g. dst = src + 1) sees the source bytes after they've been
/// rewritten by earlier iterations, exactly as the hardcoded path
/// (and real x86 MOVS) does.
///
/// Cardinal-rule shape: like [`apply_fill`], the applier reads zero
/// property names as text. The `descriptor.pointers` lookup is
/// whole-name string equality (a HashMap lookup, not a substring
/// inspection), and the position-as-byte-offset within an iteration is
/// a structural fact about the write vector's order — the same as in
/// `apply_fill`.
///
/// Returns the iteration count so the caller can charge cycles.
pub(crate) fn apply_copy(
    descriptor: &LoopDescriptor,
    program: &CompiledProgram,
    state: &mut State,
    slots: &[i32],
) -> ApplyOutcome {
    apply_copy_with_commit(descriptor, program, state, slots, CommitMode::MemoryOnly)
}

/// Variant of [`apply_copy`] that optionally commits state vars
/// (DI/SI/CX/IP/cycleCount). See [`apply_fill_with_commit`].
pub(crate) fn apply_copy_with_commit(
    descriptor: &LoopDescriptor,
    program: &CompiledProgram,
    state: &mut State,
    slots: &[i32],
    commit: CommitMode,
) -> ApplyOutcome {
    if descriptor.bulk_class != BulkClass::Copy {
        return ApplyOutcome::Unsupported("not Copy class");
    }
    if let Some(pre) = descriptor.precondition.as_ref() {
        if !crate::pattern::loop_descriptor::evaluate_precondition(pre, program, state, slots) {
            return ApplyOutcome::PreconditionNotMet;
        }
    }
    if descriptor.per_iter_cycles.is_none() {
        return ApplyOutcome::Unsupported("per_iter_cycles not extracted");
    }
    if commit == CommitMode::Full && descriptor.ip_extra_advance_slot.is_none() {
        return ApplyOutcome::Unsupported("ip_extra_advance_slot not captured");
    }
    let Some(counter) = descriptor.counter.as_ref() else {
        return ApplyOutcome::Unsupported("no counter");
    };
    let Some(n) = read_prop(program, state, slots, &counter.property) else {
        return ApplyOutcome::Unsupported("counter unresolved");
    };
    if n <= 0 {
        return ApplyOutcome::Applied { iterations: 0 };
    }

    // A Copy needs exactly two pointers: a destination and a source.
    // Anything else is shape we don't recognise — bail.
    if descriptor.pointers.len() != 2 {
        return ApplyOutcome::Unsupported("Copy expects exactly two pointers");
    }
    if descriptor.writes.is_empty() {
        return ApplyOutcome::Unsupported("Copy expects at least one write");
    }

    // Identify destination and source pointers structurally from the
    // write entries. Every write must agree on both pointer names — a
    // canonical MOVS-shape loop has all writes targeting the same dst
    // and reading from the same src. Disagreement signals a shape we
    // don't yet handle.
    let first_dst_ptr = descriptor.writes[0]
        .addr_decomposition
        .as_ref()
        .map(|(_, p)| p.as_str());
    let first_src = descriptor.writes[0]
        .val_indirect_read
        .as_ref()
        .map(|ir| ir.pointer_property.as_str());
    let (Some(dst_ptr_name), Some(src_ptr_name)) = (first_dst_ptr, first_src) else {
        return ApplyOutcome::Unsupported(
            "Copy needs addr_decomposition + val_indirect_read on first write",
        );
    };
    if dst_ptr_name == src_ptr_name {
        // The dst and src must be different stepping pointers.
        return ApplyOutcome::Unsupported("Copy dst and src pointer names match");
    }
    for w in &descriptor.writes {
        let Some((_, p)) = w.addr_decomposition.as_ref() else {
            return ApplyOutcome::Unsupported("Copy write missing addr_decomposition");
        };
        if p != dst_ptr_name {
            return ApplyOutcome::Unsupported("Copy writes disagree on dst pointer");
        }
        let Some(ir) = w.val_indirect_read.as_ref() else {
            return ApplyOutcome::Unsupported("Copy write missing val_indirect_read");
        };
        if ir.pointer_property != src_ptr_name {
            return ApplyOutcome::Unsupported("Copy writes disagree on src pointer");
        }
    }

    // Resolve the two pointer entries by name. Whole-name equality —
    // identical to how `read_prop` resolves a slot.
    let dst_entry = descriptor
        .pointers
        .iter()
        .find(|p| p.self_property == dst_ptr_name);
    let src_entry = descriptor
        .pointers
        .iter()
        .find(|p| p.self_property == src_ptr_name);
    let (Some(dst_entry), Some(src_entry)) = (dst_entry, src_entry) else {
        return ApplyOutcome::Unsupported(
            "Copy pointer names not found in descriptor.pointers",
        );
    };

    // MOVS-shape: both pointers advance by the same signed step
    // (same direction-flag bit, same base_step). Structurally enforce
    // that — if the descriptor says the two pointers move
    // independently, the cabinet is doing something we don't yet
    // model and the hardcoded path should keep the wheel.
    if dst_entry.base_step != src_entry.base_step {
        return ApplyOutcome::Unsupported("Copy pointers have different base_step");
    }
    if dst_entry.flag_property != src_entry.flag_property
        || dst_entry.flag_bit != src_entry.flag_bit
    {
        return ApplyOutcome::Unsupported(
            "Copy pointers gated by different direction flags",
        );
    }
    let step: i32 = dst_entry.base_step;
    if descriptor.writes.len() != step as usize {
        return ApplyOutcome::Unsupported(
            "writes.len() != base_step (per-iter byte count mismatch)",
        );
    }

    // Resolve segment slots and pointer values. For Copy we need both
    // segments to be cleanly decomposed — `seg_property: None` from
    // the recogniser means the source address expression had a shape
    // we couldn't simplify, and per-iter raw-Expr evaluation isn't
    // wired up yet.
    let dst_seg_name = descriptor.writes[0]
        .addr_decomposition
        .as_ref()
        .map(|(s, _)| s.as_str())
        .unwrap();
    let src_ir0 = descriptor.writes[0].val_indirect_read.as_ref();
    let src_seg_name_opt = src_ir0.and_then(|ir| ir.seg_property.as_deref());
    let Some(src_seg_name) = src_seg_name_opt else {
        return ApplyOutcome::Unsupported(
            "Copy val_indirect_read has no seg decomposition",
        );
    };
    let src_seg_scaled = src_ir0.map(|ir| ir.seg_times_sixteen).unwrap_or(false);
    // Every write must agree on its decomposed segment slots too —
    // including whether the captured base was ×16-shaped.
    for w in &descriptor.writes {
        let dst_seg = w.addr_decomposition.as_ref().map(|(s, _)| s.as_str()).unwrap();
        if dst_seg != dst_seg_name {
            return ApplyOutcome::Unsupported("Copy writes disagree on dst seg");
        }
        let src_seg = w
            .val_indirect_read
            .as_ref()
            .and_then(|ir| ir.seg_property.as_deref());
        if src_seg != Some(src_seg_name) {
            return ApplyOutcome::Unsupported("Copy writes disagree on src seg");
        }
        if w.val_indirect_read.as_ref().map(|ir| ir.seg_times_sixteen) != Some(src_seg_scaled) {
            return ApplyOutcome::Unsupported("Copy writes disagree on src seg scaling");
        }
    }
    let Some(dst_ptr_value) = read_prop(program, state, slots, dst_ptr_name) else {
        return ApplyOutcome::Unsupported("dst pointer slot unresolved");
    };
    let Some(src_ptr_value) = read_prop(program, state, slots, src_ptr_name) else {
        return ApplyOutcome::Unsupported("src pointer slot unresolved");
    };
    let Some(dst_seg_value) = read_prop(program, state, slots, dst_seg_name) else {
        return ApplyOutcome::Unsupported("dst seg slot unresolved");
    };
    let Some(src_seg_value) = read_prop(program, state, slots, src_seg_name) else {
        return ApplyOutcome::Unsupported("src seg slot unresolved");
    };
    let Some(flags) = read_prop(program, state, slots, &dst_entry.flag_property) else {
        return ApplyOutcome::Unsupported("flag_property unresolved");
    };
    let df_active = ((flags >> dst_entry.flag_bit) & 1) != 0;
    let signed_step: i32 = if df_active { -step } else { step };

    // Pointer-wrap check, both ranges (16-bit segment offset). Each
    // iteration touches `step` bytes at `ptr + i*signed_step` for
    // both src and dst.
    let n64 = n as i64;
    let step64 = step as i64;
    let bounds = |ptr: i32| -> (i64, i64) {
        if df_active {
            (ptr as i64 - (n64 - 1) * step64, ptr as i64 + step64)
        } else {
            (ptr as i64, ptr as i64 + n64 * step64)
        }
    };
    let (dst_lo, dst_hi) = bounds(dst_ptr_value);
    let (src_lo, src_hi) = bounds(src_ptr_value);
    if dst_lo < 0 || dst_hi > 0x10000 {
        return ApplyOutcome::Unsupported("dst pointer would wrap past 16-bit segment");
    }
    if src_lo < 0 || src_hi > 0x10000 {
        return ApplyOutcome::Unsupported("src pointer would wrap past 16-bit segment");
    }

    // Virtual-region overlap on the destination only — the source side
    // is read through `state.read_mem`, which transparently handles
    // packed cells / extended map / windowed byte arrays. The
    // destination is written through `bulk_store_byte` which routes
    // similarly, but the bulk path can't substitute for cabinet-emitted
    // write dispatch over virtual regions. One carve-out stays on the
    // fast path: a destination range entirely inside a cell-backed
    // (writable) windowed byte array is serviced per-byte through
    // `state.write_mem`, which performs the same key-remapped cell
    // splice the cabinet's own write rules do. Every OTHER virtual
    // destination falls back to per-tick evaluation, which runs the
    // cabinet's real write cascade — the reference semantic (a store
    // into a literal-backed rom-disk window matches no write rule and
    // drops, exactly as Chrome evaluates it; Windows 1.01's WRITE.EXE
    // hits this via Corduroy's INT 13h AH=03h on a read-only floppy,
    // 2026-07-13).
    let dst_seg_base = (dst_seg_value as i64) * 16;
    let dst_lo_linear = dst_seg_base + dst_lo;
    let dst_hi_linear = dst_seg_base + dst_hi;
    let dst_in_writable_window = state.windowed_byte_array.as_ref().is_some_and(|dw| {
        matches!(dw.backing, crate::state::WindowBacking::PackedCells { .. })
            && dst_lo_linear >= dw.window_base as i64
            && dst_hi_linear <= dw.window_end as i64
    });
    if !dst_in_writable_window
        && ranges_overlap_virtual(state, dst_lo_linear, dst_hi_linear - dst_lo_linear)
    {
        return ApplyOutcome::PerIterFallback("destination range overlaps virtual region");
    }

    // Virtual-region overlap on the SOURCE side too. `state.read_mem`
    // resolves packed cells and windowed byte arrays, but NOT regions
    // whose bytes exist only in compiled dispatch (e.g. a ROM region
    // emitted as read-only dispatch arms with no backing cells) — for
    // those it returns 0 and the copy would fabricate zeros. The one
    // carve-out is a source range entirely inside the windowed byte
    // array (either backing): read_mem resolves that exactly as the
    // cabinet's own read rules do. Everything else that's virtual
    // refuses loudly and falls back to per-tick execution, which
    // evaluates the cabinet's real dispatch.
    let src_seg_base_lin = if src_seg_scaled {
        (src_seg_value as i64) * 16
    } else {
        src_seg_value as i64
    };
    let src_lo_linear = src_seg_base_lin + src_lo;
    let src_hi_linear = src_seg_base_lin + src_hi;
    let src_in_window = state.windowed_byte_array.as_ref().is_some_and(|dw| {
        src_lo_linear >= dw.window_base as i64 && src_hi_linear <= dw.window_end as i64
    });
    if !src_in_window
        && ranges_overlap_virtual(state, src_lo_linear, src_hi_linear - src_lo_linear)
    {
        return ApplyOutcome::PerIterFallback("source range overlaps virtual region");
    }

    // Hot loop. Walk per-iter, per-write — same as the hardcoded MOVSW
    // path. For step=1 (MOVSB-shape) this is one read+write per iter;
    // for step=2 (MOVSW-shape) this is two read+writes per iter, in
    // position order (low byte at offset 0, high byte at offset 1).
    //
    // The read-then-write order within an iteration matters for
    // overlap: if dst overlaps src downstream of the iteration's
    // current position, an earlier iteration's write may overwrite a
    // future iteration's source. The hardcoded path walks per-byte in
    // hardware iteration order to preserve those semantics; this
    // applier does the same by reading and writing each byte through
    // `state.read_mem` / `bulk_store_byte` in the same order.
    // Scale the source base exactly as the cabinet's own captured shape
    // does: `var(seg)*16 + ptr` shapes scale here; `var(base) + ptr`
    // shapes captured a slot that already holds the full addend (e.g. a
    // segment-override-aware pre-scaled base).
    let src_seg_base = if src_seg_scaled {
        (src_seg_value as i64) * 16
    } else {
        src_seg_value as i64
    };
    let dst_ptr_i64 = dst_ptr_value as i64;
    let src_ptr_i64 = src_ptr_value as i64;

    let mut copied_bulk = false;
    if signed_step > 0 && !dst_in_writable_window {
        let total = n64 * (descriptor.writes.len() as i64);
        let src_linear = src_seg_base + src_ptr_i64;
        let dst_linear = dst_seg_base + dst_ptr_i64;
        if total > 0 && src_linear >= 0 && dst_linear >= 0 {
            state.bulk_copy_bytes(src_linear as usize, dst_linear as usize, total as usize);
            copied_bulk = true;
        }
    }
    if !copied_bulk {
        for i in 0..n64 {
            let iter_off = i * (signed_step as i64);
            let dst_iter_base = dst_seg_base + dst_ptr_i64 + iter_off;
            let src_iter_base = src_seg_base + src_ptr_i64 + iter_off;
            for k in 0..(descriptor.writes.len() as i64) {
                let s = src_iter_base + k;
                let d = dst_iter_base + k;
                let b = (state.read_mem(s as i32) & 0xFF) as u8;
                if dst_in_writable_window {
                    // Route through write_mem so the windowed byte array's
                    // key-remapped cell splice applies (bulk_store_byte
                    // writes raw memory/cells and would miss the window).
                    state.write_mem(d as i32, b as i32);
                } else {
                    bulk_store_byte(state, d, b);
                }
            }
        }
    }

    if commit == CommitMode::Full {
        // Both pointers advance by the same signed step (structurally
        // verified above). 16-bit wrap on commit, matching the hardcoded
        // path's `& 0xFFFF`.
        let new_dst = (dst_ptr_value + n * signed_step) & 0xFFFF;
        let new_src = (src_ptr_value + n * signed_step) & 0xFFFF;
        commit_pointer(state, &dst_entry.property, new_dst);
        commit_pointer(state, &src_entry.property, new_src);
        commit_counter(state, &counter.property, 0);
        commit_ip_and_cycles(program, state, slots, descriptor, n);
    }

    ApplyOutcome::Applied { iterations: n }
}

/// Trace gate for the flag-conditioned (CMPS/SCAS-shape) walker —
/// same env var as the deleted hardcoded path's tracer, so existing
/// debugging workflows keep working. Prints one line per fast-forward
/// with the fields that matter for divergence hunting (iteration
/// count, pointer travel, the last compared pair, committed flags).
#[cfg(not(target_arch = "wasm32"))]
fn trace_cmps_scas_enabled() -> bool {
    use std::sync::OnceLock;
    static T: OnceLock<bool> = OnceLock::new();
    *T.get_or_init(|| std::env::var("CALCITE_REP_TRACE_CMPS_SCAS").is_ok())
}
#[cfg(target_arch = "wasm32")]
fn trace_cmps_scas_enabled() -> bool {
    false
}

/// Apply a `BulkClass::ReadOnly` descriptor to state. The structural contract:
///
/// `ReadOnly` covers loops that walk pointers and possibly check a
/// flag-conditioned early-exit predicate, but never mutate memory. On x86
/// these are LODS / CMPS / SCAS; structurally any cabinet whose recogniser
/// produces a descriptor with `writes.len() == 0` and one of the three
/// shapes below would dispatch identically here.
///
/// ## Structural sub-shapes (no opcode dispatch)
///
/// The applier distinguishes the three sub-shapes structurally from
/// metadata the recogniser already produces:
///
/// | Shape | `flag_conditioned` | `pointers.len()` | Behaviour |
/// |-------|--------------------|------------------|-----------|
/// | LODS  | false              | 1                | walks `n` iters, no comparison, no exit |
/// | SCAS  | true               | 1                | walks until CX exhausted or flag-condition flips |
/// | CMPS  | true               | 2                | walks until CX exhausted or flag-condition flips |
///
/// `flag_conditioned` is captured by the recogniser at descriptor build
/// time from the predicate's disjunction shape (REPE/REPNE arms). The
/// pointer count is structural by construction. The applier never reads
/// an opcode value or a property name as a discriminator.
///
/// ## Memory side-effects
///
/// `ReadOnly` produces zero memory or extended-map mutation by definition
/// — neither LODS, CMPS, nor SCAS writes a single byte. The applier
/// therefore satisfies the dual harness's memory + extended diff trivially:
/// it never touches `state.memory` or `state.extended`.
///
/// ## Iteration count
///
/// For LODS, the iteration count equals the counter `n`.
///
/// For CMPS/SCAS, the iteration count depends on per-iter source-vs-
/// destination equality. The applier resolves every segment / pointer /
/// accumulator / rep-type slot through `descriptor.comparison_shape`,
/// which the recogniser populates structurally from the cabinet's own
/// per-opcode comparison dispatch. No literal-name reads of any
/// register or flag slot — a cabinet that names the slots anything at
/// all routes through the same code path.
///
/// LODS is structurally clean — it reads no name as text beyond the
/// `counter.property` lookup that all variants share (which is itself
/// whole-name slot equality, the same routing every applier uses).
///
/// ## State-var deltas (deferred to step 7)
///
/// As with [`apply_fill`] and [`apply_copy`], state-var commits
/// (DI/SI/CX/IP/cycleCount/flags) are deferred to step 7's full flip.
/// The dual harness diffs only memory + extended in this step.
pub(crate) fn apply_read_only(
    descriptor: &LoopDescriptor,
    program: &CompiledProgram,
    state: &mut State,
    slots: &[i32],
) -> ApplyOutcome {
    apply_read_only_with_commit(descriptor, program, state, slots, CommitMode::MemoryOnly)
}

/// Variant of [`apply_read_only`] that optionally commits state vars
/// (DI/SI/CX/IP/cycleCount/flags). See [`apply_fill_with_commit`].
pub(crate) fn apply_read_only_with_commit(
    descriptor: &LoopDescriptor,
    program: &CompiledProgram,
    state: &mut State,
    slots: &[i32],
    commit: CommitMode,
) -> ApplyOutcome {
    if descriptor.bulk_class != BulkClass::ReadOnly {
        return ApplyOutcome::Unsupported("not ReadOnly class");
    }
    if let Some(pre) = descriptor.precondition.as_ref() {
        if !crate::pattern::loop_descriptor::evaluate_precondition(pre, program, state, slots) {
            return ApplyOutcome::PreconditionNotMet;
        }
    }
    if descriptor.per_iter_cycles.is_none() {
        return ApplyOutcome::Unsupported("per_iter_cycles not extracted");
    }
    if commit == CommitMode::Full && descriptor.ip_extra_advance_slot.is_none() {
        return ApplyOutcome::Unsupported("ip_extra_advance_slot not captured");
    }
    let Some(counter) = descriptor.counter.as_ref() else {
        return ApplyOutcome::Unsupported("no counter");
    };
    let Some(n) = read_prop(program, state, slots, &counter.property) else {
        return ApplyOutcome::Unsupported("counter unresolved");
    };
    if n <= 0 {
        return ApplyOutcome::Applied { iterations: 0 };
    }
    if !descriptor.writes.is_empty() {
        // ReadOnly by definition has no writes. A descriptor with writes
        // got mis-classified upstream — bail.
        return ApplyOutcome::Unsupported("ReadOnly expects no writes");
    }
    let p_count = descriptor.pointers.len();
    if p_count == 0 || p_count > 2 {
        return ApplyOutcome::Unsupported("ReadOnly expects 1 or 2 pointers");
    }

    // Structural sub-shape selection. No opcode discriminator; no name
    // inspection. The classifier's `flag_conditioned` already captured
    // whether the predicate has the disjunction shape that gates an
    // early exit; pointer count distinguishes CMPS (2) from SCAS (1) at
    // a structural level. LODS is the no-exit case (`flag_conditioned`
    // is false; the predicate is just the counter check).
    if !descriptor.flag_conditioned {
        // LODS shape. No memory writes, no early exit, n iterations.
        //
        // A LODS-shape loop loads the accumulator from memory every
        // iteration; its post-loop accumulator is mem[last pointer
        // position]. The applier does not yet model the accumulator
        // target structurally, so a Full commit would advance pointer /
        // counter / IP while leaving the accumulator stale — silent
        // corruption. Loud-not-wrong: refuse, so the dispatcher panics
        // and the gap is extended deliberately rather than discovered
        // as misbehaviour downstream. (No smoke-set or doom cabinet
        // reaches this arm — verified by byte-identical A/B against the
        // hardcoded path, which never fast-forwarded LODS at all.)
        if p_count != 1 {
            return ApplyOutcome::Unsupported(
                "LODS shape (flag_conditioned=false) expects exactly 1 pointer",
            );
        }
        if commit == CommitMode::Full {
            return ApplyOutcome::Unsupported(
                "LODS-shape Full commit not implemented (accumulator target not modelled)",
            );
        }
        return ApplyOutcome::Applied { iterations: n };
    }

    // CMPS / SCAS shape. CMPS = 2 pointers (source + dest), SCAS = 1
    // pointer (dest only — source is the accumulator).
    //
    // All segment / accumulator / rep-type slots are resolved through
    // `descriptor.comparison_shape`, which the recogniser populates by
    // structurally identifying the cabinet's per-opcode comparison
    // dispatch. The slot names we read below are the names the cabinet
    // itself uses — no character of any name is inspected by this
    // function. A cabinet that names its segment slot anything at all
    // (`--ES`, `--zorch`, `--tape_dst_page`) gets routed identically.
    let Some(cmp) = descriptor.comparison_shape.as_ref() else {
        return ApplyOutcome::Unsupported("comparison_shape not captured");
    };
    // Scale each base exactly as its captured shape does: ×16 shapes
    // hold an unscaled 16-bit segment; flat shapes hold the full
    // (up to 20-bit) addend — masking or rescaling those reads from
    // the wrong address.
    let dst_seg_raw = read_prop(program, state, slots, &cmp.dst_seg_property).unwrap_or(0);
    let dst_seg_base = if cmp.dst_seg_times_sixteen {
        ((dst_seg_raw & 0xFFFF) as i64) * 16
    } else {
        dst_seg_raw as i64
    };
    let dst_entry = &descriptor.pointers[0];
    let Some(dst_ptr) = read_prop(program, state, slots, &dst_entry.self_property) else {
        return ApplyOutcome::Unsupported("dst pointer unresolved");
    };
    let Some(flags) = read_prop(program, state, slots, &dst_entry.flag_property) else {
        return ApplyOutcome::Unsupported("flag_property unresolved");
    };
    let df_active = ((flags >> dst_entry.flag_bit) & 1) != 0;
    let step: i32 = dst_entry.base_step;
    let signed_step: i32 = if df_active { -step } else { step };
    let is_word = cmp.width == 2;

    // CMPS gets its source pointer from the second pointer entry.
    // SCAS gets its source from an accumulator slot (the one named by
    // the comparison shape's ComparisonSource::Accumulator).
    let is_cmps = p_count == 2;
    let (src_ptr_opt, src_seg_base, scas_acc) = match &cmp.source {
        crate::pattern::loop_descriptor::ComparisonSource::Pointer {
            seg_property,
            seg_times_sixteen,
            ..
        } => {
            // CMPS-shape. Sanity: comparison's pointer source must
            // match descriptor.pointers[1], and both pointers must share
            // step + direction flag.
            if !is_cmps {
                return ApplyOutcome::Unsupported(
                    "comparison_shape says Pointer source but descriptor has 1 pointer",
                );
            }
            let src_entry = &descriptor.pointers[1];
            if src_entry.base_step != step
                || src_entry.flag_property != dst_entry.flag_property
                || src_entry.flag_bit != dst_entry.flag_bit
            {
                return ApplyOutcome::Unsupported(
                    "CMPS-shape pointers disagree on step / direction flag",
                );
            }
            let Some(src_ptr) = read_prop(program, state, slots, &src_entry.self_property) else {
                return ApplyOutcome::Unsupported("src pointer unresolved");
            };
            let src_seg_raw = read_prop(program, state, slots, seg_property).unwrap_or(0);
            let src_base = if *seg_times_sixteen {
                ((src_seg_raw & 0xFFFF) as i64) * 16
            } else {
                src_seg_raw as i64
            };
            (Some(src_ptr & 0xFFFF), src_base, 0)
        }
        crate::pattern::loop_descriptor::ComparisonSource::Accumulator {
            byte_property,
            word_property,
        } => {
            if is_cmps {
                return ApplyOutcome::Unsupported(
                    "comparison_shape says Accumulator source but descriptor has 2 pointers",
                );
            }
            let acc_slot = if is_word { word_property } else { byte_property };
            let raw = read_prop(program, state, slots, acc_slot).unwrap_or(0);
            let acc = if is_word { raw & 0xFFFF } else { raw & 0xFF };
            (None, 0i64, acc)
        }
    };
    // REPE / REPNE selector: resolved through the comparison shape's
    // rep_type_property slot. When the recogniser couldn't identify a
    // discriminator slot (a cabinet with a single unconditional
    // continue-while-equal polarity, for example), we default to 0,
    // which makes both REPE-stop and REPNE-stop branches below quiet:
    // the loop runs all `n` iterations.
    let rep_type = cmp
        .rep_type_property
        .as_ref()
        .and_then(|slot| read_prop(program, state, slots, slot))
        .unwrap_or(0);

    // Per-iter walk. Same shape as `rep_fast_forward_cmps_scas`. We don't
    // mutate memory (no `bulk_store_byte` or `bulk_fill` here); state-var
    // commits below match the hardcoded path observable-for-observable.
    let dst_mask = 0xFFFFi64;
    let mut di = (dst_ptr & 0xFFFF) as i64;
    let mut si = src_ptr_opt.map(|p| p as i64).unwrap_or(0);
    let mut iters = 0i32;
    let mut last_dst: i32 = 0;
    let mut last_src: i32 = 0;
    let mut cx_remaining: i32 = n;
    let n_max = n;
    for _ in 0..n_max {
        let dst_lin = (dst_seg_base + di) & 0xFFFFF;
        let dst_v = if is_word {
            let lo = state.read_mem(dst_lin as i32) & 0xFF;
            let hi = state.read_mem(((dst_lin + 1) & 0xFFFFF) as i32) & 0xFF;
            lo | (hi << 8)
        } else {
            state.read_mem(dst_lin as i32) & 0xFF
        };
        let src_v = if is_cmps {
            let src_lin = (src_seg_base + si) & 0xFFFFF;
            if is_word {
                let lo = state.read_mem(src_lin as i32) & 0xFF;
                let hi = state.read_mem(((src_lin + 1) & 0xFFFFF) as i32) & 0xFF;
                lo | (hi << 8)
            } else {
                state.read_mem(src_lin as i32) & 0xFF
            }
        } else {
            scas_acc
        };

        last_dst = dst_v;
        last_src = src_v;

        // Pointer advance, CX decrement.
        di = (di + signed_step as i64) & dst_mask;
        if is_cmps {
            si = (si + signed_step as i64) & dst_mask;
        }
        iters += 1;
        cx_remaining -= 1;

        // Early-exit. ZF set iff equal. REPE (rep_type=1): exit when not
        // equal. REPNE (rep_type=2): exit when equal. Other rep_type
        // values shouldn't reach a flag-conditioned descriptor — bail
        // out at iteration count and let step 7 surface it.
        let zf = src_v == dst_v;
        if rep_type == 1 && !zf {
            break;
        }
        if rep_type == 2 && zf {
            break;
        }
    }

    if commit == CommitMode::Full {
        // Pointer commits.
        commit_pointer(state, &dst_entry.property, di as i32);
        if is_cmps {
            // Find the source pointer entry's property name. We resolved
            // the pointer index earlier (descriptor.pointers[1]).
            let src_entry = &descriptor.pointers[1];
            commit_pointer(state, &src_entry.property, si as i32);
        }
        // CX commit: cx_remaining is what's left after the loop.
        // Hardcoded path masks with 0xFFFF.
        commit_counter(state, &counter.property, cx_remaining & 0xFFFF);

        // Flags commit. Mirrors `rep_fast_forward_cmps_scas`'s
        // subFlags(dst, src) order:
        //   CMPS: dst_arg = mem[DS:SI] (last_src), src_arg = mem[ES:DI] (last_dst)
        //   SCAS: dst_arg = AL/AX (scas_acc),     src_arg = mem[ES:DI] (last_dst)
        let prev_flags = read_prop(program, state, slots, &dst_entry.flag_property)
            .unwrap_or(0);
        let (fl_dst, fl_src) = if is_cmps {
            (last_src, last_dst)
        } else {
            (scas_acc, last_dst)
        };
        // If the loop ran zero iterations (n was 0 — caught above) we'd
        // skip the commit; here iters > 0 by construction, so last_*
        // are populated.
        let _ = iters;
        let new_flags = compute_sub_flags(fl_dst, fl_src, is_word, prev_flags);
        // The flag word lives on the slot named by dst_entry.flag_property
        // (the cabinet's own mirror name, e.g. `--__1flags` — resolved to
        // its committed storage by commit_pointer's address-map routing).
        commit_pointer(state, &dst_entry.flag_property, new_flags);

        #[cfg(not(target_arch = "wasm32"))]
        if trace_cmps_scas_enabled() {
            eprintln!(
                "[rep_cmps_scas:gen] rep_type={} n={} iters={} dst_ptr {:#06x}->{:#06x} \
                 last_src={:#x} last_dst={:#x} flags={:#x}",
                rep_type, n, iters, dst_ptr & 0xFFFF, di as i32, last_src, last_dst, new_flags,
            );
        }

        // IP advance and per-iter cycle charge. The `iters` value (not
        // n) is what the hardcoded path uses for the cycle multiplier on
        // CMPS/SCAS — early-exit reduces actual work done.
        commit_ip_and_cycles(program, state, slots, descriptor, iters);
    }

    ApplyOutcome::Applied { iterations: iters }
}

// ---------------------------------------------------------------------------
// State-var commit helpers.
// ---------------------------------------------------------------------------
//
// The hardcoded `rep_fast_forward` writes post-loop state vars by *bare*
// name (e.g. `state.set_var("DI", ...)`). The descriptor carries property
// names with the `--` prefix attached (`pointer.property = "--DI"`). The
// helpers below normalise this — they try `state.set_var(bare_name, ...)`
// first (the path the cabinet actually uses for register-shaped slots),
// and fall back to no-op if the slot doesn't exist.
//
// Cardinal-rule note: these helpers are name-agnostic. They never inspect
// any character of the property name they're handed — `strip_prefix("--")`
// is a uniform transformation on the whole string, identical to what
// `read_prop` does for resolution.

/// Commit a pointer (or a flag word) to its post-loop value. Used for
/// DI/SI commits in Fill/Copy and for the flags slot in ReadOnly's
/// CMPS/SCAS sub-shape.
///
/// Resolution mirrors `read_prop`: the canonical bare name
/// (`to_bare_name` — strips `--` and `__N` buffer prefixes, pure
/// function, debugger-thread-safe) addressed into the state-var table.
/// The prefix strip is what makes mirror-named slots committable — the
/// descriptor's `flag_property` is the cabinet's prior-tick mirror
/// (e.g. `--__1flags`), whose committed storage is the `flags` state
/// var. Before this routing existed, the CMPS/SCAS flags commit
/// silently no-opped on real cabinets and downstream conditional jumps
/// ran on stale flags.
fn commit_pointer(state: &mut State, property: &str, value: i32) {
    let canon = crate::eval::to_bare_name(property);
    if state.state_var_index.contains_key(canon) {
        state.set_var(canon, value);
    }
    // Anything else: the cabinet doesn't have this slot reachable
    // through the state-var path. The hardcoded `rep_fast_forward` only
    // ever wrote through `set_var` so we mirror that — slot-only
    // properties fall through silently (the recogniser would have
    // flagged a structurally invalid descriptor before we got here).
}

/// Commit the loop counter to its post-loop value (`0` for Fill/Copy,
/// CX remaining for CMPS/SCAS early exits).
fn commit_counter(state: &mut State, property: &str, value: i32) {
    commit_pointer(state, property, value);
}

/// Advance the loop's IP slot and charge the cycle counter for `iters`
/// iterations — both through the descriptor-carried names (the
/// cabinet's own `ip_property` and `per_iter_cycles.property`), never
/// hardcoded ones. A cabinet that calls its instruction-pointer slot
/// `--pc` and its cycle counter `--zorch` commits to those.
///
/// The IP advance is `ip + ip_advance_literal + extra`: the literal is
/// the exit branch's structural constant, `extra` is read from
/// `descriptor.ip_extra_advance_slot`, the cabinet's own name for the
/// stay branch's subtrahend (`calc(self − var(extra))`). See
/// `LoopDescriptor::ip_extra_advance_slot` for the fixed-point argument
/// that makes `IP + literal + extra` the exit branch's value.
///
/// Caller contract: the descriptor's `ip_extra_advance_slot` and
/// `per_iter_cycles` MUST be `Some` and the extra slot MUST resolve.
/// Those invariants are checked at the start of each
/// `apply_*_with_commit` and the applier returns `Unsupported` before
/// reaching this commit step when they don't hold; here we resolve
/// unconditionally (with `unwrap_or(0)` as belt-and-braces, since the
/// slot was already proven resolvable upstream).
fn commit_ip_and_cycles(
    program: &CompiledProgram,
    state: &mut State,
    slots: &[i32],
    descriptor: &LoopDescriptor,
    iters: i32,
) {
    let ip_var = crate::eval::to_bare_name(&descriptor.ip_property);
    let ip = state.get_var(ip_var).unwrap_or(0);
    let extra = descriptor
        .ip_extra_advance_slot
        .as_deref()
        .and_then(|slot| read_prop(program, state, slots, slot))
        .unwrap_or(0);
    let new_ip = (ip + descriptor.ip_advance_literal + extra) & 0xFFFF;
    commit_pointer(state, &descriptor.ip_property, new_ip);
    if let Some(charge) = descriptor.per_iter_cycles.as_ref() {
        let cc_var = crate::eval::to_bare_name(&charge.property);
        if let Some(cc) = state.get_var(cc_var) {
            state.set_var(cc_var, cc.wrapping_add(iters.wrapping_mul(charge.per_iter)));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::loop_descriptor::{
        BulkClass, CounterEntry, CycleCharge, IndirectRead, LoopDescriptor, PointerEntry,
        WriteEntry,
    };
    use crate::types::{CssValue, Expr, PropertyDef, PropertySyntax, StyleTest};

    /// Construct a minimal `CompiledProgram` whose only filled-in field
    /// is `loop_descriptors`. The applier reads `property_slots` too,
    /// but on cabinets that route registers through state-vars (the
    /// case for AL/AX/CX/DI/etc.) `property_slots` stays empty and the
    /// fallback `state.get_var` path takes over. That's the case all
    /// of these tests exercise.
    fn empty_program_with_descriptor(desc: LoopDescriptor) -> CompiledProgram {
        CompiledProgram {
            ops: Vec::new(),
            slot_count: 0,
            writeback: Vec::new(),
            broadcast_writes: Vec::new(),
            packed_broadcast_writes: Vec::new(),
            packed_cell_tables: Vec::new(),
            packed_exception_tables: Vec::new(),
            dispatch_tables: Vec::new(),
            chain_tables: Vec::new(),
            flat_dispatch_arrays: Vec::new(),
            property_slots: std::collections::HashMap::new(),
            functions: Vec::new(),
            windowed_byte_array: None,
            loop_descriptors: vec![desc],
        }
    }

    /// Allocate a 1 MiB state and load the named property as a state var.
    fn rigged_state(names: &[&str]) -> State {
        let mut state = State::new(1 << 20);
        let props: Vec<PropertyDef> = names
            .iter()
            .map(|n| PropertyDef {
                name: format!("--{}", n),
                syntax: PropertySyntax::Integer,
                inherits: false,
                initial_value: Some(CssValue::Integer(0)),
            })
            .collect();
        state.load_properties(&props);
        state
    }

    /// Build a STOSB-shape descriptor: one byte-step pointer (`--DI`),
    /// one write entry whose val_expr reads `--AL`. Counter is `--CX`.
    /// Address decomposition is `(--ES, --DI)`.
    fn stosb_descriptor() -> LoopDescriptor {
        LoopDescriptor {
            key_property: "--opcode".to_string(),
            key_value: 0xAA,
            ip_property: "--IP".to_string(),
            ip_self_property: "--IP_prev".to_string(),
            ip_advance_literal: 1,
            predicate_properties: vec!["--repContinue".to_string()],
            predicate: StyleTest::Single {
                property: "--repContinue".to_string(),
                value: Expr::Literal(1.0),
            },
            predicate_means_stay: true,
            counter: Some(CounterEntry {
                property: "--CX".to_string(),
                self_property: "--CX_prev".to_string(),
                step: 1,
            }),
            pointers: vec![PointerEntry {
                property: "--DI".to_string(),
                self_property: "--DI".to_string(),
                base_step: 1,
                flag_property: "--flags".to_string(),
                flag_bit: 10,
            }],
            writes: vec![WriteEntry {
                addr_property: "--mAddr0".to_string(),
                val_property: "--mVal0".to_string(),
                addr_expr: Expr::Literal(0.0),
                val_expr: Expr::Var {
                    name: "--AL".to_string(),
                    fallback: None,
                },
                addr_decomposition: Some(("--ES".to_string(), "--DI".to_string())),
                val_indirect_read: None,
            }],
            flag_conditioned: false,
            bulk_class: BulkClass::Fill,
            per_iter_cycles: Some(CycleCharge { property: "--cycleCount".to_string(), per_iter: 10 }),
            ip_extra_advance_slot: Some("--prefixLen".to_string()),
            comparison_shape: None,
            precondition: None,
        }
    }

    /// The dispatcher's loop-continuation gate: the descriptor's
    /// predicate, with its recorded polarity, decides whether a loop is
    /// mid-flight. Predicate false (e.g. a single-shot run of the same
    /// opcode, or the final iteration's tick) must NOT fast-forward —
    /// the divergence this prevents is bulk-applying a stale counter.
    #[test]
    fn loop_predicate_gates_fast_forward() {
        use crate::pattern::loop_descriptor::evaluate_loop_predicate;
        let program = empty_program_with_descriptor(stosb_descriptor());
        let desc = &program.loop_descriptors[0];
        let mut state = rigged_state(&["repContinue"]);

        state.set_var("repContinue", 0);
        assert!(
            !evaluate_loop_predicate(desc, &program, &state, &[]),
            "predicate false (advance branch taken) must gate the fast-forward off",
        );

        state.set_var("repContinue", 1);
        assert!(
            evaluate_loop_predicate(desc, &program, &state, &[]),
            "predicate true (stay branch taken) must let the fast-forward run",
        );

        // Inverted polarity: predicate true now means "the loop exited".
        let mut inverted = stosb_descriptor();
        inverted.predicate_means_stay = false;
        let program2 = empty_program_with_descriptor(inverted);
        let desc2 = &program2.loop_descriptors[0];
        state.set_var("repContinue", 1);
        assert!(!evaluate_loop_predicate(desc2, &program2, &state, &[]));
        state.set_var("repContinue", 0);
        assert!(evaluate_loop_predicate(desc2, &program2, &state, &[]));
    }

    /// Build a STOSW-shape descriptor: one word-step pointer, two write
    /// entries, low byte then high byte.
    fn stosw_descriptor() -> LoopDescriptor {
        LoopDescriptor {
            key_property: "--opcode".to_string(),
            key_value: 0xAB,
            ip_property: "--IP".to_string(),
            ip_self_property: "--IP_prev".to_string(),
            ip_advance_literal: 1,
            predicate_properties: vec!["--repContinue".to_string()],
            predicate: StyleTest::Single {
                property: "--repContinue".to_string(),
                value: Expr::Literal(1.0),
            },
            predicate_means_stay: true,
            counter: Some(CounterEntry {
                property: "--CX".to_string(),
                self_property: "--CX_prev".to_string(),
                step: 1,
            }),
            pointers: vec![PointerEntry {
                property: "--DI".to_string(),
                self_property: "--DI".to_string(),
                base_step: 2,
                flag_property: "--flags".to_string(),
                flag_bit: 10,
            }],
            writes: vec![
                WriteEntry {
                    addr_property: "--mAddr0".to_string(),
                    val_property: "--mVal0".to_string(),
                    addr_expr: Expr::Literal(0.0),
                    val_expr: Expr::Var {
                        name: "--AL".to_string(),
                        fallback: None,
                    },
                    addr_decomposition: Some(("--ES".to_string(), "--DI".to_string())),
                    val_indirect_read: None,
                },
                WriteEntry {
                    addr_property: "--mAddr1".to_string(),
                    val_property: "--mVal1".to_string(),
                    addr_expr: Expr::Literal(0.0),
                    val_expr: Expr::Var {
                        name: "--AH".to_string(),
                        fallback: None,
                    },
                    addr_decomposition: Some(("--ES".to_string(), "--DI".to_string())),
                    val_indirect_read: None,
                },
            ],
            flag_conditioned: false,
            bulk_class: BulkClass::Fill,
            per_iter_cycles: Some(CycleCharge { property: "--cycleCount".to_string(), per_iter: 10 }),
            ip_extra_advance_slot: Some("--prefixLen".to_string()),
            comparison_shape: None,
            precondition: None,
        }
    }

    /// Byte fill, forward direction: 64 bytes of 0x42 starting at
    /// ES:DI = 0xA000:0x10. Verifies the bulk-fill fast path (single
    /// write per iter collapses to one `bulk_fill` call).
    #[test]
    fn fill_byte_forward_writes_n_copies() {
        let desc = stosb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "ES", "AL", "flags"]);
        state.set_var("CX", 64);
        state.set_var("DI", 0x10);
        state.set_var("ES", 0xA000);
        state.set_var("AL", 0x42);
        state.set_var("flags", 0); // DF=0

        let outcome = apply_fill(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 64 });

        for i in 0..64 {
            assert_eq!(
                state.memory[0xA0010 + i],
                0x42,
                "byte {} should be 0x42",
                i
            );
        }
        // Boundaries untouched.
        assert_eq!(state.memory[0xA000F], 0);
        assert_eq!(state.memory[0xA0050], 0);
    }

    /// Byte fill, reverse direction (DF=1): 16 bytes of 0xAA starting
    /// at DI=0x20 walking *backward*. End-state should fill
    /// [0x11..0x21) (DI inclusive, plus 15 lower addresses).
    #[test]
    fn fill_byte_reverse_writes_lower_addresses() {
        let desc = stosb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "ES", "AL", "flags"]);
        state.set_var("CX", 16);
        state.set_var("DI", 0x20);
        state.set_var("ES", 0xA000);
        state.set_var("AL", 0xAA);
        state.set_var("flags", 1 << 10); // DF=1

        let outcome = apply_fill(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 16 });

        // 0x11..=0x20 inclusive (16 bytes ending at DI=0x20 inclusive).
        for i in 0x11..=0x20 {
            assert_eq!(
                state.memory[0xA0000 + i],
                0xAA,
                "byte {:#x} should be 0xAA",
                i
            );
        }
        // Boundaries untouched.
        assert_eq!(state.memory[0xA0010], 0);
        assert_eq!(state.memory[0xA0021], 0);
    }

    /// Word fill, forward: AX=0xBEEF, 4 iterations starting at DI=0.
    /// Byte order is low-then-high per iteration: EF BE EF BE EF BE EF BE.
    #[test]
    fn fill_word_forward_writes_low_then_high() {
        let desc = stosw_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "ES", "AL", "AH", "flags"]);
        state.set_var("CX", 4);
        state.set_var("DI", 0);
        state.set_var("ES", 0xA000);
        state.set_var("AL", 0xEF);
        state.set_var("AH", 0xBE);
        state.set_var("flags", 0);

        let outcome = apply_fill(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 4 });

        let want: [u8; 8] = [0xEF, 0xBE, 0xEF, 0xBE, 0xEF, 0xBE, 0xEF, 0xBE];
        for (i, b) in want.iter().enumerate() {
            assert_eq!(state.memory[0xA0000 + i], *b, "word-fill byte {}", i);
        }
        assert_eq!(state.memory[0xA0008], 0);
    }

    /// Word fill, reverse direction: 3 iterations starting at DI=0x10
    /// walking backward by 2.
    #[test]
    fn fill_word_reverse_walks_backward() {
        let desc = stosw_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "ES", "AL", "AH", "flags"]);
        state.set_var("CX", 3);
        state.set_var("DI", 0x10);
        state.set_var("ES", 0xA000);
        state.set_var("AL", 0x34);
        state.set_var("AH", 0x12);
        state.set_var("flags", 1 << 10); // DF=1

        let outcome = apply_fill(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 3 });

        // Iteration 0 writes (0x34, 0x12) at DI=0x10..0x11.
        // Iteration 1 writes at DI-2=0x0E..0x0F.
        // Iteration 2 writes at DI-4=0x0C..0x0D.
        for &(addr_off, want) in &[
            (0x10usize, 0x34u8), (0x11, 0x12),
            (0x0E, 0x34), (0x0F, 0x12),
            (0x0C, 0x34), (0x0D, 0x12),
        ] {
            assert_eq!(
                state.memory[0xA0000 + addr_off],
                want,
                "addr {:#x}",
                addr_off
            );
        }
        // Boundaries untouched.
        assert_eq!(state.memory[0xA000B], 0);
        assert_eq!(state.memory[0xA0012], 0);
    }

    /// Counter already zero: applier returns `Applied { iterations: 0 }`
    /// without touching memory. The hardcoded path bails before calling
    /// us (cx<=0), but the dual harness's contract is that an early-
    /// reached applier behaves identically to the hardcoded skip.
    #[test]
    fn fill_with_zero_counter_is_noop() {
        let desc = stosb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "ES", "AL", "flags"]);
        state.set_var("CX", 0);
        state.set_var("DI", 0x100);
        state.set_var("ES", 0x1000);
        state.set_var("AL", 0x55);
        state.set_var("flags", 0);

        // Pre-fill an arbitrary byte so we can verify it stays.
        state.memory[0x10100] = 0x77;
        let outcome = apply_fill(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 0 });
        assert_eq!(state.memory[0x10100], 0x77, "memory must be untouched");
    }

    /// Negative counter: same noop semantics as zero. (CSS shouldn't
    /// produce this in practice; it's a defensive check.)
    #[test]
    fn fill_with_negative_counter_is_noop() {
        let desc = stosb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "ES", "AL", "flags"]);
        state.set_var("CX", -1);
        state.set_var("DI", 0x100);
        state.set_var("ES", 0x1000);
        state.set_var("AL", 0x55);
        state.set_var("flags", 0);

        let outcome = apply_fill(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 0 });
    }

    /// The applier refuses non-`Fill` classes — Step 4 owns Fill only.
    #[test]
    fn refuses_non_fill_class() {
        let mut desc = stosb_descriptor();
        desc.bulk_class = BulkClass::Copy;
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "ES", "AL", "flags"]);
        state.set_var("CX", 1);

        let outcome = apply_fill(&desc, &prog, &mut state, &[]);
        assert!(matches!(outcome, ApplyOutcome::Unsupported(_)));
    }

    /// Missing `addr_decomposition` on a write: applier bails so the
    /// caller can fall back to the hardcoded path. (This covers
    /// pathological cabinets whose addr expressions don't match the
    /// canonical seg*16+ptr shape; doom8088 always decomposes cleanly.)
    #[test]
    fn refuses_when_addr_decomposition_missing() {
        let mut desc = stosb_descriptor();
        desc.writes[0].addr_decomposition = None;
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "ES", "AL", "flags"]);
        state.set_var("CX", 4);
        state.set_var("DI", 0);
        state.set_var("ES", 0x1000);
        state.set_var("AL", 0x55);

        let outcome = apply_fill(&desc, &prog, &mut state, &[]);
        assert!(matches!(outcome, ApplyOutcome::Unsupported(_)));
    }

    /// `val_expr` is something other than a bare Var: applier bails. A
    /// future step extends `resolve_fill_value`; until then the
    /// hardcoded path still handles these shapes.
    #[test]
    fn refuses_when_val_expr_is_not_bare_var() {
        let mut desc = stosb_descriptor();
        // Pretend the cabinet emitted `(--AL & 0xFF)` (a Calc shape) as
        // the value-side. We don't yet structurally decompose the
        // byte-extract idiom; bail.
        use crate::types::CalcOp;
        desc.writes[0].val_expr = Expr::Calc(CalcOp::Add(
            Box::new(Expr::Var { name: "--AL".to_string(), fallback: None }),
            Box::new(Expr::Literal(0.0)),
        ));
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "ES", "AL", "flags"]);
        state.set_var("CX", 4);
        state.set_var("DI", 0);
        state.set_var("ES", 0x1000);
        state.set_var("AL", 0x55);

        let outcome = apply_fill(&desc, &prog, &mut state, &[]);
        assert!(matches!(outcome, ApplyOutcome::Unsupported(_)));
    }

    /// Pointer wrap past 16-bit segment: applier bails. (Hardcoded path
    /// panics in this case; we choose Unsupported so the caller can
    /// keep its panic semantics if desired.)
    #[test]
    fn refuses_when_pointer_would_wrap() {
        let desc = stosb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "ES", "AL", "flags"]);
        state.set_var("CX", 0x100);
        state.set_var("DI", 0xFF80); // n=256 from DI=0xFF80 wraps past 0xFFFF
        state.set_var("ES", 0xA000);
        state.set_var("AL", 0x42);

        let outcome = apply_fill(&desc, &prog, &mut state, &[]);
        assert!(matches!(outcome, ApplyOutcome::Unsupported(_)));
    }

    /// Position-as-byte-offset is the structural assumption that lets
    /// the multi-write path work without an explicit per-write byte
    /// offset in the descriptor. This test verifies it: if we ever
    /// reorder `writes` (low/high swap), the output flips byte-order.
    /// That's the structural contract — any cabinet whose recogniser
    /// emits writes in source order gets correct results; any whose
    /// recogniser shuffles them gets observable wrongness, surfaced
    /// here.
    #[test]
    fn write_order_determines_byte_order_in_iteration() {
        let mut desc = stosw_descriptor();
        // Swap the writes so high comes first.
        desc.writes.swap(0, 1);
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "ES", "AL", "AH", "flags"]);
        state.set_var("CX", 1);
        state.set_var("DI", 0);
        state.set_var("ES", 0xA000);
        state.set_var("AL", 0xEF);
        state.set_var("AH", 0xBE);
        state.set_var("flags", 0);

        let outcome = apply_fill(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 1 });
        // Swapped order → BE then EF.
        assert_eq!(state.memory[0xA0000], 0xBE);
        assert_eq!(state.memory[0xA0001], 0xEF);
    }

    /// Genericity probe: a descriptor with completely unrelated property
    /// names (no `--AL`, no `--DI`, no `--CX`) should produce
    /// identical mutations as long as the structural shape is the
    /// same. Verifies the applier reads no name as text.
    #[test]
    fn genericity_unrelated_names_same_shape_same_output() {
        let desc = LoopDescriptor {
            key_property: "--opaque_key".to_string(),
            key_value: 0x77, // arbitrary
            ip_property: "--ip_x".to_string(),
            ip_self_property: "--ip_x_prev".to_string(),
            ip_advance_literal: 1,
            predicate_properties: vec!["--cont_x".to_string()],
            predicate: StyleTest::Single {
                property: "--cont_x".to_string(),
                value: Expr::Literal(1.0),
            },
            predicate_means_stay: true,
            counter: Some(CounterEntry {
                property: "--alpha".to_string(),
                self_property: "--alpha_prev".to_string(),
                step: 1,
            }),
            pointers: vec![PointerEntry {
                property: "--beta".to_string(),
                self_property: "--beta".to_string(),
                base_step: 1,
                flag_property: "--gamma".to_string(),
                flag_bit: 3, // arbitrary bit
            }],
            writes: vec![WriteEntry {
                addr_property: "--w_addr".to_string(),
                val_property: "--w_val".to_string(),
                addr_expr: Expr::Literal(0.0),
                val_expr: Expr::Var {
                    name: "--delta".to_string(),
                    fallback: None,
                },
                addr_decomposition: Some(("--epsilon".to_string(), "--beta".to_string())),
                val_indirect_read: None,
            }],
            flag_conditioned: false,
            bulk_class: BulkClass::Fill,
            per_iter_cycles: Some(CycleCharge { property: "--cycle_x".to_string(), per_iter: 10 }),
            ip_extra_advance_slot: Some("--prefixLen".to_string()),
            comparison_shape: None,
            precondition: None,
        };
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["alpha", "beta", "epsilon", "delta", "gamma"]);
        state.set_var("alpha", 8);
        state.set_var("beta", 0x40);
        state.set_var("epsilon", 0x1234);
        state.set_var("delta", 0x99);
        state.set_var("gamma", 0); // bit 3 clear → forward

        let outcome = apply_fill(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 8 });
        let base = (0x1234usize) * 16 + 0x40;
        for i in 0..8 {
            assert_eq!(state.memory[base + i], 0x99, "byte {}", i);
        }
        assert_eq!(state.memory[base + 8], 0);
    }

    /// The `IndirectRead` import is here to keep the test module's
    /// type-imports tidy; if the type were unused the compiler would
    /// warn. This bare-touch keeps it referenced without inflating any
    /// test's surface.
    #[test]
    fn indirect_read_type_referenced() {
        let _: Option<IndirectRead> = None;
    }

    /// Dual-mode equivalence: the applier's memory mutations must match
    /// what an equivalent direct call to the bulk primitives would
    /// produce. Drives the same setup through both paths on independent
    /// State clones and asserts memory equality. Smallest possible
    /// assertion of the harness's actual contract.
    ///
    /// STOSB shape: 200 bytes of 0x5A starting at ES:DI = 0x1234:0x80,
    /// forward direction.
    #[test]
    fn equivalence_byte_fill_matches_direct_bulk_fill() {
        let desc = stosb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let names = &["CX", "DI", "ES", "AL", "flags"];
        let mut state_a = rigged_state(names);
        let mut state_b = rigged_state(names);
        for s in [&mut state_a, &mut state_b] {
            s.set_var("CX", 200);
            s.set_var("DI", 0x80);
            s.set_var("ES", 0x1234);
            s.set_var("AL", 0x5A);
            s.set_var("flags", 0);
        }

        // Path A: descriptor-driven applier.
        let outcome = apply_fill(&desc, &prog, &mut state_a, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 200 });

        // Path B: direct bulk_fill, mirroring what compile.rs does for
        // STOSB. The inputs are the same; only the path through the
        // code differs.
        let es = 0x1234i64;
        let di = 0x80i64;
        let n = 200usize;
        bulk_fill(&mut state_b, es * 16 + di, n, 0x5A);

        // Memory must be byte-for-byte identical. (state_vars differ by
        // design — applier doesn't update them in step 4.)
        assert_eq!(
            state_a.memory, state_b.memory,
            "applier and direct-bulk_fill should produce identical memory"
        );
    }

    /// Same equivalence check, STOSW shape: 100 words of 0xCAFE
    /// starting at ES:DI = 0xB000:0x300, reverse direction (DF=1). DI
    /// is chosen large enough that 100 backward iterations stay in
    /// the segment.
    #[test]
    fn equivalence_word_fill_matches_direct_per_byte_writes() {
        let desc = stosw_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let names = &["CX", "DI", "ES", "AL", "AH", "flags"];
        let mut state_a = rigged_state(names);
        let mut state_b = rigged_state(names);
        for s in [&mut state_a, &mut state_b] {
            s.set_var("CX", 100);
            s.set_var("DI", 0x300);
            s.set_var("ES", 0xB000);
            s.set_var("AL", 0xFE);
            s.set_var("AH", 0xCA);
            s.set_var("flags", 1 << 10); // DF=1
        }

        // Path A: applier.
        let outcome = apply_fill(&desc, &prog, &mut state_a, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 100 });

        // Path B: mirror compile.rs's STOSW reverse loop directly.
        let es = 0xB000i64;
        let di = 0x300i64;
        let lo = 0xFEu8;
        let hi = 0xCAu8;
        let n = 100i64;
        for k in 0..n {
            let off = -k * 2; // DF=1
            let base = es * 16 + di + off;
            bulk_store_byte(&mut state_b, base, lo);
            bulk_store_byte(&mut state_b, base + 1, hi);
        }

        assert_eq!(
            state_a.memory, state_b.memory,
            "applier word-fill must match direct per-byte writes byte-for-byte"
        );
    }

    // -----------------------------------------------------------------
    // Step 5 — apply_copy (BulkClass::Copy / MOVS-shape) tests.
    // -----------------------------------------------------------------

    /// Build a MOVSB-shape descriptor: two byte-step pointers, one write
    /// entry. The write's addr_decomposition resolves to (ES, DI); its
    /// val_indirect_read resolves to (DS, SI) via a derived intermediate
    /// slot named `--_strSrcByte`.
    fn movsb_descriptor() -> LoopDescriptor {
        LoopDescriptor {
            key_property: "--opcode".to_string(),
            key_value: 0xA4,
            ip_property: "--IP".to_string(),
            ip_self_property: "--IP_prev".to_string(),
            ip_advance_literal: 1,
            predicate_properties: vec!["--repContinue".to_string()],
            predicate: StyleTest::Single {
                property: "--repContinue".to_string(),
                value: Expr::Literal(1.0),
            },
            predicate_means_stay: true,
            counter: Some(CounterEntry {
                property: "--CX".to_string(),
                self_property: "--CX_prev".to_string(),
                step: 1,
            }),
            pointers: vec![
                PointerEntry {
                    property: "--DI".to_string(),
                    self_property: "--DI".to_string(),
                    base_step: 1,
                    flag_property: "--flags".to_string(),
                    flag_bit: 10,
                },
                PointerEntry {
                    property: "--SI".to_string(),
                    self_property: "--SI".to_string(),
                    base_step: 1,
                    flag_property: "--flags".to_string(),
                    flag_bit: 10,
                },
            ],
            writes: vec![WriteEntry {
                addr_property: "--mAddr0".to_string(),
                val_property: "--mVal0".to_string(),
                addr_expr: Expr::Literal(0.0),
                val_expr: Expr::Var {
                    name: "--_strSrcByte".to_string(),
                    fallback: None,
                },
                addr_decomposition: Some(("--ES".to_string(), "--DI".to_string())),
                val_indirect_read: Some(IndirectRead {
                    seg_property: Some("--DS".to_string()),
                    seg_times_sixteen: true,
                    pointer_property: "--SI".to_string(),
                    intermediate_property: "--_strSrcByte".to_string(),
                }),
            }],
            flag_conditioned: false,
            bulk_class: BulkClass::Copy,
            per_iter_cycles: Some(CycleCharge { property: "--cycleCount".to_string(), per_iter: 17 }),
            ip_extra_advance_slot: Some("--prefixLen".to_string()),
            comparison_shape: None,
            precondition: None,
        }
    }

    /// A Copy whose source base was captured from the real kiln shape
    /// `var(base) + var(ptr)` (no ×16 in the expression): the base slot
    /// already holds the full scaled addend (kiln's `--_strSrcSeg` is
    /// `if(override: var(--segOverride); else: calc(var(--__1DS)*16))`).
    /// The applier must add it as-is — scaling it again reads from an
    /// address 16× off and silently copies garbage.
    #[test]
    fn apply_copy_flat_base_adds_unscaled() {
        let mut desc = movsb_descriptor();
        {
            let ir = desc.writes[0].val_indirect_read.as_mut().unwrap();
            ir.seg_property = Some("--srcBase".to_string());
            ir.seg_times_sixteen = false;
        }
        let program = empty_program_with_descriptor(desc);
        let mut state = rigged_state(&[
            "CX", "DI", "SI", "ES", "srcBase", "flags", "IP", "cycleCount", "prefixLen",
        ]);
        state.set_var("CX", 4);
        state.set_var("DI", 0x10);
        state.set_var("SI", 0x20);
        state.set_var("ES", 0x100); // dst linear base 0x1000
        state.set_var("srcBase", 0x2000); // FLAT base — already scaled
        state.set_var("flags", 0);
        for i in 0..4 {
            state.write_mem(0x2000 + 0x20 + i, 0xA0 + i);
        }
        let outcome = apply_copy_with_commit(
            &program.loop_descriptors[0],
            &program,
            &mut state,
            &[],
            CommitMode::MemoryOnly,
        );
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 4 });
        for i in 0..4 {
            assert_eq!(
                state.read_mem(0x1000 + 0x10 + i) & 0xFF,
                0xA0 + i,
                "byte {} must be copied from the unscaled flat base",
                i,
            );
        }
    }

    /// MOVSW-shape descriptor: two word-step pointers, two write entries
    /// (low byte at offset 0, high byte at offset 1).
    fn movsw_descriptor() -> LoopDescriptor {
        LoopDescriptor {
            key_property: "--opcode".to_string(),
            key_value: 0xA5,
            ip_property: "--IP".to_string(),
            ip_self_property: "--IP_prev".to_string(),
            ip_advance_literal: 1,
            predicate_properties: vec!["--repContinue".to_string()],
            predicate: StyleTest::Single {
                property: "--repContinue".to_string(),
                value: Expr::Literal(1.0),
            },
            predicate_means_stay: true,
            counter: Some(CounterEntry {
                property: "--CX".to_string(),
                self_property: "--CX_prev".to_string(),
                step: 1,
            }),
            pointers: vec![
                PointerEntry {
                    property: "--DI".to_string(),
                    self_property: "--DI".to_string(),
                    base_step: 2,
                    flag_property: "--flags".to_string(),
                    flag_bit: 10,
                },
                PointerEntry {
                    property: "--SI".to_string(),
                    self_property: "--SI".to_string(),
                    base_step: 2,
                    flag_property: "--flags".to_string(),
                    flag_bit: 10,
                },
            ],
            writes: vec![
                WriteEntry {
                    addr_property: "--mAddr0".to_string(),
                    val_property: "--mVal0".to_string(),
                    addr_expr: Expr::Literal(0.0),
                    val_expr: Expr::Var {
                        name: "--_strSrcByteLo".to_string(),
                        fallback: None,
                    },
                    addr_decomposition: Some(("--ES".to_string(), "--DI".to_string())),
                    val_indirect_read: Some(IndirectRead {
                        seg_property: Some("--DS".to_string()),
                        seg_times_sixteen: true,
                        pointer_property: "--SI".to_string(),
                        intermediate_property: "--_strSrcByteLo".to_string(),
                    }),
                },
                WriteEntry {
                    addr_property: "--mAddr1".to_string(),
                    val_property: "--mVal1".to_string(),
                    addr_expr: Expr::Literal(0.0),
                    val_expr: Expr::Var {
                        name: "--_strSrcByteHi".to_string(),
                        fallback: None,
                    },
                    addr_decomposition: Some(("--ES".to_string(), "--DI".to_string())),
                    val_indirect_read: Some(IndirectRead {
                        seg_property: Some("--DS".to_string()),
                        seg_times_sixteen: true,
                        pointer_property: "--SI".to_string(),
                        intermediate_property: "--_strSrcByteHi".to_string(),
                    }),
                },
            ],
            flag_conditioned: false,
            bulk_class: BulkClass::Copy,
            per_iter_cycles: Some(CycleCharge { property: "--cycleCount".to_string(), per_iter: 17 }),
            ip_extra_advance_slot: Some("--prefixLen".to_string()),
            comparison_shape: None,
            precondition: None,
        }
    }

    /// Helper to seed source bytes at DS:SI for MOVS-shape tests.
    fn seed_bytes(state: &mut State, base_linear: usize, bytes: &[u8]) {
        for (i, &b) in bytes.iter().enumerate() {
            state.memory[base_linear + i] = b;
        }
    }

    /// Byte copy, forward direction: copy 8 bytes from DS:SI = 0x1000:0x40
    /// to ES:DI = 0x2000:0x80. Source range is non-overlapping with dest.
    #[test]
    fn copy_byte_forward() {
        let desc = movsb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "SI", "ES", "DS", "flags"]);
        state.set_var("CX", 8);
        state.set_var("DI", 0x80);
        state.set_var("SI", 0x40);
        state.set_var("ES", 0x2000);
        state.set_var("DS", 0x1000);
        state.set_var("flags", 0); // DF=0
        let src_base = 0x1000 * 16 + 0x40;
        let dst_base = 0x2000 * 16 + 0x80;
        seed_bytes(&mut state, src_base, &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);

        let outcome = apply_copy(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 8 });

        for (i, &b) in [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88].iter().enumerate() {
            assert_eq!(state.memory[dst_base + i], b, "dst byte {}", i);
        }
        // Source untouched (no overlap).
        assert_eq!(state.memory[src_base], 0x11);
        assert_eq!(state.memory[src_base + 7], 0x88);
        // Boundaries.
        assert_eq!(state.memory[dst_base + 8], 0);
    }

    /// Byte copy, reverse direction (DF=1): SI=0x47, DI=0x87, n=8. Both
    /// pointers walk backward by 1 each iteration.
    #[test]
    fn copy_byte_reverse() {
        let desc = movsb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "SI", "ES", "DS", "flags"]);
        state.set_var("CX", 8);
        state.set_var("DI", 0x87);
        state.set_var("SI", 0x47);
        state.set_var("ES", 0x2000);
        state.set_var("DS", 0x1000);
        state.set_var("flags", 1 << 10); // DF=1
        let src_base = 0x1000 * 16 + 0x40;
        let dst_base = 0x2000 * 16 + 0x80;
        // Source bytes seeded across [0x40..0x48). Iter 0 reads from
        // 0x47, iter 1 from 0x46, ... iter 7 from 0x40. Same for dst.
        seed_bytes(&mut state, src_base, &[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x10, 0x20]);

        let outcome = apply_copy(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 8 });

        // Destination bytes at [0x80..0x88) should be the same source
        // bytes — reverse iteration on parallel pointers preserves byte
        // ordering between equivalent positions.
        for (i, &b) in [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x10, 0x20].iter().enumerate() {
            assert_eq!(state.memory[dst_base + i], b, "dst byte {}", i);
        }
    }

    /// Word copy, forward: copy 4 words from DS:SI to ES:DI.
    #[test]
    fn copy_word_forward() {
        let desc = movsw_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "SI", "ES", "DS", "flags"]);
        state.set_var("CX", 4);
        state.set_var("DI", 0x100);
        state.set_var("SI", 0x80);
        state.set_var("ES", 0x3000);
        state.set_var("DS", 0x1000);
        state.set_var("flags", 0);
        let src_base = 0x1000 * 16 + 0x80;
        let dst_base = 0x3000 * 16 + 0x100;
        seed_bytes(
            &mut state,
            src_base,
            &[0xCE, 0xFA, 0xEF, 0xBE, 0x0D, 0xF0, 0xAD, 0xDE],
        );

        let outcome = apply_copy(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 4 });

        for (i, &b) in [0xCE, 0xFA, 0xEF, 0xBE, 0x0D, 0xF0, 0xAD, 0xDE].iter().enumerate() {
            assert_eq!(state.memory[dst_base + i], b, "dst byte {}", i);
        }
        assert_eq!(state.memory[dst_base + 8], 0);
    }

    /// Word copy, reverse: SI / DI start at the *end* of the data and
    /// walk backward by 2.
    #[test]
    fn copy_word_reverse() {
        let desc = movsw_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "SI", "ES", "DS", "flags"]);
        state.set_var("CX", 3);
        state.set_var("DI", 0x110);
        state.set_var("SI", 0x90);
        state.set_var("ES", 0x3000);
        state.set_var("DS", 0x1000);
        state.set_var("flags", 1 << 10); // DF=1
        let src_base = 0x1000 * 16;
        let dst_base = 0x3000 * 16;
        // Iter 0: SI=0x90 → reads two bytes at [0x90, 0x91].
        // Iter 1: SI=0x8E → reads at [0x8E, 0x8F].
        // Iter 2: SI=0x8C → reads at [0x8C, 0x8D].
        // Same offsets for dst around 0x110.
        seed_bytes(&mut state, src_base + 0x8C, &[0x10, 0x11, 0x12, 0x13, 0x14, 0x15]);

        let outcome = apply_copy(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 3 });

        // dst[0x110..0x112) = src[0x90..0x92) = [0x14, 0x15]
        // dst[0x10E..0x110) = src[0x8E..0x90) = [0x12, 0x13]
        // dst[0x10C..0x10E) = src[0x8C..0x8E) = [0x10, 0x11]
        let want: &[(usize, u8)] = &[
            (0x110, 0x14), (0x111, 0x15),
            (0x10E, 0x12), (0x10F, 0x13),
            (0x10C, 0x10), (0x10D, 0x11),
        ];
        for &(off, b) in want {
            assert_eq!(state.memory[dst_base + off], b, "addr {:#x}", off);
        }
        assert_eq!(state.memory[dst_base + 0x10B], 0);
        assert_eq!(state.memory[dst_base + 0x112], 0);
    }

    /// Counter already zero: applier returns `Applied { iterations: 0 }`
    /// without touching memory.
    #[test]
    fn copy_with_zero_counter_is_noop() {
        let desc = movsb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "SI", "ES", "DS", "flags"]);
        state.set_var("CX", 0);
        state.set_var("DI", 0x80);
        state.set_var("SI", 0x40);
        state.set_var("ES", 0x2000);
        state.set_var("DS", 0x1000);
        state.set_var("flags", 0);
        state.memory[0x1000 * 16 + 0x40] = 0xAB;
        state.memory[0x2000 * 16 + 0x80] = 0xCD; // sentinel, must not be overwritten

        let outcome = apply_copy(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 0 });
        assert_eq!(state.memory[0x2000 * 16 + 0x80], 0xCD);
    }

    /// Overlapping ranges, forward: dst = src + 1, n=4. The hardcoded
    /// path's per-byte-in-iteration-order semantics produce a byte-fill
    /// effect (every dst byte ends up equal to the original src[0]).
    /// The applier must match this byte-for-byte.
    ///
    /// src = [0x11, 0x22, 0x33, 0x44] at offset 0x100..0x104
    /// dst starts at 0x101 (one byte forward of src)
    /// Iter 0: read[0x100]=0x11, write[0x101]=0x11 → src now [0x11, 0x11, 0x33, 0x44]
    /// Iter 1: read[0x101]=0x11, write[0x102]=0x11 → src now [0x11, 0x11, 0x11, 0x44]
    /// Iter 2: read[0x102]=0x11, write[0x103]=0x11 → src now [0x11, 0x11, 0x11, 0x11]
    /// Iter 3: read[0x103]=0x11, write[0x104]=0x11
    /// Final: [0x11, 0x11, 0x11, 0x11, 0x11] at 0x100..0x105
    #[test]
    fn copy_overlapping_forward_propagates_first_byte() {
        let desc = movsb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        // Use the same segment so the overlap is real.
        let mut state = rigged_state(&["CX", "DI", "SI", "ES", "DS", "flags"]);
        state.set_var("CX", 4);
        state.set_var("DI", 0x101);
        state.set_var("SI", 0x100);
        state.set_var("ES", 0x1000);
        state.set_var("DS", 0x1000);
        state.set_var("flags", 0); // DF=0 forward
        let base = 0x1000 * 16;
        seed_bytes(&mut state, base + 0x100, &[0x11, 0x22, 0x33, 0x44]);

        let outcome = apply_copy(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 4 });

        // Same pattern as the hardcoded path's per-byte forward walk:
        // first byte propagates through the destination.
        assert_eq!(state.memory[base + 0x100], 0x11); // src[0] unchanged
        assert_eq!(state.memory[base + 0x101], 0x11);
        assert_eq!(state.memory[base + 0x102], 0x11);
        assert_eq!(state.memory[base + 0x103], 0x11);
        assert_eq!(state.memory[base + 0x104], 0x11);
        assert_eq!(state.memory[base + 0x105], 0); // boundary
    }

    /// Refusal paths.
    #[test]
    fn copy_refuses_non_copy_class() {
        let mut desc = movsb_descriptor();
        desc.bulk_class = BulkClass::Fill;
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "SI", "ES", "DS", "flags"]);
        state.set_var("CX", 1);
        let outcome = apply_copy(&desc, &prog, &mut state, &[]);
        assert!(matches!(outcome, ApplyOutcome::Unsupported(_)));
    }

    #[test]
    fn copy_refuses_when_only_one_pointer() {
        let mut desc = movsb_descriptor();
        desc.pointers.pop();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "SI", "ES", "DS", "flags"]);
        state.set_var("CX", 1);
        let outcome = apply_copy(&desc, &prog, &mut state, &[]);
        assert!(matches!(outcome, ApplyOutcome::Unsupported(_)));
    }

    #[test]
    fn copy_refuses_when_val_indirect_read_missing() {
        let mut desc = movsb_descriptor();
        desc.writes[0].val_indirect_read = None;
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "SI", "ES", "DS", "flags"]);
        state.set_var("CX", 1);
        let outcome = apply_copy(&desc, &prog, &mut state, &[]);
        assert!(matches!(outcome, ApplyOutcome::Unsupported(_)));
    }

    #[test]
    fn copy_refuses_when_indirect_read_seg_missing() {
        let mut desc = movsb_descriptor();
        if let Some(ir) = desc.writes[0].val_indirect_read.as_mut() {
            ir.seg_property = None;
        }
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "SI", "ES", "DS", "flags"]);
        state.set_var("CX", 1);
        let outcome = apply_copy(&desc, &prog, &mut state, &[]);
        assert!(matches!(outcome, ApplyOutcome::Unsupported(_)));
    }

    #[test]
    fn copy_refuses_when_pointers_have_different_step() {
        let mut desc = movsb_descriptor();
        desc.pointers[1].base_step = 2;
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "SI", "ES", "DS", "flags"]);
        state.set_var("CX", 1);
        let outcome = apply_copy(&desc, &prog, &mut state, &[]);
        assert!(matches!(outcome, ApplyOutcome::Unsupported(_)));
    }

    /// Genericity probe: a Copy descriptor with completely unrelated
    /// property names (no DI/SI/ES/DS/CX/flags) must produce identical
    /// behaviour. Verifies the applier reads no name as text — every
    /// resolution is whole-name slot lookup, every structural decision
    /// is by `Expr` shape and slot identity.
    #[test]
    fn copy_genericity_unrelated_names() {
        let desc = LoopDescriptor {
            key_property: "--opaque_key".to_string(),
            key_value: 0xBC, // arbitrary
            ip_property: "--ip_x".to_string(),
            ip_self_property: "--ip_x_prev".to_string(),
            ip_advance_literal: 1,
            predicate_properties: vec!["--cont_x".to_string()],
            predicate: StyleTest::Single {
                property: "--cont_x".to_string(),
                value: Expr::Literal(1.0),
            },
            predicate_means_stay: true,
            counter: Some(CounterEntry {
                property: "--alpha".to_string(),
                self_property: "--alpha_prev".to_string(),
                step: 1,
            }),
            pointers: vec![
                PointerEntry {
                    property: "--beta".to_string(),
                    self_property: "--beta".to_string(),
                    base_step: 1,
                    flag_property: "--gamma".to_string(),
                    flag_bit: 5, // arbitrary
                },
                PointerEntry {
                    property: "--zeta".to_string(),
                    self_property: "--zeta".to_string(),
                    base_step: 1,
                    flag_property: "--gamma".to_string(),
                    flag_bit: 5,
                },
            ],
            writes: vec![WriteEntry {
                addr_property: "--w_addr".to_string(),
                val_property: "--w_val".to_string(),
                addr_expr: Expr::Literal(0.0),
                val_expr: Expr::Var {
                    name: "--theta".to_string(),
                    fallback: None,
                },
                addr_decomposition: Some(("--epsilon".to_string(), "--beta".to_string())),
                val_indirect_read: Some(IndirectRead {
                    seg_property: Some("--eta".to_string()),
                    seg_times_sixteen: true,
                    pointer_property: "--zeta".to_string(),
                    intermediate_property: "--theta".to_string(),
                }),
            }],
            flag_conditioned: false,
            bulk_class: BulkClass::Copy,
            per_iter_cycles: Some(CycleCharge { property: "--cycle_x".to_string(), per_iter: 17 }),
            ip_extra_advance_slot: Some("--prefixLen".to_string()),
            comparison_shape: None,
            precondition: None,
        };
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state =
            rigged_state(&["alpha", "beta", "zeta", "epsilon", "eta", "gamma", "theta"]);
        state.set_var("alpha", 5);
        state.set_var("beta", 0x10); // dst pointer
        state.set_var("zeta", 0x20); // src pointer
        state.set_var("epsilon", 0x4000); // dst seg
        state.set_var("eta", 0x2000); // src seg
        state.set_var("gamma", 0); // bit 5 clear → forward
        let src_base = 0x2000 * 16 + 0x20;
        let dst_base = 0x4000 * 16 + 0x10;
        seed_bytes(&mut state, src_base, &[0xA1, 0xA2, 0xA3, 0xA4, 0xA5]);

        let outcome = apply_copy(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 5 });
        for (i, &b) in [0xA1, 0xA2, 0xA3, 0xA4, 0xA5].iter().enumerate() {
            assert_eq!(state.memory[dst_base + i], b);
        }
    }

    /// Dual-mode equivalence (byte): drive the same setup through the
    /// applier and through a direct mirror of the hardcoded MOVSB loop
    /// on independent State clones. Memory must be byte-for-byte
    /// identical.
    #[test]
    fn copy_equivalence_byte_matches_direct_loop() {
        let desc = movsb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let names = &["CX", "DI", "SI", "ES", "DS", "flags"];
        let mut state_a = rigged_state(names);
        let mut state_b = rigged_state(names);
        let n = 64usize;
        let src_base = 0x1000 * 16 + 0x100;
        for s in [&mut state_a, &mut state_b] {
            s.set_var("CX", n as i32);
            s.set_var("DI", 0x200);
            s.set_var("SI", 0x100);
            s.set_var("ES", 0x4000);
            s.set_var("DS", 0x1000);
            s.set_var("flags", 0);
            // Seed src.
            for k in 0..n {
                s.memory[src_base + k] = (0x10 + k as u8) ^ 0x5A;
            }
        }

        // Path A: applier.
        let outcome = apply_copy(&desc, &prog, &mut state_a, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: n as i32 });

        // Path B: mirror the hardcoded MOVSB loop from compile.rs.
        let es_base = 0x4000i64 * 16;
        let ds_base = 0x1000i64 * 16;
        let di = 0x200i64;
        let si = 0x100i64;
        for k in 0..(n as i64) {
            let s_addr = ds_base + si + k;
            let d_addr = es_base + di + k;
            let b = (state_b.read_mem(s_addr as i32) & 0xFF) as u8;
            bulk_store_byte(&mut state_b, d_addr, b);
        }

        assert_eq!(
            state_a.memory, state_b.memory,
            "applier byte-copy must match direct loop byte-for-byte"
        );
    }

    /// Dual-mode equivalence (word, reverse): MOVSW DF=1 against a
    /// direct mirror of the hardcoded MOVSW loop.
    #[test]
    fn copy_equivalence_word_reverse_matches_direct_loop() {
        let desc = movsw_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let names = &["CX", "DI", "SI", "ES", "DS", "flags"];
        let mut state_a = rigged_state(names);
        let mut state_b = rigged_state(names);
        let n = 50i64;
        let src_base = 0x2000 * 16 + 0x500;
        for s in [&mut state_a, &mut state_b] {
            s.set_var("CX", n as i32);
            s.set_var("DI", 0x600); // walks backward from 0x600
            s.set_var("SI", 0x500); // walks backward from 0x500
            s.set_var("ES", 0x5000);
            s.set_var("DS", 0x2000);
            s.set_var("flags", 1 << 10); // DF=1
            // Seed src across [0x500-2*49 .. 0x500+2) = [0x43E .. 0x502).
            for k in 0..(n * 2) {
                let off = 0x500 - (n - 1) * 2 + k;
                s.memory[(src_base as i64 - (n - 1) * 2 + k) as usize] =
                    ((off as u8) ^ 0xC3) & 0xFF;
            }
        }

        // Path A: applier.
        let outcome = apply_copy(&desc, &prog, &mut state_a, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: n as i32 });

        // Path B: mirror the hardcoded MOVSW reverse loop.
        let es_base = 0x5000i64 * 16;
        let ds_base = 0x2000i64 * 16;
        let di = 0x600i64;
        let si = 0x500i64;
        for k in 0..n {
            let off = -k * 2; // DF=1
            let s_addr = ds_base + si + off;
            let d_addr = es_base + di + off;
            let lo = (state_b.read_mem(s_addr as i32) & 0xFF) as u8;
            let hi = (state_b.read_mem((s_addr + 1) as i32) & 0xFF) as u8;
            bulk_store_byte(&mut state_b, d_addr, lo);
            bulk_store_byte(&mut state_b, d_addr + 1, hi);
        }

        assert_eq!(
            state_a.memory, state_b.memory,
            "applier word-copy reverse must match direct loop byte-for-byte"
        );
    }

    // -----------------------------------------------------------------
    // Step 6 — apply_read_only (BulkClass::ReadOnly: LODS / CMPS / SCAS).
    // -----------------------------------------------------------------

    /// LODSB-shape: one byte-step pointer (`--SI`), no writes,
    /// `flag_conditioned=false`. The recogniser would emit this for any
    /// cabinet whose dispatch family produces a single-pointer no-write
    /// loop without an early-exit predicate.
    fn lodsb_descriptor() -> LoopDescriptor {
        LoopDescriptor {
            key_property: "--opcode".to_string(),
            key_value: 0xAC,
            ip_property: "--IP".to_string(),
            ip_self_property: "--IP_prev".to_string(),
            ip_advance_literal: 1,
            predicate_properties: vec!["--repContinue".to_string()],
            predicate: StyleTest::Single {
                property: "--repContinue".to_string(),
                value: Expr::Literal(1.0),
            },
            predicate_means_stay: true,
            counter: Some(CounterEntry {
                property: "--CX".to_string(),
                self_property: "--CX_prev".to_string(),
                step: 1,
            }),
            pointers: vec![PointerEntry {
                property: "--SI".to_string(),
                self_property: "--SI".to_string(),
                base_step: 1,
                flag_property: "--flags".to_string(),
                flag_bit: 10,
            }],
            writes: vec![],
            flag_conditioned: false,
            bulk_class: BulkClass::ReadOnly,
            per_iter_cycles: Some(CycleCharge { property: "--cycleCount".to_string(), per_iter: 15 }),
            ip_extra_advance_slot: Some("--prefixLen".to_string()),
            comparison_shape: None,
            precondition: None,
        }
    }

    /// CMPSB-shape: two byte-step pointers (`--DI` dst at index 0,
    /// `--SI` src at index 1), no writes, `flag_conditioned=true`.
    fn cmpsb_descriptor() -> LoopDescriptor {
        LoopDescriptor {
            key_property: "--opcode".to_string(),
            key_value: 0xA6,
            ip_property: "--IP".to_string(),
            ip_self_property: "--IP_prev".to_string(),
            ip_advance_literal: 1,
            predicate_properties: vec!["--repContinue".to_string(), "--repZF".to_string()],
            predicate: StyleTest::Single {
                property: "--repContinue".to_string(),
                value: Expr::Literal(1.0),
            },
            predicate_means_stay: true,
            counter: Some(CounterEntry {
                property: "--CX".to_string(),
                self_property: "--CX_prev".to_string(),
                step: 1,
            }),
            pointers: vec![
                PointerEntry {
                    property: "--DI".to_string(),
                    self_property: "--DI".to_string(),
                    base_step: 1,
                    flag_property: "--flags".to_string(),
                    flag_bit: 10,
                },
                PointerEntry {
                    property: "--SI".to_string(),
                    self_property: "--SI".to_string(),
                    base_step: 1,
                    flag_property: "--flags".to_string(),
                    flag_bit: 10,
                },
            ],
            writes: vec![],
            flag_conditioned: true,
            bulk_class: BulkClass::ReadOnly,
            per_iter_cycles: Some(CycleCharge { property: "--cycleCount".to_string(), per_iter: 22 }),
            ip_extra_advance_slot: Some("--prefixLen".to_string()),
            comparison_shape: Some(crate::pattern::loop_descriptor::ComparisonShape {
                width: 1,
                dst_seg_property: "--ES".to_string(),
                dst_seg_times_sixteen: true,
                dst_ptr_property: "--DI".to_string(),
                source: crate::pattern::loop_descriptor::ComparisonSource::Pointer {
                    seg_property: "--DS".to_string(),
                    seg_times_sixteen: true,
                    ptr_property: "--SI".to_string(),
                },
                rep_type_property: Some("--repType".to_string()),
            }),
            precondition: None,
        }
    }

    /// CMPSW-shape: word-step variant of `cmpsb_descriptor`.
    fn cmpsw_descriptor() -> LoopDescriptor {
        let mut d = cmpsb_descriptor();
        d.key_value = 0xA7;
        d.pointers[0].base_step = 2;
        d.pointers[1].base_step = 2;
        if let Some(cs) = d.comparison_shape.as_mut() {
            cs.width = 2;
        }
        d
    }

    /// SCASB-shape: one byte-step pointer (`--DI`), no writes,
    /// `flag_conditioned=true`.
    fn scasb_descriptor() -> LoopDescriptor {
        LoopDescriptor {
            key_property: "--opcode".to_string(),
            key_value: 0xAE,
            ip_property: "--IP".to_string(),
            ip_self_property: "--IP_prev".to_string(),
            ip_advance_literal: 1,
            predicate_properties: vec!["--repContinue".to_string(), "--repZF".to_string()],
            predicate: StyleTest::Single {
                property: "--repContinue".to_string(),
                value: Expr::Literal(1.0),
            },
            predicate_means_stay: true,
            counter: Some(CounterEntry {
                property: "--CX".to_string(),
                self_property: "--CX_prev".to_string(),
                step: 1,
            }),
            pointers: vec![PointerEntry {
                property: "--DI".to_string(),
                self_property: "--DI".to_string(),
                base_step: 1,
                flag_property: "--flags".to_string(),
                flag_bit: 10,
            }],
            writes: vec![],
            flag_conditioned: true,
            bulk_class: BulkClass::ReadOnly,
            per_iter_cycles: Some(CycleCharge { property: "--cycleCount".to_string(), per_iter: 15 }),
            ip_extra_advance_slot: Some("--prefixLen".to_string()),
            comparison_shape: Some(crate::pattern::loop_descriptor::ComparisonShape {
                width: 1,
                dst_seg_property: "--ES".to_string(),
                dst_seg_times_sixteen: true,
                dst_ptr_property: "--DI".to_string(),
                source: crate::pattern::loop_descriptor::ComparisonSource::Accumulator {
                    byte_property: "--AL".to_string(),
                    word_property: "--AX".to_string(),
                },
                rep_type_property: Some("--repType".to_string()),
            }),
            precondition: None,
        }
    }

    /// SCASW-shape: word-step variant of `scasb_descriptor`.
    fn scasw_descriptor() -> LoopDescriptor {
        let mut d = scasb_descriptor();
        d.key_value = 0xAF;
        d.pointers[0].base_step = 2;
        if let Some(cs) = d.comparison_shape.as_mut() {
            cs.width = 2;
        }
        d
    }

    /// LODSB forward: 16 iterations. Memory must be untouched; outcome
    /// must report 16 iterations (no early exit possible).
    #[test]
    fn read_only_lods_byte_forward() {
        let desc = lodsb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "SI", "DS", "flags"]);
        state.set_var("CX", 16);
        state.set_var("SI", 0x100);
        state.set_var("DS", 0x1000);
        state.set_var("flags", 0); // DF=0
        // Seed memory; after applier runs we re-check those exact bytes.
        let base = 0x1000 * 16 + 0x100;
        for i in 0..16 {
            state.memory[base + i] = 0x40 + i as u8;
        }
        let memory_before = state.memory.clone();

        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 16 });
        assert_eq!(state.memory, memory_before, "LODS must not mutate memory");
    }

    /// LODSB reverse direction (DF=1): same memory-untouched contract.
    /// Iteration count still equals N (no early exit for LODS).
    #[test]
    fn read_only_lods_byte_reverse() {
        let desc = lodsb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "SI", "DS", "flags"]);
        state.set_var("CX", 8);
        state.set_var("SI", 0x80);
        state.set_var("DS", 0x2000);
        state.set_var("flags", 1 << 10); // DF=1
        let memory_before = state.memory.clone();

        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 8 });
        assert_eq!(state.memory, memory_before);
    }

    /// LODS with zero counter is a noop. (Mirrors `apply_fill` /
    /// `apply_copy` semantics for cx<=0.)
    #[test]
    fn read_only_lods_zero_counter_is_noop() {
        let desc = lodsb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "SI", "DS", "flags"]);
        state.set_var("CX", 0);
        state.set_var("SI", 0x100);
        state.set_var("DS", 0x1000);

        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 0 });
    }

    /// CMPSB equal-throughout: source and destination ranges contain
    /// identical bytes. With REPE (`--repType=1`) the loop runs to
    /// counter exhaustion (no inequality to exit on). Memory unchanged.
    #[test]
    fn read_only_cmps_byte_equal_throughout_repe_completes() {
        let desc = cmpsb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state =
            rigged_state(&["CX", "DI", "SI", "ES", "DS", "AL", "AX", "flags", "repType"]);
        state.set_var("CX", 12);
        state.set_var("DI", 0x200);
        state.set_var("SI", 0x100);
        state.set_var("ES", 0x4000);
        state.set_var("DS", 0x1000);
        state.set_var("flags", 0); // DF=0
        state.set_var("repType", 1); // REPE
        // Equal bytes across both ranges.
        let dst_base = 0x4000 * 16 + 0x200;
        let src_base = 0x1000 * 16 + 0x100;
        for i in 0..12 {
            state.memory[dst_base + i] = 0x55;
            state.memory[src_base + i] = 0x55;
        }
        let memory_before = state.memory.clone();

        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        // REPE walks all 12 iters; src==dst throughout.
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 12 });
        assert_eq!(state.memory, memory_before, "CMPS must not mutate memory");
    }

    /// CMPSB mismatch mid-walk under REPE: the loop exits as soon as a
    /// pair of bytes differ.
    #[test]
    fn read_only_cmps_byte_repe_early_exit_on_mismatch() {
        let desc = cmpsb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state =
            rigged_state(&["CX", "DI", "SI", "ES", "DS", "AL", "AX", "flags", "repType"]);
        state.set_var("CX", 16);
        state.set_var("DI", 0x300);
        state.set_var("SI", 0x100);
        state.set_var("ES", 0x4000);
        state.set_var("DS", 0x1000);
        state.set_var("flags", 0);
        state.set_var("repType", 1); // REPE
        let dst_base = 0x4000 * 16 + 0x300;
        let src_base = 0x1000 * 16 + 0x100;
        for i in 0..16 {
            state.memory[dst_base + i] = 0xAA;
            state.memory[src_base + i] = 0xAA;
        }
        // Inject mismatch at offset 5.
        state.memory[src_base + 5] = 0xBB;

        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        // Iters 0..=5 run (iter 5 sees the mismatch and exits after
        // counting). Iter 5 increments `iters` to 6 then breaks.
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 6 });
    }

    /// CMPSB mismatch mid-walk under REPNE: REPNE exits on equality. So
    /// the loop runs through inequality and stops the first time
    /// src==dst.
    #[test]
    fn read_only_cmps_byte_repne_early_exit_on_match() {
        let desc = cmpsb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state =
            rigged_state(&["CX", "DI", "SI", "ES", "DS", "AL", "AX", "flags", "repType"]);
        state.set_var("CX", 16);
        state.set_var("DI", 0x400);
        state.set_var("SI", 0x100);
        state.set_var("ES", 0x4000);
        state.set_var("DS", 0x1000);
        state.set_var("flags", 0);
        state.set_var("repType", 2); // REPNE
        let dst_base = 0x4000 * 16 + 0x400;
        let src_base = 0x1000 * 16 + 0x100;
        for i in 0..16 {
            state.memory[dst_base + i] = 0xAA;
            state.memory[src_base + i] = 0xBB;
        }
        // Inject a match at offset 3.
        state.memory[src_base + 3] = 0xAA;

        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        // REPNE: iters 0,1,2 see inequality, iter 3 sees equality and
        // breaks → 4 iterations total.
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 4 });
    }

    /// CMPSW (word-step), forward, all-equal: counter exhausted.
    #[test]
    fn read_only_cmps_word_equal_throughout_repe_completes() {
        let desc = cmpsw_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state =
            rigged_state(&["CX", "DI", "SI", "ES", "DS", "AL", "AX", "flags", "repType"]);
        state.set_var("CX", 4);
        state.set_var("DI", 0x100);
        state.set_var("SI", 0x80);
        state.set_var("ES", 0x4000);
        state.set_var("DS", 0x1000);
        state.set_var("flags", 0);
        state.set_var("repType", 1); // REPE
        let dst_base = 0x4000 * 16 + 0x100;
        let src_base = 0x1000 * 16 + 0x80;
        // 4 words of 0xBEEF in both.
        for i in 0..4 {
            state.memory[dst_base + i * 2] = 0xEF;
            state.memory[dst_base + i * 2 + 1] = 0xBE;
            state.memory[src_base + i * 2] = 0xEF;
            state.memory[src_base + i * 2 + 1] = 0xBE;
        }
        let memory_before = state.memory.clone();

        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 4 });
        assert_eq!(state.memory, memory_before);
    }

    /// SCASB (single pointer, accumulator-vs-memory) under REPNE: scan
    /// for a target byte. Loop exits at the first match.
    #[test]
    fn read_only_scas_byte_repne_finds_target() {
        let desc = scasb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state =
            rigged_state(&["CX", "DI", "ES", "DS", "AL", "AX", "flags", "repType"]);
        state.set_var("CX", 32);
        state.set_var("DI", 0x500);
        state.set_var("ES", 0x4000);
        state.set_var("DS", 0); // SCAS doesn't read DS but the resolver still tries
        state.set_var("AL", 0x42);
        state.set_var("flags", 0);
        state.set_var("repType", 2); // REPNE — exit on equality
        let dst_base = 0x4000 * 16 + 0x500;
        // Fill with 0xFF, plant target at offset 7.
        for i in 0..32 {
            state.memory[dst_base + i] = 0xFF;
        }
        state.memory[dst_base + 7] = 0x42;

        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        // REPNE: iters 0..6 see 0xFF != 0x42, iter 7 sees 0x42 == 0x42
        // and breaks → 8 iterations total.
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 8 });
    }

    /// SCASB no match: loop runs through full counter.
    #[test]
    fn read_only_scas_byte_repne_no_match_full_counter() {
        let desc = scasb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state =
            rigged_state(&["CX", "DI", "ES", "DS", "AL", "AX", "flags", "repType"]);
        state.set_var("CX", 10);
        state.set_var("DI", 0x600);
        state.set_var("ES", 0x4000);
        state.set_var("AL", 0xCC);
        state.set_var("flags", 0);
        state.set_var("repType", 2); // REPNE
        let dst_base = 0x4000 * 16 + 0x600;
        for i in 0..10 {
            state.memory[dst_base + i] = 0xDD; // never matches AL=0xCC
        }
        let memory_before = state.memory.clone();

        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 10 });
        assert_eq!(state.memory, memory_before);
    }

    /// SCASW (word, accumulator AX): same shape, word reads.
    #[test]
    fn read_only_scas_word_repne_finds_target() {
        let desc = scasw_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state =
            rigged_state(&["CX", "DI", "ES", "DS", "AL", "AX", "flags", "repType"]);
        state.set_var("CX", 8);
        state.set_var("DI", 0x100);
        state.set_var("ES", 0x4000);
        state.set_var("AX", 0xDEAD);
        state.set_var("flags", 0);
        state.set_var("repType", 2); // REPNE
        let dst_base = 0x4000 * 16 + 0x100;
        // Fill with 0xBEEF; plant DEAD at word index 4.
        for i in 0..8 {
            state.memory[dst_base + i * 2] = 0xEF;
            state.memory[dst_base + i * 2 + 1] = 0xBE;
        }
        state.memory[dst_base + 4 * 2] = 0xAD;
        state.memory[dst_base + 4 * 2 + 1] = 0xDE;

        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        // 5 iters: indices 0..3 mismatch, index 4 matches.
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 5 });
    }

    /// Refusal paths.
    #[test]
    fn read_only_refuses_non_read_only_class() {
        let mut desc = lodsb_descriptor();
        desc.bulk_class = BulkClass::Fill;
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "SI", "DS", "flags"]);
        state.set_var("CX", 1);
        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        assert!(matches!(outcome, ApplyOutcome::Unsupported(_)));
    }

    #[test]
    fn read_only_refuses_when_writes_present() {
        let mut desc = lodsb_descriptor();
        desc.writes.push(WriteEntry {
            addr_property: "--mAddr0".to_string(),
            val_property: "--mVal0".to_string(),
            addr_expr: Expr::Literal(0.0),
            val_expr: Expr::Literal(0.0),
            addr_decomposition: None,
            val_indirect_read: None,
        });
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "SI", "DS", "flags"]);
        state.set_var("CX", 1);
        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        assert!(matches!(outcome, ApplyOutcome::Unsupported(_)));
    }

    #[test]
    fn read_only_refuses_when_no_pointers() {
        let mut desc = lodsb_descriptor();
        desc.pointers.clear();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "SI", "DS", "flags"]);
        state.set_var("CX", 1);
        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        assert!(matches!(outcome, ApplyOutcome::Unsupported(_)));
    }

    /// LODS shape requires exactly 1 pointer. A descriptor with
    /// `flag_conditioned=false` and 2 pointers is structurally
    /// inconsistent (there's no x86 self-loop with 2 pointers and no
    /// flag-conditioned exit) — bail.
    #[test]
    fn read_only_refuses_lods_shape_with_two_pointers() {
        let mut desc = lodsb_descriptor();
        desc.pointers.push(PointerEntry {
            property: "--DI".to_string(),
            self_property: "--DI".to_string(),
            base_step: 1,
            flag_property: "--flags".to_string(),
            flag_bit: 10,
        });
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "SI", "DI", "DS", "flags"]);
        state.set_var("CX", 4);
        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        assert!(matches!(outcome, ApplyOutcome::Unsupported(_)));
    }

    /// CMPS-shape pointers must agree on step and direction flag.
    #[test]
    fn read_only_refuses_cmps_with_mismatched_steps() {
        let mut desc = cmpsb_descriptor();
        desc.pointers[1].base_step = 2;
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state =
            rigged_state(&["CX", "DI", "SI", "ES", "DS", "AL", "AX", "flags", "repType"]);
        state.set_var("CX", 4);
        state.set_var("repType", 1);
        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        assert!(matches!(outcome, ApplyOutcome::Unsupported(_)));
    }

    /// Genericity probe: a SCAS-shape descriptor with completely
    /// unrelated property names. The applier reads `--ES`, `--AL`,
    /// `--AX`, `--repType` by literal name today (the documented
    /// cardinal-rule wart), so this probe verifies the LODS path's
    /// genericity — which is the structurally clean one.
    #[test]
    fn read_only_lods_genericity_unrelated_names() {
        let desc = LoopDescriptor {
            key_property: "--opaque_key".to_string(),
            key_value: 0x99,
            ip_property: "--ip_x".to_string(),
            ip_self_property: "--ip_x_prev".to_string(),
            ip_advance_literal: 1,
            predicate_properties: vec!["--cont_x".to_string()],
            predicate: StyleTest::Single {
                property: "--cont_x".to_string(),
                value: Expr::Literal(1.0),
            },
            predicate_means_stay: true,
            counter: Some(CounterEntry {
                property: "--alpha".to_string(),
                self_property: "--alpha_prev".to_string(),
                step: 1,
            }),
            pointers: vec![PointerEntry {
                property: "--beta".to_string(),
                self_property: "--beta".to_string(),
                base_step: 1,
                flag_property: "--gamma".to_string(),
                flag_bit: 7, // arbitrary
            }],
            writes: vec![],
            flag_conditioned: false,
            bulk_class: BulkClass::ReadOnly,
            per_iter_cycles: Some(CycleCharge { property: "--cycle_x".to_string(), per_iter: 15 }),
            ip_extra_advance_slot: Some("--prefixLen".to_string()),
            comparison_shape: None,
            precondition: None,
        };
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["alpha", "beta", "gamma"]);
        state.set_var("alpha", 11);
        state.set_var("beta", 0x2000);
        state.set_var("gamma", 0); // bit 7 clear → forward
        let memory_before = state.memory.clone();

        let outcome = apply_read_only(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 11 });
        assert_eq!(state.memory, memory_before);
    }

    /// Dual-mode equivalence (SCASB, REPNE): drive the same setup
    /// through the applier and through a direct mirror of the hardcoded
    /// SCAS loop on independent State clones. Memory must be byte-for-
    /// byte identical (both paths leave memory untouched), and
    /// iteration counts must match.
    #[test]
    fn read_only_scas_equivalence_byte_repne() {
        let desc = scasb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let names = &["CX", "DI", "ES", "DS", "AL", "AX", "flags", "repType"];
        let mut state_a = rigged_state(names);
        let mut state_b = rigged_state(names);
        let n = 64i32;
        for s in [&mut state_a, &mut state_b] {
            s.set_var("CX", n);
            s.set_var("DI", 0x80);
            s.set_var("ES", 0x4000);
            s.set_var("AL", 0x77);
            s.set_var("flags", 0);
            s.set_var("repType", 2); // REPNE
            let dst_base = 0x4000 * 16 + 0x80;
            for k in 0..(n as usize) {
                s.memory[dst_base + k] = (k as u8).wrapping_mul(3); // varied bytes
            }
            s.memory[dst_base + 23] = 0x77; // first match at offset 23
        }
        let mem_a_before = state_a.memory.clone();

        // Path A: applier.
        let outcome_a = apply_read_only(&desc, &prog, &mut state_a, &[]);

        // Path B: mirror the hardcoded SCAS loop semantics.
        let mut iters_b = 0i32;
        let mut di_b = 0x80i64;
        let es_base = 0x4000i64 * 16;
        let acc = 0x77;
        for _ in 0..n {
            let dst_lin = (es_base + di_b) & 0xFFFFF;
            let dst_v = state_b.read_mem(dst_lin as i32) & 0xFF;
            di_b = (di_b + 1) & 0xFFFF;
            iters_b += 1;
            // REPNE: exit on equality.
            if acc == dst_v {
                break;
            }
        }

        assert_eq!(
            outcome_a,
            ApplyOutcome::Applied { iterations: iters_b },
            "applier and direct loop must report same iteration count",
        );
        // Both paths leave memory unchanged.
        assert_eq!(state_a.memory, mem_a_before);
        assert_eq!(state_a.memory, state_b.memory);
    }

    /// Dual-mode equivalence (CMPSB, REPE) — full memory diff against a
    /// direct mirror.
    #[test]
    fn read_only_cmps_equivalence_byte_repe() {
        let desc = cmpsb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let names = &["CX", "DI", "SI", "ES", "DS", "AL", "AX", "flags", "repType"];
        let mut state_a = rigged_state(names);
        let mut state_b = rigged_state(names);
        let n = 32i32;
        for s in [&mut state_a, &mut state_b] {
            s.set_var("CX", n);
            s.set_var("DI", 0x200);
            s.set_var("SI", 0x100);
            s.set_var("ES", 0x4000);
            s.set_var("DS", 0x1000);
            s.set_var("flags", 0);
            s.set_var("repType", 1); // REPE
            let dst_base = 0x4000 * 16 + 0x200;
            let src_base = 0x1000 * 16 + 0x100;
            for k in 0..(n as usize) {
                s.memory[dst_base + k] = 0x33;
                s.memory[src_base + k] = 0x33;
            }
            // Inject divergence at offset 11.
            s.memory[src_base + 11] = 0x44;
        }
        let mem_a_before = state_a.memory.clone();

        let outcome_a = apply_read_only(&desc, &prog, &mut state_a, &[]);

        // Direct mirror.
        let mut iters_b = 0i32;
        let mut di_b = 0x200i64;
        let mut si_b = 0x100i64;
        let es_base = 0x4000i64 * 16;
        let ds_base = 0x1000i64 * 16;
        for _ in 0..n {
            let dst_v = state_b.read_mem(((es_base + di_b) & 0xFFFFF) as i32) & 0xFF;
            let src_v = state_b.read_mem(((ds_base + si_b) & 0xFFFFF) as i32) & 0xFF;
            di_b = (di_b + 1) & 0xFFFF;
            si_b = (si_b + 1) & 0xFFFF;
            iters_b += 1;
            // REPE: exit on inequality.
            if src_v != dst_v {
                break;
            }
        }

        assert_eq!(
            outcome_a,
            ApplyOutcome::Applied { iterations: iters_b },
        );
        assert_eq!(state_a.memory, mem_a_before);
        assert_eq!(state_a.memory, state_b.memory);
    }

    // -----------------------------------------------------------------
    // Step 7: state-var commit tests.
    //
    // These exercise the new `CommitMode::Full` codepath. The memory
    // mutations are already covered by the existing tests; here we
    // verify DI/SI/CX/IP/cycleCount/flags reach the same post-state the
    // hardcoded `rep_fast_forward` would produce.
    // -----------------------------------------------------------------

    /// `CommitMode::Full` on STOSB: DI, CX, IP, cycleCount must all
    /// advance to the values the hardcoded path would write.
    #[test]
    fn commit_full_stosb_updates_state_vars() {
        let desc = stosb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&[
            "CX", "DI", "ES", "AL", "flags", "IP", "cycleCount", "prefixLen",
        ]);
        state.set_var("CX", 32);
        state.set_var("DI", 0x100);
        state.set_var("ES", 0x2000);
        state.set_var("AL", 0x55);
        state.set_var("flags", 0); // DF=0
        state.set_var("IP", 0x1234);
        state.set_var("cycleCount", 1000);
        state.set_var("prefixLen", 1);

        let outcome = apply_fill_with_commit(&desc, &prog, &mut state, &[], CommitMode::Full);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 32 });
        // Pointer advance: DI = 0x100 + 32 * 1 = 0x120
        assert_eq!(state.get_var("DI"), Some(0x120));
        // Counter drains.
        assert_eq!(state.get_var("CX"), Some(0));
        // IP advance: 0x1234 + ip_advance_literal(=1) + prefixLen(=1) = 0x1236
        assert_eq!(state.get_var("IP"), Some(0x1236));
        // cycleCount: 1000 + 32 * 10 (STOS per_iter) = 1320
        assert_eq!(state.get_var("cycleCount"), Some(1320));
    }

    /// Genericity probe for the commit step: a cabinet that names its
    /// instruction-pointer slot `--pc_y` and its cycle counter
    /// `--zorch` gets its OWN slots advanced — the applier carries no
    /// hardcoded `IP` / `cycleCount` names. Before the descriptor
    /// carried these names, this exact test could not pass: the commit
    /// wrote literal `"IP"` / `"cycleCount"` and would silently no-op
    /// here.
    #[test]
    fn commit_full_renamed_ip_and_cycle_slots_commit_to_cabinet_names() {
        let mut desc = stosb_descriptor();
        desc.ip_property = "--pc_y".to_string();
        desc.per_iter_cycles =
            Some(CycleCharge { property: "--zorch".to_string(), per_iter: 10 });
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&[
            "CX", "DI", "ES", "AL", "flags", "pc_y", "zorch", "prefixLen",
        ]);
        state.set_var("CX", 32);
        state.set_var("DI", 0x100);
        state.set_var("ES", 0x2000);
        state.set_var("AL", 0x55);
        state.set_var("flags", 0); // DF=0
        state.set_var("pc_y", 0x1234);
        state.set_var("zorch", 1000);
        state.set_var("prefixLen", 1);

        let outcome = apply_fill_with_commit(&desc, &prog, &mut state, &[], CommitMode::Full);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 32 });
        // The cabinet's own slots advance exactly as IP/cycleCount would.
        assert_eq!(state.get_var("pc_y"), Some(0x1236));
        assert_eq!(state.get_var("zorch"), Some(1320));
        // And nothing materialised under the x86 names.
        assert_eq!(state.get_var("IP"), None);
        assert_eq!(state.get_var("cycleCount"), None);
    }

    /// `CommitMode::Full` on STOSW with DF=1 (reverse): pointer arithmetic
    /// flips sign correctly.
    #[test]
    fn commit_full_stosw_reverse_dec_pointer() {
        let desc = stosw_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&[
            "CX", "DI", "ES", "AL", "AH", "flags", "IP", "cycleCount", "prefixLen",
        ]);
        state.set_var("CX", 8);
        state.set_var("DI", 0x80);
        state.set_var("ES", 0x3000);
        state.set_var("AL", 0x11);
        state.set_var("AH", 0x22);
        state.set_var("flags", 1 << 10); // DF=1
        state.set_var("IP", 0x4000);
        state.set_var("cycleCount", 0);
        state.set_var("prefixLen", 1);

        let outcome = apply_fill_with_commit(&desc, &prog, &mut state, &[], CommitMode::Full);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 8 });
        // Reverse: DI = 0x80 + 8 * (-2) = 0x70
        assert_eq!(state.get_var("DI"), Some(0x70));
        assert_eq!(state.get_var("CX"), Some(0));
        assert_eq!(state.get_var("IP"), Some(0x4002));
        assert_eq!(state.get_var("cycleCount"), Some(80)); // 8 * 10
    }

    /// `CommitMode::Full` on MOVSB: both DI and SI advance, CX drains,
    /// IP and cycleCount move.
    #[test]
    fn commit_full_movsb_updates_both_pointers() {
        let desc = movsb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&[
            "CX", "DI", "SI", "ES", "DS", "flags", "IP", "cycleCount", "prefixLen",
        ]);
        state.set_var("CX", 16);
        state.set_var("DI", 0x200);
        state.set_var("SI", 0x800);
        state.set_var("ES", 0x4000);
        state.set_var("DS", 0x5000);
        state.set_var("flags", 0); // DF=0
        state.set_var("IP", 0x100);
        state.set_var("cycleCount", 0);
        state.set_var("prefixLen", 1);

        // Seed source bytes so reads succeed (we don't care about the
        // memory diff in this test, only state vars).
        for i in 0..16 {
            state.memory[(0x5000 * 16 + 0x800 + i) as usize] = (i & 0xFF) as u8;
        }

        let outcome = apply_copy_with_commit(&desc, &prog, &mut state, &[], CommitMode::Full);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 16 });
        assert_eq!(state.get_var("DI"), Some(0x210));
        assert_eq!(state.get_var("SI"), Some(0x810));
        assert_eq!(state.get_var("CX"), Some(0));
        assert_eq!(state.get_var("IP"), Some(0x102));
        // 16 iters * 17 cycles per MOVS iter = 272
        assert_eq!(state.get_var("cycleCount"), Some(272));
    }

    /// `CommitMode::Full` on SCASB REPNE: CX drains as iterations consume,
    /// DI advances, flags reflect the last comparison.
    #[test]
    fn commit_full_scasb_repne_finds_match() {
        let desc = scasb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&[
            "CX", "DI", "ES", "AL", "flags", "IP", "cycleCount", "prefixLen", "repType",
        ]);
        state.set_var("CX", 16);
        state.set_var("DI", 0x100);
        state.set_var("ES", 0x6000);
        state.set_var("AL", 0xAA);
        state.set_var("flags", 0); // DF=0, ZF=0
        state.set_var("IP", 0x500);
        state.set_var("cycleCount", 0);
        state.set_var("prefixLen", 1);
        state.set_var("repType", 2); // REPNE

        // Plant the match at offset 5: scan finds it after 6 iterations
        // (compares at offsets 0..=5, exits when match found at 5).
        let base = 0x6000 * 16 + 0x100;
        state.memory[base + 5] = 0xAA;

        let outcome = apply_read_only_with_commit(&desc, &prog, &mut state, &[], CommitMode::Full);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 6 });
        // DI advanced by 6 bytes.
        assert_eq!(state.get_var("DI"), Some(0x106));
        // CX = 16 - 6 = 10
        assert_eq!(state.get_var("CX"), Some(10));
        // IP advance.
        assert_eq!(state.get_var("IP"), Some(0x502));
        // cycleCount: 6 iters * 15 cycles per SCAS iter = 90
        assert_eq!(state.get_var("cycleCount"), Some(90));
        // ZF set (last compare AL == mem[ES:DI-step] was equal).
        let new_flags = state.get_var("flags").unwrap();
        assert_eq!(new_flags & 0x40, 0x40, "ZF should be set after match");
    }

    /// `CommitMode::Full` on CMPSB REPE: walks until inequality, both
    /// pointers advance, CX drains, flags carry the last SUB result.
    #[test]
    fn commit_full_cmpsb_repe_breaks_on_inequality() {
        let desc = cmpsb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&[
            "CX", "DI", "SI", "ES", "DS", "flags", "IP", "cycleCount", "prefixLen", "repType",
        ]);
        state.set_var("CX", 32);
        state.set_var("DI", 0x100);
        state.set_var("SI", 0x800);
        state.set_var("ES", 0x7000);
        state.set_var("DS", 0x8000);
        state.set_var("flags", 0);
        state.set_var("IP", 0x300);
        state.set_var("cycleCount", 0);
        state.set_var("prefixLen", 1);
        state.set_var("repType", 1); // REPE

        let dst_base = 0x7000 * 16 + 0x100;
        let src_base = 0x8000 * 16 + 0x800;
        // First 4 bytes match; 5th byte differs → REPE breaks at iter 5.
        for i in 0..4 {
            state.memory[dst_base + i] = 0xCC;
            state.memory[src_base + i] = 0xCC;
        }
        state.memory[dst_base + 4] = 0xAA;
        state.memory[src_base + 4] = 0xBB; // different

        let outcome = apply_read_only_with_commit(&desc, &prog, &mut state, &[], CommitMode::Full);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 5 });
        assert_eq!(state.get_var("DI"), Some(0x105));
        assert_eq!(state.get_var("SI"), Some(0x805));
        assert_eq!(state.get_var("CX"), Some(27)); // 32 - 5
        assert_eq!(state.get_var("IP"), Some(0x302));
        assert_eq!(state.get_var("cycleCount"), Some(110)); // 5 * 22 (CMPS)
        // Last compare: src=0xBB, dst=0xAA; ZF clear.
        let new_flags = state.get_var("flags").unwrap();
        assert_eq!(new_flags & 0x40, 0, "ZF should be clear after mismatch");
    }

    /// `CommitMode::MemoryOnly` on STOSB: state vars must NOT change
    /// (the dual-harness contract). Memory still updates.
    #[test]
    fn commit_memory_only_leaves_state_vars_untouched() {
        let desc = stosb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&[
            "CX", "DI", "ES", "AL", "flags", "IP", "cycleCount", "prefixLen",
        ]);
        state.set_var("CX", 16);
        state.set_var("DI", 0x100);
        state.set_var("ES", 0x2000);
        state.set_var("AL", 0x42);
        state.set_var("flags", 0);
        state.set_var("IP", 0x1000);
        state.set_var("cycleCount", 5000);
        state.set_var("prefixLen", 1);

        let outcome = apply_fill_with_commit(&desc, &prog, &mut state, &[], CommitMode::MemoryOnly);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 16 });
        // Pre-loop state preserved.
        assert_eq!(state.get_var("CX"), Some(16));
        assert_eq!(state.get_var("DI"), Some(0x100));
        assert_eq!(state.get_var("IP"), Some(0x1000));
        assert_eq!(state.get_var("cycleCount"), Some(5000));
        // Memory still fills.
        assert_eq!(state.memory[0x20100], 0x42);
        assert_eq!(state.memory[0x2010F], 0x42);
    }

    /// Strongest in-test verification short of running a real cabinet:
    /// `CommitMode::MemoryOnly` and `CommitMode::Full` produce identical
    /// memory mutations on independent State clones. This is what
    /// guarantees `CALCITE_REP_GENERIC=1` wouldn't drift from the dual-
    /// harness baseline on the memory side.
    #[test]
    fn full_and_memory_only_agree_on_memory_for_stosb() {
        let desc = stosb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state_full = rigged_state(&[
            "CX", "DI", "ES", "AL", "flags", "IP", "cycleCount", "prefixLen",
        ]);
        state_full.set_var("CX", 64);
        state_full.set_var("DI", 0x10);
        state_full.set_var("ES", 0xA000);
        state_full.set_var("AL", 0x99);
        state_full.set_var("flags", 0);
        state_full.set_var("IP", 0x500);
        state_full.set_var("cycleCount", 0);
        state_full.set_var("prefixLen", 1);

        let mut state_mem = rigged_state(&[
            "CX", "DI", "ES", "AL", "flags", "IP", "cycleCount", "prefixLen",
        ]);
        state_mem.set_var("CX", 64);
        state_mem.set_var("DI", 0x10);
        state_mem.set_var("ES", 0xA000);
        state_mem.set_var("AL", 0x99);
        state_mem.set_var("flags", 0);
        state_mem.set_var("IP", 0x500);
        state_mem.set_var("cycleCount", 0);
        state_mem.set_var("prefixLen", 1);

        let _ = apply_fill_with_commit(&desc, &prog, &mut state_full, &[], CommitMode::Full);
        let _ = apply_fill_with_commit(&desc, &prog, &mut state_mem, &[], CommitMode::MemoryOnly);
        assert_eq!(state_full.memory, state_mem.memory);
    }

    /// Default-shape compatibility: `apply_fill` (no commit param) is
    /// equivalent to `apply_fill_with_commit(MemoryOnly)`. This is the
    /// dual-harness invariant — existing call sites must keep their
    /// memory-only contract.
    #[test]
    fn default_apply_fill_equivalent_to_memory_only() {
        let desc = stosb_descriptor();
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state_a = rigged_state(&["CX", "DI", "ES", "AL", "flags"]);
        let mut state_b = rigged_state(&["CX", "DI", "ES", "AL", "flags"]);
        for s in [&mut state_a, &mut state_b] {
            s.set_var("CX", 8);
            s.set_var("DI", 0x40);
            s.set_var("ES", 0xB000);
            s.set_var("AL", 0x77);
            s.set_var("flags", 0);
        }

        let _ = apply_fill(&desc, &prog, &mut state_a, &[]);
        let _ = apply_fill_with_commit(&desc, &prog, &mut state_b, &[], CommitMode::MemoryOnly);
        assert_eq!(state_a.memory, state_b.memory);
    }

    // -----------------------------------------------------------------
    // Precondition (Wart 6) — applier reports PreconditionNotMet (NOT
    // Unsupported) when the cabinet's outer-guard predicate evaluates
    // false against runtime state. The dispatcher reads this as
    // "silently bail" rather than "panic".
    // -----------------------------------------------------------------

    /// When the descriptor carries a precondition whose override slot
    /// reads non-zero in state, the applier returns
    /// `ApplyOutcome::PreconditionNotMet`. The cabinet's CSS already
    /// produced the correct single-iter post-state on this tick, so
    /// the applier has nothing to do.
    #[test]
    fn precondition_not_met_returns_precondition_not_met_not_unsupported() {
        use crate::pattern::loop_descriptor::Precondition;
        let mut desc = stosb_descriptor();
        desc.precondition = Some(Precondition::NoOverrides(vec![StyleTest::Single {
            property: "--_irqActive".to_string(),
            value: Expr::Literal(1.0),
        }]));
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "ES", "AL", "flags", "_irqActive"]);
        state.set_var("CX", 64);
        state.set_var("DI", 0x10);
        state.set_var("ES", 0xA000);
        state.set_var("AL", 0x42);
        state.set_var("flags", 0);
        // Override fires → precondition NOT met.
        state.set_var("_irqActive", 1);

        let outcome = apply_fill(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::PreconditionNotMet);
        // Crucially: this is a different variant from Unsupported. The
        // dispatcher treats them differently (silent bail vs panic).
        assert!(!matches!(outcome, ApplyOutcome::Unsupported(_)));
        // No memory was written.
        for i in 0..64 {
            assert_eq!(state.memory[0xA0010 + i], 0);
        }
    }

    /// When the override slots all read zero, the precondition is met
    /// and the applier runs normally. Same descriptor + state as the
    /// failing case above, but with `_irqActive = 0`.
    #[test]
    fn precondition_met_runs_normally() {
        use crate::pattern::loop_descriptor::Precondition;
        let mut desc = stosb_descriptor();
        desc.precondition = Some(Precondition::NoOverrides(vec![StyleTest::Single {
            property: "--_irqActive".to_string(),
            value: Expr::Literal(1.0),
        }]));
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "ES", "AL", "flags", "_irqActive"]);
        state.set_var("CX", 64);
        state.set_var("DI", 0x10);
        state.set_var("ES", 0xA000);
        state.set_var("AL", 0x42);
        state.set_var("flags", 0);
        state.set_var("_irqActive", 0); // override does NOT fire

        let outcome = apply_fill(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 64 });
        for i in 0..64 {
            assert_eq!(state.memory[0xA0010 + i], 0x42);
        }
    }

    /// `precondition = None` means the descriptor has no outer guard
    /// (brainfuck / no-override cabinet shape) — applier proceeds
    /// unconditionally.
    #[test]
    fn precondition_none_runs_normally() {
        let desc = stosb_descriptor(); // precondition = None
        assert!(desc.precondition.is_none());
        let prog = empty_program_with_descriptor(desc.clone());
        let mut state = rigged_state(&["CX", "DI", "ES", "AL", "flags"]);
        state.set_var("CX", 8);
        state.set_var("DI", 0x10);
        state.set_var("ES", 0xA000);
        state.set_var("AL", 0x55);
        state.set_var("flags", 0);
        let outcome = apply_fill(&desc, &prog, &mut state, &[]);
        assert_eq!(outcome, ApplyOutcome::Applied { iterations: 8 });
    }
}
