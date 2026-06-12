//! CSS expression compiler — flattens `Expr` trees into linear `Op` sequences.
//!
//! This module compiles the parsed CSS expression trees into a flat bytecode IR
//! that operates on indexed slots instead of string-keyed HashMaps. The compiled
//! form eliminates:
//! - `HashMap<String, f64>` lookups (replaced by direct slot indexing)
//! - Recursive `Expr` tree walking (replaced by linear op execution)
//! - String allocation for buffer-prefixed names
//! - Function call save/restore overhead (inlined at compile time)

use std::collections::HashMap;

// FxHashMap for the runtime hot-path dispatch tables and broadcast-write
// address maps. These are integer-keyed and looked up per-dispatched-op
// (Op::Dispatch / Op::DispatchChain) or per-tick (broadcast writes), so
// hashing cost is on the critical path. SipHash, the std default, takes
// ~30 cycles per i32 key; FxHasher is essentially `multiply + xor` and
// runs in ~3 cycles. Flamegraph (LOGBOOK 2026-04-30) showed SipHash-related
// frames at ~17% of worker CPU on doom8088 LOAD/INGAME.
//
// Compile-time HashMaps (string-keyed dispatch_tables, function maps,
// caches in CompilerCtx, etc.) are NOT swapped — they're hit once at
// load and the swap would ripple through every cross-crate API.
type FxMap<K, V> = rustc_hash::FxHashMap<K, V>;
type FxSet<T> = rustc_hash::FxHashSet<T>;

use crate::eval::property_to_address;
use crate::pattern::broadcast_write::BroadcastWrite;
use crate::pattern::dispatch_table::{self, DispatchTable};
use crate::state::State;
use crate::types::*;

// ---------------------------------------------------------------------------
// Per-function compile profiler. Enable with CALCITE_PROFILE_COMPILE=1.
// ---------------------------------------------------------------------------

thread_local! {
    static PROF_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static PROF_TOTALS: std::cell::RefCell<Vec<(&'static str, f64, u64)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

pub fn profile_compile_init() {
    PROF_ENABLED.with(|e| e.set(std::env::var_os("CALCITE_PROFILE_COMPILE").is_some()));
    PROF_TOTALS.with(|t| t.borrow_mut().clear());
}

pub fn profile_compile_dump() {
    if !PROF_ENABLED.with(|e| e.get()) { return; }
    PROF_TOTALS.with(|t| {
        let mut entries: Vec<_> = t.borrow().clone();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("[compile profile] ---- per-scope totals (incl. children) ----");
        for (name, total, count) in &entries {
            eprintln!("  {:>8.3}s  {:>12} calls  {:>9.3}us/call  {}",
                total, count, total / *count as f64 * 1_000_000.0, name);
        }
    });
}

pub struct ProfScope {
    idx: usize,
    start: web_time::Instant,
    enabled: bool,
}

impl ProfScope {
    pub fn new(name: &'static str) -> Self {
        let enabled = PROF_ENABLED.with(|e| e.get());
        if !enabled {
            // web_time::Instant is wasm-safe (raw std::time::Instant panics on wasm).
            return ProfScope { idx: 0, start: web_time::Instant::now(), enabled: false };
        }
        let idx = PROF_TOTALS.with(|t| {
            let mut v = t.borrow_mut();
            if let Some(i) = v.iter().position(|(n, _, _)| *n == name) { i }
            else { v.push((name, 0.0, 0)); v.len() - 1 }
        });
        ProfScope { idx, start: web_time::Instant::now(), enabled: true }
    }
}

impl Drop for ProfScope {
    fn drop(&mut self) {
        if !self.enabled { return; }
        let elapsed = self.start.elapsed().as_secs_f64();
        PROF_TOTALS.with(|t| {
            let mut v = t.borrow_mut();
            v[self.idx].1 += elapsed;
            v[self.idx].2 += 1;
        });
    }
}

macro_rules! prof {
    ($name:expr) => { let _prof_guard = $crate::compile::ProfScope::new($name); };
}


// ---------------------------------------------------------------------------
// Op — flat bytecode instruction
// ---------------------------------------------------------------------------

/// Slot index type — widened to u32 to support large programs (>64K slots).
pub type Slot = u32;

/// Convert a literal `f64` to `i32` if it is a whole number in range.
/// Used by dispatch-table fast paths that require integer-valued literals.
fn lit_as_i32(v: f64) -> Option<i32> {
    if !v.is_finite() || v.fract() != 0.0 {
        return None;
    }
    if v < i32::MIN as f64 || v > i32::MAX as f64 {
        return None;
    }
    Some(v as i32)
}

/// A single operation in the compiled bytecode.
///
/// All operands are `Slot` (u32) indices into a flat `Vec<i32>` array.
/// State reads/writes use `i32` addresses matching the x86CSS convention.
#[derive(Debug, Clone)]
pub enum Op {
    // --- Loads ---
    /// slot[dst] = literal value
    LoadLit {
        dst: Slot,
        val: i32,
    },
    /// slot[dst] = slot[src]
    LoadSlot {
        dst: Slot,
        src: Slot,
    },
    /// slot[dst] = state.read_mem(addr) — compile-time-known address
    LoadState {
        dst: Slot,
        addr: i32,
    },
    /// slot[dst] = state.read_mem(slot[addr_slot] as i32) — runtime address
    LoadMem {
        dst: Slot,
        addr_slot: Slot,
    },
    /// slot[dst] = state.read_mem16(slot[addr_slot] as i32) — 16-bit word read
    LoadMem16 {
        dst: Slot,
        addr_slot: Slot,
    },
    /// Packed-cell byte read.
    ///
    /// Used by the PACK_SIZE=2 memory layout where N bytes share one state-var
    /// cell `mc{N}` (cell value = b0 | b1<<8 | ...). At runtime:
    ///   key = slot[key_slot]
    ///   N = key / pack
    ///   off = key % pack
    ///   cell_addr = packed_cell_tables[table_id][N]   // 0 if N out of bounds
    ///   cell = state.read_mem(cell_addr)
    ///   slot[dst] = (cell >> (off * 8)) & 0xFF
    LoadPackedByte {
        dst: Slot,
        key_slot: Slot,
        table_id: u32,
        pack: u8,
    },

    // --- Arithmetic ---
    Add {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    /// slot[dst] = slot[a] + val
    AddLit {
        dst: Slot,
        a: Slot,
        val: i32,
    },
    Sub {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    /// slot[dst] = slot[a] - val
    SubLit {
        dst: Slot,
        a: Slot,
        val: i32,
    },
    Mul {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    /// slot[dst] = slot[a] * val
    MulLit {
        dst: Slot,
        a: Slot,
        val: i32,
    },
    Div {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    Mod {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    Neg {
        dst: Slot,
        src: Slot,
    },
    Abs {
        dst: Slot,
        src: Slot,
    },
    Sign {
        dst: Slot,
        src: Slot,
    },
    Pow {
        dst: Slot,
        base: Slot,
        exp: Slot,
    },
    Min {
        dst: Slot,
        args: Vec<Slot>,
    },
    Max {
        dst: Slot,
        args: Vec<Slot>,
    },
    Clamp {
        dst: Slot,
        min: Slot,
        val: Slot,
        max: Slot,
    },
    Round {
        dst: Slot,
        strategy: RoundStrategy,
        val: Slot,
        interval: Slot,
    },
    Floor {
        dst: Slot,
        src: Slot,
    },

    // --- Bitwise (CSS function equivalents) ---
    /// lowerBytes(a, b) → a & ((1 << b) - 1)
    And {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    /// slot[dst] = slot[a] & ((1 << val) - 1)
    AndLit {
        dst: Slot,
        a: Slot,
        val: i32,
    },
    /// rightShift(a, b) → a >> b
    Shr {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    /// slot[dst] = slot[a] >> val
    ShrLit {
        dst: Slot,
        a: Slot,
        val: i32,
    },
    /// leftShift(a, b) → a << b
    Shl {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    /// slot[dst] = slot[a] << val
    ShlLit {
        dst: Slot,
        a: Slot,
        val: i32,
    },
    /// slot[dst] = slot[a] % val
    ModLit {
        dst: Slot,
        a: Slot,
        val: i32,
    },
    /// bit(val, idx) → (val >> idx) & 1
    Bit {
        dst: Slot,
        val: Slot,
        idx: Slot,
    },

    /// Native 16-bit bitwise AND: slot[dst] = (slot[a] & slot[b]) & 0xFFFF.
    /// Emitted by `try_compile_by_body_pattern` when a function body is the
    /// canonical 32-local bit-decomposition form.
    BitAnd16 { dst: Slot, a: Slot, b: Slot },
    /// Native 16-bit bitwise OR.
    BitOr16 { dst: Slot, a: Slot, b: Slot },
    /// Native 16-bit bitwise XOR.
    BitXor16 { dst: Slot, a: Slot, b: Slot },
    /// Native 16-bit bitwise NOT (one's complement within 16 bits).
    BitNot16 { dst: Slot, a: Slot },

    // --- Comparisons & control flow ---
    /// slot[dst] = (slot[a] == slot[b]) as i64  (integer comparison)
    CmpEq {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    /// If slot[cond] == 0, jump to target op index
    BranchIfZero {
        cond: Slot,
        target: u32,
    },
    /// Fused compare-and-branch: if slot[a] != val, jump to target.
    /// Replaces LoadLit + CmpEq + BranchIfZero triplet.
    BranchIfNotEqLit {
        a: Slot,
        val: i32,
        target: u32,
    },
    /// Fused LoadState + BranchIfNotEqLit:
    ///   slot[dst] = state.read_mem(addr);
    ///   if slot[dst] != val { pc = target }
    /// Eliminates one dispatch cycle vs the two-op sequence. The LoadState
    /// still writes `dst` so downstream ops reading that slot are unaffected.
    LoadStateAndBranchIfNotEqLit {
        dst: Slot,
        addr: i32,
        val: i32,
        target: u32,
    },

    /// Two-condition AND-guard fusion. Replaces adjacent BranchIfNotEqLit
    /// pairs (different slots) that share a target — the dominant runtime
    /// adjacency surviving the chain pass.
    ///
    /// Semantics:
    ///   if slots[a1] != val1 { pc = target; continue; }
    ///   if slots[a2] != val2 { pc = target; continue; }
    ///   pc += 2;   // skip the dead BIfNEL left at i+1 for external-jump safety
    ///
    /// On fall-through, advances pc by 2 (not 1) — the original BIfNEL at
    /// ops[i+1] is left in place as dead code so external jumps that
    /// landed on it still execute correctly. Mirrors the DispatchChain
    /// dead-code convention.
    BranchIfNotEqLit2 {
        a1: Slot,
        val1: i32,
        a2: Slot,
        val2: i32,
        target: u32,
    },
    /// Unconditional jump to target op index
    Jump {
        target: u32,
    },
    /// Dispatch chain: look up slot[a] in chain_tables[chain_id]; if found,
    /// jump to the matching body PC, else jump to miss_target. Replaces a run
    /// of BranchIfNotEqLit ops all testing the same slot against different
    /// literals (the dominant pattern in CSS-DOS bytecode).
    DispatchChain {
        a: Slot,
        chain_id: u32,
        miss_target: u32,
    },

    // --- Dispatch table ---
    /// HashMap lookup: slot[dst] = dispatch_tables[table_id].entries[slot[key]]
    /// Falls back to executing ops at fallback_target if key not found.
    Dispatch {
        dst: Slot,
        key: Slot,
        table_id: Slot,
        fallback_target: u32,
    },

    /// Flat-array dispatch fast path: slot[dst] = flat_dispatch_arrays[array_id]
    /// indexed by (slot[key] - base_key). Out-of-range keys yield `default`.
    /// Emitted when every dispatch-table entry is an integer literal and the
    /// key range is dense enough to fit in a contiguous Vec<i32>.
    DispatchFlatArray {
        dst: Slot,
        key: Slot,
        array_id: u32,
        base_key: i32,
        default: i32,
    },

    /// Call a compiled function: copy arg slots into the function's reserved
    /// parameter slots, execute its body, copy the result slot into `dst`.
    /// Compiled once per function and reused at every call site to avoid the
    /// N² recompile cost that dominates compilation of large programs.
    Call {
        dst: Slot,
        fn_id: u32,
        arg_slots: Vec<Slot>,
    },

    // --- Stores ---
    /// state.write_mem(addr, slot[src]) — compile-time-known address
    StoreState {
        addr: i32,
        src: Slot,
    },
    /// state.write_mem(slot[addr_slot], slot[src]) — runtime address
    StoreMem {
        addr_slot: Slot,
        src: Slot,
    },

    /// Bulk byte-fill:
    ///   state.memory[slots[dst_slot] .. slots[dst_slot] + slots[count_slot]]
    ///     = (slots[val_slot] & 0xFF) as u8
    /// After executing:
    ///   slots[dst_slot] += slots[count_slot];
    ///   slots[count_slot] = 0;
    ///   pc = exit_target;
    ///
    /// Emitted by the memory-fill peephole / CSS-level pattern in place of
    /// a recognised byte-fill loop. The bulk path skips write_log
    /// instrumentation (diagnostic-only; documented in the design spec).
    MemoryFill {
        dst_slot: Slot,
        val_slot: Slot,
        count_slot: Slot,
        exit_target: u32,
    },

    /// Bulk byte-copy:
    ///   state.memory.copy_within(
    ///       slots[src_slot] .. slots[src_slot] + slots[count_slot],
    ///       slots[dst_slot],
    ///   )
    /// After executing:
    ///   slots[src_slot] += slots[count_slot];
    ///   slots[dst_slot] += slots[count_slot];
    ///   slots[count_slot] = 0;
    ///   pc = exit_target;
    ///
    /// Uses `slice::copy_within` under the hood, which is correct for overlapping
    /// ranges. Out-of-bounds source or destination ranges are clipped silently
    /// (matches `Op::MemoryFill` and `Op::StoreMem` discipline). Negative src or
    /// dst is a no-op for memory but still advances slots logically — matches
    /// what iterating the loop would do.
    MemoryCopy {
        src_slot: Slot,
        dst_slot: Slot,
        count_slot: Slot,
        exit_target: u32,
    },

    /// Replicated body: executes a periodic op sequence detected by the
    /// `replicated_body` recogniser. At runtime, for each rep `k` in
    /// `0..reps`, the body's ops are evaluated with operand fields adjusted
    /// by `k * stride` for any operand classified as Linear by the
    /// classifier. Constant operands keep their template value.
    ///
    /// `body` holds the rep-0 ops verbatim (template form). `strides` is a
    /// parallel array — one entry per body op — listing
    /// `(operand_index, stride)` for each Linear operand. Operand indices
    /// are the canonical extraction order from
    /// `pattern::replicated_body::extract_op_fields`.
    ///
    /// Body ops must be straight-line (no branch/jump/dispatch/bulk-mem
    /// variants) and must not contain variable-arity Vec fields
    /// (Min/Max/Call) — the recogniser bails on those before constructing
    /// this op.
    ReplicatedBody {
        body: Box<[Op]>,
        strides: Box<[Box<[(u8, i32)]>]>,
        reps: u32,
    },
}

// ---------------------------------------------------------------------------
// CompiledProgram — the output of the compiler
// ---------------------------------------------------------------------------

/// A compiled CSS program ready for efficient execution.
///
/// All string-keyed property lookups have been replaced with slot indices.
/// The `ops` vector is executed linearly (with branches) against a flat slot array.
#[derive(Debug)]
pub struct CompiledProgram {
    /// The flat bytecode instruction stream.
    pub ops: Vec<Op>,
    /// Number of slots needed (the slot array size).
    pub slot_count: Slot,
    /// Mapping from slot index → state address for write-back.
    /// Only includes canonical properties (not buffer copies or byte halves).
    pub writeback: Vec<(Slot, i32)>,
    /// Broadcast writes — compiled separately because they need runtime HashMap lookup.
    pub broadcast_writes: Vec<CompiledBroadcastWrite>,
    /// Dispatch table data (kept for Dispatch op lookups at runtime).
    pub dispatch_tables: Vec<CompiledDispatchTable>,
    /// Dispatch chain tables (for DispatchChain op — runs of same-slot BranchIfNotEqLit).
    pub chain_tables: Vec<DispatchChainTable>,
    /// Flat-array dispatch tables (for DispatchFlatArray op — dense literal-only lookups).
    pub flat_dispatch_arrays: Vec<FlatDispatchArray>,
    /// Compiled function bodies (for `Op::Call` — reused across call sites).
    pub functions: Vec<CompiledFunction>,
    /// Mapping from property name → slot index (for reading computed values after execution).
    pub property_slots: HashMap<String, Slot>,
    /// Packed-cell address tables: `tables[id][n]` is the state-var address of cell
    /// `mcN` (always negative; 0 means "cell not present" → read returns 0).
    /// Used by `Op::LoadPackedByte` for the PACK_SIZE>1 memory layout.
    pub packed_cell_tables: Vec<Vec<i32>>,
    /// Per-`packed_cell_tables[id]` literal-exception arrays. Looked up FIRST in
    /// LoadPackedByte runtime; if returns Some(byte), used instead of the packed
    /// extract. Dense base-keyed Vec avoids HashMap overhead in the hot path.
    pub packed_exception_tables: Vec<PackedExceptionTable>,
    /// Packed-broadcast write ports (one per CSS-DOS write slot). Each port
    /// covers all packed `--mc{N}` cells through one slot — at writeback time
    /// the executor checks the gate, looks up the cell from the byte address,
    /// and splices in the new byte. Replaces ~190K ops/tick of nested
    /// `--applySlot` calls with ~6 port checks.
    pub packed_broadcast_writes: Vec<CompiledPackedBroadcastWrite>,
    /// Optional descriptor for a "windowed byte array" read pattern detected
    /// in `--readMem`'s exception entries — i.e. a contiguous range of
    /// addresses whose dispatch entries all redirect to a single
    /// flat-array dispatch with the same shape
    /// `--funcname((cell_value) * stride + literal_offset)`. The
    /// `key_cell_property` name is resolved to a state-var slot by
    /// `Evaluator::wire_state_for_windowed_byte_array`, which builds the
    /// runtime `state::WindowedByteArray` and installs it on `State` so reads
    /// inside the window collapse to one state-var read + one array index,
    /// instead of walking the inline-exception CmpEq chain on every byte.
    pub windowed_byte_array: Option<CompiledWindowedByteArray>,
    /// Mirror of `Evaluator::loop_descriptors`. Populated at compile time
    /// by the same recogniser; carried on the program so the per-tick
    /// path can consult it without re-recognising. Phase 1 of the
    /// rep_fast_forward genericity mission: populated but not yet
    /// consumed at runtime.
    pub loop_descriptors: Vec<crate::pattern::loop_descriptor::LoopDescriptor>,
}

/// Compile-time half of the windowed-byte-array fast path. The compiler can't
/// resolve state-var slots (they're assigned by `State::load_properties`), so
/// the descriptor carries the property NAME; the wiring step looks the slot up
/// and builds the runtime `state::WindowedByteArray`.
#[derive(Debug, Clone)]
pub struct CompiledWindowedByteArray {
    pub window_base: i32,
    pub window_end: i32,
    pub key_cell_property: String,
    pub stride: i32,
    pub byte_array_base_key: i32,
    pub byte_array: std::sync::Arc<Vec<i32>>,
}

/// Dense base-keyed lookup table for packed-byte literal exceptions.
/// Keys outside `[base, base + values.len())` have no exception.
/// Inside that range, `values[key - base] == i32::MIN` also means "no exception".
#[derive(Debug, Clone)]
pub struct PackedExceptionTable {
    pub base: i32,
    pub values: Vec<i32>,
}

impl PackedExceptionTable {
    pub fn empty() -> Self {
        PackedExceptionTable { base: 0, values: Vec::new() }
    }
    #[inline(always)]
    pub fn get(&self, key: i32) -> Option<i32> {
        let idx = key.wrapping_sub(self.base);
        if (idx as u32) < self.values.len() as u32 {
            let v = unsafe { *self.values.get_unchecked(idx as usize) };
            if v != i32::MIN { Some(v) } else { None }
        } else {
            None
        }
    }
}

/// A function compiled once and reused at every call site via `Op::Call`.
///
/// Each call copies the argument slots into `param_slots` (and initializes any
/// locals by re-running `body_ops`), then reads the result from `result_slot`.
#[derive(Debug)]
pub struct CompiledFunction {
    /// Reserved slots for the parameters, in declaration order.
    pub param_slots: Vec<Slot>,
    /// Compiled body ops. Reads from `param_slots`, writes to `result_slot`
    /// (and any intermediate/local slots). Self-contained — slots referenced
    /// here are either params, locals, or temporaries scoped to this body.
    pub body_ops: Vec<Op>,
    /// Slot holding the result after body_ops runs.
    pub result_slot: Slot,
}

/// A dispatch chain table — maps an i32 key to a body PC.
/// Used by the `DispatchChain` op to skip past linear runs of
/// `BranchIfNotEqLit` that all test the same slot.
#[derive(Debug, Default)]
pub struct DispatchChainTable {
    /// Key → target op index (body start).
    pub entries: FxMap<i32, u32>,
    /// Optional dense-array fast path. When the key range is contiguous (or
    /// nearly so), we populate a `Vec<u32>` indexed by `key - flat_base`.
    /// `u32::MAX` sentinel means "no entry — fall through to HashMap or miss".
    /// Enabled when `flat_range()` below the density threshold.
    pub flat_table: Option<FlatChainTable>,
}

#[derive(Debug)]
pub struct FlatChainTable {
    pub base: i32,
    /// `targets[i] = u32::MAX` means key (base+i) is not in the table.
    pub targets: Vec<u32>,
}

/// A compiled broadcast write.
#[derive(Debug)]
pub struct CompiledBroadcastWrite {
    /// Slot holding the destination address.
    pub dest_slot: Slot,
    /// Ops to evaluate the value expression (result in value_slot).
    pub value_ops: Vec<Op>,
    /// Slot holding the evaluated value.
    pub value_slot: Slot,
    /// Address → state address mapping for the broadcast.
    pub address_map: FxMap<i64, i32>,
    /// Spillover ops (for word writes).
    pub spillover: Option<CompiledSpillover>,
    /// Optional "outer gate" slot: if set, the broadcast only fires when
    /// this slot reads as 1. Matches `BroadcastWrite::gate_property` from
    /// the recogniser — see its docs for the CSS shape this absorbs.
    pub gate_slot: Option<Slot>,
}

/// Compiled spillover for word-write broadcast.
#[derive(Debug)]
pub struct CompiledSpillover {
    /// Slot holding the guard property value.
    pub guard_slot: Slot,
    /// Map from dest address → (ops to compute high byte, result slot).
    pub entries: FxMap<i64, (Vec<Op>, Slot)>,
}

/// A compiled packed-broadcast write port — one per memory write slot.
///
/// At runtime, when `gate_slot` reads 1, the executor reads the byte address
/// from `addr_slot`, looks up the cell's state address in `addr_to_cell_addr`
/// (dense base-keyed Vec), reads the previous cell value, splices in the byte
/// from `val_slot` based on `addr & 1`, and writes the result back.
///
/// One CompiledPackedBroadcastWrite covers all packed cells through one slot.
/// Slot 0 is applied last so it wins same-cell collisions (matching the CSS
/// `--applySlot` chain's outermost-wins semantics).
///
/// Width semantics: `width_slot` reads 1 (byte splice at `addr_slot`) or 2
/// (word splice with lo at `addr_slot` and hi at `addr_slot+1` — may straddle
/// two adjacent cells when the address is odd-aligned).
#[derive(Debug)]
pub struct CompiledPackedBroadcastWrite {
    /// Slot holding the per-tick gate (`--_slotKLive`). Skip the port when 0.
    pub gate_slot: Slot,
    /// Slot holding the byte address (`--memAddrK`).
    pub addr_slot: Slot,
    /// Slot holding the byte (width=1) or 16-bit word (width=2) value (`--memValK`).
    pub val_slot: Slot,
    /// Slot holding the per-tick width gate (`--_slotKWidth`). 1 or 2.
    pub width_slot: Slot,
    /// Pack size — bytes per cell (currently always 2).
    pub pack: u8,
    /// Dense cell-index → cell-state-address lookup. Index by `byte_addr / pack`.
    /// 0 means "no cell at this index — drop the write" (state-var addresses
    /// are always negative, so 0 is a safe sentinel).
    pub cell_table: Vec<i32>,
}

/// A flat-array dispatch table — dense `Vec<i32>` keyed by (key - base_key).
/// Used by the `DispatchFlatArray` op when every dispatch entry is an integer
/// literal and the key range is small enough to fit in contiguous memory.
#[derive(Debug)]
pub struct FlatDispatchArray {
    /// Values indexed by (key - base_key). Out-of-range indices yield the
    /// default value baked into the op.
    /// Stored behind an `Arc` so other parts of the runtime (e.g. the
    /// `State::windowed_byte_array` short-circuit) can share the backing buffer
    /// without cloning multi-MB byte payloads.
    pub values: std::sync::Arc<Vec<i32>>,
}

/// A compiled dispatch table — kept for runtime HashMap lookup.
#[derive(Debug)]
pub struct CompiledDispatchTable {
    /// Compiled ops for each dispatch entry, keyed by the dispatch value.
    /// Each entry is (ops, result_slot).
    pub entries: FxMap<i64, (Vec<Op>, Slot)>,
    /// Compiled ops for the fallback expression.
    pub fallback_ops: Vec<Op>,
    /// Slot holding the fallback result.
    pub fallback_slot: Slot,
}

// ---------------------------------------------------------------------------
// Body-pattern analysis — detect mathematical patterns in function bodies
// ---------------------------------------------------------------------------

/// Check if an expression is `var(name)`.
pub(crate) fn is_var_ref(expr: &Expr, param_name: &str) -> bool {
    matches!(expr, Expr::Var { name, .. } if name == param_name)
}

/// Check if an expression is `calc(readFn(var(param)) + readFn(calc(var(param) + 1)) * 256)`
/// — a 16-bit word read pattern (little-endian). If matched, returns the
/// name of the inner read function (e.g. `--readMem`) so the caller can
/// verify that function is a pure identity read before shortcutting to
/// `LoadMem16`. The two calls must use the same function name.
fn is_word_read_pattern(expr: &Expr, param: &str) -> Option<String> {
    // Top level: calc(A + B)
    let Expr::Calc(CalcOp::Add(lhs, rhs)) = expr else { return None };

    // LHS: FunctionCall("--readMem", [Var(param)])
    let Expr::FunctionCall { name: name_lo, args: args_lo } = lhs.as_ref() else { return None };
    if !name_lo.ends_with("readMem") || args_lo.len() != 1 || !is_var_ref(&args_lo[0], param) {
        return None;
    }

    // RHS: calc(FunctionCall("--readMem", [calc(Var(param) + 1)]) * 256)
    let Expr::Calc(CalcOp::Mul(hi_call, multiplier)) = rhs.as_ref() else { return None };
    if !matches!(multiplier.as_ref(), Expr::Literal(v) if (*v - 256.0).abs() < f64::EPSILON) {
        return None;
    }
    let Expr::FunctionCall { name: name_hi, args: args_hi } = hi_call.as_ref() else { return None };
    if !name_hi.ends_with("readMem") || args_hi.len() != 1 { return None }
    if name_hi != name_lo { return None }

    // The argument should be calc(Var(param) + 1)
    let Expr::Calc(CalcOp::Add(inner_var, one)) = &args_hi[0] else { return None };
    if !is_var_ref(inner_var, param) { return None }
    if !matches!(one.as_ref(), Expr::Literal(v) if (*v - 1.0).abs() < f64::EPSILON) {
        return None;
    }
    Some(name_lo.clone())
}

/// Check if an expression is `mod(var(a), pow(2, var(b)))` — bitmask pattern.
pub(crate) fn is_mod_pow2(expr: &Expr, a: &str, b: &str) -> bool {
    if let Expr::Calc(CalcOp::Mod(lhs, rhs)) = expr {
        if is_var_ref(lhs, a) {
            if let Expr::Calc(CalcOp::Pow(base, exp)) = rhs.as_ref() {
                return matches!(base.as_ref(), Expr::Literal(v) if (*v - 2.0).abs() < f64::EPSILON)
                    && is_var_ref(exp, b);
            }
        }
    }
    false
}

/// Check if an expression is `round(down, var(a) / pow(2, var(b)), 1)` — right shift.
pub(crate) fn is_right_shift(expr: &Expr, a: &str, b: &str) -> bool {
    if let Expr::Calc(CalcOp::Round(RoundStrategy::Down, val, interval)) = expr {
        // interval must be 1
        if !matches!(interval.as_ref(), Expr::Literal(v) if (*v - 1.0).abs() < f64::EPSILON) {
            return false;
        }
        // val must be var(a) / pow(2, var(b))
        if let Expr::Calc(CalcOp::Div(num, den)) = val.as_ref() {
            if is_var_ref(num, a) {
                if let Expr::Calc(CalcOp::Pow(base, exp)) = den.as_ref() {
                    return matches!(base.as_ref(), Expr::Literal(v) if (*v - 2.0).abs() < f64::EPSILON)
                        && is_var_ref(exp, b);
                }
            }
        }
    }
    false
}

/// Check if an expression is `var(a) * pow(2, var(b))` — left shift.
pub(crate) fn is_left_shift(expr: &Expr, a: &str, b: &str) -> bool {
    if let Expr::Calc(CalcOp::Mul(lhs, rhs)) = expr {
        if is_var_ref(lhs, a) {
            if let Expr::Calc(CalcOp::Pow(base, exp)) = rhs.as_ref() {
                return matches!(base.as_ref(), Expr::Literal(v) if (*v - 2.0).abs() < f64::EPSILON)
                    && is_var_ref(exp, b);
            }
        }
    }
    false
}

/// Check if an expression is `var(a) * var(local_name)` (in either order).
pub(crate) fn is_mul_refs(expr: &Expr, a: &str, local_name: &str) -> bool {
    if let Expr::Calc(CalcOp::Mul(lhs, rhs)) = expr {
        return (is_var_ref(lhs, a) && is_var_ref(rhs, local_name))
            || (is_var_ref(lhs, local_name) && is_var_ref(rhs, a));
    }
    false
}

/// Check if an expression is a power-of-2 dispatch table on `param`.
///
/// Pattern: `if(style(param:0): 1; style(param:1): 2; style(param:2): 4; ...)`
/// where entry K maps to 2^K.
pub(crate) fn is_pow2_dispatch(expr: &Expr, param: &str) -> bool {
    if let Expr::StyleCondition { branches, .. } = expr {
        if branches.len() < 4 {
            return false;
        }
        for branch in branches {
            if let StyleTest::Single { property, value } = &branch.condition {
                if property != param {
                    return false;
                }
                if let Expr::Literal(key_val) = value {
                    let k = *key_val as u32;
                    if let Expr::Literal(then_val) = &branch.then {
                        let expected = if k < 32 {
                            (1u64 << k) as f64
                        } else {
                            return false;
                        };
                        if (*then_val - expected).abs() > f64::EPSILON {
                            return false;
                        }
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
        return true;
    }
    false
}

/// Check if an expression is a bit-extract pattern: `mod(shift_body, 2)`.
///
/// Where `shift_body` is either:
/// - Directly `round(down, var(a) / pow(2, var(b)), 1)` (inline right-shift), or
/// - A function call to a function whose body IS a right-shift pattern.
pub(crate) fn is_bit_extract(
    expr: &Expr,
    a: &str,
    b: &str,
    functions: &HashMap<String, FunctionDef>,
) -> bool {
    if let Expr::Calc(CalcOp::Mod(inner, modulus)) = expr {
        // modulus must be 2
        if !matches!(modulus.as_ref(), Expr::Literal(v) if (*v - 2.0).abs() < f64::EPSILON) {
            return false;
        }
        // inner is an inline right-shift?
        if is_right_shift(inner, a, b) {
            return true;
        }
        // inner is a function call whose body is a right-shift pattern?
        if let Expr::FunctionCall { name, args } = inner.as_ref() {
            if args.len() == 2 && is_var_ref(&args[0], a) && is_var_ref(&args[1], b) {
                if let Some(func) = functions.get(name.as_str()) {
                    if func.parameters.len() == 2 && func.locals.is_empty() {
                        return is_right_shift(
                            &func.result,
                            &func.parameters[0].name,
                            &func.parameters[1].name,
                        );
                    }
                }
            }
        }
    }
    false
}

/// Which bitwise op a function body implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BitwiseKind {
    And,
    Or,
    Xor,
    Not,
}

/// True if `expr` is exactly `Literal(v)` for the given f64.
fn is_literal(expr: &Expr, v: f64) -> bool {
    matches!(expr, Expr::Literal(x) if (*x - v).abs() < f64::EPSILON)
}

/// True if `expr` extracts bit `i` of `var(param)` in the canonical form
/// `mod(var(param), 2)` (i == 0) or `mod(round(down, var(param)/2^i, 1), 2)` otherwise.
/// The divisor can be `Literal(2^i)` or `pow(2, Literal(i))`.
fn is_bit_i_of(expr: &Expr, param: &str, i: u32) -> bool {
    // Top level is mod(_, 2).
    let Expr::Calc(CalcOp::Mod(inner, modulus)) = expr else { return false; };
    if !is_literal(modulus, 2.0) { return false; }
    if i == 0 {
        return is_var_ref(inner, param);
    }
    // inner = round(down, var(param) / 2^i, 1)  (interval may be omitted or 1)
    let Expr::Calc(CalcOp::Round(RoundStrategy::Down, val, interval)) = inner.as_ref() else {
        return false;
    };
    if !is_literal(interval, 1.0) { return false; }
    let Expr::Calc(CalcOp::Div(num, den)) = val.as_ref() else { return false; };
    if !is_var_ref(num, param) { return false; }
    let expected = (1u64 << i) as f64;
    // Accept either Literal(2^i) or pow(2, Literal(i)).
    if is_literal(den, expected) { return true; }
    if let Expr::Calc(CalcOp::Pow(base, exp)) = den.as_ref() {
        if is_literal(base, 2.0) && is_literal(exp, i as f64) {
            return true;
        }
    }
    false
}

/// Flatten an N-ary Add tree into a Vec of leaf terms.
fn flatten_add(expr: &Expr, out: &mut Vec<Expr>) {
    if let Expr::Calc(CalcOp::Add(lhs, rhs)) = expr {
        flatten_add(lhs, out);
        flatten_add(rhs, out);
    } else {
        out.push(expr.clone());
    }
}

/// If `term` is `inner * factor` or `factor * inner` (both literal factor), return (inner, factor).
/// A bare `inner` with factor=1.0 also matches.
fn split_scaled_term(term: &Expr) -> (&Expr, f64) {
    if let Expr::Calc(CalcOp::Mul(lhs, rhs)) = term {
        if let Expr::Literal(f) = rhs.as_ref() {
            return (lhs.as_ref(), *f);
        }
        if let Expr::Literal(f) = lhs.as_ref() {
            return (rhs.as_ref(), *f);
        }
    }
    (term, 1.0)
}

/// True if `expr` is a reference to local name `local` (Var { name == local }).
fn is_local_ref(expr: &Expr, local: &str) -> bool {
    is_var_ref(expr, local)
}

/// Classify the per-bit combine of two locals `an`, `bn` into a bitwise kind:
/// - And: `var(an) * var(bn)`
/// - Or:  `min(1, var(an) + var(bn))`
/// - Xor: `min(1, var(an) + var(bn)) - var(an) * var(bn)`
/// Returns None if shape doesn't match any of these.
fn classify_pair_combine(expr: &Expr, an: &str, bn: &str) -> Option<BitwiseKind> {
    // AND: an * bn (either order)
    if let Expr::Calc(CalcOp::Mul(lhs, rhs)) = expr {
        if (is_local_ref(lhs, an) && is_local_ref(rhs, bn))
            || (is_local_ref(lhs, bn) && is_local_ref(rhs, an))
        {
            return Some(BitwiseKind::And);
        }
    }
    // OR: min(1, an + bn)
    if is_min_1_add(expr, an, bn) {
        return Some(BitwiseKind::Or);
    }
    // XOR: min(1, an + bn) - an * bn
    if let Expr::Calc(CalcOp::Sub(lhs, rhs)) = expr {
        if is_min_1_add(lhs, an, bn) {
            if let Expr::Calc(CalcOp::Mul(ma, mb)) = rhs.as_ref() {
                if (is_local_ref(ma, an) && is_local_ref(mb, bn))
                    || (is_local_ref(ma, bn) && is_local_ref(mb, an))
                {
                    return Some(BitwiseKind::Xor);
                }
            }
        }
    }
    None
}

/// True if `expr` is `min(1, var(an) + var(bn))` (either arg order for + and min).
fn is_min_1_add(expr: &Expr, an: &str, bn: &str) -> bool {
    let Expr::Calc(CalcOp::Min(args)) = expr else { return false; };
    if args.len() != 2 { return false; }
    let (one_arg, add_arg) = if is_literal(&args[0], 1.0) {
        (&args[0], &args[1])
    } else if is_literal(&args[1], 1.0) {
        (&args[1], &args[0])
    } else {
        return false;
    };
    let _ = one_arg;
    let Expr::Calc(CalcOp::Add(lhs, rhs)) = add_arg else { return false; };
    (is_local_ref(lhs, an) && is_local_ref(rhs, bn))
        || (is_local_ref(lhs, bn) && is_local_ref(rhs, an))
}

/// True if `expr` is `1 - var(an)` (NOT of a single bit local).
fn is_one_minus_local(expr: &Expr, an: &str) -> bool {
    if let Expr::Calc(CalcOp::Sub(lhs, rhs)) = expr {
        return is_literal(lhs, 1.0) && is_local_ref(rhs, an);
    }
    false
}

/// Attempt to classify a function body as a canonical 16-bit bitwise op
/// (AND/OR/XOR/NOT) implemented as bit decomposition + reassembly.
///
/// Returns `Some(kind)` if the body matches; `None` otherwise.
///
/// Genericity: this never looks at the function name, only at body shape.
/// Any 2-param (or 1-param for NOT) function with the right locals and result
/// tree is detected, regardless of what the user named it.
pub(crate) fn classify_bitwise_decomposition(func: &FunctionDef) -> Option<BitwiseKind> {
    let params = &func.parameters;

    // Case 1: 2-param, 32 locals → AND / OR / XOR.
    if params.len() == 2 && func.locals.len() == 32 {
        let pa = &params[0].name;
        let pb = &params[1].name;
        // Locals must be a1..a16 then b1..b16 (by order), each a bit-extract
        // of the corresponding param. We don't require exact names — we match
        // by position: first 16 locals decompose param 0, next 16 decompose param 1.
        let a_names: Vec<&str> = func.locals[..16].iter().map(|l| l.name.as_str()).collect();
        let b_names: Vec<&str> = func.locals[16..].iter().map(|l| l.name.as_str()).collect();
        for i in 0..16 {
            if !is_bit_i_of(&func.locals[i].value, pa, i as u32) { return None; }
            if !is_bit_i_of(&func.locals[16 + i].value, pb, i as u32) { return None; }
        }
        // Result: sum of 16 terms, each (a_i OP b_i) scaled by 2^i.
        let mut terms = Vec::new();
        flatten_add(&func.result, &mut terms);
        if terms.len() != 16 { return None; }
        let mut kind: Option<BitwiseKind> = None;
        for (i, term) in terms.iter().enumerate() {
            let (inner, factor) = split_scaled_term(term);
            let expected = (1u64 << i) as f64;
            if (factor - expected).abs() > f64::EPSILON { return None; }
            let this_kind = classify_pair_combine(inner, a_names[i], b_names[i])?;
            match kind {
                None => kind = Some(this_kind),
                Some(k) if k == this_kind => {}
                _ => return None,
            }
        }
        return kind.filter(|k| matches!(k, BitwiseKind::And | BitwiseKind::Or | BitwiseKind::Xor));
    }

    // Case 2: 1-param, 16 locals → NOT.
    if params.len() == 1 && func.locals.len() == 16 {
        let pa = &params[0].name;
        let a_names: Vec<&str> = func.locals.iter().map(|l| l.name.as_str()).collect();
        for i in 0..16 {
            if !is_bit_i_of(&func.locals[i].value, pa, i as u32) { return None; }
        }
        let mut terms = Vec::new();
        flatten_add(&func.result, &mut terms);
        if terms.len() != 16 { return None; }
        for (i, term) in terms.iter().enumerate() {
            let (inner, factor) = split_scaled_term(term);
            let expected = (1u64 << i) as f64;
            if (factor - expected).abs() > f64::EPSILON { return None; }
            if !is_one_minus_local(inner, a_names[i]) { return None; }
        }
        return Some(BitwiseKind::Not);
    }

    None
}

/// Check if a dispatch table is an identity-read: every entry maps key K → state[K].
pub(crate) fn is_dispatch_identity_read(table: &DispatchTable) -> bool {
    prof!("is_dispatch_identity_read");
    if table.entries.len() < 4 {
        return false;
    }
    for (&key, expr) in &table.entries {
        match expr {
            Expr::Var { name, .. } => {
                if let Some(addr) = property_to_address(name) {
                    if addr as i64 != key {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

/// Strip CSS prop name prefixes (`--`, optional `__0`/`__1`/`__2`).
/// Returns `"mc42"` for `"--__1mc42"`.
fn strip_prop_prefixes(name: &str) -> &str {
    let after_dashes = name.strip_prefix("--").unwrap_or(name);
    after_dashes
        .strip_prefix("__0")
        .or_else(|| after_dashes.strip_prefix("__1"))
        .or_else(|| after_dashes.strip_prefix("__2"))
        .unwrap_or(after_dashes)
}

/// Parse `"mc42"` → `Some(42)`.  Returns None for any other shape.
fn parse_packed_cell_name(bare: &str) -> Option<u64> {
    let rest = bare.strip_prefix("mc")?;
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

/// If `expr` is `mod(var(--mcN), 256)`, return Some(N). Otherwise None.
fn match_low_byte_extract(expr: &Expr) -> Option<u64> {
    let Expr::Calc(CalcOp::Mod(lhs, rhs)) = expr else { return None };
    let Expr::Var { name, .. } = lhs.as_ref() else { return None };
    let n = parse_packed_cell_name(strip_prop_prefixes(name))?;
    if !matches!(rhs.as_ref(), Expr::Literal(v) if (*v - 256.0).abs() < f64::EPSILON) {
        return None;
    }
    Some(n)
}

/// If `expr` is `round(down, var(--mcN) / 256)`, return Some(N). Otherwise None.
/// (Interval defaults to literal 1 in the parser when not specified.)
fn match_high_byte_extract(expr: &Expr) -> Option<u64> {
    let Expr::Calc(CalcOp::Round(RoundStrategy::Down, val, interval)) = expr else { return None };
    if !matches!(interval.as_ref(), Expr::Literal(v) if (*v - 1.0).abs() < f64::EPSILON) {
        return None;
    }
    let Expr::Calc(CalcOp::Div(num, den)) = val.as_ref() else { return None };
    let Expr::Var { name, .. } = num.as_ref() else { return None };
    let n = parse_packed_cell_name(strip_prop_prefixes(name))?;
    if !matches!(den.as_ref(), Expr::Literal(v) if (*v - 256.0).abs() < f64::EPSILON) {
        return None;
    }
    Some(n)
}

/// Check if a dispatch table is a packed-byte read with PACK_SIZE=2.
///
/// Each entry K must be:
/// - K even (=2N): `mod(var(--mcN), 256)` (low byte)
/// - K odd  (=2N+1): `round(down, var(--mcN) / 256)` (high byte)
///
/// Returns `Some(2)` if the whole table matches; otherwise `None`.
/// Future PACK_SIZE values can extend this to return `Some(pack)`.
/// Classify a dispatch table as "near-packed-byte-read": most entries follow the
/// packed-byte-extract shape (key K → byte K%pack of cell `mc{K/pack}`), with a
/// minority of exception entries (literals for ROM, function calls for MMIO, etc.).
///
/// Returns `Some((pack, exception_keys))` if ≥80% of entries match the packed
/// shape AND every packed entry is consistent with a single `pack` value.
/// Returns `None` otherwise.
pub(crate) fn classify_near_packed_byte(table: &DispatchTable) -> Option<(u8, Vec<i64>)> {
    prof!("classify_near_packed_byte");
    if table.entries.len() < 100 {
        return None;
    }
    // Currently we only recognise pack=2 (low/high byte via mod/round-down-div).
    // Higher packs would require additional shape detection.
    let pack: u8 = 2;
    let mut exceptions = Vec::new();
    let mut packed_count: usize = 0;
    for (&key, expr) in &table.entries {
        if key < 0 {
            exceptions.push(key);
            continue;
        }
        let expected_n = (key as u64) / pack as u64;
        let off = (key as u64) % pack as u64;
        let cell_n = if off == 0 {
            match_low_byte_extract(expr)
        } else {
            match_high_byte_extract(expr)
        };
        if cell_n == Some(expected_n) {
            packed_count += 1;
        } else {
            exceptions.push(key);
        }
    }
    // Require ≥80% packed shape — looser than near-identity (90%) because the
    // exceptions for readMem include ROM (~600 bytes) and MMIO function calls.
    if packed_count * 10 < table.entries.len() * 8 {
        return None;
    }
    log::info!(
        "near-packed-byte ({}): {} packed entries, {} exceptions (lit handled in O(1) table)",
        if pack == 2 { "PACK_SIZE=2" } else { "?" },
        packed_count, exceptions.len()
    );
    Some((pack, exceptions))
}

/// Decoded shape of a single rom-disk-style dispatch entry.
#[derive(Debug, Clone)]
struct DiskEntryShape {
    /// The wrapping function call's name (e.g. `"--readDiskByte"`).
    func_name: String,
    /// The property name that holds the lookup key (e.g. `"--__1mc632"`).
    key_property: String,
    /// Multiplier applied to the cell value before adding offset.
    stride: i32,
    /// Per-entry literal offset added to `cell * stride`.
    offset: i32,
}

/// Try to match `mod(var(P), 256) + round(down, var(P) / 256) * 256` —
/// the algebraic identity for "the u16 stored as low/high bytes in cell P".
/// Returns the property name P on a match.
fn match_u16_recompose(expr: &Expr) -> Option<&str> {
    let Expr::Calc(CalcOp::Add(lo, hi_term)) = expr else { return None };
    let Expr::Calc(CalcOp::Mod(lo_var, lo_div)) = lo.as_ref() else { return None };
    let Expr::Var { name: p1, .. } = lo_var.as_ref() else { return None };
    let Expr::Literal(d) = lo_div.as_ref() else { return None };
    if *d != 256.0 { return None }
    let Expr::Calc(CalcOp::Mul(round_term, mul_lit)) = hi_term.as_ref() else { return None };
    let Expr::Literal(m) = mul_lit.as_ref() else { return None };
    if *m != 256.0 { return None }
    let Expr::Calc(CalcOp::Round(RoundStrategy::Down, val, interval)) = round_term.as_ref() else { return None };
    if !matches!(interval.as_ref(), Expr::Literal(v) if *v == 1.0) { return None }
    let Expr::Calc(CalcOp::Div(hi_var, hi_div)) = val.as_ref() else { return None };
    let Expr::Var { name: p2, .. } = hi_var.as_ref() else { return None };
    let Expr::Literal(d2) = hi_div.as_ref() else { return None };
    if *d2 != 256.0 { return None }
    if p1 != p2 { return None }
    Some(p1)
}

/// Try to decode one dispatch entry as a `--func(cell * stride + offset)`
/// shape with the inner cell expression being a u16 recomposition. Returns
/// None for any other shape — the caller treats that as "this entry is not
/// part of a disk window" and falls through to the generic inline-exception
/// path.
fn decode_disk_entry(expr: &Expr) -> Option<DiskEntryShape> {
    let Expr::FunctionCall { name: func_name, args } = expr else { return None };
    if args.len() != 1 { return None }
    let Expr::Calc(CalcOp::Add(mul_term, off_lit)) = &args[0] else { return None };
    let Expr::Literal(off) = off_lit.as_ref() else { return None };
    let Expr::Calc(CalcOp::Mul(cell_term, stride_lit)) = mul_term.as_ref() else { return None };
    let Expr::Literal(stride) = stride_lit.as_ref() else { return None };
    let key_property = match_u16_recompose(cell_term.as_ref())?.to_string();
    Some(DiskEntryShape {
        func_name: func_name.clone(),
        key_property,
        stride: *stride as i32,
        offset: *off as i32,
    })
}

/// Scan the inline-exception entries of a `--readMem` dispatch table and
/// extract a "windowed byte array" descriptor if a contiguous run of keys
/// all decode to `--func((u16 cell) * stride + offset)` with offset matching
/// `key - window_base`, all sharing the same (func, cell, stride).
///
/// The flat array `--func` compiles to is looked up in `flat_dispatch_cache`;
/// a missing entry means `--func` itself isn't yet recognised as a flat
/// dispatch — recognition is order-dependent (the inner function gets
/// compiled first by demand) but in practice for CSS-DOS `--readDiskByte` is
/// compiled before `--readMem`'s exceptions are walked here, so the cache hit
/// is reliable. If we miss, we just bail and the read path falls through to
/// the inline-exception chain (correct, but slower).
///
/// Returns `None` if no qualifying group is found. A real but small group
/// (e.g. <16 entries) is treated as unrecognised — the perf win isn't worth
/// the runtime check; classify-and-bail keeps the code path predictable.
fn recognise_windowed_byte_array(
    table: &DispatchTable,
    inline_exception_keys: &[i64],
    flat_dispatch_cache: &HashMap<String, (u32, i32, i32)>,
    compiled_flat_arrays: &[FlatDispatchArray],
) -> Option<CompiledWindowedByteArray> {
    use std::collections::BTreeMap;
    // Decode each inline-exception entry. Skip ones that aren't disk-shape.
    let mut decoded: BTreeMap<i64, DiskEntryShape> = BTreeMap::new();
    for &k in inline_exception_keys {
        if let Some(entry) = table.entries.get(&k) {
            if let Some(shape) = decode_disk_entry(entry) {
                decoded.insert(k, shape);
            }
        }
    }
    if decoded.is_empty() {
        return None;
    }
    // Find the longest contiguous run with consistent (func, key_property,
    // stride) and offset == (key - first_key). BTreeMap is ordered so a
    // single linear sweep suffices.
    let mut best: Option<(i64, i64, DiskEntryShape)> = None; // (first_key, last_key, shape)
    let mut run_start: Option<(i64, DiskEntryShape)> = None;
    let mut prev_key: Option<i64> = None;
    for (&k, shape) in &decoded {
        let extends = match (&run_start, prev_key) {
            (Some((start_k, start_shape)), Some(pk))
                if k == pk + 1
                    && start_shape.func_name == shape.func_name
                    && start_shape.key_property == shape.key_property
                    && start_shape.stride == shape.stride
                    && shape.offset as i64 == k - start_k =>
            {
                true
            }
            _ => false,
        };
        if !extends {
            // Close out previous run, if any, and start fresh. A new run can
            // start here only if shape.offset == 0 (otherwise it can't anchor
            // a window whose offset == key - window_base).
            if let (Some((sk, ss)), Some(pk)) = (&run_start, prev_key) {
                let len = (pk - sk + 1) as usize;
                let better = best.as_ref().map_or(true, |b| (b.1 - b.0 + 1) < len as i64);
                if better {
                    best = Some((*sk, pk, ss.clone()));
                }
            }
            if shape.offset == 0 {
                run_start = Some((k, shape.clone()));
            } else {
                run_start = None;
            }
        }
        prev_key = Some(k);
    }
    if let (Some((sk, ss)), Some(pk)) = (&run_start, prev_key) {
        let len = (pk - sk + 1) as usize;
        let better = best.as_ref().map_or(true, |b| (b.1 - b.0 + 1) < len as i64);
        if better {
            best = Some((*sk, pk, ss.clone()));
        }
    }
    let (window_base, window_last, shape) = best?;
    let span = (window_last - window_base + 1) as usize;
    if span < 16 {
        // Below this threshold the runtime check costs more than it saves.
        return None;
    }

    // Look up the inner function's flat array. If it isn't in the cache, the
    // recogniser's preconditions weren't met — skip silently.
    let &(array_id, base_key, _default) = flat_dispatch_cache.get(&shape.func_name)?;
    let arr = compiled_flat_arrays.get(array_id as usize)?;

    log::info!(
        "[disk-window] recognised: window=[0x{:X},0x{:X}) {} entries, key cell {}, stride {}, inner func {} (array_id {} base_key {} len {})",
        window_base, window_last + 1, span, shape.key_property, shape.stride,
        shape.func_name, array_id, base_key, arr.values.len()
    );

    Some(CompiledWindowedByteArray {
        window_base: window_base as i32,
        window_end: (window_last + 1) as i32,
        key_cell_property: shape.key_property,
        stride: shape.stride,
        byte_array_base_key: base_key,
        byte_array: arr.values.clone(),
    })
}

/// Classify a dispatch table as "near-identity-read": most entries map key K → state[K],
/// but a small number have non-identity expressions (e.g., computed values for special
/// addresses like self-modifying code patches).
///
/// Returns `Some(exception_keys)` if the table is mostly identity reads (≥90%), with
/// `exception_keys` listing the keys that are NOT identity reads. Returns `None` if the
/// table doesn't qualify.
fn classify_near_identity_read(table: &DispatchTable) -> Option<Vec<i64>> {
    prof!("classify_near_identity_read");
    if table.entries.len() < 100 {
        return None;
    }
    let mut exceptions = Vec::new();
    for (&key, expr) in &table.entries {
        let is_identity = matches!(expr, Expr::Var { name, .. } if {
            property_to_address(name).is_some_and(|addr| addr as i64 == key)
        });
        if !is_identity {
            exceptions.push(key);
        }
    }
    // Must be at least 90% identity
    let identity_count = table.entries.len() - exceptions.len();
    if identity_count * 10 >= table.entries.len() * 9 {
        Some(exceptions)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Constant folding — simplify Expr trees before compilation
// ---------------------------------------------------------------------------

/// Recursively fold constant expressions and eliminate identity operations.
fn const_fold(expr: &Expr) -> Expr {
    prof!("const_fold");
    match expr {
        Expr::Calc(op) => fold_calc(op),
        Expr::StyleCondition {
            branches, fallback, ..
        } => {
            let folded_branches: Vec<StyleBranch> = branches
                .iter()
                .map(|b| StyleBranch {
                    condition: fold_test(&b.condition),
                    then: const_fold(&b.then),
                })
                .collect();
            let folded_fallback = const_fold(fallback);
            Expr::StyleCondition {
                branches: folded_branches,
                fallback: Box::new(folded_fallback),
            }
        }
        Expr::FunctionCall { name, args } => Expr::FunctionCall {
            name: name.clone(),
            args: args.iter().map(const_fold).collect(),
        },
        _ => expr.clone(),
    }
}

/// Fold a StyleTest's value expressions.
fn fold_test(test: &StyleTest) -> StyleTest {
    match test {
        StyleTest::Single { property, value } => StyleTest::Single {
            property: property.clone(),
            value: const_fold(value),
        },
        StyleTest::And(tests) => StyleTest::And(tests.iter().map(fold_test).collect()),
        StyleTest::Or(tests) => StyleTest::Or(tests.iter().map(fold_test).collect()),
    }
}

/// Fold a CalcOp, returning a simplified Expr.
fn fold_calc(op: &CalcOp) -> Expr {
    match op {
        CalcOp::Add(a, b) => {
            let fa = const_fold(a);
            let fb = const_fold(b);
            match (&fa, &fb) {
                (Expr::Literal(x), Expr::Literal(y)) => Expr::Literal(x + y),
                (_, Expr::Literal(v)) if *v == 0.0 => fa,
                (Expr::Literal(v), _) if *v == 0.0 => fb,
                _ => Expr::Calc(CalcOp::Add(Box::new(fa), Box::new(fb))),
            }
        }
        CalcOp::Sub(a, b) => {
            let fa = const_fold(a);
            let fb = const_fold(b);
            match (&fa, &fb) {
                (Expr::Literal(x), Expr::Literal(y)) => Expr::Literal(x - y),
                (_, Expr::Literal(v)) if *v == 0.0 => fa,
                _ => Expr::Calc(CalcOp::Sub(Box::new(fa), Box::new(fb))),
            }
        }
        CalcOp::Mul(a, b) => {
            let fa = const_fold(a);
            let fb = const_fold(b);
            match (&fa, &fb) {
                (Expr::Literal(x), Expr::Literal(y)) => Expr::Literal(x * y),
                (_, Expr::Literal(v)) if *v == 1.0 => fa,
                (Expr::Literal(v), _) if *v == 1.0 => fb,
                (_, Expr::Literal(v)) if *v == 0.0 => Expr::Literal(0.0),
                (Expr::Literal(v), _) if *v == 0.0 => Expr::Literal(0.0),
                _ => Expr::Calc(CalcOp::Mul(Box::new(fa), Box::new(fb))),
            }
        }
        CalcOp::Div(a, b) => {
            let fa = const_fold(a);
            let fb = const_fold(b);
            match (&fa, &fb) {
                (Expr::Literal(x), Expr::Literal(y)) if *y != 0.0 => Expr::Literal(x / y),
                (_, Expr::Literal(v)) if *v == 1.0 => fa,
                _ => Expr::Calc(CalcOp::Div(Box::new(fa), Box::new(fb))),
            }
        }
        CalcOp::Mod(a, b) => {
            let fa = const_fold(a);
            let fb = const_fold(b);
            match (&fa, &fb) {
                (Expr::Literal(x), Expr::Literal(y)) if *y != 0.0 => Expr::Literal(x % y),
                _ => Expr::Calc(CalcOp::Mod(Box::new(fa), Box::new(fb))),
            }
        }
        CalcOp::Pow(a, b) => {
            let fa = const_fold(a);
            let fb = const_fold(b);
            match (&fa, &fb) {
                (Expr::Literal(x), Expr::Literal(y)) => Expr::Literal(x.powf(*y)),
                (_, Expr::Literal(v)) if *v == 0.0 => Expr::Literal(1.0),
                (_, Expr::Literal(v)) if *v == 1.0 => fa,
                _ => Expr::Calc(CalcOp::Pow(Box::new(fa), Box::new(fb))),
            }
        }
        CalcOp::Negate(a) => {
            let fa = const_fold(a);
            match &fa {
                Expr::Literal(v) => Expr::Literal(-v),
                _ => Expr::Calc(CalcOp::Negate(Box::new(fa))),
            }
        }
        CalcOp::Abs(a) => {
            let fa = const_fold(a);
            match &fa {
                Expr::Literal(v) => Expr::Literal(v.abs()),
                _ => Expr::Calc(CalcOp::Abs(Box::new(fa))),
            }
        }
        CalcOp::Sign(a) => {
            let fa = const_fold(a);
            match &fa {
                Expr::Literal(v) => Expr::Literal(if *v > 0.0 {
                    1.0
                } else if *v < 0.0 {
                    -1.0
                } else {
                    0.0
                }),
                _ => Expr::Calc(CalcOp::Sign(Box::new(fa))),
            }
        }
        CalcOp::Min(args) => {
            let folded: Vec<Expr> = args.iter().map(const_fold).collect();
            if folded.iter().all(|e| matches!(e, Expr::Literal(_))) {
                let min = folded
                    .iter()
                    .map(|e| match e {
                        Expr::Literal(v) => *v,
                        _ => unreachable!(),
                    })
                    .fold(f64::INFINITY, f64::min);
                Expr::Literal(min)
            } else {
                Expr::Calc(CalcOp::Min(folded))
            }
        }
        CalcOp::Max(args) => {
            let folded: Vec<Expr> = args.iter().map(const_fold).collect();
            if folded.iter().all(|e| matches!(e, Expr::Literal(_))) {
                let max = folded
                    .iter()
                    .map(|e| match e {
                        Expr::Literal(v) => *v,
                        _ => unreachable!(),
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                Expr::Literal(max)
            } else {
                Expr::Calc(CalcOp::Max(folded))
            }
        }
        CalcOp::Clamp(min, val, max) => {
            let fmin = const_fold(min);
            let fval = const_fold(val);
            let fmax = const_fold(max);
            match (&fmin, &fval, &fmax) {
                (Expr::Literal(mn), Expr::Literal(v), Expr::Literal(mx)) => {
                    Expr::Literal(v.clamp(*mn, *mx))
                }
                _ => Expr::Calc(CalcOp::Clamp(
                    Box::new(fmin),
                    Box::new(fval),
                    Box::new(fmax),
                )),
            }
        }
        CalcOp::Round(strategy, val, interval) => {
            let fval = const_fold(val);
            let fint = const_fold(interval);
            match (&fval, &fint) {
                (Expr::Literal(v), Expr::Literal(i)) if *i != 0.0 => {
                    let result = match strategy {
                        RoundStrategy::Nearest => (v / i).round() * i,
                        RoundStrategy::Up => (v / i).ceil() * i,
                        RoundStrategy::Down => (v / i).floor() * i,
                        RoundStrategy::ToZero => (v / i).trunc() * i,
                    };
                    Expr::Literal(result)
                }
                _ => Expr::Calc(CalcOp::Round(*strategy, Box::new(fval), Box::new(fint))),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compiler — translates Evaluator data into CompiledProgram
// ---------------------------------------------------------------------------

/// Compiler state — tracks slot allocation and property→slot mapping.
///
/// Functions are borrowed to avoid cloning large ASTs (e.g., readMem's
/// 1M-node result expression). Dispatch tables remain owned because the
/// compile_dispatch_call method uses a remove/insert pattern for borrow
/// splitting.
struct Compiler<'a> {
    /// Next available slot index.
    next_slot: Slot,
    /// Map from property name → slot index.
    property_slots: HashMap<String, Slot>,
    /// Functions available for inlining (borrowed — avoids cloning 1M-node ASTs).
    functions: &'a HashMap<String, FunctionDef>,
    /// Recognised dispatch tables (mutably borrowed — the remove/insert
    /// pattern in compile_dispatch_body_then_call needs ownership of one
    /// table at a time, but cloning the whole map costs ~11 s in wasm on
    /// a multi-million-entry cabinet).
    dispatch_tables: &'a mut HashMap<String, DispatchTable>,
    /// Compiled dispatch table data (populated during compilation).
    compiled_dispatches: Vec<CompiledDispatchTable>,
    /// Compiled flat-array dispatch tables (fast path for literal-only dispatches).
    compiled_flat_arrays: Vec<FlatDispatchArray>,
    /// Cache: dispatch table name → compiled table_id.
    dispatch_cache: HashMap<String, Slot>,
    /// Cache: dispatch table name → (array_id, base_key, default) for the flat fast path.
    flat_dispatch_cache: HashMap<String, (u32, i32, i32)>,
    /// Cache: dispatch table name → identity-read classification result.
    identity_read_cache: HashMap<String, bool>,
    /// Cache: dispatch table name → near-identity exception keys (or None).
    near_identity_cache: HashMap<String, Option<Vec<i64>>>,
    /// Cache: dispatch table name → near-packed-byte (pack, exception keys) or None.
    near_packed_cache: HashMap<String, Option<(u8, Vec<i64>)>>,
    /// Cache: dispatch table name → table_id into `packed_cell_tables`.
    packed_cell_table_cache: HashMap<String, u32>,
    /// Packed-cell address tables (one per near-packed-byte dispatch).
    /// `packed_cell_tables[id][N]` = state-var address (negative) for cell `mcN`,
    /// or `0` if N has no entry (read returns 0).
    packed_cell_tables: Vec<Vec<i32>>,
    /// Per-table literal exception array, parallel to packed_cell_tables.
    packed_exception_tables: Vec<PackedExceptionTable>,
    /// Compiled function bodies, indexed by `fn_id`. Populated by
    /// `compile_general_function` on first encounter and reused thereafter.
    compiled_functions: Vec<CompiledFunction>,
    /// Cache: function name → fn_id in `compiled_functions`.
    function_cache: HashMap<String, u32>,
    /// "Windowed byte-array" descriptor extracted from the cabinet's
    /// `--readMem` exception entries (see [`recognise_windowed_byte_array`]).
    /// `None` until pattern recognition runs; consumed by `CompiledProgram`
    /// at end-of-compile.
    recognised_windowed_byte_array: Option<CompiledWindowedByteArray>,
}

impl<'a> Compiler<'a> {
    fn new(
        functions: &'a HashMap<String, FunctionDef>,
        dispatch_tables: &'a mut HashMap<String, DispatchTable>,
    ) -> Self {
        Compiler {
            next_slot: 0,
            property_slots: HashMap::new(),
            functions,
            dispatch_tables,
            compiled_dispatches: Vec::new(),
            compiled_flat_arrays: Vec::new(),
            dispatch_cache: HashMap::new(),
            flat_dispatch_cache: HashMap::new(),
            identity_read_cache: HashMap::new(),
            near_identity_cache: HashMap::new(),
            near_packed_cache: HashMap::new(),
            packed_cell_table_cache: HashMap::new(),
            packed_cell_tables: Vec::new(),
            packed_exception_tables: Vec::new(),
            compiled_functions: Vec::new(),
            function_cache: HashMap::new(),
            recognised_windowed_byte_array: None,
        }
    }

    /// Allocate a fresh temporary slot.
    fn alloc(&mut self) -> Slot {
        let s = self.next_slot;
        self.next_slot += 1;
        s
    }

    /// Compile an Expr into ops, returning the slot holding the result.
    fn compile_expr(&mut self, expr: &Expr, ops: &mut Vec<Op>) -> Slot {
        prof!("compile_expr");
        // Constant-fold before compiling
        let folded = {
            prof!("compile_expr:const_fold");
            const_fold(expr)
        };
        let expr = &folded;
        match expr {
            Expr::Literal(v) => {
                let dst = self.alloc();
                ops.push(Op::LoadLit {
                    dst,
                    val: *v as i32,
                });
                dst
            }

            Expr::StringLiteral(_) | Expr::Concat(_) => {
                let dst = self.alloc();
                ops.push(Op::LoadLit { dst, val: 0 });
                dst
            }

            Expr::Var { name, fallback } => self.compile_var(name, fallback.as_deref(), ops),

            Expr::Calc(calc_op) => self.compile_calc(calc_op, ops),

            Expr::StyleCondition {
                branches, fallback, ..
            } => self.compile_style_condition(branches, fallback, ops),

            Expr::FunctionCall { name, args } => self.compile_function_call(name, args, ops),
        }
    }

    /// Compile a variable reference.
    fn compile_var(&mut self, name: &str, fallback: Option<&Expr>, ops: &mut Vec<Op>) -> Slot {
        prof!("compile_var");
        // If it's a property we've already computed in this tick, use its slot directly.
        // But NOT for buffer-prefixed names (--__0*, --__1*, --__2*) — those explicitly
        // read the previous tick's state, not the current tick's computed value.
        if !is_buffer_copy(name) {
            if let Some(&s) = self.property_slots.get(name) {
                return s;
            }
        }

        // State-mapped property: load from state at compile-time-known address
        if let Some(addr) = property_to_address(name) {
            let dst = self.alloc();
            ops.push(Op::LoadState { dst, addr });
            return dst;
        }


        // Unknown property — use fallback or 0
        if let Some(fb) = fallback {
            return self.compile_expr(fb, ops);
        }

        let dst = self.alloc();
        ops.push(Op::LoadLit { dst, val: 0 });
        dst
    }

    /// Compile a CalcOp.
    fn compile_calc(&mut self, op: &CalcOp, ops: &mut Vec<Op>) -> Slot {
        prof!("compile_calc");
        match op {
            CalcOp::Add(a, b) => {
                // Lit-operand fast path: if either side is a literal, emit AddLit.
                if let Expr::Literal(lv) = b.as_ref() {
                    if let Some(val) = lit_as_i32(*lv) {
                        let sa = self.compile_expr(a, ops);
                        let dst = self.alloc();
                        ops.push(Op::AddLit { dst, a: sa, val });
                        return dst;
                    }
                }
                if let Expr::Literal(lv) = a.as_ref() {
                    if let Some(val) = lit_as_i32(*lv) {
                        let sb = self.compile_expr(b, ops);
                        let dst = self.alloc();
                        ops.push(Op::AddLit { dst, a: sb, val });
                        return dst;
                    }
                }
                let sa = self.compile_expr(a, ops);
                let sb = self.compile_expr(b, ops);
                let dst = self.alloc();
                ops.push(Op::Add { dst, a: sa, b: sb });
                dst
            }
            CalcOp::Sub(a, b) => {
                if let Expr::Literal(lv) = b.as_ref() {
                    if let Some(val) = lit_as_i32(*lv) {
                        let sa = self.compile_expr(a, ops);
                        let dst = self.alloc();
                        ops.push(Op::SubLit { dst, a: sa, val });
                        return dst;
                    }
                }
                let sa = self.compile_expr(a, ops);
                let sb = self.compile_expr(b, ops);
                let dst = self.alloc();
                ops.push(Op::Sub { dst, a: sa, b: sb });
                dst
            }
            CalcOp::Mul(a, b) => {
                if let Expr::Literal(lv) = b.as_ref() {
                    if let Some(val) = lit_as_i32(*lv) {
                        let sa = self.compile_expr(a, ops);
                        let dst = self.alloc();
                        ops.push(Op::MulLit { dst, a: sa, val });
                        return dst;
                    }
                }
                if let Expr::Literal(lv) = a.as_ref() {
                    if let Some(val) = lit_as_i32(*lv) {
                        let sb = self.compile_expr(b, ops);
                        let dst = self.alloc();
                        ops.push(Op::MulLit { dst, a: sb, val });
                        return dst;
                    }
                }
                let sa = self.compile_expr(a, ops);
                let sb = self.compile_expr(b, ops);
                let dst = self.alloc();
                ops.push(Op::Mul { dst, a: sa, b: sb });
                dst
            }
            CalcOp::Div(a, b) => {
                let sa = self.compile_expr(a, ops);
                let sb = self.compile_expr(b, ops);
                let dst = self.alloc();
                ops.push(Op::Div { dst, a: sa, b: sb });
                dst
            }
            CalcOp::Mod(a, b) => {
                if let Expr::Literal(lv) = b.as_ref() {
                    if let Some(val) = lit_as_i32(*lv) {
                        if val != 0 {
                            let sa = self.compile_expr(a, ops);
                            let dst = self.alloc();
                            ops.push(Op::ModLit { dst, a: sa, val });
                            return dst;
                        }
                    }
                }
                let sa = self.compile_expr(a, ops);
                let sb = self.compile_expr(b, ops);
                let dst = self.alloc();
                ops.push(Op::Mod { dst, a: sa, b: sb });
                dst
            }
            CalcOp::Min(args) => {
                let slots: Vec<Slot> = args.iter().map(|a| self.compile_expr(a, ops)).collect();
                let dst = self.alloc();
                ops.push(Op::Min { dst, args: slots });
                dst
            }
            CalcOp::Max(args) => {
                let slots: Vec<Slot> = args.iter().map(|a| self.compile_expr(a, ops)).collect();
                let dst = self.alloc();
                ops.push(Op::Max { dst, args: slots });
                dst
            }
            CalcOp::Clamp(min, val, max) => {
                let smin = self.compile_expr(min, ops);
                let sval = self.compile_expr(val, ops);
                let smax = self.compile_expr(max, ops);
                let dst = self.alloc();
                ops.push(Op::Clamp {
                    dst,
                    min: smin,
                    val: sval,
                    max: smax,
                });
                dst
            }
            CalcOp::Round(strategy, val, interval) => {
                let sval = self.compile_expr(val, ops);
                let sint = self.compile_expr(interval, ops);
                let dst = self.alloc();
                ops.push(Op::Round {
                    dst,
                    strategy: *strategy,
                    val: sval,
                    interval: sint,
                });
                dst
            }
            CalcOp::Pow(base, exp) => {
                let sb = self.compile_expr(base, ops);
                let se = self.compile_expr(exp, ops);
                let dst = self.alloc();
                ops.push(Op::Pow {
                    dst,
                    base: sb,
                    exp: se,
                });
                dst
            }
            CalcOp::Sign(val) => {
                let sv = self.compile_expr(val, ops);
                let dst = self.alloc();
                ops.push(Op::Sign { dst, src: sv });
                dst
            }
            CalcOp::Abs(val) => {
                let sv = self.compile_expr(val, ops);
                let dst = self.alloc();
                ops.push(Op::Abs { dst, src: sv });
                dst
            }
            CalcOp::Negate(val) => {
                let sv = self.compile_expr(val, ops);
                let dst = self.alloc();
                ops.push(Op::Neg { dst, src: sv });
                dst
            }
        }
    }

    /// Compile a StyleCondition (if/else chain) into branch ops.
    ///
    /// If all branches test the same property against integer literals (a dispatch
    /// table pattern), compiles to a `Dispatch` op (O(1) HashMap lookup) instead of
    /// a linear branch chain. This handles both function-body dispatch tables and
    /// v2-style per-register opcode dispatches in assignments.
    fn compile_style_condition(
        &mut self,
        branches: &[StyleBranch],
        fallback: &Expr,
        ops: &mut Vec<Op>,
    ) -> Slot {
        prof!("compile_style_condition");
        // Try dispatch table optimization for large single-key chains
        if let Some(table) = dispatch_table::recognise_dispatch(branches, fallback) {
            return self.compile_inline_dispatch(&table, ops);
        }


        // Fall back to linear branch chain
        self.compile_style_condition_linear(branches, fallback, ops)
    }

    /// Compile a recognised dispatch table directly (no function call wrapper).
    fn compile_inline_dispatch(
        &mut self,
        table: &DispatchTable,
        ops: &mut Vec<Op>,
    ) -> Slot {
        let _dt = web_time::Instant::now();
        let key_slot = self.compile_var(&table.key_property, None, ops);

        let mut compiled_entries: FxMap<i64, (Vec<Op>, Slot)> = FxMap::default();
        for (&key_val, entry_expr) in &table.entries {
            let mut entry_ops = Vec::new();
            let result = self.compile_expr(entry_expr, &mut entry_ops);
            compiled_entries.insert(key_val, (entry_ops, result));
        }

        let mut fallback_ops = Vec::new();
        let fallback_slot = self.compile_expr(&table.fallback, &mut fallback_ops);

        let table_id = self.compiled_dispatches.len() as Slot;
        if compiled_entries.len() >= 100 {
            log::info!("[compile detail] inline dispatch on {}: {} entries, {:.2}s",
                table.key_property, compiled_entries.len(), _dt.elapsed().as_secs_f64());
        }
        self.compiled_dispatches.push(CompiledDispatchTable {
            entries: compiled_entries,
            fallback_ops,
            fallback_slot,
        });

        let dst = self.alloc();
        ops.push(Op::Dispatch {
            dst,
            key: key_slot,
            table_id,
            fallback_target: 0,
        });
        dst
    }

    /// Compile a StyleCondition as a linear branch chain (fallback path).
    fn compile_style_condition_linear(
        &mut self,
        branches: &[StyleBranch],
        fallback: &Expr,
        ops: &mut Vec<Op>,
    ) -> Slot {
        // Result goes into a single destination slot
        let result_slot = self.alloc();

        // We emit a chain: for each branch, test condition, if true compute then
        // and jump to end, else fall through to next branch.
        // Patch targets are filled in after all branches are emitted.
        //
        // IMPORTANT: Each branch's 'then' expression is compiled in isolation
        // (its own entry_ops) and executed via a Dispatch-like mechanism. This
        // prevents slot aliasing across branches — a slot set by one branch's
        // compilation won't be stale-read by another branch at runtime.
        let mut jump_to_end: Vec<usize> = Vec::new();

        for branch in branches {
            // Fast path: if the condition is a conjunction of `style(--prop: literal)`
            // singletons, emit a chain of BranchIfNotEqLit ops going straight to the
            // next-branch target. Avoids the generic style_test machinery (Mul, intermediate
            // bool slots) and skips the fuse pass entirely. Covers the dominant BIOS
            // pattern `style(--opcode: N) and style(--uOp: K)`.
            let mut fast_branch_idxs: Vec<usize> = Vec::new();
            let used_fast = collect_literal_and_tests(&branch.condition)
                .map(|tests| {
                    for (prop, lit) in &tests {
                        let prop_slot = self.compile_var(prop, None, ops);
                        fast_branch_idxs.push(ops.len());
                        ops.push(Op::BranchIfNotEqLit {
                            a: prop_slot,
                            val: *lit,
                            target: 0, // patched after branch body emitted
                        });
                    }
                    true
                })
                .unwrap_or(false);

            let branch_idx = if used_fast {
                usize::MAX // sentinel; fast_branch_idxs handles patching
            } else {
                let cond_slot = self.compile_style_test(&branch.condition, ops);
                let idx = ops.len();
                ops.push(Op::BranchIfZero {
                    cond: cond_slot,
                    target: 0,
                });
                idx
            };
            // keep the original `branch_idx` name working below
            let _ = ();

            // Save and restore property_slots around each branch to prevent
            // cross-branch slot aliasing. Without this, a function call in
            // branch A can leave a parameter binding in property_slots that
            // branch B's compilation reuses — leading to stale slot reads
            // at runtime when only one branch executes.
            let saved_props: Vec<(String, Slot)> = self.property_slots.iter()
                .map(|(k, &v)| (k.clone(), v))
                .collect();

            // Condition true: compute 'then' value
            let ops_before = ops.len();
            let then_slot = self.compile_expr(&branch.then, ops);
            let ops_after = ops.len();
            // Log ops for the branch that produces memAddr for opcode 214 uOp 1
            if ops_after - ops_before > 10 && ops_after - ops_before < 5000 {
                log::warn!("[linear branch] {} ops (idx {}-{}) result_slot={} then_slot={}",
                    ops_after - ops_before, ops_before, ops_after, result_slot, then_slot);
                for j in ops_before..std::cmp::min(ops_after, ops_before + 30) {
                    log::warn!("  [{:4}] {:?}", j, ops[j]);
                }
            }
            ops.push(Op::LoadSlot {
                dst: result_slot,
                src: then_slot,
            });

            // Restore property_slots to prevent cross-branch contamination
            self.property_slots.clear();
            for (k, v) in saved_props {
                self.property_slots.insert(k, v);
            }

            // Jump to end
            jump_to_end.push(ops.len());
            ops.push(Op::Jump { target: 0 }); // target patched later

            // Patch the branch(s) to jump here (the next branch / fallback)
            let next_idx = ops.len() as u32;
            if used_fast {
                for idx in fast_branch_idxs.iter().copied() {
                    if let Op::BranchIfNotEqLit { target, .. } = &mut ops[idx] {
                        *target = next_idx;
                    }
                }
            } else if let Some(Op::BranchIfZero { target: tgt, .. }) = ops.get_mut(branch_idx) {
                *tgt = next_idx;
            }
        }

        // Fallback
        let fb_slot = self.compile_expr(fallback, ops);
        ops.push(Op::LoadSlot {
            dst: result_slot,
            src: fb_slot,
        });

        // Patch all jump-to-end targets
        let end_idx = ops.len() as u32;
        for idx in jump_to_end {
            if let Op::Jump { target } = &mut ops[idx] {
                *target = end_idx;
            }
        }

        result_slot
    }

    /// Compile a StyleTest into a boolean (0 or 1) in a slot.
    fn compile_style_test(&mut self, test: &StyleTest, ops: &mut Vec<Op>) -> Slot {
        match test {
            StyleTest::Single { property, value } => {
                let prop_slot = self.compile_var(property, None, ops);
                let val_slot = self.compile_expr(value, ops);
                let dst = self.alloc();
                ops.push(Op::CmpEq {
                    dst,
                    a: prop_slot,
                    b: val_slot,
                });
                dst
            }
            StyleTest::And(tests) => {
                // All must be true: short-circuit chain
                // Start with 1 (true), AND each result
                let result = self.alloc();
                ops.push(Op::LoadLit {
                    dst: result,
                    val: 1,
                });

                for t in tests {
                    let t_slot = self.compile_style_test(t, ops);
                    // If result is already 0, skip (BranchIfZero past the mul)
                    let check_idx = ops.len();
                    ops.push(Op::BranchIfZero {
                        cond: result,
                        target: 0,
                    });
                    // result = result * t_slot (both are 0 or 1, so this is AND)
                    ops.push(Op::Mul {
                        dst: result,
                        a: result,
                        b: t_slot,
                    });
                    let after = ops.len() as u32;
                    if let Op::BranchIfZero { target, .. } = &mut ops[check_idx] {
                        *target = after;
                    }
                }
                result
            }
            StyleTest::Or(tests) => {
                // Any must be true
                let result = self.alloc();
                ops.push(Op::LoadLit {
                    dst: result,
                    val: 0,
                });
                let mut jumps_to_end = Vec::new();

                for t in tests {
                    let t_slot = self.compile_style_test(t, ops);
                    // result = t_slot (store latest)
                    ops.push(Op::LoadSlot {
                        dst: result,
                        src: t_slot,
                    });
                    // If result is now nonzero, we're done — but BranchIfZero
                    // only jumps on zero, so we need the inverse logic.
                    // We'll use: if result != 0, jump to end.
                    // Implement as: branch-if-zero past the jump, then jump to end.
                    let check_idx = ops.len();
                    ops.push(Op::BranchIfZero {
                        cond: result,
                        target: 0,
                    });
                    jumps_to_end.push(ops.len());
                    ops.push(Op::Jump { target: 0 });
                    let after = ops.len() as u32;
                    if let Op::BranchIfZero { target, .. } = &mut ops[check_idx] {
                        *target = after;
                    }
                }

                let end = ops.len() as u32;
                for idx in jumps_to_end {
                    if let Op::Jump { target } = &mut ops[idx] {
                        *target = end;
                    }
                }
                result
            }
        }
    }

    /// Compile a function call — uses body-pattern analysis for optimisation.
    ///
    /// Instead of matching function names, this analyses the function's body
    /// structure to detect mathematical patterns that can be compiled to
    /// efficient native operations. This is fully generic — it works for
    /// any CSS function with the right shape.
    fn compile_function_call(&mut self, name: &str, args: &[Expr], ops: &mut Vec<Op>) -> Slot {
        prof!("compile_function_call");
        // Try body-pattern analysis on the function definition.
        // Since self.functions is &'a (external borrow), this reference doesn't
        // conflict with &mut self for compile_expr calls.
        if let Some(func) = self.functions.get(name) {
            if let Some(slot) = self.try_compile_by_body_pattern(func, args, ops) {
                return slot;
            }
        }

        // Dispatch table: check for identity-read pattern, then fall back to
        // compiled dispatch lookup.
        if self.dispatch_tables.contains_key(name) {
            // Identity-read: every entry maps key K → state[K] — direct memory read
            if args.len() == 1 && self.check_dispatch_identity_read(name) {
                let addr_slot = self.compile_expr(&args[0], ops);
                let dst = self.alloc();
                ops.push(Op::LoadMem { dst, addr_slot });
                return dst;
            }
            // Near-identity-read: mostly identity with a few exception entries.
            // Compile as LoadMem + small exception dispatch instead of full 116K-entry table.
            if args.len() == 1 {
                if let Some(exception_keys) = self.check_near_identity_read(name) {
                    return self.compile_near_identity_dispatch(name, args, &exception_keys, ops);
                }
                // Near-packed-byte-read: CSS-DOS PACK_SIZE=2 memory layout.
                // Compile as LoadPackedByte + small exception dispatch.
                if let Some((pack, exception_keys)) = self.check_near_packed_byte(name) {
                    return self.compile_near_packed_byte_dispatch(
                        name, args, pack, &exception_keys, ops,
                    );
                }
            }
            return self.compile_dispatch_call(name, args, ops);
        }

        // General function: inline the body
        self.compile_general_function(name, args, ops)
    }

    /// Try to compile a function call by analysing its body pattern.
    ///
    /// Returns `Some(result_slot)` if the body matches a known mathematical
    /// pattern that can be compiled to efficient native ops.
    fn try_compile_by_body_pattern(
        &mut self,
        func: &FunctionDef,
        args: &[Expr],
        ops: &mut Vec<Op>,
    ) -> Option<Slot> {
        prof!("try_compile_by_body_pattern");
        let params = &func.parameters;

        // Identity: 1 param, no locals, result = var(param)
        if params.len() == 1 && func.locals.is_empty() && is_var_ref(&func.result, &params[0].name)
        {
            return args.first().map(|a| self.compile_expr(a, ops));
        }

        // Word read: 1 param, body = calc(readMem(param) + readMem(param+1) * 256)
        // Compile as LoadMem16 instead of two separate readMem calls — BUT ONLY
        // if the inner readMem is a pure identity read. If it has literal
        // exception entries (e.g. BIOS ROM constants encoded as
        // `style(--at: N): 201` inside --readMem's dispatch), LoadMem16 would
        // bypass those and read 0 from state. In that case we must fall through
        // to general compilation so the two nested readMem calls each go
        // through compile_near_identity_dispatch and pick up the exceptions.
        if params.len() == 1 && func.locals.is_empty() {
            if let Some(inner_name) = is_word_read_pattern(&func.result, &params[0].name) {
                let inner_is_pure_identity = self
                    .dispatch_tables
                    .get(&inner_name)
                    .map(is_dispatch_identity_read)
                    .unwrap_or(true);
                if inner_is_pure_identity {
                    let addr_slot = self.compile_expr(&args[0], ops);
                    let dst = self.alloc();
                    ops.push(Op::LoadMem16 { dst, addr_slot });
                    return Some(dst);
                }
                // Else: fall through to general function-call compilation.
            }
        }

        // 2-param patterns with no locals
        if params.len() == 2 && func.locals.is_empty() {
            let p0 = &params[0].name;
            let p1 = &params[1].name;

            // Bitmask: mod(a, pow(2, b)) → And
            if is_mod_pow2(&func.result, p0, p1) {
                // If b is a literal, emit AndLit with precomputed mask.
                if let Expr::Literal(lv) = &args[1] {
                    if let Some(bits) = lit_as_i32(*lv) {
                        if (0..=31).contains(&bits) {
                            let mask = ((1i64 << bits) - 1) as i32;
                            let sa = self.compile_expr(&args[0], ops);
                            let dst = self.alloc();
                            ops.push(Op::AndLit { dst, a: sa, val: mask });
                            return Some(dst);
                        }
                    }
                }
                let sa = self.compile_expr(&args[0], ops);
                let sb = self.compile_expr(&args[1], ops);
                let dst = self.alloc();
                ops.push(Op::And { dst, a: sa, b: sb });
                return Some(dst);
            }

            // Right shift: round(down, a / pow(2, b), 1) → Shr
            if is_right_shift(&func.result, p0, p1) {
                if let Expr::Literal(lv) = &args[1] {
                    if let Some(val) = lit_as_i32(*lv) {
                        if (0..32).contains(&val) {
                            let sa = self.compile_expr(&args[0], ops);
                            let dst = self.alloc();
                            ops.push(Op::ShrLit { dst, a: sa, val });
                            return Some(dst);
                        }
                    }
                }
                let sa = self.compile_expr(&args[0], ops);
                let sb = self.compile_expr(&args[1], ops);
                let dst = self.alloc();
                ops.push(Op::Shr { dst, a: sa, b: sb });
                return Some(dst);
            }

            // Left shift: a * pow(2, b) → Shl
            if is_left_shift(&func.result, p0, p1) {
                if let Expr::Literal(lv) = &args[1] {
                    if let Some(val) = lit_as_i32(*lv) {
                        if (0..32).contains(&val) {
                            let sa = self.compile_expr(&args[0], ops);
                            let dst = self.alloc();
                            ops.push(Op::ShlLit { dst, a: sa, val });
                            return Some(dst);
                        }
                    }
                }
                let sa = self.compile_expr(&args[0], ops);
                let sb = self.compile_expr(&args[1], ops);
                let dst = self.alloc();
                ops.push(Op::Shl { dst, a: sa, b: sb });
                return Some(dst);
            }

            // Bit extract: mod(rightShift_body(a, b), 2) → Bit
            // i.e. mod(round(down, a / pow(2, b), 1), 2)
            if is_bit_extract(&func.result, p0, p1, &self.functions) {
                let sv = self.compile_expr(&args[0], ops);
                let si = self.compile_expr(&args[1], ops);
                let dst = self.alloc();
                ops.push(Op::Bit {
                    dst,
                    val: sv,
                    idx: si,
                });
                return Some(dst);
            }
        }

        // 16-bit bitwise decomposition (AND/OR/XOR) — 2-param, 32-local form.
        // Pattern: per-bit extract of each param, result = sum of per-bit
        // combine * 2^i. See classify_bitwise_decomposition for the shape.
        if params.len() == 2 && func.locals.len() == 32 {
            if let Some(kind) = classify_bitwise_decomposition(func) {
                let sa = self.compile_expr(&args[0], ops);
                let sb = self.compile_expr(&args[1], ops);
                let dst = self.alloc();
                let op = match kind {
                    BitwiseKind::And => Op::BitAnd16 { dst, a: sa, b: sb },
                    BitwiseKind::Or => Op::BitOr16 { dst, a: sa, b: sb },
                    BitwiseKind::Xor => Op::BitXor16 { dst, a: sa, b: sb },
                    BitwiseKind::Not => unreachable!("NOT is 1-param"),
                };
                ops.push(op);
                return Some(dst);
            }
        }

        // 16-bit bitwise NOT — 1-param, 16-local form.
        if params.len() == 1 && func.locals.len() == 16 {
            if let Some(BitwiseKind::Not) = classify_bitwise_decomposition(func) {
                let sa = self.compile_expr(&args[0], ops);
                let dst = self.alloc();
                ops.push(Op::BitNot16 { dst, a: sa });
                return Some(dst);
            }
        }

        // 2-param with 1 local: left-shift via power-of-2 dispatch table
        // Pattern: local = dispatch_on(b) {0→1, 1→2, 2→4, ...}, result = a * local
        if params.len() == 2 && func.locals.len() == 1 {
            let p0 = &params[0].name;
            let p1 = &params[1].name;
            let local = &func.locals[0];

            if is_mul_refs(&func.result, p0, &local.name) && is_pow2_dispatch(&local.value, p1) {
                let sa = self.compile_expr(&args[0], ops);
                let sb = self.compile_expr(&args[1], ops);
                let dst = self.alloc();
                ops.push(Op::Shl { dst, a: sa, b: sb });
                return Some(dst);
            }
        }

        None
    }

    /// Check if a dispatch table is an identity-read pattern (cached).
    fn check_dispatch_identity_read(&mut self, name: &str) -> bool {
        prof!("check_dispatch_identity_read");
        if let Some(&cached) = self.identity_read_cache.get(name) {
            return cached;
        }
        let result = self.dispatch_tables
            .get(name)
            .is_some_and(is_dispatch_identity_read);
        self.identity_read_cache.insert(name.to_string(), result);
        result
    }

    /// Classify a dispatch table as near-identity-read (cached).
    fn check_near_identity_read(&mut self, name: &str) -> Option<Vec<i64>> {
        prof!("check_near_identity_read");
        if let Some(cached) = self.near_identity_cache.get(name) {
            return cached.clone();
        }
        let result = self.dispatch_tables
            .get(name)
            .and_then(classify_near_identity_read);
        self.near_identity_cache.insert(name.to_string(), result.clone());
        result
    }

    /// Compile a near-identity-read dispatch table.
    ///
    /// For tables where 99%+ of entries are identity reads (key K → state[K]),
    /// we emit LoadMem as the default path and only compile a small dispatch table
    /// for the exception entries. This avoids recompiling 116K+ entries on each call.
    fn compile_near_identity_dispatch(
        &mut self,
        name: &str,
        args: &[Expr],
        exception_keys: &[i64],
        ops: &mut Vec<Op>,
    ) -> Slot {
        prof!("compile_near_identity_dispatch");
        let table = self.dispatch_tables.remove(name).unwrap();
        let func = self.functions.get(name);

        // Bind arguments to parameter slots.
        //
        // IMPORTANT: We must ensure the parameter's LoadLit/LoadSlot is in THIS ops
        // context. If the argument compiles to an existing slot (e.g., Var("--at") from
        // an outer function scope), that slot's value was set by ops in a different
        // dispatch branch that may not execute. We allocate a fresh local slot and
        // emit a copy to guarantee the value is available in this ops sequence.
        let saved: Vec<(String, Option<Slot>)> = if let Some(ref f) = func {
            f.parameters
                .iter()
                .enumerate()
                .map(|(i, param)| {
                    let old = self.property_slots.get(&param.name).copied();
                    let compiled_slot = args
                        .get(i)
                        .map(|a| self.compile_expr(a, ops))
                        .unwrap_or_else(|| {
                            let s = self.alloc();
                            ops.push(Op::LoadLit { dst: s, val: 0 });
                            s
                        });
                    // Always emit a fresh copy into this ops context to prevent
                    // stale reads from dispatch entries that didn't execute.
                    let local_slot = self.alloc();
                    ops.push(Op::LoadSlot { dst: local_slot, src: compiled_slot });
                    self.property_slots.insert(param.name.clone(), local_slot);
                    (param.name.clone(), old)
                })
                .collect()
        } else {
            Vec::new()
        };

        // Compile the key (address) lookup
        let key_slot = self.compile_var(&table.key_property, None, ops);

        // Default path: LoadMem — works for all identity-read entries
        let dst = self.alloc();
        ops.push(Op::LoadMem {
            dst,
            addr_slot: key_slot,
        });

        // Build a small dispatch table for just the exception entries
        if !exception_keys.is_empty() {
            let mut compiled_entries = HashMap::new();
            for &key_val in exception_keys {
                if let Some(entry_expr) = table.entries.get(&key_val) {
                    let mut entry_ops = Vec::new();
                    let result = self.compile_expr(entry_expr, &mut entry_ops);
                    compiled_entries.insert(key_val, (entry_ops, result));
                }
            }
            // Instead of a nested Dispatch table (whose fallback would reference
            // slots from this ops context — broken by slot compaction since the
            // fallback_ops are compacted in a separate scope), inline the exception
            // check as a branch chain in the calling ops. This keeps all slot
            // references in the same scope, avoiding cross-scope aliasing bugs.
            //
            // For each exception key K with value V:
            //   CmpEq(key_slot, K) → if match, compute V, jump to end
            // Default: use LoadMem result (already in dst)
            let result_slot = self.alloc();
            ops.push(Op::LoadSlot { dst: result_slot, src: dst }); // default = LoadMem result

            let mut jumps_to_end: Vec<usize> = Vec::new();
            for (&key_val, (entry_ops, entry_result)) in &compiled_entries {
                let lit = self.alloc();
                ops.push(Op::LoadLit { dst: lit, val: key_val as i32 });
                let cmp = self.alloc();
                ops.push(Op::CmpEq { dst: cmp, a: key_slot, b: lit });
                let branch_idx = ops.len();
                ops.push(Op::BranchIfZero { cond: cmp, target: 0 }); // patched later

                // Inline the exception entry's ops
                ops.extend_from_slice(entry_ops);
                ops.push(Op::LoadSlot { dst: result_slot, src: *entry_result });
                jumps_to_end.push(ops.len());
                ops.push(Op::Jump { target: 0 }); // patched later

                let next_idx = ops.len() as u32;
                if let Op::BranchIfZero { target, .. } = &mut ops[branch_idx] {
                    *target = next_idx;
                }
            }
            let end_idx = ops.len() as u32;
            for idx in jumps_to_end {
                if let Op::Jump { target } = &mut ops[idx] {
                    *target = end_idx;
                }
            }

            let override_dst = result_slot;
            // Use the dispatch result (which falls back to LoadMem result for non-exceptions)
            // Restore and return
            self.dispatch_tables.insert(name.to_string(), table);
            for (param_name, old) in saved {
                match old {
                    Some(s) => { self.property_slots.insert(param_name, s); }
                    None => { self.property_slots.remove(&param_name); }
                }
            }
            return override_dst;
        }

        // No exceptions — just LoadMem
        self.dispatch_tables.insert(name.to_string(), table);
        for (param_name, old) in saved {
            match old {
                Some(s) => { self.property_slots.insert(param_name, s); }
                None => { self.property_slots.remove(&param_name); }
            }
        }
        dst
    }

    /// Classify a dispatch table as near-packed-byte (cached).
    fn check_near_packed_byte(&mut self, name: &str) -> Option<(u8, Vec<i64>)> {
        prof!("check_near_packed_byte");
        if let Some(cached) = self.near_packed_cache.get(name) {
            return cached.clone();
        }
        let result = self.dispatch_tables.get(name).and_then(classify_near_packed_byte);
        self.near_packed_cache.insert(name.to_string(), result.clone());
        result
    }

    /// Build (or reuse) a packed_cell_table for `name`, returning the table_id.
    /// The table maps cell index N → state-var address (negative). N out of bounds
    /// or with no entry has the value 0 (read returns 0).
    ///
    /// Also populates the parallel `packed_exception_tables[id]` with any entries
    /// whose expression is a compile-time literal (e.g., ROM bytes). These are
    /// handled inside Op::LoadPackedByte at runtime via an O(1) HashMap lookup
    /// BEFORE the packed-byte extract, avoiding per-call-site inline branches.
    fn get_or_build_packed_cell_table(&mut self, name: &str, pack: u8) -> u32 {
        if let Some(&id) = self.packed_cell_table_cache.get(name) {
            return id;
        }
        let table = self.dispatch_tables.get(name).expect("dispatch table exists");
        // Find max N referenced by packed entries + collect literal exceptions.
        let mut max_n: i64 = -1;
        let mut literal_exceptions: Vec<(i64, i32)> = Vec::new();
        for (&key, expr) in &table.entries {
            if key < 0 { continue; }
            let expected_n = (key as u64) / pack as u64;
            let off = (key as u64) % pack as u64;
            let cell_n = if off == 0 {
                match_low_byte_extract(expr)
            } else {
                match_high_byte_extract(expr)
            };
            if cell_n == Some(expected_n) {
                max_n = max_n.max(expected_n as i64);
            } else if let Expr::Literal(v) = expr {
                literal_exceptions.push((key, *v as i32));
            }
            // Non-literal exceptions (e.g. function calls) fall through to the
            // inline-branch path in compile_near_packed_byte_dispatch.
        }
        let len = (max_n + 1) as usize;
        let mut addrs = vec![0i32; len];
        for n in 0..len {
            let cell_name = format!("mc{}", n);
            if let Some(addr) = crate::eval::property_to_address(&cell_name) {
                addrs[n] = addr;
            }
        }
        // Build the dense exception table.
        let exc_table = if literal_exceptions.is_empty() {
            PackedExceptionTable::empty()
        } else {
            let min_k = literal_exceptions.iter().map(|(k, _)| *k).min().unwrap();
            let max_k = literal_exceptions.iter().map(|(k, _)| *k).max().unwrap();
            let span = (max_k - min_k + 1) as usize;
            // Cap span to avoid pathological tables (e.g. one exception at key 0
            // and another at key 999999). Above this cap, fall back to inline
            // dispatch by emitting an empty table.
            const MAX_SPAN: usize = 1_000_000;
            if span > MAX_SPAN {
                PackedExceptionTable::empty()
            } else {
                let mut values = vec![i32::MIN; span];
                for (k, v) in &literal_exceptions {
                    values[(*k - min_k) as usize] = *v;
                }
                PackedExceptionTable { base: min_k as i32, values }
            }
        };
        let id = self.packed_cell_tables.len() as u32;
        self.packed_cell_tables.push(addrs);
        self.packed_exception_tables.push(exc_table);
        self.packed_cell_table_cache.insert(name.to_string(), id);
        id
    }

    /// Compile a near-packed-byte-read dispatch table.
    ///
    /// Default path: `Op::LoadPackedByte` (key → cell N → byte). Exception entries
    /// (literals, function calls, anything not the packed-byte shape) go through
    /// a small CmpEq/BranchIfZero chain — same approach as compile_near_identity_dispatch.
    fn compile_near_packed_byte_dispatch(
        &mut self,
        name: &str,
        args: &[Expr],
        pack: u8,
        exception_keys: &[i64],
        ops: &mut Vec<Op>,
    ) -> Slot {
        prof!("compile_near_packed_byte_dispatch");
        let table_id = self.get_or_build_packed_cell_table(name, pack);
        let table = self.dispatch_tables.remove(name).unwrap();
        let func = self.functions.get(name);

        // Bind arguments to parameter slots (mirror compile_near_identity_dispatch).
        let saved: Vec<(String, Option<Slot>)> = if let Some(ref f) = func {
            f.parameters
                .iter()
                .enumerate()
                .map(|(i, param)| {
                    let old = self.property_slots.get(&param.name).copied();
                    let compiled_slot = args
                        .get(i)
                        .map(|a| self.compile_expr(a, ops))
                        .unwrap_or_else(|| {
                            let s = self.alloc();
                            ops.push(Op::LoadLit { dst: s, val: 0 });
                            s
                        });
                    let local_slot = self.alloc();
                    ops.push(Op::LoadSlot { dst: local_slot, src: compiled_slot });
                    self.property_slots.insert(param.name.clone(), local_slot);
                    (param.name.clone(), old)
                })
                .collect()
        } else {
            Vec::new()
        };

        // Compile the key (address) lookup.
        let key_slot = self.compile_var(&table.key_property, None, ops);

        // Default path: LoadPackedByte.
        let dst = self.alloc();
        ops.push(Op::LoadPackedByte {
            dst,
            key_slot,
            table_id,
            pack,
        });

        // Exception entries → CmpEq/BranchIfZero chain. Literal exceptions are
        // already handled by the LoadPackedByte runtime via the per-table
        // packed_exception_tables HashMap (O(1) lookup, shared across call sites).
        // Only inline non-literal exceptions (function calls, MMIO, etc.).
        let inline_exceptions: Vec<i64> = exception_keys
            .iter()
            .copied()
            .filter(|k| !matches!(table.entries.get(k), Some(Expr::Literal(_))))
            .collect();
        if !inline_exceptions.is_empty() {
            let mut compiled_entries: Vec<(i64, Vec<Op>, Slot)> = Vec::new();
            for &key_val in &inline_exceptions {
                if let Some(entry_expr) = table.entries.get(&key_val) {
                    let mut entry_ops = Vec::new();
                    let result = self.compile_expr(entry_expr, &mut entry_ops);
                    compiled_entries.push((key_val, entry_ops, result));
                }
            }
            // Try to recognise a "windowed byte array" shape across the inline
            // exceptions: a contiguous run of keys whose entries all decode as
            // `--func((u16 cell) * stride + offset)` with offset = key - window_base
            // and the same (func, cell, stride). On success the descriptor is
            // stashed on self for `CompiledProgram` to pick up later. Runs
            // AFTER per-entry compilation so the inner function (e.g.
            // `--readDiskByte`) is in `flat_dispatch_cache`. The inline
            // branches are emitted regardless — the runtime State short-circuit
            // catches reads first, but the inline path is the correct fallback
            // for any cabinet/future shape that doesn't fit the recogniser.
            if self.recognised_windowed_byte_array.is_none() && name == "--readMem" {
                self.recognised_windowed_byte_array = recognise_windowed_byte_array(
                    &table,
                    &inline_exceptions,
                    &self.flat_dispatch_cache,
                    &self.compiled_flat_arrays,
                );
            }
            let result_slot = self.alloc();
            ops.push(Op::LoadSlot { dst: result_slot, src: dst });

            let mut jumps_to_end: Vec<usize> = Vec::new();
            for (key_val, entry_ops, entry_result) in &compiled_entries {
                let lit = self.alloc();
                ops.push(Op::LoadLit { dst: lit, val: *key_val as i32 });
                let cmp = self.alloc();
                ops.push(Op::CmpEq { dst: cmp, a: key_slot, b: lit });
                let branch_idx = ops.len();
                ops.push(Op::BranchIfZero { cond: cmp, target: 0 });

                ops.extend_from_slice(entry_ops);
                ops.push(Op::LoadSlot { dst: result_slot, src: *entry_result });
                jumps_to_end.push(ops.len());
                ops.push(Op::Jump { target: 0 });

                let next_idx = ops.len() as u32;
                if let Op::BranchIfZero { target, .. } = &mut ops[branch_idx] {
                    *target = next_idx;
                }
            }
            let end_idx = ops.len() as u32;
            for idx in jumps_to_end {
                if let Op::Jump { target } = &mut ops[idx] {
                    *target = end_idx;
                }
            }

            self.dispatch_tables.insert(name.to_string(), table);
            for (param_name, old) in saved {
                match old {
                    Some(s) => { self.property_slots.insert(param_name, s); }
                    None => { self.property_slots.remove(&param_name); }
                }
            }
            return result_slot;
        }

        self.dispatch_tables.insert(name.to_string(), table);
        for (param_name, old) in saved {
            match old {
                Some(s) => { self.property_slots.insert(param_name, s); }
                None => { self.property_slots.remove(&param_name); }
            }
        }
        dst
    }

    /// Maximum span (max_key - min_key + 1) permitted for the flat-array
    /// dispatch fast path. Chosen so that each flat array stays under
    /// ~40 MB (10M * 4 bytes) even in the worst case.
    const FLAT_DISPATCH_MAX_SPAN: i64 = 10_000_000;

    /// Try to build a flat-array dispatch representation of `table`.
    ///
    /// Returns `Some((array_id, base_key, default))` if every entry is a
    /// literal integer representable as i32, the fallback is a literal
    /// representable as i32, and the key range is dense enough. The
    /// resulting array is stored in `self.compiled_flat_arrays`, indexed
    /// by (key - base_key).
    ///
    /// Parameterless calls are cached — subsequent calls to the same
    /// dispatch table reuse the same `array_id`.
    fn try_build_flat_dispatch(
        &mut self,
        name: &str,
        table: &DispatchTable,
    ) -> Option<(u32, i32, i32)> {
        // Fallback must be a literal.
        let default = match &table.fallback {
            Expr::Literal(v) => lit_as_i32(*v)?,
            _ => return None,
        };

        // Every entry must be a literal integer fitting in i32, and the
        // key must fit in i32.
        let mut min_key: i64 = i64::MAX;
        let mut max_key: i64 = i64::MIN;
        for (&k, expr) in &table.entries {
            if k < i32::MIN as i64 || k > i32::MAX as i64 {
                return None;
            }
            match expr {
                Expr::Literal(v) => {
                    if lit_as_i32(*v).is_none() {
                        return None;
                    }
                }
                _ => return None,
            }
            if k < min_key { min_key = k; }
            if k > max_key { max_key = k; }
        }

        let span = max_key - min_key + 1;
        if span <= 0 || span > Self::FLAT_DISPATCH_MAX_SPAN {
            return None;
        }

        // Reuse cached result by function name. The flat array depends only
        // on the dispatch table entries (all literals), not on the runtime
        // parameter value — so caching by name is safe even for parameterized
        // dispatches. At runtime the key slot varies per call site; the array
        // is shared.
        if let Some(&cached) = self.flat_dispatch_cache.get(name) {
            return Some(cached);
        }

        let base_key = min_key as i32;
        let mut values = vec![default; span as usize];
        for (&k, expr) in &table.entries {
            if let Expr::Literal(v) = expr {
                // Safe: bounds and conversion validated above.
                let idx = (k - min_key) as usize;
                values[idx] = lit_as_i32(*v).unwrap();
            }
        }

        let array_id = self.compiled_flat_arrays.len() as u32;
        self.compiled_flat_arrays.push(FlatDispatchArray { values: std::sync::Arc::new(values) });
        let result = (array_id, base_key, default);
        self.flat_dispatch_cache.insert(name.to_string(), result);
        Some(result)
    }

    /// Compile a dispatch table function call.
    ///
    /// Dispatch tables are compiled once per program and reused via `Op::Call`.
    /// The first call compiles the table's entries against reserved parameter
    /// slots into a self-contained body that ends with an `Op::Dispatch` (or
    /// flat-array fast path). Subsequent calls emit only `Op::Call`.
    fn compile_dispatch_call(&mut self, name: &str, args: &[Expr], ops: &mut Vec<Op>) -> Slot {
        prof!("compile_dispatch_call");
        let func = self.functions.get(name).map(|f| f as *const FunctionDef);
        // SAFETY: self.functions is &'a external; never mutated during compile.
        let func = func.map(|p| unsafe { &*p });

        // Cache hit: emit a single Op::Call.
        if let Some(&fn_id) = self.function_cache.get(name) {
            let mut arg_slots: Vec<Slot> = Vec::with_capacity(args.len());
            for a in args {
                arg_slots.push(self.compile_expr(a, ops));
            }
            let param_count = func.map_or(0, |f| f.parameters.len());
            while arg_slots.len() < param_count {
                let s = self.alloc();
                ops.push(Op::LoadLit { dst: s, val: 0 });
                arg_slots.push(s);
            }
            arg_slots.truncate(param_count);
            let dst = self.alloc();
            ops.push(Op::Call { dst, fn_id, arg_slots });
            return dst;
        }

        // First call: compile the dispatch into a cached body.
        self.compile_dispatch_body_then_call(name, args, ops)
    }

    /// Compile a dispatch table function's body once (reserved params + dispatch op),
    /// cache it under a fresh `fn_id`, then emit `Op::Call` at the current call site.
    fn compile_dispatch_body_then_call(
        &mut self,
        name: &str,
        args: &[Expr],
        ops: &mut Vec<Op>,
    ) -> Slot {
        prof!("compile_dispatch_body_then_call");
        let _dt = web_time::Instant::now();
        let table = self.dispatch_tables.remove(name).unwrap();
        let func = self.functions.get(name);

        // Reserve parameter slots for the body. Op::Call writes these per call.
        let param_slots: Vec<Slot> = func
            .as_ref()
            .map(|f| f.parameters.iter().map(|_| self.alloc()).collect())
            .unwrap_or_default();

        // Bind param/local names to reserved slots while we compile the body.
        let saved: Vec<(String, Option<Slot>)> = if let Some(ref f) = func {
            let mut saved: Vec<(String, Option<Slot>)> = f.parameters
                .iter()
                .zip(&param_slots)
                .map(|(param, &pslot)| {
                    let old = self.property_slots.get(&param.name).copied();
                    self.property_slots.insert(param.name.clone(), pslot);
                    (param.name.clone(), old)
                })
                .collect();
            saved
        } else {
            Vec::new()
        };

        // Build the body ops into a fresh Vec. Locals, key lookup, entry
        // compilation, and the final Dispatch/DispatchFlatArray op all go
        // into body_ops — reads of the parameters resolve to the reserved
        // param slots which Op::Call will write at runtime.
        let mut body_ops: Vec<Op> = Vec::new();

        // Locals.
        let mut saved = saved;
        if let Some(f) = func {
            for local in &f.locals {
                let old = self.property_slots.get(&local.name).copied();
                let val_slot = self.compile_expr(&local.value, &mut body_ops);
                self.property_slots.insert(local.name.clone(), val_slot);
                saved.push((local.name.clone(), old));
            }
        }

        // Compile the key lookup into the body.
        let key_slot = self.compile_var(&table.key_property, None, &mut body_ops);

        // ---- Flat-byte-array fast path ----
        let single_param = func.map_or(true, |f| f.parameters.len() <= 1);
        let flat = if single_param && !table.entries.is_empty() {
            self.try_build_flat_dispatch(name, &table)
        } else {
            None
        };
        let result_slot = if let Some((array_id, base_key, default)) = flat {
            let dst = self.alloc();
            body_ops.push(Op::DispatchFlatArray {
                dst,
                key: key_slot,
                array_id,
                base_key,
                default,
            });
            dst
        } else {
            // Compile each dispatch entry into its own op sequence (body uses
            // reserved param slots, so entries' var() refs resolve correctly).
            let mut compiled_entries: FxMap<i64, (Vec<Op>, Slot)> = FxMap::default();
            for (&key_val, entry_expr) in &table.entries {
                let mut entry_ops = Vec::new();
                let result = self.compile_expr(entry_expr, &mut entry_ops);
                compiled_entries.insert(key_val, (entry_ops, result));
            }
            let mut fallback_ops = Vec::new();
            let fallback_slot = self.compile_expr(&table.fallback, &mut fallback_ops);

            let table_id = self.compiled_dispatches.len() as Slot;
            self.compiled_dispatches.push(CompiledDispatchTable {
                entries: compiled_entries,
                fallback_ops,
                fallback_slot,
            });

            if table.entries.len() >= 100 {
                log::info!("[compile detail] dispatch_call {} compiled: {} entries, {:.2}s",
                    name, table.entries.len(), _dt.elapsed().as_secs_f64());
            }

            let dst = self.alloc();
            body_ops.push(Op::Dispatch {
                dst,
                key: key_slot,
                table_id,
                fallback_target: 0,
            });
            dst
        };

        // Restore property bindings.
        self.dispatch_tables.insert(name.to_string(), table);
        for (param_name, old) in saved {
            match old {
                Some(s) => { self.property_slots.insert(param_name, s); }
                None => { self.property_slots.remove(&param_name); }
            }
        }

        // Register the compiled function and cache under fn_id.
        let fn_id = self.compiled_functions.len() as u32;
        self.compiled_functions.push(CompiledFunction {
            param_slots,
            body_ops,
            result_slot,
        });
        self.function_cache.insert(name.to_string(), fn_id);

        // Emit Op::Call at this call site.
        let param_count = func.map_or(0, |f| f.parameters.len());
        let mut arg_slots: Vec<Slot> = Vec::with_capacity(args.len());
        for a in args {
            arg_slots.push(self.compile_expr(a, ops));
        }
        while arg_slots.len() < param_count {
            let s = self.alloc();
            ops.push(Op::LoadLit { dst: s, val: 0 });
            arg_slots.push(s);
        }
        arg_slots.truncate(param_count);

        let dst = self.alloc();
        ops.push(Op::Call { dst, fn_id, arg_slots });
        dst
    }

    /// Compile a general function call.
    ///
    /// The function body is compiled **once** per program into an `Op::Call`
    /// target — a self-contained `Vec<Op>` referencing reserved parameter
    /// slots. Every call site emits a single `Op::Call { fn_id, arg_slots,
    /// dst }` that copies args into the reserved slots and runs the cached
    /// body. This avoids the N² recompile cost of the old inlining path.
    fn compile_general_function(&mut self, name: &str, args: &[Expr], ops: &mut Vec<Op>) -> Slot {
        prof!("compile_general_function");
        let func = match self.functions.get(name) {
            Some(f) => f as *const FunctionDef,
            None => {
                let dst = self.alloc();
                ops.push(Op::LoadLit { dst, val: 0 });
                return dst;
            }
        };
        // SAFETY: self.functions is &'a (external borrow). Never mutated during compile.
        let func = unsafe { &*func };

        let fn_id = if let Some(&id) = self.function_cache.get(name) {
            id
        } else {
            self.compile_function_body(name, func)
        };

        // Compile arguments at the call site (they execute in the caller's ops context).
        let mut arg_slots: Vec<Slot> = Vec::with_capacity(args.len());
        for a in args {
            arg_slots.push(self.compile_expr(a, ops));
        }
        // Pad missing arguments with zero literals so Op::Call can count on a
        // full vector of arg slots matching `param_slots`.
        while arg_slots.len() < func.parameters.len() {
            let s = self.alloc();
            ops.push(Op::LoadLit { dst: s, val: 0 });
            arg_slots.push(s);
        }
        arg_slots.truncate(func.parameters.len());

        let dst = self.alloc();
        ops.push(Op::Call { dst, fn_id, arg_slots });
        dst
    }

    /// Compile a function body once. Reserves param slots, binds them in
    /// `property_slots`, compiles locals and the result into a standalone
    /// body op stream, and records the result in `compiled_functions`.
    fn compile_function_body(&mut self, name: &str, func: &FunctionDef) -> u32 {
        prof!("compile_function_body");
        // Reserve a slot per parameter. Op::Call writes these before the body runs.
        let param_slots: Vec<Slot> = func.parameters.iter().map(|_| self.alloc()).collect();

        // Save & bind property_slots for parameters and locals so that
        // var() references inside the body resolve to the reserved slots.
        let mut saved: Vec<(String, Option<Slot>)> =
            Vec::with_capacity(func.parameters.len() + func.locals.len());
        for (param, &pslot) in func.parameters.iter().zip(&param_slots) {
            let old = self.property_slots.get(&param.name).copied();
            self.property_slots.insert(param.name.clone(), pslot);
            saved.push((param.name.clone(), old));
        }

        let mut body_ops: Vec<Op> = Vec::new();

        // Compile locals. Their value ops go into body_ops (runs on every call).
        for local in &func.locals {
            let old = self.property_slots.get(&local.name).copied();
            let val_slot = self.compile_expr(&local.value, &mut body_ops);
            self.property_slots.insert(local.name.clone(), val_slot);
            saved.push((local.name.clone(), old));
        }

        // Compile the result expression into the body.
        let result_slot = self.compile_expr(&func.result, &mut body_ops);

        // Restore caller's property bindings.
        for (name_s, old) in saved {
            match old {
                Some(s) => {
                    self.property_slots.insert(name_s, s);
                }
                None => {
                    self.property_slots.remove(&name_s);
                }
            }
        }

        let fn_id = self.compiled_functions.len() as u32;
        self.compiled_functions.push(CompiledFunction {
            param_slots,
            body_ops,
            result_slot,
        });
        self.function_cache.insert(name.to_string(), fn_id);
        fn_id
    }
}

/// If `test` is a (possibly nested) `And` of `Single { value: integer literal }` tests,
/// return a flat list of (property, literal) pairs. Otherwise `None`.
///
/// Used by `compile_style_condition_linear` to emit direct BranchIfNotEqLit chains
/// in place of the generic compile_style_test machinery (which emits Mul/CmpEq/branch
/// per term and never fuses). Covers the dominant BIOS pattern:
/// `style(--opcode: N) and style(--uOp: K)`.
fn collect_literal_and_tests(test: &StyleTest) -> Option<Vec<(String, i32)>> {
    let mut out: Vec<(String, i32)> = Vec::new();
    fn walk(t: &StyleTest, out: &mut Vec<(String, i32)>) -> bool {
        match t {
            StyleTest::Single { property, value } => {
                if let Expr::Literal(v) = value {
                    if v.fract() == 0.0 && *v >= i32::MIN as f64 && *v <= i32::MAX as f64 {
                        out.push((property.clone(), *v as i32));
                        return true;
                    }
                }
                false
            }
            StyleTest::And(tests) => tests.iter().all(|t| walk(t, out)),
            StyleTest::Or(_) => false,
        }
    }
    if walk(test, &mut out) && !out.is_empty() {
        Some(out)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Public API — compile an evaluator's data into a CompiledProgram
// ---------------------------------------------------------------------------

/// Compile the evaluator's assignments and broadcast writes into a `CompiledProgram`.
pub fn render_progress(phase: &str, done: usize, total: usize, elapsed: f64) {
    let pct = if total > 0 { (done as f64 / total as f64 * 100.0).min(100.0) } else { 0.0 };
    let width = 30usize;
    let filled = ((pct / 100.0) * width as f64) as usize;
    let bar: String = (0..width)
        .map(|i| if i < filled { '=' } else if i == filled { '>' } else { ' ' })
        .collect();
    eprint!(
        "\r{phase} [{bar}] {pct:5.1}%  {done}/{total}  {elapsed:.1}s                    ",
        phase = phase, bar = bar, pct = pct, done = done, total = total, elapsed = elapsed
    );
    let _ = std::io::Write::flush(&mut std::io::stderr());
}

pub fn compile(
    assignments: &[Assignment],
    broadcast_writes: &[BroadcastWrite],
    packed_broadcast_ports: &[crate::pattern::packed_broadcast_write::PackedSlotPort],
    functions: &HashMap<String, FunctionDef>,
    dispatch_tables: &mut HashMap<String, DispatchTable>,
) -> CompiledProgram {
    profile_compile_init();
    prof!("compile");
    let _ct = web_time::Instant::now();
    let mut compiler = Compiler::new(functions, dispatch_tables);
    log::info!("[compile detail] Compiler::new: {:.2}s", _ct.elapsed().as_secs_f64());
    crate::compile_stats::record("compile.compiler_new", _ct.elapsed().as_secs_f64());
    let mut ops = Vec::new();
    let mut writeback = Vec::new();

    // ---- Progress meter ----
    let bw_total: usize = broadcast_writes.iter().map(|bw| bw.address_map.len()).sum();
    let total_work = assignments.len() + bw_total;
    let show_progress = std::env::var_os("CALCITE_NO_PROGRESS").is_none()
        && total_work >= 10_000;
    let progress_start = web_time::Instant::now();
    let mut last_render = web_time::Instant::now();
    let mut work_done: usize = 0;

    let _ct = web_time::Instant::now();
    // Compile each assignment.
    for assignment in assignments.iter() {
        let _at = web_time::Instant::now();
        let result_slot = compiler.compile_expr(&assignment.value, &mut ops);
        let elapsed = _at.elapsed();
        if elapsed.as_millis() >= 100 {
            log::info!("[compile detail] assignment {} took {:.2}s", assignment.property, elapsed.as_secs_f64());
        }
        // Register this property slot so later assignments can reference it
        compiler
            .property_slots
            .insert(assignment.property.clone(), result_slot);

        // Track writeback for canonical properties
        if !is_buffer_copy(&assignment.property) {
            if let Some(addr) = property_to_address(&assignment.property) {
                writeback.push((result_slot, addr));
            }
        }

        work_done += 1;
        if show_progress && last_render.elapsed().as_millis() >= 100 {
            render_progress("Compiling", work_done, total_work, progress_start.elapsed().as_secs_f64());
            last_render = web_time::Instant::now();
        }
    }
    log::info!("[compile detail] assignments ({} items): {:.2}s, {} ops, {} dispatch tables",
        assignments.len(), _ct.elapsed().as_secs_f64(), ops.len(), compiler.compiled_dispatches.len());
    crate::compile_stats::record("compile.assignments", _ct.elapsed().as_secs_f64());


    let _ct = web_time::Instant::now();
    // Compile broadcast writes
    let compiled_bw: Vec<_> = broadcast_writes
        .iter()
        .map(|bw| {
            let _bt = web_time::Instant::now();
            let result = compile_broadcast_write(bw, &mut compiler);
            log::info!("[compile detail] broadcast write {}: {:.2}s ({} addrs, {} spillover)",
                bw.dest_property, _bt.elapsed().as_secs_f64(),
                bw.address_map.len(), bw.spillover_map.len());
            work_done += bw.address_map.len();
            if show_progress && last_render.elapsed().as_millis() >= 100 {
                render_progress("Compiling", work_done, total_work, progress_start.elapsed().as_secs_f64());
                last_render = web_time::Instant::now();
            }
            result
        })
        .collect();
    if show_progress {
        render_progress("Compiling", total_work, total_work, progress_start.elapsed().as_secs_f64());
        eprintln!();
    }
    log::info!("[compile detail] broadcast writes total ({} items): {:.2}s",
        broadcast_writes.len(), _ct.elapsed().as_secs_f64());
    crate::compile_stats::record("compile.broadcast_writes", _ct.elapsed().as_secs_f64());

    // Compile packed-broadcast write ports. Each port becomes a
    // CompiledPackedBroadcastWrite whose runtime arm reads gate/addr/val,
    // looks up the cell's state address in cell_table, splices the byte,
    // and writes back. property_slots was populated above by the assignment
    // loop, so the lookup for `gate`/`addr`/`val` slots resolves directly
    // (these properties are always registered by the assignment loop because
    // the recogniser only fires when their assignments exist).
    let _ct = web_time::Instant::now();
    let mut compiled_packed_bw: Vec<CompiledPackedBroadcastWrite> =
        Vec::with_capacity(packed_broadcast_ports.len());
    for port in packed_broadcast_ports {
        // Resolve slots. If any is missing, skip the port — better to fall
        // back to the non-absorbed `--mc{N}` evaluation than to silently
        // splice from slot 0.
        let (Some(&gate_slot), Some(&addr_slot), Some(&val_slot), Some(&width_slot)) = (
            compiler.property_slots.get(&port.gate_property),
            compiler.property_slots.get(&port.addr_property),
            compiler.property_slots.get(&port.val_property),
            compiler.property_slots.get(&port.width_property),
        ) else {
            log::warn!(
                "Packed broadcast port {} {} {} {}: missing slot binding, skipping",
                port.gate_property, port.addr_property, port.val_property, port.width_property
            );
            continue;
        };
        // Build the dense cell_table indexed by cell index (byte_addr / pack).
        let pack = port.pack as usize;
        let max_cell_idx: usize = port
            .address_map
            .keys()
            .map(|&byte_addr| (byte_addr as usize) / pack)
            .max()
            .unwrap_or(0);
        let mut cell_table: Vec<i32> = vec![0; max_cell_idx + 1];
        for (&byte_addr, var_name) in &port.address_map {
            let idx = (byte_addr as usize) / pack;
            if let Some(state_addr) = property_to_address(var_name) {
                cell_table[idx] = state_addr;
            }
        }
        compiled_packed_bw.push(CompiledPackedBroadcastWrite {
            gate_slot,
            addr_slot,
            val_slot,
            width_slot,
            pack: port.pack,
            cell_table,
        });
    }
    log::info!(
        "[compile detail] packed broadcast writes ({} ports): {:.2}s",
        compiled_packed_bw.len(),
        _ct.elapsed().as_secs_f64()
    );

    let mut program = CompiledProgram {
        ops,
        slot_count: compiler.next_slot,
        writeback,
        broadcast_writes: compiled_bw,
        dispatch_tables: compiler.compiled_dispatches,
        chain_tables: Vec::new(),
        flat_dispatch_arrays: compiler.compiled_flat_arrays,
        functions: compiler.compiled_functions,
        property_slots: compiler.property_slots,
        packed_cell_tables: compiler.packed_cell_tables,
        packed_exception_tables: compiler.packed_exception_tables,
        packed_broadcast_writes: compiled_packed_bw,
        windowed_byte_array: compiler.recognised_windowed_byte_array.take(),
        loop_descriptors: Vec::new(),
    };

    // Expand all Op::Call sites inline. Op::Call was a compile-time device to
    // avoid O(N) per-call-site recompilation; at runtime we want the flat
    // linear op stream the evaluator is optimized for. After this pass,
    // program.ops (and all dispatch entries, broadcast value_ops) contain no
    // Op::Call ops — the bodies are inlined.
    let _ct = web_time::Instant::now();
    let inlined = inline_calls(&mut program);
    log::info!("[compile detail] inline calls: {} sites inlined, {:.2}s",
        inlined, _ct.elapsed().as_secs_f64());
    crate::compile_stats::record("compile.inline_calls", _ct.elapsed().as_secs_f64());

    // Copy propagation + DCE. Disable with CALCITE_NO_COPY_ELIM=1 (native
    // only; wasm has no env) for A/B measurement.
    let copy_elim_enabled = {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::env::var_os("CALCITE_NO_COPY_ELIM").is_none()
        }
        #[cfg(target_arch = "wasm32")]
        {
            true
        }
    };
    let _ct = web_time::Instant::now();
    let (copies_propagated, copy_ops_removed) = if copy_elim_enabled {
        copy_propagate_and_dce(&mut program)
    } else {
        (0, 0)
    };
    log::info!("[compile detail] copy-elim: {} reads redirected, {} ops removed, {:.2}s",
        copies_propagated, copy_ops_removed, _ct.elapsed().as_secs_f64());
    crate::compile_stats::record("compile.copy_elim", _ct.elapsed().as_secs_f64());

    let _ct = web_time::Instant::now();
    let fused = fuse_cmp_branch(&mut program);
    log::info!("[compile detail] fuse_cmp_branch: {} fused, {:.2}s", fused, _ct.elapsed().as_secs_f64());
    crate::compile_stats::record("compile.fuse_cmp_branch", _ct.elapsed().as_secs_f64());

    let _ct = web_time::Instant::now();
    let chains = build_dispatch_chains(&mut program);
    log::info!("[compile detail] dispatch chains: {} chains built, {:.2}s", chains, _ct.elapsed().as_secs_f64());
    crate::compile_stats::record("compile.dispatch_chains", _ct.elapsed().as_secs_f64());

    let _ct = web_time::Instant::now();
    let lsfused = fuse_loadstate_branch(&mut program);
    log::info!("[compile detail] fuse_loadstate_branch: {} fused, {:.2}s", lsfused, _ct.elapsed().as_secs_f64());
    crate::compile_stats::record("compile.fuse_loadstate", _ct.elapsed().as_secs_f64());

    // BIfNEL2 fusion: HARDCODED OFF for current bench measurement
    // (2026-05-08, isolating the keyboard-revert win from the BIF2 win).
    // To re-enable, flip BIF2_FUSE_ENABLED. The pass itself is intact.
    const BIF2_FUSE_ENABLED: bool = false;
    let _ct = web_time::Instant::now();
    let bif2_fused: usize = if BIF2_FUSE_ENABLED {
        fuse_diff_slot_bifnel_pairs(&mut program)
    } else {
        0
    };
    log::info!("[compile detail] fuse_diff_slot_bifnel_pairs: {} fused, {:.2}s", bif2_fused, _ct.elapsed().as_secs_f64());

    let _ct = web_time::Instant::now();
    compact_slots(&mut program);
    log::info!("[compile detail] slot compaction: {:.2}s", _ct.elapsed().as_secs_f64());
    crate::compile_stats::record("compile.slot_compaction", _ct.elapsed().as_secs_f64());

    // Replicated-body recognition: fold long unrolled straight-line regions
    // (e.g. emitter-unrolled column drawers) into a single Op::ReplicatedBody.
    // Must run AFTER all peephole passes and slot compaction so the slot-
    // walking utility functions never encounter the new variant.
    let _ct = web_time::Instant::now();
    let folded = recognise_replicated_bodies(&mut program);
    log::info!("[compile detail] replicated_body: {} regions folded, {:.2}s",
        folded, _ct.elapsed().as_secs_f64());
    crate::compile_stats::record("compile.replicated_body", _ct.elapsed().as_secs_f64());

    profile_compile_dump();
    program
}

// ---------------------------------------------------------------------------
// Op::Call inlining — expand every Call site back into its body_ops.
// ---------------------------------------------------------------------------
//
// The Op::Call mechanism is a compile-time device: it lets us compile each
// function body exactly once instead of once per call site. But at runtime,
// a Call op forces a nested exec_ops() recursion, which is ~40x slower than
// running the body inline in the main linear op stream. So we expand every
// Call back into its body after compilation. Bodies are inlined with their
// original slot numbers — since execution is strictly sequential, sharing
// the same body-internal slots across call sites is safe (same invariant
// as the existing dispatch-entry slot overlay).
//
// Body ops may contain branches with absolute PC targets (Jump,
// BranchIfNotEqLit, BranchIfZero, LoadStateAndBranchIfNotEqLit,
// DispatchChain, etc.). When inlined at offset `base`, every target must
// be shifted by `base`. Dispatch sub-op entries have their own PC scope
// and don't need shifting.
//
// Bodies are inlined in topological order: leaves (no nested Calls) first.
// We fully inline each CompiledFunction.body_ops once, then use those
// flattened bodies when expanding Call sites in the main ops stream and
// elsewhere. This ensures a single-pass expansion of call sites.

fn shift_branch_targets(ops: &mut [Op], offset: u32) {
    for op in ops {
        match op {
            Op::Jump { target } => *target += offset,
            Op::BranchIfZero { target, .. } => *target += offset,
            Op::BranchIfNotEqLit { target, .. } => *target += offset,
            Op::LoadStateAndBranchIfNotEqLit { target, .. } => *target += offset,
            Op::BranchIfNotEqLit2 { target, .. } => *target += offset,
            Op::DispatchChain { miss_target, .. } => *miss_target += offset,
            // Dispatch op's fallback_target is unused (sub-ops live in dispatch tables).
            _ => {}
        }
    }
}

/// Expand a single stream of ops: every `Op::Call { fn_id, arg_slots, dst }`
/// is replaced with the function's already-inlined body_ops plus the arg→param
/// copy and result→dst copy. Returns the number of call sites expanded.
///
/// Branch targets in the *containing* stream (already-copied ops in `out`, and
/// ops still to come in `src`) must be shifted to account for inserted ops.
/// We build `out` by iterating `src` with an index map: old index N → new index
/// `shift_table[N]`. After construction, any branch op's target `T` (which
/// references an old index) is rewritten to `shift_table[T]`.
fn expand_calls_in_stream(ops: &mut Vec<Op>, flat_bodies: &[Vec<Op>], param_slots: &[Vec<Slot>], result_slots: &[Slot]) -> usize {
    let has_any = ops.iter().any(|o| matches!(o, Op::Call { .. }));
    if !has_any { return 0; }

    let src = std::mem::take(ops);
    let src_len = src.len();
    let mut out: Vec<Op> = Vec::with_capacity(src_len * 2);
    // shift_table[i] = new index of old instruction i.
    // An extra trailing entry (shift_table[src_len] = out.len() at end) lets
    // targets that pointed past-the-end (to "end of stream") remap correctly.
    let mut shift_table: Vec<u32> = Vec::with_capacity(src_len + 1);
    let mut expanded = 0usize;

    for op in src {
        shift_table.push(out.len() as u32);
        match op {
            Op::Call { dst, fn_id, arg_slots } => {
                let fid = fn_id as usize;
                let params = &param_slots[fid];
                let body = &flat_bodies[fid];
                let result = result_slots[fid];

                // arg → reserved param slot
                for (arg, &param) in arg_slots.iter().zip(params.iter()) {
                    out.push(Op::LoadSlot { dst: param, src: *arg });
                }

                // Inline body. Body targets are absolute within the body;
                // shift them by the current `out.len()` at body start.
                let base = out.len() as u32;
                let start = out.len();
                out.extend(body.iter().cloned());
                shift_branch_targets(&mut out[start..], base);

                // result → dst
                out.push(Op::LoadSlot { dst, src: result });
                expanded += 1;
            }
            other => out.push(other),
        }
    }
    // Sentinel for targets that pointed at src_len (end of stream).
    shift_table.push(out.len() as u32);

    // Rewrite branch targets in out[] using shift_table. Only rewrite targets
    // that correspond to original (non-body-inlined) ops. Body-inlined ops
    // already had their internal targets shifted by `base` above and don't
    // reference outer indices; they have different target ranges, and none
    // of them points into shift_table. We distinguish by target value: body
    // targets (shifted by `base`) are always < final out.len() and ≥ the
    // base at which they were inlined — but that's hard to track after the
    // fact.
    //
    // Simpler: rebuild out with a second pass that only rewrites targets of
    // ops that came from `src` (non-inlined). We track each op's origin via
    // a parallel `is_outer` vector.
    //
    // Rebuild approach. We redo the loop but now populate is_outer alongside.
    // This mirrors the body-shift pass above.
    //
    // (We do the rebuild here rather than inline in the first loop to keep
    // the control flow straightforward.)
    let mut is_outer = vec![false; out.len()];
    {
        // Re-run the logical layout to figure out which indices correspond to
        // original src ops. At entry, shift_table[i] = start position of op i
        // in out. If op i was not a Call, exactly one op lives at that position.
        // If it was a Call, several ops live starting there; none are "outer".
        //
        // Our rule: out[shift_table[i]] is "outer" iff src[i] was not a Call.
        // To avoid re-iterating src (which we consumed), we infer from the
        // sequence of shift_table deltas: size=1 means non-call, size>1 means call.
        // A call expansion size is `arg_count + body_len + 1` (LoadSlot dst).
        // We just check shift_table[i+1] - shift_table[i] == 1.
        for i in 0..src_len {
            let delta = shift_table[i + 1] - shift_table[i];
            if delta == 1 {
                is_outer[shift_table[i] as usize] = true;
            }
        }
    }

    for idx in 0..out.len() {
        if !is_outer[idx] { continue; }
        match &mut out[idx] {
            Op::Jump { target }
            | Op::BranchIfZero { target, .. }
            | Op::BranchIfNotEqLit { target, .. }
            | Op::LoadStateAndBranchIfNotEqLit { target, .. }
            | Op::BranchIfNotEqLit2 { target, .. }
            | Op::DispatchChain { miss_target: target, .. } => {
                let old = *target as usize;
                if old <= src_len {
                    *target = shift_table[old];
                }
            }
            _ => {}
        }
    }

    *ops = out;
    expanded
}

/// Inline every `Op::Call` in the program. Processes CompiledFunction bodies in
/// topological order (leaves first), then expands calls in the main ops stream,
/// dispatch entry/fallback ops, and broadcast value/spillover ops.
fn inline_calls(program: &mut CompiledProgram) -> usize {
    let n = program.functions.len();
    if n == 0 { return 0; }

    // Compute each function's immediate Call callees.
    let mut callees: Vec<Vec<u32>> = Vec::with_capacity(n);
    for func in &program.functions {
        let mut c = Vec::new();
        for op in &func.body_ops {
            if let Op::Call { fn_id, .. } = op {
                c.push(*fn_id);
            }
        }
        callees.push(c);
    }

    // Topological sort — leaves (no callees) first. Cycles would mean recursion,
    // which the compile-path doesn't produce; if we detect one, break arbitrarily.
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut inlined_set = vec![false; n];
    loop {
        let mut progressed = false;
        for i in 0..n {
            if inlined_set[i] { continue; }
            if callees[i].iter().all(|&c| inlined_set[c as usize]) {
                order.push(i as u32);
                inlined_set[i] = true;
                progressed = true;
            }
        }
        if !progressed { break; }
    }
    // Any leftover (cycle) — append in index order; we'll inline them without
    // recursion support.
    for i in 0..n {
        if !inlined_set[i] { order.push(i as u32); }
    }

    // Take out bodies so we can mutate them while keeping &program.functions layout.
    let mut bodies: Vec<Vec<Op>> = program.functions.iter_mut()
        .map(|f| std::mem::take(&mut f.body_ops))
        .collect();
    let param_slots: Vec<Vec<Slot>> = program.functions.iter()
        .map(|f| f.param_slots.clone())
        .collect();
    let result_slots: Vec<Slot> = program.functions.iter()
        .map(|f| f.result_slot)
        .collect();

    // Fully inline each function's body in topo order. After inlining bodies[i],
    // it contains no Op::Call ops, and any later function that calls i can
    // freely inline bodies[i].
    let mut total = 0usize;
    for &i in &order {
        let idx = i as usize;
        let mut body = std::mem::take(&mut bodies[idx]);
        total += expand_calls_in_stream(&mut body, &bodies, &param_slots, &result_slots);
        bodies[idx] = body;
    }

    // Now inline calls in the main ops stream.
    total += expand_calls_in_stream(&mut program.ops, &bodies, &param_slots, &result_slots);

    // Dispatch tables: entry ops and fallback ops.
    for table in &mut program.dispatch_tables {
        for (_key, (entry_ops, _slot)) in &mut table.entries {
            total += expand_calls_in_stream(entry_ops, &bodies, &param_slots, &result_slots);
        }
        total += expand_calls_in_stream(&mut table.fallback_ops, &bodies, &param_slots, &result_slots);
    }

    // Broadcast writes: value_ops and spillover entry ops.
    for bw in &mut program.broadcast_writes {
        total += expand_calls_in_stream(&mut bw.value_ops, &bodies, &param_slots, &result_slots);
        if let Some(ref mut spillover) = bw.spillover {
            for (_key, (spill_ops, _slot)) in &mut spillover.entries {
                total += expand_calls_in_stream(spill_ops, &bodies, &param_slots, &result_slots);
            }
        }
    }

    // Bodies are no longer needed at runtime since all Call sites are inlined.
    // Clear them to free memory; leave the CompiledFunction structs in place
    // (harmless, and the executor's Op::Call arm won't be invoked).
    program.functions.clear();
    total
}

// ---------------------------------------------------------------------------
// Copy propagation + dead-code elimination
// ---------------------------------------------------------------------------
//
// Call inlining materialises one `LoadSlot { param, arg }` per argument at
// every call site plus one `LoadSlot { dst, result }` per call, and dispatch
// entry compilation adds one result copy per entry. At runtime these are pure
// data movement: measured on a CSS-DOS cabinet, LoadSlot alone is ~28% of all
// dispatched ops and LoadLit (mostly literal call arguments) another ~8%.
//
// Two classic passes, run per op stream:
//
//  1. Forward copy propagation — while the fact `slots[dst] == slots[src]`
//     provably holds, rewrite reads of `dst` to read `src` directly. Facts
//     merge by intersection at branch targets; a single linear pass suffices
//     because these streams are forward-only CFGs (every branch target is a
//     higher index), so index order is topological order.
//  2. Backward liveness DCE — remove side-effect-free ops whose destination
//     is never read afterwards. After propagation this kills the bypassed
//     copies, plus literal/compute chains that only fed them.
//
// Soundness leans on the same invariants `compact_slots` already depends on:
//
//  - Non-external slots are tick-scratch: no stream reads a slot expecting a
//    value from a previous tick (slot-compaction's free-list reuse would
//    already corrupt such a pattern).
//  - Cross-stream slot communication happens only through declared channels:
//    dispatch entries read containing-scope slots (modelled by each table's
//    transitive read-set, applied at `Dispatch` ops) and export exactly one
//    result slot; broadcast/spillover sub-ops read main-scope slots (their
//    reads are added to the main stream's external set); writeback,
//    property_slots and packed ports read main slots at end-of-tick
//    (`collect_main_pinned`).
//  - Because pre-compaction slot numbers of inlined bodies are shared across
//    streams, a `Dispatch` may write slots aliasing the containing stream's
//    temporaries — so propagation drops ALL facts at a `Dispatch`.
//
// Streams containing a backward branch edge, or op variants that only exist
// after later passes (DispatchChain, BranchIfNotEqLit2, fused branches,
// ReplicatedBody) or `Op::Call`, are skipped untouched.
//
// Must run after `inline_calls` and before `build_dispatch_chains` / the
// fuse passes: no chain tables exist yet, so removing ops invalidates no
// chain-table body PCs, and the cleaned streams give the fuse passes more
// adjacent triplets to work with.

/// Copy-propagation facts: `map[dst] = root` means `slots[dst] == slots[root]`
/// here. Values are always roots (never themselves keys), maintained by
/// `add`/`kill`. `dependents` is the reverse index used to invalidate facts
/// when a root is overwritten.
#[derive(Default, Clone)]
struct CopyFacts {
    map: FxMap<Slot, Slot>,
    dependents: FxMap<Slot, Vec<Slot>>,
}

impl CopyFacts {
    fn root(&self, s: Slot) -> Slot {
        self.map.get(&s).copied().unwrap_or(s)
    }
    /// Slot `w` was overwritten: drop the fact about `w` and every fact
    /// rooted at `w`.
    fn kill(&mut self, w: Slot) {
        self.map.remove(&w);
        if let Some(deps) = self.dependents.remove(&w) {
            for d in deps {
                if self.map.get(&d) == Some(&w) {
                    self.map.remove(&d);
                }
            }
        }
    }
    /// Record `slots[dst] == slots[src]`. Caller must `kill(dst)` first and
    /// guarantee `src` is already a root (reads are rewritten before adding).
    fn add(&mut self, dst: Slot, src: Slot) {
        self.map.insert(dst, src);
        self.dependents.entry(src).or_default().push(dst);
    }
    fn clear(&mut self) {
        self.map.clear();
        self.dependents.clear();
    }
    /// Recompute `dependents` from `map` (after wholesale map replacement).
    fn rebuild_deps(&mut self) {
        self.dependents.clear();
        for (&d, &r) in &self.map {
            self.dependents.entry(r).or_default().push(d);
        }
    }
    /// Keep only facts present (with equal root) in `other`.
    fn intersect_map(a: &mut FxMap<Slot, Slot>, b: &FxMap<Slot, Slot>) {
        a.retain(|k, v| b.get(k) == Some(v));
    }
}

/// Reusable buffers for `copy_opt_stream`. A cabinet can have hundreds of
/// thousands of (mostly tiny) dispatch-entry streams; allocating the
/// working buffers per stream dominates the pass cost (especially on the
/// wasm allocator), so the driver allocates once and reuses.
#[derive(Default)]
struct CopyOptScratch {
    is_label: Vec<bool>,
    noop: Vec<bool>,
    dead: Vec<bool>,
    barrier_live: FxSet<Slot>,
    facts: CopyFacts,
    pending: FxMap<usize, FxMap<Slot, Slot>>,
    live: FxSet<Slot>,
    live_at: FxMap<usize, FxSet<Slot>>,
    new_indices: Vec<u32>,
}

/// The real control-flow target of an op at this pipeline stage, if any.
/// `Dispatch.fallback_target` is excluded — it is a placeholder (always 0)
/// that the executor never reads.
fn op_real_target(op: &Op) -> Option<u32> {
    match op {
        Op::Jump { target }
        | Op::BranchIfZero { target, .. }
        | Op::BranchIfNotEqLit { target, .. }
        | Op::LoadStateAndBranchIfNotEqLit { target, .. } => Some(*target),
        Op::MemoryFill { exit_target, .. } | Op::MemoryCopy { exit_target, .. } => {
            Some(*exit_target)
        }
        _ => None,
    }
}

/// Rewrite every pure-read operand of `op` through `facts`. Returns the
/// number of operands redirected. Operands that are both read AND written
/// by the op (MemoryFill/MemoryCopy pointer slots) are left alone — the
/// write-back must keep landing in the original slot.
fn rewrite_op_reads(op: &mut Op, facts: &CopyFacts) -> usize {
    let mut n = 0usize;
    macro_rules! rw {
        ($s:expr) => {{
            let r = facts.root(*$s);
            if r != *$s {
                *$s = r;
                n += 1;
            }
        }};
    }
    match op {
        Op::LoadSlot { src, .. } => rw!(src),
        Op::LoadMem { addr_slot, .. } | Op::LoadMem16 { addr_slot, .. } => rw!(addr_slot),
        Op::LoadPackedByte { key_slot, .. } => rw!(key_slot),
        Op::Add { a, b, .. }
        | Op::Sub { a, b, .. }
        | Op::Mul { a, b, .. }
        | Op::Div { a, b, .. }
        | Op::Mod { a, b, .. }
        | Op::And { a, b, .. }
        | Op::Shr { a, b, .. }
        | Op::Shl { a, b, .. }
        | Op::BitAnd16 { a, b, .. }
        | Op::BitOr16 { a, b, .. }
        | Op::BitXor16 { a, b, .. }
        | Op::CmpEq { a, b, .. } => {
            rw!(a);
            rw!(b);
        }
        Op::AddLit { a, .. }
        | Op::SubLit { a, .. }
        | Op::MulLit { a, .. }
        | Op::AndLit { a, .. }
        | Op::ShrLit { a, .. }
        | Op::ShlLit { a, .. }
        | Op::ModLit { a, .. }
        | Op::BitNot16 { a, .. } => rw!(a),
        Op::Neg { src, .. } | Op::Abs { src, .. } | Op::Sign { src, .. } | Op::Floor { src, .. } => {
            rw!(src)
        }
        Op::Pow { base, exp, .. } => {
            rw!(base);
            rw!(exp);
        }
        Op::Min { args, .. } | Op::Max { args, .. } => {
            for a in args.iter_mut() {
                rw!(a);
            }
        }
        Op::Clamp { min, val, max, .. } => {
            rw!(min);
            rw!(val);
            rw!(max);
        }
        Op::Round { val, interval, .. } => {
            rw!(val);
            rw!(interval);
        }
        Op::Bit { val, idx, .. } => {
            rw!(val);
            rw!(idx);
        }
        Op::BranchIfZero { cond, .. } => rw!(cond),
        Op::BranchIfNotEqLit { a, .. } => rw!(a),
        Op::Dispatch { key, .. } | Op::DispatchFlatArray { key, .. } => rw!(key),
        Op::StoreState { src, .. } => rw!(src),
        Op::StoreMem { addr_slot, src } => {
            rw!(addr_slot);
            rw!(src);
        }
        // val_slot is read-only; dst/src/count are read+written — not rewritable.
        Op::MemoryFill { val_slot, .. } => rw!(val_slot),
        Op::MemoryCopy { .. } => {}
        Op::LoadLit { .. } | Op::LoadState { .. } | Op::Jump { .. }
        | Op::LoadStateAndBranchIfNotEqLit { .. } => {}
        // Excluded by the pre-scan bail in copy_opt_stream.
        Op::BranchIfNotEqLit2 { .. }
        | Op::DispatchChain { .. }
        | Op::Call { .. }
        | Op::ReplicatedBody { .. } => {
            unreachable!("rewrite_op_reads on op excluded by copy_opt_stream pre-scan")
        }
    }
    n
}

/// Is this op free of side effects (removable when its dst is dead)?
/// `Dispatch` is NOT pure: entries may contain StoreState/StoreMem.
fn op_is_pure(op: &Op) -> bool {
    matches!(
        op,
        Op::LoadLit { .. }
            | Op::LoadSlot { .. }
            | Op::LoadState { .. }
            | Op::LoadMem { .. }
            | Op::LoadMem16 { .. }
            | Op::LoadPackedByte { .. }
            | Op::Add { .. }
            | Op::AddLit { .. }
            | Op::Sub { .. }
            | Op::SubLit { .. }
            | Op::Mul { .. }
            | Op::MulLit { .. }
            | Op::Div { .. }
            | Op::Mod { .. }
            | Op::ModLit { .. }
            | Op::Neg { .. }
            | Op::Abs { .. }
            | Op::Sign { .. }
            | Op::Pow { .. }
            | Op::Min { .. }
            | Op::Max { .. }
            | Op::Clamp { .. }
            | Op::Round { .. }
            | Op::Floor { .. }
            | Op::And { .. }
            | Op::AndLit { .. }
            | Op::Shr { .. }
            | Op::ShrLit { .. }
            | Op::Shl { .. }
            | Op::ShlLit { .. }
            | Op::Bit { .. }
            | Op::BitAnd16 { .. }
            | Op::BitOr16 { .. }
            | Op::BitXor16 { .. }
            | Op::BitNot16 { .. }
            | Op::CmpEq { .. }
            | Op::DispatchFlatArray { .. }
    )
}

/// Run copy propagation + DCE on one op stream. `external` is the set of
/// slots read after the stream finishes (live-out). `table_reads[t]` is the
/// transitive read-set of dispatch table `t`. Returns (reads redirected,
/// ops removed).
fn copy_opt_stream(
    ops: &mut Vec<Op>,
    external: &FxSet<Slot>,
    table_reads: &[FxSet<Slot>],
    scratch: &mut CopyOptScratch,
) -> (usize, usize) {
    let len = ops.len();
    if len == 0 {
        return (0, 0);
    }
    let CopyOptScratch {
        is_label,
        noop,
        dead,
        barrier_live,
        facts,
        pending,
        live,
        live_at,
        new_indices,
    } = scratch;

    // --- Pre-scan: bail on shapes this pass doesn't model, mark labels ---
    // `barrier_live` collects the transitive read-sets of every dispatch
    // table referenced from this stream. Those slots are treated as
    // always-live (checked at the removal decision) instead of being
    // injected into the tracked live set — the big tables' read-sets have
    // tens of thousands of slots, and carrying them in `live` makes every
    // per-label snapshot a huge clone. Conservative (a slot read by some
    // dispatch is unremovable even below the last Dispatch op), but those
    // slots are genuine cross-stream parameters anyway.
    is_label.clear();
    is_label.resize(len + 1, false);
    barrier_live.clear();
    for (i, op) in ops.iter().enumerate() {
        match op {
            Op::BranchIfNotEqLit2 { .. }
            | Op::DispatchChain { .. }
            | Op::Call { .. }
            | Op::ReplicatedBody { .. } => return (0, 0),
            Op::Dispatch { table_id, .. } => {
                if let Some(reads) = table_reads.get(*table_id as usize) {
                    barrier_live.extend(reads.iter().copied());
                }
            }
            _ => {}
        }
        if let Some(t) = op_real_target(op) {
            if (t as usize) <= i {
                // Backward (or self) edge — this pass assumes forward-only CFGs.
                return (0, 0);
            }
            if (t as usize) <= len {
                is_label[t as usize] = true;
            }
        }
    }

    // --- Forward copy propagation ---
    facts.clear();
    // Facts snapshot arriving at each label via jumps (intersected per edge).
    pending.clear();
    // Whether the previous op can fall through into the current one.
    let mut ft_reachable = true;
    noop.clear();
    noop.resize(len, false);
    let mut propagated = 0usize;

    for i in 0..len {
        if is_label[i] {
            match (ft_reachable, pending.remove(&i)) {
                (true, Some(p)) => {
                    CopyFacts::intersect_map(&mut facts.map, &p);
                    facts.rebuild_deps();
                }
                (true, None) => {
                    // Label with no recorded jump-in snapshot (shouldn't happen
                    // on forward-only CFGs) — be conservative.
                    facts.clear();
                }
                (false, Some(p)) => {
                    facts.map = p;
                    facts.rebuild_deps();
                }
                (false, None) => facts.clear(), // unreachable code
            }
            ft_reachable = true;
        }

        let op = &mut ops[i];
        propagated += rewrite_op_reads(op, facts);

        // Helper to merge the current facts into a jump target's pending set.
        macro_rules! snapshot_to {
            ($t:expr) => {{
                let t = $t as usize;
                if t <= len {
                    match pending.entry(t) {
                        std::collections::hash_map::Entry::Occupied(mut e) => {
                            CopyFacts::intersect_map(e.get_mut(), &facts.map);
                        }
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(facts.map.clone());
                        }
                    }
                }
            }};
        }

        match op {
            Op::LoadSlot { dst, src } => {
                if *dst == *src {
                    // No-op after rewriting: value unchanged, facts unchanged.
                    noop[i] = true;
                } else {
                    let (d, s) = (*dst, *src);
                    facts.kill(d);
                    // Cap the fact-map size: every branch clones it for the
                    // target's pending snapshot, so an unbounded map makes
                    // branchy streams quadratic. Copies are consumed close to
                    // where they're made; dropping facts past the cap only
                    // costs optimisation, not correctness.
                    if facts.map.len() < 128 {
                        facts.add(d, s);
                    }
                }
            }
            Op::Jump { target } => {
                let t = *target;
                snapshot_to!(t);
                ft_reachable = false;
            }
            Op::BranchIfZero { target, .. } | Op::BranchIfNotEqLit { target, .. } => {
                let t = *target;
                snapshot_to!(t);
            }
            Op::LoadStateAndBranchIfNotEqLit { dst, target, .. } => {
                let (d, t) = (*dst, *target);
                facts.kill(d);
                snapshot_to!(t);
            }
            Op::MemoryFill { dst_slot, count_slot, exit_target, .. } => {
                let (a, b, t) = (*dst_slot, *count_slot, *exit_target);
                facts.kill(a);
                facts.kill(b);
                snapshot_to!(t);
                ft_reachable = false;
            }
            Op::MemoryCopy { src_slot, dst_slot, count_slot, exit_target } => {
                let (a, b, c, t) = (*src_slot, *dst_slot, *count_slot, *exit_target);
                facts.kill(a);
                facts.kill(b);
                facts.kill(c);
                snapshot_to!(t);
                ft_reachable = false;
            }
            Op::Dispatch { .. } => {
                // Entry ops may write slots aliasing this scope's temporaries
                // (inlined bodies share slot numbers across streams).
                facts.clear();
            }
            Op::StoreState { .. } | Op::StoreMem { .. } => {}
            other => {
                if let Some(d) = op_dst(other) {
                    facts.kill(d);
                }
            }
        }
    }

    // --- Backward liveness DCE ---
    // `live` tracks only ordinary (scratch) slots. External and
    // dispatch-read slots are NEVER inserted — their producers are kept
    // unconditionally by the membership checks at the removal decision, so
    // tracking them would only bloat the per-label snapshot clones (the
    // main stream reads tens of thousands of pinned property slots).
    live.clear();
    // live-in set at each label index (available because targets are forward:
    // in the reverse walk a target is always visited before its branch).
    live_at.clear();
    let live_in = |live_at: &FxMap<usize, FxSet<Slot>>, t: u32| {
        if (t as usize) >= len {
            FxSet::default()
        } else {
            live_at.get(&(t as usize)).cloned().unwrap_or_default()
        }
    };
    dead.clear();
    dead.resize(len, false);
    let mut removed = 0usize;

    macro_rules! gen_read {
        ($s:expr) => {{
            let s = $s;
            if !external.contains(&s) && !barrier_live.contains(&s) {
                live.insert(s);
            }
        }};
    }

    for i in (0..len).rev() {
        let op = &ops[i];
        if noop[i] {
            dead[i] = true;
        } else {
            match op {
                Op::Jump { target } => {
                    *live = live_in(live_at, *target);
                }
                Op::BranchIfZero { cond, target } => {
                    live.extend(live_in(live_at, *target));
                    gen_read!(*cond);
                }
                Op::BranchIfNotEqLit { a, target, .. } => {
                    live.extend(live_in(live_at, *target));
                    gen_read!(*a);
                }
                Op::LoadStateAndBranchIfNotEqLit { dst, target, .. } => {
                    live.extend(live_in(live_at, *target));
                    live.remove(dst);
                }
                Op::MemoryFill { dst_slot, val_slot, count_slot, exit_target } => {
                    *live = live_in(live_at, *exit_target);
                    gen_read!(*dst_slot);
                    gen_read!(*val_slot);
                    gen_read!(*count_slot);
                }
                Op::MemoryCopy { src_slot, dst_slot, count_slot, exit_target } => {
                    *live = live_in(live_at, *exit_target);
                    gen_read!(*src_slot);
                    gen_read!(*dst_slot);
                    gen_read!(*count_slot);
                }
                Op::Dispatch { dst, key, .. } => {
                    // Table read-set handled via barrier_live.
                    live.remove(dst);
                    gen_read!(*key);
                }
                Op::StoreState { src, .. } => {
                    gen_read!(*src);
                }
                Op::StoreMem { addr_slot, src } => {
                    gen_read!(*addr_slot);
                    gen_read!(*src);
                }
                other => {
                    debug_assert!(op_is_pure(other));
                    let d = op_dst(other).expect("pure op without dst");
                    if !live.contains(&d) && !external.contains(&d) && !barrier_live.contains(&d) {
                        dead[i] = true;
                    } else if let Op::LoadSlot { dst: r, src: x } = *other {
                        // Store-forwarding: if the previous op is the sole
                        // producer of `x` and `x` is dead after this copy,
                        // retarget the producer to write `r` directly and
                        // drop the copy. Adjacency plus !is_label[i] means no
                        // other path can observe the intermediate state, and
                        // forwarding can't change `r`'s value at any read.
                        let fwd = i >= 1
                            && !is_label[i]
                            && !noop[i - 1]
                            && !live.contains(&x)
                            && !external.contains(&x)
                            && !barrier_live.contains(&x)
                            && match &ops[i - 1] {
                                // Producer must write exactly `x`, never
                                // branch, and have no other dst.
                                p if op_is_pure(p) => op_dst(p) == Some(x),
                                Op::Dispatch { dst, .. } => *dst == x,
                                _ => false,
                            };
                        if fwd {
                            dead[i] = true;
                            // Retarget the producer. Reads were already
                            // final for ops[i-1] (the forward pass is done),
                            // and the backward walk hasn't visited it yet.
                            let prev = &mut ops[i - 1];
                            match prev {
                                Op::LoadLit { dst, .. }
                                | Op::LoadSlot { dst, .. }
                                | Op::LoadState { dst, .. }
                                | Op::LoadMem { dst, .. }
                                | Op::LoadMem16 { dst, .. }
                                | Op::LoadPackedByte { dst, .. }
                                | Op::Add { dst, .. }
                                | Op::AddLit { dst, .. }
                                | Op::Sub { dst, .. }
                                | Op::SubLit { dst, .. }
                                | Op::Mul { dst, .. }
                                | Op::MulLit { dst, .. }
                                | Op::Div { dst, .. }
                                | Op::Mod { dst, .. }
                                | Op::ModLit { dst, .. }
                                | Op::Neg { dst, .. }
                                | Op::Abs { dst, .. }
                                | Op::Sign { dst, .. }
                                | Op::Pow { dst, .. }
                                | Op::Min { dst, .. }
                                | Op::Max { dst, .. }
                                | Op::Clamp { dst, .. }
                                | Op::Round { dst, .. }
                                | Op::Floor { dst, .. }
                                | Op::And { dst, .. }
                                | Op::AndLit { dst, .. }
                                | Op::Shr { dst, .. }
                                | Op::ShrLit { dst, .. }
                                | Op::Shl { dst, .. }
                                | Op::ShlLit { dst, .. }
                                | Op::Bit { dst, .. }
                                | Op::BitAnd16 { dst, .. }
                                | Op::BitOr16 { dst, .. }
                                | Op::BitXor16 { dst, .. }
                                | Op::BitNot16 { dst, .. }
                                | Op::CmpEq { dst, .. }
                                | Op::DispatchFlatArray { dst, .. }
                                | Op::Dispatch { dst, .. } => *dst = r,
                                _ => unreachable!("forwarding producer checked above"),
                            }
                        } else {
                            live.remove(&r);
                            gen_read!(x);
                        }
                    } else {
                        live.remove(&d);
                        for r in op_slots_read(other) {
                            gen_read!(r);
                        }
                    }
                }
            }
        }
        if is_label[i] {
            live_at.insert(i, live.clone());
        }
    }

    // --- Remove dead ops, remapping branch targets ---
    if dead.iter().any(|&d| d) {
        new_indices.clear();
        new_indices.resize(len, 0u32);
        let mut offset = 0u32;
        for i in 0..len {
            new_indices[i] = i as u32 - offset;
            if dead[i] {
                offset += 1;
            }
        }
        let new_end = len as u32 - offset;

        for op in ops.iter_mut() {
            let t = match op {
                Op::Jump { target }
                | Op::BranchIfZero { target, .. }
                | Op::BranchIfNotEqLit { target, .. }
                | Op::LoadStateAndBranchIfNotEqLit { target, .. } => target,
                Op::MemoryFill { exit_target, .. } | Op::MemoryCopy { exit_target, .. } => {
                    exit_target
                }
                _ => continue,
            };
            let old = *t as usize;
            *t = if old < len { new_indices[old] } else { new_end };
        }

        let mut write = 0usize;
        for read in 0..len {
            if dead[read] {
                continue;
            }
            if write != read {
                ops.swap(write, read);
            }
            write += 1;
        }
        removed = len - write;
        ops.truncate(write);
    }

    (propagated, removed)
}

/// Drive `copy_opt_stream` over every op stream in the program.
/// Returns (reads redirected, ops removed).
fn copy_propagate_and_dce(program: &mut CompiledProgram) -> (usize, usize) {
    // Transitive read-set per dispatch table: every slot any entry/fallback op
    // reads (plus result/fallback slots, which the Dispatch op itself reads),
    // closed over nested Dispatch references. Over-approximates "slots a
    // Dispatch on this table may consume from the containing scope".
    let n_tables = program.dispatch_tables.len();
    let mut table_reads: Vec<FxSet<Slot>> = Vec::with_capacity(n_tables);
    let mut table_refs: Vec<Vec<usize>> = Vec::with_capacity(n_tables);
    for table in &program.dispatch_tables {
        let mut reads: FxSet<Slot> = FxSet::default();
        let mut refs: Vec<usize> = Vec::new();
        let mut collect = |ops: &[Op]| {
            let mut local_reads = Vec::new();
            let mut local_refs = Vec::new();
            for op in ops {
                local_reads.extend(op_slots_read(op));
                if let Op::Dispatch { table_id, .. } = op {
                    local_refs.push(*table_id as usize);
                }
            }
            (local_reads, local_refs)
        };
        let (r, f) = collect(&table.fallback_ops);
        reads.extend(r);
        refs.extend(f);
        reads.insert(table.fallback_slot);
        for (entry_ops, result_slot) in table.entries.values() {
            let (r, f) = collect(entry_ops);
            reads.extend(r);
            refs.extend(f);
            reads.insert(*result_slot);
        }
        table_reads.push(reads);
        table_refs.push(refs);
    }
    loop {
        let mut changed = false;
        for i in 0..n_tables {
            let mut add: Vec<Slot> = Vec::new();
            for &c in &table_refs[i] {
                if c < n_tables && c != i {
                    add.extend(table_reads[c].iter().filter(|s| !table_reads[i].contains(s)));
                }
            }
            if !add.is_empty() {
                changed = true;
                table_reads[i].extend(add);
            }
        }
        if !changed {
            break;
        }
    }

    // Main-stream live-out: end-of-tick readers (writeback, property_slots,
    // broadcast/packed-port control slots) plus everything the broadcast
    // value/spillover sub-op streams read (they execute after the main
    // stream within the same tick).
    let mut main_external: FxSet<Slot> = collect_main_pinned(program).into_iter().collect();
    for bw in &program.broadcast_writes {
        for op in &bw.value_ops {
            main_external.extend(op_slots_read(op));
        }
        main_external.insert(bw.value_slot);
        if let Some(ref sp) = bw.spillover {
            for (spill_ops, spill_slot) in sp.entries.values() {
                for op in spill_ops {
                    main_external.extend(op_slots_read(op));
                }
                main_external.insert(*spill_slot);
            }
        }
    }

    let mut propagated = 0usize;
    let mut removed = 0usize;
    let mut scratch = CopyOptScratch::default();
    let mut ext_single: FxSet<Slot> = FxSet::default();

    let (p, r) = copy_opt_stream(&mut program.ops, &main_external, &table_reads, &mut scratch);
    propagated += p;
    removed += r;

    for table in &mut program.dispatch_tables {
        for (entry_ops, result_slot) in table.entries.values_mut() {
            ext_single.clear();
            ext_single.insert(*result_slot);
            let (p, r) = copy_opt_stream(entry_ops, &ext_single, &table_reads, &mut scratch);
            propagated += p;
            removed += r;
        }
        ext_single.clear();
        ext_single.insert(table.fallback_slot);
        let (p, r) = copy_opt_stream(&mut table.fallback_ops, &ext_single, &table_reads, &mut scratch);
        propagated += p;
        removed += r;
    }

    for bw in &mut program.broadcast_writes {
        ext_single.clear();
        ext_single.insert(bw.value_slot);
        let (p, r) = copy_opt_stream(&mut bw.value_ops, &ext_single, &table_reads, &mut scratch);
        propagated += p;
        removed += r;
        if let Some(ref mut sp) = bw.spillover {
            for (spill_ops, spill_slot) in sp.entries.values_mut() {
                ext_single.clear();
                ext_single.insert(*spill_slot);
                let (p, r) = copy_opt_stream(spill_ops, &ext_single, &table_reads, &mut scratch);
                propagated += p;
                removed += r;
            }
        }
    }

    (propagated, removed)
}

// ---------------------------------------------------------------------------
// Peephole: fuse LoadLit + CmpEq + BranchIfZero → BranchIfNotEqLit
// ---------------------------------------------------------------------------

/// Fuse `LoadLit(dst=X, val=N) + CmpEq(dst=Y, a=P, b=X) + BranchIfZero(cond=Y, target=T)`
/// into a single `BranchIfNotEqLit(a=P, val=N, target=T)`.
///
/// The fused op skips two intermediate slot writes and two match-dispatch cycles
/// in the interpreter, which matters because these triplets make up ~99% of ops
/// in CSS-DOS v4 programs.
///
/// Also applies to dispatch table entry ops and broadcast write value ops.
///
/// Returns the number of triplets fused.
fn fuse_cmp_branch(program: &mut CompiledProgram) -> usize {
    let mut total = 0;
    total += fuse_ops(&mut program.ops);
    for table in &mut program.dispatch_tables {
        for (_key, (entry_ops, _slot)) in &mut table.entries {
            total += fuse_ops(entry_ops);
        }
        total += fuse_ops(&mut table.fallback_ops);
    }
    for bw in &mut program.broadcast_writes {
        total += fuse_ops(&mut bw.value_ops);
        if let Some(ref mut spillover) = bw.spillover {
            for (_key, (spill_ops, _slot)) in &mut spillover.entries {
                total += fuse_ops(spill_ops);
            }
        }
    }
    total
}

/// Fuse triplets in a single ops array. Returns number fused.
///
/// After fusion, the three original ops are replaced by:
///   [BranchIfNotEqLit, Nop(LoadLit 0→0), Nop(LoadLit 0→0)]
/// The nops are then stripped in a second pass, with branch targets adjusted.
fn fuse_ops(ops: &mut Vec<Op>) -> usize {
    if ops.len() < 3 {
        return 0;
    }

    // Pass 1: identify fusable triplets and mark them
    let mut fused_at: Vec<usize> = Vec::new(); // indices of the LoadLit in each triplet

    let mut i = 0;
    while i + 2 < ops.len() {
        // Pattern: LoadLit(dst=X, val=N), CmpEq(dst=Y, a=P, b=X), BranchIfZero(cond=Y, target=T)
        if let (
            Op::LoadLit { dst: lit_dst, val },
            Op::CmpEq { dst: cmp_dst, a: cmp_a, b: cmp_b },
            Op::BranchIfZero { cond, target },
        ) = (&ops[i], &ops[i + 1], &ops[i + 2])
        {
            if cmp_b == lit_dst && cond == cmp_dst {
                // Fusable! Replace with BranchIfNotEqLit
                let a = *cmp_a;
                let v = *val;
                let t = *target;
                ops[i] = Op::BranchIfNotEqLit { a, val: v, target: t };
                // Mark the other two as nops (LoadLit 0→slot 0 is harmless but we'll strip them)
                ops[i + 1] = Op::LoadLit { dst: 0, val: 0 };
                ops[i + 2] = Op::LoadLit { dst: 0, val: 0 };
                fused_at.push(i);
                i += 3;
                continue;
            }
        }
        i += 1;
    }

    if fused_at.is_empty() {
        return 0;
    }

    let fused_count = fused_at.len();

    // Pass 2: strip the nop slots (indices i+1 and i+2 for each fused triplet)
    // Build a set of indices to remove
    let mut remove_set = Vec::with_capacity(fused_count * 2);
    for &idx in &fused_at {
        remove_set.push(idx + 1);
        remove_set.push(idx + 2);
    }
    remove_set.sort();

    // Build old→new index mapping for branch target adjustment
    let mut new_indices = vec![0u32; ops.len()];
    let mut offset: u32 = 0;
    let mut remove_pos = 0;
    for old_idx in 0..ops.len() {
        if remove_pos < remove_set.len() && remove_set[remove_pos] == old_idx {
            remove_pos += 1;
            // This op is being removed — map to next valid op
            new_indices[old_idx] = old_idx as u32 - offset;
            offset += 1;
        } else {
            new_indices[old_idx] = old_idx as u32 - offset;
        }
    }
    // For targets pointing past the end
    let new_end = ops.len() as u32 - offset;

    // Pass 3: adjust all branch/jump targets
    for op in ops.iter_mut() {
        match op {
            Op::BranchIfZero { target, .. }
            | Op::BranchIfNotEqLit { target, .. }
            | Op::LoadStateAndBranchIfNotEqLit { target, .. }
            | Op::BranchIfNotEqLit2 { target, .. }
            | Op::DispatchChain { miss_target: target, .. }
            | Op::Jump { target } => {
                let old = *target as usize;
                *target = if old < new_indices.len() {
                    new_indices[old]
                } else {
                    new_end
                };
            }
            _ => {}
        }
    }

    // Pass 4: actually remove the nop ops
    let mut write = 0;
    let mut remove_pos = 0;
    for read in 0..ops.len() {
        if remove_pos < remove_set.len() && remove_set[remove_pos] == read {
            remove_pos += 1;
            continue; // skip
        }
        if write != read {
            ops.swap(write, read);
        }
        write += 1;
    }
    ops.truncate(write);

    fused_count
}

// ---------------------------------------------------------------------------
// Peephole: fuse LoadState + BranchIfNotEqLit → LoadStateAndBranchIfNotEqLit
// ---------------------------------------------------------------------------

/// Fuse `LoadState(dst=X, addr=A) + BranchIfNotEqLit(a=X, val=N, target=T)`
/// into a single `LoadStateAndBranchIfNotEqLit(dst=X, addr=A, val=N, target=T)`.
///
/// The fused op still writes slot X so downstream code reading X is unaffected.
/// Saves one dispatch cycle in the interpreter hot loop.
///
/// Must run AFTER `build_dispatch_chains` so we only fuse isolated BranchIfNotEqLit
/// ops left behind after chain collapsing.
// ---------------------------------------------------------------------------
// Peephole: fuse adjacent BranchIfNotEqLit pairs sharing a target → BranchIfNotEqLit2
// ---------------------------------------------------------------------------
//
// Static structure (doom8088): 1395 adjacent BIfNEL pairs in the static op
// stream, all different-slot. 1330 of those (95.3%) share their `target` —
// they form an AND-guard:
//
//   BranchIfNotEqLit { a: A, val: vA, target: T }
//   BranchIfNotEqLit { a: B, val: vB, target: T }
//
// = "if (slots[A] != vA OR slots[B] != vB) goto T". When both equal,
// fall through to ops[i+2].
//
// Runtime impact (200K-tick window post-stage_loading): BIfNEL→BIfNEL is
// 12.35% of dispatched ops — the largest adjacency in the profile.
//
// Fusion replaces ops[i] with `BranchIfNotEqLit2 { a1, val1, a2, val2,
// target: T }`. ops[i+1] is left in place as the original BIfNEL so any
// external jump that lands on it still executes correctly (same dead-code
// convention as `build_dispatch_chains`). The fused op explicitly advances
// pc by 2 on fall-through to skip it.
//
// Constraints:
// - ops[i+1] must NOT be a jump target (would skip the second check). We
//   build is_target by scanning all ops for branch destinations, chain-
//   table body PCs, and dispatch fallback targets.
// - The two slots must differ (same-slot pairs go through chain pass).
// - Targets must match.
fn fuse_diff_slot_bifnel_pairs(program: &mut CompiledProgram) -> usize {
    let mut total = 0;
    total += fuse_bif2_ops(&mut program.ops, &program.chain_tables);
    for table in &mut program.dispatch_tables {
        for (_k, (entry_ops, _s)) in &mut table.entries {
            total += fuse_bif2_ops(entry_ops, &program.chain_tables);
        }
        total += fuse_bif2_ops(&mut table.fallback_ops, &program.chain_tables);
    }
    for bw in &mut program.broadcast_writes {
        total += fuse_bif2_ops(&mut bw.value_ops, &program.chain_tables);
        if let Some(ref mut sp) = bw.spillover {
            for (_k, (spill_ops, _s)) in &mut sp.entries {
                total += fuse_bif2_ops(spill_ops, &program.chain_tables);
            }
        }
    }
    total
}

fn fuse_bif2_ops(ops: &mut [Op], chain_tables: &[DispatchChainTable]) -> usize {
    let n = ops.len();
    if n < 2 { return 0; }

    // Build is_target: every op index that is a branch destination, a chain-
    // table body PC pointing into this array, or a flat-table chain target.
    let mut is_target = vec![false; n];
    let mark = |is_t: &mut Vec<bool>, t: u32| {
        let t = t as usize;
        if t < is_t.len() { is_t[t] = true; }
    };
    for op in ops.iter() {
        match op {
            Op::BranchIfZero { target, .. }
            | Op::BranchIfNotEqLit { target, .. }
            | Op::LoadStateAndBranchIfNotEqLit { target, .. }
            | Op::BranchIfNotEqLit2 { target, .. }
            | Op::Jump { target } => mark(&mut is_target, *target),
            Op::DispatchChain { miss_target, chain_id, .. } => {
                mark(&mut is_target, *miss_target);
                let table = &chain_tables[*chain_id as usize];
                for (_v, body_pc) in &table.entries {
                    mark(&mut is_target, *body_pc);
                }
                if let Some(flat) = &table.flat_table {
                    for &t in &flat.targets {
                        if t != u32::MAX { mark(&mut is_target, t); }
                    }
                }
            }
            Op::Dispatch { fallback_target, .. } => mark(&mut is_target, *fallback_target),
            Op::MemoryFill { exit_target, .. } | Op::MemoryCopy { exit_target, .. } => {
                mark(&mut is_target, *exit_target);
            }
            _ => {}
        }
    }

    let mut fused = 0;
    let mut i = 0;
    while i + 1 < ops.len() {
        // Snapshot inputs to avoid double-borrow.
        let (a0, v0, t0) = match &ops[i] {
            Op::BranchIfNotEqLit { a, val, target } => (*a, *val, *target),
            _ => { i += 1; continue; }
        };
        let (a1, v1, t1) = match &ops[i + 1] {
            Op::BranchIfNotEqLit { a, val, target } => (*a, *val, *target),
            _ => { i += 1; continue; }
        };

        // Constraints. Targets must match; slots must differ; ops[i+1] must
        // not itself be a branch destination (otherwise external jumps that
        // land on it expect a single-condition test).
        if t0 != t1 { i += 1; continue; }
        if a0 == a1 { i += 1; continue; }
        if is_target[i + 1] { i += 1; continue; }

        ops[i] = Op::BranchIfNotEqLit2 {
            a1: a0, val1: v0,
            a2: a1, val2: v1,
            target: t0,
        };
        // ops[i+1] stays as-is (dead BIfNEL preserved for external-jump safety).
        // The fused op's eval arm explicitly advances pc by 2 on fall-through.
        fused += 1;
        i += 2;
    }
    fused
}

fn fuse_loadstate_branch(program: &mut CompiledProgram) -> usize {
    // Gather chain-table body PCs that point into each ops array so we
    // can remap them after fusion. chain_tables are shared across all
    // op arrays (indexed by chain_id via DispatchChain), so we pair each
    // ops array with the subset of chain tables reachable from it.
    // Simpler approach: remap chain_tables in-place per ops array, using
    // the ops array's own DispatchChain references.
    let mut total = 0;
    total += fuse_ls_ops(&mut program.ops, &mut program.chain_tables);
    for table in &mut program.dispatch_tables {
        for (_k, (entry_ops, _s)) in &mut table.entries {
            total += fuse_ls_ops(entry_ops, &mut program.chain_tables);
        }
        total += fuse_ls_ops(&mut table.fallback_ops, &mut program.chain_tables);
    }
    for bw in &mut program.broadcast_writes {
        total += fuse_ls_ops(&mut bw.value_ops, &mut program.chain_tables);
        if let Some(ref mut sp) = bw.spillover {
            for (_k, (spill_ops, _s)) in &mut sp.entries {
                total += fuse_ls_ops(spill_ops, &mut program.chain_tables);
            }
        }
    }
    total
}

fn fuse_ls_ops(ops: &mut Vec<Op>, chain_tables: &mut [DispatchChainTable]) -> usize {
    if ops.len() < 2 {
        return 0;
    }

    // Collect the chain_ids referenced from THIS ops array so we can remap
    // their body PCs after fusion. DispatchChain ops encode the chain_id.
    let mut referenced_chain_ids: Vec<u32> = ops.iter().filter_map(|op| {
        if let Op::DispatchChain { chain_id, .. } = op { Some(*chain_id) } else { None }
    }).collect();
    referenced_chain_ids.sort();
    referenced_chain_ids.dedup();

    // Identify any op index that is a branch/jump target — we can't fuse
    // if the BranchIfNotEqLit is a jump target (fusing would skip the LoadState
    // when landed on directly).
    let mut is_target = vec![false; ops.len() + 1];
    for op in ops.iter() {
        match op {
            Op::BranchIfZero { target, .. }
            | Op::BranchIfNotEqLit { target, .. }
            | Op::LoadStateAndBranchIfNotEqLit { target, .. }
            | Op::BranchIfNotEqLit2 { target, .. }
            | Op::Jump { target } => {
                let t = *target as usize;
                if t < is_target.len() { is_target[t] = true; }
            }
            Op::DispatchChain { miss_target, .. } => {
                let t = *miss_target as usize;
                if t < is_target.len() { is_target[t] = true; }
            }
            Op::Dispatch { fallback_target, .. } => {
                let t = *fallback_target as usize;
                if t < is_target.len() { is_target[t] = true; }
            }
            _ => {}
        }
    }
    // Also mark chain-table body PCs — these point INTO this ops array.
    for &cid in &referenced_chain_ids {
        for (_v, body_pc) in &chain_tables[cid as usize].entries {
            let t = *body_pc as usize;
            if t < is_target.len() { is_target[t] = true; }
        }
    }

    let mut fused_at: Vec<usize> = Vec::new();
    let mut i = 0;
    while i + 1 < ops.len() {
        let pair = (&ops[i], &ops[i + 1]);
        if let (
            Op::LoadState { dst: ls_dst, addr },
            Op::BranchIfNotEqLit { a: br_a, val, target },
        ) = pair
        {
            if ls_dst == br_a && !is_target[i + 1] {
                let dst = *ls_dst;
                let addr = *addr;
                let val = *val;
                let target = *target;
                ops[i] = Op::LoadStateAndBranchIfNotEqLit { dst, addr, val, target };
                // Mark the second slot for removal by replacing with a no-op placeholder.
                // We strip it in a second pass.
                ops[i + 1] = Op::LoadLit { dst: 0, val: 0 };
                fused_at.push(i + 1);
                i += 2;
                continue;
            }
        }
        i += 1;
    }

    if fused_at.is_empty() {
        return 0;
    }

    let fused_count = fused_at.len();

    // Strip removed slots and remap branch targets
    let mut remove_set: Vec<usize> = fused_at;
    remove_set.sort();

    let mut new_indices = vec![0u32; ops.len()];
    let mut offset: u32 = 0;
    let mut remove_pos = 0;
    for old_idx in 0..ops.len() {
        if remove_pos < remove_set.len() && remove_set[remove_pos] == old_idx {
            remove_pos += 1;
            new_indices[old_idx] = old_idx as u32 - offset;
            offset += 1;
        } else {
            new_indices[old_idx] = old_idx as u32 - offset;
        }
    }
    let new_end = ops.len() as u32 - offset;

    for op in ops.iter_mut() {
        match op {
            Op::BranchIfZero { target, .. }
            | Op::BranchIfNotEqLit { target, .. }
            | Op::LoadStateAndBranchIfNotEqLit { target, .. }
            | Op::BranchIfNotEqLit2 { target, .. }
            | Op::Jump { target } => {
                let old = *target as usize;
                *target = if old < new_indices.len() { new_indices[old] } else { new_end };
            }
            Op::DispatchChain { miss_target, .. } => {
                let old = *miss_target as usize;
                *miss_target = if old < new_indices.len() { new_indices[old] } else { new_end };
            }
            Op::Dispatch { fallback_target, .. } => {
                let old = *fallback_target as usize;
                *fallback_target = if old < new_indices.len() { new_indices[old] } else { new_end };
            }
            _ => {}
        }
    }

    // Remap chain-table body PCs that point into this ops array.
    for &cid in &referenced_chain_ids {
        for (_v, body_pc) in chain_tables[cid as usize].entries.iter_mut() {
            let old = *body_pc as usize;
            *body_pc = if old < new_indices.len() { new_indices[old] } else { new_end };
        }
        // Also remap flat_table targets if present.
        if let Some(ref mut flat) = chain_tables[cid as usize].flat_table {
            for t in flat.targets.iter_mut() {
                if *t == u32::MAX { continue; }
                let old = *t as usize;
                *t = if old < new_indices.len() { new_indices[old] } else { new_end };
            }
        }
    }

    let mut write = 0;
    let mut remove_pos = 0;
    for read in 0..ops.len() {
        if remove_pos < remove_set.len() && remove_set[remove_pos] == read {
            remove_pos += 1;
            continue;
        }
        if write != read {
            ops.swap(write, read);
        }
        write += 1;
    }
    ops.truncate(write);

    fused_count
}

// ---------------------------------------------------------------------------
// Replicated-body recognition — collapse unrolled straight-line regions
// ---------------------------------------------------------------------------

/// Walks every op array in the program, finds maximal straight-line regions
/// (no incoming branch targets in the middle, no branch/jump/dispatch/bulk-
/// memory/Vec-bearing variants), and where such a region is K-periodic with
/// K >= MIN_REPS, replaces the region with one `Op::ReplicatedBody`.
///
/// Returns the number of regions folded.
///
/// Must run after `compact_slots` — folding alters op-array length and
/// branch targets, and the slot-walking utility functions
/// (`op_slots_read`, `op_dst`, `map_op_slots`, `seed_from_parent`) panic on
/// `ReplicatedBody` because they have no sensible answer for it.
fn recognise_replicated_bodies(program: &mut CompiledProgram) -> usize {
    let mut total = 0;
    total += recognise_replicated_in(&mut program.ops, &mut program.chain_tables);
    for table in &mut program.dispatch_tables {
        for (_k, (entry_ops, _s)) in &mut table.entries {
            total += recognise_replicated_in(entry_ops, &mut program.chain_tables);
        }
        total += recognise_replicated_in(&mut table.fallback_ops, &mut program.chain_tables);
    }
    for bw in &mut program.broadcast_writes {
        total += recognise_replicated_in(&mut bw.value_ops, &mut program.chain_tables);
        if let Some(ref mut sp) = bw.spillover {
            for (_k, (spill_ops, _s)) in &mut sp.entries {
                total += recognise_replicated_in(spill_ops, &mut program.chain_tables);
            }
        }
    }
    total
}

/// Per-ops-array recogniser. See `recognise_replicated_bodies` for context.
fn recognise_replicated_in(
    ops: &mut Vec<Op>,
    chain_tables: &mut [DispatchChainTable],
) -> usize {
    use crate::pattern::replicated_body::{
        classify_body, detect_period, BailReason, OperandClass, MIN_PERIOD, MIN_REPS,
    };

    // Diagnostic: count straight-line region lengths so we can tell whether
    // the issue is "no straight-line regions long enough" or "regions exist
    // but aren't periodic". Disabled by default; flip CALCITE_DBG_REPL=1 to
    // enable. The histogram is printed at end of recogniser.
    let dbg_enabled = std::env::var_os("CALCITE_DBG_REPL").is_some();
    let mut dbg_region_lens: Vec<usize> = Vec::new();
    let mut dbg_classify_attempts: usize = 0;
    let mut dbg_classify_bails: Vec<&'static str> = Vec::new();

    if ops.len() < MIN_PERIOD * MIN_REPS {
        return 0;
    }

    // Build is_target: every op index that is a branch destination, a chain-
    // table body PC pointing into this array, or a flat-table chain target.
    // We may not fold a region that contains an internal jump target (the
    // jumper would land in the middle of the body and skip earlier ops).
    let referenced_chain_ids: Vec<u32> = {
        let mut v: Vec<u32> = ops.iter().filter_map(|op| {
            if let Op::DispatchChain { chain_id, .. } = op { Some(*chain_id) } else { None }
        }).collect();
        v.sort();
        v.dedup();
        v
    };

    let mut is_target = vec![false; ops.len() + 1];
    for op in ops.iter() {
        match op {
            Op::BranchIfZero { target, .. }
            | Op::BranchIfNotEqLit { target, .. }
            | Op::LoadStateAndBranchIfNotEqLit { target, .. }
            | Op::BranchIfNotEqLit2 { target, .. }
            | Op::Jump { target } => mark_target(&mut is_target, *target),
            Op::DispatchChain { miss_target, .. } => mark_target(&mut is_target, *miss_target),
            Op::Dispatch { fallback_target, .. } => mark_target(&mut is_target, *fallback_target),
            Op::MemoryFill { exit_target, .. } | Op::MemoryCopy { exit_target, .. } => {
                mark_target(&mut is_target, *exit_target);
            }
            _ => {}
        }
    }
    for &cid in &referenced_chain_ids {
        for (_v, body_pc) in &chain_tables[cid as usize].entries {
            mark_target(&mut is_target, *body_pc);
        }
        if let Some(ref flat) = chain_tables[cid as usize].flat_table {
            for &t in &flat.targets {
                if t != u32::MAX {
                    mark_target(&mut is_target, t);
                }
            }
        }
    }

    // is_eligible: an op may live inside a body iff it has no branch field,
    // no Vec-bearing field, and no exit_target. Equivalent to "extract_op_fields
    // returns is_branch=false and the variant is not Min/Max/Call".
    let eligible = |op: &Op| -> bool {
        match op {
            Op::Min { .. } | Op::Max { .. } | Op::Call { .. } => false,
            Op::BranchIfZero { .. }
            | Op::BranchIfNotEqLit { .. }
            | Op::LoadStateAndBranchIfNotEqLit { .. }
            | Op::Jump { .. }
            | Op::DispatchChain { .. }
            | Op::Dispatch { .. }
            | Op::MemoryFill { .. }
            | Op::MemoryCopy { .. }
            | Op::ReplicatedBody { .. } => false,
            _ => true,
        }
    };

    // Scan: for each potential region start, find the maximal eligible run
    // (no internal target, all ops eligible). Try to fold; if it folds, jump
    // past it and continue. If it doesn't, advance one op and retry.
    let mut folded_regions: Vec<(usize, usize, Op)> = Vec::new(); // (start, end_exclusive, replacement)
    let mut i = 0;
    while i < ops.len() {
        if !eligible(&ops[i]) {
            i += 1;
            continue;
        }
        // Find region end: stop at first ineligible op or first internal
        // target (a target at i is allowed — it's the region head).
        let region_start = i;
        let mut end = i + 1;
        while end < ops.len() && eligible(&ops[end]) && !is_target[end] {
            end += 1;
        }
        let region_len = end - region_start;
        if dbg_enabled {
            dbg_region_lens.push(region_len);
        }
        if region_len < MIN_PERIOD * MIN_REPS {
            i = end;
            continue;
        }
        let region = &ops[region_start..end];

        // Try every period that divides the region and yields >= MIN_REPS.
        // detect_period takes the full slice and finds the smallest period
        // where the *whole* slice is periodic. If the region's length isn't
        // periodic, try truncating the tail to find a periodic prefix.
        if let Some(p) = detect_period(region) {
            // Periodic for the full region. Try to classify.
            dbg_classify_attempts += 1;
            match classify_body(region, p.period, p.reps) {
                Ok(bt) => {
                    let folded = build_replicated_body_op(&bt, region);
                    folded_regions.push((region_start, end, folded));
                    i = end;
                    continue;
                }
                Err(bail) => {
                    if dbg_enabled {
                        dbg_classify_bails.push(match bail {
                            BailReason::BranchInBody { .. } => "BranchInBody",
                            BailReason::StructuralMismatch { .. } => "StructuralMismatch",
                            BailReason::NonAffineOperand { .. } => "NonAffineOperand",
                            BailReason::ArityMismatch { .. } => "ArityMismatch",
                        });
                    }
                }
            }
        } else {
            // The region wasn't fully periodic. Try the largest periodic
            // prefix at every candidate period >= MIN_PERIOD up to
            // region_len / MIN_REPS. This catches "16 reps + a couple of
            // trailing odd ops".
            let max_period = region_len / MIN_REPS;
            'periods: for period in MIN_PERIOD..=max_period {
                let reps_max = region_len / period;
                if reps_max < MIN_REPS {
                    break;
                }
                let prefix_reps = crate::pattern::replicated_body::longest_periodic_prefix(
                    region, 0, period,
                );
                if prefix_reps < MIN_REPS {
                    continue;
                }
                let prefix_len = period * prefix_reps;
                let prefix = &region[..prefix_len];
                if let Ok(bt) = classify_body(prefix, period, prefix_reps) {
                    let folded = build_replicated_body_op(&bt, prefix);
                    folded_regions.push((region_start, region_start + prefix_len, folded));
                    i = region_start + prefix_len;
                    break 'periods;
                }
            }
            // If no period worked, advance past the region (it's not foldable).
            if folded_regions.last().map(|(_s, e, _)| *e) != Some(end)
                && folded_regions.last().map(|(s, _e, _)| *s) != Some(region_start)
            {
                i = end;
                continue;
            }
            // If we did fold something here, `i` was already updated above.
            continue;
        }
        i = end;
    }

    let _ = OperandClass::Constant(0); // keep import alive

    if dbg_enabled {
        // Per-array histogram: how big are the straight-line regions, and
        // how many classify attempts bailed and why?
        if !dbg_region_lens.is_empty() {
            let max = *dbg_region_lens.iter().max().unwrap_or(&0);
            let n_long: usize = dbg_region_lens.iter().filter(|&&l| l >= MIN_PERIOD * MIN_REPS).count();
            let n_total = dbg_region_lens.len();
            eprintln!(
                "[repl-dbg] ops_len={} regions={} long(>={})={} max_region={} classify_attempts={} bails={:?}",
                ops.len(),
                n_total,
                MIN_PERIOD * MIN_REPS,
                n_long,
                max,
                dbg_classify_attempts,
                dbg_classify_bails,
            );
        }
    }

    if folded_regions.is_empty() {
        return 0;
    }

    let n_folded = folded_regions.len();

    // Build a new ops array with folded regions replaced. Track the index
    // remapping so we can fix branch targets.
    let mut new_ops: Vec<Op> = Vec::with_capacity(ops.len());
    let mut new_indices = vec![0u32; ops.len() + 1];
    let mut region_iter = folded_regions.into_iter().peekable();
    let mut old = 0;
    while old < ops.len() {
        new_indices[old] = new_ops.len() as u32;
        if let Some((rstart, _rend, _)) = region_iter.peek() {
            if *rstart == old {
                let (_, rend, replacement) = region_iter.next().unwrap();
                new_ops.push(replacement);
                // All old indices inside [rstart, rend) collapse onto the
                // single inserted op. We don't expect any branch targets to
                // land inside (we built `is_target` to prevent that), but if
                // one does (from a chain body PC we missed), point it at the
                // ReplicatedBody op.
                for inside in (old + 1)..rend {
                    new_indices[inside] = new_ops.len() as u32 - 1;
                }
                old = rend;
                continue;
            }
        }
        new_ops.push(std::mem::replace(&mut ops[old], Op::LoadLit { dst: 0, val: 0 }));
        old += 1;
    }
    new_indices[ops.len()] = new_ops.len() as u32;
    let new_end = new_ops.len() as u32;

    // Remap all branch/jump/dispatch targets in the rewritten array.
    for op in new_ops.iter_mut() {
        match op {
            Op::BranchIfZero { target, .. }
            | Op::BranchIfNotEqLit { target, .. }
            | Op::LoadStateAndBranchIfNotEqLit { target, .. }
            | Op::BranchIfNotEqLit2 { target, .. }
            | Op::Jump { target } => {
                let o = *target as usize;
                *target = if o < new_indices.len() { new_indices[o] } else { new_end };
            }
            Op::DispatchChain { miss_target, .. } => {
                let o = *miss_target as usize;
                *miss_target = if o < new_indices.len() { new_indices[o] } else { new_end };
            }
            Op::Dispatch { fallback_target, .. } => {
                let o = *fallback_target as usize;
                *fallback_target = if o < new_indices.len() { new_indices[o] } else { new_end };
            }
            Op::MemoryFill { exit_target, .. } | Op::MemoryCopy { exit_target, .. } => {
                let o = *exit_target as usize;
                *exit_target = if o < new_indices.len() { new_indices[o] } else { new_end };
            }
            _ => {}
        }
    }

    // Remap chain-table body PCs and flat-table targets that point into this
    // array.
    for &cid in &referenced_chain_ids {
        for (_v, body_pc) in chain_tables[cid as usize].entries.iter_mut() {
            let o = *body_pc as usize;
            *body_pc = if o < new_indices.len() { new_indices[o] } else { new_end };
        }
        if let Some(ref mut flat) = chain_tables[cid as usize].flat_table {
            for t in flat.targets.iter_mut() {
                if *t == u32::MAX { continue; }
                let o = *t as usize;
                *t = if o < new_indices.len() { new_indices[o] } else { new_end };
            }
        }
    }

    *ops = new_ops;
    n_folded
}

fn mark_target(is_target: &mut [bool], target: u32) {
    let t = target as usize;
    if t < is_target.len() {
        is_target[t] = true;
    }
}

/// Translate a classified BodyTemplate into the runtime Op::ReplicatedBody.
/// `region` is the original op slice the template was classified from; we
/// take the first `period` ops as the body templates.
fn build_replicated_body_op(
    bt: &crate::pattern::replicated_body::BodyTemplate,
    region: &[Op],
) -> Op {
    use crate::pattern::replicated_body::OperandClass;
    let body: Vec<Op> = region[..bt.period].to_vec();

    // For each body op, build the strides list — operand_index, stride for
    // every Linear-classified operand. Constant operands are dropped.
    let strides: Vec<Box<[(u8, i32)]>> = bt.ops.iter().map(|op_t| {
        let v: Vec<(u8, i32)> = op_t.operands.iter().enumerate().filter_map(|(i, c)| {
            match c {
                OperandClass::Constant(_) => None,
                OperandClass::Linear { stride, .. } => Some((i as u8, *stride as i32)),
            }
        }).collect();
        v.into_boxed_slice()
    }).collect();

    Op::ReplicatedBody {
        body: body.into_boxed_slice(),
        strides: strides.into_boxed_slice(),
        reps: bt.reps as u32,
    }
}

// ---------------------------------------------------------------------------
// Dispatch-chain recognition — collapse runs of same-slot BranchIfNotEqLit
// ---------------------------------------------------------------------------

/// Detect runs of `BranchIfNotEqLit` ops all testing the same slot against
/// different literals, chained such that each branch's miss target points
/// directly to the next branch. Replace the first op with `DispatchChain`
/// which does a single HashMap lookup and jumps to the matching body (or
/// to the chain-end miss target).
///
/// Safety: intermediate `BranchIfNotEqLit` ops are left in place (as dead
/// code) so any external jumps that land on them still work correctly.
fn build_dispatch_chains(program: &mut CompiledProgram) -> usize {
    let mut total = 0;
    // Main ops
    total += chains_in_ops(&mut program.ops, &mut program.chain_tables);
    // Dispatch table entries and fallbacks
    for table in &mut program.dispatch_tables {
        for (_k, (entry_ops, _s)) in &mut table.entries {
            total += chains_in_ops(entry_ops, &mut program.chain_tables);
        }
        total += chains_in_ops(&mut table.fallback_ops, &mut program.chain_tables);
    }
    // Broadcast value ops and spillovers
    for bw in &mut program.broadcast_writes {
        total += chains_in_ops(&mut bw.value_ops, &mut program.chain_tables);
        if let Some(ref mut sp) = bw.spillover {
            for (_k, (spill_ops, _s)) in &mut sp.entries {
                total += chains_in_ops(spill_ops, &mut program.chain_tables);
            }
        }
    }
    total
}

/// Minimum chain length to convert via the HashMap path. Below this, linear
/// BranchIfNotEqLit is probably faster than a HashMap lookup (which costs
/// ~30ns vs ~5ns/op).
const MIN_CHAIN_LEN: usize = 6;

/// Minimum chain length to convert when the dense flat-array path applies.
/// A flat-array lookup costs about as much as a single BranchIfNotEqLit, so
/// replacing even a two-probe chain saves an interpreter dispatch whenever
/// the first probe misses.
const MIN_FLAT_CHAIN_LEN: usize = 2;

fn chains_in_ops(ops: &mut [Op], chain_tables: &mut Vec<DispatchChainTable>) -> usize {
    let mut built = 0;
    let len = ops.len();
    let mut i = 0;

    while i < len {
        // Is this a potential chain start?
        let (key_slot, first_val, first_target) = match &ops[i] {
            Op::BranchIfNotEqLit { a, val, target } => (*a, *val, *target),
            _ => { i += 1; continue; }
        };

        // Walk the chain: at each step, if the previous target points at another
        // BranchIfNotEqLit on the same slot, extend; else that's the miss target.
        let mut entries: Vec<(i32, u32)> = vec![(first_val, (i as u32) + 1)];
        let mut miss = first_target;
        let mut last_branch_pc = i;

        loop {
            let next_pc = miss as usize;
            if next_pc >= len { break; }
            match &ops[next_pc] {
                Op::BranchIfNotEqLit { a, val, target } if *a == key_slot => {
                    entries.push((*val, (next_pc as u32) + 1));
                    miss = *target;
                    last_branch_pc = next_pc;
                }
                _ => break,
            }
        }

        if entries.len() >= MIN_FLAT_CHAIN_LEN {
            // Build the table.
            let mut table = DispatchChainTable::default();
            for (v, pc) in &entries {
                // If there are duplicate keys (shouldn't normally happen), first wins.
                table.entries.entry(*v).or_insert(*pc);
            }
            // Build a dense-array fast path when the key range is small enough.
            // Density threshold: range / count <= 2 (i.e., at least half the
            // indices are real entries, rest are misses).
            if !table.entries.is_empty() {
                let mut min_k = i32::MAX;
                let mut max_k = i32::MIN;
                for &k in table.entries.keys() {
                    if k < min_k { min_k = k; }
                    if k > max_k { max_k = k; }
                }
                let range = (max_k as i64) - (min_k as i64) + 1;
                let count = table.entries.len() as i64;
                // Cap absolute range to avoid huge sparse tables for outlier keys.
                if range <= 256 && range <= count * 3 {
                    let mut targets = vec![u32::MAX; range as usize];
                    for (&k, &pc) in &table.entries {
                        targets[(k - min_k) as usize] = pc;
                    }
                    table.flat_table = Some(FlatChainTable { base: min_k, targets });
                }
            }
            // Short chains only pay off through the flat-array path; without
            // it, linear probes beat a HashMap lookup, so leave them alone.
            if table.flat_table.is_some() || entries.len() >= MIN_CHAIN_LEN {
                let chain_id = chain_tables.len() as u32;
                chain_tables.push(table);
                ops[i] = Op::DispatchChain { a: key_slot, chain_id, miss_target: miss };
                built += 1;
                // Advance past the whole chain so we don't nest chains.
                i = last_branch_pc + 1;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    built
}

// ---------------------------------------------------------------------------
// Slot compaction — register allocation post-pass
// ---------------------------------------------------------------------------

/// Compact slot allocation across the entire compiled program.
///
/// The compiler allocates slots monotonically (SSA-style), producing one slot
/// per temporary. This pass renumbers slots so that dead temporaries are reused,
/// dramatically reducing `slot_count` and the per-tick memset cost.
///
/// Dispatch table entries and broadcast write sub-ops are compacted separately:
/// since only one entry executes per tick, all entries in a table can share the
/// same slot range.
fn compact_slots(program: &mut CompiledProgram) {
    #[cfg(not(target_arch = "wasm32"))]
    let compact_start = web_time::Instant::now();
    let before = program.slot_count;

    // Phase 1: compact main op stream
    let main_pinned = collect_main_pinned(program);
    let mut alloc = SlotAllocator::new();

    // Pre-assign pinned slots so they get stable new indices
    for &s in &main_pinned {
        alloc.assign(s);
    }

    // Compute liveness for main ops (includes dispatch sub-op references)
    let (_last_use, dying_at) = compute_liveness(&program.ops, &program.dispatch_tables);
    let mut slot_map: HashMap<Slot, Slot> = HashMap::new();

    // Map pinned slots first
    for &s in &main_pinned {
        slot_map.insert(s, alloc.get(s));
    }

    // Walk ops, mapping slots and freeing dead ones
    let pinned_set: std::collections::HashSet<Slot> = main_pinned.iter().copied().collect();
    for i in 0..program.ops.len() {
        map_op_slots(&mut program.ops[i], &mut slot_map, &mut alloc);
        // Free slots that die at this op (O(1) amortized via reverse index)
        if let Some(dying) = dying_at.get(&i) {
            for &orig_slot in dying {
                if !pinned_set.contains(&orig_slot) {
                    if let Some(&mapped) = slot_map.get(&orig_slot) {
                        alloc.free(mapped);
                    }
                }
            }
        }
    }

    let main_high = alloc.high_water;

    // Remap writeback
    for entry in &mut program.writeback {
        entry.0 = slot_map[&entry.0];
    }

    // Remap property_slots
    for val in program.property_slots.values_mut() {
        *val = slot_map[val];
    }

    // Debug: check for properties incorrectly sharing a physical slot.
    // Recompute liveness here so we can report intervals.
    {
        let (liveness, _) = compute_liveness(&program.ops, &program.dispatch_tables);
        let mut slot_to_names: std::collections::HashMap<Slot, Vec<String>> = std::collections::HashMap::new();
        for (name, &slot) in &program.property_slots {
            slot_to_names.entry(slot).or_default().push(name.clone());
        }
        for (phys_slot, names) in &slot_to_names {
            if names.len() > 1 {
                // For each original slot that maps to this physical slot, report its last-use
                let info: Vec<String> = slot_map.iter()
                    .filter(|(_, &v)| v == *phys_slot)
                    .map(|(&orig, _)| {
                        let lu = liveness.get(&orig).copied().unwrap_or(9999);
                        format!("orig={} last_use_op={}", orig, lu)
                    })
                    .collect();
                log::warn!("[slot-compaction] physical slot {} shared by {:?} — orig slots: {:?}", phys_slot, names, info);
            }
        }
    }

    // Reusable scratch buffers — avoids repeated HashMap allocations
    let mut scratch = SubOpScratch::new();

    // Phase 2: compact dispatch table entries with nesting-depth awareness.
    //
    // Dispatch entries overlay from main_high (since only one entry executes per
    // dispatch op). BUT if an entry contains a nested Dispatch op, the inner
    // dispatch's entries also overlay from main_high — clobbering the outer
    // entry's local slots.
    //
    // Fix: compute nesting depth for each table. Tables at depth 0 (leaf — no
    // nested Dispatch ops) overlay from main_high. Tables at depth N overlay
    // from a progressively higher base, so each nesting level gets its own
    // non-overlapping slot range.

    let num_tables = program.dispatch_tables.len();
    let mut refs_tables: Vec<Vec<usize>> = Vec::with_capacity(num_tables);
    for table in &program.dispatch_tables {
        let mut refs = Vec::new();
        let collect_refs = |ops: &[Op], out: &mut Vec<usize>| {
            for op in ops {
                if let Op::Dispatch { table_id, .. } = op {
                    out.push(*table_id as usize);
                }
            }
        };
        let mut entry_refs = Vec::new();
        collect_refs(&table.fallback_ops, &mut entry_refs);
        for (entry_ops, _) in table.entries.values() {
            collect_refs(entry_ops, &mut entry_refs);
        }
        refs.extend(entry_refs);
        refs_tables.push(refs);
    }

    // Compute depth via iterative fixed-point (handles cycles gracefully)
    let mut depth: Vec<u32> = vec![0; num_tables];
    for _ in 0..num_tables {
        let mut changed = false;
        for i in 0..num_tables {
            for &child in &refs_tables[i] {
                if child < num_tables {
                    let new_depth = depth[child] + 1;
                    if new_depth > depth[i] {
                        depth[i] = new_depth;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    let max_depth = depth.iter().copied().max().unwrap_or(0);

    // Compact layer by layer
    let mut layer_base = main_high;
    for d in 0..=max_depth {
        let mut layer_high: Slot = layer_base;
        for (idx, table) in program.dispatch_tables.iter_mut().enumerate() {
            if depth[idx] != d {
                continue;
            }
            let mut table_high: Slot = 0;
            let fb_high = compact_sub_ops(
                &mut table.fallback_ops,
                &mut table.fallback_slot,
                layer_base,
                &slot_map,
                &mut scratch,
            );
            table_high = table_high.max(fb_high);
            for (entry_ops, result_slot) in table.entries.values_mut() {
                let entry_high = compact_sub_ops(entry_ops, result_slot, layer_base, &slot_map, &mut scratch);
                table_high = table_high.max(entry_high);
            }
            layer_high = layer_high.max(table_high);
            alloc.high_water = alloc.high_water.max(table_high);
        }
        layer_base = layer_high;
    }

    // Phase 3: compact broadcast write sub-ops.
    // Broadcast writes execute after the main ops and all dispatches have
    // completed, but their value_ops run `exec_ops` which can trigger nested
    // Dispatch ops. Use layer_base (above all dispatch layers) to avoid
    // colliding with any dispatch entry slots.
    for bw in &mut program.broadcast_writes {
        // dest_slot is in main scope — already mapped
        bw.dest_slot = slot_map.get(&bw.dest_slot).copied().unwrap_or(bw.dest_slot);
        // gate_slot is also main-scope (compiled via compile_var with no dispatch)
        if let Some(gs) = bw.gate_slot {
            bw.gate_slot = Some(slot_map.get(&gs).copied().unwrap_or(gs));
        }

        // value_ops get their own compact range starting from layer_base
        let bw_high = compact_sub_ops(&mut bw.value_ops, &mut bw.value_slot, layer_base, &slot_map, &mut scratch);
        alloc.high_water = alloc.high_water.max(bw_high);

        // Spillover entries
        if let Some(ref mut spillover) = bw.spillover {
            spillover.guard_slot = slot_map
                .get(&spillover.guard_slot)
                .copied()
                .unwrap_or(spillover.guard_slot);
            for (spill_ops, spill_slot) in spillover.entries.values_mut() {
                let sp_high = compact_sub_ops(spill_ops, spill_slot, layer_base, &slot_map, &mut scratch);
                alloc.high_water = alloc.high_water.max(sp_high);
            }
        }
    }

    // Remap packed-broadcast write ports. gate/addr/val are main-scope
    // property slots — they were either pre-pinned via property_slots or via
    // collect_main_pinned, and slot_map covers them. unwrap_or(orig) is a
    // safety belt: if for some reason the slot wasn't seen, leave it (the
    // out-of-range read at runtime will panic loudly rather than silently
    // splice from the wrong slot).
    for port in &mut program.packed_broadcast_writes {
        port.gate_slot = slot_map.get(&port.gate_slot).copied().unwrap_or(port.gate_slot);
        port.addr_slot = slot_map.get(&port.addr_slot).copied().unwrap_or(port.addr_slot);
        port.val_slot = slot_map.get(&port.val_slot).copied().unwrap_or(port.val_slot);
        port.width_slot = slot_map.get(&port.width_slot).copied().unwrap_or(port.width_slot);
    }

    program.slot_count = alloc.high_water;

    #[cfg(not(target_arch = "wasm32"))]
    log::info!(
        "Slot compaction: {} → {} slots ({:.1}% reduction, {:.2}s)",
        before,
        program.slot_count,
        (1.0 - program.slot_count as f64 / before.max(1) as f64) * 100.0,
        compact_start.elapsed().as_secs_f64(),
    );
    #[cfg(target_arch = "wasm32")]
    log::info!(
        "Slot compaction: {} → {} slots ({:.1}% reduction)",
        before,
        program.slot_count,
        (1.0 - program.slot_count as f64 / before.max(1) as f64) * 100.0,
    );
}

/// Collect slots that are "pinned" — referenced outside the main op stream
/// (writeback, property_slots, broadcast dest/guard slots).
fn collect_main_pinned(program: &CompiledProgram) -> Vec<Slot> {
    let mut pinned = Vec::new();
    for &(slot, _) in &program.writeback {
        pinned.push(slot);
    }
    for &slot in program.property_slots.values() {
        pinned.push(slot);
    }
    for bw in &program.broadcast_writes {
        pinned.push(bw.dest_slot);
        if let Some(gs) = bw.gate_slot {
            pinned.push(gs);
        }
        if let Some(ref spillover) = bw.spillover {
            pinned.push(spillover.guard_slot);
        }
    }
    // Packed broadcast write ports: gate/addr/val slots are read at writeback
    // time after the main op stream completes, so they must survive compaction
    // even though no main op references them by index. They should already be
    // pinned via property_slots above (the recogniser only fires on properties
    // the assignment loop registered), but listing them explicitly is cheap
    // insurance.
    for port in &program.packed_broadcast_writes {
        pinned.push(port.gate_slot);
        pinned.push(port.addr_slot);
        pinned.push(port.val_slot);
        pinned.push(port.width_slot);
    }
    pinned.sort_unstable();
    pinned.dedup();
    pinned
}

/// Compute the last op index at which each slot is read or written,
/// and build a reverse index from op index → slots that die after that op.
///
/// Dispatch sub-ops (entry results, fallback results, sub-op reads) are
/// treated as reads at the Dispatch op's index, so parameter slots that
/// are only referenced inside dispatch entries stay alive until the
/// Dispatch executes.
fn compute_liveness(
    ops: &[Op],
    dispatch_tables: &[CompiledDispatchTable],
) -> (HashMap<Slot, usize>, HashMap<usize, Vec<Slot>>) {
    let mut last_use: HashMap<Slot, usize> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        for s in op_slots_read(op).into_iter().chain(op_dst(op)) {
            last_use.insert(s, i);
        }
        // For Dispatch ops, also mark slots referenced by the table's
        // sub-ops and result slots as live at this index.
        if let Op::Dispatch { table_id, .. } = op {
            if let Some(table) = dispatch_tables.get(*table_id as usize) {
                // Fallback
                last_use.entry(table.fallback_slot).and_modify(|v| *v = (*v).max(i)).or_insert(i);
                for s in table.fallback_ops.iter().flat_map(op_slots_read) {
                    last_use.entry(s).and_modify(|v| *v = (*v).max(i)).or_insert(i);
                }
                // Entries
                for (entry_ops, result_slot) in table.entries.values() {
                    last_use.entry(*result_slot).and_modify(|v| *v = (*v).max(i)).or_insert(i);
                    for s in entry_ops.iter().flat_map(op_slots_read) {
                        last_use.entry(s).and_modify(|v| *v = (*v).max(i)).or_insert(i);
                    }
                }
            }
        }
    }
    // Build reverse: op_index → [slots dying here]
    let mut dying_at: HashMap<usize, Vec<Slot>> = HashMap::new();
    for (&slot, &idx) in &last_use {
        dying_at.entry(idx).or_default().push(slot);
    }
    (last_use, dying_at)
}

/// Compact a sub-op stream (dispatch entry, broadcast value, spillover).
/// These get a fresh allocator starting from `base_slot`, and return the
/// high-water mark.
///
/// Reusable scratch buffers are passed in to avoid per-call allocation overhead
/// (critical when called 40M+ times for large dispatch tables).
fn compact_sub_ops(
    ops: &mut [Op],
    result_slot: &mut Slot,
    base_slot: Slot,
    parent_slot_map: &HashMap<Slot, Slot>,
    scratch: &mut SubOpScratch,
) -> Slot {
    if ops.is_empty() {
        if let Some(&mapped) = parent_slot_map.get(result_slot) {
            *result_slot = mapped;
        }
        return base_slot;
    }

    // Ultra-fast path: single LoadLit (the overwhelming majority of dispatch entries).
    // No parent slots, no HashMap lookups needed — just renumber the destination.
    if ops.len() == 1 {
        if let Op::LoadLit { dst, .. } = &mut ops[0] {
            let orig = *dst;
            *dst = base_slot;
            // result_slot is the same as the LoadLit's dst
            if *result_slot == orig {
                *result_slot = base_slot;
            } else if let Some(&mapped) = parent_slot_map.get(result_slot) {
                *result_slot = mapped;
            }
            return base_slot + 1;
        }
    }

    // Fast path for tiny sub-ops (1-2 ops): no liveness analysis needed.
    // Just remap parent-scope slots and assign fresh locals sequentially.
    if ops.len() <= 2 {
        return compact_sub_ops_tiny(ops, result_slot, base_slot, parent_slot_map, scratch);
    }

    // General path: full liveness analysis with reusable scratch buffers
    scratch.clear();
    let mut alloc = SlotAllocator::with_base(base_slot);

    // Compute liveness into scratch vectors
    compute_liveness_into(ops, &mut scratch.last_use, &mut scratch.dying_at);

    // Pre-assign the result slot (if not already mapped from parent)
    let orig_result = *result_slot;
    if !parent_slot_map.contains_key(&orig_result) {
        alloc.assign(orig_result);
        scratch.local_map.insert(orig_result, alloc.get(orig_result));
    }

    for (i, op) in ops.iter_mut().enumerate() {
        seed_from_parent(op, &mut scratch.local_map, parent_slot_map);
        map_op_slots(op, &mut scratch.local_map, &mut alloc);
        if let Some(dying) = scratch.dying_at.get(&i) {
            for &orig_slot in dying {
                if orig_slot != orig_result && !parent_slot_map.contains_key(&orig_slot) {
                    if let Some(&mapped) = scratch.local_map.get(&orig_slot) {
                        alloc.free(mapped);
                    }
                }
            }
        }
    }

    *result_slot = scratch
        .local_map
        .get(&orig_result)
        .or_else(|| parent_slot_map.get(&orig_result))
        .copied()
        .unwrap_or(orig_result);
    alloc.high_water
}

/// Fast-path compaction for sub-op streams with 1-2 ops.
///
/// With so few ops, liveness analysis is pointless — no slot can die before
/// the end. We just remap parent-scope slots and assign sequential new slots
/// for any locals.
fn compact_sub_ops_tiny(
    ops: &mut [Op],
    result_slot: &mut Slot,
    base_slot: Slot,
    parent_slot_map: &HashMap<Slot, Slot>,
    scratch: &mut SubOpScratch,
) -> Slot {
    scratch.local_map.clear();
    let mut alloc = SlotAllocator::with_base(base_slot);

    let orig_result = *result_slot;
    if !parent_slot_map.contains_key(&orig_result) {
        alloc.assign(orig_result);
        scratch.local_map.insert(orig_result, alloc.get(orig_result));
    }

    for op in ops.iter_mut() {
        seed_from_parent(op, &mut scratch.local_map, parent_slot_map);
        map_op_slots(op, &mut scratch.local_map, &mut alloc);
    }

    *result_slot = scratch
        .local_map
        .get(&orig_result)
        .or_else(|| parent_slot_map.get(&orig_result))
        .copied()
        .unwrap_or(orig_result);
    alloc.high_water
}

/// Reusable scratch buffers for `compact_sub_ops` to avoid per-call allocation.
struct SubOpScratch {
    local_map: HashMap<Slot, Slot>,
    last_use: HashMap<Slot, usize>,
    dying_at: HashMap<usize, Vec<Slot>>,
}

impl SubOpScratch {
    fn new() -> Self {
        Self {
            local_map: HashMap::new(),
            last_use: HashMap::new(),
            dying_at: HashMap::new(),
        }
    }

    fn clear(&mut self) {
        self.local_map.clear();
        self.last_use.clear();
        self.dying_at.clear();
    }
}

/// Compute liveness into caller-provided maps (avoids allocation per call).
fn compute_liveness_into(
    ops: &[Op],
    last_use: &mut HashMap<Slot, usize>,
    dying_at: &mut HashMap<usize, Vec<Slot>>,
) {
    for (i, op) in ops.iter().enumerate() {
        for s in op_slots_read(op).into_iter().chain(op_dst(op)) {
            last_use.insert(s, i);
        }
    }
    for (&slot, &idx) in last_use.iter() {
        dying_at.entry(idx).or_default().push(slot);
    }
}

/// Simple slot allocator with a free list.
struct SlotAllocator {
    next: Slot,
    high_water: Slot,
    free_list: Vec<Slot>,
    /// Tracks original→new assignments for pre-assigned (pinned) slots.
    assigned: HashMap<Slot, Slot>,
}

impl SlotAllocator {
    fn new() -> Self {
        Self {
            next: 0,
            high_water: 0,
            free_list: Vec::new(),
            assigned: HashMap::new(),
        }
    }

    fn with_base(base: Slot) -> Self {
        Self {
            next: base,
            high_water: base,
            free_list: Vec::new(),
            assigned: HashMap::new(),
        }
    }

    /// Pre-assign a slot (for pinned slots). Call before mapping.
    fn assign(&mut self, original: Slot) {
        if self.assigned.contains_key(&original) {
            return;
        }
        let new = self.alloc();
        self.assigned.insert(original, new);
    }

    /// Get the mapped slot for a pre-assigned original.
    fn get(&self, original: Slot) -> Slot {
        self.assigned[&original]
    }

    /// Allocate a slot — reuse from free list or bump.
    fn alloc(&mut self) -> Slot {
        if let Some(s) = self.free_list.pop() {
            s
        } else {
            let s = self.next;
            self.next += 1;
            self.high_water = self.high_water.max(self.next);
            s
        }
    }

    /// Return a slot to the free list.
    fn free(&mut self, slot: Slot) {
        self.free_list.push(slot);
    }

    /// Get or allocate a mapping for an original slot.
    fn get_or_alloc(&mut self, original: Slot, slot_map: &mut HashMap<Slot, Slot>) -> Slot {
        if let Some(&mapped) = slot_map.get(&original) {
            mapped
        } else if let Some(&mapped) = self.assigned.get(&original) {
            slot_map.insert(original, mapped);
            mapped
        } else {
            let mapped = self.alloc();
            slot_map.insert(original, mapped);
            mapped
        }
    }
}

/// Get all slots read by an op (not including dst).
fn op_slots_read(op: &Op) -> Vec<Slot> {
    match op {
        Op::LoadLit { .. } => vec![],
        Op::LoadSlot { src, .. } => vec![*src],
        Op::LoadState { .. } => vec![],
        Op::LoadMem { addr_slot, .. } => vec![*addr_slot],
        Op::LoadMem16 { addr_slot, .. } => vec![*addr_slot],
        Op::LoadPackedByte { key_slot, .. } => vec![*key_slot],
        Op::Add { a, b, .. }
        | Op::Sub { a, b, .. }
        | Op::Mul { a, b, .. }
        | Op::Div { a, b, .. }
        | Op::Mod { a, b, .. }
        | Op::And { a, b, .. }
        | Op::Shr { a, b, .. }
        | Op::Shl { a, b, .. } => vec![*a, *b],
        Op::AddLit { a, .. } | Op::SubLit { a, .. } | Op::MulLit { a, .. }
        | Op::AndLit { a, .. } | Op::ShrLit { a, .. } | Op::ShlLit { a, .. }
        | Op::ModLit { a, .. } => vec![*a],
        Op::Neg { src, .. }
        | Op::Abs { src, .. }
        | Op::Sign { src, .. }
        | Op::Floor { src, .. } => {
            vec![*src]
        }
        Op::Pow { base, exp, .. } => vec![*base, *exp],
        Op::Min { args, .. } | Op::Max { args, .. } => args.clone(),
        Op::Clamp { min, val, max, .. } => vec![*min, *val, *max],
        Op::Round { val, interval, .. } => vec![*val, *interval],
        Op::Bit { val, idx, .. } => vec![*val, *idx],
        Op::BitAnd16 { a, b, .. } | Op::BitOr16 { a, b, .. } | Op::BitXor16 { a, b, .. } => vec![*a, *b],
        Op::BitNot16 { a, .. } => vec![*a],
        Op::CmpEq { a, b, .. } => vec![*a, *b],
        Op::BranchIfZero { cond, .. } => vec![*cond],
        Op::BranchIfNotEqLit { a, .. } => vec![*a],
        Op::LoadStateAndBranchIfNotEqLit { .. } => vec![],
        Op::BranchIfNotEqLit2 { a1, a2, .. } => vec![*a1, *a2],
        Op::Jump { .. } => vec![],
        Op::DispatchChain { a, .. } => vec![*a],
        Op::Dispatch { key, .. } => vec![*key],
        Op::DispatchFlatArray { key, .. } => vec![*key],
        Op::StoreState { src, .. } => vec![*src],
        Op::StoreMem { addr_slot, src, .. } => vec![*addr_slot, *src],
        // Call reads the arg slots at the call site; param/local slots
        // inside body_ops belong to the body's own scope and are listed
        // by the compaction walk when it recurses into body_ops.
        Op::Call { arg_slots, .. } => arg_slots.clone(),
        Op::MemoryFill { dst_slot, val_slot, count_slot, .. } => vec![*dst_slot, *val_slot, *count_slot],
        Op::MemoryCopy { src_slot, dst_slot, count_slot, .. } => vec![*src_slot, *dst_slot, *count_slot],
        // ReplicatedBody is inserted by the recogniser AFTER slot compaction
        // and peephole passes; this utility should never see it.
        Op::ReplicatedBody { .. } => unreachable!(
            "op_slots_read called on ReplicatedBody — recogniser ran before slot compaction"
        ),
    }
}

/// Get the destination slot of an op, if any.
fn op_dst(op: &Op) -> Option<Slot> {
    match op {
        Op::LoadLit { dst, .. }
        | Op::LoadSlot { dst, .. }
        | Op::LoadState { dst, .. }
        | Op::LoadMem { dst, .. }
        | Op::LoadMem16 { dst, .. }
        | Op::LoadPackedByte { dst, .. }
        | Op::Add { dst, .. }
        | Op::Sub { dst, .. }
        | Op::Mul { dst, .. }
        | Op::Div { dst, .. }
        | Op::Mod { dst, .. }
        | Op::Neg { dst, .. }
        | Op::Abs { dst, .. }
        | Op::Sign { dst, .. }
        | Op::Pow { dst, .. }
        | Op::Min { dst, .. }
        | Op::Max { dst, .. }
        | Op::Clamp { dst, .. }
        | Op::Round { dst, .. }
        | Op::Floor { dst, .. }
        | Op::And { dst, .. }
        | Op::Shr { dst, .. }
        | Op::Shl { dst, .. }
        | Op::Bit { dst, .. }
        | Op::BitAnd16 { dst, .. }
        | Op::BitOr16 { dst, .. }
        | Op::BitXor16 { dst, .. }
        | Op::BitNot16 { dst, .. }
        | Op::CmpEq { dst, .. }
        | Op::Dispatch { dst, .. }
        | Op::DispatchFlatArray { dst, .. }
        | Op::Call { dst, .. } => Some(*dst),
        Op::AddLit { dst, .. } | Op::SubLit { dst, .. } | Op::MulLit { dst, .. }
        | Op::AndLit { dst, .. } | Op::ShrLit { dst, .. } | Op::ShlLit { dst, .. }
        | Op::ModLit { dst, .. } => Some(*dst),
        Op::LoadStateAndBranchIfNotEqLit { dst, .. } => Some(*dst),
        Op::BranchIfZero { .. } | Op::BranchIfNotEqLit { .. } | Op::BranchIfNotEqLit2 { .. } | Op::Jump { .. } | Op::DispatchChain { .. } | Op::StoreState { .. } | Op::StoreMem { .. } | Op::MemoryFill { .. } | Op::MemoryCopy { .. } => {
            None
        }
        Op::ReplicatedBody { .. } => unreachable!(
            "op_dst called on ReplicatedBody — recogniser ran before slot compaction"
        ),
    }
}

/// Remap all slot references in an op, allocating new slots as needed.
fn map_op_slots(op: &mut Op, slot_map: &mut HashMap<Slot, Slot>, alloc: &mut SlotAllocator) {
    match op {
        Op::LoadLit { dst, .. } => {
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::LoadSlot { dst, src } => {
            *src = alloc.get_or_alloc(*src, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::LoadState { dst, .. } => {
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::LoadMem { dst, addr_slot } => {
            *addr_slot = alloc.get_or_alloc(*addr_slot, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::LoadMem16 { dst, addr_slot } => {
            *addr_slot = alloc.get_or_alloc(*addr_slot, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::LoadPackedByte { dst, key_slot, .. } => {
            *key_slot = alloc.get_or_alloc(*key_slot, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::Add { dst, a, b }
        | Op::Sub { dst, a, b }
        | Op::Mul { dst, a, b }
        | Op::Div { dst, a, b }
        | Op::Mod { dst, a, b }
        | Op::And { dst, a, b }
        | Op::Shr { dst, a, b }
        | Op::Shl { dst, a, b } => {
            *a = alloc.get_or_alloc(*a, slot_map);
            *b = alloc.get_or_alloc(*b, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::AddLit { dst, a, .. } | Op::SubLit { dst, a, .. } | Op::MulLit { dst, a, .. }
        | Op::AndLit { dst, a, .. } | Op::ShrLit { dst, a, .. } | Op::ShlLit { dst, a, .. }
        | Op::ModLit { dst, a, .. } => {
            *a = alloc.get_or_alloc(*a, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::Neg { dst, src }
        | Op::Abs { dst, src }
        | Op::Sign { dst, src }
        | Op::Floor { dst, src } => {
            *src = alloc.get_or_alloc(*src, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::Pow { dst, base, exp } => {
            *base = alloc.get_or_alloc(*base, slot_map);
            *exp = alloc.get_or_alloc(*exp, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::Min { dst, args } | Op::Max { dst, args } => {
            for a in args.iter_mut() {
                *a = alloc.get_or_alloc(*a, slot_map);
            }
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::Clamp { dst, min, val, max } => {
            *min = alloc.get_or_alloc(*min, slot_map);
            *val = alloc.get_or_alloc(*val, slot_map);
            *max = alloc.get_or_alloc(*max, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::Round {
            dst, val, interval, ..
        } => {
            *val = alloc.get_or_alloc(*val, slot_map);
            *interval = alloc.get_or_alloc(*interval, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::Bit { dst, val, idx } => {
            *val = alloc.get_or_alloc(*val, slot_map);
            *idx = alloc.get_or_alloc(*idx, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::BitAnd16 { dst, a, b } | Op::BitOr16 { dst, a, b } | Op::BitXor16 { dst, a, b } => {
            *a = alloc.get_or_alloc(*a, slot_map);
            *b = alloc.get_or_alloc(*b, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::BitNot16 { dst, a } => {
            *a = alloc.get_or_alloc(*a, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::CmpEq { dst, a, b } => {
            *a = alloc.get_or_alloc(*a, slot_map);
            *b = alloc.get_or_alloc(*b, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::BranchIfZero { cond, .. } => {
            *cond = alloc.get_or_alloc(*cond, slot_map);
        }
        Op::BranchIfNotEqLit { a, .. } => {
            *a = alloc.get_or_alloc(*a, slot_map);
        }
        Op::LoadStateAndBranchIfNotEqLit { dst, .. } => {
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::BranchIfNotEqLit2 { a1, a2, .. } => {
            *a1 = alloc.get_or_alloc(*a1, slot_map);
            *a2 = alloc.get_or_alloc(*a2, slot_map);
        }
        Op::DispatchChain { a, .. } => {
            *a = alloc.get_or_alloc(*a, slot_map);
        }
        Op::Jump { .. } => {}
        Op::Dispatch { dst, key, .. } => {
            *key = alloc.get_or_alloc(*key, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::DispatchFlatArray { dst, key, .. } => {
            *key = alloc.get_or_alloc(*key, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::StoreState { src, .. } => {
            *src = alloc.get_or_alloc(*src, slot_map);
        }
        Op::StoreMem { addr_slot, src } => {
            *addr_slot = alloc.get_or_alloc(*addr_slot, slot_map);
            *src = alloc.get_or_alloc(*src, slot_map);
        }
        Op::Call { dst, arg_slots, .. } => {
            for s in arg_slots.iter_mut() {
                *s = alloc.get_or_alloc(*s, slot_map);
            }
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::MemoryFill { dst_slot, val_slot, count_slot, .. } => {
            *dst_slot = alloc.get_or_alloc(*dst_slot, slot_map);
            *val_slot = alloc.get_or_alloc(*val_slot, slot_map);
            *count_slot = alloc.get_or_alloc(*count_slot, slot_map);
        }
        Op::MemoryCopy { src_slot, dst_slot, count_slot, .. } => {
            *src_slot = alloc.get_or_alloc(*src_slot, slot_map);
            *dst_slot = alloc.get_or_alloc(*dst_slot, slot_map);
            *count_slot = alloc.get_or_alloc(*count_slot, slot_map);
        }
        Op::ReplicatedBody { .. } => unreachable!(
            "map_op_slots called on ReplicatedBody — recogniser ran before slot compaction"
        ),
    }
}

/// Seed a local slot map from a parent map for **read-only** slots referenced by an op.
/// This avoids cloning the entire parent map — only slots actually used get copied.
///
/// IMPORTANT: We must NOT seed `dst` slots. A sub-op's `dst` is always a local
/// temporary that should get a fresh slot allocation. If we seed it from the parent
/// map, a local temp whose original slot number coincidentally matches a parent-scope
/// slot will be aliased to the parent slot — corrupting the parent's value when the
/// sub-op writes to it.
fn seed_from_parent(
    op: &Op,
    local_map: &mut HashMap<Slot, Slot>,
    parent_map: &HashMap<Slot, Slot>,
) {
    let mut seed = |s: Slot| {
        if !local_map.contains_key(&s) {
            if let Some(&mapped) = parent_map.get(&s) {
                local_map.insert(s, mapped);
            }
        }
    };
    match op {
        Op::LoadLit { .. } => {}
        Op::LoadSlot { src, .. } => { seed(*src); }
        Op::LoadState { .. } => {}
        Op::LoadMem { addr_slot, .. } | Op::LoadMem16 { addr_slot, .. } => { seed(*addr_slot); }
        Op::LoadPackedByte { key_slot, .. } => { seed(*key_slot); }
        Op::Add { a, b, .. } | Op::Sub { a, b, .. } | Op::Mul { a, b, .. }
        | Op::Div { a, b, .. } | Op::Mod { a, b, .. } | Op::And { a, b, .. }
        | Op::Shr { a, b, .. } | Op::Shl { a, b, .. } => { seed(*a); seed(*b); }
        Op::AddLit { a, .. } | Op::SubLit { a, .. } | Op::MulLit { a, .. }
        | Op::AndLit { a, .. } | Op::ShrLit { a, .. } | Op::ShlLit { a, .. }
        | Op::ModLit { a, .. } => { seed(*a); }
        Op::Neg { src, .. } | Op::Abs { src, .. } | Op::Sign { src, .. }
        | Op::Floor { src, .. } => { seed(*src); }
        Op::Pow { base, exp, .. } => { seed(*base); seed(*exp); }
        Op::Bit { val, idx, .. } => { seed(*val); seed(*idx); }
        Op::BitAnd16 { a, b, .. } | Op::BitOr16 { a, b, .. } | Op::BitXor16 { a, b, .. } => { seed(*a); seed(*b); }
        Op::BitNot16 { a, .. } => { seed(*a); }
        Op::CmpEq { a, b, .. } => { seed(*a); seed(*b); }
        Op::Min { args, .. } | Op::Max { args, .. } => { for a in args { seed(*a); } }
        Op::Clamp { min, val, max, .. } => { seed(*min); seed(*val); seed(*max); }
        Op::Round { val, interval, .. } => { seed(*val); seed(*interval); }
        Op::BranchIfZero { cond, .. } => { seed(*cond); }
        Op::BranchIfNotEqLit { a, .. } => { seed(*a); }
        Op::LoadStateAndBranchIfNotEqLit { .. } => {}
        Op::BranchIfNotEqLit2 { a1, a2, .. } => { seed(*a1); seed(*a2); }
        Op::DispatchChain { a, .. } => { seed(*a); }
        Op::Jump { .. } => {}
        Op::Dispatch { key, .. } => { seed(*key); }
        Op::DispatchFlatArray { key, .. } => { seed(*key); }
        Op::StoreState { src, .. } => { seed(*src); }
        Op::StoreMem { addr_slot, src } => { seed(*addr_slot); seed(*src); }
        Op::Call { arg_slots, .. } => { for s in arg_slots { seed(*s); } }
        Op::MemoryFill { dst_slot, val_slot, count_slot, .. } => { seed(*dst_slot); seed(*val_slot); seed(*count_slot); }
        Op::MemoryCopy { src_slot, dst_slot, count_slot, .. } => { seed(*src_slot); seed(*dst_slot); seed(*count_slot); }
        Op::ReplicatedBody { .. } => unreachable!(
            "seed_from_parent called on ReplicatedBody — recogniser ran before slot compaction"
        ),
    }
}

/// Compile a single broadcast write.
fn compile_broadcast_write(bw: &BroadcastWrite, compiler: &mut Compiler) -> CompiledBroadcastWrite {
    prof!("compile_broadcast_write");
    let _t0 = web_time::Instant::now();
    // Compile dest property resolution
    let dest_slot = compiler.compile_var(&bw.dest_property, None, &mut Vec::new());

    // Compile value expression
    let mut value_ops = Vec::new();
    let value_slot = compiler.compile_expr(&bw.value_expr, &mut value_ops);
    let _t1 = web_time::Instant::now();

    // Build address map: address → state address (for direct write_mem)
    // Optimised: broadcast entries are bare names like "m12345" or "AX".
    // Parse directly instead of allocating format!("--{name}") for each of 1M entries.
    let mut address_map: FxMap<i64, i32> =
        FxMap::with_capacity_and_hasher(bw.address_map.len(), Default::default());
    for (&addr, var_name) in &bw.address_map {
        let state_addr = if let Some(rest) = var_name.strip_prefix('m') {
            rest.parse::<i32>().ok()
        } else {
            // Register or other named property — fall back to full resolution
            property_to_address(var_name)
                .or_else(|| property_to_address(&format!("--{var_name}")))
        };
        if let Some(sa) = state_addr {
            address_map.insert(addr, sa);
        }
    }
    log::info!("[bw detail] {} address_map ({} entries): {:.2}s",
        bw.dest_property, bw.address_map.len(), _t1.elapsed().as_secs_f64());
    let _t2 = web_time::Instant::now();

    // Compile spillover
    let spillover = if !bw.spillover_map.is_empty() {
        bw.spillover_guard.as_ref().map(|guard| {
            let guard_slot = compiler.compile_var(guard, None, &mut Vec::new());
            let mut entries: FxMap<i64, (Vec<Op>, Slot)> = FxMap::default();
            for (&addr, (_var_name, val_expr)) in &bw.spillover_map {
                let mut spill_ops = Vec::new();
                let spill_slot = compiler.compile_expr(val_expr, &mut spill_ops);
                entries.insert(addr, (spill_ops, spill_slot));
            }
            CompiledSpillover {
                guard_slot,
                entries,
            }
        })
    } else {
        None
    };

    // Compile the outer gate property if this broadcast write was recognised
    // as sitting inside `style(--gate: 1): if(...)`. The executor checks
    // this slot once per tick and skips the whole broadcast on mismatch.
    let gate_slot = bw
        .gate_property
        .as_deref()
        .map(|g| compiler.compile_var(g, None, &mut Vec::new()));

    CompiledBroadcastWrite {
        dest_slot,
        value_ops,
        value_slot,
        address_map,
        spillover,
        gate_slot,
    }
}

// ---------------------------------------------------------------------------
// Executor — runs a CompiledProgram against State
// ---------------------------------------------------------------------------

/// Splice one byte through a packed-broadcast port at a guest byte address.
/// Reads the cell containing `byte_addr`, replaces the byte at `byte_addr`,
/// writes back, keeps the flat shadow + write log in sync.
///
/// `val` is masked to 0xFF by the caller (or is implicitly within range).
/// `byte_addr` is in guest memory — the cell index is `byte_addr / pack`.
/// If the cell isn't in the port's table the splice is silently dropped
/// (matches the v1 behaviour: ports cover every writable cell).
#[inline]
fn packed_splice_byte(
    port: &CompiledPackedBroadcastWrite,
    state: &mut State,
    byte_addr: i32,
    val: i32,
) {
    if byte_addr < 0 {
        return;
    }
    let pack = port.pack as i32;
    let cell_idx = (byte_addr / pack) as usize;
    let off = byte_addr % pack;
    if cell_idx >= port.cell_table.len() {
        return;
    }
    let cell_state_addr = port.cell_table[cell_idx];
    if cell_state_addr == 0 {
        return;
    }
    let prev = state.read_mem(cell_state_addr);
    let new_cell = if off == 0 {
        (prev & !0xFF) | val
    } else {
        // off == 1: replace high byte (bits 8..15)
        (prev & 0xFF) | (val << 8)
    };
    if new_cell != prev {
        // Write directly to state_vars — the writeback logic for
        // pack=2 cabinets logs single-byte guest writes (not whole-cell
        // i32 writes) so the affine projectors can detect byte-fill loops.
        // See the long comment in execute()'s previous version of this
        // splice for why this matters for pack=2 perf.
        let sidx = (-cell_state_addr - 1) as usize;
        if sidx < state.state_vars.len() {
            state.state_vars[sidx] = new_cell;
        }
        // Keep the flat shadow in step.
        let addr_u = byte_addr as usize;
        if addr_u < state.memory.len() {
            state.memory[addr_u] = val as u8;
        }
        if let Some(ref mut log) = state.write_log {
            log.push((byte_addr, val));
        }
    }
}

/// Splice an even-aligned 16-bit word through a packed-broadcast port.
/// Caller guarantees `byte_addr & 1 == 0` so both bytes land in the same
/// cell and the whole cell can be replaced in one read-modify-write.
///
/// `val` is a 16-bit value pre-masked to 0..=65535 by the caller. The
/// affine projectors expect *byte-level* write_log entries even for word
/// writes (so a STOSW loop's per-byte stride is visible), so we still
/// emit two write_log entries — but the cell update itself is one op.
#[inline]
fn packed_splice_word_aligned(
    port: &CompiledPackedBroadcastWrite,
    state: &mut State,
    byte_addr: i32,
    val: i32,
) {
    debug_assert!(byte_addr >= 0 && byte_addr & 1 == 0);
    let pack = port.pack as i32;
    let cell_idx = (byte_addr / pack) as usize;
    if cell_idx >= port.cell_table.len() {
        return;
    }
    let cell_state_addr = port.cell_table[cell_idx];
    if cell_state_addr == 0 {
        return;
    }
    let prev = state.read_mem(cell_state_addr);
    if val != prev {
        let sidx = (-cell_state_addr - 1) as usize;
        if sidx < state.state_vars.len() {
            state.state_vars[sidx] = val;
        }
        let addr_u = byte_addr as usize;
        if addr_u + 1 < state.memory.len() {
            state.memory[addr_u]     = (val & 0xFF) as u8;
            state.memory[addr_u + 1] = ((val >> 8) & 0xFF) as u8;
        }
        // Affine projectors lock on byte-stride patterns, so log byte
        // writes (matching what byte-mode would produce) rather than
        // logging a single (addr, word) entry.
        if let Some(ref mut log) = state.write_log {
            log.push((byte_addr, val & 0xFF));
            log.push((byte_addr + 1, (val >> 8) & 0xFF));
        }
    }
}

/// Execute a compiled program for one tick.
pub fn execute(program: &CompiledProgram, state: &mut State, slots: &mut Vec<i32>) {
    // Size slots appropriately; keep stale values between ticks. The compiler
    // guarantees slots are written before being read within a tick (every read
    // is dominated by a write on every control-flow path reaching it), so the
    // initial value doesn't matter.
    if slots.len() != program.slot_count as usize {
        slots.resize(program.slot_count as usize, 0);
    }

    // Op-adjacency profile: when enabled, dispatch through the profiling
    // variant. The check is one TLS Cell::get per tick (negligible). When
    // disabled, the production path runs unchanged with zero overhead.
    let op_profile_active = crate::pattern::op_profile::op_profile_is_enabled();

    if op_profile_active {
        // Reset prev-op tracker at tick boundaries — dispatched-op pairs
        // shouldn't span ticks (each tick is a logical sequence).
        crate::pattern::op_profile::op_profile_break_sequence();
        exec_ops_with_op_profile(&program.ops, &program.dispatch_tables, &program.chain_tables, &program.flat_dispatch_arrays, &program.functions, &program.packed_cell_tables, &program.packed_exception_tables, state, slots);
    } else {
        // Execute main ops
        exec_ops(&program.ops, &program.dispatch_tables, &program.chain_tables, &program.flat_dispatch_arrays, &program.functions, &program.packed_cell_tables, &program.packed_exception_tables, state, slots);
    }

    // Writeback: apply computed values to state
    for &(slot, addr) in &program.writeback {
        let value = slots[slot as usize];
        let old = state.read_mem(addr);
        if old != value {
            state.write_mem(addr, value);
        }
    }

    // Execute broadcast writes
    for bw in &program.broadcast_writes {
        // Gated broadcasts skip entirely when the gate is not 1. Mirrors
        // the interpreted path in eval.rs.
        if let Some(gate_slot) = bw.gate_slot {
            if slots[gate_slot as usize] != 1 {
                continue;
            }
        }
        let dest = slots[bw.dest_slot as usize];
        let dest_i64 = dest as i64;
        if bw.address_map.contains_key(&dest_i64) {
            if op_profile_active {
                exec_ops_with_op_profile(&bw.value_ops, &program.dispatch_tables, &program.chain_tables, &program.flat_dispatch_arrays, &program.functions, &program.packed_cell_tables, &program.packed_exception_tables, state, slots);
            } else {
                exec_ops(&bw.value_ops, &program.dispatch_tables, &program.chain_tables, &program.flat_dispatch_arrays, &program.functions, &program.packed_cell_tables, &program.packed_exception_tables, state, slots);
            }
            let value = slots[bw.value_slot as usize];
            state.write_mem(dest, value);
        }
        // Spillover
        if let Some(ref spillover) = bw.spillover {
            let guard = slots[spillover.guard_slot as usize];
            if guard == 1 {
                if let Some((ref spill_ops, spill_slot)) = spillover.entries.get(&dest_i64) {
                    if op_profile_active {
                        exec_ops_with_op_profile(spill_ops, &program.dispatch_tables, &program.chain_tables, &program.flat_dispatch_arrays, &program.functions, &program.packed_cell_tables, &program.packed_exception_tables, state, slots);
                    } else {
                        exec_ops(spill_ops, &program.dispatch_tables, &program.chain_tables, &program.flat_dispatch_arrays, &program.functions, &program.packed_cell_tables, &program.packed_exception_tables, state, slots);
                    }
                    let value = slots[*spill_slot as usize];
                    state.write_mem(dest + 1, value);
                }
            }
        }
    }

    // Packed broadcast writes: one port per CSS-DOS write slot. Replaces
    // the absorbed `--mc{N}: --applySlot(...)` chain. We iterate ports in
    // REVERSE order so slot 0 fires LAST — matching the CSS semantics where
    // slot 0 is the outermost --applySlot wrapper and wins same-cell
    // collisions over later slots.
    //
    // Width: each port reads a per-tick width slot (1 or 2). Width=1 splices
    // a single byte at byte_addr. Width=2 writes a 16-bit word; when
    // byte_addr is even the lo and hi halves both land in cell
    // `byte_addr/2` (single read-modify-write of the whole cell), when odd
    // they straddle (lo into b1 of cell N, hi into b0 of cell N+1).
    //
    // The aligned-width=2 path is hot — every PUSH/CALL/INT/IRQ frame and
    // every word `MOV [m], r16` to even-aligned memory hits it. Inlining
    // the splice and specialising the aligned case lets us avoid the
    // double read-modify-write the naive two-byte-splice version would do.
    for port in program.packed_broadcast_writes.iter().rev() {
        if slots[port.gate_slot as usize] != 1 {
            continue;
        }
        let byte_addr = slots[port.addr_slot as usize];
        if byte_addr < 0 {
            continue;
        }
        let val = slots[port.val_slot as usize];
        let width = slots[port.width_slot as usize];
        if width == 2 {
            if byte_addr & 1 == 0 {
                // Aligned word: single cell, single read-modify-write.
                // The whole cell becomes the 16-bit word value (masked to
                // u16 so a sign-extended val doesn't bleed into bit 16+).
                packed_splice_word_aligned(port, state, byte_addr, val & 0xFFFF);
            } else {
                // Straddle: lo half goes to cell N's high byte; hi half
                // goes to cell N+1's low byte. Two cells, two splices.
                packed_splice_byte(port, state, byte_addr, val & 0xFF);
                packed_splice_byte(port, state, byte_addr + 1, (val >> 8) & 0xFF);
            }
        } else {
            // Byte splice (width=1, the default).
            packed_splice_byte(port, state, byte_addr, val & 0xFF);
        }
    }

    // Runtime REP fast-forward: after the CSS has applied one iteration of a
    // REP string op, collapse the remaining iterations into one bulk memory
    // operation. See rep_fast_forward() for the detection rules.
    // Gated by CALCITE_REP_FASTFWD=0 env var for A/B debugging of
    // regressions; on by default.
    if rep_fastfwd_enabled() {
        rep_fast_forward(program, state, slots);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn rep_fastfwd_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        match std::env::var("CALCITE_REP_FASTFWD").as_deref() {
            Ok("0") | Ok("false") | Ok("off") => false,
            _ => true,
        }
    })
}

#[cfg(target_arch = "wasm32")]
fn rep_fastfwd_enabled() -> bool { true }

/// Post-tick descriptor-driven applier for self-loop opcodes ("REP"
/// in x86 terms; structurally: any opcode whose CSS dispatch loops on
/// its own slot with the cabinet's outer guard predicate still true).
/// Runs after the CSS has applied exactly one iteration.
///
/// Routing is purely descriptor-driven. There is **no** opcode-byte
/// table, no `matches!(opcode, 0xAA | ...)` test — the cabinet's
/// recogniser produced a `Vec<LoopDescriptor>` at load time, keyed by
/// dispatch-key value, and this function looks up the descriptor for
/// the current opcode and dispatches on its `BulkClass`.
///
/// Three outcomes:
/// - `Applied { iterations }` — applier mutated state, done.
/// - `PreconditionNotMet` — the cabinet's own outer guard predicate
///   evaluated false; CSS already produced the correct post-state and
///   calcite has nothing to do. Silent bail.
/// - `Unsupported(reason)` — recogniser produced a descriptor but the
///   applier refused. **Panic loudly**, because the per-iter CSS
///   fallback is 10-1000x slower per iteration and a long loop would
///   strand the user on a frozen screen with no error.
///
/// `BulkClass::PerIter` (the recogniser's "I can't bulk this") also
/// panics — same reason. The fix is to extend the recogniser or
/// applier, never to make the bail silent.
///
/// No env-var toggle. This is the only path.
fn rep_fast_forward(program: &CompiledProgram, state: &mut State, slots: &[i32]) {
    // Resolve a CSS property name to its current i32 value. Tries the
    // compiled program's `property_slots` first; falls back to state
    // vars by bare name. Used here exclusively to (a) read each
    // descriptor's own dispatch-key slot during routing, and (b)
    // populate the panic diagnostic when an applier surfaces an
    // unrecognised shape. Every name passed in comes off a descriptor —
    // no literal property name appears anywhere in this dispatcher.
    fn read_prop(
        program: &CompiledProgram,
        state: &State,
        slots: &[i32],
        name: &str,
    ) -> Option<i32> {
        if let Some(&s) = program.property_slots.get(name) {
            return Some(slots[s as usize]);
        }
        let bare = name.strip_prefix("--").unwrap_or(name);
        state.get_var(bare)
    }

    rep_diag_bail("00-entered");

    // Which descriptor's dispatch family fired this tick? Each
    // descriptor carries its own key property name (`key_property`,
    // captured structurally by the recogniser) and the key value its
    // shapes belong to. Read the cabinet's own key slot and compare.
    // Descriptors from the same dispatch family share a key property,
    // so cache the last read — in practice this is one slot read per
    // tick, same as the old literal-name read, without the literal.
    //
    // No `matches!(opcode, 0xAA | 0xAB | ...)` here. No `is_stos_movs`
    // / `is_cmps_scas` derived tables. The cabinet's recogniser already
    // enumerated every self-loop key it produced at load time; if the
    // current key value isn't on that list (or the cabinet has no
    // recognised dispatch at all), calcite has no fast-forward to do
    // and the CSS single-iter result stands.
    let mut last_key_read: Option<(&str, Option<i64>)> = None;
    let mut found: Option<&crate::pattern::loop_descriptor::LoopDescriptor> = None;
    for d in &program.loop_descriptors {
        let v = match last_key_read {
            Some((p, v)) if p == d.key_property.as_str() => v,
            _ => {
                let v = read_prop(program, state, slots, &d.key_property).map(i64::from);
                last_key_read = Some((d.key_property.as_str(), v));
                v
            }
        };
        if v == Some(d.key_value) {
            found = Some(d);
            break;
        }
    }
    let Some(descriptor) = found else {
        rep_diag_bail("no-descriptor");
        return;
    };

    // Loop-continuation gate: did the CSS take the stay branch of the
    // IP body this tick? If not, there is no loop in flight — either
    // the op ran single-shot (no rep-style prefix active) or this tick
    // WAS the final iteration and the CSS already advanced IP past the
    // loop. Both mean the post-tick state is already correct and
    // collapsing "remaining" iterations would corrupt it (the
    // single-shot case would bulk-apply a stale counter). This is the
    // descriptor-driven equivalent of the deleted hardcoded path's
    // "hasREP != 1" and "REPE/REPNE already exited via post-tick ZF"
    // entry guards, folded into the cabinet's own predicate.
    if !crate::pattern::loop_descriptor::evaluate_loop_predicate(descriptor, program, state, slots)
    {
        rep_diag_bail("loop-not-continuing");
        return;
    }

    use crate::pattern::loop_descriptor::BulkClass;
    use crate::pattern::rep_applier::{
        apply_copy_with_commit, apply_fill_with_commit, apply_read_only_with_commit,
        ApplyOutcome, CommitMode,
    };

    let outcome = match descriptor.bulk_class {
        BulkClass::Fill => apply_fill_with_commit(
            descriptor, program, state, slots, CommitMode::Full,
        ),
        BulkClass::Copy => apply_copy_with_commit(
            descriptor, program, state, slots, CommitMode::Full,
        ),
        BulkClass::ReadOnly => apply_read_only_with_commit(
            descriptor, program, state, slots, CommitMode::Full,
        ),
        BulkClass::PerIter => {
            // Recogniser produced a descriptor but classified it as
            // PerIter — no bulk applier handles this shape. Panic
            // loudly. Per-iter CSS fallback runs at 10-1000x slowdown
            // per iteration, which on a long REP strands the user on
            // a frozen screen with no error. The fix is to extend the
            // recogniser/applier to handle this shape, never to make
            // the bail silent.
            rep_fast_forward_panic(
                "bulk-class-per-iter",
                descriptor,
                &describe_loop_state(program, state, slots, descriptor),
                "descriptor.bulk_class is PerIter — recogniser produced a descriptor but no bulk applier handles this shape. Extend the recogniser/applier; do not silently fall back to per-iter CSS.",
            );
        }
    };

    match outcome {
        ApplyOutcome::Applied { iterations } => {
            // Applier committed memory + state vars. Tally diagnostic
            // counters and return.
            rep_diag_fire(descriptor.key_value, iterations);
        }
        ApplyOutcome::PreconditionNotMet => {
            // The cabinet's own outer guard predicate evaluated false
            // (e.g., IRQ vectored this tick, segment override prefix
            // in effect, CMPS/SCAS already exited via ZF). The CSS
            // already produced the correct single-iter post-state and
            // calcite has nothing to do. Silent bail — this is NOT a
            // recogniser gap, it's the cabinet telling us so.
            rep_diag_bail("precondition-not-met");
        }
        ApplyOutcome::Unsupported(reason) => {
            // Recogniser produced a descriptor but the applier
            // refused. Distinct from PreconditionNotMet: this is a
            // genuine recogniser/applier gap that would otherwise
            // silently route the user into a slow per-iter CSS path.
            // Panic with the loop's own slot state so the developer
            // can diagnose. See the BulkClass::PerIter arm above for
            // the rationale on why we don't fall back silently.
            rep_fast_forward_panic(
                "applier-unsupported",
                descriptor,
                &describe_loop_state(program, state, slots, descriptor),
                reason,
            );
        }
    }
}

/// Render the current values of every slot a loop descriptor names —
/// key, IP, counter, pointers, predicate participants — for panic
/// diagnostics. Every property printed comes off the descriptor (the
/// cabinet's own names); nothing is read by literal name, so the dump
/// is equally meaningful for an x86 cabinet (`--CX=412 --DI=80`) and
/// any other (`--whisksLeft=3 --bowlPos=9`).
fn describe_loop_state(
    program: &CompiledProgram,
    state: &State,
    slots: &[i32],
    descriptor: &crate::pattern::loop_descriptor::LoopDescriptor,
) -> String {
    fn read_prop(
        program: &CompiledProgram,
        state: &State,
        slots: &[i32],
        name: &str,
    ) -> Option<i32> {
        if let Some(&s) = program.property_slots.get(name) {
            return Some(slots[s as usize]);
        }
        let bare = name.strip_prefix("--").unwrap_or(name);
        state.get_var(bare)
    }
    let mut names: Vec<&str> = vec![descriptor.ip_property.as_str()];
    if let Some(c) = descriptor.counter.as_ref() {
        names.push(c.property.as_str());
    }
    for p in &descriptor.pointers {
        names.push(p.property.as_str());
    }
    for p in &descriptor.predicate_properties {
        names.push(p.as_str());
    }
    names
        .into_iter()
        .map(|n| match read_prop(program, state, slots, n) {
            Some(v) => format!("{n}={v:#x}"),
            None => format!("{n}=<unresolved>"),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Compute the 8086 flag word produced by SUB(dst, src), preserving the
/// upper control bits (TF=0x100, IF=0x200, DF=0x400) from `prev_flags`.
/// Mirrors kiln's `--subFlags8` / `--subFlags16` + `+ and(prev, 1792)`.
pub(crate) fn compute_sub_flags(dst: i32, src: i32, is_word: bool, prev_flags: i32) -> i32 {
    let mask: i32 = if is_word { 0xFFFF } else { 0xFF };
    let sign_bit: i32 = if is_word { 0x8000 } else { 0x80 };
    let dst_u = dst & mask;
    let src_u = src & mask;
    let res = (dst_u.wrapping_sub(src_u)) & mask;

    // CF: borrow from MSB → src > dst as unsigned.
    let cf = if (src_u as u32) > (dst_u as u32) { 1 } else { 0 };
    // PF: parity of low byte of result.
    let low = (res & 0xFF) as u8;
    let pf = if (low.count_ones() & 1) == 0 { 4 } else { 0 };
    // AF: borrow from bit 4 of low nibble.
    let af = if ((src_u & 0xF) as u32) > ((dst_u & 0xF) as u32) { 0x10 } else { 0 };
    // ZF.
    let zf = if res == 0 { 0x40 } else { 0 };
    // SF.
    let sf = if (res & sign_bit) != 0 { 0x80 } else { 0 };
    // OF: signed overflow of dst - src.
    //   sign(dst) != sign(src) AND sign(res) != sign(dst)
    let of = if ((dst_u ^ src_u) & sign_bit) != 0 && ((dst_u ^ res) & sign_bit) != 0 {
        0x800
    } else {
        0
    };
    // Bit 1 (always set on x86 flags).
    let base = 2;
    // Preserve TF/IF/DF (bits 8/9/10 = 0x700) from previous flags. (kiln
    // uses 1792 = 0x700.)
    let preserved = prev_flags & 0x700;

    cf + pf + af + zf + sf + of + base + preserved
}

#[cfg(not(target_arch = "wasm32"))]
mod rep_diag {
    use std::sync::Mutex;
    use std::sync::OnceLock;
    use std::collections::HashMap;
    /// Fire counters keyed by dispatch-key value — the cabinet's own
    /// key space (opcode bytes for a CPU cabinet, whatever else for
    /// any other). No per-mnemonic fields: the diagnostics carry no
    /// ISA vocabulary.
    #[derive(Default, Clone)]
    pub(super) struct Fires {
        /// (fire count, iterations elided) per dispatch-key value.
        pub per_key: HashMap<i64, (u64, i64)>,
        pub total_iters: i64,
    }
    pub(super) static COUNTS: OnceLock<Mutex<HashMap<&'static str, u64>>> = OnceLock::new();
    pub(super) static FIRES: OnceLock<Mutex<Fires>> = OnceLock::new();
    pub(super) static ENABLED: OnceLock<bool> = OnceLock::new();
    pub(super) fn enabled() -> bool {
        *ENABLED.get_or_init(|| std::env::var("CALCITE_REP_DIAG").is_ok())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn rep_diag_bail(reason: &'static str) {
    if !rep_diag::enabled() { return; }
    let m = rep_diag::COUNTS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    *m.lock().unwrap().entry(reason).or_insert(0) += 1;
}
#[cfg(target_arch = "wasm32")]
fn rep_diag_bail(_reason: &'static str) {}

/// Panic with the offending loop's state. Called when a fast-forward
/// bail would otherwise drop into per-iter CSS execution. There is no
/// slow path: per-iter CSS for a long recognised loop is so slow that
/// conformance can't progress within a sensible budget. When this
/// fires, the fix is to extend the recogniser/applier to handle the
/// new variant — never to make the bail silent.
///
/// The diagnostic prints only descriptor-carried names and values (see
/// `describe_loop_state`) — the cabinet's own vocabulary, whatever its
/// ISA or lack of one.
fn rep_fast_forward_panic(
    reason: &'static str,
    descriptor: &crate::pattern::loop_descriptor::LoopDescriptor,
    loop_state: &str,
    advice: &str,
) -> ! {
    panic!(
        "rep_fast_forward: {reason} — dispatch {key}={val:#x} ({class:?})\n  loop state: {loop_state}\n  {advice}\n  No slow path: extend the recogniser/applier to handle this variant.",
        key = descriptor.key_property,
        val = descriptor.key_value,
        class = descriptor.bulk_class,
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn rep_diag_fire(key_value: i64, n: i32) {
    if !rep_diag::enabled() { return; }
    let m = rep_diag::FIRES.get_or_init(|| std::sync::Mutex::new(rep_diag::Fires::default()));
    let mut m = m.lock().unwrap();
    let e = m.per_key.entry(key_value).or_insert((0, 0));
    e.0 += 1;
    e.1 += n as i64;
    m.total_iters += n as i64;
}
#[cfg(target_arch = "wasm32")]
fn rep_diag_fire(_key_value: i64, _n: i32) {}

pub fn rep_diag_reset() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        rep_diag::COUNTS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
            .lock().unwrap().clear();
        *rep_diag::FIRES.get_or_init(|| std::sync::Mutex::new(rep_diag::Fires::default()))
            .lock().unwrap() = rep_diag::Fires::default();
    }
}

pub fn rep_diag_report() -> String {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let bails = rep_diag::COUNTS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
            .lock().unwrap().clone();
        let fires = rep_diag::FIRES.get_or_init(|| std::sync::Mutex::new(rep_diag::Fires::default()))
            .lock().unwrap().clone();
        let mut s = String::new();
        s.push_str("rep_fast_forward fires (per dispatch-key value):\n");
        let mut keys: Vec<_> = fires.per_key.iter().collect();
        keys.sort_by_key(|(k, _)| **k);
        for (k, (count, iters)) in keys {
            s.push_str(&format!("  key={k:#x}: fires={count} iters={iters}\n"));
        }
        s.push_str(&format!("rep_fast_forward total_iters={}\n", fires.total_iters));
        s.push_str("rep_fast_forward bails:\n");
        let mut items: Vec<_> = bails.iter().collect();
        items.sort_by(|a, b| b.1.cmp(a.1));
        for (k, v) in items {
            s.push_str(&format!("  {}: {}\n", k, v));
        }
        return s;
    }
    #[cfg(target_arch = "wasm32")]
    { String::new() }
}

/// Returns true if [start, start+len) overlaps a memory region that isn't
/// backed by plain `state.memory` byte storage. CSS-DOS exposes three such
/// regions via dispatch in `--readMem`:
///   - [0x0500, 0x0502)   — BDA keyboard head bridge to `--keyboard` state var.
///   - [0xD0000, 0xD0200) — ROM-disk window synthesised by `--readDiskByte`.
///   - [0xF0000, 0x100000) — BIOS ROM, routed to state.extended.
/// Plus writes to the BIOS region also route to state.extended. The
/// fast-forward collapses iterations into a single `state.memory`
/// operation, which would be wrong for any of these ranges.
#[inline]
pub(crate) fn ranges_overlap_virtual(state: &State, start: i64, len: i64) -> bool {
    if len <= 0 { return false; }
    let end = start + len;
    let overlaps = |a: i64, b: i64, c: i64, d: i64| a < d && c < b;
    // Extended-map range: a State invariant, not a cabinet feature.
    if overlaps(start, end, 0xF_0000, 0x10_0000) {
        return true;
    }
    // Cabinet-registered virtual regions (windowed byte arrays, etc).
    for region in &state.virtual_regions {
        if overlaps(start, end, region.start as i64, region.end as i64) {
            return true;
        }
    }
    false
}

pub(crate) fn bulk_store_byte(state: &mut State, addr: i64, val: u8) {
    // Mirrors MemoryFill/MemoryCopy discipline (bulk path is
    // diagnostic-silent: no write_log entries). Routes through
    // `bulk_fill_byte(count=1)` so packed cabinets land the byte in the
    // correct state-var cell in addition to the flat shadow.
    if addr < 0 { return; }
    if addr >= 0xF0000 {
        state.extended.insert(addr as i32, val as i32);
        return;
    }
    state.bulk_fill_byte(addr as usize, 1, val);
}

/// Effective upper bound for guest memory. Packed cabinets keep the flat
/// `state.memory` at its default small size and store real bytes in packed
/// cells; we must count those too or bulk writes above `state.memory.len()`
/// silently drop (which for a long time was the cause of pack=2 diverging
/// from pack=1 on REP STOS/MOVS at high linear addresses).
#[inline]
fn effective_guest_mem_end(state: &State) -> usize {
    let flat = state.memory.len();
    if state.packed_cell_size > 0 && !state.packed_cell_table.is_empty() {
        let packed_end = state
            .packed_cell_table
            .len()
            .saturating_mul(state.packed_cell_size as usize);
        return flat.max(packed_end);
    }
    flat
}

#[inline]
pub(crate) fn bulk_fill(state: &mut State, dst: i64, count: usize, val: u8) {
    if count == 0 || dst < 0 { return; }
    let mem_len = effective_guest_mem_end(state);
    if (dst as usize) >= mem_len { return; }
    let lo = dst as usize;
    let hi = (dst as i64 + count as i64).min(mem_len as i64) as usize;
    if hi > lo {
        state.bulk_fill_byte(lo, hi - lo, val);
    }
}


/// Execute a sequence of ops against the slot array.
///
/// Hot path: uses unchecked slot indexing. Safety invariants, established by the
/// compiler:
/// - All slot indices in ops are < slots.len() (compiler allocates monotonically
///   then slot-compaction never introduces out-of-range indices).
/// - All branch/jump targets are <= ops.len() (fuse_cmp_branch adjusts them,
///   compile_style_condition_linear uses placeholders patched after the fact).
/// - Dispatch table_id is always a valid index into dispatch_tables.
/// Thin wrapper around `exec_ops` with empty ancillary tables. Used by
/// tests that drive the evaluator with a hand-constructed op stream,
/// bypassing parse/compile. Exposed via `#[doc(hidden)]` rather than
/// `#[cfg(test)]` so integration tests in `tests/` can call it.
#[doc(hidden)]
pub fn exec_ops_for_test(ops: &[Op], state: &mut State, slots: &mut [i32]) {
    let dispatch_tables: Vec<CompiledDispatchTable> = Vec::new();
    let chain_tables: Vec<DispatchChainTable> = Vec::new();
    let flat_dispatch_arrays: Vec<FlatDispatchArray> = Vec::new();
    let functions: Vec<CompiledFunction> = Vec::new();
    let packed_cell_tables: Vec<Vec<i32>> = Vec::new();
    let packed_exception_tables: Vec<PackedExceptionTable> = Vec::new();
    exec_ops(ops, &dispatch_tables, &chain_tables, &flat_dispatch_arrays, &functions, &packed_cell_tables, &packed_exception_tables, state, slots);
}

fn exec_ops(
    ops: &[Op],
    dispatch_tables: &[CompiledDispatchTable],
    chain_tables: &[DispatchChainTable],
    flat_dispatch_arrays: &[FlatDispatchArray],
    functions: &[CompiledFunction],
    packed_cell_tables: &[Vec<i32>],
    packed_exception_tables: &[PackedExceptionTable],
    state: &mut State,
    slots: &mut [i32],
) {
    let len = ops.len();
    let mut pc: usize = 0;

    // Shorthand helpers. All indices come from the compiler and are in-range;
    // debug builds still check via assert.
    macro_rules! sload {
        ($i:expr) => {{
            let idx = $i as usize;
            debug_assert!(idx < slots.len());
            unsafe { *slots.get_unchecked(idx) }
        }};
    }
    macro_rules! sstore {
        ($i:expr, $v:expr) => {{
            let idx = $i as usize;
            let v = $v;
            debug_assert!(idx < slots.len());
            unsafe { *slots.get_unchecked_mut(idx) = v; }
        }};
    }

    while pc < len {
        // Safety: pc < len checked by loop condition.
        let op = unsafe { ops.get_unchecked(pc) };
        match op {
            // Hot path: 96%+ of ops in CSS-DOS programs are this one variant.
            // Keep it first so the compiler can (hopefully) lay out the jump table
            // with this case at the predicted target.
            Op::BranchIfNotEqLit { a, val, target } => {
                if sload!(*a) != *val {
                    pc = *target as usize;
                    continue;
                }
            }
            Op::LoadStateAndBranchIfNotEqLit { dst, addr, val, target } => {
                let v = state.read_mem(*addr);
                sstore!(*dst, v);
                if v != *val {
                    pc = *target as usize;
                    continue;
                }
            }
            Op::BranchIfNotEqLit2 { a1, val1, a2, val2, target } => {
                if sload!(*a1) != *val1 || sload!(*a2) != *val2 {
                    pc = *target as usize;
                    continue;
                }
                // Skip the dead BIfNEL at i+1 left for external-jump safety.
                // The natural `pc += 1` at end-of-loop will advance the rest.
                pc += 1;
            }
            Op::LoadLit { dst, val } => {
                sstore!(*dst, *val);
            }
            Op::LoadSlot { dst, src } => {
                sstore!(*dst, sload!(*src));
            }
            Op::LoadState { dst, addr } => {
                sstore!(*dst, state.read_mem(*addr));
            }
            Op::LoadMem { dst, addr_slot } => {
                let addr = sload!(*addr_slot);
                sstore!(*dst, state.read_mem(addr));
            }
            Op::LoadMem16 { dst, addr_slot } => {
                let addr = sload!(*addr_slot);
                let v = if addr < 0 { state.read_mem(addr) } else { state.read_mem16(addr) };
                sstore!(*dst, v);
            }
            Op::LoadPackedByte { dst, key_slot, table_id, pack } => {
                let key = sload!(*key_slot);
                let tid = *table_id as usize;
                let v = if key < 0 {
                    0
                } else if let Some(lit) = packed_exception_tables[tid].get(key) {
                    lit
                } else {
                    let pack_i = *pack as i32;
                    let n = (key / pack_i) as usize;
                    let off = key % pack_i;
                    let table = &packed_cell_tables[tid];
                    let cell_addr = if n < table.len() { table[n] } else { 0 };
                    let cell = if cell_addr == 0 { 0 } else { state.read_mem(cell_addr) };
                    // (cell >> (off*8)) & 0xFF — matches Op::LoadPackedByte's
                    // documented semantics, equivalent to the prior
                    // rem_euclid/div_euclid pair for off∈{0,1} (pack=2 today).
                    // Two's-complement byte extraction works for negative cells
                    // because the arithmetic-shift sign extension is masked off.
                    (cell >> (off * 8)) & 0xFF
                };
                sstore!(*dst, v);
            }
            Op::Add { dst, a, b } => {
                sstore!(*dst, sload!(*a).wrapping_add(sload!(*b)));
            }
            Op::AddLit { dst, a, val } => {
                sstore!(*dst, sload!(*a).wrapping_add(*val));
            }
            Op::Sub { dst, a, b } => {
                sstore!(*dst, sload!(*a).wrapping_sub(sload!(*b)));
            }
            Op::SubLit { dst, a, val } => {
                sstore!(*dst, sload!(*a).wrapping_sub(*val));
            }
            Op::Mul { dst, a, b } => {
                sstore!(*dst, sload!(*a).wrapping_mul(sload!(*b)));
            }
            Op::MulLit { dst, a, val } => {
                sstore!(*dst, sload!(*a).wrapping_mul(*val));
            }
            Op::Div { dst, a, b } => {
                let divisor = sload!(*b);
                let v = if divisor == 0 { 0 } else { sload!(*a) / divisor };
                sstore!(*dst, v);
            }
            Op::Mod { dst, a, b } => {
                let divisor = sload!(*b);
                let v = if divisor == 0 { 0 } else { sload!(*a) % divisor };
                sstore!(*dst, v);
            }
            Op::Neg { dst, src } => {
                sstore!(*dst, sload!(*src).wrapping_neg());
            }
            Op::Abs { dst, src } => {
                sstore!(*dst, sload!(*src).wrapping_abs());
            }
            Op::Sign { dst, src } => {
                let v = sload!(*src);
                sstore!(*dst, if v > 0 { 1 } else if v < 0 { -1 } else { 0 });
            }
            Op::Pow { dst, base, exp } => {
                let b = sload!(*base);
                let e = sload!(*exp);
                sstore!(*dst, if e < 0 { 0 } else { b.wrapping_pow(e as u32) });
            }
            Op::Min { dst, args } => {
                let mut v = i32::MAX;
                for &a in args {
                    v = v.min(sload!(a));
                }
                sstore!(*dst, v);
            }
            Op::Max { dst, args } => {
                let mut v = i32::MIN;
                for &a in args {
                    v = v.max(sload!(a));
                }
                sstore!(*dst, v);
            }
            Op::Clamp { dst, min, val, max } => {
                let min_v = sload!(*min);
                let val_v = sload!(*val);
                let max_v = sload!(*max);
                sstore!(*dst, val_v.clamp(min_v, max_v));
            }
            Op::Round { dst, strategy, val, interval } => {
                let v = sload!(*val);
                let i = sload!(*interval);
                let r = if i == 0 {
                    v
                } else {
                    match strategy {
                        RoundStrategy::Down => v.div_euclid(i) * i,
                        RoundStrategy::Up => (v + i - 1).div_euclid(i) * i,
                        RoundStrategy::Nearest => ((v + i / 2).div_euclid(i)) * i,
                        RoundStrategy::ToZero => (v / i) * i,
                    }
                };
                sstore!(*dst, r);
            }
            Op::Floor { dst, src } => {
                sstore!(*dst, sload!(*src));
            }
            Op::And { dst, a, b } => {
                let av = sload!(*a);
                let bv = sload!(*b) as u32;
                sstore!(*dst, if bv >= 32 { av } else { av & ((1i32 << bv) - 1) });
            }
            Op::AndLit { dst, a, val } => {
                // val is precomputed mask (not shift count) — see emit.
                sstore!(*dst, sload!(*a) & *val);
            }
            Op::Shr { dst, a, b } => {
                let av = sload!(*a);
                let bv = sload!(*b) as u32;
                sstore!(*dst, if bv >= 32 { 0 } else { av >> bv });
            }
            Op::ShrLit { dst, a, val } => {
                // val is shift count (0..32). Compiler validates at emit.
                sstore!(*dst, sload!(*a) >> *val);
            }
            Op::Shl { dst, a, b } => {
                let av = sload!(*a);
                let bv = sload!(*b) as u32;
                sstore!(*dst, if bv >= 32 { 0 } else { av << bv });
            }
            Op::ShlLit { dst, a, val } => {
                sstore!(*dst, sload!(*a) << *val);
            }
            Op::ModLit { dst, a, val } => {
                let divisor = *val;
                sstore!(*dst, if divisor == 0 { 0 } else { sload!(*a) % divisor });
            }
            Op::Bit { dst, val, idx } => {
                let v = sload!(*val);
                let i = sload!(*idx) as u32;
                sstore!(*dst, if i >= 32 { 0 } else { (v >> i) & 1 });
            }
            Op::BitAnd16 { dst, a, b } => {
                let av = sload!(*a) as u32 & 0xFFFF;
                let bv = sload!(*b) as u32 & 0xFFFF;
                sstore!(*dst, (av & bv) as i32);
            }
            Op::BitOr16 { dst, a, b } => {
                let av = sload!(*a) as u32 & 0xFFFF;
                let bv = sload!(*b) as u32 & 0xFFFF;
                sstore!(*dst, (av | bv) as i32);
            }
            Op::BitXor16 { dst, a, b } => {
                let av = sload!(*a) as u32 & 0xFFFF;
                let bv = sload!(*b) as u32 & 0xFFFF;
                sstore!(*dst, (av ^ bv) as i32);
            }
            Op::BitNot16 { dst, a } => {
                let av = sload!(*a) as u32 & 0xFFFF;
                sstore!(*dst, ((!av) & 0xFFFF) as i32);
            }
            Op::CmpEq { dst, a, b } => {
                sstore!(*dst, if sload!(*a) == sload!(*b) { 1 } else { 0 });
            }
            Op::BranchIfZero { cond, target } => {
                if sload!(*cond) == 0 {
                    pc = *target as usize;
                    continue;
                }
            }
            Op::Jump { target } => {
                pc = *target as usize;
                continue;
            }
            Op::DispatchChain { a, chain_id, miss_target } => {
                let key = sload!(*a);
                let table = unsafe { chain_tables.get_unchecked(*chain_id as usize) };
                // Fast path: flat-array lookup if available.
                if let Some(ref flat) = table.flat_table {
                    let idx = key.wrapping_sub(flat.base);
                    if (idx as u32) < flat.targets.len() as u32 {
                        let t = unsafe { *flat.targets.get_unchecked(idx as usize) };
                        if t != u32::MAX {
                            pc = t as usize;
                            continue;
                        }
                    }
                    pc = *miss_target as usize;
                    continue;
                }
                // Fallback: HashMap lookup.
                match table.entries.get(&key) {
                    Some(&body_pc) => { pc = body_pc as usize; }
                    None => { pc = *miss_target as usize; }
                }
                continue;
            }
            Op::Dispatch { dst, key, table_id, .. } => {
                let key_val = sload!(*key) as i64;
                let table = unsafe { dispatch_tables.get_unchecked(*table_id as usize) };
                if let Some((entry_ops, result_slot)) = table.entries.get(&key_val) {
                    exec_ops(entry_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots);
                    sstore!(*dst, sload!(*result_slot));
                } else {
                    exec_ops(&table.fallback_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots);
                    sstore!(*dst, sload!(table.fallback_slot));
                }
            }
            Op::DispatchFlatArray { dst, key, array_id, base_key, default } => {
                let key_val = sload!(*key);
                let arr = unsafe { flat_dispatch_arrays.get_unchecked(*array_id as usize) };
                let idx = key_val.wrapping_sub(*base_key);
                let v = if idx < 0 || (idx as usize) >= arr.values.len() {
                    *default
                } else {
                    unsafe { *arr.values.get_unchecked(idx as usize) }
                };
                sstore!(*dst, v);
            }
            Op::Call { dst, fn_id, arg_slots } => {
                let func = unsafe { functions.get_unchecked(*fn_id as usize) };
                // Copy arg values into the reserved parameter slots.
                for (arg, &param) in arg_slots.iter().zip(func.param_slots.iter()) {
                    sstore!(param, sload!(*arg));
                }
                // Run the cached body (uses its own internal slots + the param slots).
                exec_ops(&func.body_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots);
                sstore!(*dst, sload!(func.result_slot));
            }
            Op::StoreState { addr, src } => {
                state.write_mem(*addr, sload!(*src));
            }
            Op::StoreMem { addr_slot, src } => {
                state.write_mem(sload!(*addr_slot), sload!(*src));
            }
            Op::MemoryFill { dst_slot, val_slot, count_slot, exit_target } => {
                let dst = sload!(*dst_slot);
                let count = sload!(*count_slot);
                let val_byte = (sload!(*val_slot) & 0xFF) as u8;
                let mem_len = effective_guest_mem_end(state);
                if count > 0 && dst >= 0 && (dst as usize) < mem_len {
                    let lo = dst as usize;
                    let hi = (dst as i64 + count as i64).min(mem_len as i64) as usize;
                    if hi > lo {
                        state.bulk_fill_byte(lo, hi - lo, val_byte);
                    }
                }
                sstore!(*dst_slot, dst.wrapping_add(count));
                sstore!(*count_slot, 0i32);
                pc = *exit_target as usize;
                continue;
            }
            Op::MemoryCopy { src_slot, dst_slot, count_slot, exit_target } => {
                let src = sload!(*src_slot);
                let dst = sload!(*dst_slot);
                let count = sload!(*count_slot);
                let mem_len = effective_guest_mem_end(state);
                if count > 0 && src >= 0 && dst >= 0
                    && (src as usize) < mem_len && (dst as usize) < mem_len
                {
                    // Clip count to fit within memory on both ends.
                    let max_by_src = (mem_len as i64) - src as i64;
                    let max_by_dst = (mem_len as i64) - dst as i64;
                    let n = (count as i64).min(max_by_src).min(max_by_dst) as usize;
                    if n > 0 {
                        let s = src as usize;
                        let d = dst as usize;
                        state.bulk_copy_bytes(s, d, n);
                    }
                }
                sstore!(*src_slot, src.wrapping_add(count));
                sstore!(*dst_slot, dst.wrapping_add(count));
                sstore!(*count_slot, 0i32);
                pc = *exit_target as usize;
                continue;
            }
            Op::ReplicatedBody { body, strides, reps } => {
                exec_replicated_body(
                    body, strides, *reps,
                    dispatch_tables, chain_tables, flat_dispatch_arrays,
                    functions, packed_cell_tables, packed_exception_tables,
                    state, slots,
                );
            }
        }
        pc += 1;
    }
}

/// Op-adjacency-profiling variant of `exec_ops`. Byte-for-byte equivalent
/// behaviour to `exec_ops` plus a `pattern::op_profile::op_profile_tick`
/// call per dispatch. Recurses into itself for Dispatch entries / fallback
/// and Call bodies so adjacency captures pairs that cross those boundaries
/// — matches the comment on `LAST_OP_KIND` in `pattern/op_profile.rs`.
///
/// Only entered from `execute` (and the parallel broadcast/spillover paths)
/// when `op_profile_is_enabled()` is true. Cold-path duplicate of the
/// production loop — see `execute` for the dispatch.
#[allow(clippy::too_many_arguments)]
fn exec_ops_with_op_profile(
    ops: &[Op],
    dispatch_tables: &[CompiledDispatchTable],
    chain_tables: &[DispatchChainTable],
    flat_dispatch_arrays: &[FlatDispatchArray],
    functions: &[CompiledFunction],
    packed_cell_tables: &[Vec<i32>],
    packed_exception_tables: &[PackedExceptionTable],
    state: &mut State,
    slots: &mut [i32],
) {
    use crate::pattern::op_profile::{op_kind_index, op_profile_tick};

    let len = ops.len();
    let mut pc: usize = 0;

    macro_rules! sload {
        ($i:expr) => {{
            let idx = $i as usize;
            debug_assert!(idx < slots.len());
            unsafe { *slots.get_unchecked(idx) }
        }};
    }
    macro_rules! sstore {
        ($i:expr, $v:expr) => {{
            let idx = $i as usize;
            let v = $v;
            debug_assert!(idx < slots.len());
            unsafe { *slots.get_unchecked_mut(idx) = v; }
        }};
    }

    while pc < len {
        let op = unsafe { ops.get_unchecked(pc) };
        op_profile_tick(op_kind_index(op));
        match op {
            Op::BranchIfNotEqLit { a, val, target } => {
                if sload!(*a) != *val {
                    pc = *target as usize;
                    continue;
                }
            }
            Op::LoadStateAndBranchIfNotEqLit { dst, addr, val, target } => {
                let v = state.read_mem(*addr);
                sstore!(*dst, v);
                if v != *val {
                    pc = *target as usize;
                    continue;
                }
            }
            Op::BranchIfNotEqLit2 { a1, val1, a2, val2, target } => {
                if sload!(*a1) != *val1 || sload!(*a2) != *val2 {
                    pc = *target as usize;
                    continue;
                }
                // Skip the dead BIfNEL at i+1 left for external-jump safety.
                // The natural `pc += 1` at end-of-loop will advance the rest.
                pc += 1;
            }
            Op::LoadLit { dst, val } => {
                sstore!(*dst, *val);
            }
            Op::LoadSlot { dst, src } => {
                sstore!(*dst, sload!(*src));
            }
            Op::LoadState { dst, addr } => {
                sstore!(*dst, state.read_mem(*addr));
            }
            Op::LoadMem { dst, addr_slot } => {
                let addr = sload!(*addr_slot);
                sstore!(*dst, state.read_mem(addr));
            }
            Op::LoadMem16 { dst, addr_slot } => {
                let addr = sload!(*addr_slot);
                let v = if addr < 0 { state.read_mem(addr) } else { state.read_mem16(addr) };
                sstore!(*dst, v);
            }
            Op::LoadPackedByte { dst, key_slot, table_id, pack } => {
                let key = sload!(*key_slot);
                let tid = *table_id as usize;
                let v = if key < 0 {
                    0
                } else if let Some(lit) = packed_exception_tables[tid].get(key) {
                    lit
                } else {
                    let pack_i = *pack as i32;
                    let n = (key / pack_i) as usize;
                    let off = key % pack_i;
                    let table = &packed_cell_tables[tid];
                    let cell_addr = if n < table.len() { table[n] } else { 0 };
                    let cell = if cell_addr == 0 { 0 } else { state.read_mem(cell_addr) };
                    (cell >> (off * 8)) & 0xFF
                };
                sstore!(*dst, v);
            }
            Op::Add { dst, a, b } => {
                sstore!(*dst, sload!(*a).wrapping_add(sload!(*b)));
            }
            Op::AddLit { dst, a, val } => {
                sstore!(*dst, sload!(*a).wrapping_add(*val));
            }
            Op::Sub { dst, a, b } => {
                sstore!(*dst, sload!(*a).wrapping_sub(sload!(*b)));
            }
            Op::SubLit { dst, a, val } => {
                sstore!(*dst, sload!(*a).wrapping_sub(*val));
            }
            Op::Mul { dst, a, b } => {
                sstore!(*dst, sload!(*a).wrapping_mul(sload!(*b)));
            }
            Op::MulLit { dst, a, val } => {
                sstore!(*dst, sload!(*a).wrapping_mul(*val));
            }
            Op::Div { dst, a, b } => {
                let divisor = sload!(*b);
                let v = if divisor == 0 { 0 } else { sload!(*a) / divisor };
                sstore!(*dst, v);
            }
            Op::Mod { dst, a, b } => {
                let divisor = sload!(*b);
                let v = if divisor == 0 { 0 } else { sload!(*a) % divisor };
                sstore!(*dst, v);
            }
            Op::Neg { dst, src } => {
                sstore!(*dst, sload!(*src).wrapping_neg());
            }
            Op::Abs { dst, src } => {
                sstore!(*dst, sload!(*src).wrapping_abs());
            }
            Op::Sign { dst, src } => {
                let v = sload!(*src);
                sstore!(*dst, if v > 0 { 1 } else if v < 0 { -1 } else { 0 });
            }
            Op::Pow { dst, base, exp } => {
                let b = sload!(*base);
                let e = sload!(*exp);
                sstore!(*dst, if e < 0 { 0 } else { b.wrapping_pow(e as u32) });
            }
            Op::Min { dst, args } => {
                let mut v = i32::MAX;
                for &a in args {
                    v = v.min(sload!(a));
                }
                sstore!(*dst, v);
            }
            Op::Max { dst, args } => {
                let mut v = i32::MIN;
                for &a in args {
                    v = v.max(sload!(a));
                }
                sstore!(*dst, v);
            }
            Op::Clamp { dst, min, val, max } => {
                let min_v = sload!(*min);
                let val_v = sload!(*val);
                let max_v = sload!(*max);
                sstore!(*dst, val_v.clamp(min_v, max_v));
            }
            Op::Round { dst, strategy, val, interval } => {
                let v = sload!(*val);
                let i = sload!(*interval);
                let r = if i == 0 {
                    v
                } else {
                    match strategy {
                        RoundStrategy::Down => v.div_euclid(i) * i,
                        RoundStrategy::Up => (v + i - 1).div_euclid(i) * i,
                        RoundStrategy::Nearest => ((v + i / 2).div_euclid(i)) * i,
                        RoundStrategy::ToZero => (v / i) * i,
                    }
                };
                sstore!(*dst, r);
            }
            Op::Floor { dst, src } => {
                sstore!(*dst, sload!(*src));
            }
            Op::And { dst, a, b } => {
                let av = sload!(*a);
                let bv = sload!(*b) as u32;
                sstore!(*dst, if bv >= 32 { av } else { av & ((1i32 << bv) - 1) });
            }
            Op::AndLit { dst, a, val } => {
                sstore!(*dst, sload!(*a) & *val);
            }
            Op::Shr { dst, a, b } => {
                let av = sload!(*a);
                let bv = sload!(*b) as u32;
                sstore!(*dst, if bv >= 32 { 0 } else { av >> bv });
            }
            Op::ShrLit { dst, a, val } => {
                sstore!(*dst, sload!(*a) >> *val);
            }
            Op::Shl { dst, a, b } => {
                let av = sload!(*a);
                let bv = sload!(*b) as u32;
                sstore!(*dst, if bv >= 32 { 0 } else { av << bv });
            }
            Op::ShlLit { dst, a, val } => {
                sstore!(*dst, sload!(*a) << *val);
            }
            Op::ModLit { dst, a, val } => {
                let divisor = *val;
                sstore!(*dst, if divisor == 0 { 0 } else { sload!(*a) % divisor });
            }
            Op::Bit { dst, val, idx } => {
                let v = sload!(*val);
                let i = sload!(*idx) as u32;
                sstore!(*dst, if i >= 32 { 0 } else { (v >> i) & 1 });
            }
            Op::BitAnd16 { dst, a, b } => {
                let av = sload!(*a) as u32 & 0xFFFF;
                let bv = sload!(*b) as u32 & 0xFFFF;
                sstore!(*dst, (av & bv) as i32);
            }
            Op::BitOr16 { dst, a, b } => {
                let av = sload!(*a) as u32 & 0xFFFF;
                let bv = sload!(*b) as u32 & 0xFFFF;
                sstore!(*dst, (av | bv) as i32);
            }
            Op::BitXor16 { dst, a, b } => {
                let av = sload!(*a) as u32 & 0xFFFF;
                let bv = sload!(*b) as u32 & 0xFFFF;
                sstore!(*dst, (av ^ bv) as i32);
            }
            Op::BitNot16 { dst, a } => {
                let av = sload!(*a) as u32 & 0xFFFF;
                sstore!(*dst, ((!av) & 0xFFFF) as i32);
            }
            Op::CmpEq { dst, a, b } => {
                sstore!(*dst, if sload!(*a) == sload!(*b) { 1 } else { 0 });
            }
            Op::BranchIfZero { cond, target } => {
                if sload!(*cond) == 0 {
                    pc = *target as usize;
                    continue;
                }
            }
            Op::Jump { target } => {
                pc = *target as usize;
                continue;
            }
            Op::DispatchChain { a, chain_id, miss_target } => {
                let key = sload!(*a);
                let table = unsafe { chain_tables.get_unchecked(*chain_id as usize) };
                if let Some(ref flat) = table.flat_table {
                    let idx = key.wrapping_sub(flat.base);
                    if (idx as u32) < flat.targets.len() as u32 {
                        let t = unsafe { *flat.targets.get_unchecked(idx as usize) };
                        if t != u32::MAX {
                            pc = t as usize;
                            continue;
                        }
                    }
                    pc = *miss_target as usize;
                    continue;
                }
                match table.entries.get(&key) {
                    Some(&body_pc) => { pc = body_pc as usize; }
                    None => { pc = *miss_target as usize; }
                }
                continue;
            }
            Op::Dispatch { dst, key, table_id, .. } => {
                let key_val = sload!(*key) as i64;
                let table = unsafe { dispatch_tables.get_unchecked(*table_id as usize) };
                if let Some((entry_ops, result_slot)) = table.entries.get(&key_val) {
                    exec_ops_with_op_profile(entry_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots);
                    sstore!(*dst, sload!(*result_slot));
                } else {
                    exec_ops_with_op_profile(&table.fallback_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots);
                    sstore!(*dst, sload!(table.fallback_slot));
                }
            }
            Op::DispatchFlatArray { dst, key, array_id, base_key, default } => {
                let key_val = sload!(*key);
                let arr = unsafe { flat_dispatch_arrays.get_unchecked(*array_id as usize) };
                let idx = key_val.wrapping_sub(*base_key);
                let v = if idx < 0 || (idx as usize) >= arr.values.len() {
                    *default
                } else {
                    unsafe { *arr.values.get_unchecked(idx as usize) }
                };
                sstore!(*dst, v);
            }
            Op::Call { dst, fn_id, arg_slots } => {
                let func = unsafe { functions.get_unchecked(*fn_id as usize) };
                for (arg, &param) in arg_slots.iter().zip(func.param_slots.iter()) {
                    sstore!(param, sload!(*arg));
                }
                exec_ops_with_op_profile(&func.body_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots);
                sstore!(*dst, sload!(func.result_slot));
            }
            Op::StoreState { addr, src } => {
                state.write_mem(*addr, sload!(*src));
            }
            Op::StoreMem { addr_slot, src } => {
                state.write_mem(sload!(*addr_slot), sload!(*src));
            }
            Op::MemoryFill { dst_slot, val_slot, count_slot, exit_target } => {
                let dst = sload!(*dst_slot);
                let count = sload!(*count_slot);
                let val_byte = (sload!(*val_slot) & 0xFF) as u8;
                let mem_len = effective_guest_mem_end(state);
                if count > 0 && dst >= 0 && (dst as usize) < mem_len {
                    let lo = dst as usize;
                    let hi = (dst as i64 + count as i64).min(mem_len as i64) as usize;
                    if hi > lo {
                        state.bulk_fill_byte(lo, hi - lo, val_byte);
                    }
                }
                sstore!(*dst_slot, dst.wrapping_add(count));
                sstore!(*count_slot, 0i32);
                pc = *exit_target as usize;
                continue;
            }
            Op::MemoryCopy { src_slot, dst_slot, count_slot, exit_target } => {
                let src = sload!(*src_slot);
                let dst = sload!(*dst_slot);
                let count = sload!(*count_slot);
                let mem_len = effective_guest_mem_end(state);
                if count > 0 && src >= 0 && dst >= 0
                    && (src as usize) < mem_len && (dst as usize) < mem_len
                {
                    let max_by_src = (mem_len as i64) - src as i64;
                    let max_by_dst = (mem_len as i64) - dst as i64;
                    let n = (count as i64).min(max_by_src).min(max_by_dst) as usize;
                    if n > 0 {
                        let s = src as usize;
                        let d = dst as usize;
                        state.bulk_copy_bytes(s, d, n);
                    }
                }
                sstore!(*src_slot, src.wrapping_add(count));
                sstore!(*dst_slot, dst.wrapping_add(count));
                sstore!(*count_slot, 0i32);
                pc = *exit_target as usize;
                continue;
            }
            Op::ReplicatedBody { body, strides, reps } => {
                // Replicated bodies execute via exec_replicated_body, which
                // calls back into exec_ops (not the profiled variant). For
                // op-profile correctness we re-implement the rep loop here
                // to dispatch through the profiled path. Bodies are small
                // (<= 16 ops typical) and contain no branches/dispatches/
                // bulk-mem ops by recogniser invariant.
                use crate::pattern::replicated_body::apply_strides;
                debug_assert_eq!(body.len(), strides.len());
                let mut scratch: Vec<Op> = Vec::with_capacity(body.len());
                for k in 0..*reps {
                    scratch.clear();
                    for (template, op_strides) in body.iter().zip(strides.iter()) {
                        scratch.push(apply_strides(template, k, op_strides));
                    }
                    exec_ops_with_op_profile(
                        &scratch,
                        dispatch_tables, chain_tables, flat_dispatch_arrays,
                        functions, packed_cell_tables, packed_exception_tables,
                        state, slots,
                    );
                }
            }
        }
        pc += 1;
    }
}

/// Execute a replicated-body op: for each rep `k` in `0..reps`, apply the
/// per-op strides to materialise the rep's ops, then run them.
///
/// The body never contains branches/dispatches/bulk-mem ops (recogniser
/// rejects those), so we can reuse `exec_ops` straight — its loop runs
/// linearly through the body, exits when pc == body.len(), and we repeat.
///
/// Allocates a small scratch `Vec<Op>` of length `body.len()` once and
/// reuses it across reps.
#[allow(clippy::too_many_arguments)]
fn exec_replicated_body(
    body: &[Op],
    strides: &[Box<[(u8, i32)]>],
    reps: u32,
    dispatch_tables: &[CompiledDispatchTable],
    chain_tables: &[DispatchChainTable],
    flat_dispatch_arrays: &[FlatDispatchArray],
    functions: &[CompiledFunction],
    packed_cell_tables: &[Vec<i32>],
    packed_exception_tables: &[PackedExceptionTable],
    state: &mut State,
    slots: &mut [i32],
) {
    use crate::pattern::replicated_body::apply_strides;
    debug_assert_eq!(body.len(), strides.len());

    // Scratch buffer: one materialised Op per body position.
    // Allocated fresh each call — bodies are small (typically <= 16 ops),
    // so the cost is dwarfed by the rep loop. If profiling shows this
    // matters, hoist into a reusable Vec on the call frame.
    let mut scratch: Vec<Op> = Vec::with_capacity(body.len());

    for k in 0..reps {
        scratch.clear();
        for (template, op_strides) in body.iter().zip(strides.iter()) {
            scratch.push(apply_strides(template, k, op_strides));
        }
        exec_ops(
            &scratch,
            dispatch_tables, chain_tables, flat_dispatch_arrays,
            functions, packed_cell_tables, packed_exception_tables,
            state, slots,
        );
    }
}


// ---------------------------------------------------------------------------
// Profiled execution — granular timing within execute()
// ---------------------------------------------------------------------------

use crate::eval::TickProfile;

/// Execute with granular profiling. Fills in the execute-related fields of TickProfile.
pub fn execute_profiled(
    program: &CompiledProgram,
    state: &mut State,
    slots: &mut Vec<i32>,
    profile: &mut TickProfile,
) {
    use web_time::Instant;

    // Slot reset
    let t0 = Instant::now();
    slots.clear();
    slots.resize(program.slot_count as usize, 0);
    let t1 = Instant::now();
    profile.slot_reset += t1.duration_since(t0).as_secs_f64();

    // Main ops (profiled — fills dispatch_time, counts, etc. directly)
    exec_ops_profiled(&program.ops, &program.dispatch_tables, &program.chain_tables, &program.flat_dispatch_arrays, &program.functions, &program.packed_cell_tables, &program.packed_exception_tables, state, slots, profile);
    let t2 = Instant::now();
    profile.main_ops += t2.duration_since(t1).as_secs_f64();
    profile.linear_ops = profile.main_ops - profile.dispatch_time;

    // Writeback
    let mut wb_changed: u64 = 0;
    for &(slot, addr) in &program.writeback {
        let value = slots[slot as usize];
        let old = state.read_mem(addr);
        if old != value {
            state.write_mem(addr, value);
            wb_changed += 1;
        }
    }
    let t3 = Instant::now();
    profile.writeback += t3.duration_since(t2).as_secs_f64();
    profile.writeback_count += program.writeback.len() as u64;
    profile.writeback_changed_count += wb_changed;

    // Broadcast writes
    let mut bw_fired: u64 = 0;
    for bw in &program.broadcast_writes {
        // Gated broadcasts skip entirely when the gate is not 1.
        if let Some(gate_slot) = bw.gate_slot {
            if slots[gate_slot as usize] != 1 {
                continue;
            }
        }
        let dest = slots[bw.dest_slot as usize];
        let dest_i64 = dest as i64;
        if bw.address_map.contains_key(&dest_i64) {
            exec_ops(&bw.value_ops, &program.dispatch_tables, &program.chain_tables, &program.flat_dispatch_arrays, &program.functions, &program.packed_cell_tables, &program.packed_exception_tables, state, slots);
            let value = slots[bw.value_slot as usize];
            state.write_mem(dest, value);
            bw_fired += 1;
        }
        if let Some(ref spillover) = bw.spillover {
            let guard = slots[spillover.guard_slot as usize];
            if guard == 1 {
                if let Some((ref spill_ops, spill_slot)) = spillover.entries.get(&dest_i64) {
                    exec_ops(spill_ops, &program.dispatch_tables, &program.chain_tables, &program.flat_dispatch_arrays, &program.functions, &program.packed_cell_tables, &program.packed_exception_tables, state, slots);
                    let value = slots[*spill_slot as usize];
                    state.write_mem(dest + 1, value);
                }
            }
        }
    }
    // Packed broadcast writes (mirror the unprofiled path; see execute()).
    for port in program.packed_broadcast_writes.iter().rev() {
        if slots[port.gate_slot as usize] != 1 {
            continue;
        }
        let byte_addr = slots[port.addr_slot as usize];
        if byte_addr < 0 {
            continue;
        }
        let val = slots[port.val_slot as usize];
        let width = slots[port.width_slot as usize];
        if width == 2 {
            if byte_addr & 1 == 0 {
                packed_splice_word_aligned(port, state, byte_addr, val & 0xFFFF);
            } else {
                packed_splice_byte(port, state, byte_addr, val & 0xFF);
                packed_splice_byte(port, state, byte_addr + 1, (val >> 8) & 0xFF);
            }
        } else {
            packed_splice_byte(port, state, byte_addr, val & 0xFF);
        }
    }
    let t4 = Instant::now();
    profile.broadcast += t4.duration_since(t3).as_secs_f64();
    profile.broadcast_fire_count += bw_fired;
    profile.broadcast_total_count += program.broadcast_writes.len() as u64;
}

/// Like exec_ops but fills profile with granular counters and timings.
fn exec_ops_profiled(
    ops: &[Op],
    dispatch_tables: &[CompiledDispatchTable],
    chain_tables: &[DispatchChainTable],
    flat_dispatch_arrays: &[FlatDispatchArray],
    functions: &[CompiledFunction],
    packed_cell_tables: &[Vec<i32>],
    packed_exception_tables: &[PackedExceptionTable],
    state: &mut State,
    slots: &mut [i32],
    profile: &mut TickProfile,
) {
    use web_time::Instant;

    macro_rules! count_op {
        ($profile:expr, $name:expr) => {
            *$profile.op_counts.entry($name).or_insert(0) += 1;
        };
    }

    let len = ops.len();
    let mut pc: usize = 0;

    while pc < len {
        profile.main_ops_count += 1;
        match &ops[pc] {
            Op::LoadLit { dst, val } => {
                count_op!(profile, "LoadLit");
                slots[*dst as usize] = *val;
            }
            Op::LoadSlot { dst, src } => {
                count_op!(profile, "LoadSlot");
                slots[*dst as usize] = slots[*src as usize];
            }
            Op::LoadState { dst, addr } => {
                count_op!(profile, "LoadState");
                slots[*dst as usize] = state.read_mem(*addr);
            }
            Op::LoadMem { dst, addr_slot } => {
                count_op!(profile, "LoadMem");
                let addr = slots[*addr_slot as usize];
                slots[*dst as usize] = state.read_mem(addr);
            }
            Op::LoadMem16 { dst, addr_slot } => {
                count_op!(profile, "LoadMem16");
                let addr = slots[*addr_slot as usize];
                if addr < 0 {
                    slots[*dst as usize] = state.read_mem(addr);
                } else {
                    slots[*dst as usize] = state.read_mem16(addr);
                }
            }
            Op::LoadPackedByte { dst, key_slot, table_id, pack } => {
                count_op!(profile, "LoadPackedByte");
                let key = slots[*key_slot as usize];
                let tid = *table_id as usize;
                let v = if key < 0 {
                    0
                } else if let Some(lit) = packed_exception_tables[tid].get(key) {
                    lit
                } else {
                    let pack_i = *pack as i32;
                    let n = (key / pack_i) as usize;
                    let off = key % pack_i;
                    let table = &packed_cell_tables[tid];
                    let cell_addr = if n < table.len() { table[n] } else { 0 };
                    let cell = if cell_addr == 0 { 0 } else { state.read_mem(cell_addr) };
                    (cell >> (off * 8)) & 0xFF
                };
                slots[*dst as usize] = v;
            }
            Op::Add { dst, a, b } => {
                count_op!(profile, "Add");
                slots[*dst as usize] = slots[*a as usize].wrapping_add(slots[*b as usize]);
            }
            Op::AddLit { dst, a, val } => {
                count_op!(profile, "AddLit");
                slots[*dst as usize] = slots[*a as usize].wrapping_add(*val);
            }
            Op::Sub { dst, a, b } => {
                count_op!(profile, "Sub");
                slots[*dst as usize] = slots[*a as usize].wrapping_sub(slots[*b as usize]);
            }
            Op::SubLit { dst, a, val } => {
                count_op!(profile, "SubLit");
                slots[*dst as usize] = slots[*a as usize].wrapping_sub(*val);
            }
            Op::Mul { dst, a, b } => {
                count_op!(profile, "Mul");
                slots[*dst as usize] = slots[*a as usize].wrapping_mul(slots[*b as usize]);
            }
            Op::MulLit { dst, a, val } => {
                count_op!(profile, "MulLit");
                slots[*dst as usize] = slots[*a as usize].wrapping_mul(*val);
            }
            Op::Div { dst, a, b } => {
                count_op!(profile, "Div");
                let divisor = slots[*b as usize];
                slots[*dst as usize] = if divisor == 0 {
                    0
                } else {
                    slots[*a as usize] / divisor
                };
            }
            Op::Mod { dst, a, b } => {
                count_op!(profile, "Mod");
                let divisor = slots[*b as usize];
                slots[*dst as usize] = if divisor == 0 {
                    0
                } else {
                    slots[*a as usize] % divisor
                };
            }
            Op::Neg { dst, src } => {
                count_op!(profile, "Neg");
                slots[*dst as usize] = slots[*src as usize].wrapping_neg();
            }
            Op::Abs { dst, src } => {
                count_op!(profile, "Abs");
                slots[*dst as usize] = slots[*src as usize].wrapping_abs();
            }
            Op::Sign { dst, src } => {
                count_op!(profile, "Sign");
                let v = slots[*src as usize];
                slots[*dst as usize] = if v > 0 {
                    1
                } else if v < 0 {
                    -1
                } else {
                    0
                };
            }
            Op::Pow { dst, base, exp } => {
                count_op!(profile, "Pow");
                let b = slots[*base as usize];
                let e = slots[*exp as usize];
                slots[*dst as usize] = if e < 0 { 0 } else { b.wrapping_pow(e as u32) };
            }
            Op::Min { dst, args } => {
                count_op!(profile, "Min");
                let mut v = i32::MAX;
                for &a in args {
                    v = v.min(slots[a as usize]);
                }
                slots[*dst as usize] = v;
            }
            Op::Max { dst, args } => {
                count_op!(profile, "Max");
                let mut v = i32::MIN;
                for &a in args {
                    v = v.max(slots[a as usize]);
                }
                slots[*dst as usize] = v;
            }
            Op::Clamp { dst, min, val, max } => {
                count_op!(profile, "Clamp");
                let min_v = slots[*min as usize];
                let val_v = slots[*val as usize];
                let max_v = slots[*max as usize];
                slots[*dst as usize] = val_v.clamp(min_v, max_v);
            }
            Op::Round {
                dst,
                strategy,
                val,
                interval,
            } => {
                count_op!(profile, "Round");
                let v = slots[*val as usize];
                let i = slots[*interval as usize];
                slots[*dst as usize] = if i == 0 {
                    v
                } else {
                    match strategy {
                        RoundStrategy::Down => v.div_euclid(i) * i,
                        RoundStrategy::Up => (v + i - 1).div_euclid(i) * i,
                        RoundStrategy::Nearest => ((v + i / 2).div_euclid(i)) * i,
                        RoundStrategy::ToZero => (v / i) * i,
                    }
                };
            }
            Op::Floor { dst, src } => {
                count_op!(profile, "Floor");
                slots[*dst as usize] = slots[*src as usize];
            }
            Op::And { dst, a, b } => {
                count_op!(profile, "And");
                let av = slots[*a as usize];
                let bv = slots[*b as usize] as u32;
                slots[*dst as usize] = if bv >= 32 {
                    av
                } else {
                    av & ((1i32 << bv) - 1)
                };
            }
            Op::Shr { dst, a, b } => {
                count_op!(profile, "Shr");
                let av = slots[*a as usize];
                let bv = slots[*b as usize] as u32;
                slots[*dst as usize] = if bv >= 32 { 0 } else { av >> bv };
            }
            Op::Shl { dst, a, b } => {
                count_op!(profile, "Shl");
                let av = slots[*a as usize];
                let bv = slots[*b as usize] as u32;
                slots[*dst as usize] = if bv >= 32 { 0 } else { av << bv };
            }
            Op::AndLit { dst, a, val } => {
                count_op!(profile, "AndLit");
                slots[*dst as usize] = slots[*a as usize] & *val;
            }
            Op::ShrLit { dst, a, val } => {
                count_op!(profile, "ShrLit");
                slots[*dst as usize] = slots[*a as usize] >> *val;
            }
            Op::ShlLit { dst, a, val } => {
                count_op!(profile, "ShlLit");
                slots[*dst as usize] = slots[*a as usize] << *val;
            }
            Op::ModLit { dst, a, val } => {
                count_op!(profile, "ModLit");
                let divisor = *val;
                slots[*dst as usize] = if divisor == 0 { 0 } else { slots[*a as usize] % divisor };
            }
            Op::Bit { dst, val, idx } => {
                count_op!(profile, "Bit");
                let v = slots[*val as usize];
                let i = slots[*idx as usize] as u32;
                slots[*dst as usize] = if i >= 32 { 0 } else { (v >> i) & 1 };
            }
            Op::BitAnd16 { dst, a, b } => {
                count_op!(profile, "BitAnd16");
                let av = slots[*a as usize] as u32 & 0xFFFF;
                let bv = slots[*b as usize] as u32 & 0xFFFF;
                slots[*dst as usize] = (av & bv) as i32;
            }
            Op::BitOr16 { dst, a, b } => {
                count_op!(profile, "BitOr16");
                let av = slots[*a as usize] as u32 & 0xFFFF;
                let bv = slots[*b as usize] as u32 & 0xFFFF;
                slots[*dst as usize] = (av | bv) as i32;
            }
            Op::BitXor16 { dst, a, b } => {
                count_op!(profile, "BitXor16");
                let av = slots[*a as usize] as u32 & 0xFFFF;
                let bv = slots[*b as usize] as u32 & 0xFFFF;
                slots[*dst as usize] = (av ^ bv) as i32;
            }
            Op::BitNot16 { dst, a } => {
                count_op!(profile, "BitNot16");
                let av = slots[*a as usize] as u32 & 0xFFFF;
                slots[*dst as usize] = ((!av) & 0xFFFF) as i32;
            }
            Op::CmpEq { dst, a, b } => {
                count_op!(profile, "CmpEq");
                slots[*dst as usize] = if slots[*a as usize] == slots[*b as usize] {
                    1
                } else {
                    0
                };
            }
            Op::BranchIfZero { cond, target } => {
                count_op!(profile, "BranchIfZero");
                if slots[*cond as usize] == 0 {
                    profile.branches_taken += 1;
                    pc = *target as usize;
                    continue;
                } else {
                    profile.branches_not_taken += 1;
                }
            }
            Op::BranchIfNotEqLit { a, val, target } => {
                count_op!(profile, "BranchIfNotEqLit");
                if slots[*a as usize] != *val {
                    profile.branches_taken += 1;
                    pc = *target as usize;
                    continue;
                } else {
                    profile.branches_not_taken += 1;
                }
            }
            Op::LoadStateAndBranchIfNotEqLit { dst, addr, val, target } => {
                count_op!(profile, "LoadStateAndBranchIfNotEqLit");
                let v = state.read_mem(*addr);
                slots[*dst as usize] = v;
                if v != *val {
                    profile.branches_taken += 1;
                    pc = *target as usize;
                    continue;
                } else {
                    profile.branches_not_taken += 1;
                }
            }
            Op::BranchIfNotEqLit2 { a1, val1, a2, val2, target } => {
                count_op!(profile, "BranchIfNotEqLit2");
                if slots[*a1 as usize] != *val1 || slots[*a2 as usize] != *val2 {
                    profile.branches_taken += 1;
                    pc = *target as usize;
                    continue;
                } else {
                    profile.branches_not_taken += 1;
                    pc += 2;
                    continue;
                }
            }
            Op::Jump { target } => {
                count_op!(profile, "Jump");
                pc = *target as usize;
                continue;
            }
            Op::DispatchChain { a, chain_id, miss_target } => {
                count_op!(profile, "DispatchChain");
                let key = slots[*a as usize];
                let table = &chain_tables[*chain_id as usize];
                match table.entries.get(&key) {
                    Some(&body_pc) => { pc = body_pc as usize; }
                    None => { pc = *miss_target as usize; }
                }
                continue;
            }
            Op::Dispatch {
                dst, key, table_id, ..
            } => {
                count_op!(profile, "Dispatch");
                profile.dispatch_count += 1;
                let td = Instant::now();
                let key_val = slots[*key as usize] as i64;
                let table = &dispatch_tables[*table_id as usize];
                if let Some((entry_ops, result_slot)) = table.entries.get(&key_val) {
                    profile.dispatch_sub_ops_count += entry_ops.len() as u64;
                    exec_ops(entry_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots);
                    slots[*dst as usize] = slots[*result_slot as usize];
                } else {
                    profile.dispatch_sub_ops_count += table.fallback_ops.len() as u64;
                    exec_ops(&table.fallback_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots);
                    slots[*dst as usize] = slots[table.fallback_slot as usize];
                }
                profile.dispatch_time += td.elapsed().as_secs_f64();
            }
            Op::DispatchFlatArray { dst, key, array_id, base_key, default } => {
                count_op!(profile, "DispatchFlatArray");
                let key_val = slots[*key as usize];
                let arr = &flat_dispatch_arrays[*array_id as usize];
                let idx = key_val.wrapping_sub(*base_key);
                let v = if idx < 0 || (idx as usize) >= arr.values.len() {
                    *default
                } else {
                    arr.values[idx as usize]
                };
                slots[*dst as usize] = v;
            }
            Op::Call { dst, fn_id, arg_slots } => {
                count_op!(profile, "Call");
                let func = &functions[*fn_id as usize];
                for (arg, &param) in arg_slots.iter().zip(func.param_slots.iter()) {
                    slots[param as usize] = slots[*arg as usize];
                }
                exec_ops(&func.body_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots);
                slots[*dst as usize] = slots[func.result_slot as usize];
            }
            Op::StoreState { addr, src } => {
                count_op!(profile, "StoreState");
                state.write_mem(*addr, slots[*src as usize]);
            }
            Op::StoreMem { addr_slot, src } => {
                count_op!(profile, "StoreMem");
                state.write_mem(slots[*addr_slot as usize], slots[*src as usize]);
            }
            Op::MemoryFill { dst_slot, val_slot, count_slot, exit_target } => {
                count_op!(profile, "MemoryFill");
                let dst = slots[*dst_slot as usize];
                let count = slots[*count_slot as usize];
                let val_byte = (slots[*val_slot as usize] & 0xFF) as u8;
                let mem_len = effective_guest_mem_end(state);
                if count > 0 && dst >= 0 && (dst as usize) < mem_len {
                    let lo = dst as usize;
                    let hi = (dst as i64 + count as i64).min(mem_len as i64) as usize;
                    if hi > lo {
                        state.bulk_fill_byte(lo, hi - lo, val_byte);
                    }
                }
                slots[*dst_slot as usize] = dst.wrapping_add(count);
                slots[*count_slot as usize] = 0;
                pc = *exit_target as usize;
                continue;
            }
            Op::MemoryCopy { src_slot, dst_slot, count_slot, exit_target } => {
                count_op!(profile, "MemoryCopy");
                let src = slots[*src_slot as usize];
                let dst = slots[*dst_slot as usize];
                let count = slots[*count_slot as usize];
                let mem_len = effective_guest_mem_end(state);
                if count > 0 && src >= 0 && dst >= 0
                    && (src as usize) < mem_len && (dst as usize) < mem_len
                {
                    let max_by_src = (mem_len as i64) - src as i64;
                    let max_by_dst = (mem_len as i64) - dst as i64;
                    let n = (count as i64).min(max_by_src).min(max_by_dst) as usize;
                    if n > 0 {
                        let s = src as usize;
                        let d = dst as usize;
                        state.bulk_copy_bytes(s, d, n);
                    }
                }
                slots[*src_slot as usize] = src.wrapping_add(count);
                slots[*dst_slot as usize] = dst.wrapping_add(count);
                slots[*count_slot as usize] = 0;
                pc = *exit_target as usize;
                continue;
            }
            Op::ReplicatedBody { body, strides, reps } => {
                count_op!(profile, "ReplicatedBody");
                // Profiled path delegates to the production runner — accuracy
                // of inner-op counts is sacrificed in favour of code size.
                // The outer loop still increments main_ops_count once per
                // ReplicatedBody dispatch, which is the meaningful thing to
                // measure (we collapsed N inner-loop iterations into 1).
                exec_replicated_body(
                    body, strides, *reps,
                    dispatch_tables, chain_tables, flat_dispatch_arrays,
                    functions, packed_cell_tables, packed_exception_tables,
                    state, slots,
                );
            }
        }
        pc += 1;
    }
}

// ---------------------------------------------------------------------------
// Traced execution (for debugging compiled vs interpreted divergence)
// ---------------------------------------------------------------------------

/// A single traced op execution.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TraceEntry {
    /// Program counter (op index in the ops array)
    pub pc: usize,
    /// Human-readable description of the op
    pub op: String,
    /// The op's destination slot (if any)
    pub dst_slot: Option<Slot>,
    /// The value written to the destination slot (if any)
    pub dst_value: Option<i32>,
    /// Key input slot values referenced by this op
    pub inputs: Vec<(Slot, i32)>,
    /// For branches: whether the branch was taken
    pub branch_taken: Option<bool>,
    /// Nesting depth (0 for main ops, incremented for dispatch sub-ops)
    pub depth: u32,
}

/// Execute ops with full tracing. Returns (result_slot_values, trace_log).
/// If `op_range` is Some((start, end)), only traces ops in that range.
/// `target_slot` is the slot whose computation we're interested in.
pub fn exec_ops_traced(
    ops: &[Op],
    dispatch_tables: &[CompiledDispatchTable],
    chain_tables: &[DispatchChainTable],
    flat_dispatch_arrays: &[FlatDispatchArray],
    functions: &[CompiledFunction],
    packed_cell_tables: &[Vec<i32>],
    packed_exception_tables: &[PackedExceptionTable],
    state: &mut State,
    slots: &mut [i32],
    op_range: Option<(usize, usize)>,
    depth: u32,
) -> Vec<TraceEntry> {
    let mut trace = Vec::new();
    let len = ops.len();
    let mut pc: usize = 0;
    let (range_start, range_end) = op_range.unwrap_or((0, len));
    let in_range = |pc: usize| pc >= range_start && pc < range_end;

    while pc < len {
        let should_trace = in_range(pc);
        match &ops[pc] {
            Op::LoadLit { dst, val } => {
                slots[*dst as usize] = *val;
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("LoadLit dst={} val={}", dst, val),
                        dst_slot: Some(*dst), dst_value: Some(*val),
                        inputs: vec![], branch_taken: None, depth,
                    });
                }
            }
            Op::LoadSlot { dst, src } => {
                let src_val = slots[*src as usize];
                slots[*dst as usize] = src_val;
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("LoadSlot dst={} src={}", dst, src),
                        dst_slot: Some(*dst), dst_value: Some(src_val),
                        inputs: vec![(*src, src_val)], branch_taken: None, depth,
                    });
                }
            }
            Op::LoadState { dst, addr } => {
                let val = state.read_mem(*addr);
                slots[*dst as usize] = val;
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("LoadState dst={} addr={}", dst, addr),
                        dst_slot: Some(*dst), dst_value: Some(val),
                        inputs: vec![], branch_taken: None, depth,
                    });
                }
            }
            Op::LoadMem { dst, addr_slot } => {
                let addr = slots[*addr_slot as usize];
                let val = state.read_mem(addr);
                slots[*dst as usize] = val;
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("LoadMem dst={} addr_slot={} addr={} → {}", dst, addr_slot, addr, val),
                        dst_slot: Some(*dst), dst_value: Some(val),
                        inputs: vec![(*addr_slot, addr)], branch_taken: None, depth,
                    });
                }
            }
            Op::LoadMem16 { dst, addr_slot } => {
                let addr = slots[*addr_slot as usize];
                let val = if addr < 0 { state.read_mem(addr) } else { state.read_mem16(addr) };
                slots[*dst as usize] = val;
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("LoadMem16 dst={} addr_slot={} addr={} → {}", dst, addr_slot, addr, val),
                        dst_slot: Some(*dst), dst_value: Some(val),
                        inputs: vec![(*addr_slot, addr)], branch_taken: None, depth,
                    });
                }
            }
            Op::LoadPackedByte { dst, key_slot, table_id, pack } => {
                let key = slots[*key_slot as usize];
                let tid = *table_id as usize;
                let val = if key < 0 {
                    0
                } else if let Some(lit) = packed_exception_tables[tid].get(key) {
                    lit
                } else {
                    let pack_i = *pack as i32;
                    let n = (key / pack_i) as usize;
                    let off = key % pack_i;
                    let table = &packed_cell_tables[tid];
                    let cell_addr = if n < table.len() { table[n] } else { 0 };
                    let cell = if cell_addr == 0 { 0 } else { state.read_mem(cell_addr) };
                    (cell >> (off * 8)) & 0xFF
                };
                slots[*dst as usize] = val;
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("LoadPackedByte dst={} key_slot={} key={} table_id={} pack={} → {}", dst, key_slot, key, table_id, pack, val),
                        dst_slot: Some(*dst), dst_value: Some(val),
                        inputs: vec![(*key_slot, key)], branch_taken: None, depth,
                    });
                }
            }
            Op::Add { dst, a, b } => {
                let av = slots[*a as usize];
                let bv = slots[*b as usize];
                let val = av.wrapping_add(bv);
                slots[*dst as usize] = val;
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("Add dst={} a={}({}) b={}({}) → {}", dst, a, av, b, bv, val),
                        dst_slot: Some(*dst), dst_value: Some(val),
                        inputs: vec![(*a, av), (*b, bv)], branch_taken: None, depth,
                    });
                }
            }
            Op::Sub { dst, a, b } => {
                let av = slots[*a as usize];
                let bv = slots[*b as usize];
                let val = av.wrapping_sub(bv);
                slots[*dst as usize] = val;
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("Sub dst={} a={}({}) b={}({}) → {}", dst, a, av, b, bv, val),
                        dst_slot: Some(*dst), dst_value: Some(val),
                        inputs: vec![(*a, av), (*b, bv)], branch_taken: None, depth,
                    });
                }
            }
            Op::Mul { dst, a, b } => {
                let av = slots[*a as usize];
                let bv = slots[*b as usize];
                let val = av.wrapping_mul(bv);
                slots[*dst as usize] = val;
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("Mul dst={} a={}({}) b={}({}) → {}", dst, a, av, b, bv, val),
                        dst_slot: Some(*dst), dst_value: Some(val),
                        inputs: vec![(*a, av), (*b, bv)], branch_taken: None, depth,
                    });
                }
            }
            Op::AddLit { dst, a, val } => {
                let av = slots[*a as usize];
                let r = av.wrapping_add(*val);
                slots[*dst as usize] = r;
                if should_trace {
                    trace.push(TraceEntry { pc, op: format!("AddLit dst={} a={}({}) +{} → {}", dst, a, av, val, r), dst_slot: Some(*dst), dst_value: Some(r), inputs: vec![(*a, av)], branch_taken: None, depth });
                }
            }
            Op::SubLit { dst, a, val } => {
                let av = slots[*a as usize];
                let r = av.wrapping_sub(*val);
                slots[*dst as usize] = r;
                if should_trace {
                    trace.push(TraceEntry { pc, op: format!("SubLit dst={} a={}({}) -{} → {}", dst, a, av, val, r), dst_slot: Some(*dst), dst_value: Some(r), inputs: vec![(*a, av)], branch_taken: None, depth });
                }
            }
            Op::MulLit { dst, a, val } => {
                let av = slots[*a as usize];
                let r = av.wrapping_mul(*val);
                slots[*dst as usize] = r;
                if should_trace {
                    trace.push(TraceEntry { pc, op: format!("MulLit dst={} a={}({}) *{} → {}", dst, a, av, val, r), dst_slot: Some(*dst), dst_value: Some(r), inputs: vec![(*a, av)], branch_taken: None, depth });
                }
            }
            Op::AndLit { dst, a, val } => {
                let av = slots[*a as usize];
                let r = av & *val;
                slots[*dst as usize] = r;
                if should_trace { trace.push(TraceEntry { pc, op: format!("AndLit dst={} a={}({}) &{} → {}", dst, a, av, val, r), dst_slot: Some(*dst), dst_value: Some(r), inputs: vec![(*a, av)], branch_taken: None, depth }); }
            }
            Op::ShrLit { dst, a, val } => {
                let av = slots[*a as usize];
                let r = av >> *val;
                slots[*dst as usize] = r;
                if should_trace { trace.push(TraceEntry { pc, op: format!("ShrLit dst={} a={}({}) >>{} → {}", dst, a, av, val, r), dst_slot: Some(*dst), dst_value: Some(r), inputs: vec![(*a, av)], branch_taken: None, depth }); }
            }
            Op::ShlLit { dst, a, val } => {
                let av = slots[*a as usize];
                let r = av << *val;
                slots[*dst as usize] = r;
                if should_trace { trace.push(TraceEntry { pc, op: format!("ShlLit dst={} a={}({}) <<{} → {}", dst, a, av, val, r), dst_slot: Some(*dst), dst_value: Some(r), inputs: vec![(*a, av)], branch_taken: None, depth }); }
            }
            Op::ModLit { dst, a, val } => {
                let av = slots[*a as usize];
                let r = if *val == 0 { 0 } else { av % *val };
                slots[*dst as usize] = r;
                if should_trace { trace.push(TraceEntry { pc, op: format!("ModLit dst={} a={}({}) %{} → {}", dst, a, av, val, r), dst_slot: Some(*dst), dst_value: Some(r), inputs: vec![(*a, av)], branch_taken: None, depth }); }
            }
            Op::Div { dst, a, b } => {
                let av = slots[*a as usize]; let bv = slots[*b as usize];
                let val = if bv == 0 { 0 } else { av / bv };
                slots[*dst as usize] = val;
                if should_trace { trace.push(TraceEntry { pc, op: format!("Div dst={} → {}", dst, val), dst_slot: Some(*dst), dst_value: Some(val), inputs: vec![(*a, av), (*b, bv)], branch_taken: None, depth }); }
            }
            Op::Mod { dst, a, b } => {
                let av = slots[*a as usize]; let bv = slots[*b as usize];
                let val = if bv == 0 { 0 } else { av % bv };
                slots[*dst as usize] = val;
                if should_trace { trace.push(TraceEntry { pc, op: format!("Mod dst={} → {}", dst, val), dst_slot: Some(*dst), dst_value: Some(val), inputs: vec![(*a, av), (*b, bv)], branch_taken: None, depth }); }
            }
            Op::Neg { dst, src } => {
                let val = slots[*src as usize].wrapping_neg();
                slots[*dst as usize] = val;
                if should_trace { trace.push(TraceEntry { pc, op: format!("Neg dst={} → {}", dst, val), dst_slot: Some(*dst), dst_value: Some(val), inputs: vec![(*src, slots[*src as usize])], branch_taken: None, depth }); }
            }
            Op::Abs { dst, src } => {
                let val = slots[*src as usize].wrapping_abs();
                slots[*dst as usize] = val;
                if should_trace { trace.push(TraceEntry { pc, op: format!("Abs dst={} → {}", dst, val), dst_slot: Some(*dst), dst_value: Some(val), inputs: vec![], branch_taken: None, depth }); }
            }
            Op::Sign { dst, src } => {
                let v = slots[*src as usize];
                let val = if v > 0 { 1 } else if v < 0 { -1 } else { 0 };
                slots[*dst as usize] = val;
                if should_trace { trace.push(TraceEntry { pc, op: format!("Sign dst={} → {}", dst, val), dst_slot: Some(*dst), dst_value: Some(val), inputs: vec![(*src, v)], branch_taken: None, depth }); }
            }
            Op::Pow { dst, base, exp } => {
                let b = slots[*base as usize]; let e = slots[*exp as usize];
                let val = if e < 0 { 0 } else { b.wrapping_pow(e as u32) };
                slots[*dst as usize] = val;
                if should_trace { trace.push(TraceEntry { pc, op: format!("Pow dst={} → {}", dst, val), dst_slot: Some(*dst), dst_value: Some(val), inputs: vec![(*base, b), (*exp, e)], branch_taken: None, depth }); }
            }
            Op::Min { dst, args } => {
                let mut v = i32::MAX; for &a in args { v = v.min(slots[a as usize]); }
                slots[*dst as usize] = v;
                if should_trace { trace.push(TraceEntry { pc, op: format!("Min dst={} → {}", dst, v), dst_slot: Some(*dst), dst_value: Some(v), inputs: args.iter().map(|&a| (a, slots[a as usize])).collect(), branch_taken: None, depth }); }
            }
            Op::Max { dst, args } => {
                let mut v = i32::MIN; for &a in args { v = v.max(slots[a as usize]); }
                slots[*dst as usize] = v;
                if should_trace { trace.push(TraceEntry { pc, op: format!("Max dst={} → {}", dst, v), dst_slot: Some(*dst), dst_value: Some(v), inputs: args.iter().map(|&a| (a, slots[a as usize])).collect(), branch_taken: None, depth }); }
            }
            Op::Clamp { dst, min, val, max } => {
                let val_v = slots[*val as usize].clamp(slots[*min as usize], slots[*max as usize]);
                slots[*dst as usize] = val_v;
                if should_trace { trace.push(TraceEntry { pc, op: format!("Clamp dst={} → {}", dst, val_v), dst_slot: Some(*dst), dst_value: Some(val_v), inputs: vec![], branch_taken: None, depth }); }
            }
            Op::Round { dst, strategy, val, interval } => {
                let v = slots[*val as usize]; let i = slots[*interval as usize];
                let val_r = if i == 0 { v } else {
                    match strategy {
                        RoundStrategy::Down => v.div_euclid(i) * i,
                        RoundStrategy::Up => (v + i - 1).div_euclid(i) * i,
                        RoundStrategy::Nearest => ((v + i / 2).div_euclid(i)) * i,
                        RoundStrategy::ToZero => (v / i) * i,
                    }
                };
                slots[*dst as usize] = val_r;
                if should_trace { trace.push(TraceEntry { pc, op: format!("Round dst={} → {}", dst, val_r), dst_slot: Some(*dst), dst_value: Some(val_r), inputs: vec![], branch_taken: None, depth }); }
            }
            Op::Floor { dst, src } => {
                slots[*dst as usize] = slots[*src as usize];
                if should_trace { trace.push(TraceEntry { pc, op: format!("Floor dst={}", dst), dst_slot: Some(*dst), dst_value: Some(slots[*dst as usize]), inputs: vec![], branch_taken: None, depth }); }
            }
            Op::And { dst, a, b } => {
                let av = slots[*a as usize]; let bv = slots[*b as usize] as u32;
                let val = if bv >= 32 { av } else { av & ((1i32 << bv) - 1) };
                slots[*dst as usize] = val;
                if should_trace { trace.push(TraceEntry { pc, op: format!("And dst={} a={}({}) b={}({}) → {}", dst, a, av, b, bv, val), dst_slot: Some(*dst), dst_value: Some(val), inputs: vec![(*a, av), (*b, bv as i32)], branch_taken: None, depth }); }
            }
            Op::Shr { dst, a, b } => {
                let av = slots[*a as usize]; let bv = slots[*b as usize] as u32;
                let val = if bv >= 32 { 0 } else { av >> bv };
                slots[*dst as usize] = val;
                if should_trace { trace.push(TraceEntry { pc, op: format!("Shr dst={} → {}", dst, val), dst_slot: Some(*dst), dst_value: Some(val), inputs: vec![(*a, av), (*b, bv as i32)], branch_taken: None, depth }); }
            }
            Op::Shl { dst, a, b } => {
                let av = slots[*a as usize]; let bv = slots[*b as usize] as u32;
                let val = if bv >= 32 { 0 } else { av << bv };
                slots[*dst as usize] = val;
                if should_trace { trace.push(TraceEntry { pc, op: format!("Shl dst={} → {}", dst, val), dst_slot: Some(*dst), dst_value: Some(val), inputs: vec![(*a, av), (*b, bv as i32)], branch_taken: None, depth }); }
            }
            Op::Bit { dst, val, idx } => {
                let v = slots[*val as usize]; let i = slots[*idx as usize] as u32;
                let val_r = if i >= 32 { 0 } else { (v >> i) & 1 };
                slots[*dst as usize] = val_r;
                if should_trace { trace.push(TraceEntry { pc, op: format!("Bit dst={} → {}", dst, val_r), dst_slot: Some(*dst), dst_value: Some(val_r), inputs: vec![], branch_taken: None, depth }); }
            }
            Op::BitAnd16 { dst, a, b } => {
                let av = slots[*a as usize] as u32 & 0xFFFF;
                let bv = slots[*b as usize] as u32 & 0xFFFF;
                let r = (av & bv) as i32;
                slots[*dst as usize] = r;
                if should_trace { trace.push(TraceEntry { pc, op: format!("BitAnd16 dst={} → {}", dst, r), dst_slot: Some(*dst), dst_value: Some(r), inputs: vec![(*a, av as i32), (*b, bv as i32)], branch_taken: None, depth }); }
            }
            Op::BitOr16 { dst, a, b } => {
                let av = slots[*a as usize] as u32 & 0xFFFF;
                let bv = slots[*b as usize] as u32 & 0xFFFF;
                let r = (av | bv) as i32;
                slots[*dst as usize] = r;
                if should_trace { trace.push(TraceEntry { pc, op: format!("BitOr16 dst={} → {}", dst, r), dst_slot: Some(*dst), dst_value: Some(r), inputs: vec![(*a, av as i32), (*b, bv as i32)], branch_taken: None, depth }); }
            }
            Op::BitXor16 { dst, a, b } => {
                let av = slots[*a as usize] as u32 & 0xFFFF;
                let bv = slots[*b as usize] as u32 & 0xFFFF;
                let r = (av ^ bv) as i32;
                slots[*dst as usize] = r;
                if should_trace { trace.push(TraceEntry { pc, op: format!("BitXor16 dst={} → {}", dst, r), dst_slot: Some(*dst), dst_value: Some(r), inputs: vec![(*a, av as i32), (*b, bv as i32)], branch_taken: None, depth }); }
            }
            Op::BitNot16 { dst, a } => {
                let av = slots[*a as usize] as u32 & 0xFFFF;
                let r = ((!av) & 0xFFFF) as i32;
                slots[*dst as usize] = r;
                if should_trace { trace.push(TraceEntry { pc, op: format!("BitNot16 dst={} → {}", dst, r), dst_slot: Some(*dst), dst_value: Some(r), inputs: vec![(*a, av as i32)], branch_taken: None, depth }); }
            }
            Op::CmpEq { dst, a, b } => {
                let av = slots[*a as usize]; let bv = slots[*b as usize];
                let val = if av == bv { 1 } else { 0 };
                slots[*dst as usize] = val;
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("CmpEq dst={} a={}({}) b={}({}) → {}", dst, a, av, b, bv, val),
                        dst_slot: Some(*dst), dst_value: Some(val),
                        inputs: vec![(*a, av), (*b, bv)], branch_taken: None, depth,
                    });
                }
            }
            Op::BranchIfZero { cond, target } => {
                let cv = slots[*cond as usize];
                let taken = cv == 0;
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("BranchIfZero cond={}({}) target={} {}", cond, cv, target, if taken { "TAKEN" } else { "not taken" }),
                        dst_slot: None, dst_value: None,
                        inputs: vec![(*cond, cv)], branch_taken: Some(taken), depth,
                    });
                }
                if taken {
                    pc = *target as usize;
                    continue;
                }
            }
            Op::BranchIfNotEqLit { a, val, target } => {
                let a_val = slots[*a as usize];
                let taken = a_val != *val;
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("BranchIfNotEqLit a={} val={} target={}", a, val, target),
                        dst_slot: None, dst_value: None,
                        inputs: vec![(*a, a_val)], branch_taken: Some(taken), depth,
                    });
                }
                if taken {
                    pc = *target as usize;
                    continue;
                }
            }
            Op::LoadStateAndBranchIfNotEqLit { dst, addr, val, target } => {
                let v = state.read_mem(*addr);
                slots[*dst as usize] = v;
                let taken = v != *val;
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("LoadStateAndBranchIfNotEqLit dst={} addr={} val={} target={}", dst, addr, val, target),
                        dst_slot: Some(*dst), dst_value: Some(v),
                        inputs: vec![], branch_taken: Some(taken), depth,
                    });
                }
                if taken {
                    pc = *target as usize;
                    continue;
                }
            }
            Op::BranchIfNotEqLit2 { a1, val1, a2, val2, target } => {
                let v1 = slots[*a1 as usize];
                let v2 = slots[*a2 as usize];
                let taken = v1 != *val1 || v2 != *val2;
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("BranchIfNotEqLit2 a1={} val1={} a2={} val2={} target={}", a1, val1, a2, val2, target),
                        dst_slot: None, dst_value: None,
                        inputs: vec![(*a1, v1), (*a2, v2)], branch_taken: Some(taken), depth,
                    });
                }
                if taken {
                    pc = *target as usize;
                    continue;
                } else {
                    pc += 2;
                    continue;
                }
            }
            Op::Jump { target } => {
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("Jump target={}", target),
                        dst_slot: None, dst_value: None,
                        inputs: vec![], branch_taken: None, depth,
                    });
                }
                pc = *target as usize;
                continue;
            }
            Op::DispatchChain { a, chain_id, miss_target } => {
                let key = slots[*a as usize];
                let table = &chain_tables[*chain_id as usize];
                let (next_pc, hit) = match table.entries.get(&key) {
                    Some(&body_pc) => (body_pc as usize, true),
                    None => (*miss_target as usize, false),
                };
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("DispatchChain a={}({}) chain={} → {} ({})", a, key, chain_id, next_pc, if hit { "hit" } else { "miss" }),
                        dst_slot: None, dst_value: None,
                        inputs: vec![(*a, key)], branch_taken: Some(hit), depth,
                    });
                }
                pc = next_pc;
                continue;
            }
            Op::Dispatch { dst, key, table_id, .. } => {
                let key_val = slots[*key as usize] as i64;
                let table = &dispatch_tables[*table_id as usize];
                if let Some((entry_ops, result_slot)) = table.entries.get(&key_val) {
                    if should_trace {
                        let mut sub = exec_ops_traced(entry_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots, None, depth + 1);
                        trace.push(TraceEntry {
                            pc, op: format!("Dispatch dst={} key={}({}) table={} → entry (result_slot={}={})", dst, key, key_val, table_id, result_slot, slots[*result_slot as usize]),
                            dst_slot: Some(*dst), dst_value: Some(slots[*result_slot as usize]),
                            inputs: vec![(*key, key_val as i32)], branch_taken: None, depth,
                        });
                        trace.append(&mut sub);
                    } else {
                        exec_ops(entry_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots);
                    }
                    slots[*dst as usize] = slots[*result_slot as usize];
                } else {
                    if should_trace {
                        let mut sub = exec_ops_traced(&table.fallback_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots, None, depth + 1);
                        exec_ops(&table.fallback_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots);
                        trace.push(TraceEntry {
                            pc, op: format!("Dispatch dst={} key={}({}) table={} → fallback (fallback_slot={}={})", dst, key, key_val, table_id, table.fallback_slot, slots[table.fallback_slot as usize]),
                            dst_slot: Some(*dst), dst_value: Some(slots[table.fallback_slot as usize]),
                            inputs: vec![(*key, key_val as i32)], branch_taken: None, depth,
                        });
                        trace.append(&mut sub);
                    } else {
                        exec_ops(&table.fallback_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots);
                    }
                    slots[*dst as usize] = slots[table.fallback_slot as usize];
                }
            }
            Op::DispatchFlatArray { dst, key, array_id, base_key, default } => {
                let key_val = slots[*key as usize];
                let arr = &flat_dispatch_arrays[*array_id as usize];
                let idx = key_val.wrapping_sub(*base_key);
                let v = if idx < 0 || (idx as usize) >= arr.values.len() {
                    *default
                } else {
                    arr.values[idx as usize]
                };
                slots[*dst as usize] = v;
                if should_trace {
                    trace.push(TraceEntry {
                        pc, op: format!("DispatchFlatArray dst={} key={}({}) array={} base={} default={} → {}", dst, key, key_val, array_id, base_key, default, v),
                        dst_slot: Some(*dst), dst_value: Some(v),
                        inputs: vec![(*key, key_val)], branch_taken: None, depth,
                    });
                }
            }
            Op::Call { dst, fn_id, arg_slots } => {
                let func = &functions[*fn_id as usize];
                for (arg, &param) in arg_slots.iter().zip(func.param_slots.iter()) {
                    slots[param as usize] = slots[*arg as usize];
                }
                if should_trace {
                    let mut sub = exec_ops_traced(&func.body_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots, None, depth + 1);
                    let result = slots[func.result_slot as usize];
                    trace.push(TraceEntry {
                        pc, op: format!("Call dst={} fn_id={} args={:?} → {}", dst, fn_id, arg_slots, result),
                        dst_slot: Some(*dst), dst_value: Some(result),
                        inputs: arg_slots.iter().map(|&s| (s, slots[s as usize])).collect(),
                        branch_taken: None, depth,
                    });
                    trace.append(&mut sub);
                } else {
                    exec_ops(&func.body_ops, dispatch_tables, chain_tables, flat_dispatch_arrays, functions, packed_cell_tables, packed_exception_tables, state, slots);
                }
                slots[*dst as usize] = slots[func.result_slot as usize];
            }
            Op::StoreState { addr, src } => {
                let val = slots[*src as usize];
                state.write_mem(*addr, val);
                if should_trace { trace.push(TraceEntry { pc, op: format!("StoreState addr={} src={}({})", addr, src, val), dst_slot: None, dst_value: None, inputs: vec![(*src, val)], branch_taken: None, depth }); }
            }
            Op::StoreMem { addr_slot, src } => {
                let addr = slots[*addr_slot as usize]; let val = slots[*src as usize];
                state.write_mem(addr, val);
                if should_trace { trace.push(TraceEntry { pc, op: format!("StoreMem addr={}({}) src={}({})", addr_slot, addr, src, val), dst_slot: None, dst_value: None, inputs: vec![(*addr_slot, addr), (*src, val)], branch_taken: None, depth }); }
            }
            Op::MemoryFill { dst_slot, val_slot, count_slot, exit_target } => {
                let dst = slots[*dst_slot as usize];
                let count = slots[*count_slot as usize];
                let val = slots[*val_slot as usize];
                let val_byte = (val & 0xFF) as u8;
                let mem_len = effective_guest_mem_end(state);
                if count > 0 && dst >= 0 && (dst as usize) < mem_len {
                    let lo = dst as usize;
                    let hi = (dst as i64 + count as i64).min(mem_len as i64) as usize;
                    if hi > lo {
                        state.bulk_fill_byte(lo, hi - lo, val_byte);
                    }
                }
                let new_dst = dst.wrapping_add(count);
                slots[*dst_slot as usize] = new_dst;
                slots[*count_slot as usize] = 0;
                if should_trace {
                    trace.push(TraceEntry {
                        pc,
                        op: format!("MemoryFill dst={}({}) val={}({:#x}) count={}({}) → exit={}",
                            dst_slot, dst, val_slot, val_byte, count_slot, count, exit_target),
                        dst_slot: Some(*dst_slot),
                        dst_value: Some(new_dst),
                        inputs: vec![(*dst_slot, dst), (*val_slot, val), (*count_slot, count)],
                        branch_taken: None,
                        depth,
                    });
                }
                pc = *exit_target as usize;
                continue;
            }
            Op::MemoryCopy { src_slot, dst_slot, count_slot, exit_target } => {
                let src = slots[*src_slot as usize];
                let dst = slots[*dst_slot as usize];
                let count = slots[*count_slot as usize];
                let mem_len = effective_guest_mem_end(state);
                if count > 0 && src >= 0 && dst >= 0
                    && (src as usize) < mem_len && (dst as usize) < mem_len
                {
                    let max_by_src = (mem_len as i64) - src as i64;
                    let max_by_dst = (mem_len as i64) - dst as i64;
                    let n = (count as i64).min(max_by_src).min(max_by_dst) as usize;
                    if n > 0 {
                        let s = src as usize;
                        let d = dst as usize;
                        state.bulk_copy_bytes(s, d, n);
                    }
                }
                let new_src = src.wrapping_add(count);
                let new_dst = dst.wrapping_add(count);
                slots[*src_slot as usize] = new_src;
                slots[*dst_slot as usize] = new_dst;
                slots[*count_slot as usize] = 0;
                if should_trace {
                    trace.push(TraceEntry {
                        pc,
                        op: format!("MemoryCopy src={}({}) dst={}({}) count={}({}) → exit={}",
                            src_slot, src, dst_slot, dst, count_slot, count, exit_target),
                        dst_slot: Some(*dst_slot),
                        dst_value: Some(new_dst),
                        inputs: vec![(*src_slot, src), (*dst_slot, dst), (*count_slot, count)],
                        branch_taken: None,
                        depth,
                    });
                }
                pc = *exit_target as usize;
                continue;
            }
            Op::ReplicatedBody { body, strides, reps } => {
                if should_trace {
                    trace.push(TraceEntry {
                        pc,
                        op: format!("ReplicatedBody body_len={} reps={}", body.len(), reps),
                        dst_slot: None,
                        dst_value: None,
                        inputs: vec![],
                        branch_taken: None,
                        depth,
                    });
                }
                exec_replicated_body(
                    body, strides, *reps,
                    dispatch_tables, chain_tables, flat_dispatch_arrays,
                    functions, packed_cell_tables, packed_exception_tables,
                    state, slots,
                );
            }
        }
        pc += 1;
    }
    trace
}

/// Trace execution of a specific property's computation.
/// Returns the trace log for ops in the range that computes the named property.
pub fn trace_property(
    program: &CompiledProgram,
    state: &mut State,
    slots: &mut Vec<i32>,
    property_name: &str,
) -> Option<(i32, Vec<TraceEntry>)> {
    let target_slot = *program.property_slots.get(property_name)?;

    // Reset slots
    slots.clear();
    slots.resize(program.slot_count as usize, 0);

    // Find the op range for this property by looking for the StoreState that
    // writes to the property's state address, or the last LoadSlot that writes
    // to the property's result slot. We trace the entire ops since the property
    // computation may depend on earlier ops (state loads, condition checks).
    // But to limit output, we'll trace only ops that are "near" the target slot.
    //
    // Simple approach: run all ops with full tracing, then filter to only
    // the ops that touch the target slot or lead to it. Since this is a
    // debugging tool (not hot path), performance doesn't matter.
    let trace = exec_ops_traced(
        &program.ops,
        &program.dispatch_tables,
        &program.chain_tables,
        &program.flat_dispatch_arrays,
        &program.functions,
        &program.packed_cell_tables,
        &program.packed_exception_tables,
        state,
        slots,
        None, // trace all ops
        0,
    );

    let result_value = slots[target_slot as usize];

    // Filter trace to relevant ops: find the writeback that stores to target_slot,
    // then walk backward through the trace to find all ops that feed into it.
    // For now, return all ops that wrote to the target slot or were branches in the
    // path, plus a window around them.
    let mut relevant_slots: std::collections::HashSet<u32> = std::collections::HashSet::new();
    relevant_slots.insert(target_slot);

    // Walk backward through trace to find dependency chain
    let mut relevant: Vec<TraceEntry> = Vec::new();
    for entry in trace.iter().rev() {
        let dominated = entry.dst_slot.map(|s| relevant_slots.contains(&s)).unwrap_or(false);
        let is_branch = entry.branch_taken.is_some();
        if dominated || is_branch || entry.depth > 0 {
            if dominated {
                // Add input slots as dependencies
                for &(slot, _) in &entry.inputs {
                    relevant_slots.insert(slot);
                }
            }
            relevant.push(entry.clone());
        }
    }
    relevant.reverse();

    Some((result_value, relevant))
}

// ---------------------------------------------------------------------------
// Helpers (duplicated from eval.rs to avoid coupling)
// ---------------------------------------------------------------------------

fn is_buffer_copy(name: &str) -> bool {
    name.starts_with("--__0") || name.starts_with("--__1") || name.starts_with("--__2")
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const STATE_VAR_NAMES: &[&str] = &[
        "AX", "CX", "DX", "BX", "SP", "BP", "SI", "DI",
        "IP", "ES", "CS", "SS", "DS", "flags",
    ];

    /// Install a test address map and return a State with matching state vars.
    fn setup() -> State {
        let mut map = std::collections::HashMap::new();
        let mut state = State::default();
        for (i, &name) in STATE_VAR_NAMES.iter().enumerate() {
            map.insert(name.to_string(), -(i as i32 + 1));
            state.state_vars.push(0);
            state.state_var_names.push(name.to_string());
            state.state_var_index.insert(name.to_string(), i);
        }
        crate::eval::set_address_map(map);
        state
    }

    // --- copy_propagate_and_dce / copy_opt_stream ---

    fn run(ops: &[Op], n_slots: usize) -> Vec<i32> {
        let mut state = State::default();
        let mut slots = vec![0i32; n_slots];
        exec_ops(ops, &[], &[], &[], &[], &[], &[], &mut state, &mut slots);
        slots
    }

    fn ext(slots: &[Slot]) -> FxSet<Slot> {
        slots.iter().copied().collect()
    }

    fn opt_stream(
        ops: &mut Vec<Op>,
        external: &FxSet<Slot>,
        table_reads: &[FxSet<Slot>],
    ) -> (usize, usize) {
        copy_opt_stream(ops, external, table_reads, &mut CopyOptScratch::default())
    }

    #[test]
    fn copy_elim_collapses_chain() {
        let mut ops = vec![
            Op::LoadLit { dst: 0, val: 7 },
            Op::LoadSlot { dst: 1, src: 0 },
            Op::LoadSlot { dst: 2, src: 1 },
            Op::Add { dst: 3, a: 2, b: 2 },
        ];
        let before = run(&ops, 8);
        let (prop, removed) = opt_stream(&mut ops,&ext(&[3]), &[]);
        assert!(prop >= 2, "expected chain reads redirected, got {prop}");
        assert_eq!(removed, 2, "both copies should die");
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[1], Op::Add { dst: 3, a: 0, b: 0 }));
        let after = run(&ops, 8);
        assert_eq!(before[3], after[3]);
        assert_eq!(after[3], 14);
    }

    #[test]
    fn copy_elim_diamond_facts_survive_labels() {
        // 0: LoadLit cond
        // 1: LoadSlot 1 <- 0 (the copy under test)
        // 2: BranchIfZero(cond=1) -> 5
        // 3: AddLit 2 = slot1 + 100
        // 4: Jump -> 6
        // 5: AddLit 2 = slot1 + 200    (label, single pred)
        // 6: Add 3 = slot2 + slot1     (join label, two preds)
        let build = |k: i32| {
            vec![
                Op::LoadLit { dst: 0, val: k },
                Op::LoadSlot { dst: 1, src: 0 },
                Op::BranchIfZero { cond: 1, target: 5 },
                Op::AddLit { dst: 2, a: 1, val: 100 },
                Op::Jump { target: 6 },
                Op::AddLit { dst: 2, a: 1, val: 200 },
                Op::Add { dst: 3, a: 2, b: 1 },
            ]
        };
        for k in [0, 5] {
            let mut ops = build(k);
            let before = run(&ops, 8);
            let (_, removed) = opt_stream(&mut ops,&ext(&[3]), &[]);
            assert_eq!(removed, 1, "copy should die after facts cross the labels (k={k})");
            assert_eq!(ops.len(), 6);
            let after = run(&ops, 8);
            assert_eq!(before[3], after[3], "semantics changed for k={k}");
        }
    }

    #[test]
    fn copy_elim_copy_back_is_noop() {
        let mut ops = vec![
            Op::LoadLit { dst: 0, val: 9 },
            Op::LoadSlot { dst: 1, src: 0 },
            Op::LoadSlot { dst: 0, src: 1 }, // copy-back: a no-op once src is rewritten
            Op::Add { dst: 2, a: 0, b: 1 },
        ];
        let before = run(&ops, 8);
        let (_, removed) = opt_stream(&mut ops,&ext(&[2]), &[]);
        assert_eq!(removed, 2);
        assert_eq!(ops.len(), 2);
        let after = run(&ops, 8);
        assert_eq!(before[2], after[2]);
        assert_eq!(after[2], 18);
    }

    #[test]
    fn copy_elim_dead_chain_cascade() {
        let mut ops = vec![
            Op::LoadLit { dst: 0, val: 1 },
            Op::LoadLit { dst: 1, val: 2 },
            Op::Add { dst: 2, a: 0, b: 1 },
            Op::LoadLit { dst: 3, val: 5 },
        ];
        let (_, removed) = opt_stream(&mut ops,&ext(&[3]), &[]);
        assert_eq!(removed, 3, "whole dead compute chain should cascade away");
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], Op::LoadLit { dst: 3, val: 5 }));
    }

    #[test]
    fn copy_elim_dispatch_kills_facts_and_keeps_table_reads() {
        let mut table_reads: Vec<FxSet<Slot>> = vec![FxSet::default()];
        table_reads[0].insert(7);
        let mut ops = vec![
            Op::LoadLit { dst: 5, val: 1 },
            Op::LoadSlot { dst: 7, src: 5 }, // read inside the dispatch entry
            Op::Dispatch { dst: 8, key: 5, table_id: 0, fallback_target: 0 },
            Op::Add { dst: 9, a: 8, b: 7 },
        ];
        let (_, removed) = opt_stream(&mut ops,&ext(&[9]), &table_reads);
        assert_eq!(removed, 0, "copy feeding a dispatch entry must survive");
        assert_eq!(ops.len(), 4);
        // The read of slot 7 after the Dispatch must NOT have been rewritten
        // to 5 — the dispatch entry may alias-write either slot.
        assert!(matches!(ops[3], Op::Add { dst: 9, a: 8, b: 7 }));
    }

    #[test]
    fn copy_elim_external_dst_kept() {
        let mut ops = vec![
            Op::LoadLit { dst: 0, val: 3 },
            Op::LoadSlot { dst: 10, src: 0 }, // external (e.g. writeback)
            Op::LoadSlot { dst: 11, src: 0 }, // dead
        ];
        let (_, removed) = opt_stream(&mut ops,&ext(&[10]), &[]);
        // The dead copy dies, and store-forwarding folds the external copy
        // into its producer: LoadLit writes slot 10 directly.
        assert_eq!(removed, 2);
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], Op::LoadLit { dst: 10, val: 3 }));
    }

    #[test]
    fn copy_elim_store_forwarding_result_copy() {
        // The dispatch-entry shape: compute, then copy into the result slot.
        let mut ops = vec![
            Op::LoadLit { dst: 0, val: 6 },
            Op::AddLit { dst: 1, a: 0, val: 1 },
            Op::LoadSlot { dst: 2, src: 1 }, // result copy
        ];
        let before = run(&ops, 8);
        let (_, removed) = opt_stream(&mut ops,&ext(&[2]), &[]);
        assert_eq!(removed, 1);
        assert!(matches!(ops[1], Op::AddLit { dst: 2, a: 0, val: 1 }));
        let after = run(&ops, 8);
        assert_eq!(before[2], after[2]);
        assert_eq!(after[2], 7);
    }

    #[test]
    fn copy_elim_no_forwarding_across_label() {
        // A branch target sits on the copy: another path reaches the copy
        // without the producer, so the producer must not be retargeted.
        let build = |k: i32| {
            vec![
                Op::LoadLit { dst: 0, val: k },
                Op::LoadLit { dst: 1, val: 10 },
                Op::BranchIfZero { cond: 0, target: 4 },
                Op::AddLit { dst: 1, a: 1, val: 90 },
                Op::LoadSlot { dst: 2, src: 1 }, // label: reached two ways
            ]
        };
        for k in [0, 1] {
            let mut ops = build(k);
            let before = run(&ops, 8);
            opt_stream(&mut ops,&ext(&[2]), &[]);
            let after = run(&ops, 8);
            assert_eq!(before[2], after[2], "label forwarding broke semantics for k={k}");
        }
    }

    #[test]
    fn copy_elim_no_forwarding_when_src_still_live() {
        let mut ops = vec![
            Op::LoadLit { dst: 0, val: 6 },
            Op::AddLit { dst: 1, a: 0, val: 1 },
            Op::LoadSlot { dst: 2, src: 1 }, // copy
            Op::Add { dst: 3, a: 1, b: 2 },  // still reads slot 1
        ];
        let before = run(&ops, 8);
        opt_stream(&mut ops,&ext(&[3]), &[]);
        let after = run(&ops, 8);
        assert_eq!(before[3], after[3]);
        assert_eq!(after[3], 14);
    }

    #[test]
    fn copy_elim_backward_edge_bails() {
        let mut ops = vec![
            Op::LoadLit { dst: 0, val: 3 },
            Op::LoadSlot { dst: 1, src: 0 },
            Op::BranchIfZero { cond: 1, target: 0 },
        ];
        let orig = format!("{ops:?}");
        let (prop, removed) = opt_stream(&mut ops,&ext(&[1]), &[]);
        assert_eq!((prop, removed), (0, 0));
        assert_eq!(format!("{ops:?}"), orig, "stream with backward edge must be untouched");
    }

    #[test]
    fn copy_elim_memoryfill_pointer_slots_not_rewritten() {
        let mut ops = vec![
            Op::LoadLit { dst: 0, val: 5 },
            Op::LoadSlot { dst: 1, src: 0 },
            Op::MemoryFill { dst_slot: 1, val_slot: 0, count_slot: 3, exit_target: 3 },
        ];
        let (_, removed) = opt_stream(&mut ops,&ext(&[1]), &[]);
        assert_eq!(removed, 0);
        // dst_slot is read+written: must keep pointing at slot 1, and the
        // copy producing slot 1 must stay.
        assert!(matches!(
            ops[2],
            Op::MemoryFill { dst_slot: 1, val_slot: 0, count_slot: 3, .. }
        ));
        assert!(matches!(ops[1], Op::LoadSlot { dst: 1, src: 0 }));
    }

    #[test]
    fn copy_elim_jump_target_remap_preserves_semantics() {
        // Dead copy sits between a branch and its target; removal shifts
        // indices, so targets must be remapped.
        let build = |k: i32| {
            vec![
                Op::LoadLit { dst: 0, val: k },
                Op::BranchIfZero { cond: 0, target: 4 },
                Op::LoadSlot { dst: 1, src: 0 }, // dead (slot 1 never read)
                Op::LoadLit { dst: 2, val: 111 },
                Op::AddLit { dst: 3, a: 2, val: 1 },
            ]
        };
        for k in [0, 1] {
            let mut ops = build(k);
            let before = run(&ops, 8);
            let (_, removed) = opt_stream(&mut ops,&ext(&[3]), &[]);
            assert_eq!(removed, 1);
            let after = run(&ops, 8);
            assert_eq!(before[3], after[3], "branch remap broke semantics for k={k}");
        }
    }

    #[test]
    fn compile_literal() {
        let expr = Expr::Literal(42.0);
        let empty_fns = HashMap::new();
        let mut empty_dt = HashMap::new();
        let mut compiler = Compiler::new(&empty_fns, &mut empty_dt);
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);
        assert_eq!(ops.len(), 1);

        let mut state = State::default();
        let mut slots = vec![0i32; compiler.next_slot as usize];
        exec_ops(&ops, &[], &[], &[], &[], &[], &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 42);
    }

    #[test]
    fn compile_calc_add() {
        let expr = Expr::Calc(CalcOp::Add(
            Box::new(Expr::Literal(10.0)),
            Box::new(Expr::Literal(20.0)),
        ));
        let empty_fns = HashMap::new();
        let mut empty_dt = HashMap::new();
        let mut compiler = Compiler::new(&empty_fns, &mut empty_dt);
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);

        let mut state = State::default();
        let mut slots = vec![0i32; compiler.next_slot as usize];
        exec_ops(&ops, &[], &[], &[], &[], &[], &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 30);
    }

    #[test]
    fn compile_var_from_state() {
        let mut state = setup();
        let expr = Expr::Var {
            name: "--AX".to_string(),
            fallback: None,
        };
        let empty_fns = HashMap::new();
        let mut empty_dt = HashMap::new();
        let mut compiler = Compiler::new(&empty_fns, &mut empty_dt);
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);

        state.set_var("AX", 0x1234);
        let mut slots = vec![0i32; compiler.next_slot as usize];
        exec_ops(&ops, &[], &[], &[], &[], &[], &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 0x1234);
    }

    #[test]
    fn compile_style_condition() {
        let mut state = setup();
        let expr = Expr::StyleCondition {
            branches: vec![
                StyleBranch {
                    condition: StyleTest::Single {
                        property: "--AX".to_string(),
                        value: Expr::Literal(1.0),
                    },
                    then: Expr::Literal(100.0),
                },
                StyleBranch {
                    condition: StyleTest::Single {
                        property: "--AX".to_string(),
                        value: Expr::Literal(2.0),
                    },
                    then: Expr::Literal(200.0),
                },
            ],
            fallback: Box::new(Expr::Literal(0.0)),
        };
        let empty_fns = HashMap::new();
        let mut empty_dt = HashMap::new();
        let mut compiler = Compiler::new(&empty_fns, &mut empty_dt);
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);

        state.set_var("AX", 2);
        let mut slots = vec![0i32; compiler.next_slot as usize];
        exec_ops(&ops, &[], &[], &[], &[], &[], &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 200);
    }

    #[test]
    fn compile_readmem() {
        let mut state = setup(); // install address map before compilation
        // Build a dispatch table that maps key K → Var at state address K
        // (identity-read pattern, detected generically).
        use crate::pattern::dispatch_table::DispatchTable;
        let mut entries = HashMap::new();
        entries.insert(
            -1,
            Expr::Var {
                name: "--AX".to_string(),
                fallback: None,
            },
        );
        entries.insert(
            -2,
            Expr::Var {
                name: "--CX".to_string(),
                fallback: None,
            },
        );
        entries.insert(
            -3,
            Expr::Var {
                name: "--DX".to_string(),
                fallback: None,
            },
        );
        entries.insert(
            -4,
            Expr::Var {
                name: "--BX".to_string(),
                fallback: None,
            },
        );
        entries.insert(
            0,
            Expr::Var {
                name: "--m0".to_string(),
                fallback: None,
            },
        );
        let mut dispatch_tables = HashMap::new();
        dispatch_tables.insert(
            "--readMem".to_string(),
            DispatchTable {
                key_property: "--at".to_string(),
                entries,
                fallback: Expr::Literal(0.0),
            },
        );
        let functions = HashMap::new();

        let expr = Expr::FunctionCall {
            name: "--readMem".to_string(),
            args: vec![Expr::Literal(-1.0)], // AX register
        };
        let mut compiler = Compiler::new(&functions, &mut dispatch_tables);
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);

        state.set_var("AX", 42);
        let mut slots = vec![0i32; compiler.next_slot as usize];
        exec_ops(&ops, &compiler.compiled_dispatches, &[], &compiler.compiled_flat_arrays, &compiler.compiled_functions, &compiler.packed_cell_tables, &compiler.packed_exception_tables, &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 42);
    }

    #[test]
    fn compile_bitwise_ops() {
        // Build a function with body mod(a, pow(2, b)) — bitmask pattern.
        let mut functions = HashMap::new();
        functions.insert(
            "--lowerBytes".to_string(),
            FunctionDef {
                name: "--lowerBytes".to_string(),
                parameters: vec![
                    FunctionParam {
                        name: "--a".to_string(),
                        syntax: PropertySyntax::Integer,
                    },
                    FunctionParam {
                        name: "--b".to_string(),
                        syntax: PropertySyntax::Integer,
                    },
                ],
                locals: vec![],
                result: Expr::Calc(CalcOp::Mod(
                    Box::new(Expr::Var {
                        name: "--a".to_string(),
                        fallback: None,
                    }),
                    Box::new(Expr::Calc(CalcOp::Pow(
                        Box::new(Expr::Literal(2.0)),
                        Box::new(Expr::Var {
                            name: "--b".to_string(),
                            fallback: None,
                        }),
                    ))),
                )),
            },
        );

        // --lowerBytes(0xFF, 4) → 0xF
        let expr = Expr::FunctionCall {
            name: "--lowerBytes".to_string(),
            args: vec![Expr::Literal(0xFF as f64), Expr::Literal(4.0)],
        };
        let mut empty_dt = HashMap::new();
        let mut compiler = Compiler::new(&functions, &mut empty_dt);
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);

        let mut state = State::default();
        let mut slots = vec![0i32; compiler.next_slot as usize];
        exec_ops(&ops, &[], &[], &[], &[], &[], &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 15);
    }

    #[test]
    fn compile_full_program() {
        let mut state = setup();
        let assignments = vec![
            Assignment {
                property: "--AX".to_string(),
                value: Expr::Literal(42.0),
            },
            Assignment {
                property: "--m0".to_string(),
                value: Expr::Literal(255.0),
            },
        ];

        let program = compile(&assignments, &[], &[], &HashMap::new(), &mut HashMap::new());

        let mut slots = Vec::new();
        execute(&program, &mut state, &mut slots);

        assert_eq!(state.get_var("AX").unwrap(), 42);
        assert_eq!(state.memory[0], 255);
    }

    #[test]
    fn compile_function_inline() {
        let mut functions = HashMap::new();
        functions.insert(
            "--double".to_string(),
            FunctionDef {
                name: "--double".to_string(),
                parameters: vec![FunctionParam {
                    name: "--x".to_string(),
                    syntax: PropertySyntax::Integer,
                }],
                locals: vec![],
                result: Expr::Calc(CalcOp::Mul(
                    Box::new(Expr::Var {
                        name: "--x".to_string(),
                        fallback: None,
                    }),
                    Box::new(Expr::Literal(2.0)),
                )),
            },
        );

        let expr = Expr::FunctionCall {
            name: "--double".to_string(),
            args: vec![Expr::Literal(21.0)],
        };
        let mut empty_dt = HashMap::new();
        let mut compiler = Compiler::new(&functions, &mut empty_dt);
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);

        let mut state = State::default();
        let mut slots = vec![0i32; compiler.next_slot as usize];
        exec_ops(&ops, &[], &[], &[], &compiler.compiled_functions, &[], &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 42);
    }

    #[test]
    fn compile_value_forwarding() {
        let mut state = setup();
        // Assignment A computes --AX = 10
        // Assignment B computes --CX = var(--AX) + 5
        // B should see A's value without a state lookup
        let assignments = vec![
            Assignment {
                property: "--AX".to_string(),
                value: Expr::Literal(10.0),
            },
            Assignment {
                property: "--CX".to_string(),
                value: Expr::Calc(CalcOp::Add(
                    Box::new(Expr::Var {
                        name: "--AX".to_string(),
                        fallback: None,
                    }),
                    Box::new(Expr::Literal(5.0)),
                )),
            },
        ];

        let program = compile(&assignments, &[], &[], &HashMap::new(), &mut HashMap::new());

        let mut slots = Vec::new();
        execute(&program, &mut state, &mut slots);

        assert_eq!(state.get_var("AX").unwrap(), 10);
        assert_eq!(state.get_var("CX").unwrap(), 15);
    }

    #[test]
    fn compile_dispatch_table() {
        use crate::pattern::dispatch_table::DispatchTable;

        let mut functions = HashMap::new();
        functions.insert(
            "--lookup".to_string(),
            FunctionDef {
                name: "--lookup".to_string(),
                parameters: vec![FunctionParam {
                    name: "--key".to_string(),
                    syntax: PropertySyntax::Integer,
                }],
                locals: vec![],
                result: Expr::Literal(0.0),
            },
        );

        let mut entries = HashMap::new();
        entries.insert(0, Expr::Literal(100.0));
        entries.insert(1, Expr::Literal(200.0));
        entries.insert(2, Expr::Literal(300.0));
        entries.insert(42, Expr::Literal(999.0));

        let mut dispatch_tables = HashMap::new();
        dispatch_tables.insert(
            "--lookup".to_string(),
            DispatchTable {
                key_property: "--key".to_string(),
                entries,
                fallback: Expr::Literal(0.0),
            },
        );

        let assignments = vec![Assignment {
            property: "--result".to_string(),
            value: Expr::FunctionCall {
                name: "--lookup".to_string(),
                args: vec![Expr::Literal(42.0)],
            },
        }];

        let program = compile(&assignments, &[], &[], &functions, &mut dispatch_tables);

        let mut state = State::default();
        let mut slots = Vec::new();
        execute(&program, &mut state, &mut slots);

        // --result is not a state-mapped property, so check the slot directly
        // Find the result slot from writeback — it won't be there since --result
        // isn't a register. Check by looking at dispatch result.
        //
        // Because every entry here is an integer literal and the key range is
        // dense, compile_dispatch_call takes the flat-array fast path and
        // emits a single `DispatchFlatArray` op (no `CompiledDispatchTable`
        // entry is created). Assert the fast path fired instead.
        assert_eq!(program.dispatch_tables.len(), 0);
        assert_eq!(program.flat_dispatch_arrays.len(), 1);
        let arr = &program.flat_dispatch_arrays[0];
        // Span is 0..=42 → 43 slots; spot-check a few values.
        assert_eq!(arr.values.len(), 43);
        assert_eq!(arr.values[0], 100);
        assert_eq!(arr.values[1], 200);
        assert_eq!(arr.values[2], 300);
        assert_eq!(arr.values[42], 999);
        // Gap entries default to 0.
        assert_eq!(arr.values[5], 0);
    }

    #[test]
    fn dispatch_flat_array_literal_fast_path() {
        // Single-parameter dispatch where every entry is an integer literal
        // and keys are dense → should compile to DispatchFlatArray and return
        // the correct value both in range and out of range.
        use crate::pattern::dispatch_table::DispatchTable;

        let mut functions = HashMap::new();
        functions.insert(
            "--readByte".to_string(),
            FunctionDef {
                name: "--readByte".to_string(),
                parameters: vec![FunctionParam {
                    name: "--idx".to_string(),
                    syntax: PropertySyntax::Integer,
                }],
                locals: vec![],
                result: Expr::Literal(0.0),
            },
        );

        let mut entries = HashMap::new();
        for i in 0..16i64 {
            entries.insert(i, Expr::Literal((i * 7 + 3) as f64));
        }
        let mut dispatch_tables = HashMap::new();
        dispatch_tables.insert(
            "--readByte".to_string(),
            DispatchTable {
                key_property: "--idx".to_string(),
                entries,
                fallback: Expr::Literal(-1.0),
            },
        );

        // Build a compiler and compile a call manually so we can inspect ops.
        let expr = Expr::FunctionCall {
            name: "--readByte".to_string(),
            args: vec![Expr::Literal(5.0)],
        };
        let mut compiler = Compiler::new(&functions, &mut dispatch_tables);
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);
        // Op::Call wraps the body, which contains the DispatchFlatArray op.
        let has_flat = compiler.compiled_functions.iter()
            .any(|f| f.body_ops.iter().any(|o| matches!(o, Op::DispatchFlatArray { .. })));
        assert!(
            has_flat || ops.iter().any(|o| matches!(o, Op::DispatchFlatArray { .. })),
            "expected DispatchFlatArray op somewhere (top-level ops {:?}, functions {})",
            ops, compiler.compiled_functions.len()
        );

        // Execute and check the value.
        let mut state = State::default();
        let mut slots = vec![0i32; compiler.next_slot as usize];
        exec_ops(
            &ops,
            &compiler.compiled_dispatches,
            &[],
            &compiler.compiled_flat_arrays,
            &compiler.compiled_functions,
            &compiler.packed_cell_tables,
            &compiler.packed_exception_tables,
            &mut state,
            &mut slots,
        );
        assert_eq!(slots[slot as usize], 5 * 7 + 3);

        // Out-of-range key falls back to the default.
        let expr2 = Expr::FunctionCall {
            name: "--readByte".to_string(),
            args: vec![Expr::Literal(999.0)],
        };
        let mut ops2 = Vec::new();
        let slot2 = compiler.compile_expr(&expr2, &mut ops2);
        let mut slots2 = vec![0i32; compiler.next_slot as usize];
        exec_ops(
            &ops2,
            &compiler.compiled_dispatches,
            &[],
            &compiler.compiled_flat_arrays,
            &compiler.compiled_functions,
            &compiler.packed_cell_tables,
            &compiler.packed_exception_tables,
            &mut state,
            &mut slots2,
        );
        assert_eq!(slots2[slot2 as usize], -1);
    }

    #[test]
    fn replicated_body_matches_unrolled_execution() {
        // Build an unrolled body of K=8 reps × period=3:
        //   LoadLit  { dst: 100 + k*3,     val: 7 }       ; constant val
        //   AddLit   { dst: 100 + k*3,     a: 100 + k*3, val: k }   ; mix const+linear
        //   LoadSlot { dst: 200,           src: 100 + k*3 }         ; reads strided slot
        // After execution, slot 200 == 7 + (K-1) ; slots 100..(100+K*3) hold 7+k.

        let _ = setup();
        let period = 3usize;
        let reps = 8u32;

        let mut original_ops = Vec::new();
        for k in 0..reps {
            let s = 100 + k * 3;
            original_ops.push(Op::LoadLit { dst: s, val: 7 });
            original_ops.push(Op::AddLit { dst: s, a: s, val: k as i32 });
            original_ops.push(Op::LoadSlot { dst: 200, src: s });
        }

        // Run the unrolled version.
        let mut state_a = State::default();
        let mut slots_a = vec![0i32; 256];
        exec_ops_for_test(&original_ops, &mut state_a, &mut slots_a);

        // Build the ReplicatedBody form. Body = first rep verbatim.
        let body: Vec<Op> = original_ops[..period].to_vec();
        // Strides per body op:
        //   op0 LoadLit  { dst, val }       : dst stride 3, val const
        //     => [(0, 3)]
        //   op1 AddLit   { dst, a, val }    : dst stride 3, a stride 3, val stride 1
        //     => [(0, 3), (1, 3), (2, 1)]
        //   op2 LoadSlot { dst, src }       : dst const, src stride 3
        //     => [(1, 3)]
        let strides_per_op: Vec<Box<[(u8, i32)]>> = vec![
            Box::from([(0u8, 3i32)] as [(u8, i32); 1]),
            Box::from([(0u8, 3i32), (1u8, 3i32), (2u8, 1i32)] as [(u8, i32); 3]),
            Box::from([(1u8, 3i32)] as [(u8, i32); 1]),
        ];
        let folded_op = Op::ReplicatedBody {
            body: body.into_boxed_slice(),
            strides: strides_per_op.into_boxed_slice(),
            reps,
        };

        let mut state_b = State::default();
        let mut slots_b = vec![0i32; 256];
        exec_ops_for_test(&[folded_op], &mut state_b, &mut slots_b);

        // Slots used by the body (100..100+reps*3, plus 200) must match.
        for k in 0..reps {
            let s = (100 + k * 3) as usize;
            assert_eq!(
                slots_a[s], slots_b[s],
                "slot {} (rep {}) differs: a={} b={}",
                s, k, slots_a[s], slots_b[s]
            );
        }
        assert_eq!(slots_a[200], slots_b[200], "slot 200 (last LoadSlot) differs");
        // And specifically, slot 200 should equal 7 + (reps-1) = 14.
        assert_eq!(slots_a[200], 7 + (reps as i32 - 1));
    }

    #[test]
    fn replicated_body_with_state_writes() {
        // Body that touches state memory: StoreState { addr, src } with addr
        // striding by 4. After execution, state[0x100], state[0x104], ... hold
        // the values from slot 50 (constant per rep but easier to seed).

        let _ = setup();
        let mut state_a = State::default();
        let mut slots_a = vec![0i32; 256];
        slots_a[50] = 0xCAFE;

        let reps = 8u32;
        let mut original_ops = Vec::new();
        for k in 0..reps {
            let addr = 0x100 + (k as i32) * 4;
            original_ops.push(Op::StoreState { addr, src: 50 });
        }
        exec_ops_for_test(&original_ops, &mut state_a, &mut slots_a);

        // Folded: body = first StoreState; addr (operand 0) strides by 4.
        let body = vec![Op::StoreState { addr: 0x100, src: 50 }];
        let strides_per_op: Vec<Box<[(u8, i32)]>> =
            vec![Box::from([(0u8, 4i32)] as [(u8, i32); 1])];
        let folded = Op::ReplicatedBody {
            body: body.into_boxed_slice(),
            strides: strides_per_op.into_boxed_slice(),
            reps,
        };
        let mut state_b = State::default();
        let mut slots_b = vec![0i32; 256];
        slots_b[50] = 0xCAFE;
        exec_ops_for_test(&[folded], &mut state_b, &mut slots_b);

        // Both runs must produce identical state at every written address.
        // (We don't assert what the value *is* — `state.read_mem` may truncate,
        // sign-extend, or mask depending on address class. The conformance
        // claim is that ReplicatedBody and unrolled execution agree.)
        for k in 0..reps {
            let addr = 0x100 + (k as i32) * 4;
            assert_eq!(
                state_a.read_mem(addr),
                state_b.read_mem(addr),
                "state[{:#x}] differs",
                addr
            );
        }
    }
}
