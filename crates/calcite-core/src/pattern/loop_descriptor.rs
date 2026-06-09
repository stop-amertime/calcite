//! Self-loop recogniser: extract structural descriptors of opcode-keyed
//! self-repeating instructions from the parsed dispatch table family.
//!
//! ## What this is
//!
//! Some opcodes that a cabinet emits are *self-loops*: when the dispatched
//! body executes, it leaves the program counter unchanged (or moved back to
//! itself) so the same opcode runs again on the next tick, gated by a
//! counter. Each iteration touches memory and steps a pointer. The classic
//! examples come from x86 `REP STOSB` / `REP MOVSB` / `REP CMPSB` etc., but
//! the structural shape is general — any cabinet that wants a CPU-level
//! self-iterating instruction will emit the same shape.
//!
//! Per the calcite cardinal rule (see `../CLAUDE.md`), this recogniser may
//! NOT look at characters in any property/slot/function name. Names are
//! opaque tokens. The recogniser determines structure from:
//!
//! - **Slot identity**: whether two `var(--name)` references point at the
//!   same slot (string equality on the whole name).
//! - **Expression shape**: the tree shape of `Expr` nodes (literals,
//!   arithmetic, conditionals, calls).
//! - **Repetition**: which slots appear in multiple per-opcode bodies, and
//!   in which roles.
//!
//! It does NOT do prefix sniffing, suffix sniffing, character searches, or
//! any other content-based decision on names. A 6502 cabinet, a brainfuck
//! cabinet, or an arbitrary non-emulator cabinet whose CSS happens to emit
//! the same shape MUST trigger the same recognition with no calcite-side
//! change.
//!
//! ## What it produces
//!
//! For each opcode value V where the per-V dispatch bodies, taken together,
//! match the self-loop signature, this module yields a [`LoopDescriptor`]
//! describing the structural facts: counter slot, pointer slots and their
//! step formulas, write descriptors, exit predicate, and IP-advance
//! formula.
//!
//! Phase 1 of the [rep_fast_forward genericity mission][plan] only emits
//! descriptors; the descriptor-driven runtime applier is phase 2.
//!
//! [plan]: ../../../../CSS-DOS/docs/plans/2026-05-06-rep-fast-forward-genericity.md
//!
//! ## Recognition strategy at a glance
//!
//! Given the parsed assignments on `.cpu`, we look for a family of
//! opcode-keyed dispatches: assignments whose RHS is
//! `Expr::StyleCondition` and whose branches all test the same single
//! property (the "dispatch key", typically the opcode latch slot). Among
//! the family, an *IP slot* is one whose per-V body has the
//! "stay-here-or-advance" shape: a `StyleCondition` whose two outcomes are
//! (a) the slot's own prior value minus a literal/slot offset and (b) the
//! slot's own prior value plus a literal. The predicate of that
//! `StyleCondition` is the *loop predicate* — the structural shape that
//! tells the rest of the per-V bodies whether iteration is continuing.
//!
//! From there, for each other family member we look for known shapes
//! against that predicate:
//! - "counter": `if(<predicate>: self; else: max(0, self - 1))`
//!   (self meaning the same opaque slot read on both sides).
//! - "pointer": `if(<rep-guard>: lowerBytes(self ± k - bit(flags, n) * 2k, 16); else: self)`.
//! - "memwrite": for assignments belonging to the memwrite family, the
//!   per-V body of the address half is gated to `-1` when the rep-guard
//!   says "no fire", and to a real address expression when it should
//!   write.
//!
//! The pieces are returned as a [`LoopDescriptor`].

use std::collections::{HashMap, HashSet};

use crate::types::*;

/// Structural description of a self-loop opcode discovered in a cabinet's
/// dispatch family.
///
/// This is purely a description of CSS shape — it carries no x86, 6502, or
/// any other ISA assumptions. The runtime applier (phase 2) reads these
/// descriptors and walks them; phase 1 only emits them.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopDescriptor {
    /// The dispatch key property name (e.g. the opcode latch).
    /// Stored as an opaque token; the recogniser does not inspect it.
    pub key_property: String,
    /// The dispatch key value this descriptor describes (e.g. one specific
    /// opcode literal).
    pub key_value: i64,
    /// Property name whose dispatch body has the "stay-here-or-advance"
    /// shape. The runtime applier writes the advance value directly when
    /// fast-forwarding.
    pub ip_property: String,
    /// Slot read by both branches of the IP body — the "prior" mirror of
    /// the IP slot. Carried as an opaque name; the runtime resolves it to
    /// a slot index.
    pub ip_self_property: String,
    /// Literal offset added by the IP-advance branch (the per-iter IP step
    /// taken when the loop exits).
    pub ip_advance_literal: i32,
    /// Property names that participate in the loop's continuation predicate.
    /// Captured for diagnostic / verification purposes; the predicate
    /// itself is also stored verbatim in [`Self::predicate`].
    pub predicate_properties: Vec<String>,
    /// The full predicate expression as it appears in the IP body's
    /// `StyleCondition`. The runtime applier evaluates this against the
    /// post-tick slot view to decide whether to fast-forward at all.
    pub predicate: StyleTest,
    /// Whether `predicate` evaluating true selects the *stay* branch of
    /// the IP body. Captured from which orientation of the stay/advance
    /// match succeeded (`true` = stay body was the branch `then`,
    /// `false` = the emitter inverted the shape and stay is the
    /// fallback). The runtime gate ([`evaluate_loop_predicate`]) uses
    /// this polarity: the loop is mid-flight iff the predicate's
    /// post-tick truth value equals `predicate_means_stay` — i.e. the
    /// CSS took the stay branch this tick, so the IP slot held its
    /// value and more iterations remain to collapse.
    ///
    /// **Cardinal-rule probe.** Pure shape: both orientations are
    /// structural facts about which branch holds the `calc(self − …)`
    /// body. No name or upstream meaning involved.
    pub predicate_means_stay: bool,
    /// The counter slot, if one was found.
    pub counter: Option<CounterEntry>,
    /// Pointer slots and their step formulas. Order is recogniser-dependent
    /// (see implementation); descriptors compare equal regardless of order.
    pub pointers: Vec<PointerEntry>,
    /// Write descriptors gathered from memwrite-slot assignments belonging
    /// to the same dispatch family.
    pub writes: Vec<WriteEntry>,
    /// Whether the predicate is a flag-conditioned exit (i.e. the loop
    /// exits not just on counter zero but also on a flag-bit condition,
    /// CMPS/SCAS-style). Phase 1 only sets this when the predicate's
    /// shape itself has a flag-bit conjunction; phase 2 will use it to
    /// drive the right runtime walker.
    pub flag_conditioned: bool,
    /// Bulk-applier classification computed structurally at descriptor
    /// build time. Phase 3a populates this; phase 3b's runtime applier
    /// dispatches on it. The classifier is purely shape-based — it does
    /// not look at any name, only at whether write-value expressions
    /// transitively reference any pointer-slot mirror.
    pub bulk_class: BulkClass,
    /// Per-iteration cycle cost charged when the applier fast-forwards
    /// one iteration of this loop.
    ///
    /// **Structurally derived, not opcode-keyed.** At recogniser time we
    /// scan the dispatch family for the single family member whose
    /// per-key body has the shape `Calc(Add(Var(X), Literal(K)))` (or
    /// the commutative `Literal(K) + Var(X)`) for the largest number
    /// of keys, with `X` the same opaque slot reference across all
    /// matching keys. That member is "the cycle counter": the slot
    /// dispatched per opcode as a fixed-slot-plus-per-opcode-literal
    /// shape. For this descriptor's `key_value`, the literal `K` from
    /// that member's body is captured here.
    ///
    /// `Some(K)` when the dispatch family contains such a member AND
    /// it has a body for this opcode of the matching shape.
    /// `None` otherwise — the applier returns `Unsupported` and the
    /// dispatcher panics, because charging the wrong cycle count would
    /// break cabinets whose progression is gated on cycleCount-derived
    /// timers.
    ///
    /// **Cardinal-rule probe.** A 6502 / brainfuck / non-emulator
    /// cabinet whose CSS emits the same `var(X) + Literal(K)`-per-key
    /// dispatch family produces the same `Some(K)`; one without that
    /// shape produces `None`. Slot names are opaque tokens; renaming
    /// the cycle counter slot (e.g. `--cycleCount` → `--zorch`) does
    /// not affect the extraction.
    pub per_iter_cycles: Option<i32>,
    /// Name of an extra slot whose value contributes to the IP slot's
    /// post-loop advance, captured from the stay branch's subtrahend:
    /// the per-key IP body `if(<pred>: calc(self − var(extra)); else:
    /// calc(self + L))` yields `Some("--extra")`.
    ///
    /// Why the subtrahend is the extra advance: on a "stay" tick the IP
    /// slot is a fixed point — its newly assigned value equals its
    /// current value — so `self − extra (+ any wrapper addend W) = IP`,
    /// pinning `self = IP + extra − W`. The exit branch then produces
    /// `self + L + W = IP + extra + L`. Wrapper contributions cancel:
    /// the post-loop advance over the current IP is `extra + L`
    /// (`ip_advance_literal` is `L`), derivable from the two branch
    /// shapes alone. The runtime applier reads the extra slot's current
    /// value and commits `IP + L + extra` after fast-forwarding.
    ///
    /// `Some(slot_name)` when the stay subtrahend is a bare `Var`;
    /// `None` when it is any other shape (e.g. a literal) — the applier
    /// then reports the loop unsupported rather than guessing.
    ///
    /// **Cardinal-rule probe.** A 6502 or brainfuck cabinet whose
    /// stay branch subtracts a slot of any name (`--introBytes`,
    /// `--zorch`, etc.) produces `Some` of that name — the recogniser
    /// captures whatever name the cabinet used, without inspecting the
    /// characters of that name. The structural fact is "the stay branch
    /// is `calc(self − var(...))`"; no upstream meaning is consulted.
    pub ip_extra_advance_slot: Option<String>,
    /// Structural comparison shape for flag-conditioned ReadOnly loops
    /// (CMPS / SCAS family). `Some` when the dispatch family contains a
    /// comparison member whose per-key body for this opcode is shaped
    /// `Calc(Sub(a, b))` (or commutative reassociations) with the
    /// operands tracing through pointer mirrors or accumulator slots;
    /// `None` otherwise.
    ///
    /// **Cardinal-rule probe.** A 6502 cabinet that emits a comparison
    /// dispatch with the same `Calc(Sub(...))`-per-opcode shape produces
    /// an equivalent ComparisonShape using the cabinet's own slot names.
    /// Renaming the segment / pointer / accumulator slots does not affect
    /// the match. A cabinet without a comparison dispatch produces
    /// `None`, and the applier returns `Unsupported` — the dispatcher
    /// panics, which is correct because the flag-conditioned ReadOnly
    /// shape MUST have a comparison source somewhere in the cabinet.
    pub comparison_shape: Option<ComparisonShape>,
    /// The cabinet's own outer-guard predicate for whether the rep
    /// dispatch's normal branch fires this tick. Captured structurally
    /// from the `StyleCondition` wrappers that the recogniser strips on
    /// its way down to the dispatch (see `find_inner_dispatch`'s
    /// fallback-descent rule): every wrapper whose `fallback` contains
    /// the inner dispatch contributes its branch conditions to a set
    /// of "override conditions". The dispatch fires iff none of those
    /// conditions match the current state — i.e., the wrapper's
    /// fallback (else) branch was reached.
    ///
    /// `Some(Precondition::NoOverrides(tests))` when at least one
    /// outer wrapper was stripped on the way to the inner dispatch.
    /// `None` when the IP slot's assignment was already a bare
    /// single-key dispatch with no wrappers (a brainfuck / 6502
    /// cabinet without TF/IRQ override plumbing). In the `None`
    /// case the applier treats the precondition as trivially true
    /// and proceeds — fast-forward is always safe per shape alone.
    ///
    /// **Cardinal-rule probe.** The `StyleTest`s captured here are the
    /// cabinet's own wrapper-branch conditions, opaque slot names and
    /// all. A cabinet whose wrappers use entirely different slot names
    /// (or none at all) produces a structurally-equivalent
    /// `Precondition` (or `None`) without any character-level
    /// inspection of slot names by the recogniser.
    pub precondition: Option<Precondition>,
}

/// The cabinet's own outer-guard predicate for whether a recognised
/// self-loop's dispatch body fires this tick.
///
/// Stored separately from `predicate` (which is the *inner* loop-
/// continuation predicate read off the IP body's `StyleCondition`):
/// `Precondition` describes the *outer* `StyleCondition` wrappers
/// that the recogniser stripped to find the dispatch in the first
/// place. The dispatch fires iff this precondition holds, evaluated
/// against the post-tick slot view.
///
/// Currently only one shape is captured (`NoOverrides`); adding more
/// shapes is a matter of extending this enum and the matching
/// evaluator arm in [`evaluate_precondition`].
#[derive(Debug, Clone, PartialEq)]
pub enum Precondition {
    /// "None of the listed wrapper-branch conditions hold." Captures
    /// the structural fact that the recogniser descended into a
    /// `StyleCondition`'s `fallback` to find the dispatch: that
    /// fallback fires iff every branch condition above it evaluated
    /// false.
    ///
    /// An empty `Vec` is degenerate — it means the precondition is
    /// trivially true. The recogniser doesn't emit an empty
    /// `NoOverrides`; the enclosing `Option<Precondition>` is `None`
    /// in that case.
    NoOverrides(Vec<StyleTest>),
}

/// How a flag-conditioned ReadOnly loop (CMPS / SCAS family) performs
/// its per-iter comparison, captured structurally from the cabinet's
/// own per-opcode comparison dispatch.
///
/// The shape is identified by finding a dispatch family member whose
/// per-key body for the opcode of this descriptor is a `Calc(Sub(a, b))`
/// (or `Calc(Add(a, Negate(b)))`) where one operand traces — through any
/// intermediate slot bodies — to a memory read keyed on a pointer
/// mirror. The other operand is either another such memory read (CMPS)
/// or a bare `Var` (SCAS, the accumulator). No opcode-byte tables, no
/// "this slot is named --AL" reads — the matcher only inspects `Expr`
/// shape and whole-name slot identity.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonShape {
    /// Width of one comparison step, in bytes. Structurally equal to
    /// the loop's pointer step magnitude (`pointers[0].base_step`).
    /// 8086 byte/word string ops give 1 or 2; a hypothetical wider
    /// loop on another ISA would give a larger value.
    pub width: u8,
    /// Destination-side segment slot. Captured from the address
    /// decomposition `var(seg) * 16 + var(ptr)` or `var(base) + var(ptr)`
    /// of one operand of the comparison `Sub`. The slot name is opaque —
    /// no character of any name is inspected.
    pub dst_seg_property: String,
    /// Whether the destination base sat inside a `* 16` in the captured
    /// shape (see [`ComparisonSource::Pointer::seg_times_sixteen`]).
    pub dst_seg_times_sixteen: bool,
    /// Destination-side pointer slot. Equals the destination pointer
    /// entry's `self_property`.
    pub dst_ptr_property: String,
    /// Source operand of the comparison.
    pub source: ComparisonSource,
    /// Name of the slot whose values, paired with the comparison-derived
    /// flag-bit slot's values in the IP-body predicate's disjunctive
    /// branches, select the comparison-result sense (REPE vs REPNE on
    /// x86; conceptually "continue while equal" vs "continue while
    /// unequal" on any ISA with this loop family).
    ///
    /// Identified structurally as the predicate slot that:
    ///   1. Takes more than one distinct literal value across the
    ///      disjunctive branches of the IP predicate.
    ///   2. Is NOT the comparison-derived flag-bit slot (the slot whose
    ///      top-level body transitively depends on the comparison
    ///      dispatch's output).
    ///
    /// `Some(slot)` when exactly one predicate slot satisfies both
    /// rules. `None` when no such slot exists (a cabinet with a single
    /// unconditional REPE-like exit polarity, or one whose discriminator
    /// is encoded differently).
    pub rep_type_property: Option<String>,
}

/// Source operand of a flag-conditioned ReadOnly comparison.
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonSource {
    /// CMPS-shape: source is a memory read through a second stepping
    /// pointer. Carries the source-side segment slot and pointer slot
    /// (matching the descriptor's second pointer entry).
    Pointer {
        seg_property: String,
        /// Whether the captured base sat inside a `* 16` in the shape
        /// (`var(seg)*16 + ptr` → true; flat `var(base) + ptr` → false,
        /// the slot already holds the full addend — e.g. the cabinet's
        /// segment-override-aware pre-scaled base). The applier scales
        /// exactly as the captured shape does.
        seg_times_sixteen: bool,
        ptr_property: String,
    },
    /// SCAS-shape: source is an accumulator slot read once per iter.
    /// The width-1 (byte) variant uses `byte_property`; the width-2
    /// (word) variant uses `word_property`. Captured per-descriptor —
    /// each opcode (byte / word) gets its own descriptor with the slot
    /// for *that* width populated and the other left empty.
    Accumulator {
        byte_property: String,
        word_property: String,
    },
}

/// Coarse classification of how a recognised self-loop's per-iter
/// memory writes can be collapsed into a bulk operation.
///
/// The classifier is structural and does not encode opcode knowledge.
/// It looks at:
///
/// - Whether the descriptor has any write entries (`writes.len()`).
/// - Whether each write's value expression transitively references the
///   `self_property` of any pointer entry (i.e. the prior-tick mirror
///   the cabinet uses to read the source pointer's pre-iter value).
///
/// "Transitively references" means: there is some `Expr::Var` or
/// `StyleTest::Single { property, .. }` somewhere in the value
/// expression tree whose property name equals one of the pointer's
/// `self_property` strings (whole-name equality — no substring or
/// character inspection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BulkClass {
    /// No memory writes (CMPS / SCAS / LODS). Bulk applier walks
    /// counter-many iterations doing reads and predicate checks; no
    /// memory mutation.
    ReadOnly,
    /// All write-value expressions are independent of pointer-slot
    /// state (typical STOS: every iteration writes the same constant
    /// from an accumulator). Collapses to a flat memset over the
    /// iterated address range.
    Fill,
    /// At least one write-value expression reads through a pointer
    /// slot's mirror (typical MOVS: writes byte fetched from the
    /// per-iter source pointer). Collapses to a memcpy along the
    /// stepped address range, modulo overlap rules.
    Copy,
    /// Write-value expression depends on something other than a
    /// pointer mirror, or has shape we don't recognise structurally.
    /// Bulk applier falls back to per-iter evaluation.
    PerIter,
}

/// A counter slot — one whose per-V body decrements itself when the
/// loop predicate says "fire" and saturates at zero.
#[derive(Debug, Clone, PartialEq)]
pub struct CounterEntry {
    /// The property the dispatch body is for.
    pub property: String,
    /// The "self" slot read on both sides of the body. In well-formed
    /// shapes this equals `property` (or its prior-tick mirror); the
    /// recogniser only requires that both branches read the same slot,
    /// not that it bears a particular name.
    pub self_property: String,
    /// The decrement amount. Almost always 1 in practice but recorded
    /// generically.
    pub step: i32,
}

/// A pointer slot — one whose per-V body advances by ±k under a
/// direction-flag bit, gated by the rep-guard.
#[derive(Debug, Clone, PartialEq)]
pub struct PointerEntry {
    /// The dispatch body's target property.
    pub property: String,
    /// The "self" slot read by the body's update branch.
    pub self_property: String,
    /// The base step magnitude (positive). The actual signed step is
    /// `base_step` when the direction-flag bit is 0, `-base_step` when 1.
    pub base_step: i32,
    /// The flag slot the direction bit is read from.
    pub flag_property: String,
    /// Bit position of the direction flag inside `flag_property`.
    pub flag_bit: u32,
}

/// A memwrite descriptor — captured as raw expression slices since phase
/// 1 doesn't yet drive a runtime; phase 2 compiles these into the applier.
#[derive(Debug, Clone, PartialEq)]
pub struct WriteEntry {
    /// The address-side property the kiln-style memwrite slot was on
    /// (e.g. one of the `--memAddrN`-shaped slots, but the recogniser
    /// treats the name as opaque).
    pub addr_property: String,
    /// The value-side property paired with the address.
    pub val_property: String,
    /// Address expression for this opcode. Already unwrapped from the
    /// per-V `StyleCondition` branch.
    pub addr_expr: Expr,
    /// Value expression for this opcode.
    pub val_expr: Expr,
    /// Structural decomposition of `addr_expr` into a `(segment_slot,
    /// pointer_slot)` pair, when the address has the canonical
    /// "segment-shifted-by-16 plus pointer" shape that bulk appliers
    /// can iterate over.
    ///
    /// The recogniser searches `addr_expr` for any sub-expression of
    /// shape `calc(seg_var * 16 + ptr_var)` or `calc(ptr_var + seg_var
    /// * 16)`, where `seg_var` and `ptr_var` are both `Expr::Var`
    /// references. The literal `16` is the only number-content the
    /// matcher is allowed to read — it is the canonical 8086-style
    /// segment shift, structurally identical to "scale a base by a
    /// fixed page size and add an offset" in any other ISA. (The
    /// genericity probe: a 6502 / brainfuck / non-emulator cabinet
    /// that emits the same shape — base * K + index, K constant — must
    /// decompose identically.)
    ///
    /// `Some((seg_property, pointer_property))` when the shape matches;
    /// `None` otherwise (e.g. for shapes the structural matcher can't
    /// simplify — bulk appliers fall back to per-iter `addr_expr`
    /// evaluation in that case).
    pub addr_decomposition: Option<(String, String)>,
    /// Indirect-read intermediate decomposition for the value side.
    ///
    /// When `val_expr` is a bare `Var(name)` whose dispatch body in the
    /// cabinet has the canonical "read function-call keyed on pointer
    /// mirror" shape, this captures the structural fact. The matcher
    /// allows: `body = FunctionCall(_, args)` where `args` contains —
    /// anywhere in their tree — a `Var` reference to one of the
    /// descriptor's pointer `self_property` slots.
    ///
    /// Optionally extracts a segment slot when the call's argument tree
    /// has the clean shape `calc(var(seg_slot) + var(ptr_mirror))` (or
    /// the reversed orientation). Otherwise `seg_property` is `None`
    /// and the runtime must evaluate the address expression as-is.
    ///
    /// This is the structural meat of phase 3b step 2: the cabinet
    /// writes a byte that's the result of a memory read keyed on the
    /// loop's source pointer. Recognising the indirect-read intermediate
    /// at compile time lets the bulk classifier promote `Fill` → `Copy`
    /// for MOVS-style loops that route their source byte through a
    /// derived intermediate slot. Pure structural shape — no character
    /// inspection of any name.
    pub val_indirect_read: Option<IndirectRead>,
}

/// Decomposition of a value-side indirect read through a pointer mirror.
///
/// Captured structurally: the cabinet's `val_expr` is a bare `Var(name)`
/// whose dispatch body is `FunctionCall(_, args)` with the args tree
/// referencing one of the descriptor's pointer mirror slots. The matcher
/// inspects only `Expr` shapes and slot identity (whole-name equality);
/// no character of any name is read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndirectRead {
    /// Base slot of the read address, when the call's argument
    /// decomposes cleanly as `var(base) + var(ptr)` or
    /// `var(seg) * 16 + var(ptr)` (either operand order, with any
    /// trailing literal byte-offset addends peeled off — those are the
    /// within-iteration offsets the applier models positionally). The
    /// slot may itself be an intermediate (e.g. a `StyleCondition`
    /// honouring a segment override and resolving to a pre-scaled
    /// base) — runtime resolution is the runtime's problem.
    pub seg_property: Option<String>,
    /// Whether the captured base appeared inside a `* 16` in the shape.
    /// `true` (`var(seg) * 16 + ptr`): the slot holds an unscaled value
    /// and the applier multiplies by 16, exactly as the CSS does.
    /// `false` (`var(base) + ptr`): the slot already holds the full
    /// addend and the applier adds it as-is. The scaling is a fact of
    /// the captured shape — applying a different scaling than the
    /// cabinet's own expression would diverge from Chrome.
    pub seg_times_sixteen: bool,
    /// The pointer mirror slot the indirect read keys on. Always one of
    /// the descriptor's pointer `self_property` names.
    pub pointer_property: String,
    /// The intermediate slot name itself (the bare `Var` the val_expr
    /// reads). Carried so the runtime can re-resolve via slot index when
    /// it needs more than the seg/ptr pair.
    pub intermediate_property: String,
}

/// Recognise self-loop opcodes in a dispatch family.
///
/// Inputs:
///
/// - `assignments`: the cabinet's top-level assignments (e.g. the body of
///   `.cpu`). These are scanned for the dispatch family.
///
/// The function returns one [`LoopDescriptor`] per recognised opcode.
///
/// **Cardinal-rule contract.** The implementation only inspects:
/// - Slot/property *identity* (string-equality on the entire name).
/// - Expression-tree *shape* (variant, arity, literals).
/// - Slot *repetition* (how many times a name appears, and in which
///   structural positions).
///
/// It does NOT inspect characters of any name. Renaming every slot in the
/// cabinet (preserving identity) MUST produce identical descriptors modulo
/// the new names.
pub fn recognise_loops(assignments: &[Assignment]) -> Vec<LoopDescriptor> {
    // 1. Find the dispatch family — assignments whose RHS is a top-level
    //    StyleCondition keyed on a single property, all sharing the same
    //    key.
    let family = collect_dispatch_family(assignments);
    let Some(family) = family else { return Vec::new() };

    // Build a name → body lookup over all top-level assignments. The
    // val-side indirect-read recogniser uses this to peek inside
    // intermediate slots like the cabinet's `--_strSrcByte`-shaped
    // pre-computed read. Whole-name equality only — the lookup is just a
    // HashMap on the property string the cabinet emitted.
    let assignment_index = build_assignment_index(assignments);

    let mut out = Vec::new();
    for &key_value in &family.keys {
        if let Some(desc) = recognise_one_opcode(&family, key_value, &assignment_index) {
            out.push(desc);
        }
    }
    out.sort_by_key(|d| d.key_value);
    out
}

fn build_assignment_index(assignments: &[Assignment]) -> HashMap<&str, &Expr> {
    let mut idx: HashMap<&str, &Expr> = HashMap::with_capacity(assignments.len());
    for a in assignments {
        idx.insert(a.property.as_str(), &a.value);
    }
    idx
}

// ---------------------------------------------------------------------------
// Dispatch family — assignments keyed on a common single property.
// ---------------------------------------------------------------------------

/// A group of assignments whose RHS is a top-level `StyleCondition` whose
/// branches all test a single common property (the dispatch key). The set
/// of values across all members is the *key set*.
struct DispatchFamily<'a> {
    key_property: String,
    /// Set of all key values that appear in any member's branches.
    keys: Vec<i64>,
    /// Per-member: property name → list of (key_value, body Expr).
    /// A member is one assignment, indexed by its property name.
    members: HashMap<&'a str, FamilyMember<'a>>,
}

struct FamilyMember<'a> {
    /// Per-key body. A body is the `then` of the matching StyleBranch.
    bodies: HashMap<i64, &'a Expr>,
    /// The fallback Expr (applied when no key matches).
    #[allow(dead_code)]
    fallback: &'a Expr,
    /// Position of this assignment in the source's `assignments` list.
    /// Used purely structurally — the recogniser does not interpret the
    /// number, only compares it across members. Cabinets that emit
    /// related slots in matched order (kiln pairs an address slot with
    /// the value slot that immediately follows) get correct pairing for
    /// free; cabinets that don't degrade to first-by-position pairing,
    /// which is no worse than the old name-sort heuristic.
    assignment_index: usize,
}

fn collect_dispatch_family<'a>(assignments: &'a [Assignment]) -> Option<DispatchFamily<'a>> {
    // For each assignment, try to extract a single-key dispatch shape:
    //   if(style(P:K1): B1; style(P:K2): B2; ...; else: F)
    // (StyleCondition where every branch is StyleTest::Single on the
    // same P, with K_i a literal.)
    let mut by_key: HashMap<String, HashMap<&'a str, FamilyMember<'a>>> = HashMap::new();

    for (idx, asn) in assignments.iter().enumerate() {
        let Some((key_prop, mut member)) = extract_single_key_dispatch(asn) else {
            continue;
        };
        member.assignment_index = idx;
        by_key.entry(key_prop).or_default().insert(asn.property.as_str(), member);
    }

    // Pick the largest family (in member count). If two are tied, prefer
    // the one with more keys.
    let (best_key_prop, best_members) = by_key
        .into_iter()
        .max_by_key(|(_, m)| {
            let n_members = m.len();
            let n_keys: usize = m.values().map(|fm| fm.bodies.len()).sum();
            (n_members, n_keys)
        })?;

    if best_members.len() < 2 {
        // A loop needs at least an IP body and one of {counter, pointer,
        // memwrite}. Two members is the minimum.
        return None;
    }

    let mut all_keys: HashSet<i64> = HashSet::new();
    for m in best_members.values() {
        for &k in m.bodies.keys() {
            all_keys.insert(k);
        }
    }
    let mut keys: Vec<i64> = all_keys.into_iter().collect();
    keys.sort_unstable();

    Some(DispatchFamily {
        key_property: best_key_prop,
        keys,
        members: best_members,
    })
}

fn extract_single_key_dispatch<'a>(
    asn: &'a Assignment,
) -> Option<(String, FamilyMember<'a>)> {
    let dispatch_expr = find_inner_dispatch(&asn.value)?;

    // Try the strict shape first: every branch on the same property
    // with a Literal value. This matches register dispatches like --CX
    // where the wrapper is peeled off before reaching here.
    if let Some(strict) = extract_strict_single_key(dispatch_expr, asn.property.as_str()) {
        return Some(strict);
    }

    // Fall back to the dominant-key shape: a StyleCondition where most
    // branches are keyed on one property but a few wrapper branches
    // (TF/IRQ override) are keyed on others. Used by the memwrite slot
    // assignments (--memAddrN / --memValN) where kiln folds the
    // wrapper into the same chain.
    let (key_prop, bodies_vec, fallback) = extract_dominant_dispatch_key(dispatch_expr)?;
    if bodies_vec.is_empty() {
        return None;
    }
    let mut bodies: HashMap<i64, &Expr> = HashMap::new();
    for (k, body) in bodies_vec {
        bodies.insert(k, body);
    }
    Some((
        key_prop,
        FamilyMember {
            bodies,
            fallback,
            assignment_index: 0, // overwritten by caller
        },
    ))
}

fn extract_strict_single_key<'a>(
    dispatch_expr: &'a Expr,
    _member_prop: &'a str,
) -> Option<(String, FamilyMember<'a>)> {
    let Expr::StyleCondition { branches, fallback } = dispatch_expr else { return None };
    if branches.is_empty() {
        return None;
    }
    let key_prop = match &branches[0].condition {
        StyleTest::Single { property, .. } => property.clone(),
        _ => return None,
    };
    let mut bodies: HashMap<i64, &Expr> = HashMap::new();
    for branch in branches {
        let StyleTest::Single { property, value } = &branch.condition else {
            return None;
        };
        if property != &key_prop {
            return None;
        }
        let Expr::Literal(v) = value else {
            return None;
        };
        bodies.insert(*v as i64, &branch.then);
    }
    Some((
        key_prop,
        FamilyMember {
            bodies,
            fallback: fallback.as_ref(),
            assignment_index: 0, // overwritten by caller
        },
    ))
}

/// Find the inner "single-key dispatch" StyleCondition inside an
/// expression that may have outer wrappers. Recognises the structural
/// shape: a StyleCondition whose every branch tests the SAME single
/// property against an integer literal. The wrappers we descend through
/// are also purely structural:
///
/// - `Calc(<inner>, <anything>)` arithmetic — kiln wraps the IP
///   dispatch in `calc(<dispatch> + var(--prefixLen))`. We descend
///   into either side of any binary calc op, and into the singleton arg
///   of unary ones, looking for an inner dispatch.
/// - `StyleCondition { branches, fallback }` whose `branches` keys are
///   on a different property than what the inner dispatch keys on. The
///   TF / IRQ wrapper kiln emits for every register has shape
///   `if(style(--_tf: 1): X; style(--_irqActive: 1): Y; else: <real>)`
///   where the real dispatch (keyed on `--opcode`) lives in the else.
///   The wrapper's two override branches return non-dispatch values
///   (constants or var-reads). We descend into the fallback whenever it
///   itself contains a single-key dispatch on a property different from
///   the outer's branch keys.
///
/// Returns `Some(<the StyleCondition that IS the single-key dispatch>)`,
/// or `None` if no such inner dispatch exists.
fn find_inner_dispatch(expr: &Expr) -> Option<&Expr> {
    // Direct hit?
    if is_single_key_dispatch(expr) {
        return Some(expr);
    }
    // Descend into Calc binary/unary ops.
    if let Expr::Calc(op) = expr {
        if let Some(inner) = descend_calc_for_dispatch(op) {
            return Some(inner);
        }
    }
    if let Expr::StyleCondition { fallback, branches } = expr {
        if !branches.is_empty() {
            // Try the fallback (TF/IRQ wrapper case: real dispatch
            // lives in the else).
            if let Some(inner) = find_inner_dispatch(fallback) {
                return Some(inner);
            }
            // Mixed-key StyleCondition (memwrite-slot case: the
            // wrapper branches are folded into the same chain as the
            // dispatch branches, all sharing a single
            // `if(... ;... ;... ;else: ...)`). Accept this expr as the
            // dispatch point if some single property dominates the
            // branches' keys. The downstream
            // `extract_dominant_dispatch_key` then picks out only the
            // dispatch-keyed branches.
            if has_dominant_dispatch_key(expr) {
                return Some(expr);
            }
        }
    }
    None
}

fn has_dominant_dispatch_key(expr: &Expr) -> bool {
    let Expr::StyleCondition { branches, .. } = expr else { return false };
    if branches.is_empty() {
        return false;
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for b in branches {
        if let StyleTest::Single { property, value } = &b.condition {
            if matches!(value, Expr::Literal(_)) {
                *counts.entry(property.as_str()).or_default() += 1;
            }
        }
    }
    let total: usize = counts.values().sum();
    if total < 2 {
        return false;
    }
    let max_count = counts.values().copied().max().unwrap_or(0);
    max_count * 2 >= total
}

fn descend_calc_for_dispatch(op: &CalcOp) -> Option<&Expr> {
    match op {
        CalcOp::Add(a, b)
        | CalcOp::Sub(a, b)
        | CalcOp::Mul(a, b)
        | CalcOp::Div(a, b)
        | CalcOp::Mod(a, b)
        | CalcOp::Pow(a, b) => find_inner_dispatch(a).or_else(|| find_inner_dispatch(b)),
        CalcOp::Min(args) | CalcOp::Max(args) => args.iter().find_map(find_inner_dispatch),
        CalcOp::Clamp(a, b, c) => find_inner_dispatch(a)
            .or_else(|| find_inner_dispatch(b))
            .or_else(|| find_inner_dispatch(c)),
        CalcOp::Round(_, a, b) => find_inner_dispatch(a).or_else(|| find_inner_dispatch(b)),
        CalcOp::Sign(a) | CalcOp::Abs(a) | CalcOp::Negate(a) => find_inner_dispatch(a),
    }
}

fn is_single_key_dispatch(expr: &Expr) -> bool {
    let Expr::StyleCondition { branches, .. } = expr else { return false };
    if branches.is_empty() {
        return false;
    }
    let key = match &branches[0].condition {
        StyleTest::Single { property, .. } => property,
        _ => return false,
    };
    branches.iter().all(|b| match &b.condition {
        StyleTest::Single { property, value } => {
            property == key && matches!(value, Expr::Literal(_))
        }
        _ => false,
    })
}

/// Like [`is_single_key_dispatch`], but tolerant of "wrapper" branches
/// keyed on a different property than the dispatch's main key. Used
/// when a memwrite-slot StyleCondition has TF/IRQ override branches
/// keyed on `_tf` / `_irqActive` interleaved with the opcode-keyed
/// dispatch branches, all in one `if(...)` chain.
///
/// Returns the dispatch property name and the matching branches if
/// at least one Single::Literal branch on a single common property
/// dominates the branch set (more than half of all
/// `StyleTest::Single { value: Literal }` branches share its key).
fn extract_dominant_dispatch_key<'a>(
    expr: &'a Expr,
) -> Option<(String, Vec<(i64, &'a Expr)>, &'a Expr)> {
    let Expr::StyleCondition { branches, fallback } = expr else { return None };
    if branches.is_empty() {
        return None;
    }
    // Count occurrences of each Single-property used with a Literal
    // value.
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for b in branches {
        if let StyleTest::Single { property, value } = &b.condition {
            if matches!(value, Expr::Literal(_)) {
                *counts.entry(property.as_str()).or_default() += 1;
            }
        }
    }
    let total: usize = counts.values().sum();
    if total == 0 {
        return None;
    }
    // Pick the property with the most occurrences. Must have a clear
    // majority (>= 50% of literal-keyed branches) — otherwise we don't
    // have a coherent dispatch.
    let (dom_key, dom_count) = counts.iter().max_by_key(|(_, c)| **c)?;
    if *dom_count * 2 < total {
        return None;
    }
    let dom_key = dom_key.to_string();
    let mut bodies: Vec<(i64, &Expr)> = Vec::new();
    for b in branches {
        if let StyleTest::Single { property, value } = &b.condition {
            if property == &dom_key {
                if let Expr::Literal(v) = value {
                    bodies.push((*v as i64, &b.then));
                }
            }
        }
    }
    Some((dom_key, bodies, fallback.as_ref()))
}

// ---------------------------------------------------------------------------
// Per-opcode recognition.
// ---------------------------------------------------------------------------

fn recognise_one_opcode<'a>(
    family: &DispatchFamily<'a>,
    key_value: i64,
    assignment_index: &HashMap<&'a str, &'a Expr>,
) -> Option<LoopDescriptor> {
    // Step 1: find a member with the IP-stay-or-advance shape for this
    // key. This is the killer signature; without it there is no loop.
    let mut ip_member: Option<(&str, IpShape)> = None;
    for (&prop, member) in &family.members {
        let Some(body) = member.bodies.get(&key_value) else { continue };
        if let Some(shape) = match_ip_stay_or_advance(body) {
            ip_member = Some((prop, shape));
            break;
        }
    }
    let (ip_prop, ip_shape) = ip_member?;

    // Step 2: walk the rest of the family for this key, classifying each
    // body against the predicate from the IP shape.
    let predicate = ip_shape.predicate.clone();
    let predicate_properties = collect_property_names_from_predicate(&predicate);
    let flag_conditioned = predicate_mentions_flag_bit(&predicate);

    let mut counter: Option<CounterEntry> = None;
    let mut pointers: Vec<PointerEntry> = Vec::new();
    let mut writes_addr: HashMap<&str, (&str, &Expr)> = HashMap::new(); // prop → (val_paired_property_name_unknown_yet, addr_expr)
    let mut writes_val: HashMap<&str, &Expr> = HashMap::new();

    for (&prop, member) in &family.members {
        if prop == ip_prop {
            continue;
        }
        let Some(body) = member.bodies.get(&key_value) else { continue };

        if let Some(c) = match_counter(body, &predicate, prop) {
            // Keep the FIRST counter — there should be at most one in a
            // well-formed loop. Multiple counters mean the predicate fits
            // multiple decrement bodies; we conservatively keep the first
            // (sorted) to make output deterministic.
            if counter.is_none() {
                counter = Some(c);
            }
            continue;
        }

        if let Some(p) = match_pointer(body, prop) {
            pointers.push(p);
            continue;
        }

        // Otherwise: try to classify as memwrite address or value.
        // We do this *purely* by shape: an address expression is a
        // StyleCondition (or unconditional value) producing either
        // `-1` (gated off) or an arithmetic address; a value expression
        // produces a slot read or a literal. We tentatively bin the body
        // by whether it has a `-1` literal in either branch position.
        match classify_memwrite_side(body) {
            MemwriteSide::AddressLike => {
                writes_addr.insert(prop, (prop, body));
            }
            MemwriteSide::ValueLike => {
                writes_val.insert(prop, body);
            }
        }
    }

    // Pair address/value memwrite slots by assignment-index proximity.
    //
    // The cabinet's emitter (in CSS-DOS, kiln) pairs related slots by
    // emitting them adjacent in the cabinet source. We exploit that
    // structurally: for each address slot, the matching value slot is
    // the one whose assignment_index is closest to it among unpaired
    // value slots, with ties broken in favour of the immediately-after
    // position. This is purely positional — no character-level reads.
    //
    // Cabinets that don't co-locate addr/val slots degrade to "pick the
    // closest available", which is no worse than the old name-sort
    // heuristic and still deterministic.
    let mut writes: Vec<WriteEntry> = Vec::new();
    let mut addr_props: Vec<&str> = writes_addr.keys().copied().collect();
    // Sort addresses by assignment_index so pairing is left-to-right
    // through the cabinet source (deterministic regardless of HashMap
    // iteration order).
    addr_props.sort_by_key(|p| family.members[*p].assignment_index);
    // Pointer mirrors — the prior-tick `self_property` slots the
    // pointer step bodies read. The val-side indirect-read recogniser
    // uses these to decide whether an intermediate slot's body is
    // structurally a "read keyed on a pointer" (Copy-shape) versus
    // anything else (Fill / PerIter).
    let pointer_mirrors: HashSet<&str> = pointers
        .iter()
        .map(|p| p.self_property.as_str())
        .collect();
    for ap in addr_props {
        let (_, addr_expr) = writes_addr[ap];
        let addr_idx = family.members[ap].assignment_index;
        // Find the unpaired value property whose assignment_index is
        // closest to addr_idx, preferring "immediately after" over
        // "immediately before" on a tie.
        let mut best: Option<(&str, usize, isize)> = None; // (prop, |delta|, signed_delta)
        for vp in writes_val.keys().copied() {
            let v_idx = family.members[vp].assignment_index;
            let signed = v_idx as isize - addr_idx as isize;
            let abs = signed.unsigned_abs();
            let candidate = (vp, abs, signed);
            best = match best {
                None => Some(candidate),
                Some((_, prev_abs, prev_signed)) => {
                    if abs < prev_abs || (abs == prev_abs && signed > prev_signed) {
                        Some(candidate)
                    } else {
                        best
                    }
                }
            };
        }

        let addr_decomposition = decompose_addr_expr(addr_expr);
        if let Some((vp, _, _)) = best {
            let ve = writes_val[vp];
            writes_val.remove(vp);
            let val_indirect_read =
                recognise_indirect_read(ve, &pointer_mirrors, assignment_index);
            writes.push(WriteEntry {
                addr_property: ap.to_string(),
                val_property: vp.to_string(),
                addr_expr: addr_expr.clone(),
                val_expr: ve.clone(),
                addr_decomposition,
                val_indirect_read,
            });
        } else {
            writes.push(WriteEntry {
                addr_property: ap.to_string(),
                val_property: String::new(),
                addr_expr: addr_expr.clone(),
                val_expr: Expr::Literal(0.0),
                addr_decomposition,
                val_indirect_read: None,
            });
        }
    }

    // A real loop has at least a counter or a pointer or a write. An IP
    // body alone is not enough — that would describe a bare unconditional
    // jump-back, with no termination. Refuse those (cardinal-rule:
    // unbounded loops aren't safe to fast-forward).
    if counter.is_none() && pointers.is_empty() && writes.is_empty() {
        return None;
    }

    // For determinism, sort pointer entries by property name.
    pointers.sort_by(|a, b| a.property.cmp(&b.property));

    let bulk_class = classify_bulk(&pointers, &writes);

    let per_iter_cycles = extract_per_iter_cycles(family, key_value);

    let ip_extra_advance_slot = ip_shape.extra_advance_slot.clone();

    let comparison_shape = if flag_conditioned && writes.is_empty() && !pointers.is_empty() {
        extract_comparison_shape(family, key_value, &pointers, &predicate, assignment_index)
    } else {
        None
    };

    let precondition = extract_precondition(assignment_index, ip_prop);

    Some(LoopDescriptor {
        key_property: family.key_property.clone(),
        key_value,
        ip_property: ip_prop.to_string(),
        ip_self_property: ip_shape.self_property,
        ip_advance_literal: ip_shape.advance_literal,
        predicate_properties,
        predicate,
        predicate_means_stay: ip_shape.predicate_means_stay,
        counter,
        pointers,
        writes,
        flag_conditioned,
        bulk_class,
        per_iter_cycles,
        ip_extra_advance_slot,
        comparison_shape,
        precondition,
    })
}

/// Walk the IP slot's top-level assignment expression and capture every
/// `StyleCondition` wrapper whose `fallback` is what gets descended into
/// to find the inner dispatch. The branches of each such wrapper are
/// "override conditions": each branch's `condition` is a `StyleTest`
/// that, when true, replaces the dispatch's normal result for this
/// tick. The dispatch fires iff *none* of those branch conditions
/// match the current state — that's the cabinet's own outer guard.
///
/// **Cardinal-rule shape.** This matcher inspects only:
/// - `Expr` node shape (`StyleCondition`, `Calc`).
/// - The downstream `find_inner_dispatch` rule (used to identify which
///   side of a wrapper contains the dispatch).
///
/// It does NOT inspect any character of any property name. The returned
/// `StyleTest`s carry whatever opaque slot names the cabinet used. A
/// cabinet with no outer wrappers (a bare dispatch on the IP slot)
/// produces `None`. A cabinet whose wrappers test different slot names
/// produces `Some(Precondition::NoOverrides(<those tests verbatim>))`
/// — renaming the override slots does not affect the extraction.
fn extract_precondition(
    assignment_index: &HashMap<&str, &Expr>,
    ip_prop: &str,
) -> Option<Precondition> {
    let top = *assignment_index.get(ip_prop)?;
    let mut overrides: Vec<StyleTest> = Vec::new();
    collect_override_branches(top, &mut overrides);
    if overrides.is_empty() {
        None
    } else {
        Some(Precondition::NoOverrides(overrides))
    }
}

/// Recursively walk an expression, collecting any `StyleCondition`
/// wrapper-branch conditions encountered on the way down to the inner
/// dispatch. A wrapper is identified by the same rule
/// `find_inner_dispatch` uses for descent: the dispatch lives in the
/// `fallback`, and the branches are "override" cases that produce a
/// non-dispatch result.
///
/// Mirrors `find_inner_dispatch`'s descent rule. Differs only in what
/// it returns — instead of the inner dispatch `Expr`, it collects
/// override branch conditions along the way.
fn collect_override_branches(expr: &Expr, out: &mut Vec<StyleTest>) {
    // Direct dispatch — no wrapper at this level, stop.
    if is_single_key_dispatch(expr) {
        return;
    }
    // Descend through Calc — record nothing; calc wrappers don't
    // gate the dispatch.
    if let Expr::Calc(op) = expr {
        descend_calc_for_overrides(op, out);
        return;
    }
    // StyleCondition: if the fallback contains the inner dispatch, the
    // branch conditions here are override gates.
    if let Expr::StyleCondition { branches, fallback } = expr {
        if !branches.is_empty() {
            if find_inner_dispatch(fallback).is_some() {
                for b in branches {
                    out.push(b.condition.clone());
                }
                collect_override_branches(fallback, out);
                return;
            }
            // The other accepted shape in find_inner_dispatch:
            // mixed-key StyleCondition with a dominant dispatch key.
            // In this case the dispatch IS the StyleCondition itself
            // (with some branches being wrapper overrides folded into
            // the same chain). The folded wrapper branches are NOT
            // captured as preconditions here — they're already part of
            // the dispatch structure that the per-key body extraction
            // skips over. Treat as no-wrapper-at-this-level.
        }
    }
}

fn descend_calc_for_overrides(op: &CalcOp, out: &mut Vec<StyleTest>) {
    match op {
        CalcOp::Add(a, b)
        | CalcOp::Sub(a, b)
        | CalcOp::Mul(a, b)
        | CalcOp::Div(a, b)
        | CalcOp::Mod(a, b)
        | CalcOp::Pow(a, b) => {
            if find_inner_dispatch(a).is_some() {
                collect_override_branches(a, out);
            } else if find_inner_dispatch(b).is_some() {
                collect_override_branches(b, out);
            }
        }
        CalcOp::Min(args) | CalcOp::Max(args) => {
            for arg in args {
                if find_inner_dispatch(arg).is_some() {
                    collect_override_branches(arg, out);
                    break;
                }
            }
        }
        CalcOp::Clamp(a, b, c) => {
            for arg in [a.as_ref(), b.as_ref(), c.as_ref()] {
                if find_inner_dispatch(arg).is_some() {
                    collect_override_branches(arg, out);
                    break;
                }
            }
        }
        CalcOp::Round(_, a, b) => {
            if find_inner_dispatch(a).is_some() {
                collect_override_branches(a, out);
            } else if find_inner_dispatch(b).is_some() {
                collect_override_branches(b, out);
            }
        }
        CalcOp::Sign(a) | CalcOp::Abs(a) | CalcOp::Negate(a) => {
            collect_override_branches(a, out);
        }
    }
}

/// Evaluate a captured [`Precondition`] against the current runtime
/// slot view. Used by the applier at entry to decide whether the
/// cabinet's normal dispatch branch is what just fired this tick —
/// when this returns `false`, the cabinet's outer wrapper took over
/// and produced its own (correct) post-state; the applier has
/// nothing to do.
///
/// Cardinal-rule note: only slot identity (whole-name lookup via
/// `program.property_slots` / `state.state_vars`) is used to read
/// each tested slot's value. The `Expr` on the right-hand side of a
/// `StyleTest::Single` is evaluated by [`evaluate_style_test_rhs`],
/// which handles the literal-only shapes the recogniser captures.
pub(crate) fn evaluate_precondition(
    pre: &Precondition,
    program: &crate::compile::CompiledProgram,
    state: &crate::state::State,
    slots: &[i32],
) -> bool {
    match pre {
        Precondition::NoOverrides(overrides) => {
            // Dispatch fires iff *every* override condition is false.
            !overrides
                .iter()
                .any(|t| evaluate_style_test_runtime(t, program, state, slots))
        }
    }
}

/// Evaluate the loop-continuation gate for a descriptor against the
/// post-tick slot view: did the CSS take the *stay* branch of the IP
/// body this tick?
///
/// `true` means the loop is mid-flight — the dispatch held the IP slot
/// at its value this tick and there are remaining iterations for the
/// applier to collapse. `false` means there is nothing to fast-forward:
/// either the op ran as a single-shot (no active loop) or the CSS just
/// executed the final iteration and already advanced past the loop.
///
/// This is the structural generalisation of the hardcoded path's
/// "hasREP != 1 → return" and "REPE/REPNE already exited via post-tick
/// ZF → return" entry guards: the cabinet's own continuation predicate
/// (read off the IP body) folds all of those into one test. Evaluating
/// it against the same post-tick slots the IP body read this tick
/// reproduces this tick's stay/advance decision exactly.
///
/// Cardinal-rule note: the predicate is the cabinet's own `StyleTest`,
/// opaque names and all; `predicate_means_stay` is a structural fact
/// about which branch held the stay body. Nothing here reads a name's
/// characters or assumes upstream meaning.
pub(crate) fn evaluate_loop_predicate(
    descriptor: &LoopDescriptor,
    program: &crate::compile::CompiledProgram,
    state: &crate::state::State,
    slots: &[i32],
) -> bool {
    evaluate_style_test_runtime(&descriptor.predicate, program, state, slots)
        == descriptor.predicate_means_stay
}

/// Evaluate a single [`StyleTest`] against the runtime slot view.
///
/// Mirrors the resolver inside `rep_applier::read_prop`: try
/// `program.property_slots` first, then `state.state_vars` by bare
/// name. Comparison is integer equality (consistent with how the rest
/// of calcite-core evaluates `style()` tests — see
/// `Evaluator::eval_style_test`).
///
/// `StyleTest::Single` is the only shape the precondition recogniser
/// emits today (outer wrapper branch conditions are always single
/// `style(--prop: value)` tests in kiln-emitted cabinets), but
/// `And`/`Or` are handled too so the evaluator works on hand-built
/// test descriptors.
fn evaluate_style_test_runtime(
    test: &StyleTest,
    program: &crate::compile::CompiledProgram,
    state: &crate::state::State,
    slots: &[i32],
) -> bool {
    match test {
        StyleTest::Single { property, value } => {
            let prop_val = read_prop_runtime(program, state, slots, property);
            let test_val = evaluate_style_test_rhs(value);
            prop_val == test_val
        }
        StyleTest::And(tests) => tests
            .iter()
            .all(|t| evaluate_style_test_runtime(t, program, state, slots)),
        StyleTest::Or(tests) => tests
            .iter()
            .any(|t| evaluate_style_test_runtime(t, program, state, slots)),
    }
}

/// Resolve a property name's current value through the same routing the
/// compiler gives reads of that name (see `rep_applier::read_prop` for
/// the full rationale): buffer-copy names skip the slot table and read
/// the committed state var via the canonical bare name (`to_bare_name`
/// — pure function, debugger-thread-safe, unlike the thread-local
/// address map); everything else tries the compiled slot view first.
/// Missing slots resolve to 0 (matches the `Evaluator::resolve_property`
/// default for unset numeric slots).
fn read_prop_runtime(
    program: &crate::compile::CompiledProgram,
    state: &crate::state::State,
    slots: &[i32],
    name: &str,
) -> i64 {
    if !crate::eval::is_buffer_copy(name) {
        if let Some(&s) = program.property_slots.get(name) {
            return slots[s as usize] as i64;
        }
    }
    state
        .get_var(crate::eval::to_bare_name(name))
        .map(|v| v as i64)
        .unwrap_or(0)
}

/// Evaluate the RHS of a `StyleTest::Single`. The recogniser only
/// captures `Expr::Literal` RHS values (cabinet outer wrappers test
/// against constants like `1` / `0`). Anything else is treated as
/// the comparison failing — the captured shape didn't fit a literal,
/// so we conservatively report "this override fired" by returning a
/// value the LHS can never match.
fn evaluate_style_test_rhs(value: &Expr) -> i64 {
    match value {
        Expr::Literal(v) => *v as i64,
        // Captured wrapper conditions today only test against literals.
        // If a future cabinet shape emits a non-literal RHS we'd need
        // a richer evaluator; for now bail to a sentinel that won't
        // match any LHS slot read.
        _ => i64::MIN,
    }
}

/// Extract the structural comparison shape for a flag-conditioned
/// ReadOnly loop (CMPS / SCAS family) at this opcode.
///
/// **Rule.** Walk the dispatch family looking for a "comparison" member:
/// a member whose per-key body for `key_value` has shape
/// `Calc(Sub(a, b))` (or `Calc(Add(a, Negate(b)))`), where the two
/// operands transitively reference the loop's pointer mirror slots or
/// accumulator-shaped bare `Var`s. The cabinet's comparison dispatch
/// (8086: `--_cmpDiff`) emits exactly this shape per-opcode, mapping the
/// upstream "compute the difference" intent onto a pure CSS arithmetic
/// dispatch.
///
/// The matcher inspects only:
/// - The shape of `Expr` nodes (`Calc(Sub(...))`, `FunctionCall`, `Var`).
/// - Whole-name equality (pointer mirror lookup, intermediate body
///   lookup via `assignment_index`).
///
/// It does NOT inspect characters of any name, look at the literal `0xA6`
/// vs `0xA7`, or assume any particular function name for the comparison.
///
/// **Cardinal-rule probe.** A cabinet whose comparison dispatch member
/// uses different slot names produces an equivalent `ComparisonShape`
/// with those names. A cabinet without such a member produces `None`.
fn extract_comparison_shape<'a>(
    family: &DispatchFamily<'a>,
    key_value: i64,
    pointers: &[PointerEntry],
    predicate: &StyleTest,
    assignment_index: &HashMap<&'a str, &'a Expr>,
) -> Option<ComparisonShape> {
    // Pointer mirror set: the prior-tick slot names the pointer step
    // bodies read. Used as the "reaches a pointer" test for the
    // comparison operands.
    let mirrors: HashSet<&str> = pointers.iter().map(|p| p.self_property.as_str()).collect();

    // Exclude family members that have been classified upstream as the
    // pointer / counter / IP slots — their bodies are loop machinery,
    // not data comparison. (The pointer step body has a Calc(Sub(...))
    // shape too, computing the next pointer value; we'd match it
    // spuriously without this filter.)
    let pointer_props: HashSet<&str> = pointers.iter().map(|p| p.property.as_str()).collect();

    // Search family members for one whose body for this key is a Sub
    // with two operands that each trace to a memory read keyed on a
    // pointer mirror (CMPS) or one such + a bare-Var operand (SCAS).
    let mut found: Option<(&Expr, &Expr)> = None;
    for (mprop, member) in &family.members {
        if pointer_props.contains(*mprop) {
            continue;
        }
        let Some(body) = member.bodies.get(&key_value) else { continue };
        if let Some(pair) = match_subtraction_operands(body) {
            let (a, b) = pair;
            // Comparison shape: both operands trace to data (either a
            // memory read through a pointer mirror via an intermediate
            // FunctionCall, or — for SCAS — one operand is a bare Var
            // accumulator). We require:
            //   - At least one operand traces to a memory read through
            //     a pointer mirror (the "destination" side).
            //   - The other operand is either a bare Var (accumulator)
            //     or also traces to a memory read through a (different)
            //     pointer mirror.
            // We do NOT accept "operand contains a Var referencing the
            // pointer mirror without going through an intermediate
            // FunctionCall read" — that's the pointer-advance body
            // shape, not a data comparison.
            let a_reads = operand_reads_through_pointer(a, &mirrors, assignment_index);
            let b_reads = operand_reads_through_pointer(b, &mirrors, assignment_index);
            let a_acc = matches!(a, Expr::Var { fallback: None, .. })
                && !mirrors.contains(match_bare_var(a).as_deref().unwrap_or(""));
            let b_acc = matches!(b, Expr::Var { fallback: None, .. })
                && !mirrors.contains(match_bare_var(b).as_deref().unwrap_or(""));
            if (a_reads && b_reads) || (a_reads && b_acc) || (b_reads && a_acc) {
                found = Some(pair);
                break;
            }
        }
    }
    let (lhs, rhs) = found?;

    // Identify which operand is the destination side. Convention:
    // `pointers[0]` is the destination (alphabetic sort places `DI`
    // before `SI`, but this is purely a convention — the comparison
    // arithmetic is symmetric for the bytes-equal check that gates
    // early exit, and the applier uses the same convention).
    if pointers.is_empty() {
        return None;
    }
    let dst_ptr_mirror = pointers[0].self_property.as_str();
    let src_ptr_mirror_opt = pointers.get(1).map(|p| p.self_property.as_str());

    // Resolve dst operand and dst segment.
    let dst_operand = if reaches_specific_pointer_mirror(lhs, dst_ptr_mirror, assignment_index) {
        lhs
    } else if reaches_specific_pointer_mirror(rhs, dst_ptr_mirror, assignment_index) {
        rhs
    } else {
        return None;
    };
    let (dst_seg_property, dst_ptr_property, dst_seg_times_sixteen) =
        trace_segmented_read(dst_operand, dst_ptr_mirror, assignment_index)?;

    // The OTHER operand is the source.
    let src_operand = if std::ptr::eq(dst_operand, lhs) { rhs } else { lhs };

    let source = if let Some(src_ptr_mirror) = src_ptr_mirror_opt {
        // CMPS shape: src operand must trace to the second pointer
        // mirror and decompose to a segmented memory read.
        let (src_seg, src_ptr, src_scaled) =
            trace_segmented_read(src_operand, src_ptr_mirror, assignment_index)?;
        ComparisonSource::Pointer {
            seg_property: src_seg,
            seg_times_sixteen: src_scaled,
            ptr_property: src_ptr,
        }
    } else {
        // SCAS shape: src operand is an accumulator. Either a bare
        // Var (byte: `var(--AL)`) or a Calc of bare Vars (word:
        // `calc(var(--AH) * 256 + var(--AL))` or similar).
        extract_accumulator(src_operand)?
    };

    let width = pointers[0].base_step as u8;
    let rep_type_property =
        identify_rep_type_slot(predicate, family, key_value, assignment_index);

    Some(ComparisonShape {
        width,
        dst_seg_property,
        dst_seg_times_sixteen,
        dst_ptr_property,
        source,
        rep_type_property,
    })
}

/// Match `Calc(Sub(a, b))` or `Calc(Add(a, Negate(b)))` and return the
/// two operands `(a, b)` (in the order they appear in the subtraction:
/// `lhs - rhs`).
fn match_subtraction_operands(expr: &Expr) -> Option<(&Expr, &Expr)> {
    // Peel an outer StyleCondition fallback (the `repGuardReg` wrapper
    // typically wraps `Calc(...)` in `if(...: oldVal; else: <real>)`).
    let inner = match expr {
        Expr::StyleCondition { fallback, .. } => fallback.as_ref(),
        other => other,
    };
    // Peel a top-level Add of two sub-expressions: the cabinet may
    // emit `calc(subFlags(a, b) + and(flags, mask))` where one side is
    // a flag-mask term we want to skip past.
    if let Expr::Calc(CalcOp::Add(left, right)) = inner {
        if let Some(pair) = match_subtraction_operands_inner(left) {
            return Some(pair);
        }
        if let Some(pair) = match_subtraction_operands_inner(right) {
            return Some(pair);
        }
    }
    match_subtraction_operands_inner(inner)
}

fn match_subtraction_operands_inner(expr: &Expr) -> Option<(&Expr, &Expr)> {
    match expr {
        Expr::Calc(CalcOp::Sub(a, b)) => Some((a.as_ref(), b.as_ref())),
        // FunctionCall with exactly 2 args is treated as a comparison
        // primitive when its args reach the pointer mirrors (see
        // `reaches_pointer_mirror` check by the caller). This lets us
        // pick up cabinets that route the compare through a named
        // function (e.g. `subFlags8(src, dst)`) rather than a bare Sub.
        Expr::FunctionCall { args, .. } if args.len() == 2 => Some((&args[0], &args[1])),
        _ => None,
    }
}

/// True iff the operand represents a memory read through one of the
/// pointer mirror slots. Accepted shapes:
///
/// 1. A bare `Var(name)` whose top-level body is a `FunctionCall(_, args)`
///    where the args tree references a pointer mirror (canonical 8086
///    `--_strDstByte = --readMem(calc(--ES * 16 + --DI))` shape).
/// 2. A `Calc` tree containing such a Var (word concat shape:
///    `calc(var(--lo) + var(--hi) * 256)` where `--lo` is the
///    intermediate read).
/// 3. A direct `FunctionCall(_, args)` whose args reference the mirror.
///
/// This is more restrictive than `reaches_pointer_mirror` — that helper
/// also returns true for operands that just *contain* a `Var(pointer)`
/// reference in some Calc tree (the pointer-advance body shape). For
/// comparison-operand identification we need to filter those out.
fn operand_reads_through_pointer<'a>(
    expr: &Expr,
    mirrors: &HashSet<&str>,
    assignment_index: &HashMap<&'a str, &'a Expr>,
) -> bool {
    match expr {
        Expr::Var { name, fallback: None } => {
            if let Some(body) = assignment_index.get(name.as_str()) {
                if let Expr::FunctionCall { args, .. } = body {
                    return args.iter().any(|a| expr_references_any(a, mirrors));
                }
            }
            false
        }
        Expr::Calc(op) => operand_reads_through_pointer_in_calc(op, mirrors, assignment_index),
        Expr::FunctionCall { args, .. } => args.iter().any(|a| expr_references_any(a, mirrors)),
        _ => false,
    }
}

fn operand_reads_through_pointer_in_calc<'a>(
    op: &CalcOp,
    mirrors: &HashSet<&str>,
    assignment_index: &HashMap<&'a str, &'a Expr>,
) -> bool {
    match op {
        CalcOp::Add(a, b)
        | CalcOp::Sub(a, b)
        | CalcOp::Mul(a, b)
        | CalcOp::Div(a, b)
        | CalcOp::Mod(a, b)
        | CalcOp::Pow(a, b) => {
            operand_reads_through_pointer(a, mirrors, assignment_index)
                || operand_reads_through_pointer(b, mirrors, assignment_index)
        }
        _ => false,
    }
}

/// True iff the expression — possibly tracing through a single layer of
/// intermediate slot (a top-level `Var(name)` whose body is in
/// `assignment_index`) — references one of the pointer mirror slots.
fn reaches_pointer_mirror<'a>(
    expr: &Expr,
    mirrors: &HashSet<&str>,
    assignment_index: &HashMap<&'a str, &'a Expr>,
) -> bool {
    if expr_references_any(expr, mirrors) {
        return true;
    }
    // Trace one layer through an intermediate slot's body.
    if let Expr::Var { name, fallback: None } = expr {
        if let Some(body) = assignment_index.get(name.as_str()) {
            return expr_references_any(body, mirrors);
        }
    }
    // Also walk Calc trees for a top-level word-concat shape:
    //   calc(var(intermediate) + var(intermediate_hi) * 256)
    // each sub-Var of which may be an intermediate.
    if let Expr::Calc(op) = expr {
        match op {
            CalcOp::Add(a, b)
            | CalcOp::Sub(a, b)
            | CalcOp::Mul(a, b)
            | CalcOp::Div(a, b)
            | CalcOp::Mod(a, b)
            | CalcOp::Pow(a, b) => {
                if reaches_pointer_mirror(a, mirrors, assignment_index)
                    || reaches_pointer_mirror(b, mirrors, assignment_index)
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Like `operand_reads_through_pointer`, but restricted to a single
/// specific mirror name. Used to disambiguate which operand is the
/// destination side once we've identified the comparison pair.
fn reaches_specific_pointer_mirror<'a>(
    expr: &Expr,
    mirror: &str,
    assignment_index: &HashMap<&'a str, &'a Expr>,
) -> bool {
    let mut set: HashSet<&str> = HashSet::new();
    set.insert(mirror);
    operand_reads_through_pointer(expr, &set, assignment_index)
}

/// Trace an operand of the comparison subtraction down to the segment
/// slot it reads through. The operand shape may be:
///
/// - A bare `Var(intermediate)` whose top-level body is
///   `FunctionCall(_, [<calc(seg*16 + ptr)>])` (8086's
///   `--readMem(calc(--ES * 16 + --DI))`-shaped pre-computed byte).
/// - A `Calc(Add(...))` that itself contains the same shape, for word
///   reads composed as `low + hi * 256`.
/// - A direct `FunctionCall(_, args)` of similar shape.
///
/// Returns `(seg_property, ptr_property)` — the names the cabinet uses
/// for the segment base and the pointer mirror. The pointer mirror
/// matches `pointer_mirror_target`.
fn trace_segmented_read<'a>(
    expr: &Expr,
    pointer_mirror_target: &str,
    assignment_index: &HashMap<&'a str, &'a Expr>,
) -> Option<(String, String, bool)> {
    // Case 1: intermediate slot — peek through to its body.
    if let Expr::Var { name, fallback: None } = expr {
        if let Some(body) = assignment_index.get(name.as_str()) {
            return trace_segmented_read(body, pointer_mirror_target, assignment_index);
        }
    }
    // Case 2: FunctionCall — descend into args.
    if let Expr::FunctionCall { args, .. } = expr {
        for a in args {
            if let Some(triple) = trace_segmented_read(a, pointer_mirror_target, assignment_index)
            {
                return Some(triple);
            }
            // Direct ×16 decomposition at the arg level — the slot holds
            // an unscaled segment the shape multiplies by 16.
            if let Some((seg, ptr)) = match_seg_plus_ptr(a, pointer_mirror_target) {
                return Some((seg, ptr, true));
            }
            // The 8086 source-side dispatches through an intermediate
            // `--_strSrcSeg = if(...: var(--segOverride); else: calc(--DS * 16))`
            // shape — so the base "slot" is itself a Var(intermediate)
            // holding the already-scaled addend. Capture the
            // intermediate's name and the ptr name from the surrounding
            // shape, with the no-scaling flag.
            if let Some((seg, ptr)) = match_segvar_plus_ptr(a, pointer_mirror_target) {
                return Some((seg, ptr, false));
            }
        }
    }
    // Case 3: Calc tree — descend through arithmetic ops to find the
    // canonical seg-pointer shape buried inside.
    if let Expr::Calc(op) = expr {
        return calc_trace_segmented_read(op, pointer_mirror_target, assignment_index);
    }
    None
}

fn calc_trace_segmented_read<'a>(
    op: &CalcOp,
    pointer_mirror_target: &str,
    assignment_index: &HashMap<&'a str, &'a Expr>,
) -> Option<(String, String, bool)> {
    match op {
        CalcOp::Add(a, b)
        | CalcOp::Sub(a, b)
        | CalcOp::Mul(a, b)
        | CalcOp::Div(a, b)
        | CalcOp::Mod(a, b)
        | CalcOp::Pow(a, b) => trace_segmented_read(a, pointer_mirror_target, assignment_index)
            .or_else(|| trace_segmented_read(b, pointer_mirror_target, assignment_index)),
        _ => None,
    }
}

/// Match `calc(var(seg) * 16 + var(ptr))` or the reversed orientation,
/// returning `(seg_name, ptr_name)` when `ptr_name == pointer_mirror_target`.
fn match_seg_plus_ptr(expr: &Expr, pointer_mirror_target: &str) -> Option<(String, String)> {
    let (seg, ptr) = match_segmented_address(expr)?;
    if ptr == pointer_mirror_target {
        Some((seg, ptr))
    } else {
        None
    }
}

/// Match `calc(var(seg_intermediate) + var(ptr))` (where the seg side is
/// already shifted by the intermediate) — the 8086 `--_strSrcSeg` case.
/// Returns `(seg_intermediate, ptr)` when `ptr == pointer_mirror_target`.
fn match_segvar_plus_ptr(expr: &Expr, pointer_mirror_target: &str) -> Option<(String, String)> {
    let Expr::Calc(CalcOp::Add(a, b)) = expr else { return None };
    let (av, bv) = (match_bare_var(a), match_bare_var(b));
    match (av, bv) {
        (Some(left), Some(right)) => {
            if left == pointer_mirror_target {
                Some((right, left))
            } else if right == pointer_mirror_target {
                Some((left, right))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Extract the accumulator-side operand of a SCAS-shape comparison.
/// Accepted shapes:
/// - Bare `Var(name)` — byte accumulator: `byte_property = name`,
///   `word_property = name`.
/// - `Calc(Add(var(low), Calc(Mul(var(hi), Literal(256)))))` or
///   reassociations — word accumulator. Currently captured as the bare
///   form by recording the outermost Var (low byte) as `byte_property`
///   and the parent Var as `word_property` when the cabinet uses
///   `var(--__1AX)` as the word read. For the synthetic-cabinet tests
///   we accept a bare Var for both byte and word forms.
fn extract_accumulator(expr: &Expr) -> Option<ComparisonSource> {
    // Strip one layer of intermediate Var.
    if let Some(name) = match_bare_var(expr) {
        return Some(ComparisonSource::Accumulator {
            byte_property: name.clone(),
            word_property: name,
        });
    }
    // For word concat shapes (var(lo) + var(hi) * 256), capture the
    // low-byte var as byte_property and itself as word_property; the
    // applier reads `word_property` when width=2.
    if let Expr::Calc(CalcOp::Add(a, b)) = expr {
        if let (Some(lo), _) = (match_bare_var(a), b) {
            return Some(ComparisonSource::Accumulator {
                byte_property: lo.clone(),
                word_property: lo,
            });
        }
        if let (_, Some(lo)) = (a, match_bare_var(b)) {
            return Some(ComparisonSource::Accumulator {
                byte_property: lo.clone(),
                word_property: lo,
            });
        }
    }
    None
}

/// Identify the rep-type discriminator slot among predicate slots.
///
/// **Rule.** Among `StyleTest::Single` slots in the IP-body predicate,
/// pick the one that:
///   1. Takes more than one distinct literal value across the
///      disjunctive branches (i.e. the slot is keyed differently per
///      branch — characteristic of a discriminator).
///   2. Is NOT the comparison-result flag-bit slot — defined as the
///      slot whose top-level body transitively depends on the
///      comparison member's output. The comparison member is identified
///      structurally above (same routine that produced the dst/src
///      operands).
///
/// When exactly one slot satisfies both, return it. When zero or more
/// than one do, return `None` — the applier will conservatively bail
/// (Unsupported) rather than guess.
fn identify_rep_type_slot<'a>(
    predicate: &StyleTest,
    family: &DispatchFamily<'a>,
    key_value: i64,
    assignment_index: &HashMap<&'a str, &'a Expr>,
) -> Option<String> {
    // Collect per-disjunctive-branch single-test slot→value mappings.
    let branches = match predicate {
        StyleTest::Or(parts) => parts.clone(),
        StyleTest::And(_) | StyleTest::Single { .. } => vec![predicate.clone()],
    };
    if branches.len() < 2 {
        return None;
    }

    // For each slot, collect the set of distinct literal values it
    // takes across branches.
    let mut per_slot_values: HashMap<String, HashSet<i64>> = HashMap::new();
    for b in &branches {
        let pairs = collect_single_pairs(b);
        for (prop, val) in pairs {
            per_slot_values.entry(prop).or_default().insert(val);
        }
    }

    // The comparison-flag-bit slot is the one whose top-level body
    // transitively depends on the comparison member's output. Identify
    // the comparison-output property: the property of the family
    // member we picked above (whose body is the Sub). We re-derive it
    // here for locality.
    let cmp_output_props: HashSet<&str> = family
        .members
        .iter()
        .filter_map(|(prop, member)| {
            let body = member.bodies.get(&key_value)?;
            if match_subtraction_operands(body).is_some() {
                Some(*prop)
            } else {
                None
            }
        })
        .collect();

    // Candidates: slots with >1 distinct value AND whose top-level body
    // does NOT transitively reach any comparison output.
    let mut candidates: Vec<String> = Vec::new();
    for (slot, values) in &per_slot_values {
        if values.len() < 2 {
            continue;
        }
        // Reach test: does this slot's body depend on the comparison
        // output (i.e. eventually read from `cmp_output_props`)?
        if let Some(body) = assignment_index.get(slot.as_str()) {
            if expr_references_any(body, &cmp_output_props) {
                continue;
            }
            // Indirect: walk through one layer of intermediate Var
            // references in case the dependency is two hops deep.
            if expr_reaches_through_intermediates(body, &cmp_output_props, assignment_index, 3) {
                continue;
            }
        }
        candidates.push(slot.clone());
    }

    if candidates.len() == 1 {
        return Some(candidates.into_iter().next().unwrap());
    }
    None
}

/// Collect (slot, literal_value) pairs from a flat conjunction
/// `StyleTest`. Returns one entry per `StyleTest::Single { property,
/// value: Literal(_) }` in the tree.
fn collect_single_pairs(test: &StyleTest) -> Vec<(String, i64)> {
    let mut out = Vec::new();
    walk_test(test, &mut |t| {
        if let StyleTest::Single { property, value } = t {
            if let Expr::Literal(v) = value {
                out.push((property.clone(), *v as i64));
            }
        }
    });
    out
}

/// Walk through up to `depth` layers of intermediate `Var(name)` slot
/// references, returning true if any slot's body reaches the targets.
fn expr_reaches_through_intermediates<'a>(
    expr: &Expr,
    targets: &HashSet<&str>,
    assignment_index: &HashMap<&'a str, &'a Expr>,
    depth: u32,
) -> bool {
    if depth == 0 {
        return false;
    }
    if expr_references_any(expr, targets) {
        return true;
    }
    // Collect intermediate Var names referenced anywhere in the tree
    // and recurse on their bodies.
    let mut seen: HashSet<String> = HashSet::new();
    collect_var_names(expr, &mut seen);
    for n in seen {
        if let Some(body) = assignment_index.get(n.as_str()) {
            if expr_reaches_through_intermediates(body, targets, assignment_index, depth - 1) {
                return true;
            }
        }
    }
    false
}

fn collect_var_names(expr: &Expr, out: &mut HashSet<String>) {
    match expr {
        Expr::Var { name, fallback } => {
            out.insert(name.clone());
            if let Some(fb) = fallback {
                collect_var_names(fb, out);
            }
        }
        Expr::Literal(_) | Expr::StringLiteral(_) => {}
        Expr::Calc(op) => collect_var_names_in_calc(op, out),
        Expr::StyleCondition { branches, fallback } => {
            for b in branches {
                collect_var_names_in_test(&b.condition, out);
                collect_var_names(&b.then, out);
            }
            collect_var_names(fallback, out);
        }
        Expr::FunctionCall { args, .. } => {
            for a in args {
                collect_var_names(a, out);
            }
        }
        Expr::Concat(parts) => {
            for p in parts {
                collect_var_names(p, out);
            }
        }
    }
}

fn collect_var_names_in_calc(op: &CalcOp, out: &mut HashSet<String>) {
    match op {
        CalcOp::Add(a, b)
        | CalcOp::Sub(a, b)
        | CalcOp::Mul(a, b)
        | CalcOp::Div(a, b)
        | CalcOp::Mod(a, b)
        | CalcOp::Pow(a, b) => {
            collect_var_names(a, out);
            collect_var_names(b, out);
        }
        CalcOp::Min(args) | CalcOp::Max(args) => {
            for a in args {
                collect_var_names(a, out);
            }
        }
        CalcOp::Clamp(a, b, c) => {
            collect_var_names(a, out);
            collect_var_names(b, out);
            collect_var_names(c, out);
        }
        CalcOp::Round(_, a, b) => {
            collect_var_names(a, out);
            collect_var_names(b, out);
        }
        CalcOp::Sign(a) | CalcOp::Abs(a) | CalcOp::Negate(a) => collect_var_names(a, out),
    }
}

fn collect_var_names_in_test(test: &StyleTest, out: &mut HashSet<String>) {
    match test {
        StyleTest::Single { property: _, value } => collect_var_names(value, out),
        StyleTest::And(parts) | StyleTest::Or(parts) => {
            for p in parts {
                collect_var_names_in_test(p, out);
            }
        }
    }
}

/// Find the per-iteration cycle cost literal for one opcode, structurally.
///
/// **Rule.** Among the dispatch family's members, pick the one with the
/// highest count of bodies matching the shape
/// `Calc(Add(Var(X), Literal(K)))` (with X the same opaque slot reference
/// across all matching keys for that member). Call that member "the
/// cycle-counter family member". For the requested `key_value`, return
/// `Some(K)` if its body in that member matches the shape; `None`
/// otherwise (no such family member exists, or this opcode is not one
/// of the per-key bodies the chosen member covers).
///
/// **Why this rule.** A CPU cabinet that wants per-instruction cycle
/// accounting emits exactly this shape: one slot dispatched on the
/// opcode key, each branch returning `var(self_mirror) + per_opcode_K`.
/// No matter what the slot name is (`--cycleCount`, `--zorch`,
/// `--moodMeter`), no matter the cabinet's ISA, the dispatch shape is
/// the same. The "most participants wins" tiebreaker is itself
/// structural — the cycle counter is the family member with the
/// broadest opcode coverage of the self-add-literal shape.
///
/// **Cardinal-rule guarantee.** No characters of any name are inspected.
/// The recogniser compares `Var.name` strings only for whole-string
/// equality (to detect "same X across keys"). A cabinet without a
/// per-iter cycle-style dispatch returns `None`, and the applier bails
/// loudly rather than charging a fabricated cost.
fn extract_per_iter_cycles(family: &DispatchFamily<'_>, key_value: i64) -> Option<i32> {
    // For each member, compute the (best_X, count) over its bodies: the
    // single Var slot reference shared by the most "Var(X) + Literal(K)"
    // bodies, and how many such bodies use it.
    let mut best_member: Option<(&str, usize, &HashMap<i64, &Expr>, String)> = None;
    for (&prop, member) in &family.members {
        let Some((shared_x, count)) = best_shared_var_for_self_add_literal(&member.bodies) else {
            continue;
        };
        let replace = match best_member.as_ref() {
            None => true,
            Some(&(prev_prop, prev_count, _, _)) => {
                if count > prev_count {
                    true
                } else if count < prev_count {
                    false
                } else {
                    // Tie on count — break deterministically by
                    // lexicographic property-name order so the chosen
                    // member is stable across HashMap iteration orders.
                    prop < prev_prop
                }
            }
        };
        if replace {
            best_member = Some((prop, count, &member.bodies, shared_x));
        }
    }

    let (_, _, bodies, shared_x) = best_member?;
    let body = bodies.get(&key_value)?;
    let (x, k) = decompose_var_plus_literal(body)?;
    if x != shared_x {
        return None;
    }
    Some(k)
}

/// Decompose an expression matching shape `Calc(Add(Var(X), Literal(K)))`
/// (or commutative `Literal(K) + Var(X)`) into `(X, K)`. Returns `None`
/// for any other shape, including `Calc(Add(Var(X), Var(Y)))` (two vars)
/// and `Calc(Add(Literal(K1), Literal(K2)))` (two literals).
fn decompose_var_plus_literal(expr: &Expr) -> Option<(String, i32)> {
    let Expr::Calc(CalcOp::Add(a, b)) = expr else { return None };
    match (a.as_ref(), b.as_ref()) {
        (Expr::Var { name, fallback: _ }, Expr::Literal(k)) => Some((name.clone(), *k as i32)),
        (Expr::Literal(k), Expr::Var { name, fallback: _ }) => Some((name.clone(), *k as i32)),
        _ => None,
    }
}

/// Given a member's per-key bodies, find the single `Var.name` X that
/// appears as the var operand in the largest number of bodies whose
/// shape is `Calc(Add(Var(X), Literal(K)))`. Returns `(X, count)` for
/// the best X, or `None` if no body matches the shape.
fn best_shared_var_for_self_add_literal(
    bodies: &HashMap<i64, &Expr>,
) -> Option<(String, usize)> {
    let mut by_x: HashMap<String, usize> = HashMap::new();
    for body in bodies.values() {
        if let Some((x, _)) = decompose_var_plus_literal(body) {
            *by_x.entry(x).or_insert(0) += 1;
        }
    }
    by_x.into_iter()
        // Tiebreak by lexicographic name so the choice is deterministic.
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
}

/// Classify the bulk-applier shape of a recognised loop, structurally.
///
/// See [`BulkClass`]. Two ways a loop's writes get tagged as `Copy`:
///
/// 1. The write's `val_expr` directly references one of the pointer
///    entries' `self_property` slots (e.g. a cabinet that emits
///    `var(--ptrMirror)` inline as the value).
/// 2. The write's `val_expr` is a bare `Var(name)` whose dispatch body
///    elsewhere has the indirect-read shape — captured at descriptor
///    build time as `WriteEntry.val_indirect_read`.
///
/// Pure structural shape — whole-name identity only, no substring or
/// character inspection.
fn classify_bulk(pointers: &[PointerEntry], writes: &[WriteEntry]) -> BulkClass {
    if writes.is_empty() {
        return BulkClass::ReadOnly;
    }
    let mirrors: HashSet<&str> = pointers
        .iter()
        .map(|p| p.self_property.as_str())
        .collect();
    let mut any_copy = false;
    for w in writes {
        if expr_references_any(&w.val_expr, &mirrors) {
            any_copy = true;
            continue;
        }
        if w.val_indirect_read.is_some() {
            any_copy = true;
        }
    }
    if any_copy {
        BulkClass::Copy
    } else {
        BulkClass::Fill
    }
}

/// Recognise an indirect-read intermediate on a write's value expression.
///
/// The MOVS-style shape: the value-side dispatch entry is a bare
/// `Var(name)`, and `name`'s top-level assignment body is a
/// `FunctionCall` (any opaque name) whose argument tree references —
/// somewhere — one of the loop's pointer mirror slots. That tells us the
/// per-iter source byte is a memory read keyed on a stepping pointer:
/// the canonical Copy-shape, exposed via a derived intermediate.
///
/// When the FunctionCall's first argument has the clean shape
/// `calc(var(seg) + var(ptr_mirror))` (or the reversed orientation),
/// the segment slot is captured too. Otherwise `seg_property` is `None`
/// and the runtime evaluates the address argument verbatim.
///
/// Cardinal-rule shape:
/// - Inputs are `&Expr` and a name set; this matcher does not split or
///   substring any name.
/// - The function's name is opaque — calcite does not encode any
///   "this is the read primitive" knowledge. The structural fact is:
///   "a function call keyed on a stepping pointer" — generic across
///   any cabinet that uses a function-call shape for memory access.
fn recognise_indirect_read<'a>(
    val_expr: &Expr,
    mirrors: &HashSet<&str>,
    assignment_index: &HashMap<&'a str, &'a Expr>,
) -> Option<IndirectRead> {
    // Step 1: val_expr must be a bare Var with no fallback. A val_expr
    // that's already a complex expression doesn't fit the "intermediate
    // hides the dependency" pattern this matcher is for — direct
    // references through pointer mirrors are caught by the existing
    // `expr_references_any` path in `classify_bulk`.
    let intermediate_name = match val_expr {
        Expr::Var { name, fallback: None } => name.as_str(),
        _ => return None,
    };
    // Step 2: look up the intermediate's body. If it's not a top-level
    // assignment in the cabinet, we can't trace through it.
    let body = *assignment_index.get(intermediate_name)?;
    // Step 3: the body must be a FunctionCall (any name, any arg
    // count). The function's name is opaque to the matcher; the
    // structural fact "this is a call expression" is what marks it as a
    // candidate read primitive.
    let Expr::FunctionCall { args, .. } = body else { return None };
    if args.is_empty() {
        return None;
    }
    // Step 4: somewhere in the args' expression trees there must be a
    // Var reference matching one of the loop's pointer mirrors.
    let mut pointer_property: Option<String> = None;
    for arg in args {
        if let Some(name) = first_pointer_mirror_referenced(arg, mirrors) {
            pointer_property = Some(name);
            break;
        }
    }
    let pointer_property = pointer_property?;
    // Step 5: try to extract a clean `(base, ptr)` decomposition from
    // the first argument: `calc(base + ptr)` or `calc(seg*16 + ptr)`
    // (either orientation, trailing literal offsets peeled). Otherwise
    // leave it None — the runtime can still evaluate the address
    // argument verbatim per-iter.
    let decomposed = decompose_indirect_addr(&args[0], &pointer_property);
    let (seg_property, seg_times_sixteen) = match decomposed {
        Some((name, scaled)) => (Some(name), scaled),
        None => (None, false),
    };

    Some(IndirectRead {
        seg_property,
        seg_times_sixteen,
        pointer_property,
        intermediate_property: intermediate_name.to_string(),
    })
}

/// Return the first pointer-mirror name referenced by `expr` (depth-first
/// pre-order) if any. Whole-name identity check — no substring or
/// character inspection.
fn first_pointer_mirror_referenced(expr: &Expr, mirrors: &HashSet<&str>) -> Option<String> {
    match expr {
        Expr::Var { name, fallback } => {
            if mirrors.contains(name.as_str()) {
                return Some(name.clone());
            }
            if let Some(fb) = fallback {
                return first_pointer_mirror_referenced(fb, mirrors);
            }
            None
        }
        Expr::Literal(_) | Expr::StringLiteral(_) => None,
        Expr::Calc(op) => first_pointer_mirror_in_calc(op, mirrors),
        Expr::StyleCondition { branches, fallback } => {
            for b in branches {
                if let Some(n) = first_pointer_mirror_in_test(&b.condition, mirrors) {
                    return Some(n);
                }
                if let Some(n) = first_pointer_mirror_referenced(&b.then, mirrors) {
                    return Some(n);
                }
            }
            first_pointer_mirror_referenced(fallback, mirrors)
        }
        Expr::FunctionCall { args, .. } => args
            .iter()
            .find_map(|a| first_pointer_mirror_referenced(a, mirrors)),
        Expr::Concat(parts) => parts
            .iter()
            .find_map(|p| first_pointer_mirror_referenced(p, mirrors)),
    }
}

fn first_pointer_mirror_in_calc(op: &CalcOp, mirrors: &HashSet<&str>) -> Option<String> {
    match op {
        CalcOp::Add(a, b)
        | CalcOp::Sub(a, b)
        | CalcOp::Mul(a, b)
        | CalcOp::Div(a, b)
        | CalcOp::Mod(a, b)
        | CalcOp::Pow(a, b) => first_pointer_mirror_referenced(a, mirrors)
            .or_else(|| first_pointer_mirror_referenced(b, mirrors)),
        CalcOp::Min(args) | CalcOp::Max(args) => args
            .iter()
            .find_map(|a| first_pointer_mirror_referenced(a, mirrors)),
        CalcOp::Clamp(a, b, c) => first_pointer_mirror_referenced(a, mirrors)
            .or_else(|| first_pointer_mirror_referenced(b, mirrors))
            .or_else(|| first_pointer_mirror_referenced(c, mirrors)),
        CalcOp::Round(_, a, b) => first_pointer_mirror_referenced(a, mirrors)
            .or_else(|| first_pointer_mirror_referenced(b, mirrors)),
        CalcOp::Sign(a) | CalcOp::Abs(a) | CalcOp::Negate(a) => {
            first_pointer_mirror_referenced(a, mirrors)
        }
    }
}

fn first_pointer_mirror_in_test(test: &StyleTest, mirrors: &HashSet<&str>) -> Option<String> {
    match test {
        StyleTest::Single { property, value } => {
            if mirrors.contains(property.as_str()) {
                return Some(property.clone());
            }
            first_pointer_mirror_referenced(value, mirrors)
        }
        StyleTest::And(parts) | StyleTest::Or(parts) => parts
            .iter()
            .find_map(|p| first_pointer_mirror_in_test(p, mirrors)),
    }
}

/// Try to extract a segment slot from an indirect-read function call's
/// first argument. Accepts the shape `calc(var(seg) + var(ptr))` (or
/// the reversed orientation `calc(var(ptr) + var(seg))`) where one
/// operand is the pointer mirror we already identified. The other
/// operand must be a bare `Var` whose name we capture as the segment
/// slot — its name is opaque to the matcher (the segment slot may
/// itself be an intermediate that the runtime resolves later).
///
/// Returns `None` for arg shapes the structural matcher can't simplify
/// (e.g. extra arithmetic, deep nesting). The runtime falls back to
/// evaluating the full argument expression in those cases.
fn decompose_indirect_addr(arg: &Expr, pointer_property: &str) -> Option<(String, bool)> {
    let Expr::Calc(CalcOp::Add(left, right)) = arg else { return None };
    // Peel a trailing literal byte-offset addend (`<inner> + K` /
    // `K + <inner>`): the within-iteration offset of this write's read.
    // The applier models that offset positionally (write k reads at
    // +k), mirroring the dst side, so only the base decomposition is
    // captured here.
    if matches!(right.as_ref(), Expr::Literal(_)) {
        return decompose_indirect_addr(left, pointer_property);
    }
    if matches!(left.as_ref(), Expr::Literal(_)) {
        return decompose_indirect_addr(right, pointer_property);
    }
    // `var(base) + var(ptr)` — the base slot contributes as-is.
    if let (Some(p), Some(s)) = (match_bare_var(left), match_bare_var(right)) {
        if p == pointer_property {
            return Some((s, false));
        }
        if s == pointer_property {
            return Some((p, false));
        }
    }
    // `var(seg) * 16 + var(ptr)` — the base slot contributes ×16. Same
    // shape grammar as the dst side's `match_segmented_address`.
    if let Some(p) = match_bare_var(right) {
        if p == pointer_property {
            if let Some(seg) = match_var_times_sixteen(left) {
                return Some((seg, true));
            }
        }
    }
    if let Some(p) = match_bare_var(left) {
        if p == pointer_property {
            if let Some(seg) = match_var_times_sixteen(right) {
                return Some((seg, true));
            }
        }
    }
    None
}

/// True iff `expr` transitively references any `Expr::Var { name, .. }`
/// or `StyleTest::Single { property, .. }` whose name is in `names`.
/// Whole-name equality only — no substring matching.
fn expr_references_any(expr: &Expr, names: &HashSet<&str>) -> bool {
    match expr {
        Expr::Var { name, fallback } => {
            if names.contains(name.as_str()) {
                return true;
            }
            if let Some(fb) = fallback {
                return expr_references_any(fb, names);
            }
            false
        }
        Expr::Literal(_) | Expr::StringLiteral(_) => false,
        Expr::Calc(op) => calc_references_any(op, names),
        Expr::StyleCondition { branches, fallback } => {
            for b in branches {
                if test_references_any(&b.condition, names) || expr_references_any(&b.then, names) {
                    return true;
                }
            }
            expr_references_any(fallback, names)
        }
        Expr::FunctionCall { args, .. } => {
            args.iter().any(|a| expr_references_any(a, names))
        }
        Expr::Concat(parts) => parts.iter().any(|p| expr_references_any(p, names)),
    }
}

fn calc_references_any(op: &CalcOp, names: &HashSet<&str>) -> bool {
    match op {
        CalcOp::Add(a, b)
        | CalcOp::Sub(a, b)
        | CalcOp::Mul(a, b)
        | CalcOp::Div(a, b)
        | CalcOp::Mod(a, b)
        | CalcOp::Pow(a, b) => {
            expr_references_any(a, names) || expr_references_any(b, names)
        }
        CalcOp::Min(args) | CalcOp::Max(args) => {
            args.iter().any(|a| expr_references_any(a, names))
        }
        CalcOp::Clamp(a, b, c) => {
            expr_references_any(a, names)
                || expr_references_any(b, names)
                || expr_references_any(c, names)
        }
        CalcOp::Round(_, a, b) => {
            expr_references_any(a, names) || expr_references_any(b, names)
        }
        CalcOp::Sign(a) | CalcOp::Abs(a) | CalcOp::Negate(a) => expr_references_any(a, names),
    }
}

fn test_references_any(test: &StyleTest, names: &HashSet<&str>) -> bool {
    match test {
        StyleTest::Single { property, value } => {
            names.contains(property.as_str()) || expr_references_any(value, names)
        }
        StyleTest::And(parts) | StyleTest::Or(parts) => {
            parts.iter().any(|p| test_references_any(p, names))
        }
    }
}

// ---------------------------------------------------------------------------
// Shape matchers.
// ---------------------------------------------------------------------------

/// The shape extracted from an IP-body that matches stay-or-advance.
#[derive(Debug, Clone)]
struct IpShape {
    /// The slot read by both branches.
    self_property: String,
    /// The "advance" literal — the integer offset added on the exit branch.
    advance_literal: i32,
    /// The predicate guarding "stay" vs "advance".
    predicate: StyleTest,
    /// Whether `predicate` evaluating true selects the *stay* branch.
    /// `true` when the stay body was the branch's `then`; `false` when
    /// kiln (or another emitter) inverted the shape and the stay body
    /// is the fallback. The runtime gate needs this to know which
    /// polarity of the predicate means "the loop is still running".
    predicate_means_stay: bool,
    /// The stay branch's subtrahend, when it is a bare `Var` — the
    /// cabinet's own slot for the extra per-instruction advance.
    ///
    /// Why the *subtrahend* is the extra advance: on a "stay" tick the
    /// IP slot is a fixed point — its new value equals its old value —
    /// so `self − S (+ any wrapper addend W) = IP_current`, which pins
    /// `self = IP_current + S − W`. The exit branch then produces
    /// `self + L + W = IP_current + S + L`. Any wrapper contribution W
    /// cancels; the post-loop advance over the current IP is exactly
    /// `S + L`, derivable from the two branch shapes alone.
    extra_advance_slot: Option<String>,
}

/// Match an IP-body whose shape is "stay-here-or-advance".
///
/// Two structural variants are accepted, both purely in terms of CSS
/// shape — the recogniser does not look at any property name:
///
/// 1. **Single-predicate (STOS/MOVS/LODS form).**
///    `if(<pred>: <X>; else: <Y>)` where one of `<X>` / `<Y>` is
///    `calc(self - <subtrahend>)` (the loop-stay branch) and the other
///    is `calc(self + <integer literal>)` (the loop-advance branch).
///    The two outcomes share the same `self` slot.
///
/// 2. **Disjunctive-predicate (CMPS/SCAS form).**
///    `if(<P1>: stay; <P2>: stay; ...; <Pn>: stay; else: advance)` —
///    multiple branches all yielding the same stay body, with the
///    fallback being the advance body. Or symmetrically, multiple
///    branches all advancing with the fallback staying. The synthesised
///    predicate is `Or(P1, P2, ..., Pn)`.
///
/// The predicate stored in the descriptor is the test as it appears (or
/// the synthesised disjunction), NOT normalised — phase 2 evaluates it
/// directly. If the stay branch was the `else` (i.e. kiln emitted the
/// inverted shape), phase 2 just reads the predicate and inverts its
/// outcome there; phase 1's job is only to extract the structural fact
/// that this is an IP body, not to canonicalise it.
fn match_ip_stay_or_advance(body: &Expr) -> Option<IpShape> {
    let Expr::StyleCondition { branches, fallback } = body else { return None };
    if branches.is_empty() {
        return None;
    }
    let else_ = fallback.as_ref();

    if branches.len() == 1 {
        // Single-branch form (STOS/MOVS/LODS). Try both orientations,
        // recording which one matched: predicate-true selects stay only
        // when the stay body was the `then`.
        let then = &branches[0].then;
        if let Some(s) = match_ip_orientation(then, else_, &branches[0].condition, true) {
            return Some(s);
        }
        return match_ip_orientation(else_, then, &branches[0].condition, false);
    }

    // Multi-branch form (CMPS/SCAS): all branch `then`s must be
    // structurally equal; fallback is the other side. Two orientations:
    //
    //   - All branches stay; fallback advances. Predicate = OR(branch
    //     conditions).
    //   - All branches advance; fallback stays. Predicate = OR(branch
    //     conditions), but inverted in meaning. We capture it as-is and
    //     let phase 2 choose the polarity.
    //
    // Equality is recursive structural equality on Expr (PartialEq).
    let first_then = &branches[0].then;
    if !branches.iter().all(|b| &b.then == first_then) {
        return None;
    }
    let conditions: Vec<StyleTest> = branches.iter().map(|b| b.condition.clone()).collect();
    let synth_predicate = StyleTest::Or(conditions);

    // Try (then=stay, else=advance) first, then the inverse.
    if let Some(s) = match_ip_orientation(first_then, else_, &synth_predicate, true) {
        return Some(s);
    }
    match_ip_orientation(else_, first_then, &synth_predicate, false)
}

fn match_ip_orientation(
    stay: &Expr,
    advance: &Expr,
    predicate: &StyleTest,
    predicate_means_stay: bool,
) -> Option<IpShape> {
    let (stay_self, stay_subtrahend) = match_calc_sub_var(stay)?;
    let (advance_self, advance_lit) = match_calc_add_var_lit(advance)?;
    if stay_self != advance_self {
        return None;
    }
    // Capture the stay subtrahend when it is a bare Var — see the
    // `IpShape::extra_advance_slot` doc for why the subtrahend (not any
    // outer wrapper addend) is the structurally-correct extra advance.
    // A non-Var subtrahend yields None: the applier reports Unsupported
    // rather than committing an IP it can't derive.
    let extra_advance_slot = match stay_subtrahend {
        Expr::Var { name, .. } => Some(name.clone()),
        _ => None,
    };
    Some(IpShape {
        self_property: stay_self,
        advance_literal: advance_lit,
        predicate: predicate.clone(),
        predicate_means_stay,
        extra_advance_slot,
    })
}

/// Match `calc(var(name) - <anything>)`. Returns the var name and the
/// subtrahend.
fn match_calc_sub_var(expr: &Expr) -> Option<(String, &Expr)> {
    let Expr::Calc(CalcOp::Sub(a, b)) = expr else { return None };
    let Expr::Var { name, .. } = a.as_ref() else { return None };
    Some((name.clone(), b.as_ref()))
}

/// Match `calc(var(name) + <integer literal>)`. Returns var name and the
/// integer literal value as i32.
fn match_calc_add_var_lit(expr: &Expr) -> Option<(String, i32)> {
    let Expr::Calc(CalcOp::Add(a, b)) = expr else { return None };
    let Expr::Var { name, .. } = a.as_ref() else { return None };
    let Expr::Literal(v) = b.as_ref() else { return None };
    if v.fract() != 0.0 {
        return None;
    }
    Some((name.clone(), *v as i32))
}

/// Match the counter shape:
/// `if(<pred-equiv-or-rep-guard>: self; else: max(0, calc(self - 1)))`.
/// We accept any predicate equivalent to the loop predicate's *negation*
/// being the trigger for decrement — but at this level we don't try to
/// prove logical equivalence. Two acceptable shapes:
///
/// 1. `if(<P>: self; else: max(0, self - 1))` — decrement on else.
/// 2. The exact rep-guard kiln emits, which encodes "no rep prefix → keep
///    self; else decrement". Recognised structurally as
///    `if(<some-style-cond>: var(self); else: max(0, calc(var(self) - 1)))`.
///
/// Returns the structural Counter even if the inner predicate doesn't
/// textually match `predicate`; the runtime path in phase 2 evaluates
/// each predicate independently anyway. Phase 1 only cares about the
/// shape.
fn match_counter(body: &Expr, _predicate: &StyleTest, prop: &str) -> Option<CounterEntry> {
    let Expr::StyleCondition { branches, fallback } = body else { return None };
    if branches.len() != 1 {
        return None;
    }
    let then = &branches[0].then;
    let else_ = fallback.as_ref();

    // Either (then=self, else=decrement) or (then=decrement, else=self).
    // The recogniser doesn't care which orientation the cabinet picks.
    if let Some(c) = match_counter_orientation(then, else_, prop) {
        return Some(c);
    }
    match_counter_orientation(else_, then, prop)
}

fn match_counter_orientation(
    self_branch: &Expr,
    decrement_branch: &Expr,
    prop: &str,
) -> Option<CounterEntry> {
    // self_branch = var(self)
    let Expr::Var { name: self_name, .. } = self_branch else { return None };

    // decrement_branch = max(0, calc(self - step))
    let Expr::Calc(CalcOp::Max(args)) = decrement_branch else { return None };
    if args.len() != 2 {
        return None;
    }
    let zero_ok = matches!(&args[0], Expr::Literal(v) if *v == 0.0);
    if !zero_ok {
        return None;
    }
    let Expr::Calc(CalcOp::Sub(a, b)) = &args[1] else { return None };
    let Expr::Var { name: dec_name, .. } = a.as_ref() else { return None };
    let Expr::Literal(step) = b.as_ref() else { return None };
    if step.fract() != 0.0 || *step <= 0.0 {
        return None;
    }
    if dec_name != self_name {
        return None;
    }

    Some(CounterEntry {
        property: prop.to_string(),
        self_property: self_name.clone(),
        step: *step as i32,
    })
}

/// Match the pointer shape:
///
///   `if(<gate>: <A>; else: <B>)`
///
/// where exactly ONE of `<A>` / `<B>` is `var(self)` (the guard
/// branch) and the OTHER is the update expression
/// `funcCall(calc(var(self) + k - call(var(flag), n) * 2k), 16)` —
/// kiln's `--lowerBytes(... , 16)` shape for a 16-bit modular pointer
/// step under a direction flag.
///
/// The recogniser does NOT inspect the function names — it accepts ANY
/// outer 2-arg function call whose second arg is the literal 16, AND
/// any inner 2-arg function call whose second arg is a small integer
/// (the flag bit position). The shape is what matters; the names are
/// how the cabinet exposes that shape, and a 6502 cabinet calling its
/// equivalents `--lowBytes(_, 8)` would still get caught (modulo the
/// "16" being whatever modulus the cabinet uses; we accept any literal
/// modulus ≥ 2).
fn match_pointer(body: &Expr, prop: &str) -> Option<PointerEntry> {
    let Expr::StyleCondition { branches, fallback } = body else { return None };
    if branches.len() != 1 {
        return None;
    }
    let then = &branches[0].then;
    let else_ = fallback.as_ref();

    // Exactly one of {then, else} should be a bare var(self) (the guard
    // branch), and the other should be the update expression. Try both
    // orderings and return whichever matches.
    if let Some(p) = match_pointer_with_orientation(then, else_, prop) {
        return Some(p);
    }
    match_pointer_with_orientation(else_, then, prop)
}

/// Try matching with `guard_branch` = the `var(self)` side and
/// `update_branch` = the update-expression side.
fn match_pointer_with_orientation(
    guard_branch: &Expr,
    update_branch: &Expr,
    prop: &str,
) -> Option<PointerEntry> {
    let Expr::Var { name: self_guard, .. } = guard_branch else { return None };

    // update_branch = outerCall(inner, modulus_literal) where inner is
    //   calc(var(self) + k - innerCall(var(flag), bit) * (2k))
    let Expr::FunctionCall { name: _outer_name, args: outer_args } = update_branch else {
        return None;
    };
    if outer_args.len() != 2 {
        return None;
    }
    let Expr::Literal(modulus) = &outer_args[1] else { return None };
    if modulus.fract() != 0.0 || *modulus < 2.0 {
        return None;
    }

    // inner: calc(<calc(var(self) + k)> - <innerCall(var(flag), bit)> * (2k))
    //
    // Tree shape from kiln (after parsing `calc(a + b - c * d)`):
    //   Calc::Sub(Calc::Add(Var(self), Lit(k)), Calc::Mul(Call(flag, bit), Lit(2k)))
    let Expr::Calc(CalcOp::Sub(addexpr, mulexpr)) = &outer_args[0] else { return None };

    let Expr::Calc(CalcOp::Add(self_var, k_lit)) = addexpr.as_ref() else { return None };
    let Expr::Var { name: self_name, .. } = self_var.as_ref() else { return None };
    if self_name != self_guard {
        return None;
    }
    let Expr::Literal(k) = k_lit.as_ref() else { return None };
    if k.fract() != 0.0 || *k <= 0.0 {
        return None;
    }
    let base_step = *k as i32;

    let Expr::Calc(CalcOp::Mul(callexpr, twok_lit)) = mulexpr.as_ref() else { return None };
    let Expr::Literal(twok) = twok_lit.as_ref() else { return None };
    if (*twok - 2.0 * (*k)).abs() > f64::EPSILON {
        return None;
    }
    let Expr::FunctionCall { name: _inner_name, args: inner_args } = callexpr.as_ref() else {
        return None;
    };
    if inner_args.len() != 2 {
        return None;
    }
    let Expr::Var { name: flag_name, .. } = &inner_args[0] else { return None };
    let Expr::Literal(bit) = &inner_args[1] else { return None };
    if bit.fract() != 0.0 || *bit < 0.0 || *bit > 31.0 {
        return None;
    }

    Some(PointerEntry {
        property: prop.to_string(),
        self_property: self_name.clone(),
        base_step,
        flag_property: flag_name.clone(),
        flag_bit: *bit as u32,
    })
}

#[derive(Debug)]
enum MemwriteSide {
    AddressLike,
    ValueLike,
}

/// Tentative classifier for a per-V body that didn't match counter or
/// pointer. The kiln-emitted memwrite address slots collapse to `-1`
/// when no write should fire (gated through `repGuardAddr`); value
/// slots don't have that sentinel. So if we see a `-1` literal as the
/// "stay" branch of a `StyleCondition`, treat it as address-like.
/// Everything else is value-like.
///
/// Phase 1 only tags these tentatively; phase 2 will pair them by slot
/// index based on the cabinet's memwrite-slot assignment ordering. We
/// keep the address/value bodies separate here so the descriptor still
/// captures useful structural info even before pairing.
fn classify_memwrite_side(body: &Expr) -> MemwriteSide {
    if expression_contains_neg_one_literal(body) {
        MemwriteSide::AddressLike
    } else {
        MemwriteSide::ValueLike
    }
}

/// Search `expr` for any sub-expression of shape `(var * 16) + var` or
/// `var + (var * 16)`. Returns the `(scale_var, offset_var)` pair on the
/// first match found via post-order tree walk.
///
/// The bulk applier (phase 3b runtime path) needs the segment/pointer
/// pair to step through memory iteration-by-iteration. The recogniser
/// finds it once at compile time so the applier can stay name-blind.
///
/// Cardinal-rule note: this matcher reads the literal **value** `16`
/// (the page-size constant the canonical 8086 segment shift uses), but
/// inspects no character of any property name. Any cabinet — ISA-
/// flavoured or otherwise — whose write-address has shape
/// `calc(base * 16 + offset)` decomposes identically.
fn decompose_addr_expr(expr: &Expr) -> Option<(String, String)> {
    // Try to match this node directly.
    if let Some(pair) = match_segmented_address(expr) {
        return Some(pair);
    }
    // Otherwise descend through the structures the recogniser already
    // knows how to peek inside (StyleCondition branches, Calc/FunCall
    // children). The address expression typically appears wrapped in
    // `if(active_guard: -1; else: <real_address>)`; finding the shape
    // anywhere inside that wrapper is sufficient.
    match expr {
        Expr::Calc(op) => decompose_in_calc(op),
        Expr::StyleCondition { branches, fallback } => {
            for b in branches {
                if let Some(pair) = decompose_addr_expr(&b.then) {
                    return Some(pair);
                }
            }
            decompose_addr_expr(fallback)
        }
        Expr::FunctionCall { args, .. } => {
            for a in args {
                if let Some(pair) = decompose_addr_expr(a) {
                    return Some(pair);
                }
            }
            None
        }
        Expr::Concat(parts) => {
            for p in parts {
                if let Some(pair) = decompose_addr_expr(p) {
                    return Some(pair);
                }
            }
            None
        }
        Expr::Literal(_) | Expr::StringLiteral(_) | Expr::Var { .. } => None,
    }
}

fn decompose_in_calc(op: &CalcOp) -> Option<(String, String)> {
    match op {
        CalcOp::Add(a, b) | CalcOp::Sub(a, b) | CalcOp::Mul(a, b)
        | CalcOp::Div(a, b) | CalcOp::Mod(a, b) | CalcOp::Pow(a, b) => {
            decompose_addr_expr(a).or_else(|| decompose_addr_expr(b))
        }
        CalcOp::Min(args) | CalcOp::Max(args) => {
            for a in args { if let Some(p) = decompose_addr_expr(a) { return Some(p); } }
            None
        }
        CalcOp::Clamp(a, b, c) => decompose_addr_expr(a)
            .or_else(|| decompose_addr_expr(b))
            .or_else(|| decompose_addr_expr(c)),
        CalcOp::Round(_, a, b) => decompose_addr_expr(a).or_else(|| decompose_addr_expr(b)),
        CalcOp::Sign(a) | CalcOp::Abs(a) | CalcOp::Negate(a) => decompose_addr_expr(a),
    }
}

/// Match the canonical "segment * 16 + pointer" shape at this exact
/// node. Returns `Some((seg_name, ptr_name))` when both halves are bare
/// `Expr::Var` references and the multiplication's other operand is the
/// literal 16. Recognises both operand orderings of the outer `+`.
fn match_segmented_address(expr: &Expr) -> Option<(String, String)> {
    let Expr::Calc(CalcOp::Add(left, right)) = expr else { return None };
    // Try left = (seg * 16), right = pointer.
    if let Some(seg) = match_var_times_sixteen(left) {
        if let Some(ptr) = match_bare_var(right) {
            return Some((seg, ptr));
        }
    }
    // Try right = (seg * 16), left = pointer.
    if let Some(seg) = match_var_times_sixteen(right) {
        if let Some(ptr) = match_bare_var(left) {
            return Some((seg, ptr));
        }
    }
    None
}

/// `var(--name) * 16` or `16 * var(--name)`, returning the var name.
fn match_var_times_sixteen(expr: &Expr) -> Option<String> {
    let Expr::Calc(CalcOp::Mul(a, b)) = expr else { return None };
    if let (Some(name), true) = (match_bare_var(a), match_lit_eq(b, 16.0)) {
        return Some(name);
    }
    if let (Some(name), true) = (match_bare_var(b), match_lit_eq(a, 16.0)) {
        return Some(name);
    }
    None
}

fn match_bare_var(expr: &Expr) -> Option<String> {
    if let Expr::Var { name, fallback: None } = expr {
        Some(name.clone())
    } else {
        None
    }
}

fn match_lit_eq(expr: &Expr, target: f64) -> bool {
    matches!(expr, Expr::Literal(v) if *v == target)
}

fn expression_contains_neg_one_literal(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(v) => *v == -1.0,
        Expr::Calc(op) => calc_contains_neg_one(op),
        Expr::StyleCondition { branches, fallback } => {
            for b in branches {
                if expression_contains_neg_one_literal(&b.then) {
                    return true;
                }
            }
            expression_contains_neg_one_literal(fallback)
        }
        Expr::FunctionCall { args, .. } => args.iter().any(expression_contains_neg_one_literal),
        Expr::Concat(parts) => parts.iter().any(expression_contains_neg_one_literal),
        _ => false,
    }
}

fn calc_contains_neg_one(op: &CalcOp) -> bool {
    match op {
        CalcOp::Add(a, b)
        | CalcOp::Sub(a, b)
        | CalcOp::Mul(a, b)
        | CalcOp::Div(a, b)
        | CalcOp::Mod(a, b)
        | CalcOp::Pow(a, b) => {
            expression_contains_neg_one_literal(a) || expression_contains_neg_one_literal(b)
        }
        CalcOp::Min(args) | CalcOp::Max(args) => {
            args.iter().any(expression_contains_neg_one_literal)
        }
        CalcOp::Clamp(a, b, c) => {
            expression_contains_neg_one_literal(a)
                || expression_contains_neg_one_literal(b)
                || expression_contains_neg_one_literal(c)
        }
        CalcOp::Round(_, a, b) => {
            expression_contains_neg_one_literal(a) || expression_contains_neg_one_literal(b)
        }
        CalcOp::Sign(a) | CalcOp::Abs(a) | CalcOp::Negate(a) => {
            expression_contains_neg_one_literal(a)
        }
    }
}

// ---------------------------------------------------------------------------
// Predicate helpers.
// ---------------------------------------------------------------------------

fn collect_property_names_from_predicate(test: &StyleTest) -> Vec<String> {
    let mut out = Vec::new();
    walk_test(test, &mut |t| {
        if let StyleTest::Single { property, .. } = t {
            if !out.contains(property) {
                out.push(property.clone());
            }
        }
    });
    out
}

fn predicate_mentions_flag_bit(test: &StyleTest) -> bool {
    // Heuristic: if the predicate has more than one Single test on
    // distinct properties, AND-combined, treat it as flag-conditioned.
    // The exact "what flag, what bit" extraction lives in phase 2; phase
    // 1 only flags the descriptor as needing flag-aware semantics.
    let mut props: HashSet<String> = HashSet::new();
    walk_test(test, &mut |t| {
        if let StyleTest::Single { property, .. } = t {
            props.insert(property.clone());
        }
    });
    props.len() > 1
}

fn walk_test<F: FnMut(&StyleTest)>(test: &StyleTest, f: &mut F) {
    f(test);
    match test {
        StyleTest::Single { .. } => {}
        StyleTest::And(parts) | StyleTest::Or(parts) => {
            for p in parts {
                walk_test(p, f);
            }
        }
    }
}

#[cfg(test)]
mod tests;
