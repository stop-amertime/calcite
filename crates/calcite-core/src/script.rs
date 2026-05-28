//! Generic script-primitive layer: `WatchRegistry`, watches, predicates, and
//! actions. Hosts (calcite-cli, calcite-wasm, embedders) register watches on
//! a registry; per-tick (or at advancement boundaries) they call
//! [`WatchRegistry::poll`] which evaluates the watches against the current
//! [`State`] and pushes [`MeasurementEvent`]s into a buffer the host drains.
//!
//! ## Primitives
//!
//! * [`WatchKind::Stride`] — fires every N ticks (cheap; elapsed-since-last-fire).
//! * [`WatchKind::Burst`] — fires on K consecutive ticks every N (cheap).
//! * [`WatchKind::At`] — fires exactly once at a target tick (cheap;
//!   replaces the old `--script-event` scheduling).
//! * [`WatchKind::Edge`] — fires on a byte-value transition at a memory
//!   address (touches state.read_mem; expensive; gate it).
//! * [`WatchKind::Cond`] — fires when a memory predicate holds (touches
//!   state.read_mem; expensive; gate it).
//! * [`WatchKind::Halt`] — fires when byte at addr is non-zero AND
//!   requests the host stop the run (cheap).
//!
//! ## Actions
//!
//! Each watch carries one or more [`Action`]s. When the watch fires they
//! are applied in declaration order. Available actions:
//!
//! * [`Action::Emit`] — push a [`MeasurementEvent`] into the registry's
//!   buffer. The event records the tick, watch name, and any sampled
//!   state-var values listed in `WatchSpec::sample_vars`.
//! * [`Action::DumpMemRange`] — read `len` bytes of guest memory starting
//!   at `addr` into the next emitted event (or write to a path; both are
//!   supported, see [`DumpSink`]).
//! * [`Action::Snapshot`] — produce a state snapshot (in-memory or to a
//!   path templated with `{tick}` / `{name}`).
//! * [`Action::SetVar`] — set a named state variable (the generic
//!   primitive that CSS-DOS-side profiles compose on for keyboard
//!   injection — `SetVar { "keyboard", scancode }` etc.).
//! * [`Action::Halt`] — request the host stop the run after this watch
//!   completes its other actions.
//!
//! ## Cheap-vs-expensive gating
//!
//! Some primitives ([`WatchKind::Cond`], [`WatchKind::Edge`]) read guest
//! memory and are too expensive to evaluate per-tick on hot loops. Each
//! watch may name a `gate` (the name of another registered watch); the
//! gated watch is only evaluated on ticks where its gate fires. The
//! typical setup is:
//!
//! ```text
//! --watch poll:stride:every=50000
//! --watch reached_ingame:cond:0x3a3c4=0:gate=poll:then=emit,halt
//! ```
//!
//! The expensive `Cond` watch only runs every 50K ticks, gated by the
//! cheap `Stride`. Watches without a `gate` are evaluated every tick the
//! registry is polled — fine for `Stride` (cheap) but expensive for
//! `Cond` (avoid).
//!
//! ## Genericity
//!
//! No primitive in this module knows about x86, BIOS, video memory, DOS,
//! or any specific cabinet. The same registry is useful for testing a
//! sieve cabinet, a clock cabinet, or any other CSS program. Upstream
//! knowledge (the address of `_g_gamestate` in DOOM, that text VRAM
//! lives at 0xB8000 with stride 2, etc.) lives in the profiles that
//! *register* watches, not in calcite-core.

use crate::state::State;
use std::cell::Cell;

/// A single registered watch: a firing condition plus the actions that
/// run when it fires.
#[derive(Debug)]
pub struct WatchSpec {
    /// Identifier emitted with every event this watch produces.
    pub name: String,
    /// What kind of firing condition this watch represents.
    pub kind: WatchKind,
    /// Name of another watch that gates this one (the gated watch is
    /// only evaluated on ticks where the named gate watch fires). Useful
    /// for keeping expensive `Cond` / `Edge` evaluations off the hot
    /// path. `None` means "evaluate every poll".
    pub gate: Option<String>,
    /// Actions to run, in order, when this watch fires.
    pub actions: Vec<Action>,
    /// State-variable names whose current value is recorded in
    /// [`MeasurementEvent::sampled_vars`] when this watch emits. Empty by
    /// default. Useful for "regs"-style snapshots (give it the names of
    /// registers / cycle counters / tick counters etc.).
    pub sample_vars: Vec<String>,
}

/// What kind of firing condition a watch represents. See module-level
/// docs for the cheap/expensive split.
#[derive(Debug)]
pub enum WatchKind {
    /// Fire every `every` ticks. Cheap. Uses `last_fired_at` to track
    /// the last tick we fired on so we don't depend on the host calling
    /// poll() at tick values divisible by `every` — fires whenever
    /// `tick - last_fired_at >= every`. The CLI naturally aligns its
    /// poll cadence to stride boundaries; the wasm bridge polls on
    /// adaptive batch boundaries that may not align.
    Stride {
        every: u32,
        /// Internal: tick of the most recent fire. Initialised to 0 so
        /// the first poll past `every` fires regardless of starting
        /// tick value.
        last_fired_at: Cell<u32>,
    },
    /// Fire on `count` consecutive ticks every `every` ticks. Cheap.
    Burst { every: u32, count: u32 },
    /// Fire exactly once at tick `tick`. Cheap. Replaces the old
    /// `--script-event TICK:KIND[:ARGS]` scheduling: a host that wants
    /// "press Enter at tick 2_000_000" registers an `At { tick }` watch
    /// with `[SetVar { "keyboard", 0x1c0d }]` actions.
    At { tick: u32 },
    /// Fire whenever the byte at `addr` transitions to a different
    /// value than the last poll. Expensive — memory read on every gate
    /// fire — gate it with a stride watch unless polling a tiny set.
    /// `last` and `primed` are interior-mutable so registry methods
    /// don't need `&mut self` for predicate evaluation.
    Edge {
        addr: i32,
        /// Internal: byte value seen on the previous fire. Initialised
        /// on the first poll.
        last: Cell<u8>,
        primed: Cell<bool>,
    },
    /// Fire when all `tests` hold simultaneously. Expensive — gate it.
    /// If `repeat` is false, fires once and is then disabled. If `true`,
    /// fires on every gated poll where the predicate is true (sustain
    /// mode) — useful for actions that need to keep firing while a
    /// state holds, e.g. `setvar_pulse=keyboard,...` to spam a key
    /// while the title screen is up.
    Cond {
        tests: Vec<Predicate>,
        repeat: bool,
        /// Internal: whether this watch already fired (used when
        /// `repeat=false`).
        fired: Cell<bool>,
        /// Internal: was the predicate true on the last evaluation?
        /// Reserved for future fall-detection; not currently consulted
        /// by the eval path.
        last_held: Cell<bool>,
    },
    /// Fire when byte at `addr` is non-zero. Implicitly requests halt
    /// regardless of declared actions — this is the idiomatic "stop the
    /// run when the program writes a halt code" primitive. Cheap.
    Halt { addr: i32 },
}

/// A single boolean predicate over guest memory. Combines via AND inside
/// a [`WatchKind::Cond`].
#[derive(Debug, Clone)]
pub enum Predicate {
    /// Byte at linear `addr` equals `val`.
    ByteEq { addr: i32, val: u8 },
    /// Byte at linear `addr` does not equal `val`.
    ByteNe { addr: i32, val: u8 },
    /// `needle` appears (in order) in guest memory starting at `base`,
    /// with consecutive needle bytes located at `base`, `base + stride`,
    /// `base + 2*stride`, …, scanned across `[base, base + max_window)`.
    /// Matches anywhere inside the window.
    ///
    /// Generic primitive for "is text X visible somewhere in this
    /// memory region with this stride" — replaces the old
    /// `vram_text:NEEDLE` helper, which hard-coded text-VRAM at 0xB8000
    /// with stride 2. CSS-DOS-side profiles compose this primitive
    /// with the appropriate `(base, stride, max_window)` for the
    /// upstream layer's video shape; calcite stays generic.
    BytePatternAt {
        base: i32,
        stride: u32,
        max_window: u32,
        needle: Vec<u8>,
    },
}

/// One side-effect to perform when a watch fires.
#[derive(Debug, Clone)]
pub enum Action {
    /// Push a [`MeasurementEvent`] into the registry's event buffer.
    /// The event captures the tick, watch name, and any
    /// `sample_vars` values.
    Emit,
    /// Read `len` bytes of guest memory starting at `addr`. The
    /// result is delivered via the configured [`DumpSink`]: by
    /// default, attached to the next emitted event under
    /// [`MeasurementEvent::dumps`]; alternatively, written to a file
    /// path (the host opts into the file sink at registry-construction
    /// time on native, and gets the in-memory sink on wasm).
    DumpMemRange {
        addr: i32,
        len: u32,
        /// Path template for file sinks. `{tick}` and `{name}` are
        /// substituted. Ignored for in-memory sinks.
        path_template: String,
    },
    /// Snapshot the current state. Same template substitution as
    /// `DumpMemRange`.
    Snapshot { path_template: String },
    /// Write `value` to the named state variable (e.g. `keyboard`).
    /// Generic; upstream knowledge about WHICH variable to set lives in
    /// the host that registers the watch.
    SetVar { name: String, value: i32 },
    /// Set state variable `name` to `value` immediately, then schedule
    /// a release (set to 0) after `hold_ticks` ticks have passed.
    /// Generic edge-pair primitive; CSS-DOS-side profiles use it for
    /// keyboard taps (`SetVarPulse { "keyboard", 0x1c0d, 50_000 }`)
    /// where the cabinet's keyboard handler needs a make-then-break
    /// edge to register a press. The release is dispatched by
    /// [`WatchRegistry::poll`] when its `tick` argument reaches the
    /// scheduled release-tick.
    SetVarPulse { name: String, value: i32, hold_ticks: u32 },
    /// Set a pseudo-class match edge active immediately, then release
    /// (set inactive) after `hold_ticks` ticks. Generic edge-pair
    /// primitive; CSS-DOS-side profiles use it for keyboard taps that
    /// drive the cabinet's `&:has(#SEL:PSEUDO) { --PROP: V }` rules
    /// through calcite's input-edge recogniser. Mirror of `SetVarPulse`
    /// but on the pseudo-class surface.
    PseudoActivePulse {
        pseudo: String,
        selector: String,
        hold_ticks: u32,
    },
    /// Request the host stop the run. Effective from
    /// [`WatchRegistry::halt_requested`] after this watch's other
    /// actions have run.
    Halt,
}

/// Where bulk dump payloads (`DumpMemRange`, `Snapshot`) go on this
/// platform. Native CLIs use `File`; wasm uses `Memory` because
/// filesystem isn't available.
#[derive(Debug, Clone, Copy)]
pub enum DumpSink {
    /// Attach payload bytes directly to the emitted event. Caller
    /// drains via [`WatchRegistry::drain_events`].
    Memory,
    /// Write payload bytes to a path computed from the action's
    /// `path_template`. The path is also recorded on the event so
    /// downstream tooling can correlate.
    File,
}

/// A single payload chunk attached to an emitted event.
#[derive(Debug, Clone)]
pub struct DumpedRange {
    /// Source address of the range (or 0 for snapshots).
    pub addr: i32,
    /// Bytes (only populated when sink == Memory).
    pub bytes: Vec<u8>,
    /// Path written to (only populated when sink == File).
    pub path: Option<String>,
    /// Tag: "mem" for `DumpMemRange`, "snapshot" for `Snapshot`.
    pub tag: &'static str,
}

/// One measurement event produced by a fired [`Action::Emit`] (or
/// recorded automatically when an [`Action::DumpMemRange`] /
/// [`Action::Snapshot`] runs without an explicit `Emit`).
#[derive(Debug, Clone)]
pub struct MeasurementEvent {
    /// The tick at which the watch fired.
    pub tick: u32,
    /// `WatchSpec::name` of the watch that fired.
    pub watch_name: String,
    /// Values of state vars listed in `WatchSpec::sample_vars`, in the
    /// same order. Names that don't resolve to a state var record `0`.
    pub sampled_vars: Vec<(String, i32)>,
    /// Memory or snapshot payloads attached to this event (when the
    /// dump sink is Memory).
    pub dumps: Vec<DumpedRange>,
    /// Set when the watch's actions include a `Halt`, or this is a
    /// `WatchKind::Halt` firing.
    pub halted: bool,
}

/// Registry of watches and produced events. One instance per run; reset
/// per fresh execution.
#[derive(Debug)]
pub struct WatchRegistry {
    pub(crate) watches: Vec<WatchSpec>,
    /// Per-watch fire-count. Used by `Burst` to know which of the
    /// `count` consecutive ticks we're on, and so the host can report.
    pub(crate) fire_counts: Vec<u32>,
    /// Per-watch "this watch fired in the most recent poll() call".
    /// Used by gates: a watch with `gate=foo` only evaluates on ticks
    /// where the `foo` watch's `gate_fired[foo_idx]` is true after the
    /// pre-pass.
    pub(crate) gate_fired: Vec<bool>,
    /// Output buffer; events accumulate here until `drain_events()`.
    pub(crate) events: Vec<MeasurementEvent>,
    /// Set when any fired watch has a `Halt` action or is a
    /// `WatchKind::Halt`.
    pub(crate) halt_requested: bool,
    /// Where dump bytes go on this build (Memory on wasm, configurable
    /// on native).
    pub(crate) dump_sink: DumpSink,
    /// Pending `SetVarPulse` releases. Each entry is a (release_tick,
    /// var_name) pair scheduled when the pulse fired. `poll` checks at
    /// the top of every call and releases (writes 0 to the named var)
    /// any whose release_tick has passed. Sorted on insertion so the
    /// front entries are always the soonest-due.
    pub(crate) pending_releases: Vec<PendingRelease>,
    /// Surfaces that had releases dispatched at the start of THIS
    /// poll. Pulse actions that target one of these skip — gives the
    /// engine a full inter-poll batch with the surface in its
    /// "released" state between pulses, the break edge edge-detectors
    /// need.
    pub(crate) released_this_poll: Vec<PendingReleaseKind>,
}

/// A pulsed release scheduled for a future tick. `kind` carries which
/// surface the make-edge wrote to, so `poll` knows whether to set the
/// state var to 0 or flip a pseudo-class edge to inactive.
#[derive(Debug, Clone)]
pub(crate) struct PendingRelease {
    pub release_tick: u32,
    pub kind: PendingReleaseKind,
}

#[derive(Debug, Clone)]
pub(crate) enum PendingReleaseKind {
    /// `SetVarPulse` release — set the named state var to 0.
    Var { name: String },
    /// `PseudoActivePulse` release — set the (pseudo, selector) match
    /// edge inactive on `state.pseudo_active`.
    Pseudo { pseudo: String, selector: String },
}

impl PendingReleaseKind {
    /// True if `self` and `other` target the same surface.
    pub(crate) fn same_target(&self, other: &PendingReleaseKind) -> bool {
        match (self, other) {
            (Self::Var { name: a }, Self::Var { name: b }) => a == b,
            (
                Self::Pseudo { pseudo: ap, selector: as_ },
                Self::Pseudo { pseudo: bp, selector: bs },
            ) => ap == bp && as_ == bs,
            _ => false,
        }
    }
}

impl WatchRegistry {
    /// New empty registry. Defaults to `DumpSink::Memory` so embedders
    /// (wasm callers, tests) get inline payloads they can read back via
    /// `drain_events`. Native CLIs should call `set_dump_sink(File)`.
    pub fn new() -> Self {
        Self {
            watches: Vec::new(),
            fire_counts: Vec::new(),
            gate_fired: Vec::new(),
            events: Vec::new(),
            halt_requested: false,
            dump_sink: DumpSink::Memory,
            pending_releases: Vec::new(),
            released_this_poll: Vec::new(),
        }
    }

    /// Choose where bulk dump payloads go.
    pub fn set_dump_sink(&mut self, sink: DumpSink) {
        self.dump_sink = sink;
    }

    /// Register a watch. Returns the index assigned to it.
    pub fn register(&mut self, spec: WatchSpec) -> usize {
        let idx = self.watches.len();
        self.watches.push(spec);
        self.fire_counts.push(0);
        self.gate_fired.push(false);
        idx
    }

    /// Consume the buffered events. After this call the buffer is
    /// empty. Halt-state is preserved.
    pub fn drain_events(&mut self) -> Vec<MeasurementEvent> {
        std::mem::take(&mut self.events)
    }

    /// Has any fired watch requested a halt?
    pub fn halt_requested(&self) -> bool {
        self.halt_requested
    }

    /// Number of registered watches.
    pub fn len(&self) -> usize {
        self.watches.len()
    }

    /// Is the registry empty?
    pub fn is_empty(&self) -> bool {
        self.watches.is_empty()
    }

    /// Iterate over watch names in registration order. Useful for the
    /// host to validate gate references at parse time.
    pub fn watch_names(&self) -> impl Iterator<Item = &str> {
        self.watches.iter().map(|w| w.name.as_str())
    }

    /// Resolve a gate name to its index. `O(n)` but only run once at
    /// register-time / for diagnostic.
    pub fn name_index(&self, name: &str) -> Option<usize> {
        self.watches.iter().position(|w| w.name == name)
    }
}

impl Default for WatchRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// `poll` and the action/predicate machinery live in `script_eval.rs` so
// this file stays a clean surface declaration.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_smoke() {
        let mut r = WatchRegistry::new();
        let idx = r.register(WatchSpec {
            name: "tick".to_string(),
            kind: WatchKind::Stride { every: 100, last_fired_at: Cell::new(0) },
            gate: None,
            actions: vec![Action::Emit],
            sample_vars: Vec::new(),
        });
        assert_eq!(idx, 0);
        assert_eq!(r.len(), 1);
        assert!(!r.halt_requested());
        assert_eq!(r.name_index("tick"), Some(0));
        assert_eq!(r.name_index("missing"), None);
    }
}
