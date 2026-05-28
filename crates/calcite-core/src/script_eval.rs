//! Evaluator for the [`crate::script`] watch layer.
//!
//! The host calls [`poll`] every tick (or at known advancement
//! boundaries) with the registry, the state, and the current tick.
//! `poll` evaluates watches, runs their actions, and pushes events into
//! the registry's buffer.

use crate::script::*;
use crate::state::State;

/// Evaluate every watch in the registry against the current tick and
/// state. Mutates the registry (event buffer, internal counters) and
/// the state (`SetVar` actions). Reads guest memory via `state.read_mem`.
///
/// Cost model:
/// * `Stride`/`Burst`/`At`/`Halt` are cheap: a small integer compare and
///   no memory reads.
/// * `Edge` reads exactly one byte per evaluation.
/// * `Cond` reads up to `tests.len()` bytes per evaluation, plus a
///   linear scan of `max_window / stride` bytes per `BytePatternAt`
///   test it contains.
///
/// Watches with `gate = Some(name)` are only evaluated on ticks where
/// the named gate watch has fired this poll. Two-phase: first pass
/// evaluates ungated watches (recording their fire status), second
/// pass evaluates gated ones whose gate fired.
pub fn poll(reg: &mut WatchRegistry, state: &mut State, tick: u32) {
    // Pending `SetVarPulse` releases: any whose release_tick has passed
    // get applied (var → 0) before we re-evaluate watches. Vars that
    // released THIS poll go into `released_this_poll` so subsequent
    // pulses on the same var skip — that gives the engine a full
    // inter-poll batch where the var sits at 0 (the break edge a
    // cabinet's edge detector needs to register a key-up).
    reg.released_this_poll.clear();
    if !reg.pending_releases.is_empty() {
        let mut keep: Vec<PendingRelease> = Vec::with_capacity(reg.pending_releases.len());
        for pr in std::mem::take(&mut reg.pending_releases) {
            if tick >= pr.release_tick {
                match &pr.kind {
                    PendingReleaseKind::Var { name } => {
                        state.set_var(name, 0);
                    }
                    PendingReleaseKind::Pseudo { pseudo, selector } => {
                        state.set_pseudo_class_active(pseudo, selector, false);
                    }
                }
                reg.released_this_poll.push(pr.kind);
            } else {
                keep.push(pr);
            }
        }
        reg.pending_releases = keep;
    }

    if reg.is_empty() {
        return;
    }

    // Phase 1: evaluate ungated watches; record gate-fire status.
    let mut fired: Vec<bool> = vec![false; reg.len()];
    for i in 0..reg.len() {
        if reg.watches[i].gate.is_none() {
            fired[i] = evaluate_watch(reg, state, i, tick);
        }
    }

    // Phase 2: evaluate gated watches whose gate fired in phase 1.
    for i in 0..reg.len() {
        let Some(gate_name) = reg.watches[i].gate.clone() else {
            continue;
        };
        let Some(g_idx) = reg.name_index(&gate_name) else {
            // Unknown gate: silently skip. The CLI validates gates at
            // parse time so an unknown gate is a misuse the test catches.
            continue;
        };
        if fired[g_idx] {
            fired[i] = evaluate_watch(reg, state, i, tick);
        }
    }

    // Update gate_fired snapshot for any future external use.
    for (i, &f) in fired.iter().enumerate() {
        reg.gate_fired[i] = f;
    }
}

/// Evaluate a single watch. Returns whether it fired this poll.
fn evaluate_watch(
    reg: &mut WatchRegistry,
    state: &mut State,
    idx: usize,
    tick: u32,
) -> bool {
    let fires = match &reg.watches[idx].kind {
        WatchKind::Stride { every, last_fired_at } => {
            let every = *every;
            if every == 0 || tick == 0 {
                false
            } else {
                // Fire if at least `every` ticks have elapsed since
                // last fire. This is host-call-cadence independent;
                // the previous `tick % every == 0` check required the
                // host to poll at tick values that align with `every`,
                // which doesn't hold when the bridge polls on adaptive
                // batch boundaries instead of fixed strides.
                let last = last_fired_at.get();
                if tick.saturating_sub(last) >= every {
                    last_fired_at.set(tick);
                    true
                } else {
                    false
                }
            }
        }
        WatchKind::Burst { every, count } => {
            let every = *every;
            let count = *count;
            if every == 0 || count == 0 {
                false
            } else {
                let phase = tick % every;
                phase < count
            }
        }
        WatchKind::At { tick: t } => *t == tick,
        WatchKind::Edge { addr, last, primed } => {
            let addr = *addr;
            let cur = (state.read_mem(addr) & 0xFF) as u8;
            if !primed.get() {
                last.set(cur);
                primed.set(true);
                false
            } else if cur != last.get() {
                last.set(cur);
                true
            } else {
                false
            }
        }
        WatchKind::Cond {
            tests,
            repeat,
            fired,
            last_held,
        } => {
            if !*repeat && fired.get() {
                false
            } else {
                let holds = tests.iter().all(|t| predicate_holds(t, state));
                if holds {
                    if *repeat {
                        // Sustain mode: fire on every gated poll while
                        // the predicate holds. Re-arms after a fall.
                        // (Rising-edge-only mode would be a separate
                        // variant; the documented behaviour is the
                        // sustain shape that matches the upstream
                        // `:then=spam` use case the new primitives
                        // replace.)
                        last_held.set(true);
                        true
                    } else {
                        fired.set(true);
                        true
                    }
                } else {
                    last_held.set(false);
                    false
                }
            }
        }
        WatchKind::Halt { addr } => {
            let v = state.read_mem(*addr) & 0xFF;
            v != 0
        }
    };

    if !fires {
        return false;
    }

    reg.fire_counts[idx] += 1;

    // Capture context the actions need (immutably) so we can release
    // the borrow before mutating state.
    let watch_name = reg.watches[idx].name.clone();
    let sample_vars = reg.watches[idx].sample_vars.clone();
    let actions = reg.watches[idx].actions.clone();
    let dump_sink = reg.dump_sink;
    // `WatchKind::Halt` always implies halt request.
    let mut implicit_halt = matches!(reg.watches[idx].kind, WatchKind::Halt { .. });

    // Run actions; gather event payloads.
    let mut event = MeasurementEvent {
        tick,
        watch_name: watch_name.clone(),
        sampled_vars: Vec::new(),
        dumps: Vec::new(),
        halted: false,
    };
    for name in &sample_vars {
        let v = state.get_var(name).unwrap_or(0);
        event.sampled_vars.push((name.clone(), v));
    }

    let mut wants_emit = false;
    // SetVarPulse releases scheduled by this watch. Pushed onto the
    // registry after the action loop so we don't tangle borrow-of-actions
    // with mutable-borrow-of-registry.
    let mut pending_pulse_releases: Vec<PendingRelease> = Vec::new();
    for action in &actions {
        match action {
            Action::Emit => {
                wants_emit = true;
            }
            Action::DumpMemRange {
                addr,
                len,
                path_template,
            } => {
                let mut buf = vec![0u8; *len as usize];
                for (i, slot) in buf.iter_mut().enumerate() {
                    *slot = (state.read_mem(*addr + i as i32) & 0xFF) as u8;
                }
                let path = match dump_sink {
                    DumpSink::File => Some(render_template(
                        path_template,
                        tick,
                        &watch_name,
                    )),
                    DumpSink::Memory => None,
                };
                if let Some(p) = path.as_deref() {
                    if let Err(e) = std::fs::write(p, &buf) {
                        log::warn!("watch {watch_name}: dump-mem write {p}: {e}");
                    }
                }
                let bytes = match dump_sink {
                    DumpSink::Memory => buf,
                    DumpSink::File => Vec::new(),
                };
                event.dumps.push(DumpedRange {
                    addr: *addr,
                    bytes,
                    path,
                    tag: "mem",
                });
                wants_emit = true; // dumps imply an event for correlation
            }
            Action::Snapshot { path_template } => {
                let blob = state.snapshot();
                let path = match dump_sink {
                    DumpSink::File => Some(render_template(
                        path_template,
                        tick,
                        &watch_name,
                    )),
                    DumpSink::Memory => None,
                };
                if let Some(p) = path.as_deref() {
                    if let Err(e) = std::fs::write(p, &blob) {
                        log::warn!("watch {watch_name}: snapshot write {p}: {e}");
                    }
                }
                let bytes = match dump_sink {
                    DumpSink::Memory => blob,
                    DumpSink::File => Vec::new(),
                };
                event.dumps.push(DumpedRange {
                    addr: 0,
                    bytes,
                    path,
                    tag: "snapshot",
                });
                wants_emit = true;
            }
            Action::SetVar { name, value } => {
                state.set_var(name, *value);
            }
            Action::SetVarPulse { name, value, hold_ticks } => {
                // Skip if (a) this var has a pending release queued
                // for a future tick (a previous pulse is still mid-
                // hold) or (b) this var was JUST released at the top
                // of this poll. Both prevent a sustain-cond + pulse
                // race that would overwrite the break edge before the
                // engine ever ticks with the var at 0. The engine
                // sees: pulse → batch ticks → release → batch ticks
                // → pulse → ... — a clean make/break/make/break
                // cadence at 2× the gating poll-stride.
                let target = PendingReleaseKind::Var { name: name.clone() };
                let already_pending = reg.pending_releases
                    .iter()
                    .any(|pr| pr.kind.same_target(&target))
                    || pending_pulse_releases
                        .iter()
                        .any(|pr| pr.kind.same_target(&target));
                let just_released = reg.released_this_poll
                    .iter()
                    .any(|k| k.same_target(&target));
                if !already_pending && !just_released {
                    // Make edge: write the value now.
                    state.set_var(name, *value);
                    pending_pulse_releases.push(PendingRelease {
                        release_tick: tick.saturating_add(*hold_ticks),
                        kind: target,
                    });
                }
            }
            Action::PseudoActivePulse { pseudo, selector, hold_ticks } => {
                // Same skip-if-pending shape as SetVarPulse but on the
                // pseudo-class surface. `set_pseudo_class_active` flips
                // the host-state edge; calcite's input-edge recogniser
                // converts that into the cabinet's CSS-declared property
                // value at the next `apply_input_edges` (pre-tick).
                let target = PendingReleaseKind::Pseudo {
                    pseudo: pseudo.clone(),
                    selector: selector.clone(),
                };
                let already_pending = reg.pending_releases
                    .iter()
                    .any(|pr| pr.kind.same_target(&target))
                    || pending_pulse_releases
                        .iter()
                        .any(|pr| pr.kind.same_target(&target));
                let just_released = reg.released_this_poll
                    .iter()
                    .any(|k| k.same_target(&target));
                if !already_pending && !just_released {
                    state.set_pseudo_class_active(pseudo, selector, true);
                    pending_pulse_releases.push(PendingRelease {
                        release_tick: tick.saturating_add(*hold_ticks),
                        kind: target,
                    });
                }
            }
            Action::Halt => {
                implicit_halt = true;
            }
        }
    }

    if implicit_halt {
        reg.halt_requested = true;
        event.halted = true;
        wants_emit = true;
    }

    if wants_emit {
        reg.events.push(event);
    }

    // Append SetVarPulse releases scheduled by this watch. The
    // skip-if-pending check inside the action loop guarantees we never
    // queue two entries for the same var simultaneously.
    reg.pending_releases.extend(pending_pulse_releases);

    true
}

fn predicate_holds(p: &Predicate, state: &State) -> bool {
    match p {
        Predicate::ByteEq { addr, val } => {
            (state.read_mem(*addr) & 0xFF) as u8 == *val
        }
        Predicate::ByteNe { addr, val } => {
            (state.read_mem(*addr) & 0xFF) as u8 != *val
        }
        Predicate::BytePatternAt {
            base,
            stride,
            max_window,
            needle,
        } => {
            if needle.is_empty() {
                return true;
            }
            if *stride == 0 {
                return false;
            }
            let n = needle.len();
            let stride = *stride as i32;
            let max_window = *max_window as i32;
            // Number of "slots" of stride within the window.
            let slots = max_window / stride;
            let n_i = n as i32;
            if slots < n_i {
                return false;
            }
            // Try each starting slot.
            for s in 0..=(slots - n_i) {
                let mut ok = true;
                for j in 0..n_i {
                    let addr = *base + (s + j) * stride;
                    let b = (state.read_mem(addr) & 0xFF) as u8;
                    if b != needle[j as usize] {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    return true;
                }
            }
            false
        }
    }
}

/// Substitute `{tick}` and `{name}` placeholders in a path template.
fn render_template(template: &str, tick: u32, name: &str) -> String {
    template
        .replace("{tick}", &tick.to_string())
        .replace("{name}", name)
}

impl WatchRegistry {
    /// Number of times a watch has fired, by index. Mostly used by
    /// host diagnostics.
    pub fn fire_count(&self, idx: usize) -> u32 {
        self.fire_counts[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// Tiny no-op State for predicate/scheduling tests that don't
    /// exercise memory.
    fn empty_state() -> State {
        State::default()
    }

    #[test]
    fn stride_fires_every_n() {
        let mut reg = WatchRegistry::new();
        reg.register(WatchSpec {
            name: "tick100".to_string(),
            kind: WatchKind::Stride { every: 100, last_fired_at: std::cell::Cell::new(0) },
            gate: None,
            actions: vec![Action::Emit],
            sample_vars: Vec::new(),
        });
        let mut state = empty_state();
        let mut count = 0;
        for tick in 0..=1000 {
            poll(&mut reg, &mut state, tick);
            count += reg.drain_events().len();
        }
        // ticks 100, 200, ..., 1000 = 10 fires (tick=0 excluded).
        assert_eq!(count, 10);
    }

    #[test]
    fn stride_fires_on_unaligned_poll_cadence() {
        // Regression: the wasm bridge calls poll() at adaptive batch
        // boundaries that are NOT multiples of `every`. A `tick % every
        // == 0` check would never fire here; elapsed-since-last-fire
        // must. Poll cadence below is deliberately coprime-ish to 100.
        let mut reg = WatchRegistry::new();
        reg.register(WatchSpec {
            name: "tick100".to_string(),
            kind: WatchKind::Stride { every: 100, last_fired_at: std::cell::Cell::new(0) },
            gate: None,
            actions: vec![Action::Emit],
            sample_vars: Vec::new(),
        });
        let mut state = empty_state();
        let mut count = 0;
        // Poll at 137, 274, 411, ... — never a multiple of 100.
        let mut tick = 0u32;
        while tick <= 10_000 {
            tick += 137;
            poll(&mut reg, &mut state, tick);
            count += reg.drain_events().len();
        }
        // 10_000 / 100 = 100 stride windows; with a 137-tick poll step
        // the watch fires once per poll once a full `every` has elapsed
        // since the last fire. It must fire substantially (not zero,
        // which is the bug) — at least once per ~137-tick window past
        // the first, i.e. ~70+.
        assert!(count >= 70, "stride starved on unaligned poll: only {count} fires");
    }

    #[test]
    fn burst_fires_count_consecutive() {
        let mut reg = WatchRegistry::new();
        reg.register(WatchSpec {
            name: "burst".to_string(),
            kind: WatchKind::Burst { every: 100, count: 3 },
            gate: None,
            actions: vec![Action::Emit],
            sample_vars: Vec::new(),
        });
        let mut state = empty_state();
        let mut count = 0;
        for tick in 0..1000 {
            poll(&mut reg, &mut state, tick);
            count += reg.drain_events().len();
        }
        // Phase < 3 of every 100: ticks 0,1,2,100,101,102,...,900,901,902 = 10 * 3 = 30.
        assert_eq!(count, 30);
    }

    #[test]
    fn at_fires_exactly_once() {
        let mut reg = WatchRegistry::new();
        reg.register(WatchSpec {
            name: "scheduled".to_string(),
            kind: WatchKind::At { tick: 500 },
            gate: None,
            actions: vec![Action::Emit],
            sample_vars: Vec::new(),
        });
        let mut state = empty_state();
        let mut events = Vec::new();
        for tick in 0..1000 {
            poll(&mut reg, &mut state, tick);
            events.extend(reg.drain_events());
        }
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].tick, 500);
    }

    #[test]
    fn pseudo_pulse_make_then_release() {
        // Schedule a pseudo_pulse at tick 100 with hold=50. Verify
        // (a) the (active, kb-1) edge flips on at tick 100,
        // (b) it stays on through tick 149,
        // (c) it flips off at tick ≥ 150.
        let mut reg = WatchRegistry::new();
        reg.register(WatchSpec {
            name: "tap".to_string(),
            kind: WatchKind::At { tick: 100 },
            gate: None,
            actions: vec![Action::PseudoActivePulse {
                pseudo: "active".to_string(),
                selector: "kb-1".to_string(),
                hold_ticks: 50,
            }],
            sample_vars: Vec::new(),
        });
        let mut state = empty_state();
        // Pre-fire: edge inactive.
        poll(&mut reg, &mut state, 99);
        assert!(!state.pseudo_class_active("active", "kb-1"));
        // At tick 100 the watch fires; make-edge flips on.
        poll(&mut reg, &mut state, 100);
        assert!(state.pseudo_class_active("active", "kb-1"));
        // Mid-hold: still active.
        poll(&mut reg, &mut state, 149);
        assert!(state.pseudo_class_active("active", "kb-1"));
        // Past release_tick (100+50=150): release flips off.
        poll(&mut reg, &mut state, 150);
        assert!(!state.pseudo_class_active("active", "kb-1"));
    }

    #[test]
    fn predicate_byte_pattern_at() {
        let mut state = empty_state();
        // Write "FOO" at addr 100, 102, 104 (stride 2).
        state.write_mem(100, b'F' as i32);
        state.write_mem(102, b'O' as i32);
        state.write_mem(104, b'O' as i32);
        let p = Predicate::BytePatternAt {
            base: 100,
            stride: 2,
            max_window: 200,
            needle: b"FOO".to_vec(),
        };
        assert!(predicate_holds(&p, &state));
        // Different needle — should miss.
        let p_miss = Predicate::BytePatternAt {
            base: 100,
            stride: 2,
            max_window: 200,
            needle: b"BAR".to_vec(),
        };
        assert!(!predicate_holds(&p_miss, &state));
    }

    #[test]
    fn template_substitution() {
        assert_eq!(render_template("a-{tick}-{name}.bin", 42, "x"), "a-42-x.bin");
        assert_eq!(render_template("plain", 1, "y"), "plain");
    }

    #[test]
    fn edge_primed_by_first_poll() {
        let mut reg = WatchRegistry::new();
        reg.register(WatchSpec {
            name: "edge".to_string(),
            kind: WatchKind::Edge {
                addr: 0x100,
                last: Cell::new(0),
                primed: Cell::new(false),
            },
            gate: None,
            actions: vec![Action::Emit],
            sample_vars: Vec::new(),
        });
        let mut state = empty_state();
        // Tick 0: state mem is all zero. First poll primes (no fire).
        poll(&mut reg, &mut state, 0);
        assert_eq!(reg.drain_events().len(), 0);
        // Tick 1: still zero. No transition.
        poll(&mut reg, &mut state, 1);
        assert_eq!(reg.drain_events().len(), 0);
        // Write a non-zero byte; tick 2 should see the transition.
        state.write_mem(0x100, 0x42);
        poll(&mut reg, &mut state, 2);
        assert_eq!(reg.drain_events().len(), 1);
        // Tick 3 same value — no fire.
        poll(&mut reg, &mut state, 3);
        assert_eq!(reg.drain_events().len(), 0);
    }
}
