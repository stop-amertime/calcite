//! Integration tests for the script-primitive layer. Each test uses
//! the smallest possible CSS cabinet that exercises the primitive
//! end-to-end through the real Evaluator + State; no x86 content.
//!
//! These tests double as living documentation for the API: they show
//! exactly how a host (CLI, wasm bridge, future bench harness)
//! composes a WatchRegistry with run_batch_silent / poll to drive
//! measurements.

use calcite_core::eval::Evaluator;
use calcite_core::parser::parse_css;
use calcite_core::script::*;
use calcite_core::script_eval::poll;
use calcite_core::State;
use std::cell::Cell;

/// A 1-property cabinet that increments a counter every tick.
///
/// Includes a stub `--opcode` property because calcite-core's
/// `rep_fast_forward` post-tick hook panics on cabinets without that
/// slot (pre-existing issue documented in CSS-DOS LOGBOOK 2026-05-01).
/// Setting it to a non-string-op value (here 0x90 NOP) makes the hook
/// short-circuit cleanly.
const COUNTER_CSS: &str = r#"
@property --n {
    syntax: "<integer>";
    inherits: true;
    initial-value: 0;
}
@property --opcode {
    syntax: "<integer>";
    inherits: true;
    initial-value: 144;
}
.cpu {
    --n: calc(var(--n) + 1);
}
"#;

fn build(css: &str) -> (Evaluator, State) {
    let parsed = parse_css(css).expect("parse");
    let mut state = State::default();
    state.load_properties(&parsed.properties);
    let evaluator = Evaluator::from_parsed(&parsed);
    (evaluator, state)
}

/// Drive the engine tick-by-tick for `n_ticks`, polling the registry
/// after each tick. Returns the events drained from the registry.
fn run_with_polling(
    evaluator: &mut Evaluator,
    state: &mut State,
    reg: &mut WatchRegistry,
    n_ticks: u32,
) -> Vec<MeasurementEvent> {
    let mut events = Vec::new();
    for tick in 1..=n_ticks {
        evaluator.run_batch_silent(state, 1);
        poll(reg, state, tick);
        events.extend(reg.drain_events());
        if reg.halt_requested() {
            break;
        }
    }
    events
}

#[test]
fn stride_fires_every_n_ticks_against_real_engine() {
    let (mut ev, mut state) = build(COUNTER_CSS);
    let mut reg = WatchRegistry::new();
    reg.register(WatchSpec {
        name: "every100".to_string(),
        kind: WatchKind::Stride { every: 100, last_fired_at: Cell::new(0) },
        gate: None,
        actions: vec![Action::Emit],
        sample_vars: vec!["n".to_string()],
    });
    let events = run_with_polling(&mut ev, &mut state, &mut reg, 1000);
    // 1000 ticks / 100 = 10 fires (at 100, 200, ..., 1000).
    assert_eq!(events.len(), 10);
    // Sampled --n values follow the same cadence.
    assert_eq!(events[0].sampled_vars, vec![("n".to_string(), 100)]);
    assert_eq!(events[9].sampled_vars, vec![("n".to_string(), 1000)]);
    assert_eq!(events[0].tick, 100);
    assert_eq!(events[9].tick, 1000);
}

#[test]
fn cond_fires_once_when_byte_predicate_holds() {
    // We don't need a CSS cabinet that writes the byte itself  the
    // test injects the byte transition via state.write_mem at tick
    // 500. This isolates the primitive from any cabinet shape.
    let (mut ev, mut state) = build(COUNTER_CSS);
    let mut reg = WatchRegistry::new();

    // Cheap stride watch that gates the expensive cond.
    reg.register(WatchSpec {
        name: "poll".to_string(),
        kind: WatchKind::Stride { every: 50, last_fired_at: Cell::new(0) },
        gate: None,
        actions: vec![],
        sample_vars: vec![],
    });
    // Expensive cond: byte at 0x200 == 1. Gated by `poll`. Halts when
    // it fires.
    reg.register(WatchSpec {
        name: "byte_set".to_string(),
        kind: WatchKind::Cond {
            tests: vec![Predicate::ByteEq { addr: 0x200, val: 1 }],
            repeat: false,
            fired: Cell::new(false),
            last_held: Cell::new(false),
        },
        gate: Some("poll".to_string()),
        actions: vec![Action::Emit, Action::Halt],
        sample_vars: vec!["n".to_string()],
    });

    let mut events = Vec::new();
    let mut halted = false;
    for tick in 1..=1000 {
        if tick == 500 {
            state.write_mem(0x200, 1);
        }
        ev.run_batch_silent(&mut state, 1);
        poll(&mut reg, &mut state, tick);
        events.extend(reg.drain_events());
        if reg.halt_requested() {
            halted = true;
            break;
        }
    }
    assert!(halted, "cond should have requested halt");
    assert_eq!(events.len(), 1);
    // Byte was written before the run_batch+poll at tick 500; stride
    // 50 fires there, the gated cond evaluates and matches.
    assert_eq!(events[0].tick, 500);
    assert_eq!(events[0].watch_name, "byte_set");
    assert!(events[0].halted);
    assert_eq!(events[0].sampled_vars, vec![("n".to_string(), 500)]);
}

#[test]
fn edge_fires_on_zero_to_one_transition() {
    let (mut ev, mut state) = build(COUNTER_CSS);
    let mut reg = WatchRegistry::new();
    reg.register(WatchSpec {
        name: "poll".to_string(),
        kind: WatchKind::Stride { every: 1, last_fired_at: Cell::new(0) },
        gate: None,
        actions: vec![],
        sample_vars: vec![],
    });
    reg.register(WatchSpec {
        name: "edge".to_string(),
        kind: WatchKind::Edge {
            addr: 0x300,
            last: Cell::new(0),
            primed: Cell::new(false),
        },
        gate: Some("poll".to_string()),
        actions: vec![Action::Emit],
        sample_vars: vec![],
    });

    let mut events = Vec::new();
    for tick in 1..=1000 {
        if tick == 700 {
            state.write_mem(0x300, 0xAB);
        }
        if tick == 800 {
            state.write_mem(0x300, 0x00);
        }
        ev.run_batch_silent(&mut state, 1);
        poll(&mut reg, &mut state, tick);
        events.extend(reg.drain_events());
    }
    // First tick primes the edge watch (records last=0); the 0->0xAB
    // transition fires at tick 700; the 0xAB->0 transition fires at 800.
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].tick, 700);
    assert_eq!(events[1].tick, 800);
}

#[test]
fn halt_kind_stops_run_and_emits() {
    let (mut ev, mut state) = build(COUNTER_CSS);
    let mut reg = WatchRegistry::new();
    reg.register(WatchSpec {
        name: "halt_byte".to_string(),
        kind: WatchKind::Halt { addr: 0x400 },
        gate: None,
        actions: vec![Action::Emit],
        sample_vars: vec![],
    });

    let mut events = Vec::new();
    let mut last_tick = 0u32;
    for tick in 1..=2000 {
        if tick == 1234 {
            state.write_mem(0x400, 0xFF);
        }
        ev.run_batch_silent(&mut state, 1);
        poll(&mut reg, &mut state, tick);
        events.extend(reg.drain_events());
        last_tick = tick;
        if reg.halt_requested() {
            break;
        }
    }
    assert_eq!(last_tick, 1234);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].tick, 1234);
    assert!(events[0].halted);
}

#[test]
fn at_kind_fires_exactly_once_at_target_tick() {
    let (mut ev, mut state) = build(COUNTER_CSS);
    let mut reg = WatchRegistry::new();
    reg.register(WatchSpec {
        name: "scheduled".to_string(),
        kind: WatchKind::At { tick: 333 },
        gate: None,
        actions: vec![Action::Emit, Action::SetVar {
            name: "n".to_string(),
            value: 999_999,
        }],
        sample_vars: vec![],
    });
    let events = run_with_polling(&mut ev, &mut state, &mut reg, 500);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].tick, 333);
    // SetVar action ran at tick 333 (after the 333rd run_batch); the
    // remaining 167 ticks (334..=500) each increment --n. So the
    // final value is 999_999 + 167 = 1_000_166.
    assert_eq!(state.get_var("n"), Some(1_000_166));
}

#[test]
fn dump_mem_range_attaches_payload_to_event() {
    let (mut ev, mut state) = build(COUNTER_CSS);
    // Pre-populate a 16-byte region.
    for i in 0..16 {
        state.write_mem(0x500 + i, (i + 1) as i32);
    }
    let mut reg = WatchRegistry::new();
    reg.set_dump_sink(DumpSink::Memory);
    reg.register(WatchSpec {
        name: "dump_at_100".to_string(),
        kind: WatchKind::At { tick: 100 },
        gate: None,
        actions: vec![Action::DumpMemRange {
            addr: 0x500,
            len: 16,
            path_template: "ignored".to_string(),
        }],
        sample_vars: vec![],
    });
    let events = run_with_polling(&mut ev, &mut state, &mut reg, 200);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].dumps.len(), 1);
    let dump = &events[0].dumps[0];
    assert_eq!(dump.tag, "mem");
    assert_eq!(dump.addr, 0x500);
    assert_eq!(dump.bytes, (1u8..=16).collect::<Vec<u8>>());
    // Memory sink leaves path None.
    assert!(dump.path.is_none());
}

#[test]
fn cond_with_byte_pattern_at() {
    // Compose the primitive that replaces the old vram_text:NEEDLE
    // helper: walking memory at (base, stride) for an ASCII needle.
    let (mut ev, mut state) = build(COUNTER_CSS);
    let mut reg = WatchRegistry::new();
    reg.register(WatchSpec {
        name: "poll".to_string(),
        kind: WatchKind::Stride { every: 25, last_fired_at: Cell::new(0) },
        gate: None,
        actions: vec![],
        sample_vars: vec![],
    });
    reg.register(WatchSpec {
        name: "needle_visible".to_string(),
        kind: WatchKind::Cond {
            tests: vec![Predicate::BytePatternAt {
                base: 0x100,
                stride: 2,
                max_window: 200,
                needle: b"HELLO".to_vec(),
            }],
            repeat: false,
            fired: Cell::new(false),
            last_held: Cell::new(false),
        },
        gate: Some("poll".to_string()),
        actions: vec![Action::Emit, Action::Halt],
        sample_vars: vec![],
    });

    // Inject "HELLO" at stride 2 starting at 0x100 + 10, at tick 350.
    let mut halted = false;
    let mut events = Vec::new();
    for tick in 1..=1000 {
        if tick == 350 {
            for (i, &b) in b"HELLO".iter().enumerate() {
                state.write_mem(0x100 + 10 + (i as i32) * 2, b as i32);
            }
        }
        ev.run_batch_silent(&mut state, 1);
        poll(&mut reg, &mut state, tick);
        events.extend(reg.drain_events());
        if reg.halt_requested() {
            halted = true;
            break;
        }
    }
    assert!(halted);
    assert_eq!(events.len(), 1);
    // Stride 25; first poll-tick >= 350 is 350.
    assert_eq!(events[0].tick, 350);
    assert_eq!(events[0].watch_name, "needle_visible");
}

#[test]
fn setvar_pulse_writes_make_then_break_after_hold_ticks() {
    // SetVarPulse writes VALUE immediately, schedules write-of-0 after
    // HOLD_TICKS more polled ticks. This is the generic edge-pair
    // primitive that CSS-DOS-side profiles use for keyboard taps —
    // the cabinet's edge detector needs both make and break to
    // register a press.
    //
    // Add a `--keyboard` property to the test cabinet so the host can
    // watch it from the State side.
    let css = r#"
@property --n {
    syntax: "<integer>";
    inherits: true;
    initial-value: 0;
}
@property --keyboard {
    syntax: "<integer>";
    inherits: true;
    initial-value: 0;
}
@property --opcode {
    syntax: "<integer>";
    inherits: true;
    initial-value: 144;
}
.cpu {
    --n: calc(var(--n) + 1);
}
"#;
    let (mut ev, mut state) = build(css);
    let mut reg = WatchRegistry::new();

    // Fire a pulse at tick 100, hold for 50 ticks. Expected timeline:
    //   tick 99:   keyboard = 0  (initial)
    //   tick 100:  pulse fires → keyboard = 0x1c0d, release scheduled
    //              for tick 100 + 50 = 150
    //   tick 100..149: keyboard = 0x1c0d
    //   tick 150:  release dispatched at top of poll → keyboard = 0
    //   tick 150..: keyboard = 0
    reg.register(WatchSpec {
        name: "tap".to_string(),
        kind: WatchKind::At { tick: 100 },
        gate: None,
        actions: vec![Action::SetVarPulse {
            name: "keyboard".to_string(),
            value: 0x1c0d,
            hold_ticks: 50,
        }],
        sample_vars: vec![],
    });

    // Sample the var at specific ticks.
    let mut samples: Vec<(u32, i32)> = Vec::new();
    for tick in 1..=300 {
        ev.run_batch_silent(&mut state, 1);
        poll(&mut reg, &mut state, tick);
        let _ = reg.drain_events();
        if matches!(tick, 99 | 100 | 120 | 149 | 150 | 200) {
            let v = state.get_var("keyboard").unwrap_or(0);
            samples.push((tick, v));
        }
    }
    // tick 99: pre-pulse → 0
    assert_eq!(samples.iter().find(|(t,_)| *t==99).unwrap().1, 0);
    // tick 100: make edge → 0x1c0d
    assert_eq!(samples.iter().find(|(t,_)| *t==100).unwrap().1, 0x1c0d);
    // tick 120 / 149: still held
    assert_eq!(samples.iter().find(|(t,_)| *t==120).unwrap().1, 0x1c0d);
    assert_eq!(samples.iter().find(|(t,_)| *t==149).unwrap().1, 0x1c0d);
    // tick 150: release dispatched at top of poll → 0
    assert_eq!(samples.iter().find(|(t,_)| *t==150).unwrap().1, 0);
    assert_eq!(samples.iter().find(|(t,_)| *t==200).unwrap().1, 0);
}

#[test]
fn setvar_pulse_skips_when_release_pending() {
    // A second pulse on the same var while the first is still
    // mid-hold is a no-op — skip rather than re-arm. This is the
    // semantic the doom-loading bench needs: a sustain `cond:repeat`
    // that fires on every gated poll while a stage holds must NOT
    // re-pulse on every poll, because the same-poll race
    // (release-fires-then-pulse-fires) would write the value back
    // before the engine ever sees zero, so the cabinet's edge
    // detector never registers the break edge. Skipping while a
    // release is queued gives the engine a full inter-poll batch
    // with the var = 0 between releases, producing the make/break
    // pair the cabinet needs.
    let css = r#"
@property --keyboard { syntax: "<integer>"; inherits: true; initial-value: 0; }
@property --opcode { syntax: "<integer>"; inherits: true; initial-value: 144; }
.cpu {}
"#;
    let (mut ev, mut state) = build(css);
    let mut reg = WatchRegistry::new();

    // Two taps at 100 and 150, both with hold=100. With skip-while-
    // pending, tap2 is a no-op (release from tap1 is still queued at
    // 200 when tap2 fires). Release happens at 200; var stays 0
    // afterwards.
    reg.register(WatchSpec {
        name: "tap1".to_string(),
        kind: WatchKind::At { tick: 100 },
        gate: None,
        actions: vec![Action::SetVarPulse {
            name: "keyboard".to_string(),
            value: 0x1c0d,
            hold_ticks: 100,
        }],
        sample_vars: vec![],
    });
    reg.register(WatchSpec {
        name: "tap2".to_string(),
        kind: WatchKind::At { tick: 150 },
        gate: None,
        actions: vec![Action::SetVarPulse {
            name: "keyboard".to_string(),
            value: 0x1c0d,
            hold_ticks: 100,
        }],
        sample_vars: vec![],
    });

    let mut samples: Vec<(u32, i32)> = Vec::new();
    for tick in 1..=400 {
        ev.run_batch_silent(&mut state, 1);
        poll(&mut reg, &mut state, tick);
        let _ = reg.drain_events();
        if matches!(tick, 99 | 100 | 149 | 150 | 199 | 200 | 201) {
            samples.push((tick, state.get_var("keyboard").unwrap_or(0)));
        }
    }
    let at = |t: u32| samples.iter().find(|(x,_)| *x==t).unwrap().1;
    assert_eq!(at(99),  0);
    assert_eq!(at(100), 0x1c0d);  // first tap's make
    assert_eq!(at(149), 0x1c0d);  // still held
    assert_eq!(at(150), 0x1c0d);  // tap2 was a no-op (skipped)
    assert_eq!(at(199), 0x1c0d);  // still held (release at 200)
    assert_eq!(at(200), 0);       // tap1's release fires
    assert_eq!(at(201), 0);       // stays released
}

#[test]
fn sustain_cond_pulse_alternates_make_and_break_at_2x_poll_stride() {
    // The doom-loading bench's pattern: a sustain `cond:repeat` with
    // setvar_pulse should produce a clean make/break alternation at
    // twice the poll stride (one stride for the held make, one stride
    // for the released break). Without skip-when-released-this-poll,
    // the make would re-fire the same poll the release dispatched and
    // the engine would never see the break edge.
    let css = r#"
@property --keyboard { syntax: "<integer>"; inherits: true; initial-value: 0; }
@property --opcode { syntax: "<integer>"; inherits: true; initial-value: 144; }
@property --target_byte { syntax: "<integer>"; inherits: true; initial-value: 1; }
.cpu {}
"#;
    let (mut ev, mut state) = build(css);
    let mut reg = WatchRegistry::new();

    // Cheap stride gate at 100 ticks (substitute for the 50K poll
    // stride the doom bench uses; same shape, faster test).
    reg.register(WatchSpec {
        name: "poll".to_string(),
        kind: WatchKind::Stride { every: 100, last_fired_at: Cell::new(0) },
        gate: None,
        actions: vec![],
        sample_vars: vec![],
    });
    // Sustain cond: predicate is "byte at 0x300 == 1" (always true in
    // this test — no cabinet writes 0x300). Sustain mode means it
    // fires on every gated poll while held.
    state.write_mem(0x300, 1);
    reg.register(WatchSpec {
        name: "tap".to_string(),
        kind: WatchKind::Cond {
            tests: vec![Predicate::ByteEq { addr: 0x300, val: 1 }],
            repeat: true,  // sustain
            fired: std::cell::Cell::new(false),
            last_held: std::cell::Cell::new(false),
        },
        gate: Some("poll".to_string()),
        actions: vec![Action::SetVarPulse {
            name: "keyboard".to_string(),
            value: 0x1c0d,
            hold_ticks: 100,
        }],
        sample_vars: vec![],
    });

    // Sample at every poll boundary to see the cadence.
    let mut samples: Vec<(u32, i32)> = Vec::new();
    for tick in 1..=600 {
        ev.run_batch_silent(&mut state, 1);
        poll(&mut reg, &mut state, tick);
        let _ = reg.drain_events();
        // Sample MID-batch (between polls) so we see what the engine
        // sees: tick 50, 150, 250, ... — halfway between polls.
        if tick % 100 == 50 {
            samples.push((tick, state.get_var("keyboard").unwrap_or(0)));
        }
    }
    // Expected mid-batch values:
    //   tick 50: keyboard = 0 (no poll has fired yet)
    //   tick 150: keyboard = 0x1c0d (poll@100 pulsed, release scheduled @200)
    //   tick 250: keyboard = 0 (poll@200 released, skipped pulse this poll)
    //   tick 350: keyboard = 0x1c0d (poll@300 pulsed, release @400)
    //   tick 450: keyboard = 0 (poll@400 released, skipped)
    //   tick 550: keyboard = 0x1c0d (poll@500 pulsed)
    let at = |t: u32| samples.iter().find(|(x,_)| *x==t).unwrap().1;
    assert_eq!(at(50),  0);
    assert_eq!(at(150), 0x1c0d);
    assert_eq!(at(250), 0);
    assert_eq!(at(350), 0x1c0d);
    assert_eq!(at(450), 0);
    assert_eq!(at(550), 0x1c0d);
}
