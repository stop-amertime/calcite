//! Parser for the calcite-cli `--watch` flag.
//!
//! Syntax:
//!
//! ```text
//! NAME:KIND:SPEC[:gate=NAME][:sample=VAR1,VAR2][:then=ACTION1+ACTION2+...]
//! ```
//!
//! Field delimiter is `:`. Within `sample=` the variable list uses `,`.
//! Within `then=` the action list uses `+` (so action sub-args can use
//! `,` and `:` freely  important for `dump=ADDR,LEN,PATH` whose PATH
//! may contain Windows-style `C:\foo`).
//!
//! KIND:SPEC variants:
//!
//! - `stride:every=N`
//! - `burst:every=N,count=K`
//! - `at:tick=T`
//! - `edge:addr=A`
//! - `cond:TEST1[,TEST2...][,repeat]`
//! - `halt:addr=A`
//!
//! TEST: `ADDR=VAL` | `ADDR!=VAL` | `pattern@BASE:STRIDE:WINDOW=NEEDLE`
//!
//! ACTION (within `then=`):
//!
//! - `emit`
//! - `halt`
//! - `setvar=NAME,VALUE`
//! - `setvar_pulse=NAME,VALUE,HOLD_TICKS`  (make→break edge pair;
//!                                          writes VALUE now, schedules
//!                                          write-of-0 after HOLD_TICKS)
//! - `pseudo_pulse=PSEUDO,SELECTOR,HOLD_TICKS`
//!                                         (pseudo-class make→break;
//!                                          flips edge active now,
//!                                          schedules release after
//!                                          HOLD_TICKS — drives the
//!                                          cabinet's `:has(...:pseudo)`
//!                                          input-edge rules)
//! - `dump=ADDR,LEN[,PATH]`     (PATH supports `{tick}` / `{name}`)
//! - `snapshot[=PATH]`
//!
//! Numbers may be decimal or `0x`-prefixed.

use crate::script::*;
use std::cell::Cell;

fn parse_int(s: &str, what: &str) -> Result<i64, String> {
    let s = s.trim();
    let v = if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i64::from_str_radix(h, 16)
    } else {
        s.parse::<i64>()
    };
    v.map_err(|e| format!("invalid {what} {s:?}: {e}"))
}

fn require_kv<'a>(s: &'a str, key: &str) -> Result<&'a str, String> {
    let prefix = format!("{key}=");
    s.strip_prefix(&prefix)
        .ok_or_else(|| format!("expected {key}=...; got {s:?}"))
}

/// Split a `then=...` payload into individual ACTION strings on `+`.
fn split_actions(s: &str) -> Vec<&str> {
    s.split('+').filter(|t| !t.is_empty()).collect()
}

/// Split a `cond` SPEC into TEST tokens. Tests use `,` and may
/// optionally include the literal token `repeat` to mean "fire on
/// every gated poll while the predicate holds (sustain), instead of
/// the default of firing once and disabling".
fn split_tests(s: &str) -> Vec<&str> {
    s.split(',').map(str::trim).filter(|t| !t.is_empty()).collect()
}

fn parse_predicate(tok: &str) -> Result<Predicate, String> {
    if let Some(rest) = tok.strip_prefix("pattern@") {
        // pattern@BASE:STRIDE:WINDOW=NEEDLE
        let (head, needle) = rest.split_once('=')
            .ok_or_else(|| format!("pattern@ test missing =NEEDLE: {tok:?}"))?;
        let parts: Vec<&str> = head.split(':').collect();
        if parts.len() != 3 {
            return Err(format!(
                "pattern@ expects BASE:STRIDE:WINDOW, got {head:?}"
            ));
        }
        let base = parse_int(parts[0], "BASE")? as i32;
        let stride = parse_int(parts[1], "STRIDE")? as u32;
        let max_window = parse_int(parts[2], "WINDOW")? as u32;
        Ok(Predicate::BytePatternAt {
            base,
            stride,
            max_window,
            needle: needle.as_bytes().to_vec(),
        })
    } else if let Some(idx) = tok.find("!=") {
        let (a, v) = tok.split_at(idx);
        let v = &v[2..];
        let addr = parse_int(a, "ADDR")? as i32;
        let val = parse_int(v, "VAL")? as u8;
        Ok(Predicate::ByteNe { addr, val })
    } else if let Some(idx) = tok.find('=') {
        let (a, v) = tok.split_at(idx);
        let v = &v[1..];
        let addr = parse_int(a, "ADDR")? as i32;
        let val = parse_int(v, "VAL")? as u8;
        Ok(Predicate::ByteEq { addr, val })
    } else {
        Err(format!(
            "test {tok:?} not ADDR=VAL / ADDR!=VAL / pattern@BASE:STRIDE:WINDOW=NEEDLE"
        ))
    }
}

fn parse_action(tok: &str) -> Result<Action, String> {
    let tok = tok.trim();
    match tok {
        "emit" => Ok(Action::Emit),
        "halt" => Ok(Action::Halt),
        _ => {
            if let Some(rest) = tok.strip_prefix("setvar=") {
                let parts: Vec<&str> = rest.split(',').collect();
                if parts.len() != 2 {
                    return Err(format!("setvar expects NAME,VALUE; got {rest:?}"));
                }
                let value = parse_int(parts[1], "VALUE")? as i32;
                Ok(Action::SetVar {
                    name: parts[0].to_string(),
                    value,
                })
            } else if let Some(rest) = tok.strip_prefix("setvar_pulse=") {
                // setvar_pulse=NAME,VALUE,HOLD_TICKS — write VALUE now,
                // schedule a write-of-0 HOLD_TICKS later. Generic edge-pair
                // primitive; CSS-DOS-side profiles use it for keyboard taps.
                let parts: Vec<&str> = rest.split(',').collect();
                if parts.len() != 3 {
                    return Err(format!(
                        "setvar_pulse expects NAME,VALUE,HOLD_TICKS; got {rest:?}"
                    ));
                }
                let value = parse_int(parts[1], "VALUE")? as i32;
                let hold_ticks = parse_int(parts[2], "HOLD_TICKS")? as u32;
                if hold_ticks == 0 {
                    return Err(format!(
                        "setvar_pulse HOLD_TICKS must be > 0 (would never release): {rest:?}"
                    ));
                }
                Ok(Action::SetVarPulse {
                    name: parts[0].to_string(),
                    value,
                    hold_ticks,
                })
            } else if let Some(rest) = tok.strip_prefix("pseudo_pulse=") {
                // pseudo_pulse=PSEUDO,SELECTOR,HOLD_TICKS — flip the
                // (PSEUDO, SELECTOR) match edge active now, schedule a
                // release HOLD_TICKS later. Generic edge-pair primitive
                // for the pseudo-class input surface; CSS-DOS-side
                // profiles use it for click-driven keyboard taps that
                // exercise the cabinet's `&:has(#SEL:PSEUDO) { --PROP: V }`
                // rules through calcite's input-edge recogniser.
                let parts: Vec<&str> = rest.split(',').collect();
                if parts.len() != 3 {
                    return Err(format!(
                        "pseudo_pulse expects PSEUDO,SELECTOR,HOLD_TICKS; got {rest:?}"
                    ));
                }
                let hold_ticks = parse_int(parts[2], "HOLD_TICKS")? as u32;
                if hold_ticks == 0 {
                    return Err(format!(
                        "pseudo_pulse HOLD_TICKS must be > 0 (would never release): {rest:?}"
                    ));
                }
                Ok(Action::PseudoActivePulse {
                    pseudo: parts[0].to_string(),
                    selector: parts[1].to_string(),
                    hold_ticks,
                })
            } else if let Some(rest) = tok.strip_prefix("dump=") {
                // ADDR,LEN[,PATH]. ADDR/LEN can use 0x prefix; PATH may
                // contain commas (rare) by being the trailing field.
                let mut iter = rest.splitn(3, ',');
                let addr_s = iter.next().unwrap_or("");
                let len_s = iter.next().ok_or_else(|| {
                    format!("dump expects ADDR,LEN[,PATH]; got {rest:?}")
                })?;
                let path = iter.next().map(|s| s.to_string())
                    .unwrap_or_else(|| "dump_{tick}.bin".to_string());
                let addr = parse_int(addr_s, "ADDR")? as i32;
                let len = parse_int(len_s, "LEN")? as u32;
                Ok(Action::DumpMemRange {
                    addr,
                    len,
                    path_template: path,
                })
            } else if let Some(rest) = tok.strip_prefix("snapshot=") {
                Ok(Action::Snapshot {
                    path_template: rest.to_string(),
                })
            } else if tok == "snapshot" {
                Ok(Action::Snapshot {
                    path_template: "snapshot_{tick}.bin".to_string(),
                })
            } else {
                Err(format!("unknown action {tok:?}"))
            }
        }
    }
}

/// Parse a single `--watch` spec string into a [`WatchSpec`].
pub fn parse_watch(spec: &str) -> Result<WatchSpec, String> {
    let parts: Vec<&str> = spec.split(':').collect();
    if parts.len() < 3 {
        return Err(format!("--watch expects NAME:KIND:SPEC[...]; got {spec:?}"));
    }
    let name = parts[0].to_string();
    if name.is_empty() {
        return Err(format!("--watch name is empty: {spec:?}"));
    }
    let kind_str = parts[1];
    // The "spec" portion may itself contain colons (pattern@BASE:STRIDE:WINDOW
    // inside cond), so we re-glue everything from index 2 up to the first
    // segment that starts with a known suffix prefix (`gate=`, `sample=`,
    // `then=`).
    let mut spec_segments: Vec<&str> = Vec::new();
    let mut suffix_segments: Vec<&str> = Vec::new();
    let mut in_suffix = false;
    for seg in &parts[2..] {
        let starts_suffix = seg.starts_with("gate=")
            || seg.starts_with("sample=")
            || seg.starts_with("then=");
        if starts_suffix {
            in_suffix = true;
        }
        if in_suffix {
            suffix_segments.push(seg);
        } else {
            spec_segments.push(seg);
        }
    }
    let spec_payload = spec_segments.join(":");

    let mut gate: Option<String> = None;
    let mut sample_vars: Vec<String> = Vec::new();
    let mut actions: Vec<Action> = Vec::new();
    for seg in suffix_segments {
        if let Some(g) = seg.strip_prefix("gate=") {
            gate = Some(g.to_string());
        } else if let Some(s) = seg.strip_prefix("sample=") {
            sample_vars = s.split(',').map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty()).collect();
        } else if let Some(a) = seg.strip_prefix("then=") {
            for tok in split_actions(a) {
                actions.push(parse_action(tok)?);
            }
        }
    }

    let kind = match kind_str {
        "stride" => {
            let every = parse_int(require_kv(&spec_payload, "every")?, "every")? as u32;
            WatchKind::Stride { every, last_fired_at: std::cell::Cell::new(0) }
        }
        "burst" => {
            // every=N,count=K
            let mut every: Option<u32> = None;
            let mut count: Option<u32> = None;
            for tok in spec_payload.split(',') {
                if let Some(v) = tok.trim().strip_prefix("every=") {
                    every = Some(parse_int(v, "every")? as u32);
                } else if let Some(v) = tok.trim().strip_prefix("count=") {
                    count = Some(parse_int(v, "count")? as u32);
                } else if !tok.trim().is_empty() {
                    return Err(format!("burst expects every=N,count=K; got {tok:?}"));
                }
            }
            WatchKind::Burst {
                every: every.ok_or("burst missing every=")?,
                count: count.ok_or("burst missing count=")?,
            }
        }
        "at" => {
            let tick = parse_int(require_kv(&spec_payload, "tick")?, "tick")? as u32;
            WatchKind::At { tick }
        }
        "edge" => {
            let addr = parse_int(require_kv(&spec_payload, "addr")?, "addr")? as i32;
            WatchKind::Edge {
                addr,
                last: Cell::new(0),
                primed: Cell::new(false),
            }
        }
        "cond" => {
            let tokens = split_tests(&spec_payload);
            let mut tests = Vec::new();
            let mut repeat = false;
            for tok in tokens {
                if tok == "repeat" {
                    repeat = true;
                    continue;
                }
                tests.push(parse_predicate(tok)?);
            }
            if tests.is_empty() {
                return Err(format!("cond {name:?} parsed to zero tests"));
            }
            WatchKind::Cond {
                tests,
                repeat,
                fired: Cell::new(false),
                last_held: Cell::new(false),
            }
        }
        "halt" => {
            let addr = parse_int(require_kv(&spec_payload, "addr")?, "addr")? as i32;
            WatchKind::Halt { addr }
        }
        other => return Err(format!("unknown --watch KIND {other:?}")),
    };

    Ok(WatchSpec {
        name,
        kind,
        gate,
        actions,
        sample_vars,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stride_basic() {
        let w = parse_watch("tick:stride:every=100").unwrap();
        assert_eq!(w.name, "tick");
        match w.kind {
            WatchKind::Stride { every, .. } => assert_eq!(every, 100),
            _ => panic!("wrong kind"),
        }
    }

    #[test]
    fn parse_cond_with_pattern_and_actions() {
        let w = parse_watch(
            "drdos:cond:0x449=0x03,pattern@0xb8000:2:4000=DR-DOS:gate=poll:then=emit+halt",
        )
        .unwrap();
        assert_eq!(w.name, "drdos");
        assert_eq!(w.gate.as_deref(), Some("poll"));
        assert_eq!(w.actions.len(), 2);
        match &w.kind {
            WatchKind::Cond { tests, repeat, .. } => {
                assert_eq!(tests.len(), 2);
                assert!(!repeat);
            }
            _ => panic!("wrong kind"),
        }
    }

    #[test]
    fn parse_dump_action_with_template_path() {
        let w = parse_watch(
            "dump_at:at:tick=2000000:then=dump=0xb8000,4000,vram_{tick}.bin",
        )
        .unwrap();
        assert_eq!(w.actions.len(), 1);
        match &w.actions[0] {
            Action::DumpMemRange {
                addr,
                len,
                path_template,
            } => {
                assert_eq!(*addr, 0xb8000);
                assert_eq!(*len, 4000);
                assert_eq!(path_template, "vram_{tick}.bin");
            }
            _ => panic!("wrong action"),
        }
    }

    #[test]
    fn parse_burst() {
        let w = parse_watch("b:burst:every=1000,count=5").unwrap();
        match w.kind {
            WatchKind::Burst { every, count } => {
                assert_eq!(every, 1000);
                assert_eq!(count, 5);
            }
            _ => panic!("wrong kind"),
        }
    }

    #[test]
    fn parse_setvar_action() {
        let w = parse_watch("kbd:at:tick=500:then=setvar=keyboard,0x1c0d").unwrap();
        match &w.actions[0] {
            Action::SetVar { name, value } => {
                assert_eq!(name, "keyboard");
                assert_eq!(*value, 0x1c0d);
            }
            _ => panic!("wrong action"),
        }
    }

    #[test]
    fn parse_pseudo_pulse_action() {
        let w = parse_watch(
            "tap:at:tick=500:then=pseudo_pulse=active,kb-enter,50000",
        )
        .unwrap();
        match &w.actions[0] {
            Action::PseudoActivePulse {
                pseudo,
                selector,
                hold_ticks,
            } => {
                assert_eq!(pseudo, "active");
                assert_eq!(selector, "kb-enter");
                assert_eq!(*hold_ticks, 50_000);
            }
            other => panic!("wrong action: {other:?}"),
        }
    }

    #[test]
    fn parse_pseudo_pulse_rejects_bad_arity() {
        assert!(parse_watch("t:at:tick=1:then=pseudo_pulse=foo,bar").is_err());
        assert!(parse_watch("t:at:tick=1:then=pseudo_pulse=foo,bar,0").is_err());
    }

    #[test]
    fn parse_sample_vars() {
        let w = parse_watch("p:stride:every=1000:sample=AX,IP,cycleCount").unwrap();
        assert_eq!(w.sample_vars, vec!["AX", "IP", "cycleCount"]);
    }
}
