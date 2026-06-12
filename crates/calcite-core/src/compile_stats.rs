//! Per-phase compile timing, recorded during parse/compile.
//!
//! Native builds can read phase timings off stderr (`RUST_LOG=info`), but
//! wasm has no env or stderr and worker console logs don't reliably reach
//! the host page. This module records the same numbers the `[compile
//! phase]` / `[compile detail]` log lines carry into a thread-local list
//! that hosts retrieve after compile — calcite-wasm exposes it as
//! `CalciteEngine::compile_phase_report()` (JSON).
//!
//! The recording is append-only and ~free (one Vec push per phase, a
//! couple dozen per compile).

use std::cell::RefCell;

thread_local! {
    static PHASES: RefCell<Vec<(String, f64)>> = const { RefCell::new(Vec::new()) };
}

/// Clear recorded phases. Called at the start of `parse_stylesheet` so one
/// engine construction yields one report.
pub fn reset() {
    PHASES.with(|p| p.borrow_mut().clear());
}

/// Record a completed phase.
pub fn record(name: &str, secs: f64) {
    PHASES.with(|p| p.borrow_mut().push((name.to_string(), secs)));
}

/// All recorded phases, in order, as a JSON array:
/// `[{"phase":"parse.fast_scan","secs":1.234}, …]`.
pub fn report_json() -> String {
    PHASES.with(|p| {
        let v = p.borrow();
        let items: Vec<String> = v
            .iter()
            .map(|(n, s)| format!("{{\"phase\":{n:?},\"secs\":{s:.3}}}"))
            .collect();
        format!("[{}]", items.join(","))
    })
}
