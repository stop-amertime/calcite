//! calc(ite) — A JIT compiler for computational CSS.
//!
//! Parses CSS files, recognises computational patterns (large if(style()) dispatch chains,
//! broadcast writes, bitwise decomposition), and compiles them into efficient native
//! operations. The primary target is running x86CSS faster than Chrome's native style
//! resolver.

/// CSS expression compiler — flattens Expr trees into flat bytecode.
pub mod compile;
/// Per-phase compile timing, retrievable by hosts (wasm phase report).
pub mod compile_stats;
/// Chrome conformance comparison utilities (requires `conformance` feature).
#[cfg(feature = "conformance")]
pub mod conformance;
/// Error types.
pub mod error;
/// Expression evaluator — runs compiled programs against flat state.
pub mod eval;
/// CSS parser — tokenisation and expression tree construction.
pub mod parser;
/// Pattern recognition — dispatch tables, broadcast writes.
pub mod pattern;
/// Generic script-primitive layer: stride / burst / at / edge / cond /
/// halt watches with emit / dump / snapshot / set-var actions. See
/// [`script`] for the design and [`script_eval::poll`] for the
/// evaluator.
pub mod script;
/// Evaluator for the [`script`] layer — kept separate from the surface
/// declaration so the surface stays readable.
pub mod script_eval;
/// Text-format parser for [`script::WatchSpec`] strings. Used by both
/// calcite-cli (`--watch`) and calcite-wasm (`engine.register_watch`)
/// so the syntax stays in one place.
pub mod script_spec;
/// Machine state — registers and memory.
pub mod state;
/// Runtime loop-period detector and affine projector.
pub mod tick_period;
/// Signature-based cycle detector (experimental, WIP — projection bug).
pub mod cycle_tracker;
/// Execution summary — event log, block segmenter, prose renderer.
pub mod summary;
/// IR type definitions — expressions, assignments, programs.
pub mod types;

pub use error::{CalciteError, Result};
pub use eval::{property_to_address, Evaluator, TickProfile};
pub use state::{State, StateDelta};
