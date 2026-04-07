//! calc(ite) — A JIT compiler for computational CSS.
//!
//! Parses CSS files, recognises computational patterns (large if(style()) dispatch chains,
//! broadcast writes, bitwise decomposition), and compiles them into efficient native
//! operations. The primary target is running x86CSS faster than Chrome's native style
//! resolver.

/// CSS expression compiler — flattens Expr trees into flat bytecode.
pub mod compile;
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
/// Machine state — registers and memory.
pub mod state;
/// IR type definitions — expressions, assignments, programs.
pub mod types;

pub use error::{CalciteError, Result};
pub use eval::{property_to_address, Evaluator};
pub use state::State;
