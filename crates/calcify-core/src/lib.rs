//! calc(ify) — A JIT compiler for computational CSS.
//!
//! Parses CSS files, recognises computational patterns (large if(style()) dispatch chains,
//! broadcast writes, bitwise decomposition), and compiles them into efficient native
//! operations. The primary target is running x86CSS faster than Chrome's native style
//! resolver.

#[cfg(feature = "conformance")]
pub mod conformance;
pub mod error;
pub mod eval;
pub mod parser;
pub mod pattern;
pub mod state;
pub mod types;

pub use error::{CalcifyError, Result};
pub use eval::{property_to_address, Evaluator};
pub use state::State;
