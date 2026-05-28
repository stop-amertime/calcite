//! Pattern recognition and compilation.
//!
//! After parsing, this module identifies computational patterns in the CSS
//! and replaces them with efficient equivalents:
//!
//! - **Large `if(style())` dispatch → hash map / array index.**
//!   Transforms `readMem()` from O(1602) to O(1).
//!
//! - **Broadcast write → direct store.**
//!   Transforms per-tick state update from O(1583) comparisons to O(1) write.
//!
//! - **Bit decomposition → native bitwise ops.** (Optional, future)
//!
//! - **`@function` inlining.** (Optional, future)

pub mod broadcast_write;
pub mod byte_period;
pub mod dispatch_table;
pub mod fusion_sim;
pub mod loop_descriptor;
pub mod op_profile;
pub mod packed_broadcast_write;
pub mod replicated_body;
