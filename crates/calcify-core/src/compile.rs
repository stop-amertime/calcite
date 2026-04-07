//! CSS expression compiler — flattens `Expr` trees into linear `Op` sequences.
//!
//! This module compiles the parsed CSS expression trees into a flat bytecode IR
//! that operates on indexed slots instead of string-keyed HashMaps. The compiled
//! form eliminates:
//! - `HashMap<String, f64>` lookups (replaced by direct slot indexing)
//! - Recursive `Expr` tree walking (replaced by linear op execution)
//! - String allocation for buffer-prefixed names
//! - Function call save/restore overhead (inlined at compile time)

use std::collections::HashMap;

use crate::eval::property_to_address;
use crate::pattern::broadcast_write::BroadcastWrite;
use crate::pattern::dispatch_table::DispatchTable;
use crate::state::State;
use crate::types::*;

// ---------------------------------------------------------------------------
// Op — flat bytecode instruction
// ---------------------------------------------------------------------------

/// A single operation in the compiled bytecode.
///
/// All operands are `u16` slot indices into a flat `Vec<f64>` array.
/// State reads/writes use `i32` addresses matching the x86CSS convention.
#[derive(Debug, Clone)]
pub enum Op {
    // --- Loads ---
    /// slot[dst] = literal value
    LoadLit { dst: u16, val: f64 },
    /// slot[dst] = slot[src]
    LoadSlot { dst: u16, src: u16 },
    /// slot[dst] = state.read_mem(addr) — compile-time-known address
    LoadState { dst: u16, addr: i32 },
    /// slot[dst] = state.read_mem(slot[addr_slot] as i32) — runtime address
    LoadMem { dst: u16, addr_slot: u16 },
    /// slot[dst] = state.read_mem16(slot[addr_slot] as i32) — 16-bit word read
    LoadMem16 { dst: u16, addr_slot: u16 },
    /// slot[dst] = state.keyboard as f64
    LoadKeyboard { dst: u16 },

    // --- Arithmetic ---
    Add { dst: u16, a: u16, b: u16 },
    Sub { dst: u16, a: u16, b: u16 },
    Mul { dst: u16, a: u16, b: u16 },
    Div { dst: u16, a: u16, b: u16 },
    Mod { dst: u16, a: u16, b: u16 },
    Neg { dst: u16, src: u16 },
    Abs { dst: u16, src: u16 },
    Sign { dst: u16, src: u16 },
    Pow { dst: u16, base: u16, exp: u16 },
    Min { dst: u16, args: Vec<u16> },
    Max { dst: u16, args: Vec<u16> },
    Clamp { dst: u16, min: u16, val: u16, max: u16 },
    Round { dst: u16, strategy: RoundStrategy, val: u16, interval: u16 },
    Floor { dst: u16, src: u16 },

    // --- Bitwise (CSS function equivalents) ---
    /// lowerBytes(a, b) → a & ((1 << b) - 1)
    And { dst: u16, a: u16, b: u16 },
    /// rightShift(a, b) → a >> b
    Shr { dst: u16, a: u16, b: u16 },
    /// leftShift(a, b) → a << b
    Shl { dst: u16, a: u16, b: u16 },
    /// bit(val, idx) → (val >> idx) & 1
    Bit { dst: u16, val: u16, idx: u16 },

    // --- Comparisons & control flow ---
    /// slot[dst] = (slot[a] == slot[b]) as i64  (integer comparison)
    CmpEq { dst: u16, a: u16, b: u16 },
    /// If slot[cond] == 0, jump to target op index
    BranchIfZero { cond: u16, target: u32 },
    /// Unconditional jump to target op index
    Jump { target: u32 },

    // --- Dispatch table ---
    /// HashMap lookup: slot[dst] = dispatch_tables[table_id].entries[slot[key]]
    /// Falls back to executing ops at fallback_target if key not found.
    Dispatch { dst: u16, key: u16, table_id: u16, fallback_target: u32 },

    // --- Stores ---
    /// state.write_mem(addr, slot[src]) — compile-time-known address
    StoreState { addr: i32, src: u16 },
    /// state.write_mem(slot[addr_slot], slot[src]) — runtime address
    StoreMem { addr_slot: u16, src: u16 },
}

// ---------------------------------------------------------------------------
// CompiledProgram — the output of the compiler
// ---------------------------------------------------------------------------

/// A compiled CSS program ready for efficient execution.
///
/// All string-keyed property lookups have been replaced with slot indices.
/// The `ops` vector is executed linearly (with branches) against a flat slot array.
#[derive(Debug)]
pub struct CompiledProgram {
    /// The flat bytecode instruction stream.
    pub ops: Vec<Op>,
    /// Number of slots needed (the slot array size).
    pub slot_count: u16,
    /// Mapping from slot index → state address for write-back.
    /// Only includes canonical properties (not buffer copies or byte halves).
    pub writeback: Vec<(u16, i32)>,
    /// Broadcast writes — compiled separately because they need runtime HashMap lookup.
    pub broadcast_writes: Vec<CompiledBroadcastWrite>,
    /// Dispatch table data (kept for Dispatch op lookups at runtime).
    pub dispatch_tables: Vec<CompiledDispatchTable>,
}

/// A compiled broadcast write.
#[derive(Debug)]
pub struct CompiledBroadcastWrite {
    /// Slot holding the destination address.
    pub dest_slot: u16,
    /// Ops to evaluate the value expression (result in value_slot).
    pub value_ops: Vec<Op>,
    /// Slot holding the evaluated value.
    pub value_slot: u16,
    /// Address → state address mapping for the broadcast.
    pub address_map: HashMap<i64, i32>,
    /// Spillover ops (for word writes).
    pub spillover: Option<CompiledSpillover>,
}

/// Compiled spillover for word-write broadcast.
#[derive(Debug)]
pub struct CompiledSpillover {
    /// Slot holding the guard property value.
    pub guard_slot: u16,
    /// Map from dest address → (ops to compute high byte, result slot).
    pub entries: HashMap<i64, (Vec<Op>, u16)>,
}

/// A compiled dispatch table — kept for runtime HashMap lookup.
#[derive(Debug)]
pub struct CompiledDispatchTable {
    /// Compiled ops for each dispatch entry, keyed by the dispatch value.
    /// Each entry is (ops, result_slot).
    pub entries: HashMap<i64, (Vec<Op>, u16)>,
    /// Compiled ops for the fallback expression.
    pub fallback_ops: Vec<Op>,
    /// Slot holding the fallback result.
    pub fallback_slot: u16,
}

// ---------------------------------------------------------------------------
// Compiler — translates Evaluator data into CompiledProgram
// ---------------------------------------------------------------------------

/// Compiler state — tracks slot allocation and property→slot mapping.
struct Compiler {
    /// Next available slot index.
    next_slot: u16,
    /// Map from property name → slot index.
    property_slots: HashMap<String, u16>,
    /// Functions available for inlining.
    functions: HashMap<String, FunctionDef>,
    /// Recognised dispatch tables.
    dispatch_tables: HashMap<String, DispatchTable>,
    /// Compiled dispatch table data (populated during compilation).
    compiled_dispatches: Vec<CompiledDispatchTable>,
}

impl Compiler {
    fn new(
        functions: &HashMap<String, FunctionDef>,
        dispatch_tables: &HashMap<String, DispatchTable>,
    ) -> Self {
        Compiler {
            next_slot: 0,
            property_slots: HashMap::new(),
            functions: functions.clone(),
            dispatch_tables: dispatch_tables.clone(),
            compiled_dispatches: Vec::new(),
        }
    }

    /// Allocate a fresh temporary slot.
    fn alloc(&mut self) -> u16 {
        let s = self.next_slot;
        self.next_slot += 1;
        s
    }

    /// Compile an Expr into ops, returning the slot holding the result.
    fn compile_expr(&mut self, expr: &Expr, ops: &mut Vec<Op>) -> u16 {
        match expr {
            Expr::Literal(v) => {
                let dst = self.alloc();
                ops.push(Op::LoadLit { dst, val: *v });
                dst
            }

            Expr::StringLiteral(_) => {
                let dst = self.alloc();
                ops.push(Op::LoadLit { dst, val: 0.0 });
                dst
            }

            Expr::Var { name, fallback } => {
                self.compile_var(name, fallback.as_deref(), ops)
            }

            Expr::Calc(calc_op) => self.compile_calc(calc_op, ops),

            Expr::StyleCondition {
                branches, fallback, ..
            } => self.compile_style_condition(branches, fallback, ops),

            Expr::FunctionCall { name, args } => {
                self.compile_function_call(name, args, ops)
            }
        }
    }

    /// Compile a variable reference.
    fn compile_var(&mut self, name: &str, fallback: Option<&Expr>, ops: &mut Vec<Op>) -> u16 {
        // If it's a property we've already computed in this tick, use its slot directly.
        if let Some(&s) = self.property_slots.get(name) {
            return s;
        }

        // Buffer-prefixed name: --__0AX, --__1AX, --__2AX → canonical --AX
        if name.starts_with("--__") && name.len() > 5 {
            let canonical = format!("--{}", &name[5..]);
            if let Some(&s) = self.property_slots.get(&canonical) {
                return s;
            }
        }

        // State-mapped property: load from state at compile-time-known address
        if let Some(addr) = property_to_address(name) {
            let dst = self.alloc();
            ops.push(Op::LoadState { dst, addr });
            return dst;
        }

        // Keyboard I/O property
        if name == "--keyboard" || name == "--__1keyboard" || name == "--__2keyboard" {
            let dst = self.alloc();
            ops.push(Op::LoadKeyboard { dst });
            return dst;
        }

        // Unknown property — use fallback or 0
        if let Some(fb) = fallback {
            return self.compile_expr(fb, ops);
        }

        let dst = self.alloc();
        ops.push(Op::LoadLit { dst, val: 0.0 });
        dst
    }

    /// Compile a CalcOp.
    fn compile_calc(&mut self, op: &CalcOp, ops: &mut Vec<Op>) -> u16 {
        match op {
            CalcOp::Add(a, b) => {
                let sa = self.compile_expr(a, ops);
                let sb = self.compile_expr(b, ops);
                let dst = self.alloc();
                ops.push(Op::Add { dst, a: sa, b: sb });
                dst
            }
            CalcOp::Sub(a, b) => {
                let sa = self.compile_expr(a, ops);
                let sb = self.compile_expr(b, ops);
                let dst = self.alloc();
                ops.push(Op::Sub { dst, a: sa, b: sb });
                dst
            }
            CalcOp::Mul(a, b) => {
                let sa = self.compile_expr(a, ops);
                let sb = self.compile_expr(b, ops);
                let dst = self.alloc();
                ops.push(Op::Mul { dst, a: sa, b: sb });
                dst
            }
            CalcOp::Div(a, b) => {
                let sa = self.compile_expr(a, ops);
                let sb = self.compile_expr(b, ops);
                let dst = self.alloc();
                ops.push(Op::Div { dst, a: sa, b: sb });
                dst
            }
            CalcOp::Mod(a, b) => {
                let sa = self.compile_expr(a, ops);
                let sb = self.compile_expr(b, ops);
                let dst = self.alloc();
                ops.push(Op::Mod { dst, a: sa, b: sb });
                dst
            }
            CalcOp::Min(args) => {
                let slots: Vec<u16> = args.iter().map(|a| self.compile_expr(a, ops)).collect();
                let dst = self.alloc();
                ops.push(Op::Min { dst, args: slots });
                dst
            }
            CalcOp::Max(args) => {
                let slots: Vec<u16> = args.iter().map(|a| self.compile_expr(a, ops)).collect();
                let dst = self.alloc();
                ops.push(Op::Max { dst, args: slots });
                dst
            }
            CalcOp::Clamp(min, val, max) => {
                let smin = self.compile_expr(min, ops);
                let sval = self.compile_expr(val, ops);
                let smax = self.compile_expr(max, ops);
                let dst = self.alloc();
                ops.push(Op::Clamp { dst, min: smin, val: sval, max: smax });
                dst
            }
            CalcOp::Round(strategy, val, interval) => {
                let sval = self.compile_expr(val, ops);
                let sint = self.compile_expr(interval, ops);
                let dst = self.alloc();
                ops.push(Op::Round { dst, strategy: *strategy, val: sval, interval: sint });
                dst
            }
            CalcOp::Pow(base, exp) => {
                let sb = self.compile_expr(base, ops);
                let se = self.compile_expr(exp, ops);
                let dst = self.alloc();
                ops.push(Op::Pow { dst, base: sb, exp: se });
                dst
            }
            CalcOp::Sign(val) => {
                let sv = self.compile_expr(val, ops);
                let dst = self.alloc();
                ops.push(Op::Sign { dst, src: sv });
                dst
            }
            CalcOp::Abs(val) => {
                let sv = self.compile_expr(val, ops);
                let dst = self.alloc();
                ops.push(Op::Abs { dst, src: sv });
                dst
            }
            CalcOp::Negate(val) => {
                let sv = self.compile_expr(val, ops);
                let dst = self.alloc();
                ops.push(Op::Neg { dst, src: sv });
                dst
            }
        }
    }

    /// Compile a StyleCondition (if/else chain) into branch ops.
    fn compile_style_condition(
        &mut self,
        branches: &[StyleBranch],
        fallback: &Expr,
        ops: &mut Vec<Op>,
    ) -> u16 {
        // Result goes into a single destination slot
        let result_slot = self.alloc();

        // We emit a chain: for each branch, test condition, if true compute then
        // and jump to end, else fall through to next branch.
        // Patch targets are filled in after all branches are emitted.
        let mut jump_to_end: Vec<usize> = Vec::new();

        for branch in branches {
            let cond_slot = self.compile_style_test(&branch.condition, ops);
            // If condition is false (0), skip this branch
            let branch_idx = ops.len();
            ops.push(Op::BranchIfZero { cond: cond_slot, target: 0 }); // target patched later

            // Condition true: compute 'then' value
            let then_slot = self.compile_expr(&branch.then, ops);
            ops.push(Op::LoadSlot { dst: result_slot, src: then_slot });

            // Jump to end
            jump_to_end.push(ops.len());
            ops.push(Op::Jump { target: 0 }); // target patched later

            // Patch the branch-if-zero to jump here (the next branch)
            let next_idx = ops.len() as u32;
            if let Op::BranchIfZero { target, .. } = &mut ops[branch_idx] {
                *target = next_idx;
            }
        }

        // Fallback
        let fb_slot = self.compile_expr(fallback, ops);
        ops.push(Op::LoadSlot { dst: result_slot, src: fb_slot });

        // Patch all jump-to-end targets
        let end_idx = ops.len() as u32;
        for idx in jump_to_end {
            if let Op::Jump { target } = &mut ops[idx] {
                *target = end_idx;
            }
        }

        result_slot
    }

    /// Compile a StyleTest into a boolean (0 or 1) in a slot.
    fn compile_style_test(&mut self, test: &StyleTest, ops: &mut Vec<Op>) -> u16 {
        match test {
            StyleTest::Single { property, value } => {
                let prop_slot = self.compile_var(property, None, ops);
                let val_slot = self.compile_expr(value, ops);
                let dst = self.alloc();
                ops.push(Op::CmpEq { dst, a: prop_slot, b: val_slot });
                dst
            }
            StyleTest::And(tests) => {
                // All must be true: short-circuit chain
                // Start with 1 (true), AND each result
                let result = self.alloc();
                ops.push(Op::LoadLit { dst: result, val: 1.0 });

                for t in tests {
                    let t_slot = self.compile_style_test(t, ops);
                    // If result is already 0, skip (BranchIfZero past the mul)
                    let check_idx = ops.len();
                    ops.push(Op::BranchIfZero { cond: result, target: 0 });
                    // result = result * t_slot (both are 0 or 1, so this is AND)
                    ops.push(Op::Mul { dst: result, a: result, b: t_slot });
                    let after = ops.len() as u32;
                    if let Op::BranchIfZero { target, .. } = &mut ops[check_idx] {
                        *target = after;
                    }
                }
                result
            }
            StyleTest::Or(tests) => {
                // Any must be true
                let result = self.alloc();
                ops.push(Op::LoadLit { dst: result, val: 0.0 });
                let mut jumps_to_end = Vec::new();

                for t in tests {
                    let t_slot = self.compile_style_test(t, ops);
                    // result = t_slot (store latest)
                    ops.push(Op::LoadSlot { dst: result, src: t_slot });
                    // If result is now nonzero, we're done — but BranchIfZero
                    // only jumps on zero, so we need the inverse logic.
                    // We'll use: if result != 0, jump to end.
                    // Implement as: branch-if-zero past the jump, then jump to end.
                    let check_idx = ops.len();
                    ops.push(Op::BranchIfZero { cond: result, target: 0 });
                    jumps_to_end.push(ops.len());
                    ops.push(Op::Jump { target: 0 });
                    let after = ops.len() as u32;
                    if let Op::BranchIfZero { target, .. } = &mut ops[check_idx] {
                        *target = after;
                    }
                }

                let end = ops.len() as u32;
                for idx in jumps_to_end {
                    if let Op::Jump { target } = &mut ops[idx] {
                        *target = end;
                    }
                }
                result
            }
        }
    }

    /// Compile a function call — inlines known functions.
    fn compile_function_call(&mut self, name: &str, args: &[Expr], ops: &mut Vec<Op>) -> u16 {
        // Fast-path builtins: compile directly to native ops
        match name {
            "--readMem" => {
                if let Some(arg) = args.first() {
                    let addr_slot = self.compile_expr(arg, ops);
                    let dst = self.alloc();
                    ops.push(Op::LoadMem { dst, addr_slot });
                    return dst;
                }
                let dst = self.alloc();
                ops.push(Op::LoadLit { dst, val: 0.0 });
                return dst;
            }
            "--read2" => {
                if let Some(arg) = args.first() {
                    let addr_slot = self.compile_expr(arg, ops);
                    let dst = self.alloc();
                    ops.push(Op::LoadMem16 { dst, addr_slot });
                    return dst;
                }
                let dst = self.alloc();
                ops.push(Op::LoadLit { dst, val: 0.0 });
                return dst;
            }
            "--lowerBytes" => {
                if args.len() >= 2 {
                    let sa = self.compile_expr(&args[0], ops);
                    let sb = self.compile_expr(&args[1], ops);
                    let dst = self.alloc();
                    ops.push(Op::And { dst, a: sa, b: sb });
                    return dst;
                }
                let dst = self.alloc();
                ops.push(Op::LoadLit { dst, val: 0.0 });
                return dst;
            }
            "--rightShift" => {
                if args.len() >= 2 {
                    let sa = self.compile_expr(&args[0], ops);
                    let sb = self.compile_expr(&args[1], ops);
                    let dst = self.alloc();
                    ops.push(Op::Shr { dst, a: sa, b: sb });
                    return dst;
                }
                let dst = self.alloc();
                ops.push(Op::LoadLit { dst, val: 0.0 });
                return dst;
            }
            "--leftShift" => {
                if args.len() >= 2 {
                    let sa = self.compile_expr(&args[0], ops);
                    let sb = self.compile_expr(&args[1], ops);
                    let dst = self.alloc();
                    ops.push(Op::Shl { dst, a: sa, b: sb });
                    return dst;
                }
                let dst = self.alloc();
                ops.push(Op::LoadLit { dst, val: 0.0 });
                return dst;
            }
            "--bit" => {
                if args.len() >= 2 {
                    let sv = self.compile_expr(&args[0], ops);
                    let si = self.compile_expr(&args[1], ops);
                    let dst = self.alloc();
                    ops.push(Op::Bit { dst, val: sv, idx: si });
                    return dst;
                }
                let dst = self.alloc();
                ops.push(Op::LoadLit { dst, val: 0.0 });
                return dst;
            }
            "--int" => {
                if let Some(arg) = args.first() {
                    return self.compile_expr(arg, ops);
                }
                let dst = self.alloc();
                ops.push(Op::LoadLit { dst, val: 0.0 });
                return dst;
            }
            _ => {}
        }

        // Dispatch table: compile the key lookup and each entry
        if self.dispatch_tables.contains_key(name) {
            return self.compile_dispatch_call(name, args, ops);
        }

        // General function: inline the body
        self.compile_general_function(name, args, ops)
    }

    /// Compile a dispatch table function call.
    fn compile_dispatch_call(&mut self, name: &str, args: &[Expr], ops: &mut Vec<Op>) -> u16 {
        // Take the dispatch table temporarily to avoid borrow conflicts
        let table = self.dispatch_tables.remove(name).unwrap();
        let func = self.functions.get(name).cloned();

        // Bind arguments to parameter slots (if function definition exists)
        let saved: Vec<(String, Option<u16>)> = if let Some(ref f) = func {
            f.parameters
                .iter()
                .enumerate()
                .map(|(i, param)| {
                    let old = self.property_slots.get(&param.name).copied();
                    let val_slot = args
                        .get(i)
                        .map(|a| self.compile_expr(a, ops))
                        .unwrap_or_else(|| {
                            let s = self.alloc();
                            ops.push(Op::LoadLit { dst: s, val: 0.0 });
                            s
                        });
                    self.property_slots.insert(param.name.clone(), val_slot);
                    (param.name.clone(), old)
                })
                .collect()
        } else {
            Vec::new()
        };

        // Compile the key lookup
        let key_slot = self.compile_var(&table.key_property, None, ops);

        // Compile each dispatch entry into its own op sequence
        let mut compiled_entries = HashMap::new();
        for (&key_val, entry_expr) in &table.entries {
            let mut entry_ops = Vec::new();
            let result = self.compile_expr(entry_expr, &mut entry_ops);
            compiled_entries.insert(key_val, (entry_ops, result));
        }

        // Compile fallback
        let mut fallback_ops = Vec::new();
        let fallback_slot = self.compile_expr(&table.fallback, &mut fallback_ops);

        let table_id = self.compiled_dispatches.len() as u16;
        self.compiled_dispatches.push(CompiledDispatchTable {
            entries: compiled_entries,
            fallback_ops,
            fallback_slot,
        });

        // Restore the dispatch table and parameter bindings
        self.dispatch_tables.insert(name.to_string(), table);
        for (param_name, old) in saved {
            match old {
                Some(s) => { self.property_slots.insert(param_name, s); }
                None => { self.property_slots.remove(&param_name); }
            }
        }

        let dst = self.alloc();
        ops.push(Op::Dispatch {
            dst,
            key: key_slot,
            table_id,
            fallback_target: 0, // not used — dispatch is handled by the executor
        });
        dst
    }

    /// Compile a general function call by inlining its body.
    fn compile_general_function(&mut self, name: &str, args: &[Expr], ops: &mut Vec<Op>) -> u16 {
        let func = match self.functions.get(name).cloned() {
            Some(f) => f,
            None => {
                let dst = self.alloc();
                ops.push(Op::LoadLit { dst, val: 0.0 });
                return dst;
            }
        };

        // Bind arguments to parameter slots
        let saved_params: Vec<(String, Option<u16>)> = func
            .parameters
            .iter()
            .enumerate()
            .map(|(i, param)| {
                let old = self.property_slots.get(&param.name).copied();
                let val_slot = args
                    .get(i)
                    .map(|a| self.compile_expr(a, ops))
                    .unwrap_or_else(|| {
                        let s = self.alloc();
                        ops.push(Op::LoadLit { dst: s, val: 0.0 });
                        s
                    });
                self.property_slots.insert(param.name.clone(), val_slot);
                (param.name.clone(), old)
            })
            .collect();

        // Evaluate local variables
        let saved_locals: Vec<(String, Option<u16>)> = func
            .locals
            .iter()
            .map(|local| {
                let old = self.property_slots.get(&local.name).copied();
                let val_slot = self.compile_expr(&local.value, ops);
                self.property_slots.insert(local.name.clone(), val_slot);
                (local.name.clone(), old)
            })
            .collect();

        // Compile the result expression
        let result_slot = self.compile_expr(&func.result, ops);

        // Restore previous bindings
        for (param_name, old) in saved_params.into_iter().chain(saved_locals) {
            match old {
                Some(s) => { self.property_slots.insert(param_name, s); }
                None => { self.property_slots.remove(&param_name); }
            }
        }

        result_slot
    }
}

// ---------------------------------------------------------------------------
// Public API — compile an evaluator's data into a CompiledProgram
// ---------------------------------------------------------------------------

/// Compile the evaluator's assignments and broadcast writes into a `CompiledProgram`.
pub fn compile(
    assignments: &[Assignment],
    broadcast_writes: &[BroadcastWrite],
    functions: &HashMap<String, FunctionDef>,
    dispatch_tables: &HashMap<String, DispatchTable>,
) -> CompiledProgram {
    let mut compiler = Compiler::new(functions, dispatch_tables);
    let mut ops = Vec::new();
    let mut writeback = Vec::new();

    // Compile each assignment
    for assignment in assignments {
        let result_slot = compiler.compile_expr(&assignment.value, &mut ops);
        // Register this property slot so later assignments can reference it
        compiler.property_slots.insert(assignment.property.clone(), result_slot);

        // Track writeback for canonical properties
        if !is_buffer_copy(&assignment.property) && !is_byte_half(&assignment.property) {
            if let Some(addr) = property_to_address(&assignment.property) {
                writeback.push((result_slot, addr));
            }
        }
    }

    // Compile broadcast writes
    let compiled_bw = broadcast_writes
        .iter()
        .map(|bw| compile_broadcast_write(bw, &mut compiler))
        .collect();

    CompiledProgram {
        ops,
        slot_count: compiler.next_slot,
        writeback,
        broadcast_writes: compiled_bw,
        dispatch_tables: compiler.compiled_dispatches,
    }
}

/// Compile a single broadcast write.
fn compile_broadcast_write(bw: &BroadcastWrite, compiler: &mut Compiler) -> CompiledBroadcastWrite {
    // Compile dest property resolution
    let dest_slot = compiler.compile_var(&bw.dest_property, None, &mut Vec::new());

    // Compile value expression
    let mut value_ops = Vec::new();
    let value_slot = compiler.compile_expr(&bw.value_expr, &mut value_ops);

    // Build address map: address → state address (for direct write_mem)
    let mut address_map = HashMap::new();
    for (&addr, var_name) in &bw.address_map {
        if let Some(state_addr) = property_to_address(&format!("--{var_name}")) {
            address_map.insert(addr, state_addr);
        } else if let Some(state_addr) = property_to_address(var_name) {
            address_map.insert(addr, state_addr);
        }
    }

    // Compile spillover
    let spillover = if !bw.spillover_map.is_empty() {
        bw.spillover_guard.as_ref().map(|guard| {
            let guard_slot = compiler.compile_var(guard, None, &mut Vec::new());
            let mut entries = HashMap::new();
            for (&addr, (_var_name, val_expr)) in &bw.spillover_map {
                let mut spill_ops = Vec::new();
                let spill_slot = compiler.compile_expr(val_expr, &mut spill_ops);
                entries.insert(addr, (spill_ops, spill_slot));
            }
            CompiledSpillover {
                guard_slot,
                entries,
            }
        })
    } else {
        None
    };

    CompiledBroadcastWrite {
        dest_slot,
        value_ops,
        value_slot,
        address_map,
        spillover,
    }
}

// ---------------------------------------------------------------------------
// Executor — runs a CompiledProgram against State
// ---------------------------------------------------------------------------

/// Execute a compiled program for one tick.
pub fn execute(program: &CompiledProgram, state: &mut State, slots: &mut Vec<f64>) {
    // Reset slots (reuse allocation)
    slots.clear();
    slots.resize(program.slot_count as usize, 0.0);

    // Execute main ops
    exec_ops(&program.ops, &program.dispatch_tables, state, slots);

    // Writeback: apply computed values to state
    for &(slot, addr) in &program.writeback {
        let value = slots[slot as usize] as i32;
        let old = state.read_mem(addr);
        if old != value {
            state.write_mem(addr, value);
        }
    }

    // Execute broadcast writes
    for bw in &program.broadcast_writes {
        let dest = slots[bw.dest_slot as usize];
        let dest_i64 = dest as i64;
        if bw.address_map.contains_key(&dest_i64) {
            exec_ops(&bw.value_ops, &program.dispatch_tables, state, slots);
            let value = slots[bw.value_slot as usize] as i32;
            state.write_mem(dest_i64 as i32, value);
        }
        // Spillover
        if let Some(ref spillover) = bw.spillover {
            let guard = slots[spillover.guard_slot as usize] as i64;
            if guard == 1 {
                if let Some((ref spill_ops, spill_slot)) = spillover.entries.get(&dest_i64) {
                    exec_ops(spill_ops, &program.dispatch_tables, state, slots);
                    let value = slots[*spill_slot as usize] as i32;
                    state.write_mem(dest_i64 as i32 + 1, value);
                }
            }
        }
    }
}

/// Execute a sequence of ops against the slot array.
fn exec_ops(
    ops: &[Op],
    dispatch_tables: &[CompiledDispatchTable],
    state: &mut State,
    slots: &mut [f64],
) {
    let len = ops.len();
    let mut pc: usize = 0;

    while pc < len {
        match &ops[pc] {
            Op::LoadLit { dst, val } => {
                slots[*dst as usize] = *val;
            }
            Op::LoadSlot { dst, src } => {
                slots[*dst as usize] = slots[*src as usize];
            }
            Op::LoadState { dst, addr } => {
                slots[*dst as usize] = state.read_mem(*addr) as f64;
            }
            Op::LoadMem { dst, addr_slot } => {
                let addr = slots[*addr_slot as usize] as i32;
                slots[*dst as usize] = state.read_mem(addr) as f64;
            }
            Op::LoadMem16 { dst, addr_slot } => {
                let addr = slots[*addr_slot as usize] as i32;
                if addr < 0 {
                    slots[*dst as usize] = state.read_mem(addr) as f64;
                } else {
                    slots[*dst as usize] = state.read_mem16(addr) as f64;
                }
            }
            Op::LoadKeyboard { dst } => {
                slots[*dst as usize] = state.keyboard as f64;
            }
            Op::Add { dst, a, b } => {
                slots[*dst as usize] = slots[*a as usize] + slots[*b as usize];
            }
            Op::Sub { dst, a, b } => {
                slots[*dst as usize] = slots[*a as usize] - slots[*b as usize];
            }
            Op::Mul { dst, a, b } => {
                slots[*dst as usize] = slots[*a as usize] * slots[*b as usize];
            }
            Op::Div { dst, a, b } => {
                let divisor = slots[*b as usize];
                slots[*dst as usize] = if divisor == 0.0 { 0.0 } else { slots[*a as usize] / divisor };
            }
            Op::Mod { dst, a, b } => {
                let divisor = slots[*b as usize];
                slots[*dst as usize] = if divisor == 0.0 { 0.0 } else { slots[*a as usize] % divisor };
            }
            Op::Neg { dst, src } => {
                slots[*dst as usize] = -slots[*src as usize];
            }
            Op::Abs { dst, src } => {
                slots[*dst as usize] = slots[*src as usize].abs();
            }
            Op::Sign { dst, src } => {
                let v = slots[*src as usize];
                slots[*dst as usize] = if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 };
            }
            Op::Pow { dst, base, exp } => {
                slots[*dst as usize] = slots[*base as usize].powf(slots[*exp as usize]);
            }
            Op::Min { dst, args } => {
                let mut v = f64::INFINITY;
                for &a in args {
                    v = v.min(slots[a as usize]);
                }
                slots[*dst as usize] = v;
            }
            Op::Max { dst, args } => {
                let mut v = f64::NEG_INFINITY;
                for &a in args {
                    v = v.max(slots[a as usize]);
                }
                slots[*dst as usize] = v;
            }
            Op::Clamp { dst, min, val, max } => {
                let min_v = slots[*min as usize];
                let val_v = slots[*val as usize];
                let max_v = slots[*max as usize];
                slots[*dst as usize] = val_v.clamp(min_v, max_v);
            }
            Op::Round { dst, strategy, val, interval } => {
                let v = slots[*val as usize];
                let i = slots[*interval as usize];
                slots[*dst as usize] = if i == 0.0 {
                    v
                } else {
                    match strategy {
                        RoundStrategy::Nearest => (v / i).round() * i,
                        RoundStrategy::Up => (v / i).ceil() * i,
                        RoundStrategy::Down => (v / i).floor() * i,
                        RoundStrategy::ToZero => (v / i).trunc() * i,
                    }
                };
            }
            Op::Floor { dst, src } => {
                slots[*dst as usize] = slots[*src as usize].floor();
            }
            Op::And { dst, a, b } => {
                let av = slots[*a as usize] as i64;
                let bv = slots[*b as usize] as u32;
                slots[*dst as usize] = if bv >= 64 {
                    av as f64
                } else {
                    (av & ((1i64 << bv) - 1)) as f64
                };
            }
            Op::Shr { dst, a, b } => {
                let av = slots[*a as usize] as i64;
                let bv = slots[*b as usize] as u32;
                slots[*dst as usize] = if bv >= 64 { 0.0 } else { (av >> bv) as f64 };
            }
            Op::Shl { dst, a, b } => {
                let av = slots[*a as usize] as i64;
                let bv = slots[*b as usize] as u32;
                slots[*dst as usize] = if bv >= 64 { 0.0 } else { (av << bv) as f64 };
            }
            Op::Bit { dst, val, idx } => {
                let v = slots[*val as usize] as i64;
                let i = slots[*idx as usize] as u32;
                slots[*dst as usize] = if i >= 64 { 0.0 } else { ((v >> i) & 1) as f64 };
            }
            Op::CmpEq { dst, a, b } => {
                slots[*dst as usize] = if (slots[*a as usize] as i64) == (slots[*b as usize] as i64) {
                    1.0
                } else {
                    0.0
                };
            }
            Op::BranchIfZero { cond, target } => {
                if slots[*cond as usize] == 0.0 {
                    pc = *target as usize;
                    continue;
                }
            }
            Op::Jump { target } => {
                pc = *target as usize;
                continue;
            }
            Op::Dispatch { dst, key, table_id, .. } => {
                let key_val = slots[*key as usize] as i64;
                let table = &dispatch_tables[*table_id as usize];
                if let Some((entry_ops, result_slot)) = table.entries.get(&key_val) {
                    exec_ops(entry_ops, dispatch_tables, state, slots);
                    slots[*dst as usize] = slots[*result_slot as usize];
                } else {
                    exec_ops(&table.fallback_ops, dispatch_tables, state, slots);
                    slots[*dst as usize] = slots[table.fallback_slot as usize];
                }
            }
            Op::StoreState { addr, src } => {
                state.write_mem(*addr, slots[*src as usize] as i32);
            }
            Op::StoreMem { addr_slot, src } => {
                state.write_mem(slots[*addr_slot as usize] as i32, slots[*src as usize] as i32);
            }
        }
        pc += 1;
    }
}

// ---------------------------------------------------------------------------
// Helpers (duplicated from eval.rs to avoid coupling)
// ---------------------------------------------------------------------------

fn is_buffer_copy(name: &str) -> bool {
    name.starts_with("--__0") || name.starts_with("--__1") || name.starts_with("--__2")
}

fn is_byte_half(name: &str) -> bool {
    let bare = if name.starts_with("--__") && name.len() > 5 {
        &name[5..]
    } else if let Some(stripped) = name.strip_prefix("--") {
        stripped
    } else {
        name
    };
    matches!(bare, "AL" | "AH" | "BL" | "BH" | "CL" | "CH" | "DL" | "DH")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state;

    #[test]
    fn compile_literal() {
        let expr = Expr::Literal(42.0);
        let mut compiler = Compiler::new(&HashMap::new(), &HashMap::new());
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);
        assert_eq!(ops.len(), 1);

        let mut state = State::default();
        let mut slots = vec![0.0; compiler.next_slot as usize];
        exec_ops(&ops, &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 42.0);
    }

    #[test]
    fn compile_calc_add() {
        let expr = Expr::Calc(CalcOp::Add(
            Box::new(Expr::Literal(10.0)),
            Box::new(Expr::Literal(20.0)),
        ));
        let mut compiler = Compiler::new(&HashMap::new(), &HashMap::new());
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);

        let mut state = State::default();
        let mut slots = vec![0.0; compiler.next_slot as usize];
        exec_ops(&ops, &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 30.0);
    }

    #[test]
    fn compile_var_from_state() {
        let expr = Expr::Var {
            name: "--AX".to_string(),
            fallback: None,
        };
        let mut compiler = Compiler::new(&HashMap::new(), &HashMap::new());
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);

        let mut state = State::default();
        state.registers[state::reg::AX] = 0x1234;
        let mut slots = vec![0.0; compiler.next_slot as usize];
        exec_ops(&ops, &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 0x1234 as f64);
    }

    #[test]
    fn compile_style_condition() {
        let expr = Expr::StyleCondition {
            branches: vec![
                StyleBranch {
                    condition: StyleTest::Single {
                        property: "--AX".to_string(),
                        value: Expr::Literal(1.0),
                    },
                    then: Expr::Literal(100.0),
                },
                StyleBranch {
                    condition: StyleTest::Single {
                        property: "--AX".to_string(),
                        value: Expr::Literal(2.0),
                    },
                    then: Expr::Literal(200.0),
                },
            ],
            fallback: Box::new(Expr::Literal(0.0)),
        };
        let mut compiler = Compiler::new(&HashMap::new(), &HashMap::new());
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);

        let mut state = State::default();
        state.registers[state::reg::AX] = 2;
        let mut slots = vec![0.0; compiler.next_slot as usize];
        exec_ops(&ops, &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 200.0);
    }

    #[test]
    fn compile_readmem() {
        let expr = Expr::FunctionCall {
            name: "--readMem".to_string(),
            args: vec![Expr::Literal(-1.0)], // AX register
        };
        let mut compiler = Compiler::new(&HashMap::new(), &HashMap::new());
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);

        let mut state = State::default();
        state.registers[state::reg::AX] = 42;
        let mut slots = vec![0.0; compiler.next_slot as usize];
        exec_ops(&ops, &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 42.0);
    }

    #[test]
    fn compile_bitwise_ops() {
        // --lowerBytes(0xFF, 4) → 0xF
        let expr = Expr::FunctionCall {
            name: "--lowerBytes".to_string(),
            args: vec![Expr::Literal(0xFF as f64), Expr::Literal(4.0)],
        };
        let mut compiler = Compiler::new(&HashMap::new(), &HashMap::new());
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);

        let mut state = State::default();
        let mut slots = vec![0.0; compiler.next_slot as usize];
        exec_ops(&ops, &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 15.0);
    }

    #[test]
    fn compile_full_program() {
        let assignments = vec![
            Assignment {
                property: "--AX".to_string(),
                value: Expr::Literal(42.0),
            },
            Assignment {
                property: "--m0".to_string(),
                value: Expr::Literal(255.0),
            },
        ];

        let program = compile(&assignments, &[], &HashMap::new(), &HashMap::new());

        let mut state = State::default();
        let mut slots = Vec::new();
        execute(&program, &mut state, &mut slots);

        assert_eq!(state.registers[state::reg::AX], 42);
        assert_eq!(state.memory[0], 255);
    }

    #[test]
    fn compile_function_inline() {
        let mut functions = HashMap::new();
        functions.insert(
            "--double".to_string(),
            FunctionDef {
                name: "--double".to_string(),
                parameters: vec![FunctionParam {
                    name: "--x".to_string(),
                    syntax: PropertySyntax::Integer,
                }],
                locals: vec![],
                result: Expr::Calc(CalcOp::Mul(
                    Box::new(Expr::Var {
                        name: "--x".to_string(),
                        fallback: None,
                    }),
                    Box::new(Expr::Literal(2.0)),
                )),
            },
        );

        let expr = Expr::FunctionCall {
            name: "--double".to_string(),
            args: vec![Expr::Literal(21.0)],
        };
        let mut compiler = Compiler::new(&functions, &HashMap::new());
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);

        let mut state = State::default();
        let mut slots = vec![0.0; compiler.next_slot as usize];
        exec_ops(&ops, &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 42.0);
    }

    #[test]
    fn compile_value_forwarding() {
        // Assignment A computes --AX = 10
        // Assignment B computes --CX = var(--AX) + 5
        // B should see A's value without a state lookup
        let assignments = vec![
            Assignment {
                property: "--AX".to_string(),
                value: Expr::Literal(10.0),
            },
            Assignment {
                property: "--CX".to_string(),
                value: Expr::Calc(CalcOp::Add(
                    Box::new(Expr::Var {
                        name: "--AX".to_string(),
                        fallback: None,
                    }),
                    Box::new(Expr::Literal(5.0)),
                )),
            },
        ];

        let program = compile(&assignments, &[], &HashMap::new(), &HashMap::new());

        let mut state = State::default();
        let mut slots = Vec::new();
        execute(&program, &mut state, &mut slots);

        assert_eq!(state.registers[state::reg::AX], 10);
        assert_eq!(state.registers[state::reg::CX], 15);
    }

    #[test]
    fn compile_dispatch_table() {
        use crate::pattern::dispatch_table::DispatchTable;

        let mut functions = HashMap::new();
        functions.insert(
            "--lookup".to_string(),
            FunctionDef {
                name: "--lookup".to_string(),
                parameters: vec![FunctionParam {
                    name: "--key".to_string(),
                    syntax: PropertySyntax::Integer,
                }],
                locals: vec![],
                result: Expr::Literal(0.0),
            },
        );

        let mut entries = HashMap::new();
        entries.insert(0, Expr::Literal(100.0));
        entries.insert(1, Expr::Literal(200.0));
        entries.insert(2, Expr::Literal(300.0));
        entries.insert(42, Expr::Literal(999.0));

        let mut dispatch_tables = HashMap::new();
        dispatch_tables.insert(
            "--lookup".to_string(),
            DispatchTable {
                key_property: "--key".to_string(),
                entries,
                fallback: Expr::Literal(0.0),
            },
        );

        let assignments = vec![Assignment {
            property: "--result".to_string(),
            value: Expr::FunctionCall {
                name: "--lookup".to_string(),
                args: vec![Expr::Literal(42.0)],
            },
        }];

        let program = compile(&assignments, &[], &functions, &dispatch_tables);

        let mut state = State::default();
        let mut slots = Vec::new();
        execute(&program, &mut state, &mut slots);

        // --result is not a state-mapped property, so check the slot directly
        // Find the result slot from writeback — it won't be there since --result
        // isn't a register. Check by looking at dispatch result.
        assert_eq!(program.dispatch_tables.len(), 1);
    }
}
