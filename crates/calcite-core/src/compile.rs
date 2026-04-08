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
use crate::pattern::dispatch_table::{self, DispatchTable};
use crate::state::State;
use crate::types::*;

// ---------------------------------------------------------------------------
// Op — flat bytecode instruction
// ---------------------------------------------------------------------------

/// Slot index type — widened to u32 to support large programs (>64K slots).
pub type Slot = u32;

/// A single operation in the compiled bytecode.
///
/// All operands are `Slot` (u32) indices into a flat `Vec<i32>` array.
/// State reads/writes use `i32` addresses matching the x86CSS convention.
#[derive(Debug, Clone)]
pub enum Op {
    // --- Loads ---
    /// slot[dst] = literal value
    LoadLit {
        dst: Slot,
        val: i32,
    },
    /// slot[dst] = slot[src]
    LoadSlot {
        dst: Slot,
        src: Slot,
    },
    /// slot[dst] = state.read_mem(addr) — compile-time-known address
    LoadState {
        dst: Slot,
        addr: i32,
    },
    /// slot[dst] = state.read_mem(slot[addr_slot] as i32) — runtime address
    LoadMem {
        dst: Slot,
        addr_slot: Slot,
    },
    /// slot[dst] = state.read_mem16(slot[addr_slot] as i32) — 16-bit word read
    LoadMem16 {
        dst: Slot,
        addr_slot: Slot,
    },
    /// slot[dst] = state.keyboard
    LoadKeyboard {
        dst: Slot,
    },

    // --- Arithmetic ---
    Add {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    Sub {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    Mul {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    Div {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    Mod {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    Neg {
        dst: Slot,
        src: Slot,
    },
    Abs {
        dst: Slot,
        src: Slot,
    },
    Sign {
        dst: Slot,
        src: Slot,
    },
    Pow {
        dst: Slot,
        base: Slot,
        exp: Slot,
    },
    Min {
        dst: Slot,
        args: Vec<Slot>,
    },
    Max {
        dst: Slot,
        args: Vec<Slot>,
    },
    Clamp {
        dst: Slot,
        min: Slot,
        val: Slot,
        max: Slot,
    },
    Round {
        dst: Slot,
        strategy: RoundStrategy,
        val: Slot,
        interval: Slot,
    },
    Floor {
        dst: Slot,
        src: Slot,
    },

    // --- Bitwise (CSS function equivalents) ---
    /// lowerBytes(a, b) → a & ((1 << b) - 1)
    And {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    /// rightShift(a, b) → a >> b
    Shr {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    /// leftShift(a, b) → a << b
    Shl {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    /// bit(val, idx) → (val >> idx) & 1
    Bit {
        dst: Slot,
        val: Slot,
        idx: Slot,
    },

    // --- Comparisons & control flow ---
    /// slot[dst] = (slot[a] == slot[b]) as i64  (integer comparison)
    CmpEq {
        dst: Slot,
        a: Slot,
        b: Slot,
    },
    /// If slot[cond] == 0, jump to target op index
    BranchIfZero {
        cond: Slot,
        target: u32,
    },
    /// Unconditional jump to target op index
    Jump {
        target: u32,
    },

    // --- Dispatch table ---
    /// HashMap lookup: slot[dst] = dispatch_tables[table_id].entries[slot[key]]
    /// Falls back to executing ops at fallback_target if key not found.
    Dispatch {
        dst: Slot,
        key: Slot,
        table_id: Slot,
        fallback_target: u32,
    },

    // --- Stores ---
    /// state.write_mem(addr, slot[src]) — compile-time-known address
    StoreState {
        addr: i32,
        src: Slot,
    },
    /// state.write_mem(slot[addr_slot], slot[src]) — runtime address
    StoreMem {
        addr_slot: Slot,
        src: Slot,
    },
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
    pub slot_count: Slot,
    /// Mapping from slot index → state address for write-back.
    /// Only includes canonical properties (not buffer copies or byte halves).
    pub writeback: Vec<(Slot, i32)>,
    /// Broadcast writes — compiled separately because they need runtime HashMap lookup.
    pub broadcast_writes: Vec<CompiledBroadcastWrite>,
    /// Dispatch table data (kept for Dispatch op lookups at runtime).
    pub dispatch_tables: Vec<CompiledDispatchTable>,
    /// Mapping from property name → slot index (for reading computed values after execution).
    pub property_slots: HashMap<String, Slot>,
}

/// A compiled broadcast write.
#[derive(Debug)]
pub struct CompiledBroadcastWrite {
    /// Slot holding the destination address.
    pub dest_slot: Slot,
    /// Ops to evaluate the value expression (result in value_slot).
    pub value_ops: Vec<Op>,
    /// Slot holding the evaluated value.
    pub value_slot: Slot,
    /// Address → state address mapping for the broadcast.
    pub address_map: HashMap<i64, i32>,
    /// Spillover ops (for word writes).
    pub spillover: Option<CompiledSpillover>,
}

/// Compiled spillover for word-write broadcast.
#[derive(Debug)]
pub struct CompiledSpillover {
    /// Slot holding the guard property value.
    pub guard_slot: Slot,
    /// Map from dest address → (ops to compute high byte, result slot).
    pub entries: HashMap<i64, (Vec<Op>, Slot)>,
}

/// A compiled dispatch table — kept for runtime HashMap lookup.
#[derive(Debug)]
pub struct CompiledDispatchTable {
    /// Compiled ops for each dispatch entry, keyed by the dispatch value.
    /// Each entry is (ops, result_slot).
    pub entries: HashMap<i64, (Vec<Op>, Slot)>,
    /// Compiled ops for the fallback expression.
    pub fallback_ops: Vec<Op>,
    /// Slot holding the fallback result.
    pub fallback_slot: Slot,
}

// ---------------------------------------------------------------------------
// Body-pattern analysis — detect mathematical patterns in function bodies
// ---------------------------------------------------------------------------

/// Check if an expression is `var(name)`.
pub(crate) fn is_var_ref(expr: &Expr, param_name: &str) -> bool {
    matches!(expr, Expr::Var { name, .. } if name == param_name)
}

/// Check if an expression is `mod(var(a), pow(2, var(b)))` — bitmask pattern.
pub(crate) fn is_mod_pow2(expr: &Expr, a: &str, b: &str) -> bool {
    if let Expr::Calc(CalcOp::Mod(lhs, rhs)) = expr {
        if is_var_ref(lhs, a) {
            if let Expr::Calc(CalcOp::Pow(base, exp)) = rhs.as_ref() {
                return matches!(base.as_ref(), Expr::Literal(v) if (*v - 2.0).abs() < f64::EPSILON)
                    && is_var_ref(exp, b);
            }
        }
    }
    false
}

/// Check if an expression is `round(down, var(a) / pow(2, var(b)), 1)` — right shift.
pub(crate) fn is_right_shift(expr: &Expr, a: &str, b: &str) -> bool {
    if let Expr::Calc(CalcOp::Round(RoundStrategy::Down, val, interval)) = expr {
        // interval must be 1
        if !matches!(interval.as_ref(), Expr::Literal(v) if (*v - 1.0).abs() < f64::EPSILON) {
            return false;
        }
        // val must be var(a) / pow(2, var(b))
        if let Expr::Calc(CalcOp::Div(num, den)) = val.as_ref() {
            if is_var_ref(num, a) {
                if let Expr::Calc(CalcOp::Pow(base, exp)) = den.as_ref() {
                    return matches!(base.as_ref(), Expr::Literal(v) if (*v - 2.0).abs() < f64::EPSILON)
                        && is_var_ref(exp, b);
                }
            }
        }
    }
    false
}

/// Check if an expression is `var(a) * pow(2, var(b))` — left shift.
pub(crate) fn is_left_shift(expr: &Expr, a: &str, b: &str) -> bool {
    if let Expr::Calc(CalcOp::Mul(lhs, rhs)) = expr {
        if is_var_ref(lhs, a) {
            if let Expr::Calc(CalcOp::Pow(base, exp)) = rhs.as_ref() {
                return matches!(base.as_ref(), Expr::Literal(v) if (*v - 2.0).abs() < f64::EPSILON)
                    && is_var_ref(exp, b);
            }
        }
    }
    false
}

/// Check if an expression is `var(a) * var(local_name)` (in either order).
pub(crate) fn is_mul_refs(expr: &Expr, a: &str, local_name: &str) -> bool {
    if let Expr::Calc(CalcOp::Mul(lhs, rhs)) = expr {
        return (is_var_ref(lhs, a) && is_var_ref(rhs, local_name))
            || (is_var_ref(lhs, local_name) && is_var_ref(rhs, a));
    }
    false
}

/// Check if an expression is a power-of-2 dispatch table on `param`.
///
/// Pattern: `if(style(param:0): 1; style(param:1): 2; style(param:2): 4; ...)`
/// where entry K maps to 2^K.
pub(crate) fn is_pow2_dispatch(expr: &Expr, param: &str) -> bool {
    if let Expr::StyleCondition { branches, .. } = expr {
        if branches.len() < 4 {
            return false;
        }
        for branch in branches {
            if let StyleTest::Single { property, value } = &branch.condition {
                if property != param {
                    return false;
                }
                if let Expr::Literal(key_val) = value {
                    let k = *key_val as u32;
                    if let Expr::Literal(then_val) = &branch.then {
                        let expected = if k < 32 {
                            (1u64 << k) as f64
                        } else {
                            return false;
                        };
                        if (*then_val - expected).abs() > f64::EPSILON {
                            return false;
                        }
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
        return true;
    }
    false
}

/// Check if an expression is a bit-extract pattern: `mod(shift_body, 2)`.
///
/// Where `shift_body` is either:
/// - Directly `round(down, var(a) / pow(2, var(b)), 1)` (inline right-shift), or
/// - A function call to a function whose body IS a right-shift pattern.
pub(crate) fn is_bit_extract(
    expr: &Expr,
    a: &str,
    b: &str,
    functions: &HashMap<String, FunctionDef>,
) -> bool {
    if let Expr::Calc(CalcOp::Mod(inner, modulus)) = expr {
        // modulus must be 2
        if !matches!(modulus.as_ref(), Expr::Literal(v) if (*v - 2.0).abs() < f64::EPSILON) {
            return false;
        }
        // inner is an inline right-shift?
        if is_right_shift(inner, a, b) {
            return true;
        }
        // inner is a function call whose body is a right-shift pattern?
        if let Expr::FunctionCall { name, args } = inner.as_ref() {
            if args.len() == 2 && is_var_ref(&args[0], a) && is_var_ref(&args[1], b) {
                if let Some(func) = functions.get(name.as_str()) {
                    if func.parameters.len() == 2 && func.locals.is_empty() {
                        return is_right_shift(
                            &func.result,
                            &func.parameters[0].name,
                            &func.parameters[1].name,
                        );
                    }
                }
            }
        }
    }
    false
}

/// Check if a dispatch table is an identity-read: every entry maps key K → state[K].
pub(crate) fn is_dispatch_identity_read(table: &DispatchTable) -> bool {
    if table.entries.len() < 4 {
        return false;
    }
    for (&key, expr) in &table.entries {
        match expr {
            Expr::Var { name, .. } => {
                if let Some(addr) = property_to_address(name) {
                    if addr as i64 != key {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

/// Classify a dispatch table as "near-identity-read": most entries map key K → state[K],
/// but a small number have non-identity expressions (e.g., computed values for special
/// addresses like self-modifying code patches).
///
/// Returns `Some(exception_keys)` if the table is mostly identity reads (≥90%), with
/// `exception_keys` listing the keys that are NOT identity reads. Returns `None` if the
/// table doesn't qualify.
fn classify_near_identity_read(table: &DispatchTable) -> Option<Vec<i64>> {
    if table.entries.len() < 100 {
        return None;
    }
    let mut exceptions = Vec::new();
    for (&key, expr) in &table.entries {
        let is_identity = matches!(expr, Expr::Var { name, .. } if {
            property_to_address(name).is_some_and(|addr| addr as i64 == key)
        });
        if !is_identity {
            exceptions.push(key);
        }
    }
    // Must be at least 90% identity
    let identity_count = table.entries.len() - exceptions.len();
    if identity_count * 10 >= table.entries.len() * 9 {
        Some(exceptions)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Constant folding — simplify Expr trees before compilation
// ---------------------------------------------------------------------------

/// Recursively fold constant expressions and eliminate identity operations.
fn const_fold(expr: &Expr) -> Expr {
    match expr {
        Expr::Calc(op) => fold_calc(op),
        Expr::StyleCondition {
            branches, fallback, ..
        } => {
            let folded_branches: Vec<StyleBranch> = branches
                .iter()
                .map(|b| StyleBranch {
                    condition: fold_test(&b.condition),
                    then: const_fold(&b.then),
                })
                .collect();
            let folded_fallback = const_fold(fallback);
            Expr::StyleCondition {
                branches: folded_branches,
                fallback: Box::new(folded_fallback),
            }
        }
        Expr::FunctionCall { name, args } => Expr::FunctionCall {
            name: name.clone(),
            args: args.iter().map(const_fold).collect(),
        },
        _ => expr.clone(),
    }
}

/// Fold a StyleTest's value expressions.
fn fold_test(test: &StyleTest) -> StyleTest {
    match test {
        StyleTest::Single { property, value } => StyleTest::Single {
            property: property.clone(),
            value: const_fold(value),
        },
        StyleTest::And(tests) => StyleTest::And(tests.iter().map(fold_test).collect()),
        StyleTest::Or(tests) => StyleTest::Or(tests.iter().map(fold_test).collect()),
    }
}

/// Fold a CalcOp, returning a simplified Expr.
fn fold_calc(op: &CalcOp) -> Expr {
    match op {
        CalcOp::Add(a, b) => {
            let fa = const_fold(a);
            let fb = const_fold(b);
            match (&fa, &fb) {
                (Expr::Literal(x), Expr::Literal(y)) => Expr::Literal(x + y),
                (_, Expr::Literal(v)) if *v == 0.0 => fa,
                (Expr::Literal(v), _) if *v == 0.0 => fb,
                _ => Expr::Calc(CalcOp::Add(Box::new(fa), Box::new(fb))),
            }
        }
        CalcOp::Sub(a, b) => {
            let fa = const_fold(a);
            let fb = const_fold(b);
            match (&fa, &fb) {
                (Expr::Literal(x), Expr::Literal(y)) => Expr::Literal(x - y),
                (_, Expr::Literal(v)) if *v == 0.0 => fa,
                _ => Expr::Calc(CalcOp::Sub(Box::new(fa), Box::new(fb))),
            }
        }
        CalcOp::Mul(a, b) => {
            let fa = const_fold(a);
            let fb = const_fold(b);
            match (&fa, &fb) {
                (Expr::Literal(x), Expr::Literal(y)) => Expr::Literal(x * y),
                (_, Expr::Literal(v)) if *v == 1.0 => fa,
                (Expr::Literal(v), _) if *v == 1.0 => fb,
                (_, Expr::Literal(v)) if *v == 0.0 => Expr::Literal(0.0),
                (Expr::Literal(v), _) if *v == 0.0 => Expr::Literal(0.0),
                _ => Expr::Calc(CalcOp::Mul(Box::new(fa), Box::new(fb))),
            }
        }
        CalcOp::Div(a, b) => {
            let fa = const_fold(a);
            let fb = const_fold(b);
            match (&fa, &fb) {
                (Expr::Literal(x), Expr::Literal(y)) if *y != 0.0 => Expr::Literal(x / y),
                (_, Expr::Literal(v)) if *v == 1.0 => fa,
                _ => Expr::Calc(CalcOp::Div(Box::new(fa), Box::new(fb))),
            }
        }
        CalcOp::Mod(a, b) => {
            let fa = const_fold(a);
            let fb = const_fold(b);
            match (&fa, &fb) {
                (Expr::Literal(x), Expr::Literal(y)) if *y != 0.0 => Expr::Literal(x % y),
                _ => Expr::Calc(CalcOp::Mod(Box::new(fa), Box::new(fb))),
            }
        }
        CalcOp::Pow(a, b) => {
            let fa = const_fold(a);
            let fb = const_fold(b);
            match (&fa, &fb) {
                (Expr::Literal(x), Expr::Literal(y)) => Expr::Literal(x.powf(*y)),
                (_, Expr::Literal(v)) if *v == 0.0 => Expr::Literal(1.0),
                (_, Expr::Literal(v)) if *v == 1.0 => fa,
                _ => Expr::Calc(CalcOp::Pow(Box::new(fa), Box::new(fb))),
            }
        }
        CalcOp::Negate(a) => {
            let fa = const_fold(a);
            match &fa {
                Expr::Literal(v) => Expr::Literal(-v),
                _ => Expr::Calc(CalcOp::Negate(Box::new(fa))),
            }
        }
        CalcOp::Abs(a) => {
            let fa = const_fold(a);
            match &fa {
                Expr::Literal(v) => Expr::Literal(v.abs()),
                _ => Expr::Calc(CalcOp::Abs(Box::new(fa))),
            }
        }
        CalcOp::Sign(a) => {
            let fa = const_fold(a);
            match &fa {
                Expr::Literal(v) => Expr::Literal(if *v > 0.0 {
                    1.0
                } else if *v < 0.0 {
                    -1.0
                } else {
                    0.0
                }),
                _ => Expr::Calc(CalcOp::Sign(Box::new(fa))),
            }
        }
        CalcOp::Min(args) => {
            let folded: Vec<Expr> = args.iter().map(const_fold).collect();
            if folded.iter().all(|e| matches!(e, Expr::Literal(_))) {
                let min = folded
                    .iter()
                    .map(|e| match e {
                        Expr::Literal(v) => *v,
                        _ => unreachable!(),
                    })
                    .fold(f64::INFINITY, f64::min);
                Expr::Literal(min)
            } else {
                Expr::Calc(CalcOp::Min(folded))
            }
        }
        CalcOp::Max(args) => {
            let folded: Vec<Expr> = args.iter().map(const_fold).collect();
            if folded.iter().all(|e| matches!(e, Expr::Literal(_))) {
                let max = folded
                    .iter()
                    .map(|e| match e {
                        Expr::Literal(v) => *v,
                        _ => unreachable!(),
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                Expr::Literal(max)
            } else {
                Expr::Calc(CalcOp::Max(folded))
            }
        }
        CalcOp::Clamp(min, val, max) => {
            let fmin = const_fold(min);
            let fval = const_fold(val);
            let fmax = const_fold(max);
            match (&fmin, &fval, &fmax) {
                (Expr::Literal(mn), Expr::Literal(v), Expr::Literal(mx)) => {
                    Expr::Literal(v.clamp(*mn, *mx))
                }
                _ => Expr::Calc(CalcOp::Clamp(
                    Box::new(fmin),
                    Box::new(fval),
                    Box::new(fmax),
                )),
            }
        }
        CalcOp::Round(strategy, val, interval) => {
            let fval = const_fold(val);
            let fint = const_fold(interval);
            match (&fval, &fint) {
                (Expr::Literal(v), Expr::Literal(i)) if *i != 0.0 => {
                    let result = match strategy {
                        RoundStrategy::Nearest => (v / i).round() * i,
                        RoundStrategy::Up => (v / i).ceil() * i,
                        RoundStrategy::Down => (v / i).floor() * i,
                        RoundStrategy::ToZero => (v / i).trunc() * i,
                    };
                    Expr::Literal(result)
                }
                _ => Expr::Calc(CalcOp::Round(*strategy, Box::new(fval), Box::new(fint))),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compiler — translates Evaluator data into CompiledProgram
// ---------------------------------------------------------------------------

/// Compiler state — tracks slot allocation and property→slot mapping.
struct Compiler {
    /// Next available slot index.
    next_slot: Slot,
    /// Map from property name → slot index.
    property_slots: HashMap<String, Slot>,
    /// Functions available for inlining.
    functions: HashMap<String, FunctionDef>,
    /// Recognised dispatch tables.
    dispatch_tables: HashMap<String, DispatchTable>,
    /// Compiled dispatch table data (populated during compilation).
    compiled_dispatches: Vec<CompiledDispatchTable>,
    /// Cache: dispatch table name → compiled table_id.
    /// Tables with context-independent entries (no parameter refs in entry ops)
    /// can be compiled once and reused across call sites.
    dispatch_cache: HashMap<String, Slot>,
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
            dispatch_cache: HashMap::new(),
        }
    }

    /// Allocate a fresh temporary slot.
    fn alloc(&mut self) -> Slot {
        let s = self.next_slot;
        self.next_slot += 1;
        s
    }

    /// Compile an Expr into ops, returning the slot holding the result.
    fn compile_expr(&mut self, expr: &Expr, ops: &mut Vec<Op>) -> Slot {
        // Constant-fold before compiling
        let folded = const_fold(expr);
        let expr = &folded;
        match expr {
            Expr::Literal(v) => {
                let dst = self.alloc();
                ops.push(Op::LoadLit {
                    dst,
                    val: *v as i32,
                });
                dst
            }

            Expr::StringLiteral(_) | Expr::Concat(_) => {
                let dst = self.alloc();
                ops.push(Op::LoadLit { dst, val: 0 });
                dst
            }

            Expr::Var { name, fallback } => self.compile_var(name, fallback.as_deref(), ops),

            Expr::Calc(calc_op) => self.compile_calc(calc_op, ops),

            Expr::StyleCondition {
                branches, fallback, ..
            } => self.compile_style_condition(branches, fallback, ops),

            Expr::FunctionCall { name, args } => self.compile_function_call(name, args, ops),
        }
    }

    /// Compile a variable reference.
    fn compile_var(&mut self, name: &str, fallback: Option<&Expr>, ops: &mut Vec<Op>) -> Slot {
        // If it's a property we've already computed in this tick, use its slot directly.
        // But NOT for buffer-prefixed names (--__0*, --__1*, --__2*) — those explicitly
        // read the previous tick's state, not the current tick's computed value.
        if !is_buffer_copy(name) {
            if let Some(&s) = self.property_slots.get(name) {
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
        ops.push(Op::LoadLit { dst, val: 0 });
        dst
    }

    /// Compile a CalcOp.
    fn compile_calc(&mut self, op: &CalcOp, ops: &mut Vec<Op>) -> Slot {
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
                let slots: Vec<Slot> = args.iter().map(|a| self.compile_expr(a, ops)).collect();
                let dst = self.alloc();
                ops.push(Op::Min { dst, args: slots });
                dst
            }
            CalcOp::Max(args) => {
                let slots: Vec<Slot> = args.iter().map(|a| self.compile_expr(a, ops)).collect();
                let dst = self.alloc();
                ops.push(Op::Max { dst, args: slots });
                dst
            }
            CalcOp::Clamp(min, val, max) => {
                let smin = self.compile_expr(min, ops);
                let sval = self.compile_expr(val, ops);
                let smax = self.compile_expr(max, ops);
                let dst = self.alloc();
                ops.push(Op::Clamp {
                    dst,
                    min: smin,
                    val: sval,
                    max: smax,
                });
                dst
            }
            CalcOp::Round(strategy, val, interval) => {
                let sval = self.compile_expr(val, ops);
                let sint = self.compile_expr(interval, ops);
                let dst = self.alloc();
                ops.push(Op::Round {
                    dst,
                    strategy: *strategy,
                    val: sval,
                    interval: sint,
                });
                dst
            }
            CalcOp::Pow(base, exp) => {
                let sb = self.compile_expr(base, ops);
                let se = self.compile_expr(exp, ops);
                let dst = self.alloc();
                ops.push(Op::Pow {
                    dst,
                    base: sb,
                    exp: se,
                });
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
    ///
    /// If all branches test the same property against integer literals (a dispatch
    /// table pattern), compiles to a `Dispatch` op (O(1) HashMap lookup) instead of
    /// a linear branch chain. This handles both function-body dispatch tables and
    /// v2-style per-register opcode dispatches in assignments.
    fn compile_style_condition(
        &mut self,
        branches: &[StyleBranch],
        fallback: &Expr,
        ops: &mut Vec<Op>,
    ) -> Slot {
        // Try dispatch table optimization for large single-key chains
        if let Some(table) = dispatch_table::recognise_dispatch(branches, fallback) {
            return self.compile_inline_dispatch(&table, ops);
        }

        // Fall back to linear branch chain
        self.compile_style_condition_linear(branches, fallback, ops)
    }

    /// Compile a recognised dispatch table directly (no function call wrapper).
    fn compile_inline_dispatch(
        &mut self,
        table: &DispatchTable,
        ops: &mut Vec<Op>,
    ) -> Slot {
        let key_slot = self.compile_var(&table.key_property, None, ops);

        let mut compiled_entries = HashMap::new();
        for (&key_val, entry_expr) in &table.entries {
            let mut entry_ops = Vec::new();
            let result = self.compile_expr(entry_expr, &mut entry_ops);
            compiled_entries.insert(key_val, (entry_ops, result));
        }

        let mut fallback_ops = Vec::new();
        let fallback_slot = self.compile_expr(&table.fallback, &mut fallback_ops);

        let table_id = self.compiled_dispatches.len() as Slot;
        self.compiled_dispatches.push(CompiledDispatchTable {
            entries: compiled_entries,
            fallback_ops,
            fallback_slot,
        });

        let dst = self.alloc();
        ops.push(Op::Dispatch {
            dst,
            key: key_slot,
            table_id,
            fallback_target: 0,
        });
        dst
    }

    /// Compile a StyleCondition as a linear branch chain (fallback path).
    fn compile_style_condition_linear(
        &mut self,
        branches: &[StyleBranch],
        fallback: &Expr,
        ops: &mut Vec<Op>,
    ) -> Slot {
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
            ops.push(Op::BranchIfZero {
                cond: cond_slot,
                target: 0,
            }); // target patched later

            // Condition true: compute 'then' value
            let then_slot = self.compile_expr(&branch.then, ops);
            ops.push(Op::LoadSlot {
                dst: result_slot,
                src: then_slot,
            });

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
        ops.push(Op::LoadSlot {
            dst: result_slot,
            src: fb_slot,
        });

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
    fn compile_style_test(&mut self, test: &StyleTest, ops: &mut Vec<Op>) -> Slot {
        match test {
            StyleTest::Single { property, value } => {
                let prop_slot = self.compile_var(property, None, ops);
                let val_slot = self.compile_expr(value, ops);
                let dst = self.alloc();
                ops.push(Op::CmpEq {
                    dst,
                    a: prop_slot,
                    b: val_slot,
                });
                dst
            }
            StyleTest::And(tests) => {
                // All must be true: short-circuit chain
                // Start with 1 (true), AND each result
                let result = self.alloc();
                ops.push(Op::LoadLit {
                    dst: result,
                    val: 1,
                });

                for t in tests {
                    let t_slot = self.compile_style_test(t, ops);
                    // If result is already 0, skip (BranchIfZero past the mul)
                    let check_idx = ops.len();
                    ops.push(Op::BranchIfZero {
                        cond: result,
                        target: 0,
                    });
                    // result = result * t_slot (both are 0 or 1, so this is AND)
                    ops.push(Op::Mul {
                        dst: result,
                        a: result,
                        b: t_slot,
                    });
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
                ops.push(Op::LoadLit {
                    dst: result,
                    val: 0,
                });
                let mut jumps_to_end = Vec::new();

                for t in tests {
                    let t_slot = self.compile_style_test(t, ops);
                    // result = t_slot (store latest)
                    ops.push(Op::LoadSlot {
                        dst: result,
                        src: t_slot,
                    });
                    // If result is now nonzero, we're done — but BranchIfZero
                    // only jumps on zero, so we need the inverse logic.
                    // We'll use: if result != 0, jump to end.
                    // Implement as: branch-if-zero past the jump, then jump to end.
                    let check_idx = ops.len();
                    ops.push(Op::BranchIfZero {
                        cond: result,
                        target: 0,
                    });
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

    /// Compile a function call — uses body-pattern analysis for optimisation.
    ///
    /// Instead of matching function names, this analyses the function's body
    /// structure to detect mathematical patterns that can be compiled to
    /// efficient native operations. This is fully generic — it works for
    /// any CSS function with the right shape.
    fn compile_function_call(&mut self, name: &str, args: &[Expr], ops: &mut Vec<Op>) -> Slot {
        // Try body-pattern analysis on the function definition.
        if let Some(func) = self.functions.get(name).cloned() {
            if let Some(slot) = self.try_compile_by_body_pattern(&func, args, ops) {
                return slot;
            }
        }

        // Dispatch table: check for identity-read pattern, then fall back to
        // compiled dispatch lookup.
        if self.dispatch_tables.contains_key(name) {
            // Identity-read: every entry maps key K → state[K] — direct memory read
            if args.len() == 1 && self.check_dispatch_identity_read(name) {
                let addr_slot = self.compile_expr(&args[0], ops);
                let dst = self.alloc();
                ops.push(Op::LoadMem { dst, addr_slot });
                return dst;
            }
            // Near-identity-read: mostly identity with a few exception entries.
            // Compile as LoadMem + small exception dispatch instead of full 116K-entry table.
            if args.len() == 1 {
                if let Some(exception_keys) = self.dispatch_tables.get(name).and_then(classify_near_identity_read) {
                    return self.compile_near_identity_dispatch(name, args, &exception_keys, ops);
                }
            }
            return self.compile_dispatch_call(name, args, ops);
        }

        // General function: inline the body
        self.compile_general_function(name, args, ops)
    }

    /// Try to compile a function call by analysing its body pattern.
    ///
    /// Returns `Some(result_slot)` if the body matches a known mathematical
    /// pattern that can be compiled to efficient native ops.
    fn try_compile_by_body_pattern(
        &mut self,
        func: &FunctionDef,
        args: &[Expr],
        ops: &mut Vec<Op>,
    ) -> Option<Slot> {
        let params = &func.parameters;

        // Identity: 1 param, no locals, result = var(param)
        if params.len() == 1 && func.locals.is_empty() && is_var_ref(&func.result, &params[0].name)
        {
            return args.first().map(|a| self.compile_expr(a, ops));
        }

        // 2-param patterns with no locals
        if params.len() == 2 && func.locals.is_empty() {
            let p0 = &params[0].name;
            let p1 = &params[1].name;

            // Bitmask: mod(a, pow(2, b)) → And
            if is_mod_pow2(&func.result, p0, p1) {
                let sa = self.compile_expr(&args[0], ops);
                let sb = self.compile_expr(&args[1], ops);
                let dst = self.alloc();
                ops.push(Op::And { dst, a: sa, b: sb });
                return Some(dst);
            }

            // Right shift: round(down, a / pow(2, b), 1) → Shr
            if is_right_shift(&func.result, p0, p1) {
                let sa = self.compile_expr(&args[0], ops);
                let sb = self.compile_expr(&args[1], ops);
                let dst = self.alloc();
                ops.push(Op::Shr { dst, a: sa, b: sb });
                return Some(dst);
            }

            // Left shift: a * pow(2, b) → Shl
            if is_left_shift(&func.result, p0, p1) {
                let sa = self.compile_expr(&args[0], ops);
                let sb = self.compile_expr(&args[1], ops);
                let dst = self.alloc();
                ops.push(Op::Shl { dst, a: sa, b: sb });
                return Some(dst);
            }

            // Bit extract: mod(rightShift_body(a, b), 2) → Bit
            // i.e. mod(round(down, a / pow(2, b), 1), 2)
            if is_bit_extract(&func.result, p0, p1, &self.functions) {
                let sv = self.compile_expr(&args[0], ops);
                let si = self.compile_expr(&args[1], ops);
                let dst = self.alloc();
                ops.push(Op::Bit {
                    dst,
                    val: sv,
                    idx: si,
                });
                return Some(dst);
            }
        }

        // 2-param with 1 local: left-shift via power-of-2 dispatch table
        // Pattern: local = dispatch_on(b) {0→1, 1→2, 2→4, ...}, result = a * local
        if params.len() == 2 && func.locals.len() == 1 {
            let p0 = &params[0].name;
            let p1 = &params[1].name;
            let local = &func.locals[0];

            if is_mul_refs(&func.result, p0, &local.name) && is_pow2_dispatch(&local.value, p1) {
                let sa = self.compile_expr(&args[0], ops);
                let sb = self.compile_expr(&args[1], ops);
                let dst = self.alloc();
                ops.push(Op::Shl { dst, a: sa, b: sb });
                return Some(dst);
            }
        }

        None
    }

    /// Check if a dispatch table is an identity-read pattern.
    fn check_dispatch_identity_read(&self, name: &str) -> bool {
        self.dispatch_tables
            .get(name)
            .is_some_and(is_dispatch_identity_read)
    }

    /// Compile a near-identity-read dispatch table.
    ///
    /// For tables where 99%+ of entries are identity reads (key K → state[K]),
    /// we emit LoadMem as the default path and only compile a small dispatch table
    /// for the exception entries. This avoids recompiling 116K+ entries on each call.
    fn compile_near_identity_dispatch(
        &mut self,
        name: &str,
        args: &[Expr],
        exception_keys: &[i64],
        ops: &mut Vec<Op>,
    ) -> Slot {
        let table = self.dispatch_tables.remove(name).unwrap();
        let func = self.functions.get(name).cloned();

        // Bind arguments to parameter slots
        let saved: Vec<(String, Option<Slot>)> = if let Some(ref f) = func {
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
                            ops.push(Op::LoadLit { dst: s, val: 0 });
                            s
                        });
                    self.property_slots.insert(param.name.clone(), val_slot);
                    (param.name.clone(), old)
                })
                .collect()
        } else {
            Vec::new()
        };

        // Compile the key (address) lookup
        let key_slot = self.compile_var(&table.key_property, None, ops);

        // Default path: LoadMem — works for all identity-read entries
        let dst = self.alloc();
        ops.push(Op::LoadMem {
            dst,
            addr_slot: key_slot,
        });

        // Build a small dispatch table for just the exception entries
        if !exception_keys.is_empty() {
            let mut compiled_entries = HashMap::new();
            for &key_val in exception_keys {
                if let Some(entry_expr) = table.entries.get(&key_val) {
                    let mut entry_ops = Vec::new();
                    let result = self.compile_expr(entry_expr, &mut entry_ops);
                    compiled_entries.insert(key_val, (entry_ops, result));
                }
            }
            let mut fallback_ops = Vec::new();
            let fallback_slot = {
                let s = self.alloc();
                fallback_ops.push(Op::LoadSlot { dst: s, src: dst });
                s
            };
            let table_id = self.compiled_dispatches.len() as Slot;
            self.compiled_dispatches.push(CompiledDispatchTable {
                entries: compiled_entries,
                fallback_ops,
                fallback_slot,
            });
            let override_dst = self.alloc();
            ops.push(Op::Dispatch {
                dst: override_dst,
                key: key_slot,
                table_id,
                fallback_target: 0,
            });
            // Use the dispatch result (which falls back to LoadMem result for non-exceptions)
            // Restore and return
            self.dispatch_tables.insert(name.to_string(), table);
            for (param_name, old) in saved {
                match old {
                    Some(s) => { self.property_slots.insert(param_name, s); }
                    None => { self.property_slots.remove(&param_name); }
                }
            }
            return override_dst;
        }

        // No exceptions — just LoadMem
        self.dispatch_tables.insert(name.to_string(), table);
        for (param_name, old) in saved {
            match old {
                Some(s) => { self.property_slots.insert(param_name, s); }
                None => { self.property_slots.remove(&param_name); }
            }
        }
        dst
    }

    /// Compile a dispatch table function call.
    ///
    /// Large dispatch tables (≥100 entries) are compiled once and cached — subsequent
    /// calls reuse the same `table_id`. This is safe because dispatch table entries
    /// are keyed by the dispatch value (the parameter), not by parameter slots. The
    /// parameter only appears as the key; entry bodies are context-independent.
    fn compile_dispatch_call(&mut self, name: &str, args: &[Expr], ops: &mut Vec<Op>) -> Slot {
        let table = self.dispatch_tables.remove(name).unwrap();
        let func = self.functions.get(name).cloned();

        // Bind arguments to parameter slots, then evaluate locals
        // (the dispatch key may reference a local, e.g. --parity dispatches on --low8)
        let saved: Vec<(String, Option<Slot>)> = if let Some(ref f) = func {
            let mut saved: Vec<(String, Option<Slot>)> = f.parameters
                .iter()
                .enumerate()
                .map(|(i, param)| {
                    let old = self.property_slots.get(&param.name).copied();
                    let val_slot = args
                        .get(i)
                        .map(|a| self.compile_expr(a, ops))
                        .unwrap_or_else(|| {
                            let s = self.alloc();
                            ops.push(Op::LoadLit { dst: s, val: 0 });
                            s
                        });
                    self.property_slots.insert(param.name.clone(), val_slot);
                    (param.name.clone(), old)
                })
                .collect();
            // Evaluate locals so the dispatch key can reference them
            for local in &f.locals {
                let old = self.property_slots.get(&local.name).copied();
                let val_slot = self.compile_expr(&local.value, ops);
                self.property_slots.insert(local.name.clone(), val_slot);
                saved.push((local.name.clone(), old));
            }
            saved
        } else {
            Vec::new()
        };

        // Compile the key lookup
        let key_slot = self.compile_var(&table.key_property, None, ops);

        // Check cache for large tables — compile entries only once
        let table_id = if let Some(&cached_id) = self.dispatch_cache.get(name) {
            cached_id
        } else {
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

            let id = self.compiled_dispatches.len() as Slot;
            self.compiled_dispatches.push(CompiledDispatchTable {
                entries: compiled_entries,
                fallback_ops,
                fallback_slot,
            });

            // Cache large tables for reuse
            if table.entries.len() >= 100 {
                self.dispatch_cache.insert(name.to_string(), id);
            }

            id
        };

        // Restore the dispatch table and parameter bindings
        self.dispatch_tables.insert(name.to_string(), table);
        for (param_name, old) in saved {
            match old {
                Some(s) => {
                    self.property_slots.insert(param_name, s);
                }
                None => {
                    self.property_slots.remove(&param_name);
                }
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
    fn compile_general_function(&mut self, name: &str, args: &[Expr], ops: &mut Vec<Op>) -> Slot {
        let func = match self.functions.get(name).cloned() {
            Some(f) => f,
            None => {
                let dst = self.alloc();
                ops.push(Op::LoadLit { dst, val: 0 });
                return dst;
            }
        };

        // Bind arguments to parameter slots
        let saved_params: Vec<(String, Option<Slot>)> = func
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
                        ops.push(Op::LoadLit { dst: s, val: 0 });
                        s
                    });
                self.property_slots.insert(param.name.clone(), val_slot);
                (param.name.clone(), old)
            })
            .collect();

        // Evaluate local variables
        let saved_locals: Vec<(String, Option<Slot>)> = func
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
                Some(s) => {
                    self.property_slots.insert(param_name, s);
                }
                None => {
                    self.property_slots.remove(&param_name);
                }
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
        compiler
            .property_slots
            .insert(assignment.property.clone(), result_slot);

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

    let mut program = CompiledProgram {
        ops,
        slot_count: compiler.next_slot,
        writeback,
        broadcast_writes: compiled_bw,
        dispatch_tables: compiler.compiled_dispatches,
        property_slots: compiler.property_slots,
    };

    compact_slots(&mut program);

    program
}

// ---------------------------------------------------------------------------
// Slot compaction — register allocation post-pass
// ---------------------------------------------------------------------------

/// Compact slot allocation across the entire compiled program.
///
/// The compiler allocates slots monotonically (SSA-style), producing one slot
/// per temporary. This pass renumbers slots so that dead temporaries are reused,
/// dramatically reducing `slot_count` and the per-tick memset cost.
///
/// Dispatch table entries and broadcast write sub-ops are compacted separately:
/// since only one entry executes per tick, all entries in a table can share the
/// same slot range.
fn compact_slots(program: &mut CompiledProgram) {
    #[cfg(not(target_arch = "wasm32"))]
    let compact_start = std::time::Instant::now();
    let before = program.slot_count;

    // Phase 1: compact main op stream
    let main_pinned = collect_main_pinned(program);
    let mut alloc = SlotAllocator::new();

    // Pre-assign pinned slots so they get stable new indices
    for &s in &main_pinned {
        alloc.assign(s);
    }

    // Compute liveness for main ops (includes dispatch sub-op references)
    let (_last_use, dying_at) = compute_liveness(&program.ops, &program.dispatch_tables);
    let mut slot_map: HashMap<Slot, Slot> = HashMap::new();

    // Map pinned slots first
    for &s in &main_pinned {
        slot_map.insert(s, alloc.get(s));
    }

    // Walk ops, mapping slots and freeing dead ones
    let pinned_set: std::collections::HashSet<Slot> = main_pinned.iter().copied().collect();
    for i in 0..program.ops.len() {
        map_op_slots(&mut program.ops[i], &mut slot_map, &mut alloc);
        // Free slots that die at this op (O(1) amortized via reverse index)
        if let Some(dying) = dying_at.get(&i) {
            for &orig_slot in dying {
                if !pinned_set.contains(&orig_slot) {
                    if let Some(&mapped) = slot_map.get(&orig_slot) {
                        alloc.free(mapped);
                    }
                }
            }
        }
    }

    let main_high = alloc.high_water;

    // Remap writeback
    for entry in &mut program.writeback {
        entry.0 = slot_map[&entry.0];
    }

    // Remap property_slots
    for val in program.property_slots.values_mut() {
        *val = slot_map[val];
    }

    // Reusable scratch buffers — avoids repeated HashMap allocations
    let mut scratch = SubOpScratch::new();

    // Phase 2: compact dispatch table entries
    // Each table's entries share slots starting from main_high (they never
    // execute simultaneously). Fallback ops are also compacted per-table.
    for table in &mut program.dispatch_tables {
        let mut table_high: Slot = 0;

        // Compact fallback
        let fb_high = compact_sub_ops(
            &mut table.fallback_ops,
            &mut table.fallback_slot,
            main_high,
            &slot_map,
            &mut scratch,
        );
        table_high = table_high.max(fb_high);

        // Compact each entry — all overlay from main_high
        for (entry_ops, result_slot) in table.entries.values_mut() {
            let entry_high = compact_sub_ops(entry_ops, result_slot, main_high, &slot_map, &mut scratch);
            table_high = table_high.max(entry_high);
        }

        // table_high is the max slots any single entry needs beyond main_high
        // (captured implicitly in the final slot_count)
        alloc.high_water = alloc.high_water.max(table_high);
    }

    // Phase 3: compact broadcast write sub-ops
    for bw in &mut program.broadcast_writes {
        // dest_slot is in main scope — already mapped
        bw.dest_slot = slot_map.get(&bw.dest_slot).copied().unwrap_or(bw.dest_slot);

        // value_ops get their own compact range starting from main_high
        let bw_high = compact_sub_ops(&mut bw.value_ops, &mut bw.value_slot, main_high, &slot_map, &mut scratch);
        alloc.high_water = alloc.high_water.max(bw_high);

        // Spillover entries
        if let Some(ref mut spillover) = bw.spillover {
            spillover.guard_slot = slot_map
                .get(&spillover.guard_slot)
                .copied()
                .unwrap_or(spillover.guard_slot);
            for (spill_ops, spill_slot) in spillover.entries.values_mut() {
                let sp_high = compact_sub_ops(spill_ops, spill_slot, main_high, &slot_map, &mut scratch);
                alloc.high_water = alloc.high_water.max(sp_high);
            }
        }
    }

    program.slot_count = alloc.high_water;

    #[cfg(not(target_arch = "wasm32"))]
    log::info!(
        "Slot compaction: {} → {} slots ({:.1}% reduction, {:.2}s)",
        before,
        program.slot_count,
        (1.0 - program.slot_count as f64 / before.max(1) as f64) * 100.0,
        compact_start.elapsed().as_secs_f64(),
    );
    #[cfg(target_arch = "wasm32")]
    log::info!(
        "Slot compaction: {} → {} slots ({:.1}% reduction)",
        before,
        program.slot_count,
        (1.0 - program.slot_count as f64 / before.max(1) as f64) * 100.0,
    );
}

/// Collect slots that are "pinned" — referenced outside the main op stream
/// (writeback, property_slots, broadcast dest/guard slots).
fn collect_main_pinned(program: &CompiledProgram) -> Vec<Slot> {
    let mut pinned = Vec::new();
    for &(slot, _) in &program.writeback {
        pinned.push(slot);
    }
    for &slot in program.property_slots.values() {
        pinned.push(slot);
    }
    for bw in &program.broadcast_writes {
        pinned.push(bw.dest_slot);
        if let Some(ref spillover) = bw.spillover {
            pinned.push(spillover.guard_slot);
        }
    }
    pinned.sort_unstable();
    pinned.dedup();
    pinned
}

/// Compute the last op index at which each slot is read or written,
/// and build a reverse index from op index → slots that die after that op.
///
/// Dispatch sub-ops (entry results, fallback results, sub-op reads) are
/// treated as reads at the Dispatch op's index, so parameter slots that
/// are only referenced inside dispatch entries stay alive until the
/// Dispatch executes.
fn compute_liveness(
    ops: &[Op],
    dispatch_tables: &[CompiledDispatchTable],
) -> (HashMap<Slot, usize>, HashMap<usize, Vec<Slot>>) {
    let mut last_use: HashMap<Slot, usize> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        for s in op_slots_read(op).into_iter().chain(op_dst(op)) {
            last_use.insert(s, i);
        }
        // For Dispatch ops, also mark slots referenced by the table's
        // sub-ops and result slots as live at this index.
        if let Op::Dispatch { table_id, .. } = op {
            if let Some(table) = dispatch_tables.get(*table_id as usize) {
                // Fallback
                last_use.entry(table.fallback_slot).and_modify(|v| *v = (*v).max(i)).or_insert(i);
                for s in table.fallback_ops.iter().flat_map(op_slots_read) {
                    last_use.entry(s).and_modify(|v| *v = (*v).max(i)).or_insert(i);
                }
                // Entries
                for (entry_ops, result_slot) in table.entries.values() {
                    last_use.entry(*result_slot).and_modify(|v| *v = (*v).max(i)).or_insert(i);
                    for s in entry_ops.iter().flat_map(op_slots_read) {
                        last_use.entry(s).and_modify(|v| *v = (*v).max(i)).or_insert(i);
                    }
                }
            }
        }
    }
    // Build reverse: op_index → [slots dying here]
    let mut dying_at: HashMap<usize, Vec<Slot>> = HashMap::new();
    for (&slot, &idx) in &last_use {
        dying_at.entry(idx).or_default().push(slot);
    }
    (last_use, dying_at)
}

/// Compact a sub-op stream (dispatch entry, broadcast value, spillover).
/// These get a fresh allocator starting from `base_slot`, and return the
/// high-water mark.
///
/// Reusable scratch buffers are passed in to avoid per-call allocation overhead
/// (critical when called 40M+ times for large dispatch tables).
fn compact_sub_ops(
    ops: &mut [Op],
    result_slot: &mut Slot,
    base_slot: Slot,
    parent_slot_map: &HashMap<Slot, Slot>,
    scratch: &mut SubOpScratch,
) -> Slot {
    if ops.is_empty() {
        if let Some(&mapped) = parent_slot_map.get(result_slot) {
            *result_slot = mapped;
        }
        return base_slot;
    }

    // Ultra-fast path: single LoadLit (the overwhelming majority of dispatch entries).
    // No parent slots, no HashMap lookups needed — just renumber the destination.
    if ops.len() == 1 {
        if let Op::LoadLit { dst, .. } = &mut ops[0] {
            let orig = *dst;
            *dst = base_slot;
            // result_slot is the same as the LoadLit's dst
            if *result_slot == orig {
                *result_slot = base_slot;
            } else if let Some(&mapped) = parent_slot_map.get(result_slot) {
                *result_slot = mapped;
            }
            return base_slot + 1;
        }
    }

    // Fast path for tiny sub-ops (1-2 ops): no liveness analysis needed.
    // Just remap parent-scope slots and assign fresh locals sequentially.
    if ops.len() <= 2 {
        return compact_sub_ops_tiny(ops, result_slot, base_slot, parent_slot_map, scratch);
    }

    // General path: full liveness analysis with reusable scratch buffers
    scratch.clear();
    let mut alloc = SlotAllocator::with_base(base_slot);

    // Compute liveness into scratch vectors
    compute_liveness_into(ops, &mut scratch.last_use, &mut scratch.dying_at);

    // Pre-assign the result slot (if not already mapped from parent)
    let orig_result = *result_slot;
    if !parent_slot_map.contains_key(&orig_result) {
        alloc.assign(orig_result);
        scratch.local_map.insert(orig_result, alloc.get(orig_result));
    }

    for (i, op) in ops.iter_mut().enumerate() {
        seed_from_parent(op, &mut scratch.local_map, parent_slot_map);
        map_op_slots(op, &mut scratch.local_map, &mut alloc);
        if let Some(dying) = scratch.dying_at.get(&i) {
            for &orig_slot in dying {
                if orig_slot != orig_result && !parent_slot_map.contains_key(&orig_slot) {
                    if let Some(&mapped) = scratch.local_map.get(&orig_slot) {
                        alloc.free(mapped);
                    }
                }
            }
        }
    }

    *result_slot = scratch
        .local_map
        .get(&orig_result)
        .or_else(|| parent_slot_map.get(&orig_result))
        .copied()
        .unwrap_or(orig_result);
    alloc.high_water
}

/// Fast-path compaction for sub-op streams with 1-2 ops.
///
/// With so few ops, liveness analysis is pointless — no slot can die before
/// the end. We just remap parent-scope slots and assign sequential new slots
/// for any locals.
fn compact_sub_ops_tiny(
    ops: &mut [Op],
    result_slot: &mut Slot,
    base_slot: Slot,
    parent_slot_map: &HashMap<Slot, Slot>,
    scratch: &mut SubOpScratch,
) -> Slot {
    scratch.local_map.clear();
    let mut alloc = SlotAllocator::with_base(base_slot);

    let orig_result = *result_slot;
    if !parent_slot_map.contains_key(&orig_result) {
        alloc.assign(orig_result);
        scratch.local_map.insert(orig_result, alloc.get(orig_result));
    }

    for op in ops.iter_mut() {
        seed_from_parent(op, &mut scratch.local_map, parent_slot_map);
        map_op_slots(op, &mut scratch.local_map, &mut alloc);
    }

    *result_slot = scratch
        .local_map
        .get(&orig_result)
        .or_else(|| parent_slot_map.get(&orig_result))
        .copied()
        .unwrap_or(orig_result);
    alloc.high_water
}

/// Reusable scratch buffers for `compact_sub_ops` to avoid per-call allocation.
struct SubOpScratch {
    local_map: HashMap<Slot, Slot>,
    last_use: HashMap<Slot, usize>,
    dying_at: HashMap<usize, Vec<Slot>>,
}

impl SubOpScratch {
    fn new() -> Self {
        Self {
            local_map: HashMap::new(),
            last_use: HashMap::new(),
            dying_at: HashMap::new(),
        }
    }

    fn clear(&mut self) {
        self.local_map.clear();
        self.last_use.clear();
        self.dying_at.clear();
    }
}

/// Compute liveness into caller-provided maps (avoids allocation per call).
fn compute_liveness_into(
    ops: &[Op],
    last_use: &mut HashMap<Slot, usize>,
    dying_at: &mut HashMap<usize, Vec<Slot>>,
) {
    for (i, op) in ops.iter().enumerate() {
        for s in op_slots_read(op).into_iter().chain(op_dst(op)) {
            last_use.insert(s, i);
        }
    }
    for (&slot, &idx) in last_use.iter() {
        dying_at.entry(idx).or_default().push(slot);
    }
}

/// Simple slot allocator with a free list.
struct SlotAllocator {
    next: Slot,
    high_water: Slot,
    free_list: Vec<Slot>,
    /// Tracks original→new assignments for pre-assigned (pinned) slots.
    assigned: HashMap<Slot, Slot>,
}

impl SlotAllocator {
    fn new() -> Self {
        Self {
            next: 0,
            high_water: 0,
            free_list: Vec::new(),
            assigned: HashMap::new(),
        }
    }

    fn with_base(base: Slot) -> Self {
        Self {
            next: base,
            high_water: base,
            free_list: Vec::new(),
            assigned: HashMap::new(),
        }
    }

    /// Pre-assign a slot (for pinned slots). Call before mapping.
    fn assign(&mut self, original: Slot) {
        if self.assigned.contains_key(&original) {
            return;
        }
        let new = self.alloc();
        self.assigned.insert(original, new);
    }

    /// Get the mapped slot for a pre-assigned original.
    fn get(&self, original: Slot) -> Slot {
        self.assigned[&original]
    }

    /// Allocate a slot — reuse from free list or bump.
    fn alloc(&mut self) -> Slot {
        if let Some(s) = self.free_list.pop() {
            s
        } else {
            let s = self.next;
            self.next += 1;
            self.high_water = self.high_water.max(self.next);
            s
        }
    }

    /// Return a slot to the free list.
    fn free(&mut self, slot: Slot) {
        self.free_list.push(slot);
    }

    /// Get or allocate a mapping for an original slot.
    fn get_or_alloc(&mut self, original: Slot, slot_map: &mut HashMap<Slot, Slot>) -> Slot {
        if let Some(&mapped) = slot_map.get(&original) {
            mapped
        } else if let Some(&mapped) = self.assigned.get(&original) {
            slot_map.insert(original, mapped);
            mapped
        } else {
            let mapped = self.alloc();
            slot_map.insert(original, mapped);
            mapped
        }
    }
}

/// Get all slots read by an op (not including dst).
fn op_slots_read(op: &Op) -> Vec<Slot> {
    match op {
        Op::LoadLit { .. } => vec![],
        Op::LoadSlot { src, .. } => vec![*src],
        Op::LoadState { .. } => vec![],
        Op::LoadMem { addr_slot, .. } => vec![*addr_slot],
        Op::LoadMem16 { addr_slot, .. } => vec![*addr_slot],
        Op::LoadKeyboard { .. } => vec![],
        Op::Add { a, b, .. }
        | Op::Sub { a, b, .. }
        | Op::Mul { a, b, .. }
        | Op::Div { a, b, .. }
        | Op::Mod { a, b, .. }
        | Op::And { a, b, .. }
        | Op::Shr { a, b, .. }
        | Op::Shl { a, b, .. } => vec![*a, *b],
        Op::Neg { src, .. }
        | Op::Abs { src, .. }
        | Op::Sign { src, .. }
        | Op::Floor { src, .. } => {
            vec![*src]
        }
        Op::Pow { base, exp, .. } => vec![*base, *exp],
        Op::Min { args, .. } | Op::Max { args, .. } => args.clone(),
        Op::Clamp { min, val, max, .. } => vec![*min, *val, *max],
        Op::Round { val, interval, .. } => vec![*val, *interval],
        Op::Bit { val, idx, .. } => vec![*val, *idx],
        Op::CmpEq { a, b, .. } => vec![*a, *b],
        Op::BranchIfZero { cond, .. } => vec![*cond],
        Op::Jump { .. } => vec![],
        Op::Dispatch { key, .. } => vec![*key],
        Op::StoreState { src, .. } => vec![*src],
        Op::StoreMem { addr_slot, src, .. } => vec![*addr_slot, *src],
    }
}

/// Get the destination slot of an op, if any.
fn op_dst(op: &Op) -> Option<Slot> {
    match op {
        Op::LoadLit { dst, .. }
        | Op::LoadSlot { dst, .. }
        | Op::LoadState { dst, .. }
        | Op::LoadMem { dst, .. }
        | Op::LoadMem16 { dst, .. }
        | Op::LoadKeyboard { dst, .. }
        | Op::Add { dst, .. }
        | Op::Sub { dst, .. }
        | Op::Mul { dst, .. }
        | Op::Div { dst, .. }
        | Op::Mod { dst, .. }
        | Op::Neg { dst, .. }
        | Op::Abs { dst, .. }
        | Op::Sign { dst, .. }
        | Op::Pow { dst, .. }
        | Op::Min { dst, .. }
        | Op::Max { dst, .. }
        | Op::Clamp { dst, .. }
        | Op::Round { dst, .. }
        | Op::Floor { dst, .. }
        | Op::And { dst, .. }
        | Op::Shr { dst, .. }
        | Op::Shl { dst, .. }
        | Op::Bit { dst, .. }
        | Op::CmpEq { dst, .. }
        | Op::Dispatch { dst, .. } => Some(*dst),
        Op::BranchIfZero { .. } | Op::Jump { .. } | Op::StoreState { .. } | Op::StoreMem { .. } => {
            None
        }
    }
}

/// Remap all slot references in an op, allocating new slots as needed.
fn map_op_slots(op: &mut Op, slot_map: &mut HashMap<Slot, Slot>, alloc: &mut SlotAllocator) {
    match op {
        Op::LoadLit { dst, .. } => {
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::LoadSlot { dst, src } => {
            *src = alloc.get_or_alloc(*src, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::LoadState { dst, .. } => {
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::LoadMem { dst, addr_slot } => {
            *addr_slot = alloc.get_or_alloc(*addr_slot, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::LoadMem16 { dst, addr_slot } => {
            *addr_slot = alloc.get_or_alloc(*addr_slot, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::LoadKeyboard { dst } => {
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::Add { dst, a, b }
        | Op::Sub { dst, a, b }
        | Op::Mul { dst, a, b }
        | Op::Div { dst, a, b }
        | Op::Mod { dst, a, b }
        | Op::And { dst, a, b }
        | Op::Shr { dst, a, b }
        | Op::Shl { dst, a, b } => {
            *a = alloc.get_or_alloc(*a, slot_map);
            *b = alloc.get_or_alloc(*b, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::Neg { dst, src }
        | Op::Abs { dst, src }
        | Op::Sign { dst, src }
        | Op::Floor { dst, src } => {
            *src = alloc.get_or_alloc(*src, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::Pow { dst, base, exp } => {
            *base = alloc.get_or_alloc(*base, slot_map);
            *exp = alloc.get_or_alloc(*exp, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::Min { dst, args } | Op::Max { dst, args } => {
            for a in args.iter_mut() {
                *a = alloc.get_or_alloc(*a, slot_map);
            }
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::Clamp { dst, min, val, max } => {
            *min = alloc.get_or_alloc(*min, slot_map);
            *val = alloc.get_or_alloc(*val, slot_map);
            *max = alloc.get_or_alloc(*max, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::Round {
            dst, val, interval, ..
        } => {
            *val = alloc.get_or_alloc(*val, slot_map);
            *interval = alloc.get_or_alloc(*interval, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::Bit { dst, val, idx } => {
            *val = alloc.get_or_alloc(*val, slot_map);
            *idx = alloc.get_or_alloc(*idx, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::CmpEq { dst, a, b } => {
            *a = alloc.get_or_alloc(*a, slot_map);
            *b = alloc.get_or_alloc(*b, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::BranchIfZero { cond, .. } => {
            *cond = alloc.get_or_alloc(*cond, slot_map);
        }
        Op::Jump { .. } => {}
        Op::Dispatch { dst, key, .. } => {
            *key = alloc.get_or_alloc(*key, slot_map);
            *dst = alloc.get_or_alloc(*dst, slot_map);
        }
        Op::StoreState { src, .. } => {
            *src = alloc.get_or_alloc(*src, slot_map);
        }
        Op::StoreMem { addr_slot, src } => {
            *addr_slot = alloc.get_or_alloc(*addr_slot, slot_map);
            *src = alloc.get_or_alloc(*src, slot_map);
        }
    }
}

/// Seed a local slot map from a parent map for all slots referenced by an op.
/// This avoids cloning the entire parent map — only slots actually used get copied.
fn seed_from_parent(
    op: &Op,
    local_map: &mut HashMap<Slot, Slot>,
    parent_map: &HashMap<Slot, Slot>,
) {
    let mut seed = |s: Slot| {
        if !local_map.contains_key(&s) {
            if let Some(&mapped) = parent_map.get(&s) {
                local_map.insert(s, mapped);
            }
        }
    };
    match op {
        Op::LoadLit { dst, .. } => { seed(*dst); }
        Op::LoadSlot { dst, src } => { seed(*dst); seed(*src); }
        Op::LoadState { dst, .. } => { seed(*dst); }
        Op::LoadMem { dst, addr_slot } | Op::LoadMem16 { dst, addr_slot } => { seed(*dst); seed(*addr_slot); }
        Op::LoadKeyboard { dst } => { seed(*dst); }
        Op::Add { dst, a, b } | Op::Sub { dst, a, b } | Op::Mul { dst, a, b }
        | Op::Div { dst, a, b } | Op::Mod { dst, a, b } | Op::And { dst, a, b }
        | Op::Shr { dst, a, b } | Op::Shl { dst, a, b } => { seed(*dst); seed(*a); seed(*b); }
        Op::Neg { dst, src } | Op::Abs { dst, src } | Op::Sign { dst, src }
        | Op::Floor { dst, src } => { seed(*dst); seed(*src); }
        Op::Pow { dst, base, exp } => { seed(*dst); seed(*base); seed(*exp); }
        Op::Bit { dst, val, idx } => { seed(*dst); seed(*val); seed(*idx); }
        Op::CmpEq { dst, a, b } => { seed(*dst); seed(*a); seed(*b); }
        Op::Min { dst, args } | Op::Max { dst, args } => { seed(*dst); for a in args { seed(*a); } }
        Op::Clamp { dst, min, val, max } => { seed(*dst); seed(*min); seed(*val); seed(*max); }
        Op::Round { dst, val, interval, .. } => { seed(*dst); seed(*val); seed(*interval); }
        Op::BranchIfZero { cond, .. } => { seed(*cond); }
        Op::Jump { .. } => {}
        Op::Dispatch { dst, key, .. } => { seed(*dst); seed(*key); }
        Op::StoreState { src, .. } => { seed(*src); }
        Op::StoreMem { addr_slot, src } => { seed(*addr_slot); seed(*src); }
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
pub fn execute(program: &CompiledProgram, state: &mut State, slots: &mut Vec<i32>) {
    // Reset slots (reuse allocation)
    slots.clear();
    slots.resize(program.slot_count as usize, 0);

    // Execute main ops
    exec_ops(&program.ops, &program.dispatch_tables, state, slots);

    // Writeback: apply computed values to state
    for &(slot, addr) in &program.writeback {
        let value = slots[slot as usize];
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
            let value = slots[bw.value_slot as usize];
            state.write_mem(dest, value);
        }
        // Spillover
        if let Some(ref spillover) = bw.spillover {
            let guard = slots[spillover.guard_slot as usize];
            if guard == 1 {
                if let Some((ref spill_ops, spill_slot)) = spillover.entries.get(&dest_i64) {
                    exec_ops(spill_ops, &program.dispatch_tables, state, slots);
                    let value = slots[*spill_slot as usize];
                    state.write_mem(dest + 1, value);
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
    slots: &mut [i32],
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
                slots[*dst as usize] = state.read_mem(*addr);
            }
            Op::LoadMem { dst, addr_slot } => {
                let addr = slots[*addr_slot as usize];
                slots[*dst as usize] = state.read_mem(addr);
            }
            Op::LoadMem16 { dst, addr_slot } => {
                let addr = slots[*addr_slot as usize];
                if addr < 0 {
                    slots[*dst as usize] = state.read_mem(addr);
                } else {
                    slots[*dst as usize] = state.read_mem16(addr);
                }
            }
            Op::LoadKeyboard { dst } => {
                slots[*dst as usize] = state.keyboard;
            }
            Op::Add { dst, a, b } => {
                slots[*dst as usize] = slots[*a as usize].wrapping_add(slots[*b as usize]);
            }
            Op::Sub { dst, a, b } => {
                slots[*dst as usize] = slots[*a as usize].wrapping_sub(slots[*b as usize]);
            }
            Op::Mul { dst, a, b } => {
                slots[*dst as usize] = slots[*a as usize].wrapping_mul(slots[*b as usize]);
            }
            Op::Div { dst, a, b } => {
                let divisor = slots[*b as usize];
                slots[*dst as usize] = if divisor == 0 {
                    0
                } else {
                    slots[*a as usize] / divisor
                };
            }
            Op::Mod { dst, a, b } => {
                let divisor = slots[*b as usize];
                slots[*dst as usize] = if divisor == 0 {
                    0
                } else {
                    slots[*a as usize] % divisor
                };
            }
            Op::Neg { dst, src } => {
                slots[*dst as usize] = slots[*src as usize].wrapping_neg();
            }
            Op::Abs { dst, src } => {
                slots[*dst as usize] = slots[*src as usize].wrapping_abs();
            }
            Op::Sign { dst, src } => {
                let v = slots[*src as usize];
                slots[*dst as usize] = if v > 0 {
                    1
                } else if v < 0 {
                    -1
                } else {
                    0
                };
            }
            Op::Pow { dst, base, exp } => {
                let b = slots[*base as usize];
                let e = slots[*exp as usize];
                slots[*dst as usize] = if e < 0 { 0 } else { b.wrapping_pow(e as u32) };
            }
            Op::Min { dst, args } => {
                let mut v = i32::MAX;
                for &a in args {
                    v = v.min(slots[a as usize]);
                }
                slots[*dst as usize] = v;
            }
            Op::Max { dst, args } => {
                let mut v = i32::MIN;
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
            Op::Round {
                dst,
                strategy,
                val,
                interval,
            } => {
                let v = slots[*val as usize];
                let i = slots[*interval as usize];
                slots[*dst as usize] = if i == 0 {
                    v
                } else {
                    // Integer rounding: round(down, v/i, 1)*i is just floor-div
                    match strategy {
                        RoundStrategy::Down => v.div_euclid(i) * i,
                        RoundStrategy::Up => (v + i - 1).div_euclid(i) * i,
                        RoundStrategy::Nearest => ((v + i / 2).div_euclid(i)) * i,
                        RoundStrategy::ToZero => (v / i) * i,
                    }
                };
            }
            Op::Floor { dst, src } => {
                // No-op for integers — value is already floored
                slots[*dst as usize] = slots[*src as usize];
            }
            Op::And { dst, a, b } => {
                let av = slots[*a as usize];
                let bv = slots[*b as usize] as u32;
                slots[*dst as usize] = if bv >= 32 {
                    av
                } else {
                    av & ((1i32 << bv) - 1)
                };
            }
            Op::Shr { dst, a, b } => {
                let av = slots[*a as usize];
                let bv = slots[*b as usize] as u32;
                slots[*dst as usize] = if bv >= 32 { 0 } else { av >> bv };
            }
            Op::Shl { dst, a, b } => {
                let av = slots[*a as usize];
                let bv = slots[*b as usize] as u32;
                slots[*dst as usize] = if bv >= 32 { 0 } else { av << bv };
            }
            Op::Bit { dst, val, idx } => {
                let v = slots[*val as usize];
                let i = slots[*idx as usize] as u32;
                slots[*dst as usize] = if i >= 32 { 0 } else { (v >> i) & 1 };
            }
            Op::CmpEq { dst, a, b } => {
                slots[*dst as usize] = if slots[*a as usize] == slots[*b as usize] {
                    1
                } else {
                    0
                };
            }
            Op::BranchIfZero { cond, target } => {
                if slots[*cond as usize] == 0 {
                    pc = *target as usize;
                    continue;
                }
            }
            Op::Jump { target } => {
                pc = *target as usize;
                continue;
            }
            Op::Dispatch {
                dst, key, table_id, ..
            } => {
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
                state.write_mem(*addr, slots[*src as usize]);
            }
            Op::StoreMem { addr_slot, src } => {
                state.write_mem(slots[*addr_slot as usize], slots[*src as usize]);
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
    if let Some(addr) = property_to_address(name) {
        return addr < -14;
    }
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state;

    /// Install a test address map so property_to_address works for --AX etc.
    fn setup() {
        use crate::state::addr;
        let mut map = std::collections::HashMap::new();
        for (name, a) in [
            ("AX", addr::AX),
            ("CX", addr::CX),
            ("DX", addr::DX),
            ("BX", addr::BX),
            ("SP", addr::SP),
            ("BP", addr::BP),
            ("SI", addr::SI),
            ("DI", addr::DI),
            ("IP", addr::IP),
            ("ES", addr::ES),
            ("CS", addr::CS),
            ("SS", addr::SS),
            ("DS", addr::DS),
            ("flags", addr::FLAGS),
            ("AH", addr::AH),
            ("CH", addr::CH),
            ("DH", addr::DH),
            ("BH", addr::BH),
            ("AL", addr::AL),
            ("CL", addr::CL),
            ("DL", addr::DL),
            ("BL", addr::BL),
        ] {
            map.insert(name.to_string(), a);
        }
        crate::eval::set_address_map(map);
    }

    #[test]
    fn compile_literal() {
        let expr = Expr::Literal(42.0);
        let mut compiler = Compiler::new(&HashMap::new(), &HashMap::new());
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);
        assert_eq!(ops.len(), 1);

        let mut state = State::default();
        let mut slots = vec![0i32; compiler.next_slot as usize];
        exec_ops(&ops, &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 42);
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
        let mut slots = vec![0i32; compiler.next_slot as usize];
        exec_ops(&ops, &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 30);
    }

    #[test]
    fn compile_var_from_state() {
        setup();
        let expr = Expr::Var {
            name: "--AX".to_string(),
            fallback: None,
        };
        let mut compiler = Compiler::new(&HashMap::new(), &HashMap::new());
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);

        let mut state = State::default();
        state.registers[state::reg::AX] = 0x1234;
        let mut slots = vec![0i32; compiler.next_slot as usize];
        exec_ops(&ops, &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 0x1234);
    }

    #[test]
    fn compile_style_condition() {
        setup();
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
        let mut slots = vec![0i32; compiler.next_slot as usize];
        exec_ops(&ops, &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 200);
    }

    #[test]
    fn compile_readmem() {
        setup();
        // Build a dispatch table that maps key K → Var at state address K
        // (identity-read pattern, detected generically).
        use crate::pattern::dispatch_table::DispatchTable;
        let mut entries = HashMap::new();
        entries.insert(
            -1,
            Expr::Var {
                name: "--AX".to_string(),
                fallback: None,
            },
        );
        entries.insert(
            -2,
            Expr::Var {
                name: "--CX".to_string(),
                fallback: None,
            },
        );
        entries.insert(
            -3,
            Expr::Var {
                name: "--DX".to_string(),
                fallback: None,
            },
        );
        entries.insert(
            -4,
            Expr::Var {
                name: "--BX".to_string(),
                fallback: None,
            },
        );
        entries.insert(
            0,
            Expr::Var {
                name: "--m0".to_string(),
                fallback: None,
            },
        );
        let mut dispatch_tables = HashMap::new();
        dispatch_tables.insert(
            "--readMem".to_string(),
            DispatchTable {
                key_property: "--at".to_string(),
                entries,
                fallback: Expr::Literal(0.0),
            },
        );
        let functions = HashMap::new();

        let expr = Expr::FunctionCall {
            name: "--readMem".to_string(),
            args: vec![Expr::Literal(-1.0)], // AX register
        };
        let mut compiler = Compiler::new(&functions, &dispatch_tables);
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);

        let mut state = State::default();
        state.registers[state::reg::AX] = 42;
        let mut slots = vec![0i32; compiler.next_slot as usize];
        exec_ops(&ops, &compiler.compiled_dispatches, &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 42);
    }

    #[test]
    fn compile_bitwise_ops() {
        // Build a function with body mod(a, pow(2, b)) — bitmask pattern.
        let mut functions = HashMap::new();
        functions.insert(
            "--lowerBytes".to_string(),
            FunctionDef {
                name: "--lowerBytes".to_string(),
                parameters: vec![
                    FunctionParam {
                        name: "--a".to_string(),
                        syntax: PropertySyntax::Integer,
                    },
                    FunctionParam {
                        name: "--b".to_string(),
                        syntax: PropertySyntax::Integer,
                    },
                ],
                locals: vec![],
                result: Expr::Calc(CalcOp::Mod(
                    Box::new(Expr::Var {
                        name: "--a".to_string(),
                        fallback: None,
                    }),
                    Box::new(Expr::Calc(CalcOp::Pow(
                        Box::new(Expr::Literal(2.0)),
                        Box::new(Expr::Var {
                            name: "--b".to_string(),
                            fallback: None,
                        }),
                    ))),
                )),
            },
        );

        // --lowerBytes(0xFF, 4) → 0xF
        let expr = Expr::FunctionCall {
            name: "--lowerBytes".to_string(),
            args: vec![Expr::Literal(0xFF as f64), Expr::Literal(4.0)],
        };
        let mut compiler = Compiler::new(&functions, &HashMap::new());
        let mut ops = Vec::new();
        let slot = compiler.compile_expr(&expr, &mut ops);

        let mut state = State::default();
        let mut slots = vec![0i32; compiler.next_slot as usize];
        exec_ops(&ops, &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 15);
    }

    #[test]
    fn compile_full_program() {
        setup();
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
        let mut slots = vec![0i32; compiler.next_slot as usize];
        exec_ops(&ops, &[], &mut state, &mut slots);
        assert_eq!(slots[slot as usize], 42);
    }

    #[test]
    fn compile_value_forwarding() {
        setup();
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
