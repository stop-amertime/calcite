//! The evaluator — runs compiled programs against the flat state.
//!
//! Two modes:
//! 1. **Interpreted**: Evaluates `Expr` trees directly against `State`.
//!    This is the Phase 1-2 path: parse CSS → evaluate expressions.
//! 2. **Compiled**: Uses pattern-recognised structures (dispatch tables,
//!    direct writes) for O(1) operations. (Phase 2+)

use std::collections::HashMap;

use crate::pattern::broadcast_write::{self, BroadcastWrite};
use crate::pattern::dispatch_table::{self, DispatchTable};
use crate::state::State;
use crate::types::*;

/// The main evaluator.
#[derive(Debug)]
pub struct Evaluator {
    /// Parsed @function definitions, keyed by name.
    pub functions: HashMap<String, FunctionDef>,
    /// Property assignments to execute each tick (in declaration order).
    pub assignments: Vec<Assignment>,
    /// Recognised dispatch tables for large if(style()) chains in functions.
    pub dispatch_tables: HashMap<String, DispatchTable>,
    /// Recognised broadcast write patterns.
    pub broadcast_writes: Vec<BroadcastWrite>,
}

/// The result of running a batch of ticks.
#[derive(Debug, Clone, Default)]
pub struct TickResult {
    pub changes: Vec<(String, String)>,
    pub ticks_executed: u32,
}

impl Evaluator {
    /// Build an evaluator from a `ParsedProgram`.
    pub fn from_parsed(program: &ParsedProgram) -> Self {
        let functions: HashMap<String, FunctionDef> = program
            .functions
            .iter()
            .map(|f| (f.name.clone(), f.clone()))
            .collect();

        // Recognise dispatch tables in function result expressions
        let mut dispatch_tables = HashMap::new();
        for func in &program.functions {
            if let Expr::StyleCondition {
                branches, fallback, ..
            } = &func.result
            {
                if let Some(table) = dispatch_table::recognise_dispatch(branches, fallback) {
                    log::info!(
                        "Recognised dispatch table in @function {}: {} entries on {}",
                        func.name,
                        table.entries.len(),
                        table.key_property,
                    );
                    dispatch_tables.insert(func.name.clone(), table);
                }
            }
        }

        // Recognise broadcast write patterns in assignments
        let broadcast_writes = broadcast_write::recognise_broadcast(&program.assignments);
        for bw in &broadcast_writes {
            log::info!(
                "Recognised broadcast write: {} → {} targets",
                bw.dest_property,
                bw.address_map.len(),
            );
        }

        Evaluator {
            functions,
            assignments: program.assignments.clone(),
            dispatch_tables,
            broadcast_writes,
        }
    }

    /// Run a single tick: evaluate all assignments against the state.
    pub fn tick(&self, state: &mut State) -> TickResult {
        let mut env = EvalEnv::new(&self.functions, &self.dispatch_tables);

        // Execute all assignments in declaration order
        for assignment in &self.assignments {
            let value = env.eval_expr(&assignment.value, state);
            // Store the computed value in the environment for subsequent declarations
            env.properties.insert(assignment.property.clone(), value);
        }

        // Apply broadcast writes efficiently
        for bw in &self.broadcast_writes {
            let dest = env.resolve_property(&bw.dest_property, state);
            let dest_i64 = dest as i64;
            // Find the matching target and write directly
            for (addr, var_name) in &bw.address_map {
                if *addr == dest_i64 {
                    let value = env.eval_expr(&bw.value_expr, state);
                    state.write_mem(*addr as i32, value as i32);
                    env.properties.insert(var_name.clone(), value);
                    break;
                }
            }
        }

        // Apply non-broadcast property values back to state
        // (This maps custom property names to state addresses)
        let changes = self.apply_state(&env.properties, state);

        state.frame_counter += 1;

        TickResult {
            changes,
            ticks_executed: 1,
        }
    }

    /// Run a batch of ticks.
    pub fn run_batch(&self, state: &mut State, count: u32) -> TickResult {
        let mut result = TickResult::default();
        for _ in 0..count {
            let tick_result = self.tick(state);
            result.changes = tick_result.changes;
            result.ticks_executed += 1;
        }
        result
    }

    /// Apply computed property values to state and return the changes.
    fn apply_state(
        &self,
        properties: &HashMap<String, f64>,
        state: &mut State,
    ) -> Vec<(String, String)> {
        let mut changes = Vec::new();

        for (name, &value) in properties {
            let int_val = value as i32;
            // Map well-known property names to state addresses
            if let Some(addr) = property_to_address(name) {
                let old = state.read_mem(addr);
                if old != int_val {
                    state.write_mem(addr, int_val);
                    changes.push((name.clone(), int_val.to_string()));
                }
            }
        }

        changes
    }
}

/// Map a CSS custom property name to a state address.
///
/// Uses x86CSS's naming convention:
/// - `--AX`, `--CX`, ..., `--flags` → register addresses
/// - `--m0`, `--m1`, ... → memory addresses
fn property_to_address(name: &str) -> Option<i32> {
    use crate::state::addr;
    match name {
        "--AX" => Some(addr::AX),
        "--CX" => Some(addr::CX),
        "--DX" => Some(addr::DX),
        "--BX" => Some(addr::BX),
        "--SP" => Some(addr::SP),
        "--BP" => Some(addr::BP),
        "--SI" => Some(addr::SI),
        "--DI" => Some(addr::DI),
        "--IP" => Some(addr::IP),
        "--ES" => Some(addr::ES),
        "--CS" => Some(addr::CS),
        "--SS" => Some(addr::SS),
        "--DS" => Some(addr::DS),
        "--flags" => Some(addr::FLAGS),
        _ if name.starts_with("--m") => name[3..].parse::<i32>().ok(),
        _ => None,
    }
}

/// Evaluation environment for a single tick.
///
/// Holds function definitions and computed property values for the current tick.
struct EvalEnv<'a> {
    functions: &'a HashMap<String, FunctionDef>,
    dispatch_tables: &'a HashMap<String, DispatchTable>,
    /// Property values computed so far in this tick.
    properties: HashMap<String, f64>,
    /// Call depth for recursion protection.
    call_depth: usize,
}

const MAX_CALL_DEPTH: usize = 64;

impl<'a> EvalEnv<'a> {
    fn new(
        functions: &'a HashMap<String, FunctionDef>,
        dispatch_tables: &'a HashMap<String, DispatchTable>,
    ) -> Self {
        Self {
            functions,
            dispatch_tables,
            properties: HashMap::new(),
            call_depth: 0,
        }
    }

    /// Evaluate an expression to a numeric value.
    fn eval_expr(&mut self, expr: &Expr, state: &State) -> f64 {
        match expr {
            Expr::Literal(v) => *v,

            Expr::Var { name, fallback } => {
                // Check current tick's computed properties first
                if let Some(&v) = self.properties.get(name.as_str()) {
                    return v;
                }
                // Then check state
                if let Some(addr) = property_to_address(name) {
                    return state.read_mem(addr) as f64;
                }
                // Fallback
                if let Some(fb) = fallback {
                    return self.eval_expr(fb, state);
                }
                0.0
            }

            Expr::StringLiteral(_) => 0.0, // strings have no numeric value

            Expr::Calc(op) => self.eval_calc(op, state),

            Expr::StyleCondition {
                branches, fallback, ..
            } => {
                for branch in branches {
                    if self.eval_style_test(&branch.condition, state) {
                        return self.eval_expr(&branch.then, state);
                    }
                }
                self.eval_expr(fallback, state)
            }

            Expr::FunctionCall { name, args } => self.eval_function_call(name, args, state),
        }
    }

    /// Evaluate a `CalcOp`.
    fn eval_calc(&mut self, op: &CalcOp, state: &State) -> f64 {
        match op {
            CalcOp::Add(a, b) => self.eval_expr(a, state) + self.eval_expr(b, state),
            CalcOp::Sub(a, b) => self.eval_expr(a, state) - self.eval_expr(b, state),
            CalcOp::Mul(a, b) => self.eval_expr(a, state) * self.eval_expr(b, state),
            CalcOp::Div(a, b) => {
                let divisor = self.eval_expr(b, state);
                if divisor == 0.0 {
                    0.0
                } else {
                    self.eval_expr(a, state) / divisor
                }
            }
            CalcOp::Mod(a, b) => {
                let divisor = self.eval_expr(b, state);
                if divisor == 0.0 {
                    0.0
                } else {
                    self.eval_expr(a, state) % divisor
                }
            }
            CalcOp::Min(args) => args
                .iter()
                .map(|a| self.eval_expr(a, state))
                .fold(f64::INFINITY, f64::min),
            CalcOp::Max(args) => args
                .iter()
                .map(|a| self.eval_expr(a, state))
                .fold(f64::NEG_INFINITY, f64::max),
            CalcOp::Clamp(min, val, max) => {
                let min_v = self.eval_expr(min, state);
                let val_v = self.eval_expr(val, state);
                let max_v = self.eval_expr(max, state);
                val_v.clamp(min_v, max_v)
            }
            CalcOp::Round(strategy, val, interval) => {
                let v = self.eval_expr(val, state);
                let i = self.eval_expr(interval, state);
                if i == 0.0 {
                    return v;
                }
                match strategy {
                    RoundStrategy::Nearest => (v / i).round() * i,
                    RoundStrategy::Up => (v / i).ceil() * i,
                    RoundStrategy::Down => (v / i).floor() * i,
                    RoundStrategy::ToZero => (v / i).trunc() * i,
                }
            }
            CalcOp::Pow(base, exp) => self.eval_expr(base, state).powf(self.eval_expr(exp, state)),
            CalcOp::Sign(val) => {
                let v = self.eval_expr(val, state);
                if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            CalcOp::Abs(val) => self.eval_expr(val, state).abs(),
            CalcOp::Negate(val) => -self.eval_expr(val, state),
        }
    }

    /// Resolve a property value: check computed properties, then state.
    fn resolve_property(&self, name: &str, state: &State) -> f64 {
        if let Some(&v) = self.properties.get(name) {
            return v;
        }
        if let Some(addr) = property_to_address(name) {
            return state.read_mem(addr) as f64;
        }
        0.0
    }

    /// Evaluate a style test (condition inside an `if()` branch).
    fn eval_style_test(&mut self, test: &StyleTest, state: &State) -> bool {
        match test {
            StyleTest::Single { property, value } => {
                let prop_val = self.resolve_property(property, state);
                let test_val = self.eval_expr(value, state);
                (prop_val - test_val).abs() < 0.5
            }
            StyleTest::And(tests) => tests.iter().all(|t| self.eval_style_test(t, state)),
            StyleTest::Or(tests) => tests.iter().any(|t| self.eval_style_test(t, state)),
        }
    }

    /// Evaluate a @function call.
    fn eval_function_call(&mut self, name: &str, args: &[Expr], state: &State) -> f64 {
        if self.call_depth >= MAX_CALL_DEPTH {
            log::warn!("max call depth exceeded calling {name}");
            return 0.0;
        }

        // Check for a dispatch table optimisation
        if let Some(table) = self.dispatch_tables.get(name) {
            return self.eval_dispatch(table, args, state);
        }

        let func = match self.functions.get(name) {
            Some(f) => f.clone(), // Clone to avoid borrow conflict
            None => {
                log::debug!("undefined function: {name}");
                return 0.0;
            }
        };

        self.call_depth += 1;

        // Bind arguments to parameter names
        let old_props: Vec<(String, Option<f64>)> = func
            .parameters
            .iter()
            .enumerate()
            .map(|(i, param)| {
                let old = self.properties.get(&param.name).copied();
                let val = args.get(i).map(|a| self.eval_expr(a, state)).unwrap_or(0.0);
                self.properties.insert(param.name.clone(), val);
                (param.name.clone(), old)
            })
            .collect();

        // Evaluate local variables
        let old_locals: Vec<(String, Option<f64>)> = func
            .locals
            .iter()
            .map(|local| {
                let old = self.properties.get(&local.name).copied();
                let val = self.eval_expr(&local.value, state);
                self.properties.insert(local.name.clone(), val);
                (local.name.clone(), old)
            })
            .collect();

        // Evaluate result
        let result = self.eval_expr(&func.result, state);

        // Restore previous property values
        for (name, old) in old_props.into_iter().chain(old_locals) {
            match old {
                Some(v) => {
                    self.properties.insert(name, v);
                }
                None => {
                    self.properties.remove(&name);
                }
            }
        }

        self.call_depth -= 1;
        result
    }

    /// Evaluate using a dispatch table — O(1) lookup.
    fn eval_dispatch(&mut self, table: &DispatchTable, args: &[Expr], state: &State) -> f64 {
        // The key is the first argument (the dispatched property)
        let key = args
            .first()
            .map(|a| self.eval_expr(a, state))
            .unwrap_or(0.0) as i64;

        if let Some(result_expr) = table.entries.get(&key) {
            self.eval_expr(result_expr, state)
        } else {
            self.eval_expr(&table.fallback, state)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state;

    #[test]
    fn eval_literal() {
        let functions = HashMap::new();
        let dispatch = HashMap::new();
        let mut env = EvalEnv::new(&functions, &dispatch);
        let state = State::default();

        assert_eq!(env.eval_expr(&Expr::Literal(42.0), &state), 42.0);
    }

    #[test]
    fn eval_calc_operations() {
        let functions = HashMap::new();
        let dispatch = HashMap::new();
        let mut env = EvalEnv::new(&functions, &dispatch);
        let state = State::default();

        let expr = Expr::Calc(CalcOp::Add(
            Box::new(Expr::Literal(10.0)),
            Box::new(Expr::Literal(20.0)),
        ));
        assert_eq!(env.eval_expr(&expr, &state), 30.0);

        let expr = Expr::Calc(CalcOp::Mul(
            Box::new(Expr::Literal(3.0)),
            Box::new(Expr::Literal(7.0)),
        ));
        assert_eq!(env.eval_expr(&expr, &state), 21.0);

        let expr = Expr::Calc(CalcOp::Mod(
            Box::new(Expr::Literal(17.0)),
            Box::new(Expr::Literal(5.0)),
        ));
        assert_eq!(env.eval_expr(&expr, &state), 2.0);
    }

    #[test]
    fn eval_var_from_state() {
        let functions = HashMap::new();
        let dispatch = HashMap::new();
        let mut env = EvalEnv::new(&functions, &dispatch);
        let mut state = State::default();
        state.registers[state::reg::AX] = 0x1234;

        let expr = Expr::Var {
            name: "--AX".to_string(),
            fallback: None,
        };
        assert_eq!(env.eval_expr(&expr, &state), 0x1234 as f64);
    }

    #[test]
    fn eval_var_fallback() {
        let functions = HashMap::new();
        let dispatch = HashMap::new();
        let mut env = EvalEnv::new(&functions, &dispatch);
        let state = State::default();

        let expr = Expr::Var {
            name: "--nonexistent".to_string(),
            fallback: Some(Box::new(Expr::Literal(99.0))),
        };
        assert_eq!(env.eval_expr(&expr, &state), 99.0);
    }

    #[test]
    fn eval_style_condition() {
        let functions = HashMap::new();
        let dispatch = HashMap::new();
        let mut env = EvalEnv::new(&functions, &dispatch);
        let mut state = State::default();
        state.registers[state::reg::AX] = 2;

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

        assert_eq!(env.eval_expr(&expr, &state), 200.0);
    }

    #[test]
    fn eval_round() {
        let functions = HashMap::new();
        let dispatch = HashMap::new();
        let mut env = EvalEnv::new(&functions, &dispatch);
        let state = State::default();

        let expr = Expr::Calc(CalcOp::Round(
            RoundStrategy::Down,
            Box::new(Expr::Literal(7.8)),
            Box::new(Expr::Literal(1.0)),
        ));
        assert_eq!(env.eval_expr(&expr, &state), 7.0);
    }

    #[test]
    fn eval_function_call() {
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

        let dispatch = HashMap::new();
        let mut env = EvalEnv::new(&functions, &dispatch);
        let state = State::default();

        let expr = Expr::FunctionCall {
            name: "--double".to_string(),
            args: vec![Expr::Literal(21.0)],
        };
        assert_eq!(env.eval_expr(&expr, &state), 42.0);
    }

    #[test]
    fn eval_dispatch_table() {
        let functions = HashMap::new();
        let mut dispatch = HashMap::new();

        let mut entries = HashMap::new();
        entries.insert(0, Expr::Literal(100.0));
        entries.insert(1, Expr::Literal(200.0));
        entries.insert(2, Expr::Literal(300.0));
        entries.insert(42, Expr::Literal(999.0));

        dispatch.insert(
            "--lookup".to_string(),
            crate::pattern::dispatch_table::DispatchTable {
                key_property: "--key".to_string(),
                entries,
                fallback: Expr::Literal(0.0),
            },
        );

        let mut env = EvalEnv::new(&functions, &dispatch);
        let state = State::default();

        // Look up key=42 → should return 999.0
        let expr = Expr::FunctionCall {
            name: "--lookup".to_string(),
            args: vec![Expr::Literal(42.0)],
        };
        assert_eq!(env.eval_expr(&expr, &state), 999.0);

        // Look up key=99 (missing) → should return fallback 0.0
        let expr = Expr::FunctionCall {
            name: "--lookup".to_string(),
            args: vec![Expr::Literal(99.0)],
        };
        assert_eq!(env.eval_expr(&expr, &state), 0.0);
    }

    #[test]
    fn tick_applies_assignments() {
        let program = ParsedProgram {
            properties: vec![],
            functions: vec![],
            assignments: vec![
                Assignment {
                    property: "--AX".to_string(),
                    value: Expr::Literal(42.0),
                },
                Assignment {
                    property: "--m0".to_string(),
                    value: Expr::Literal(255.0),
                },
            ],
        };

        let evaluator = Evaluator::from_parsed(&program);
        let mut state = State::default();

        let result = evaluator.tick(&mut state);

        assert_eq!(state.registers[state::reg::AX], 42);
        assert_eq!(state.memory[0], 255);
        assert_eq!(result.ticks_executed, 1);
        assert!(!result.changes.is_empty());
    }
}
