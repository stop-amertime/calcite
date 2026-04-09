//! The evaluator — runs compiled programs against the flat state.
//!
//! Two modes:
//! 1. **Interpreted**: Evaluates `Expr` trees directly against `State`.
//!    This is the Phase 1-2 path: parse CSS → evaluate expressions.
//! 2. **Compiled**: Uses pattern-recognised structures (dispatch tables,
//!    direct writes) for O(1) operations. (Phase 2+)

use std::collections::{HashMap, HashSet};

use crate::compile::{
    self, is_bit_extract, is_dispatch_identity_read, is_left_shift, is_mod_pow2, is_mul_refs,
    is_pow2_dispatch, is_right_shift, is_var_ref, CompiledProgram,
};
use crate::pattern::broadcast_write::{self, BroadcastWrite};
use crate::pattern::dispatch_table::{self, DispatchTable};
use crate::state::State;
use crate::types::*;

/// A value produced by expression evaluation — either numeric or string.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Number(f64),
    Str(String),
}

impl Value {
    /// Coerce to f64. Strings become 0.0.
    pub fn as_number(&self) -> f64 {
        match self {
            Value::Number(n) => *n,
            Value::Str(_) => 0.0,
        }
    }

    /// Coerce to String. Numbers become empty string.
    pub fn as_string(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            Value::Number(_) => String::new(),
        }
    }
}

/// A precomputed function body pattern for the interpreter fast-path.
///
/// Detected at construction time by analysing function body structure,
/// so no per-call overhead. These are generic mathematical patterns —
/// no function names are matched.
#[derive(Debug, Clone, Copy)]
enum FunctionPattern {
    /// result = var(param) — identity, return arg directly.
    Identity,
    /// mod(a, pow(2, b)) — bitmask.
    Bitmask,
    /// round(down, a / pow(2, b), 1) — right shift.
    RightShift,
    /// a * pow(2, b) or dispatch-table variant — left shift.
    LeftShift,
    /// mod(right_shift(a, b), 2) — bit extract.
    BitExtract,
    /// Dispatch table where key K → state[K] — direct memory read.
    IdentityRead,
    /// 16-bit word read (identity-read dispatch + word construction).
    IdentityRead16,
}

/// The main evaluator.
///
/// Holds both the immutable compiled program (functions, dispatch tables,
/// broadcast writes, assignments) and mutable per-tick evaluation state
/// (properties map, call depth). The properties map is allocated once and
/// reused across ticks via `clear()` to avoid per-tick allocation overhead.
pub struct Evaluator {
    /// Parsed @function definitions, keyed by name.
    pub functions: HashMap<String, FunctionDef>,
    /// Property assignments to execute each tick (in declaration order).
    pub assignments: Vec<Assignment>,
    /// Recognised dispatch tables for large if(style()) chains in functions.
    pub dispatch_tables: HashMap<String, DispatchTable>,
    /// Recognised broadcast write patterns.
    pub broadcast_writes: Vec<BroadcastWrite>,
    /// Precomputed function body patterns for the interpreter fast-path.
    function_patterns: HashMap<String, FunctionPattern>,
    /// Property values computed during the current tick. Reused across ticks.
    properties: HashMap<String, Value>,
    /// Call depth for recursion protection.
    call_depth: usize,
    /// String property assignments (evaluated via interpreter after compiled pass).
    string_assignments: Vec<Assignment>,
    /// Bare names of properties with string syntax (e.g., "textBuffer").
    string_property_names: HashSet<String>,
    /// Pre-tick hooks: called before each tick to handle external side effects.
    /// The evaluator has no knowledge of what hooks do — they are registered
    /// by the caller (CLI, WASM, etc.).
    #[allow(clippy::type_complexity)]
    pre_tick_hooks: Vec<Box<dyn Fn(&mut State)>>,
    /// Compiled bytecode program (flat ops over indexed slots).
    compiled: CompiledProgram,
    /// Slot array reused across ticks (avoids per-tick allocation).
    slots: Vec<i32>,
}

/// The result of running a batch of ticks.
#[derive(Debug, Clone, Default)]
pub struct TickResult {
    /// State changes as (property_name, new_value) pairs.
    pub changes: Vec<(String, String)>,
    /// Number of ticks executed.
    pub ticks_executed: u32,
}

impl Evaluator {
    /// Build an evaluator from a `ParsedProgram`.
    pub fn from_parsed(program: &ParsedProgram) -> Self {
        let _t = std::time::Instant::now();
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

        // Build and install the CSS-derived address map from identity-read dispatch tables.
        // Only override if we found actual mappings (preserves any pre-installed test map).
        let address_map = build_address_map(&dispatch_tables);
        if !address_map.is_empty() {
            log::info!(
                "Derived {} property→address mappings from CSS structure",
                address_map.len()
            );
            set_address_map(address_map);
        }

        log::info!("[compile phase] dispatch tables: {:.2}s", _t.elapsed().as_secs_f64());
        let _t = std::time::Instant::now();
        // Recognise broadcast write patterns in assignments
        let broadcast_result = broadcast_write::recognise_broadcast(&program.assignments);
        for bw in &broadcast_result.writes {
            log::info!(
                "Recognised broadcast write: {} → {} targets",
                bw.dest_property,
                bw.address_map.len(),
            );
        }

        log::info!("[compile phase] broadcast recognition: {:.2}s", _t.elapsed().as_secs_f64());
        let _t = std::time::Instant::now();
        // Identify string properties from @property syntax
        let mut string_property_names: HashSet<String> = HashSet::new();
        for prop in &program.properties {
            if matches!(&prop.syntax, PropertySyntax::Custom(s) if
                s.contains("content-list") || s.contains("string"))
                || matches!(&prop.syntax, PropertySyntax::Any)
                    && matches!(&prop.initial_value, Some(CssValue::String(_)))
            {
                let bare = to_bare_name(&prop.name);
                string_property_names.insert(bare.to_string());
                log::info!("Detected string property: {} (bare: {})", prop.name, bare);
            }
        }

        // Filter out:
        // 1. Assignments absorbed into broadcast writes (would overwrite with stale values)
        // 2. Triple-buffer copies (--__0*, --__1*, --__2*) which are no-ops in mutable state
        let all_assignments: Vec<Assignment> = program
            .assignments
            .iter()
            .filter(|a| {
                !broadcast_result.absorbed_properties.contains(&a.property)
                    && !is_buffer_copy(&a.property)
            })
            .cloned()
            .collect();

        // Partition: string properties go to interpreter, rest to compiler
        let (string_assignments, numeric_assignments): (Vec<_>, Vec<_>) = all_assignments
            .into_iter()
            .partition(|a| string_property_names.contains(to_bare_name(&a.property)));

        if !string_assignments.is_empty() {
            log::info!(
                "String assignments (interpreter path): {}",
                string_assignments
                    .iter()
                    .map(|a| a.property.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        log::info!("[compile phase] filter+partition: {:.2}s ({} numeric, {} string)", _t.elapsed().as_secs_f64(), numeric_assignments.len(), string_assignments.len());
        let _t = std::time::Instant::now();
        // Topological sort: reorder assignments by data dependencies.
        // CSS evaluates all properties simultaneously, but our sequential evaluator
        // must process them in dependency order. Only style() test references
        // (non-buffer-prefixed) create ordering constraints.
        let assignments = topological_sort_assignments(numeric_assignments, &functions);

        log::info!("[compile phase] topological sort: {:.2}s", _t.elapsed().as_secs_f64());
        let _t = std::time::Instant::now();
        let buffer_copies = program
            .assignments
            .iter()
            .filter(|a| is_buffer_copy(&a.property))
            .count();

        log::info!(
            "Assignments: {} numeric, {} string, {} absorbed into broadcast writes, {} buffer copies skipped",
            assignments.len(),
            string_assignments.len(),
            broadcast_result.absorbed_properties.len(),
            buffer_copies,
        );
        if log::log_enabled!(log::Level::Debug) {
            for a in &assignments {
                log::debug!("  kept: {}", a.property);
            }
        }

        log::info!("[compile phase] logging: {:.2}s", _t.elapsed().as_secs_f64());
        let _t = std::time::Instant::now();
        let compiled = compile::compile(
            &assignments,
            &broadcast_result.writes,
            &functions,
            &dispatch_tables,
        );

        log::info!("[compile phase] compile::compile: {:.2}s", _t.elapsed().as_secs_f64());
        log::info!(
            "Compiled: {} ops, {} slots, {} dispatch tables, {} broadcast writes",
            compiled.ops.len(),
            compiled.slot_count,
            compiled.dispatch_tables.len(),
            compiled.broadcast_writes.len(),
        );

        // Precompute function body patterns for the interpreter fast-path.
        let function_patterns = detect_function_patterns(&functions, &dispatch_tables);
        for (name, pattern) in &function_patterns {
            log::info!("Detected body pattern {:?} in {}", pattern, name);
        }

        let slot_count = compiled.slot_count as usize;
        let properties_capacity = assignments.len();
        Evaluator {
            functions,
            assignments,
            dispatch_tables,
            broadcast_writes: broadcast_result.writes,
            function_patterns,
            pre_tick_hooks: Vec::new(),
            properties: HashMap::with_capacity(properties_capacity),
            call_depth: 0,
            string_assignments,
            string_property_names,
            compiled,
            slots: Vec::with_capacity(slot_count),
        }
    }

    /// Register a pre-tick hook that is called before each tick.
    ///
    /// Hooks inspect and modify state to implement external side effects
    /// (e.g., text output, keyboard input). The evaluator itself has no
    /// knowledge of what the hook does.
    pub fn add_pre_tick_hook(&mut self, hook: Box<dyn Fn(&mut State)>) {
        self.pre_tick_hooks.push(hook);
    }

    /// Run a single tick: evaluate all assignments against the state.
    pub fn tick(&mut self, state: &mut State) -> TickResult {
        // Run pre-tick hooks (e.g., external function dispatch).
        for hook in &self.pre_tick_hooks {
            hook(state);
        }

        // Snapshot registers for change detection
        let prev_regs = state.registers;

        // Execute numeric assignments via compiled bytecode
        compile::execute(&self.compiled, state, &mut self.slots);


        // Evaluate string assignments via interpreter (after compiled pass,
        // so state is up-to-date for var() references).
        if !self.string_assignments.is_empty() {
            self.properties.clear();
            self.call_depth = 0;
            // Populate properties from compiled slot values so string eval
            // can read intermediate computed values (e.g., --instId, --instArg1).
            for (name, &slot) in &self.compiled.property_slots {
                if (slot as usize) < self.slots.len() {
                    self.properties.insert(
                        name.clone(),
                        Value::Number(self.slots[slot as usize] as f64),
                    );
                }
            }
            for i in 0..self.string_assignments.len() {
                let assignment = &self.string_assignments[i] as *const Assignment;
                let assignment = unsafe { &*assignment };
                let value = self.eval_expr(&assignment.value, state);
                let bare = to_bare_name(&assignment.property);
                log::trace!("String property {}: {:?}", assignment.property, value);
                if let Value::Str(ref s) = value {
                    state.string_properties.insert(bare.to_string(), s.clone());
                }
                self.properties.insert(assignment.property.clone(), value);
            }
        }

        // Detect register changes
        let mut changes = Vec::new();
        let reg_names = [
            "--AX", "--CX", "--DX", "--BX", "--SP", "--BP", "--SI", "--DI", "--IP", "--ES", "--CS",
            "--SS", "--DS", "--flags",
        ];
        for (i, name) in reg_names.iter().enumerate() {
            if state.registers[i] != prev_regs[i] {
                changes.push((name.to_string(), state.registers[i].to_string()));
            }
        }

        state.frame_counter += 1;

        TickResult {
            changes,
            ticks_executed: 1,
        }
    }

    /// Run a single tick using the interpreted path (for comparison testing).
    pub fn tick_interpreted(&mut self, state: &mut State) -> TickResult {
        // Run pre-tick hooks
        for hook in &self.pre_tick_hooks {
            hook(state);
        }

        self.properties.clear();
        self.call_depth = 0;

        let assignments_ptr = self.assignments.as_ptr();
        let assignments_len = self.assignments.len();
        for i in 0..assignments_len {
            let assignment = unsafe { &*assignments_ptr.add(i) };
            let value = self.eval_expr(&assignment.value, state);
            self.properties.insert(assignment.property.clone(), value);
        }

        let bw_ptr = self.broadcast_writes.as_ptr();
        let bw_len = self.broadcast_writes.len();
        for i in 0..bw_len {
            let bw = unsafe { &*bw_ptr.add(i) };
            let dest = self.resolve_property(&bw.dest_property, state).as_number();
            let dest_i64 = dest as i64;
            if bw.address_map.contains_key(&dest_i64) {
                let value = self.eval_expr(&bw.value_expr, state).as_number();
                state.write_mem(dest_i64 as i32, value as i32);
            }
            if !bw.spillover_map.is_empty() {
                if let Some(ref guard) = bw.spillover_guard {
                    let guard_val = self.resolve_property(guard, state).as_number();
                    if (guard_val as i64) == 1 {
                        if let Some((_var_name, val_expr)) = bw.spillover_map.get(&dest_i64) {
                            let value = self.eval_expr(val_expr, state).as_number();
                            state.write_mem(dest_i64 as i32 + 1, value as i32);
                        }
                    }
                }
            }
        }

        // Evaluate string assignments
        for i in 0..self.string_assignments.len() {
            let assignment = &self.string_assignments[i] as *const Assignment;
            let assignment = unsafe { &*assignment };
            let value = self.eval_expr(&assignment.value, state);
            let bare = to_bare_name(&assignment.property);
            if let Value::Str(ref s) = value {
                state.string_properties.insert(bare.to_string(), s.clone());
            }
            self.properties.insert(assignment.property.clone(), value);
        }

        let changes = self.apply_state(state);
        state.frame_counter += 1;

        TickResult {
            changes,
            ticks_executed: 1,
        }
    }

    /// Read a computed property value from the most recent tick.
    pub fn get_property(&self, name: &str) -> Option<f64> {
        self.properties.get(name).map(|v| v.as_number())
    }

    /// Read a compiled property's slot value after execution.
    ///
    /// Returns the value computed for `name` during the most recent tick.
    /// This reads directly from the slot array, so it works for compiled
    /// numeric properties (not string assignments).
    pub fn get_slot_value(&self, name: &str) -> Option<i32> {
        self.compiled
            .property_slots
            .get(name)
            .and_then(|&slot| self.slots.get(slot as usize).copied())
    }

    /// Run a batch of ticks, returning the net state diff across all ticks.
    ///
    /// Takes a snapshot before the batch and diffs at the end, so callers
    /// see every register/memory change — not just the final tick's delta.
    pub fn run_batch(&mut self, state: &mut State, count: u32) -> TickResult {
        let snapshot = state.clone();
        for _ in 0..count {
            self.tick(state);
        }

        // Diff registers
        let mut changes = Vec::new();
        let reg_names = [
            "--AX", "--CX", "--DX", "--BX", "--SP", "--BP", "--SI", "--DI", "--IP", "--ES", "--CS",
            "--SS", "--DS", "--flags",
        ];
        for (i, name) in reg_names.iter().enumerate() {
            if state.registers[i] != snapshot.registers[i] {
                changes.push((name.to_string(), state.registers[i].to_string()));
            }
        }

        TickResult {
            changes,
            ticks_executed: count,
        }
    }

    /// Apply computed property values to state and return the changes.
    ///
    /// Only writes canonical (non-prefixed) properties to state.
    /// Buffer copies (`--__0AX`, `--__1AX`, `--__2AX`) are skipped —
    /// they exist for x86CSS's triple-buffer pipeline but carry stale values
    /// that would nondeterministically overwrite the current tick's result.
    fn apply_state(&self, state: &mut State) -> Vec<(String, String)> {
        let mut changes = Vec::new();

        for (name, value) in &self.properties {
            // Skip buffer copies — only the canonical name should write to state
            if name.starts_with("--__0") || name.starts_with("--__1") || name.starts_with("--__2") {
                continue;
            }
            // Skip byte-half properties
            if is_byte_half(name) {
                continue;
            }
            match value {
                Value::Number(n) => {
                    let int_val = *n as i32;
                    if let Some(addr) = property_to_address(name) {
                        let old = state.read_mem(addr);
                        if old != int_val {
                            state.write_mem(addr, int_val);
                            changes.push((name.clone(), int_val.to_string()));
                        }
                    }
                }
                Value::Str(_) => {
                    // String properties are written to state in the tick loop directly
                }
            }
        }

        changes
    }
}

/// Reorder assignments by data dependencies.
///
/// CSS evaluates all custom properties simultaneously, but our sequential evaluator
/// processes them in declaration order. If assignment B's expression references
/// property A (via `var(--A)` or `style(--A: ...)`), then A must be computed
/// before B. This performs a topological sort on the dependency graph while
/// preserving original order where there are no constraints.
fn topological_sort_assignments(
    assignments: Vec<Assignment>,
    functions: &HashMap<String, FunctionDef>,
) -> Vec<Assignment> {
    if assignments.len() <= 1 {
        return assignments;
    }

    let _ts = std::time::Instant::now();
    // Build a set of properties defined in this tick
    let defined: HashMap<String, usize> = assignments
        .iter()
        .enumerate()
        .map(|(i, a)| (a.property.clone(), i))
        .collect();

    // For each assignment, collect which defined properties it references
    let mut deps: Vec<Vec<usize>> = Vec::with_capacity(assignments.len());
    let mut fn_cache: HashMap<String, Vec<usize>> = HashMap::new();
    for a in &assignments {
        let mut refs = Vec::new();
        collect_style_deps(&a.value, &defined, functions, &mut fn_cache, &mut refs);
        refs.sort_unstable();
        refs.dedup();
        // Remove self-references (e.g., --IP: calc(var(--IP) + 1))
        let self_idx = defined[&a.property];
        refs.retain(|&idx| idx != self_idx);
        deps.push(refs);
    }
    log::info!("[topo detail] dep collection ({} assignments): {:.2}s", assignments.len(), _ts.elapsed().as_secs_f64());
    let _ts = std::time::Instant::now();

    // Topological sort via Kahn's algorithm, breaking ties by original order.
    // We track ALL dependency edges — even those that are already in the right
    // order — because moving one node can invalidate previously-correct ordering.
    let n = assignments.len();
    let mut in_degree = vec![0usize; n];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, dep_list) in deps.iter().enumerate() {
        for &dep in dep_list {
            // dep must come before i
            in_degree[i] += 1;
            dependents[dep].push(i);
        }
    }

    // Use a min-heap keyed on original index to preserve order where possible
    let mut ready: std::collections::BinaryHeap<std::cmp::Reverse<usize>> =
        std::collections::BinaryHeap::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            ready.push(std::cmp::Reverse(i));
        }
    }

    log::info!("[topo detail] graph build: {:.2}s", _ts.elapsed().as_secs_f64());
    let _ts = std::time::Instant::now();
    let mut result = Vec::with_capacity(n);
    while let Some(std::cmp::Reverse(idx)) = ready.pop() {
        result.push(idx);
        for &dependent in &dependents[idx] {
            in_degree[dependent] -= 1;
            if in_degree[dependent] == 0 {
                ready.push(std::cmp::Reverse(dependent));
            }
        }
    }

    // If there's a cycle, fall back to original order
    if result.len() != n {
        log::warn!("Dependency cycle detected, keeping original assignment order");
        return assignments;
    }

    // Check if any reordering actually happened
    let reordered = result.iter().enumerate().any(|(pos, &orig)| pos != orig);
    if reordered {
        let moved: Vec<_> = result
            .iter()
            .enumerate()
            .filter(|(pos, &orig)| pos != &orig)
            .map(|(_, &orig)| &assignments[orig].property)
            .collect();
        log::info!(
            "Topological sort reordered {} assignments: {:?}",
            moved.len(),
            moved
        );
        if log::log_enabled!(log::Level::Info) {
            for (new_pos, &orig_idx) in result.iter().enumerate() {
                if new_pos != orig_idx {
                    log::info!(
                        "Topological sort: {} moved from position {} to {}",
                        assignments[orig_idx].property,
                        orig_idx,
                        new_pos
                    );
                }
            }
        }
    }

    // Reconstruct in sorted order
    let mut indexed: Vec<_> = assignments.into_iter().enumerate().collect();
    let mut sorted_assignments = Vec::with_capacity(n);
    // Create a position map: original_index → new_position
    let mut pos_map = vec![0usize; n];
    for (new_pos, &orig_idx) in result.iter().enumerate() {
        pos_map[orig_idx] = new_pos;
    }
    indexed.sort_by_key(|(orig_idx, _)| pos_map[*orig_idx]);
    for (_, a) in indexed {
        sorted_assignments.push(a);
    }
    sorted_assignments
}

/// Collect indices of defined properties that are referenced by `style()` tests
/// within an expression.
///
/// Only `style(--prop: ...)` comparisons create ordering dependencies — `var(--prop)`
/// references naturally fall back to the previous tick's state value. This conservative
/// approach preserves the original declaration order except where intra-tick evaluation
/// is required for correct branching.
fn collect_style_deps(
    expr: &Expr,
    defined: &HashMap<String, usize>,
    functions: &HashMap<String, FunctionDef>,
    fn_cache: &mut HashMap<String, Vec<usize>>,
    out: &mut Vec<usize>,
) {
    match expr {
        Expr::Var { name, fallback } => {
            // Track dependency on current-tick computed properties referenced via var().
            // Skip --__1X references (previous-tick values) as they don't need ordering.
            if !name.starts_with("--__") {
                if let Some(&idx) = defined.get(name) {
                    out.push(idx);
                }
            }
            if let Some(fb) = fallback {
                collect_style_deps(fb, defined, functions, fn_cache, out);
            }
        }
        Expr::Literal(_) | Expr::StringLiteral(_) => {}
        Expr::Concat(parts) => {
            for part in parts {
                collect_style_deps(part, defined, functions, fn_cache, out);
            }
        }
        Expr::Calc(op) => match op {
            CalcOp::Add(a, b)
            | CalcOp::Sub(a, b)
            | CalcOp::Mul(a, b)
            | CalcOp::Div(a, b)
            | CalcOp::Mod(a, b)
            | CalcOp::Pow(a, b) => {
                collect_style_deps(a, defined, functions, fn_cache, out);
                collect_style_deps(b, defined, functions, fn_cache, out);
            }
            CalcOp::Negate(a) | CalcOp::Abs(a) | CalcOp::Sign(a) => {
                collect_style_deps(a, defined, functions, fn_cache, out);
            }
            CalcOp::Clamp(a, b, c) => {
                collect_style_deps(a, defined, functions, fn_cache, out);
                collect_style_deps(b, defined, functions, fn_cache, out);
                collect_style_deps(c, defined, functions, fn_cache, out);
            }
            CalcOp::Round(_, a, b) => {
                collect_style_deps(a, defined, functions, fn_cache, out);
                collect_style_deps(b, defined, functions, fn_cache, out);
            }
            CalcOp::Min(args) | CalcOp::Max(args) => {
                for a in args {
                    collect_style_deps(a, defined, functions, fn_cache, out);
                }
            }
        },
        Expr::StyleCondition { branches, fallback } => {
            for branch in branches {
                collect_style_test_deps(&branch.condition, defined, functions, fn_cache, out);
                collect_style_deps(&branch.then, defined, functions, fn_cache, out);
            }
            collect_style_deps(fallback, defined, functions, fn_cache, out);
        }
        Expr::FunctionCall { name, args } => {
            for a in args {
                collect_style_deps(a, defined, functions, fn_cache, out);
            }
            // Recurse into function body — cached to avoid walking 1M-node ASTs repeatedly
            if functions.contains_key(name) {
                if let Some(cached) = fn_cache.get(name) {
                    out.extend_from_slice(cached);
                } else {
                    let func = &functions[name];
                    let mut body_deps = Vec::new();
                    collect_style_deps(&func.result, defined, functions, fn_cache, &mut body_deps);
                    for local in &func.locals {
                        collect_style_deps(&local.value, defined, functions, fn_cache, &mut body_deps);
                    }
                    fn_cache.insert(name.clone(), body_deps.clone());
                    out.extend_from_slice(&body_deps);
                }
            }
        }
    }
}

/// Collect property references from `style()` test conditions.
///
/// These are the dependencies that create ordering requirements — a `style(--X: val)`
/// comparison must see the current tick's value of `--X`.
fn collect_style_test_deps(
    test: &StyleTest,
    defined: &HashMap<String, usize>,
    functions: &HashMap<String, FunctionDef>,
    fn_cache: &mut HashMap<String, Vec<usize>>,
    out: &mut Vec<usize>,
) {
    match test {
        StyleTest::Single { property, value } => {
            if !property.starts_with("--__") {
                if let Some(&idx) = defined.get(property) {
                    out.push(idx);
                }
            }
            collect_style_deps(value, defined, functions, fn_cache, out);
        }
        StyleTest::And(tests) | StyleTest::Or(tests) => {
            for t in tests {
                collect_style_test_deps(t, defined, functions, fn_cache, out);
            }
        }
    }
}

/// Check if a property names a byte-half register (a split-register view).
///
/// In CSS, these are read-only views computed from the full register each tick.
/// The full register formula handles byte merging on writes, so byte halves
/// must NOT write back to state (they'd clobber with stale values).
///
/// Detection: byte-half addresses have values < -14 (below the full register
/// range). This is derived from the CSS structure, not hardcoded names.
fn is_byte_half(name: &str) -> bool {
    if let Some(addr) = property_to_address(name) {
        return addr < -14;
    }
    false
}

/// Check if a property is a triple-buffer copy (`--__0*`, `--__1*`, `--__2*`).
///
/// These assignments exist for x86CSS's animation pipeline but are no-ops
/// in calcite's mutable-state model — they just copy the canonical value
/// to a buffer slot that resolves back to the same value via `resolve_property`.
fn is_buffer_copy(name: &str) -> bool {
    name.starts_with("--__0") || name.starts_with("--__1") || name.starts_with("--__2")
}

/// Extract the bare register/memory name from a CSS custom property name.
///
/// Strips the `--` prefix and any triple-buffer prefix (`__0`, `__1`, `__2`):
/// - `"--AX"` → `"AX"`
/// - `"--__0AX"` → `"AX"`
/// - `"--__1flags"` → `"flags"`
/// - `"--m42"` → `"m42"`
fn to_bare_name(name: &str) -> &str {
    let after_dashes = &name[2..]; // skip leading "--"
    if let Some(rest) = after_dashes.strip_prefix("__0") {
        rest
    } else if let Some(rest) = after_dashes.strip_prefix("__1") {
        rest
    } else if let Some(rest) = after_dashes.strip_prefix("__2") {
        rest
    } else {
        after_dashes
    }
}

/// Map a CSS custom property name to a state address.
///
/// Uses a CSS-derived address map (from identity-read dispatch tables) when
/// available. Falls back to:
/// 1. `--m{N}` memory address parsing (generic).
/// 2. Register name heuristic (when no CSS-derived map is available).
///
/// Automatically strips triple-buffer prefixes (`--__0`, `--__1`, `--__2`).
pub fn property_to_address(name: &str) -> Option<i32> {
    let canonical = to_bare_name(name);

    // Check the CSS-derived address map first.
    let found = ADDRESS_MAP.with(|map| map.borrow().get(canonical).copied());
    if found.is_some() {
        return found;
    }

    // Generic fallback: --m{N} → memory address N
    if let Some(rest) = canonical.strip_prefix('m') {
        return parse_mem_address(rest);
    }

    // Register name fallback: standard register names (AX, IP, flags, etc.)
    // Always consulted — the CSS-derived map may contain only memory cells
    // (e.g., v2 i8086-css where --readMem maps m0..mN but not registers).
    register_name_heuristic(canonical)
}

/// Map standard register names to their state addresses.
///
/// Always consulted as a fallback after the CSS-derived address map.
/// The address map may only contain memory cells (e.g., from --readMem
/// identity-read dispatch) without register entries.
fn register_name_heuristic(name: &str) -> Option<i32> {
    use crate::state::addr;
    match name {
        "AX" => Some(addr::AX),
        "CX" => Some(addr::CX),
        "DX" => Some(addr::DX),
        "BX" => Some(addr::BX),
        "SP" => Some(addr::SP),
        "BP" => Some(addr::BP),
        "SI" => Some(addr::SI),
        "DI" => Some(addr::DI),
        "IP" => Some(addr::IP),
        "ES" => Some(addr::ES),
        "CS" => Some(addr::CS),
        "SS" => Some(addr::SS),
        "DS" => Some(addr::DS),
        "flags" => Some(addr::FLAGS),
        "AH" => Some(addr::AH),
        "CH" => Some(addr::CH),
        "DH" => Some(addr::DH),
        "BH" => Some(addr::BH),
        "AL" => Some(addr::AL),
        "CL" => Some(addr::CL),
        "DL" => Some(addr::DL),
        "BL" => Some(addr::BL),
        _ => None,
    }
}

// Thread-local address map, populated from CSS analysis.
// Maps bare property names (e.g., "AX", "flags") to state addresses.
// Populated by `set_address_map()` during `Evaluator::from_parsed()`.
thread_local! {
    static ADDRESS_MAP: std::cell::RefCell<HashMap<String, i32>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Install an address map derived from CSS structure for the current thread.
///
/// Called during `Evaluator::from_parsed()` so that all subsequent
/// `property_to_address()` calls use the CSS-derived mapping.
pub fn set_address_map(map: HashMap<String, i32>) {
    ADDRESS_MAP.with(|m| {
        *m.borrow_mut() = map;
    });
}

/// Detect video memory region from the address map.
///
/// Returns `Some((base_addr, size))` if contiguous addresses starting at
/// a VGA text-mode base (0xB8000) are found. Returns `None` otherwise.
pub fn detect_video_memory() -> Option<(usize, usize)> {
    ADDRESS_MAP.with(|m| {
        let map = m.borrow();
        // Find all addresses >= 0xB8000 (VGA text-mode segment)
        let vga_base = 0xB8000i32;
        let mut min_addr = i32::MAX;
        let mut max_addr = i32::MIN;
        let mut count = 0usize;
        for &addr in map.values() {
            if addr >= vga_base {
                min_addr = min_addr.min(addr);
                max_addr = max_addr.max(addr);
                count += 1;
            }
        }
        if count >= 100 && min_addr == vga_base {
            Some((min_addr as usize, (max_addr - min_addr + 1) as usize))
        } else {
            None
        }
    })
}

/// Build an address map by extracting property→address pairs from
/// identity-read dispatch tables in the CSS.
///
/// An identity-read dispatch table maps key K → `Var("--{name}")` where
/// the property at `--{name}` is at address K. Reversing this gives us
/// the property→address mapping without any hardcoded knowledge.
pub fn build_address_map(dispatch_tables: &HashMap<String, DispatchTable>) -> HashMap<String, i32> {
    let mut map = HashMap::new();
    for table in dispatch_tables.values() {
        // We need to check identity-read without recursing into property_to_address
        // (which depends on this map). Do a structural check: each entry's Var name
        // should parse to a bare name, and we record the key→bare_name association.
        if table.entries.len() < 4 {
            continue;
        }
        // Verify this is an identity-read table by checking consistency:
        // Each entry maps an integer key to a Var whose bare name is unique.
        let mut is_identity = true;
        let mut candidates: Vec<(String, i32)> = Vec::new();
        for (&key, expr) in &table.entries {
            if let Expr::Var { name, .. } = expr {
                let bare = to_bare_name(name).to_string();
                candidates.push((bare, key as i32));
            } else {
                is_identity = false;
                break;
            }
        }
        if is_identity {
            for (bare, addr) in candidates {
                map.insert(bare, addr);
            }
        }
    }
    map
}

/// Parse a memory address from digit chars without allocating (replaces `str::parse::<i32>()`).
fn parse_mem_address(s: &str) -> Option<i32> {
    if s.is_empty() {
        return None;
    }
    let mut result: i32 = 0;
    for &b in s.as_bytes() {
        if b.is_ascii_digit() {
            result = result.checked_mul(10)?.checked_add((b - b'0') as i32)?;
        } else {
            return None;
        }
    }
    Some(result)
}

/// Detect body patterns for all functions, returning a map of name → pattern.
///
/// Analyses function body structure to find mathematical patterns that can
/// be compiled to efficient native operations. Also checks dispatch tables
/// for identity-read patterns.
fn detect_function_patterns(
    functions: &HashMap<String, FunctionDef>,
    dispatch_tables: &HashMap<String, DispatchTable>,
) -> HashMap<String, FunctionPattern> {
    let mut patterns = HashMap::new();

    for (name, func) in functions {
        let params = &func.parameters;

        // Identity: 1 param, no locals, result = var(param)
        if params.len() == 1 && func.locals.is_empty() && is_var_ref(&func.result, &params[0].name)
        {
            patterns.insert(name.clone(), FunctionPattern::Identity);
            continue;
        }

        // 2-param patterns with no locals
        if params.len() == 2 && func.locals.is_empty() {
            let p0 = &params[0].name;
            let p1 = &params[1].name;

            if is_mod_pow2(&func.result, p0, p1) {
                patterns.insert(name.clone(), FunctionPattern::Bitmask);
                continue;
            }
            if is_right_shift(&func.result, p0, p1) {
                patterns.insert(name.clone(), FunctionPattern::RightShift);
                continue;
            }
            if is_left_shift(&func.result, p0, p1) {
                patterns.insert(name.clone(), FunctionPattern::LeftShift);
                continue;
            }
            if is_bit_extract(&func.result, p0, p1, functions) {
                patterns.insert(name.clone(), FunctionPattern::BitExtract);
                continue;
            }
        }

        // 2-param with 1 local: left-shift via power-of-2 dispatch table
        if params.len() == 2 && func.locals.len() == 1 {
            let p0 = &params[0].name;
            let p1 = &params[1].name;
            let local = &func.locals[0];

            if is_mul_refs(&func.result, p0, &local.name) && is_pow2_dispatch(&local.value, p1) {
                patterns.insert(name.clone(), FunctionPattern::LeftShift);
                continue;
            }
        }

        // Dispatch table identity-read
        if let Some(table) = dispatch_tables.get(name) {
            if is_dispatch_identity_read(table) {
                patterns.insert(name.clone(), FunctionPattern::IdentityRead);
                continue;
            }
        }
    }

    // Second pass: detect 16-bit read pattern.
    // A function that calls an identity-read function and constructs a word
    // (lo + hi*256) is a 16-bit read. We detect this by checking if the function
    // calls an IdentityRead function in its body.
    let identity_read_names: Vec<String> = patterns
        .iter()
        .filter(|(_, p)| matches!(p, FunctionPattern::IdentityRead))
        .map(|(n, _)| n.clone())
        .collect();

    for (name, func) in functions {
        if patterns.contains_key(name) {
            continue;
        }
        // Check: does the function body call an identity-read function?
        // Pattern: result involves read_fn(addr) + read_fn(addr+1) * 256
        // or uses a sign check to differentiate register (single) vs memory (word).
        if func.parameters.len() == 1 && calls_identity_read(&func.result, &identity_read_names) {
            patterns.insert(name.clone(), FunctionPattern::IdentityRead16);
        }
    }

    patterns
}

/// Check if an expression tree calls any of the named identity-read functions.
fn calls_identity_read(expr: &Expr, identity_read_names: &[String]) -> bool {
    match expr {
        Expr::FunctionCall { name, .. } => {
            if identity_read_names.contains(name) {
                return true;
            }
        }
        Expr::Calc(op) => {
            return match op {
                CalcOp::Add(a, b)
                | CalcOp::Sub(a, b)
                | CalcOp::Mul(a, b)
                | CalcOp::Div(a, b)
                | CalcOp::Mod(a, b)
                | CalcOp::Pow(a, b) => {
                    calls_identity_read(a, identity_read_names)
                        || calls_identity_read(b, identity_read_names)
                }
                CalcOp::Negate(a) | CalcOp::Abs(a) | CalcOp::Sign(a) => {
                    calls_identity_read(a, identity_read_names)
                }
                CalcOp::Clamp(a, b, c) => {
                    calls_identity_read(a, identity_read_names)
                        || calls_identity_read(b, identity_read_names)
                        || calls_identity_read(c, identity_read_names)
                }
                CalcOp::Round(_, a, b) => {
                    calls_identity_read(a, identity_read_names)
                        || calls_identity_read(b, identity_read_names)
                }
                CalcOp::Min(args) | CalcOp::Max(args) => args
                    .iter()
                    .any(|a| calls_identity_read(a, identity_read_names)),
            };
        }
        Expr::StyleCondition {
            branches, fallback, ..
        } => {
            if calls_identity_read(fallback, identity_read_names) {
                return true;
            }
            for branch in branches {
                if calls_identity_read(&branch.then, identity_read_names) {
                    return true;
                }
            }
        }
        Expr::Concat(parts) => {
            for part in parts {
                if calls_identity_read(part, identity_read_names) {
                    return true;
                }
            }
        }
        _ => {}
    }
    false
}

const MAX_CALL_DEPTH: usize = 64;

// --- Evaluation methods ---
//
// These methods need &mut self to write to self.properties/call_depth, but also
// read self.functions/dispatch_tables. We use raw pointers in eval_function_call
// and eval_dispatch_raw to read from immutable program data while mutating
// properties. This is safe because functions/dispatch_tables are never modified
// during evaluation.

impl Evaluator {
    /// Evaluate an expression to a `Value` (number or string).
    fn eval_expr(&mut self, expr: &Expr, state: &State) -> Value {
        match expr {
            Expr::Literal(v) => Value::Number(*v),

            Expr::Var { name, fallback } => {
                let v = self.resolve_property(name, state);
                match &v {
                    Value::Number(n) if *n != 0.0 => return v,
                    Value::Str(s) if !s.is_empty() => return v,
                    _ => {}
                }
                // Property might genuinely be 0/empty, or might not exist.
                if self.properties.contains_key(name.as_str()) {
                    return v;
                }
                if name.starts_with("--__") && name.len() > 5 {
                    let canonical = format!("--{}", &name[5..]);
                    if self.properties.contains_key(&canonical) {
                        return v;
                    }
                }
                if property_to_address(name).is_some() {
                    return v;
                }
                if let Some(fb) = fallback {
                    return self.eval_expr(fb, state);
                }
                log::debug!("undefined variable: {name}");
                v
            }

            Expr::StringLiteral(s) => Value::Str(s.clone()),

            Expr::Calc(op) => Value::Number(self.eval_calc(op, state)),

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

            Expr::Concat(parts) => {
                let mut result = String::new();
                for part in parts {
                    match self.eval_expr(part, state) {
                        Value::Str(s) => result.push_str(&s),
                        Value::Number(_) => {}
                    }
                }
                Value::Str(result)
            }
        }
    }

    /// Evaluate a `CalcOp` (always numeric).
    fn eval_calc(&mut self, op: &CalcOp, state: &State) -> f64 {
        match op {
            CalcOp::Add(a, b) => {
                self.eval_expr(a, state).as_number() + self.eval_expr(b, state).as_number()
            }
            CalcOp::Sub(a, b) => {
                self.eval_expr(a, state).as_number() - self.eval_expr(b, state).as_number()
            }
            CalcOp::Mul(a, b) => {
                self.eval_expr(a, state).as_number() * self.eval_expr(b, state).as_number()
            }
            CalcOp::Div(a, b) => {
                let divisor = self.eval_expr(b, state).as_number();
                if divisor == 0.0 {
                    0.0
                } else {
                    self.eval_expr(a, state).as_number() / divisor
                }
            }
            CalcOp::Mod(a, b) => {
                let divisor = self.eval_expr(b, state).as_number();
                if divisor == 0.0 {
                    0.0
                } else {
                    self.eval_expr(a, state).as_number() % divisor
                }
            }
            CalcOp::Min(args) => args
                .iter()
                .map(|a| self.eval_expr(a, state).as_number())
                .fold(f64::INFINITY, f64::min),
            CalcOp::Max(args) => args
                .iter()
                .map(|a| self.eval_expr(a, state).as_number())
                .fold(f64::NEG_INFINITY, f64::max),
            CalcOp::Clamp(min, val, max) => {
                let min_v = self.eval_expr(min, state).as_number();
                let val_v = self.eval_expr(val, state).as_number();
                let max_v = self.eval_expr(max, state).as_number();
                val_v.clamp(min_v, max_v)
            }
            CalcOp::Round(strategy, val, interval) => {
                let v = self.eval_expr(val, state).as_number();
                let i = self.eval_expr(interval, state).as_number();
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
            CalcOp::Pow(base, exp) => self
                .eval_expr(base, state)
                .as_number()
                .powf(self.eval_expr(exp, state).as_number()),
            CalcOp::Sign(val) => {
                let v = self.eval_expr(val, state).as_number();
                if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            CalcOp::Abs(val) => self.eval_expr(val, state).as_number().abs(),
            CalcOp::Negate(val) => -self.eval_expr(val, state).as_number(),
        }
    }

    /// Resolve a property value: check computed properties, then state.
    ///
    /// For buffer-prefixed names (`--__1AX`), also checks the canonical name (`--AX`)
    /// in computed properties before falling back to state.
    fn resolve_property(&self, name: &str, state: &State) -> Value {
        if let Some(v) = self.properties.get(name) {
            return v.clone();
        }
        if name.starts_with("--__") && name.len() > 5 {
            let suffix = &name[5..];
            if !self.properties.is_empty() {
                let canonical = format!("--{suffix}");
                if let Some(v) = self.properties.get(&canonical) {
                    return v.clone();
                }
            }
            // Check string properties on state (for buffer-prefixed string vars)
            if let Some(s) = state.string_properties.get(suffix) {
                return Value::Str(s.clone());
            }
        }
        // Check string properties on state by bare name
        let bare = to_bare_name(name);
        if let Some(s) = state.string_properties.get(bare) {
            return Value::Str(s.clone());
        }
        if let Some(addr) = property_to_address(name) {
            return Value::Number(state.read_mem(addr) as f64);
        }
        // Special I/O properties not mapped to the register/memory address space
        if name == "--keyboard" || name == "--__1keyboard" || name == "--__2keyboard" {
            return Value::Number(state.keyboard as f64);
        }
        // Default: empty string for string properties, 0 for numeric
        if self.string_property_names.contains(bare) {
            return Value::Str(String::new());
        }
        Value::Number(0.0)
    }

    /// Evaluate a style test (condition inside an `if()` branch).
    fn eval_style_test(&mut self, test: &StyleTest, state: &State) -> bool {
        match test {
            StyleTest::Single { property, value } => {
                let prop_val = self.resolve_property(property, state).as_number() as i64;
                let test_val = self.eval_expr(value, state).as_number() as i64;
                prop_val == test_val
            }
            StyleTest::And(tests) => tests.iter().all(|t| self.eval_style_test(t, state)),
            StyleTest::Or(tests) => tests.iter().any(|t| self.eval_style_test(t, state)),
        }
    }

    /// Evaluate a @function call.
    fn eval_function_call(&mut self, name: &str, args: &[Expr], state: &State) -> Value {
        if self.call_depth >= MAX_CALL_DEPTH {
            log::warn!("max call depth exceeded calling {name}");
            return Value::Number(0.0);
        }

        // Fast paths based on precomputed body-pattern analysis.
        // These match mathematical patterns in function bodies, not names.
        if let Some(&pattern) = self.function_patterns.get(name) {
            match pattern {
                FunctionPattern::Identity => {
                    if let Some(arg) = args.first() {
                        return self.eval_expr(arg, state);
                    }
                    return Value::Number(0.0);
                }
                FunctionPattern::IdentityRead => {
                    if let Some(arg) = args.first() {
                        let addr = self.eval_expr(arg, state).as_number() as i32;
                        return Value::Number(state.read_mem(addr) as f64);
                    }
                    return Value::Number(0.0);
                }
                FunctionPattern::IdentityRead16 => {
                    if let Some(arg) = args.first() {
                        let addr = self.eval_expr(arg, state).as_number() as i32;
                        if addr < 0 {
                            return Value::Number(state.read_mem(addr) as f64);
                        }
                        return Value::Number(state.read_mem16(addr) as f64);
                    }
                    return Value::Number(0.0);
                }
                FunctionPattern::Bitmask => {
                    if args.len() >= 2 {
                        let a = self.eval_expr(&args[0], state).as_number() as i64;
                        let b = self.eval_expr(&args[1], state).as_number() as u32;
                        if b >= 64 {
                            return Value::Number(a as f64);
                        }
                        return Value::Number((a & ((1i64 << b) - 1)) as f64);
                    }
                    return Value::Number(0.0);
                }
                FunctionPattern::RightShift => {
                    if args.len() >= 2 {
                        let a = self.eval_expr(&args[0], state).as_number() as i64;
                        let b = self.eval_expr(&args[1], state).as_number() as u32;
                        if b >= 64 {
                            return Value::Number(0.0);
                        }
                        return Value::Number((a >> b) as f64);
                    }
                    return Value::Number(0.0);
                }
                FunctionPattern::LeftShift => {
                    if args.len() >= 2 {
                        let a = self.eval_expr(&args[0], state).as_number() as i64;
                        let b = self.eval_expr(&args[1], state).as_number() as u32;
                        if b >= 64 {
                            return Value::Number(0.0);
                        }
                        return Value::Number((a << b) as f64);
                    }
                    return Value::Number(0.0);
                }
                FunctionPattern::BitExtract => {
                    if args.len() >= 2 {
                        let val = self.eval_expr(&args[0], state).as_number() as i64;
                        let idx = self.eval_expr(&args[1], state).as_number() as u32;
                        if idx >= 64 {
                            return Value::Number(0.0);
                        }
                        return Value::Number(((val >> idx) & 1) as f64);
                    }
                    return Value::Number(0.0);
                }
            }
        }

        // Check for a dispatch table optimisation.
        if let Some(table) = self.dispatch_tables.get(name) {
            let table_key = &table.key_property as *const String;
            let table_entries = &table.entries as *const HashMap<i64, Expr>;
            let table_fallback = &table.fallback as *const Expr;
            // SAFETY: dispatch_tables is not modified during evaluation.
            return unsafe {
                self.eval_dispatch_raw(
                    name,
                    &*table_key,
                    &*table_entries,
                    &*table_fallback,
                    args,
                    state,
                )
            };
        }

        let func = match self.functions.get(name) {
            Some(f) => f as *const FunctionDef,
            None => {
                log::debug!("undefined function: {name}");
                return Value::Number(0.0);
            }
        };
        // SAFETY: functions is not modified during evaluation.
        let func = unsafe { &*func };

        self.call_depth += 1;

        // Bind arguments to parameter names
        let old_props: Vec<(String, Option<Value>)> = func
            .parameters
            .iter()
            .enumerate()
            .map(|(i, param)| {
                let old = self.properties.get(&param.name).cloned();
                let val = args
                    .get(i)
                    .map(|a| self.eval_expr(a, state))
                    .unwrap_or(Value::Number(0.0));
                self.properties.insert(param.name.clone(), val);
                (param.name.clone(), old)
            })
            .collect();

        // Evaluate local variables
        let old_locals: Vec<(String, Option<Value>)> = func
            .locals
            .iter()
            .map(|local| {
                let old = self.properties.get(&local.name).cloned();
                let val = self.eval_expr(&local.value, state);
                self.properties.insert(local.name.clone(), val);
                (local.name.clone(), old)
            })
            .collect();

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
    ///
    /// SAFETY: `entries` and `fallback` must point into self.dispatch_tables,
    /// which is not modified during evaluation.
    unsafe fn eval_dispatch_raw(
        &mut self,
        name: &str,
        key_property: &str,
        entries: &HashMap<i64, Expr>,
        fallback: &Expr,
        args: &[Expr],
        state: &State,
    ) -> Value {
        let func = self.functions.get(name).map(|f| f as *const FunctionDef);
        let old_props: Vec<(String, Option<Value>)> = if let Some(func_ptr) = func {
            let func = &*func_ptr;
            func.parameters
                .iter()
                .enumerate()
                .map(|(i, param)| {
                    let old = self.properties.get(&param.name).cloned();
                    let val = args
                        .get(i)
                        .map(|a| self.eval_expr(a, state))
                        .unwrap_or(Value::Number(0.0));
                    self.properties.insert(param.name.clone(), val);
                    (param.name.clone(), old)
                })
                .collect()
        } else {
            Vec::new()
        };

        let key = self.resolve_property(key_property, state).as_number() as i64;

        let result = if let Some(result_expr) = entries.get(&key) {
            self.eval_expr(result_expr, state)
        } else {
            self.eval_expr(fallback, state)
        };

        for (name, old) in old_props {
            match old {
                Some(v) => {
                    self.properties.insert(name, v);
                }
                None => {
                    self.properties.remove(&name);
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state;

    /// Install a test address map with standard register mappings.
    ///
    /// Tests that reference `--AX`, `--CX`, etc. must call this to set up the
    /// CSS-derived address mapping that the evaluator and compiler depend on.
    fn install_test_address_map() {
        use crate::state::addr;
        let mut map = HashMap::new();
        map.insert("AX".to_string(), addr::AX);
        map.insert("CX".to_string(), addr::CX);
        map.insert("DX".to_string(), addr::DX);
        map.insert("BX".to_string(), addr::BX);
        map.insert("SP".to_string(), addr::SP);
        map.insert("BP".to_string(), addr::BP);
        map.insert("SI".to_string(), addr::SI);
        map.insert("DI".to_string(), addr::DI);
        map.insert("IP".to_string(), addr::IP);
        map.insert("ES".to_string(), addr::ES);
        map.insert("CS".to_string(), addr::CS);
        map.insert("SS".to_string(), addr::SS);
        map.insert("DS".to_string(), addr::DS);
        map.insert("flags".to_string(), addr::FLAGS);
        map.insert("AH".to_string(), addr::AH);
        map.insert("CH".to_string(), addr::CH);
        map.insert("DH".to_string(), addr::DH);
        map.insert("BH".to_string(), addr::BH);
        map.insert("AL".to_string(), addr::AL);
        map.insert("CL".to_string(), addr::CL);
        map.insert("DL".to_string(), addr::DL);
        map.insert("BL".to_string(), addr::BL);
        set_address_map(map);
    }

    /// Helper: create a minimal Evaluator for unit tests (no assignments/patterns).
    fn test_evaluator(
        functions: HashMap<String, FunctionDef>,
        dispatch_tables: HashMap<String, DispatchTable>,
    ) -> Evaluator {
        install_test_address_map();
        let compiled = crate::compile::compile(&[], &[], &functions, &dispatch_tables);
        let function_patterns = detect_function_patterns(&functions, &dispatch_tables);
        Evaluator {
            functions,
            assignments: vec![],
            dispatch_tables,
            broadcast_writes: vec![],
            function_patterns,
            pre_tick_hooks: Vec::new(),
            properties: HashMap::with_capacity(16),
            call_depth: 0,
            string_assignments: vec![],
            string_property_names: HashSet::new(),
            compiled,
            slots: Vec::new(),
        }
    }

    #[test]
    fn eval_literal() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let state = State::default();
        assert_eq!(
            eval.eval_expr(&Expr::Literal(42.0), &state).as_number(),
            42.0
        );
    }

    #[test]
    fn eval_calc_operations() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let state = State::default();

        let expr = Expr::Calc(CalcOp::Add(
            Box::new(Expr::Literal(10.0)),
            Box::new(Expr::Literal(20.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 30.0);

        let expr = Expr::Calc(CalcOp::Mul(
            Box::new(Expr::Literal(3.0)),
            Box::new(Expr::Literal(7.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 21.0);

        let expr = Expr::Calc(CalcOp::Mod(
            Box::new(Expr::Literal(17.0)),
            Box::new(Expr::Literal(5.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 2.0);
    }

    #[test]
    fn eval_var_from_state() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let mut state = State::default();
        state.registers[state::reg::AX] = 0x1234;

        let expr = Expr::Var {
            name: "--AX".to_string(),
            fallback: None,
        };
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 0x1234 as f64);
    }

    #[test]
    fn eval_var_fallback() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let state = State::default();

        let expr = Expr::Var {
            name: "--nonexistent".to_string(),
            fallback: Some(Box::new(Expr::Literal(99.0))),
        };
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 99.0);
    }

    #[test]
    fn eval_style_condition() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
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

        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 200.0);
    }

    #[test]
    fn eval_round() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let state = State::default();

        let expr = Expr::Calc(CalcOp::Round(
            RoundStrategy::Down,
            Box::new(Expr::Literal(7.8)),
            Box::new(Expr::Literal(1.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 7.0);
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

        let mut eval = test_evaluator(functions, HashMap::new());
        let state = State::default();

        let expr = Expr::FunctionCall {
            name: "--double".to_string(),
            args: vec![Expr::Literal(21.0)],
        };
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 42.0);
    }

    #[test]
    fn eval_dispatch_table() {
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

        let mut eval = test_evaluator(functions, dispatch);
        let state = State::default();

        let expr = Expr::FunctionCall {
            name: "--lookup".to_string(),
            args: vec![Expr::Literal(42.0)],
        };
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 999.0);

        let expr = Expr::FunctionCall {
            name: "--lookup".to_string(),
            args: vec![Expr::Literal(99.0)],
        };
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 0.0);
    }

    #[test]
    fn tick_applies_assignments() {
        install_test_address_map();
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

        let mut evaluator = Evaluator::from_parsed(&program);
        let mut state = State::default();

        let result = evaluator.tick(&mut state);

        assert_eq!(state.registers[state::reg::AX], 42);
        assert_eq!(state.memory[0], 255);
        assert_eq!(result.ticks_executed, 1);
        assert!(!result.changes.is_empty());
    }

    #[test]
    fn parse_mem_address_valid() {
        assert_eq!(parse_mem_address("0"), Some(0));
        assert_eq!(parse_mem_address("42"), Some(42));
        assert_eq!(parse_mem_address("1585"), Some(1585));
    }

    #[test]
    fn parse_mem_address_invalid() {
        assert_eq!(parse_mem_address(""), None);
        assert_eq!(parse_mem_address("abc"), None);
        assert_eq!(parse_mem_address("12x"), None);
    }

    #[test]
    fn eval_division_by_zero() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let state = State::default();

        let expr = Expr::Calc(CalcOp::Div(
            Box::new(Expr::Literal(100.0)),
            Box::new(Expr::Literal(0.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 0.0);
    }

    #[test]
    fn eval_mod_by_zero() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let state = State::default();

        let expr = Expr::Calc(CalcOp::Mod(
            Box::new(Expr::Literal(17.0)),
            Box::new(Expr::Literal(0.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 0.0);
    }

    #[test]
    fn eval_negate() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let state = State::default();

        let expr = Expr::Calc(CalcOp::Negate(Box::new(Expr::Literal(42.0))));
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), -42.0);

        let expr = Expr::Calc(CalcOp::Negate(Box::new(Expr::Literal(-7.0))));
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 7.0);
    }

    #[test]
    fn eval_sign_and_abs() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let state = State::default();

        assert_eq!(
            eval.eval_expr(
                &Expr::Calc(CalcOp::Sign(Box::new(Expr::Literal(42.0)))),
                &state
            )
            .as_number(),
            1.0
        );
        assert_eq!(
            eval.eval_expr(
                &Expr::Calc(CalcOp::Sign(Box::new(Expr::Literal(-5.0)))),
                &state
            )
            .as_number(),
            -1.0
        );
        assert_eq!(
            eval.eval_expr(
                &Expr::Calc(CalcOp::Sign(Box::new(Expr::Literal(0.0)))),
                &state
            )
            .as_number(),
            0.0
        );
        assert_eq!(
            eval.eval_expr(
                &Expr::Calc(CalcOp::Abs(Box::new(Expr::Literal(-99.0)))),
                &state
            )
            .as_number(),
            99.0
        );
    }

    #[test]
    fn eval_clamp() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let state = State::default();

        // Value within range
        let expr = Expr::Calc(CalcOp::Clamp(
            Box::new(Expr::Literal(0.0)),
            Box::new(Expr::Literal(50.0)),
            Box::new(Expr::Literal(100.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 50.0);

        // Value below min
        let expr = Expr::Calc(CalcOp::Clamp(
            Box::new(Expr::Literal(10.0)),
            Box::new(Expr::Literal(5.0)),
            Box::new(Expr::Literal(100.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 10.0);

        // Value above max
        let expr = Expr::Calc(CalcOp::Clamp(
            Box::new(Expr::Literal(0.0)),
            Box::new(Expr::Literal(200.0)),
            Box::new(Expr::Literal(100.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 100.0);
    }

    #[test]
    fn eval_max_call_depth() {
        // Create a function that calls itself (infinite recursion)
        let mut functions = HashMap::new();
        functions.insert(
            "--recurse".to_string(),
            FunctionDef {
                name: "--recurse".to_string(),
                parameters: vec![],
                locals: vec![],
                result: Expr::FunctionCall {
                    name: "--recurse".to_string(),
                    args: vec![],
                },
            },
        );

        let mut eval = test_evaluator(functions, HashMap::new());
        let state = State::default();

        // Should not panic — returns 0 when depth exceeded
        let expr = Expr::FunctionCall {
            name: "--recurse".to_string(),
            args: vec![],
        };
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 0.0);
    }
}
