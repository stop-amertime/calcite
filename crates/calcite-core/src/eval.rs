//! The evaluator — runs compiled programs against the flat state.
//!
//! Two modes:
//! 1. **Interpreted**: Evaluates `Expr` trees directly against `State`.
//!    This is the Phase 1-2 path: parse CSS → evaluate expressions.
//! 2. **Compiled**: Uses pattern-recognised structures (dispatch tables,
//!    direct writes) for O(1) operations. (Phase 2+)

use std::collections::{HashMap, HashSet};
use web_time::Instant;

use crate::compile::{
    self, is_bit_extract, is_dispatch_identity_read, is_left_shift,
    is_mod_pow2, is_mul_refs, is_pow2_dispatch, is_right_shift, is_var_ref, CompiledProgram,
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
    /// Recognised packed-cell broadcast ports (one per CSS-DOS write slot).
    /// Each port covers all `--mcN` cell assignments absorbed into a single
    /// (gate, addr, val) shape. The interpreter runs these after the
    /// per-assignment loop so packed cabinets see the same memory writes
    /// the compiled path produces via `Op::PackedBroadcast*`.
    pub packed_broadcast_ports: Vec<crate::pattern::packed_broadcast_write::PackedSlotPort>,
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
    pre_tick_hooks: Vec<Box<dyn Fn(&mut State) + Send + Sync>>,
    /// Compiled bytecode program (flat ops over indexed slots).
    pub(crate) compiled: CompiledProgram,
    /// Slot array reused across ticks (avoids per-tick allocation).
    slots: Vec<i32>,
    /// Input-edge bindings recognised from nested `:has(…:pseudo)` rules.
    /// Each entry compiles a single `&:has(#SELECTOR:PSEUDO) { --PROP: VAL; }`
    /// rule into something we can apply per-tick. `value` is pre-evaluated
    /// at compile time as a literal i32 — non-literal RHS values fall back
    /// to 0 with a warning (the recogniser is structural; the value-side
    /// expression evaluator isn't worth the complexity for the current
    /// shapes kiln emits).
    pub(crate) input_edge_bindings: Vec<InputEdgeBinding>,
    /// Self-loop descriptors recognised at compile time from the
    /// cabinet's opcode-keyed dispatch family. Phase 1 of the
    /// rep_fast_forward genericity mission emits these but does not
    /// yet consume them at runtime — the hardcoded fast-forward path
    /// remains the active applier.
    pub loop_descriptors: Vec<crate::pattern::loop_descriptor::LoopDescriptor>,
}

/// Compiled form of a `ParsedProgram::input_edges` entry. The
/// state-var slot is resolved at `wire_state_for_input_edges` time
/// (right after state vars are loaded) and the result is installed on
/// `State::input_edge_groups`; the Evaluator only keeps the raw
/// bindings for host introspection via `input_edges_for_host`.
#[derive(Debug, Clone)]
pub(crate) struct InputEdgeBinding {
    pub property: String,
    pub pseudo: String,
    pub selector: String,
    pub value: i32,
}

/// Granular timing breakdown from [`Evaluator::tick_profiled`].
#[derive(Debug, Clone, Default)]
pub struct TickProfile {
    // --- Top-level phases ---
    /// Time in pre-tick hooks (seconds).
    pub hooks: f64,
    /// Time cloning state vars for change detection (seconds).
    pub snapshot: f64,
    /// Time detecting state variable changes (seconds).
    pub change_detect: f64,
    /// Time evaluating string assignments via interpreter (seconds).
    pub string_eval: f64,

    // --- Inside execute (the 98% chunk) ---
    /// Time resetting/resizing the slot array.
    pub slot_reset: f64,
    /// Time executing the main ops bytecode stream (total).
    pub main_ops: f64,
    /// Time spent inside dispatch table lookups (subset of main_ops).
    pub dispatch_time: f64,
    /// Time spent on linear ops excluding dispatches (main_ops - dispatch_time).
    pub linear_ops: f64,
    /// Time in the writeback loop (slots → state).
    pub writeback: f64,
    /// Time in broadcast write evaluation.
    pub broadcast: f64,
    /// Number of main-stream ops stepped (excluding dispatch sub-ops).
    pub main_ops_count: u64,
    /// Number of dispatch table lookups.
    pub dispatch_count: u64,
    /// Number of ops executed inside dispatch sub-programs.
    pub dispatch_sub_ops_count: u64,
    /// Number of branches taken (BranchIfZero that jumped).
    pub branches_taken: u64,
    /// Number of branches not taken.
    pub branches_not_taken: u64,
    /// Number of broadcast writes that fired (dest in address_map).
    pub broadcast_fire_count: u64,
    /// Total number of broadcast writes checked.
    pub broadcast_total_count: u64,
    /// Number of writeback entries.
    pub writeback_count: u64,
    /// Number of writeback entries that actually changed state.
    pub writeback_changed_count: u64,
    /// Per-op-type execution counts (op discriminant name → count).
    pub op_counts: HashMap<&'static str, u64>,
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
    /// Access the compiled program (for diagnostic/debug probes).
    #[doc(hidden)]
    pub fn compiled(&self) -> &CompiledProgram {
        &self.compiled
    }

    /// Populate `state.packed_cell_table` + `packed_cell_size` so
    /// `State::read_mem` / `write_mem` can route positive guest addresses
    /// through the packed `mcN` state vars instead of the flat shadow
    /// (which isn't populated for packed cabinets).
    ///
    /// Every consumer of `State` + `Evaluator` MUST call this after
    /// `state.load_properties` and before running any ticks — the wasm
    /// runtime, the CLI, the debugger, and any test harness that holds its
    /// own State. Forgetting this call is a "reads return 0 even though
    /// the CPU wrote them correctly" footgun because the write path goes
    /// directly to `state_vars[]` via slot addresses while the read path
    /// sees an empty cell table and falls through to the unused shadow.
    ///
    /// The table chosen is:
    /// - `compiled.packed_cell_tables[0]` if present — that's the dense
    ///   lookup the runtime `LoadPackedByte` op uses. It covers every
    ///   cell the CSS reads from, which is a superset of what any single
    ///   write-port covers.
    /// - Otherwise the union of every `packed_broadcast_writes[i].cell_table`
    ///   merged by cell index (entries from lower-slot ports win, matching
    ///   the write-priority order the evaluator applies per-tick).
    /// - If neither table exists, the cabinet isn't packed; nothing to do.
    pub fn wire_state_for_packed_memory(&self, state: &mut crate::State) {
        let compiled = &self.compiled;
        // Read-side table is the most complete; prefer it.
        if let Some(first) = compiled.packed_cell_tables.first() {
            state.packed_cell_table = first.clone();
            let pack = compiled
                .packed_broadcast_writes
                .first()
                .map(|bw| bw.pack)
                .unwrap_or(2);
            state.packed_cell_size = pack;
            return;
        }
        // Fallback: merge all the write-port tables. Lower slot index wins
        // on collision — that matches the "slot 0 applies last" semantics
        // so collision winners stay stable whether you read or write.
        if !compiled.packed_broadcast_writes.is_empty() {
            let mut merged: Vec<i32> = Vec::new();
            let pack = compiled.packed_broadcast_writes[0].pack;
            for port in compiled.packed_broadcast_writes.iter().rev() {
                if port.cell_table.len() > merged.len() {
                    merged.resize(port.cell_table.len(), 0);
                }
                for (i, &cell_addr) in port.cell_table.iter().enumerate() {
                    if cell_addr != 0 {
                        merged[i] = cell_addr;
                    }
                }
            }
            state.packed_cell_table = merged;
            state.packed_cell_size = pack;
        }
    }

    /// Install the compile-time-recognised "windowed byte array" descriptor
    /// on `State` so reads inside the window collapse to one state-var read +
    /// one flat-array index (instead of walking the inline-exception CmpEq
    /// chain in `--readMem` and re-evaluating the inner `--readDiskByte`
    /// dispatch on every byte).
    ///
    /// Mirrors the contract of `wire_state_for_packed_memory`: must be
    /// called after `state.load_properties` and before any ticks. Safe to
    /// call when the cabinet has no windowed-byte-array shape — descriptor
    /// is `None` and `State::read_mem` skips the short-circuit.
    pub fn wire_state_for_windowed_byte_array(&self, state: &mut crate::State) {
        let Some(ref cw) = self.compiled.windowed_byte_array else { return };
        // The compiler tracks the cell by NAME (e.g. "--__1mc632"), since
        // state-var slot indices aren't assigned until State::load_properties.
        // Resolve to the state var here. CSS-DOS double-buffers cells through
        // intermediate names `--__1mcN` / `--__2mcN` whose default sources the
        // canonical `--mcN` property; the canonical name is what
        // `load_properties` registers as a state var (with the `--` stripped).
        let bare = cw.key_cell_property.trim_start_matches("--");
        let canonical = bare
            .strip_prefix("__1")
            .or_else(|| bare.strip_prefix("__2"))
            .unwrap_or(bare);
        let Some(slot) = state.var_slot(canonical) else {
            log::warn!(
                "[windowed-byte-array] cabinet has windowed-byte-array dispatch but key cell `{}` (canonical `{}`) is not a state var; falling back to per-byte CSS",
                cw.key_cell_property, canonical
            );
            return;
        };
        state.windowed_byte_array = Some(crate::state::WindowedByteArray {
            window_base: cw.window_base,
            window_end: cw.window_end,
            key_cell_slot: slot,
            stride: cw.stride,
            byte_array_base_key: cw.byte_array_base_key,
            byte_array: cw.byte_array.clone(),
        });
        state.virtual_regions.push(crate::state::VirtualRegion {
            start: cw.window_base,
            end: cw.window_end,
            source: "windowed_byte_array",
        });
        log::info!(
            "[windowed-byte-array] installed: window=[0x{:X},0x{:X}) stride={} key_cell={} byte_array_len={}",
            cw.window_base, cw.window_end, cw.stride, cw.key_cell_property,
            cw.byte_array.len()
        );
    }

    /// Build an evaluator from a `ParsedProgram`.
    pub fn from_parsed(program: &ParsedProgram) -> Self {
        let _t = Instant::now();
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

        // Merge fast-path function-dispatch runs — dense literal entries the
        // parser absorbed without ever building Expr trees — into the
        // recognised tables. Three cases:
        //  - a table was recognised on the same key → extend its entries;
        //  - no table (the run absorbed everything, or the leftover branches
        //    were below recognise_dispatch's threshold) but the leftovers
        //    are compatible (same key, literal tests) → build the table here;
        //  - anything else → re-inject the entries as plain branches so
        //    behaviour is identical to the no-fast-path route (deferred
        //    until the functions map exists below).
        // Key collisions between a run and parsed branches keep last-insert-
        // wins semantics, matching recognise_dispatch (emitters don't
        // duplicate keys).
        let mut merged_entries = 0usize;
        let mut deferred_reinjects: Vec<&crate::types::FnDispatchRun> = Vec::new();
        for run in &program.prebuilt_fn_dispatch_runs {
            if let Some(table) = dispatch_tables.get_mut(&run.function) {
                if table.key_property == run.key_property {
                    table.entries.reserve(run.entries.len());
                    for &(k, v) in &run.entries {
                        table.entries.insert(k, Expr::Literal(v as f64));
                    }
                    merged_entries += run.entries.len();
                } else {
                    // Mixed keys: the original (unabsorbed) function would not
                    // have been recognised as a table at all. Drop the table
                    // and fall back to branch re-injection.
                    log::warn!(
                        "fn-dispatch run in {} keyed on {} conflicts with recognised table key {}; re-injecting as branches",
                        run.function, run.key_property, table.key_property,
                    );
                    dispatch_tables.remove(&run.function);
                    deferred_reinjects.push(run);
                }
            } else if let Some(table) =
                build_table_from_run(&program.functions, run)
            {
                merged_entries += run.entries.len();
                dispatch_tables.insert(run.function.clone(), table);
            } else {
                log::warn!(
                    "fn-dispatch run in {} ({} entries) not table-compatible; re-injecting as branches",
                    run.function, run.entries.len(),
                );
                deferred_reinjects.push(run);
            }
        }
        if merged_entries > 0 {
            log::info!(
                "[fast-path] merged {} fn-dispatch runs ({} literal entries) into dispatch tables",
                program.prebuilt_fn_dispatch_runs.len(),
                merged_entries,
            );
        }
        crate::compile_stats::record("eval.dispatch_recognition", _t.elapsed().as_secs_f64());

        // Build the functions map. For table-recognised functions the
        // branches are dead weight — every runtime path (interpreter
        // eval_function_call, compile_function_call) consults the dispatch
        // table before the function body — so clone those defs with an
        // empty branch list instead of deep-copying multi-hundred-K-branch
        // trees (~4 s in wasm on the doom cabinet).
        let _tc = Instant::now();
        let mut functions: HashMap<String, FunctionDef> = program
            .functions
            .iter()
            .map(|f| {
                if dispatch_tables.contains_key(&f.name) {
                    if let Expr::StyleCondition { fallback, .. } = &f.result {
                        return (
                            f.name.clone(),
                            FunctionDef {
                                name: f.name.clone(),
                                parameters: f.parameters.clone(),
                                locals: f.locals.clone(),
                                result: Expr::StyleCondition {
                                    branches: Vec::new(),
                                    fallback: fallback.clone(),
                                },
                            },
                        );
                    }
                }
                (f.name.clone(), f.clone())
            })
            .collect();
        crate::compile_stats::record("eval.functions_clone", _tc.elapsed().as_secs_f64());

        // Correctness fallback for non-table-shaped functions whose runs
        // were absorbed: put the entries back as plain branches.
        for run in deferred_reinjects {
            reinject_fn_dispatch_run(&mut functions, run);
        }

        // Merge memory address mappings from identity-read dispatch tables into
        // the address map. State var addresses (from load_properties) take priority.
        let address_map = build_address_map(&dispatch_tables);
        if !address_map.is_empty() {
            log::info!(
                "Merging {} property→address mappings from dispatch tables",
                address_map.len()
            );
            merge_address_map(address_map);
        }

        log::info!("[compile phase] dispatch tables: {:.2}s", _t.elapsed().as_secs_f64());
        crate::compile_stats::record("eval.dispatch_tables", _t.elapsed().as_secs_f64());
        let _t = Instant::now();
        // Recognise broadcast write patterns in assignments. Then merge in
        // any broadcast writes the parser fast-path already produced: those
        // never reached `program.assignments` (their Expr trees were never
        // built), so the recogniser can't see them.
        let mut broadcast_result =
            broadcast_write::recognise_broadcast(&program.assignments);
        for pre in &program.prebuilt_broadcast_writes {
            broadcast_result.writes.push(pre.clone());
        }
        for name in &program.fast_path_absorbed {
            broadcast_result.absorbed_properties.insert(name.clone());
        }
        for bw in &broadcast_result.writes {
            let gate = bw.gate_property.as_deref().unwrap_or("(ungated)");
            log::info!(
                "Recognised broadcast write: {} → {} targets  gate={}",
                bw.dest_property,
                bw.address_map.len(),
                gate,
            );
        }
        // Packed broadcast writes: CSS-DOS PACK_SIZE=2 cell writes via the
        // nested --applySlot chain. Replaces ~190K ops/tick with ~6 port
        // checks. The absorbed property set is merged into broadcast_result
        // so the assignment loop drops them.
        let mut packed_bw_result =
            crate::pattern::packed_broadcast_write::recognise_packed_broadcast(
                &program.assignments,
            );
        for name in &packed_bw_result.absorbed_properties {
            broadcast_result.absorbed_properties.insert(name.clone());
        }
        // Merge prebuilt packed ports from the parser fast-path. For any
        // (gate, addr, val, width) tuple already present in the
        // AST-recognised result, union the address_map; otherwise push a
        // new port.
        if !program.prebuilt_packed_broadcast_ports.is_empty() {
            use std::collections::HashMap;
            let mut idx_by_key: HashMap<(String, String, String, String), usize> =
                HashMap::new();
            for (i, port) in packed_bw_result.ports.iter().enumerate() {
                idx_by_key.insert(
                    (
                        port.gate_property.clone(),
                        port.addr_property.clone(),
                        port.val_property.clone(),
                        port.width_property.clone(),
                    ),
                    i,
                );
            }
            for pre in &program.prebuilt_packed_broadcast_ports {
                let key = (
                    pre.gate_property.clone(),
                    pre.addr_property.clone(),
                    pre.val_property.clone(),
                    pre.width_property.clone(),
                );
                if let Some(&i) = idx_by_key.get(&key) {
                    for (&addr, name) in &pre.address_map {
                        packed_bw_result.ports[i]
                            .address_map
                            .insert(addr, name.clone());
                    }
                } else {
                    idx_by_key.insert(key, packed_bw_result.ports.len());
                    packed_bw_result.ports.push(pre.clone());
                }
            }
        }
        for port in &packed_bw_result.ports {
            log::info!(
                "Recognised packed broadcast port: gate={} addr={} val={} width={} → {} cells (pack={})",
                port.gate_property, port.addr_property, port.val_property,
                port.width_property, port.address_map.len(), packed_bw_result.pack,
            );
        }

        log::info!("[compile phase] broadcast recognition: {:.2}s", _t.elapsed().as_secs_f64());
        crate::compile_stats::record("eval.broadcast_recognition", _t.elapsed().as_secs_f64());
        let _t = Instant::now();
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
        crate::compile_stats::record("eval.filter_partition", _t.elapsed().as_secs_f64());
        let _t = Instant::now();
        // Topological sort: reorder assignments by data dependencies.
        // CSS evaluates all properties simultaneously, but our sequential evaluator
        // must process them in dependency order. Only style() test references
        // (non-buffer-prefixed) create ordering constraints.
        let assignments = topological_sort_assignments(numeric_assignments, &functions);

        log::info!("[compile phase] topological sort: {:.2}s", _t.elapsed().as_secs_f64());
        crate::compile_stats::record("eval.topological_sort", _t.elapsed().as_secs_f64());
        let _t = Instant::now();
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
        let _t = Instant::now();
        let compiled = compile::compile(
            &assignments,
            &broadcast_result.writes,
            &packed_bw_result.ports,
            &functions,
            &mut dispatch_tables,
        );

        log::info!("[compile phase] compile::compile: {:.2}s", _t.elapsed().as_secs_f64());
        crate::compile_stats::record("eval.compile_total", _t.elapsed().as_secs_f64());
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

        // Compile input edges. Only literal values are supported in v1;
        // non-literal RHS expressions are skipped with a warning.
        let mut input_edge_bindings = Vec::with_capacity(program.input_edges.len());
        for edge in &program.input_edges {
            let value = match &edge.value {
                Expr::Literal(v) => *v as i32,
                _ => {
                    log::warn!(
                        "input edge {}:{} on {} has non-literal value; skipping",
                        edge.pseudo,
                        edge.selector,
                        edge.property,
                    );
                    continue;
                }
            };
            input_edge_bindings.push(InputEdgeBinding {
                property: edge.property.clone(),
                pseudo: edge.pseudo.clone(),
                selector: edge.selector.clone(),
                value,
            });
        }
        if !input_edge_bindings.is_empty() {
            log::info!(
                "Compiled {} input edges (pseudo-class gated assignments)",
                input_edge_bindings.len(),
            );
        }

        // Recognise opcode-keyed self-loops in the cabinet's dispatch
        // family. Phase 1 of the rep_fast_forward genericity mission:
        // descriptors are produced but not yet consumed at runtime.
        let loop_descriptors =
            crate::pattern::loop_descriptor::recognise_loops(&program.assignments);
        log::info!(
            "[loop-descriptor] recognised {} self-loop opcodes",
            loop_descriptors.len(),
        );
        let mut compiled = compiled;
        compiled.loop_descriptors = loop_descriptors.clone();

        Evaluator {
            functions,
            assignments,
            dispatch_tables,
            broadcast_writes: broadcast_result.writes,
            packed_broadcast_ports: packed_bw_result.ports,
            function_patterns,
            pre_tick_hooks: Vec::new(),
            properties: HashMap::with_capacity(properties_capacity),
            call_depth: 0,
            string_assignments,
            string_property_names,
            compiled,
            slots: Vec::with_capacity(slot_count),
            input_edge_bindings,
            loop_descriptors,
        }
    }

    /// Return the input edges the recogniser bound, as
    /// `(property, pseudo, selector, value)` tuples. Hosts use this to
    /// know which edges they can drive via `set_pseudo_class_active`.
    /// The strings are owned to make the wasm boundary easy.
    pub fn input_edges_for_host(&self) -> Vec<(String, String, String, i32)> {
        self.input_edge_bindings
            .iter()
            .map(|b| {
                (
                    b.property.clone(),
                    b.pseudo.clone(),
                    b.selector.clone(),
                    b.value,
                )
            })
            .collect()
    }

    /// Resolve the parsed input-edge bindings against the live
    /// `state.state_var_index` and install the grouped form on
    /// `state.input_edge_groups`. After this, the host's calls to
    /// `state.set_pseudo_class_active(...)` directly write the
    /// affected gated slots — there is no per-tick apply path.
    ///
    /// Call once at engine construction, after
    /// `state.load_properties`. Idempotent: re-running rebuilds the
    /// groups (useful after `engine.reset()`).
    ///
    /// Bindings whose `bare(property)` doesn't resolve to a slot are
    /// silently dropped — they correspond to recogniser hits on
    /// properties the host's filter (e.g. `program.json`) excluded,
    /// which is rare but possible.
    pub fn wire_state_for_input_edges(&self, state: &mut crate::State) {
        use std::collections::HashMap;
        if self.input_edge_bindings.is_empty() {
            state.input_edge_groups = Vec::new();
            return;
        }
        let mut by_slot: HashMap<usize, Vec<crate::state::InputEdgeGroupEntry>> =
            HashMap::new();
        for b in &self.input_edge_bindings {
            let bare = to_bare_name(&b.property);
            let slot = match state.var_slot(bare) {
                Some(s) => s,
                None => continue,
            };
            by_slot
                .entry(slot)
                .or_default()
                .push(crate::state::InputEdgeGroupEntry {
                    pseudo: b.pseudo.clone(),
                    selector: b.selector.clone(),
                    value: b.value,
                });
        }
        let mut groups: Vec<crate::state::InputEdgeGroup> = by_slot
            .into_iter()
            .map(|(slot, edges)| crate::state::InputEdgeGroup { slot, edges })
            .collect();
        // Stable order by slot for predictable diagnostics.
        groups.sort_by_key(|g| g.slot);
        state.input_edge_groups = groups;
    }

    /// Register a pre-tick hook that is called before each tick.
    ///
    /// Hooks inspect and modify state to implement external side effects
    /// (e.g., text output, keyboard input). The evaluator itself has no
    /// knowledge of what the hook does.
    pub fn add_pre_tick_hook(&mut self, hook: Box<dyn Fn(&mut State) + Send + Sync>) {
        self.pre_tick_hooks.push(hook);
    }

    /// Run a single tick: evaluate all assignments against the state.
    pub fn tick(&mut self, state: &mut State) -> TickResult {
        // Run pre-tick hooks (e.g., external function dispatch).
        for hook in &self.pre_tick_hooks {
            hook(state);
        }

        // Input-edge values were written directly into gated state-var
        // slots at `state.set_pseudo_class_active(...)` time, so the
        // per-tick path no longer needs an apply step. Removed
        // 2026-05-28 — see `wire_state_for_input_edges` and
        // `State::set_pseudo_class_active`.

        // Snapshot state vars for change detection
        let prev_vars = state.state_vars.clone();

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

        // Detect state var changes
        let mut changes = Vec::new();
        for (i, name) in state.state_var_names.iter().enumerate() {
            if i < prev_vars.len() && state.state_vars[i] != prev_vars[i] {
                changes.push((format!("--{}", name), state.state_vars[i].to_string()));
            }
        }

        state.frame_counter += 1;

        TickResult {
            changes,
            ticks_executed: 1,
        }
    }

    /// Run one tick AND produce a [`StateDelta`] capturing every mutated
    /// channel, with both pre- and post-tick values so the delta can be
    /// applied or reverted.
    ///
    /// More expensive than `tick()` because it pre-clones every channel that
    /// might change (state_vars, memory bytes, extended map, string_properties).
    /// For hot loops where you just need the state_var change summary, keep
    /// using `tick()`.
    pub fn tick_with_delta(&mut self, state: &mut State) -> (TickResult, crate::state::StateDelta) {
        use crate::state::StateDelta;

        let frame_before = state.frame_counter;
        // Full pre-images for diff. These are "expensive" only in the sense
        // of being O(state size) once per tick — still far cheaper than the
        // State::clone() the HTTP debugger used to do every 1000 ticks.
        let prev_state_vars = state.state_vars.clone();
        let prev_memory = state.memory.clone();
        let prev_extended = state.extended.clone();
        let prev_strings = state.string_properties.clone();

        let result = self.tick(state);

        let mut delta = StateDelta {
            frame_counter_before: frame_before,
            frame_counter_after: state.frame_counter,
            ..Default::default()
        };

        for (i, (old, new)) in prev_state_vars.iter().zip(state.state_vars.iter()).enumerate() {
            if old != new {
                delta.state_vars.push((i as u32, *old, *new));
            }
        }
        // Memory may have grown (it's a Vec<u8>); treat missing old as 0.
        let common_len = prev_memory.len().min(state.memory.len());
        for i in 0..common_len {
            let old = prev_memory[i];
            let new = state.memory[i];
            if old != new {
                delta.memory.push((i as u32, old, new));
            }
        }
        for i in common_len..state.memory.len() {
            let new = state.memory[i];
            if new != 0 {
                delta.memory.push((i as u32, 0, new));
            }
        }
        // Extended map: check all keys in either side.
        let mut ext_keys: std::collections::HashSet<i32> = prev_extended.keys().copied().collect();
        ext_keys.extend(state.extended.keys().copied());
        for addr in ext_keys {
            let old = prev_extended.get(&addr).copied();
            let new = state.extended.get(&addr).copied();
            if old != new {
                delta.extended.push((addr, old, new));
            }
        }
        // String properties: check all keys in either side.
        let mut str_keys: std::collections::HashSet<String> =
            prev_strings.keys().cloned().collect();
        str_keys.extend(state.string_properties.keys().cloned());
        for name in str_keys {
            let old = prev_strings.get(&name).cloned();
            let new = state.string_properties.get(&name).cloned();
            if old != new {
                delta.string_properties.push((name, old, new));
            }
        }

        (result, delta)
    }

    /// Execute one tick without change detection — for use from run_batch where
    /// the outer routine does a single diff against the snapshot at the end.
    /// Skips the per-tick state_vars.clone() and the post-tick diff loop.
    #[inline]
    fn tick_no_diff(&mut self, state: &mut State) {
        for hook in &self.pre_tick_hooks {
            hook(state);
        }
        // Input-edge values are written at set_pseudo_class_active time;
        // no per-tick apply needed.
        compile::execute(&self.compiled, state, &mut self.slots);
        if !self.string_assignments.is_empty() {
            self.properties.clear();
            self.call_depth = 0;
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
                if let Value::Str(ref s) = value {
                    state.string_properties.insert(bare.to_string(), s.clone());
                }
                self.properties.insert(assignment.property.clone(), value);
            }
        }
        state.frame_counter += 1;
    }

    /// Run a single tick with phase timing breakdown.
    ///
    /// Returns the normal TickResult plus timing for each phase:
    /// (hooks, snapshot, execute, string_eval, change_detect) in seconds.
    pub fn tick_profiled(&mut self, state: &mut State) -> (TickResult, TickProfile) {
        let mut profile = TickProfile::default();

        let t0 = Instant::now();

        // Pre-tick hooks
        for hook in &self.pre_tick_hooks {
            hook(state);
        }
        let t1 = Instant::now();
        profile.hooks += t1.duration_since(t0).as_secs_f64();

        // Input-edge slots are written at set_pseudo_class_active time;
        // no per-tick apply needed.

        // Snapshot
        let prev_vars = state.state_vars.clone();
        let t2 = Instant::now();
        profile.snapshot += t2.duration_since(t1).as_secs_f64();

        // Execute compiled bytecode — granular profiling inside
        compile::execute_profiled(&self.compiled, state, &mut self.slots, &mut profile);

        // String assignments
        let t3 = Instant::now();
        if !self.string_assignments.is_empty() {
            self.properties.clear();
            self.call_depth = 0;
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
                if let Value::Str(ref s) = value {
                    state.string_properties.insert(bare.to_string(), s.clone());
                }
                self.properties.insert(assignment.property.clone(), value);
            }
        }
        let t4 = Instant::now();
        profile.string_eval += t4.duration_since(t3).as_secs_f64();

        // Change detection
        let mut changes = Vec::new();
        for (i, name) in state.state_var_names.iter().enumerate() {
            if i < prev_vars.len() && state.state_vars[i] != prev_vars[i] {
                changes.push((format!("--{}", name), state.state_vars[i].to_string()));
            }
        }
        let t5 = Instant::now();
        profile.change_detect += t5.duration_since(t4).as_secs_f64();

        state.frame_counter += 1;

        (
            TickResult {
                changes,
                ticks_executed: 1,
            },
            profile,
        )
    }

    /// Run a single tick using the interpreted path (for comparison testing).
    pub fn tick_interpreted(&mut self, state: &mut State) -> TickResult {
        // Run pre-tick hooks
        for hook in &self.pre_tick_hooks {
            hook(state);
        }

        // Input-edge slots are written at set_pseudo_class_active time;
        // no per-tick apply needed.

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
            // Gated broadcasts skip entirely when the gate is not 1. See
            // the compiled path (compile::execute) for the same check.
            if let Some(ref gate_property) = bw.gate_property {
                let gate = self.resolve_property(gate_property, state).as_number();
                if gate as i64 != 1 {
                    continue;
                }
            }
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

        // Packed broadcast writes — the `--mcN` cell assignments were
        // absorbed out of the assignment loop (they would just be a cascade
        // of --applySlot, which semantically IS the port logic). Running
        // the ports here produces the same state-var cell updates that
        // the compiled path's packed_broadcast_writes runtime produces.
        // Without this, packed cabinets executed via the interpreter never
        // actually apply any memory writes — every `--mcN` stays at its
        // initial value. Iterate in REVERSE so slot 0 fires LAST, matching
        // the CSS semantics where slot 0 is the outermost --applySlot and
        // wins on same-cell collisions.
        let packed_ports_ptr = self.packed_broadcast_ports.as_ptr();
        let packed_ports_len = self.packed_broadcast_ports.len();
        for i in (0..packed_ports_len).rev() {
            let port = unsafe { &*packed_ports_ptr.add(i) };
            let gate = self.resolve_property(&port.gate_property, state).as_number();
            if (gate as i64) != 1 { continue; }
            let byte_addr = self.resolve_property(&port.addr_property, state).as_number() as i32;
            if byte_addr < 0 { continue; }
            let val = self.resolve_property(&port.val_property, state).as_number() as i32;
            let width = self.resolve_property(&port.width_property, state).as_number() as i32;
            // write_mem is packed-aware: it splices the single byte into
            // the state-var cell when a cell is declared, and keeps the
            // flat shadow in sync. Do not filter by address_map — the
            // compiled path doesn't; write_mem handles missing cells
            // correctly on its own (updates the shadow only).
            if width == 2 {
                state.write_mem(byte_addr, val & 0xFF);
                state.write_mem(byte_addr + 1, (val >> 8) & 0xFF);
            } else {
                state.write_mem(byte_addr, val & 0xFF);
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

    /// Snapshot all named slot values after a compiled tick. Used by the debugger to
    /// compare computed intermediate properties between compiled and interpreted paths.
    pub fn get_all_slot_values(&self) -> HashMap<String, i32> {
        self.compiled
            .property_slots
            .iter()
            .filter_map(|(name, &slot)| {
                self.slots.get(slot as usize).map(|&v| (name.clone(), v))
            })
            .collect()
    }

    /// Dump the compiled ops and dispatch tables to stderr for debugging.
    pub fn dump_compiled(&self) {
        eprintln!("=== COMPILED PROGRAM ({} ops, {} slots, {} dispatch tables, {} writeback) ===",
            self.compiled.ops.len(),
            self.compiled.slot_count,
            self.compiled.dispatch_tables.len(),
            self.compiled.writeback.len(),
        );
        eprintln!("--- main ops ---");
        for (i, op) in self.compiled.ops.iter().enumerate() {
            eprintln!("  [{:4}] {:?}", i, op);
        }
        eprintln!("--- dispatch tables ---");
        for (tid, dt) in self.compiled.dispatch_tables.iter().enumerate() {
            eprintln!("  table[{}]: {} entries, fallback_slot={}", tid, dt.entries.len(), dt.fallback_slot);
            let mut keys: Vec<i64> = dt.entries.keys().copied().collect();
            keys.sort();
            for k in &keys {
                let (ops, result) = &dt.entries[k];
                eprintln!("    key={} → slot={} ({} ops)", k, result, ops.len());
                for (j, op) in ops.iter().enumerate() {
                    eprintln!("      [{:3}] {:?}", j, op);
                }
            }
            eprintln!("    fallback ({} ops):", dt.fallback_ops.len());
            for (j, op) in dt.fallback_ops.iter().enumerate() {
                eprintln!("      [{:3}] {:?}", j, op);
            }
        }
        eprintln!("--- writeback ---");
        for (slot, addr) in &self.compiled.writeback {
            let name = self.compiled.property_slots.iter()
                .find(|(_, &s)| s == *slot)
                .map(|(n, _)| n.as_str())
                .unwrap_or("?");
            eprintln!("  slot={} ({}) → addr={}", slot, name, addr);
        }
        eprintln!("--- property_slots ---");
        let mut ps: Vec<(&String, &u32)> = self.compiled.property_slots.iter().collect();
        ps.sort_by_key(|(_, &s)| s);
        for (name, slot) in ps {
            eprintln!("  {} → slot {}", name, slot);
        }
    }

    /// Return a map of slot index → property name for all compiled properties.
    pub fn get_slot_map(&self) -> Vec<(u32, String)> {
        let mut result: Vec<(u32, String)> = self.compiled.property_slots
            .iter()
            .map(|(name, &slot)| (slot, name.clone()))
            .collect();
        result.sort_by_key(|(slot, _)| *slot);
        result
    }

    /// Return the current value of a given slot index after the last tick.
    pub fn get_slot_by_index(&self, slot: u32) -> Option<i32> {
        self.slots.get(slot as usize).copied()
    }

    /// Return the compiled ops for a single named property, plus any dispatch tables they reference.
    /// Returns None if the property has no compiled slot.
    pub fn get_ops_for_property(&self, name: &str) -> Option<Vec<String>> {
        let slot = *self.compiled.property_slots.get(name)?;
        // Find the range of ops that write to this slot
        let mut lines = Vec::new();
        lines.push(format!("property {} → slot {}", name, slot));

        // Dump all main ops (can be large — caller should filter or limit)
        lines.push(format!("main_ops: {} total", self.compiled.ops.len()));
        for (i, op) in self.compiled.ops.iter().enumerate() {
            lines.push(format!("  [{:4}] {:?}", i, op));
        }

        lines.push(format!("dispatch_tables: {}", self.compiled.dispatch_tables.len()));
        for (tid, dt) in self.compiled.dispatch_tables.iter().enumerate() {
            lines.push(format!("  table[{}]: {} entries, fallback_slot={}", tid, dt.entries.len(), dt.fallback_slot));
            let mut keys: Vec<i64> = dt.entries.keys().copied().collect();
            keys.sort();
            for k in &keys {
                let (entry_ops, result) = &dt.entries[k];
                lines.push(format!("    [{}] key={} → slot={} ({} ops)", tid, k, result, entry_ops.len()));
                for (j, op) in entry_ops.iter().enumerate() {
                    lines.push(format!("      [{:3}] {:?}", j, op));
                }
            }
            lines.push(format!("    [{}] fallback ({} ops):", tid, dt.fallback_ops.len()));
            for (j, op) in dt.fallback_ops.iter().enumerate() {
                lines.push(format!("      [{:3}] {:?}", j, op));
            }
        }
        Some(lines)
    }

    /// Dump compiled ops in a given range as debug strings.
    pub fn dump_ops_range(&self, start: usize, end: usize) -> Vec<String> {
        let mut lines = Vec::new();
        let end = end.min(self.compiled.ops.len());
        for i in start..end {
            lines.push(format!("[{:5}] {:?}", i, self.compiled.ops[i]));
        }
        lines
    }

    /// Snapshot all named property values from the most recent interpreted tick.
    /// Returns values from `self.properties` (the interpreted path's HashMap).
    pub fn get_all_interpreted_values(&self) -> HashMap<String, i32> {
        self.properties
            .iter()
            .map(|(name, v)| (name.clone(), v.as_number() as i32))
            .collect()
    }

    /// Trace the compiled execution of a specific property at the current state.
    /// Returns (compiled_value, trace_entries) or None if property not found.
    /// Does NOT modify the state (clones internally).
    pub fn trace_property(&mut self, state: &State, property_name: &str) -> Option<(i32, Vec<compile::TraceEntry>)> {
        let mut state_clone = state.clone();
        compile::trace_property(&self.compiled, &mut state_clone, &mut self.slots, property_name)
    }

    /// Run a batch of ticks, returning the net state diff across all ticks.
    ///
    /// Takes a snapshot before the batch and diffs at the end, so callers
    /// see every register/memory change — not just the final tick's delta.
    pub fn run_batch(&mut self, state: &mut State, count: u32) -> TickResult {
        // Snapshot only state_vars for diff (full state.clone() is expensive
        // and unused — we only diff state_vars here).
        let snapshot_vars = state.state_vars.clone();
        for _ in 0..count {
            self.tick_no_diff(state);
        }

        // Diff state vars
        let mut changes = Vec::new();
        for (i, name) in state.state_var_names.iter().enumerate() {
            if i < state.state_vars.len() && i < snapshot_vars.len()
                && state.state_vars[i] != snapshot_vars[i]
            {
                changes.push((format!("--{}", name), state.state_vars[i].to_string()));
            }
        }

        TickResult {
            changes,
            ticks_executed: count,
        }
    }

    /// Run a batch of ticks without computing the state-var diff.
    ///
    /// Use this when the caller doesn't consume the diff (the web bridge
    /// observes state via direct property reads after each batch). On
    /// dense cabinets — doom8088 has ~10K state vars — the snapshot+sweep
    /// is a real per-batch cost we shouldn't pay if no one's reading it.
    pub fn run_batch_silent(&mut self, state: &mut State, count: u32) {
        for _ in 0..count {
            self.tick_no_diff(state);
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
            // Byte-half properties (AL, AH, etc.) are computed CSS expressions
            // with no @property declaration — they won't be in the address map,
            // so property_to_address returns None and they're skipped below.
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

    let _ts = Instant::now();
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
    let _ts = Instant::now();

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
    let _ts = Instant::now();
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
// is_byte_half removed — byte-halves (AL, AH, etc.) are computed CSS expressions
// with no @property declaration. They never appear in the address map, so
// property_to_address() returns None for them and they're automatically
// excluded from writeback.

/// Check if a property is a triple-buffer copy (`--__0*`, `--__1*`, `--__2*`).
///
/// These assignments exist for x86CSS's animation pipeline but are no-ops
/// in calcite's mutable-state model — they just copy the canonical value
/// to a buffer slot that resolves back to the same value via `resolve_property`.
///
/// `pub(crate)` so `pattern::rep_applier`'s slot resolver can mirror the
/// compiler's routing (buffer-copy reads skip the slot table and read the
/// committed state address) instead of inventing its own.
pub(crate) fn is_buffer_copy(name: &str) -> bool {
    name.starts_with("--__0") || name.starts_with("--__1") || name.starts_with("--__2")
}

/// Extract the bare register/memory name from a CSS custom property name.
///
/// Strips the `--` prefix (if present) and any triple-buffer prefix
/// (`__0`, `__1`, `__2`):
/// - `"--AX"` → `"AX"`
/// - `"AX"` → `"AX"`
/// - `"--__0AX"` → `"AX"`
/// - `"__1flags"` → `"flags"`
/// - `"--m42"` → `"m42"`
/// - `"mc42"` → `"mc42"`
///
/// Callers may pass either the CSS form (`--foo`) or the bare form (`foo`);
/// both resolve to the same canonical name. This keeps the function safe when
/// called from code paths that build names programmatically (e.g. the packed
/// cell table builder formats `"mc{}"` without the leading `--`).
///
/// `pub(crate)` so `pattern::rep_applier` / `pattern::loop_descriptor` can
/// resolve descriptor-carried mirror names (`--__1DI` etc.) to their
/// committed state vars. Unlike `property_to_address` this is a pure
/// function — no thread-local — so it works on the debugger's tokio
/// worker threads, where the ADDRESS_MAP thread-local is empty.
pub(crate) fn to_bare_name(name: &str) -> &str {
    let after_dashes = name.strip_prefix("--").unwrap_or(name);
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
/// Uses the CSS-derived address map (populated from `@property` declarations
/// during `State::load_properties`). Falls back to `--m{N}` memory address
/// parsing. No hardcoded register names.
///
/// Automatically strips triple-buffer prefixes (`--__0`, `--__1`, `--__2`).
pub fn property_to_address(name: &str) -> Option<i32> {
    let canonical = to_bare_name(name);

    // Check the CSS-derived address map (state vars + any dispatch table entries).
    let found = ADDRESS_MAP.with(|map| map.borrow().get(canonical).copied());
    if found.is_some() {
        return found;
    }

    // Generic fallback: --m{N} → memory address N
    if let Some(rest) = canonical.strip_prefix('m') {
        return parse_mem_address(rest);
    }

    None
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
/// Called during `State::load_properties()` for state variables and during
/// `Evaluator::from_parsed()` for dispatch-table-derived memory mappings.
pub fn set_address_map(map: HashMap<String, i32>) {
    ADDRESS_MAP.with(|m| {
        *m.borrow_mut() = map;
    });
}

/// Merge additional entries into the existing address map.
///
/// Existing entries are NOT overwritten — state var addresses (installed by
/// `load_properties`) take priority over dispatch-table-derived addresses.
pub fn merge_address_map(map: HashMap<String, i32>) {
    ADDRESS_MAP.with(|m| {
        let mut existing = m.borrow_mut();
        for (k, v) in map {
            existing.entry(k).or_insert(v);
        }
    });
}

/// Build an address map by extracting property→address pairs from
/// identity-read dispatch tables in the CSS.
///
/// An identity-read dispatch table maps key K → `Var("--{name}")` where
/// the property at `--{name}` is at address K. Reversing this gives us
/// the property→address mapping without any hardcoded knowledge.
///
/// Tables do NOT need to be pure identity mappings — individual entries
/// with literal values or non-Var expressions (e.g. BIOS ROM constants,
/// helper function calls for memory-mapped devices) are skipped, but the
/// identity entries in the same table are still recorded. This matters
/// for x86-CSS's `--readMem` function which mixes identity reads of
/// writable memory with literal reads of the BIOS ROM.
pub fn build_address_map(dispatch_tables: &HashMap<String, DispatchTable>) -> HashMap<String, i32> {
    let mut map = HashMap::new();
    for table in dispatch_tables.values() {
        if table.entries.len() < 4 {
            continue;
        }
        // Walk every entry and record those that are identity-form
        // (Var("--name") with parseable bare name). Non-identity entries
        // don't disqualify the rest of the table.
        for (&key, expr) in &table.entries {
            if let Expr::Var { name, .. } = expr {
                let bare = to_bare_name(name).to_string();
                map.insert(bare, key as i32);
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
/// Build a `DispatchTable` for a function whose dense literal entries were
/// absorbed by the parser fast-path, when no table was recognised from the
/// leftover branches alone (typically because the run absorbed everything
/// and fewer than recognise_dispatch's 4-branch minimum remained). The
/// leftover branches must be compatible — single tests on the run's key
/// property with literal test values — and are folded into the table.
fn build_table_from_run(
    functions: &[FunctionDef],
    run: &crate::types::FnDispatchRun,
) -> Option<DispatchTable> {
    let func = functions.iter().find(|f| f.name == run.function)?;
    let Expr::StyleCondition { branches, fallback } = &func.result else {
        return None;
    };
    let mut entries: HashMap<i64, Expr> =
        HashMap::with_capacity(run.entries.len() + branches.len());
    for branch in branches {
        match &branch.condition {
            StyleTest::Single {
                property,
                value: Expr::Literal(v),
            } if *property == run.key_property => {
                entries.insert(*v as i64, branch.then.clone());
            }
            _ => return None,
        }
    }
    for &(k, v) in &run.entries {
        entries.insert(k, Expr::Literal(v as f64));
    }
    Some(DispatchTable {
        key_property: run.key_property.clone(),
        entries,
        fallback: (**fallback).clone(),
    })
}

/// Correctness fallback: put a fast-path-absorbed dispatch run back into its
/// function as ordinary branches (appended; key order is irrelevant because
/// style() tests on distinct literals are mutually exclusive). Used when the
/// function turns out not to be table-shaped, so the slow path sees exactly
/// what it would have parsed.
fn reinject_fn_dispatch_run(
    functions: &mut HashMap<String, FunctionDef>,
    run: &crate::types::FnDispatchRun,
) {
    let Some(func) = functions.get_mut(&run.function) else {
        log::warn!(
            "fn-dispatch run references unknown function {} — {} entries dropped",
            run.function,
            run.entries.len(),
        );
        return;
    };
    if let Expr::StyleCondition { branches, .. } = &mut func.result {
        branches.reserve(run.entries.len());
        for &(k, v) in &run.entries {
            branches.push(StyleBranch {
                condition: StyleTest::Single {
                    property: run.key_property.clone(),
                    value: Expr::Literal(k as f64),
                },
                then: Expr::Literal(v as f64),
            });
        }
    } else {
        log::warn!(
            "fn-dispatch run in {} cannot be re-injected (result is not an if()) — {} entries dropped",
            run.function,
            run.entries.len(),
        );
    }
}

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

        // Dispatch table identity-read (one byte per @property)
        if let Some(table) = dispatch_tables.get(name) {
            if is_dispatch_identity_read(table) {
                patterns.insert(name.clone(), FunctionPattern::IdentityRead);
                continue;
            }
        }
    }

    // Second pass: detect 16-bit read pattern.
    // A function that calls an identity-read function and constructs a word
    // (lo + hi*256) is a 16-bit read. The runtime fast-path uses
    // `state.read_mem16` which reads from `state.memory[]` directly — that's
    // ONLY safe for the unpacked layout where bytes live in `memory`. For the
    // PACK_SIZE=2 layout, bytes live in state_var cells (`mcN`), so we must
    // NOT promote — let those readers evaluate their bodies normally, which
    // dispatches through the underlying PackedByteRead fast path twice.
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
    /// For `--__1`-prefixed names (previous-tick reads), we skip self.properties
    /// entirely — they must come from committed state, never from this tick's
    /// computed values. Reading `--__1uOp` must return the previous tick's uOp,
    /// not the newly computed `--uOp` for this tick.
    fn resolve_property(&self, name: &str, state: &State) -> Value {
        // --__1X names: always read from committed state, never from computed properties.
        if name.starts_with("--__1") {
            let suffix = &name[5..];
            // Check string properties on state (for buffer-prefixed string vars)
            if let Some(s) = state.string_properties.get(suffix) {
                return Value::Str(s.clone());
            }
            if let Some(addr) = property_to_address(name) {
                return Value::Number(state.read_mem(addr) as f64);
            }
            return Value::Number(0.0);
        }
        if let Some(v) = self.properties.get(name) {
            return v.clone();
        }
        if name.starts_with("--__") && name.len() > 5 {
            let suffix = &name[5..];
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

    const STATE_VAR_NAMES: &[&str] = &[
        "AX", "CX", "DX", "BX", "SP", "BP", "SI", "DI",
        "IP", "ES", "CS", "SS", "DS", "flags",
    ];

    /// Install a test address map and return a State with matching state vars.
    fn install_test_address_map() -> State {
        let mut map = HashMap::new();
        let mut state = State::default();
        for (i, &name) in STATE_VAR_NAMES.iter().enumerate() {
            map.insert(name.to_string(), -(i as i32 + 1));
            state.state_vars.push(0);
            state.state_var_names.push(name.to_string());
            state.state_var_index.insert(name.to_string(), i);
        }
        set_address_map(map);
        state
    }

    /// Helper: create a minimal Evaluator for unit tests (no assignments/patterns).
    /// Also returns a State with matching state vars.
    fn test_evaluator(
        functions: HashMap<String, FunctionDef>,
        mut dispatch_tables: HashMap<String, DispatchTable>,
    ) -> (Evaluator, State) {
        let state = install_test_address_map();
        let compiled = crate::compile::compile(&[], &[], &[], &functions, &mut dispatch_tables);
        let function_patterns = detect_function_patterns(&functions, &dispatch_tables);
        let evaluator = Evaluator {
            functions,
            assignments: vec![],
            dispatch_tables,
            broadcast_writes: vec![],
            packed_broadcast_ports: vec![],
            function_patterns,
            pre_tick_hooks: Vec::new(),
            properties: HashMap::with_capacity(16),
            call_depth: 0,
            string_assignments: vec![],
            string_property_names: HashSet::new(),
            compiled,
            slots: Vec::new(),
            input_edge_bindings: Vec::new(),
            loop_descriptors: Vec::new(),
        };
        (evaluator, state)
    }

    #[test]
    fn eval_literal() {
        let (mut eval, state) = test_evaluator(HashMap::new(), HashMap::new());
        assert_eq!(
            eval.eval_expr(&Expr::Literal(42.0), &state).as_number(),
            42.0
        );
    }

    #[test]
    fn eval_calc_operations() {
        let (mut eval, state) = test_evaluator(HashMap::new(), HashMap::new());

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
        let (mut eval, mut state) = test_evaluator(HashMap::new(), HashMap::new());
        state.set_var("AX", 0x1234);

        let expr = Expr::Var {
            name: "--AX".to_string(),
            fallback: None,
        };
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 0x1234 as f64);
    }

    #[test]
    fn eval_var_fallback() {
        let (mut eval, state) = test_evaluator(HashMap::new(), HashMap::new());

        let expr = Expr::Var {
            name: "--nonexistent".to_string(),
            fallback: Some(Box::new(Expr::Literal(99.0))),
        };
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 99.0);
    }

    #[test]
    fn eval_style_condition() {
        let (mut eval, mut state) = test_evaluator(HashMap::new(), HashMap::new());
        state.set_var("AX", 2);

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
        let (mut eval, state) = test_evaluator(HashMap::new(), HashMap::new());

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

        let (mut eval, state) = test_evaluator(functions, HashMap::new());

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

        let (mut eval, state) = test_evaluator(functions, dispatch);

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
        use crate::types::PropertyDef;
        let program = ParsedProgram {
            properties: vec![
                PropertyDef {
                    name: "--AX".to_string(),
                    syntax: crate::types::PropertySyntax::Integer,
                    inherits: true,
                    initial_value: Some(crate::types::CssValue::Integer(0)),
                },
            ],
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
            ..Default::default()
        };

        let mut state = State::default();
        state.load_properties(&program.properties);
        let mut evaluator = Evaluator::from_parsed(&program);

        let result = evaluator.tick(&mut state);

        assert_eq!(state.get_var("AX").unwrap(), 42);
        assert_eq!(state.memory[0], 255);
        assert_eq!(result.ticks_executed, 1);
        assert!(!result.changes.is_empty());
    }

    #[test]
    fn input_edges_drive_state_var() {
        // Verifies apply-on-transition: each set_pseudo_class_active
        // call writes the gated slot directly, no per-tick apply
        // needed. Tests the Phase 4 path landed 2026-05-28.
        use crate::types::{InputEdge, PropertyDef};
        let program = ParsedProgram {
            properties: vec![PropertyDef {
                name: "--keyboard".to_string(),
                syntax: crate::types::PropertySyntax::Integer,
                inherits: true,
                initial_value: Some(crate::types::CssValue::Integer(0)),
            }],
            functions: vec![],
            assignments: vec![],
            input_edges: vec![
                InputEdge {
                    property: "--keyboard".to_string(),
                    pseudo: "active".to_string(),
                    selector: "kb-1".to_string(),
                    value: Expr::Literal(561.0),
                },
                InputEdge {
                    property: "--keyboard".to_string(),
                    pseudo: "active".to_string(),
                    selector: "kb-a".to_string(),
                    value: Expr::Literal(7777.0),
                },
            ],
            ..Default::default()
        };

        let mut state = State::default();
        state.load_properties(&program.properties);
        let evaluator = Evaluator::from_parsed(&program);

        // Bindings compiled from the input edges.
        assert_eq!(evaluator.input_edge_bindings.len(), 2);

        // Wire the resolved groups onto State. After this, every
        // set_pseudo_class_active call writes the slot directly.
        evaluator.wire_state_for_input_edges(&mut state);
        assert_eq!(state.input_edge_groups.len(), 1, "one gated slot (--keyboard)");
        assert_eq!(state.input_edge_groups[0].edges.len(), 2);

        // Nothing pressed → slot stays at 0.
        assert_eq!(state.get_var("keyboard").unwrap(), 0);

        // Press kb-1 → 561 written immediately.
        state.set_pseudo_class_active("active", "kb-1", true);
        assert_eq!(state.get_var("keyboard").unwrap(), 561);

        // Hold kb-1, also press kb-a → sum to 8338.
        state.set_pseudo_class_active("active", "kb-a", true);
        assert_eq!(state.get_var("keyboard").unwrap(), 8338);

        // Release kb-a → back to 561.
        state.set_pseudo_class_active("active", "kb-a", false);
        assert_eq!(state.get_var("keyboard").unwrap(), 561);

        // Release kb-1 → cleared to 0.
        state.set_pseudo_class_active("active", "kb-1", false);
        assert_eq!(state.get_var("keyboard").unwrap(), 0);
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
    fn to_bare_name_accepts_dashed_and_bare_forms() {
        // The dashed CSS form.
        assert_eq!(to_bare_name("--AX"), "AX");
        assert_eq!(to_bare_name("--__1flags"), "flags");
        assert_eq!(to_bare_name("--mc42"), "mc42");
        // The bare form — used by code paths that build names
        // programmatically (e.g. get_or_build_packed_cell_table formats
        // "mc{}" without the `--` prefix). The previous implementation
        // unconditionally did `&name[2..]`, which silently stripped the
        // first two characters of a bare name ("mc42" → "42"), causing
        // the packed cell table to be populated entirely with zeros.
        assert_eq!(to_bare_name("AX"), "AX");
        assert_eq!(to_bare_name("mc42"), "mc42");
        assert_eq!(to_bare_name("__1flags"), "flags");
    }

    #[test]
    fn eval_division_by_zero() {
        let (mut eval, state) = test_evaluator(HashMap::new(), HashMap::new());

        let expr = Expr::Calc(CalcOp::Div(
            Box::new(Expr::Literal(100.0)),
            Box::new(Expr::Literal(0.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 0.0);
    }

    #[test]
    fn eval_mod_by_zero() {
        let (mut eval, state) = test_evaluator(HashMap::new(), HashMap::new());

        let expr = Expr::Calc(CalcOp::Mod(
            Box::new(Expr::Literal(17.0)),
            Box::new(Expr::Literal(0.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 0.0);
    }

    #[test]
    fn eval_negate() {
        let (mut eval, state) = test_evaluator(HashMap::new(), HashMap::new());

        let expr = Expr::Calc(CalcOp::Negate(Box::new(Expr::Literal(42.0))));
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), -42.0);

        let expr = Expr::Calc(CalcOp::Negate(Box::new(Expr::Literal(-7.0))));
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 7.0);
    }

    #[test]
    fn eval_sign_and_abs() {
        let (mut eval, state) = test_evaluator(HashMap::new(), HashMap::new());

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
        let (mut eval, state) = test_evaluator(HashMap::new(), HashMap::new());

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

        let (mut eval, state) = test_evaluator(functions, HashMap::new());

        // Should not panic — returns 0 when depth exceeded
        let expr = Expr::FunctionCall {
            name: "--recurse".to_string(),
            args: vec![],
        };
        assert_eq!(eval.eval_expr(&expr, &state).as_number(), 0.0);
    }
}
