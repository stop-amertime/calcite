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
///
/// Holds both the immutable compiled program (functions, dispatch tables,
/// broadcast writes, assignments) and mutable per-tick evaluation state
/// (properties map, call depth). The properties map is allocated once and
/// reused across ticks via `clear()` to avoid per-tick allocation overhead.
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
    /// Property values computed during the current tick. Reused across ticks.
    properties: HashMap<String, f64>,
    /// Call depth for recursion protection.
    call_depth: usize,
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
        let broadcast_result = broadcast_write::recognise_broadcast(&program.assignments);
        for bw in &broadcast_result.writes {
            log::info!(
                "Recognised broadcast write: {} → {} targets",
                bw.dest_property,
                bw.address_map.len(),
            );
        }

        // Filter out:
        // 1. Assignments absorbed into broadcast writes (would overwrite with stale values)
        // 2. Triple-buffer copies (--__0*, --__1*, --__2*) which are no-ops in mutable state
        let assignments: Vec<Assignment> = program
            .assignments
            .iter()
            .filter(|a| {
                !broadcast_result.absorbed_properties.contains(&a.property)
                    && !is_buffer_copy(&a.property)
            })
            .cloned()
            .collect();

        // Reorder: move --modRm* assignments before --instId.
        // In CSS, all properties are computed simultaneously, but our sequential
        // evaluator processes them in declaration order. The --getInstId() function
        // references --modRm_reg via compound conditions (e.g., opcode 0xFF group),
        // so --modRm* must be computed before --instId.
        let assignments = reorder_modrm_before_instid(assignments);

        let buffer_copies = program
            .assignments
            .iter()
            .filter(|a| is_buffer_copy(&a.property))
            .count();

        log::info!(
            "Assignments: {} kept, {} absorbed into broadcast writes, {} buffer copies skipped",
            assignments.len(),
            broadcast_result.absorbed_properties.len(),
            buffer_copies,
        );
        if log::log_enabled!(log::Level::Debug) {
            for a in &assignments {
                log::debug!("  kept: {}", a.property);
            }
        }

        let properties_capacity = assignments.len();
        Evaluator {
            functions,
            assignments,
            dispatch_tables,
            broadcast_writes: broadcast_result.writes,
            properties: HashMap::with_capacity(properties_capacity),
            call_depth: 0,
        }
    }

    /// Run a single tick: evaluate all assignments against the state.
    pub fn tick(&mut self, state: &mut State) -> TickResult {
        // Handle external function calls BEFORE evaluation.
        // When IP is at an external function address, the CSS has RET (0xC3) there.
        // We capture the side effects (text output) before the tick processes the RET.
        let prev_ip = state.registers[crate::state::reg::IP];
        match prev_ip {
            0x2000 => {
                // writeChar1: output 1 char from stack argument at SP+2
                let ch = state.read_mem(state.registers[crate::state::reg::SP] + 2);
                if ch > 0 && ch < 128 {
                    state.text_buffer.push(ch as u8 as char);
                }
            }
            0x2002 => {
                // writeChar4: output 4 chars from string pointer at SP+2
                let ptr = state.read_mem16(state.registers[crate::state::reg::SP] + 2);
                for off in 0..4 {
                    let ch = state.read_mem(ptr + off);
                    if ch == 0 { break; }
                    if ch > 0 && ch < 128 {
                        state.text_buffer.push(ch as u8 as char);
                    }
                }
            }
            0x2004 => {
                // writeChar8: output 8 chars from string pointer at SP+2
                let ptr = state.read_mem16(state.registers[crate::state::reg::SP] + 2);
                for off in 0..8 {
                    let ch = state.read_mem(ptr + off);
                    if ch == 0 { break; }
                    if ch > 0 && ch < 128 {
                        state.text_buffer.push(ch as u8 as char);
                    }
                }
            }
            _ => {}
        }

        // Reuse the properties map — clear() retains the allocated capacity.
        self.properties.clear();
        self.call_depth = 0;

        // Execute all assignments in declaration order.
        // We use raw pointers to read from self.assignments while mutating self.properties.
        let assignments_ptr = self.assignments.as_ptr();
        let assignments_len = self.assignments.len();
        for i in 0..assignments_len {
            // SAFETY: assignments is not modified during tick; pointer is valid.
            let assignment = unsafe { &*assignments_ptr.add(i) };
            let value = self.eval_expr(&assignment.value, state);
            self.properties.insert(assignment.property.clone(), value);
        }

        // Apply broadcast writes: O(1) HashMap lookup per write.
        // Writes go directly to state — no need to update properties since
        // absorbed memory cells are never read by the remaining assignments
        // (memory reads go through --readMem dispatch table which reads from state).
        let bw_ptr = self.broadcast_writes.as_ptr();
        let bw_len = self.broadcast_writes.len();
        for i in 0..bw_len {
            // SAFETY: broadcast_writes is not modified during tick; pointer is valid.
            let bw = unsafe { &*bw_ptr.add(i) };
            let dest = self.resolve_property(&bw.dest_property, state);
            let dest_i64 = dest as i64;
            if bw.address_map.contains_key(&dest_i64) {
                let value = self.eval_expr(&bw.value_expr, state);
                state.write_mem(dest_i64 as i32, value as i32);
            }
            // Handle word-write spillover: if the guard (e.g., --isWordWrite) is set,
            // also write the high byte to the next address (dest + 1).
            if !bw.spillover_map.is_empty() {
                if let Some(ref guard) = bw.spillover_guard {
                    let guard_val = self.resolve_property(guard, state);
                    if (guard_val as i64) == 1 {
                        if let Some((_var_name, val_expr)) = bw.spillover_map.get(&dest_i64) {
                            let value = self.eval_expr(val_expr, state);
                            state.write_mem(dest_i64 as i32 + 1, value as i32);
                        }
                    }
                }
            }
        }

        // Apply non-broadcast property values back to state
        let changes = self.apply_state(state);

        state.frame_counter += 1;

        TickResult {
            changes,
            ticks_executed: 1,
        }
    }

    /// Read a computed property value from the most recent tick.
    ///
    /// Returns the value of the named property as computed during the last
    /// `tick()` call, or `None` if the property wasn't computed.
    pub fn get_property(&self, name: &str) -> Option<f64> {
        self.properties.get(name).copied()
    }

    /// Run a batch of ticks, returning the net state diff across all ticks.
    ///
    /// Takes a snapshot before the batch and diffs at the end, so callers
    /// see every register/memory change — not just the final tick's delta.
    pub fn run_batch(&mut self, state: &mut State, count: u32) -> TickResult {
        let snapshot = state.clone();
        let mut ticks_done: u32 = 0;
        while ticks_done < count {
            // Check if IP points to a REP prefix — if so, batch-execute natively.
            let skipped = self.try_rep_fast_path(state);
            if skipped > 0 {
                ticks_done += skipped;
                continue;
            }
            self.tick(state);
            ticks_done += 1;
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
            ticks_executed: ticks_done,
        }
    }

    /// Attempt to fast-path a REP-prefixed string instruction.
    ///
    /// If IP points to a REP/REPZ (0xF3) or REPNZ (0xF2) prefix followed by
    /// a string operation (0xA4-0xAF), execute all CX iterations natively
    /// and return the number of ticks consumed (1). Returns 0 if not applicable.
    fn try_rep_fast_path(&self, state: &mut State) -> u32 {
        let ip = state.registers[crate::state::reg::IP];
        if ip < 0 || ip as usize + 1 >= state.memory.len() {
            return 0;
        }
        let prefix = state.memory[ip as usize];
        if prefix != 0xF3 && prefix != 0xF2 {
            return 0;
        }
        let opcode = state.memory[ip as usize + 1];
        if !matches!(opcode, 0xA4..=0xAF) {
            return 0;
        }

        let cx = state.registers[crate::state::reg::CX] & 0xFFFF;
        if cx == 0 {
            state.registers[crate::state::reg::IP] = ip + 2;
            state.frame_counter += 1;
            return 1;
        }

        let ds = state.registers[crate::state::reg::DS];
        let es = state.registers[crate::state::reg::ES];
        let flags = state.registers[crate::state::reg::FLAGS];
        let df = (flags >> 10) & 1;

        let is_repnz = prefix == 0xF2;
        let is_repz = prefix == 0xF3;

        let mut si = state.registers[crate::state::reg::SI] & 0xFFFF;
        let mut di = state.registers[crate::state::reg::DI] & 0xFFFF;
        let mut remaining = cx;
        let mut last_flags = flags;
        let mem_len = state.memory.len();

        match opcode {
            // STOSB: store AL to ES:DI
            0xAA => {
                let al = (state.registers[crate::state::reg::AX] & 0xFF) as u8;
                for _ in 0..remaining {
                    let addr = ((es * 16 + di) & 0xFFFFF) as usize;
                    if addr < mem_len { state.memory[addr] = al; }
                    di = if df == 0 { (di + 1) & 0xFFFF } else { (di - 1) & 0xFFFF };
                }
                remaining = 0;
            }
            // STOSW: store AX to ES:DI
            0xAB => {
                let ax = state.registers[crate::state::reg::AX] & 0xFFFF;
                let lo = (ax & 0xFF) as u8;
                let hi = ((ax >> 8) & 0xFF) as u8;
                for _ in 0..remaining {
                    let addr = ((es * 16 + di) & 0xFFFFF) as usize;
                    if addr < mem_len { state.memory[addr] = lo; }
                    if addr + 1 < mem_len { state.memory[addr + 1] = hi; }
                    di = if df == 0 { (di + 2) & 0xFFFF } else { (di - 2) & 0xFFFF };
                }
                remaining = 0;
            }
            // MOVSB: copy byte DS:SI to ES:DI
            0xA4 => {
                for _ in 0..remaining {
                    let src = ((ds * 16 + si) & 0xFFFFF) as usize;
                    let dst = ((es * 16 + di) & 0xFFFFF) as usize;
                    let byte = if src < mem_len { state.memory[src] } else { 0 };
                    if dst < mem_len { state.memory[dst] = byte; }
                    si = if df == 0 { (si + 1) & 0xFFFF } else { (si - 1) & 0xFFFF };
                    di = if df == 0 { (di + 1) & 0xFFFF } else { (di - 1) & 0xFFFF };
                }
                remaining = 0;
            }
            // MOVSW: copy word DS:SI to ES:DI
            0xA5 => {
                for _ in 0..remaining {
                    let src = ((ds * 16 + si) & 0xFFFFF) as usize;
                    let dst = ((es * 16 + di) & 0xFFFFF) as usize;
                    let lo = if src < mem_len { state.memory[src] } else { 0 };
                    let hi = if src + 1 < mem_len { state.memory[src + 1] } else { 0 };
                    if dst < mem_len { state.memory[dst] = lo; }
                    if dst + 1 < mem_len { state.memory[dst + 1] = hi; }
                    si = if df == 0 { (si + 2) & 0xFFFF } else { (si - 2) & 0xFFFF };
                    di = if df == 0 { (di + 2) & 0xFFFF } else { (di - 2) & 0xFFFF };
                }
                remaining = 0;
            }
            // SCASB: compare AL with ES:DI byte
            0xAE => {
                let al = state.registers[crate::state::reg::AX] & 0xFF;
                while remaining > 0 {
                    let addr = ((es * 16 + di) & 0xFFFFF) as usize;
                    let byte = if addr < mem_len { state.memory[addr] as i32 } else { 0 };
                    di = if df == 0 { (di + 1) & 0xFFFF } else { (di - 1) & 0xFFFF };
                    remaining -= 1;
                    let diff = (al - byte) & 0xFF;
                    let zf = if diff == 0 { 1 } else { 0 };
                    last_flags = (last_flags & !0xC5) | (zf << 6)
                        | (if diff & 0x80 != 0 { 0x80 } else { 0 })
                        | (if al < byte { 1 } else { 0 });
                    if is_repnz && zf == 1 { break; }
                    if is_repz && zf == 0 { break; }
                }
            }
            // SCASW: compare AX with ES:DI word
            0xAF => {
                let ax = state.registers[crate::state::reg::AX] & 0xFFFF;
                while remaining > 0 {
                    let addr = ((es * 16 + di) & 0xFFFFF) as usize;
                    let lo = if addr < mem_len { state.memory[addr] as i32 } else { 0 };
                    let hi = if addr + 1 < mem_len { state.memory[addr + 1] as i32 } else { 0 };
                    let word = lo + hi * 256;
                    di = if df == 0 { (di + 2) & 0xFFFF } else { (di - 2) & 0xFFFF };
                    remaining -= 1;
                    let diff = (ax - word) & 0xFFFF;
                    let zf = if diff == 0 { 1 } else { 0 };
                    last_flags = (last_flags & !0xC5) | (zf << 6)
                        | (if diff & 0x8000 != 0 { 0x80 } else { 0 })
                        | (if ax < word { 1 } else { 0 });
                    if is_repnz && zf == 1 { break; }
                    if is_repz && zf == 0 { break; }
                }
            }
            // CMPSB: compare DS:SI byte with ES:DI byte
            0xA6 => {
                while remaining > 0 {
                    let src = ((ds * 16 + si) & 0xFFFFF) as usize;
                    let dst = ((es * 16 + di) & 0xFFFFF) as usize;
                    let a = if src < mem_len { state.memory[src] as i32 } else { 0 };
                    let b = if dst < mem_len { state.memory[dst] as i32 } else { 0 };
                    si = if df == 0 { (si + 1) & 0xFFFF } else { (si - 1) & 0xFFFF };
                    di = if df == 0 { (di + 1) & 0xFFFF } else { (di - 1) & 0xFFFF };
                    remaining -= 1;
                    let diff = (a - b) & 0xFF;
                    let zf = if diff == 0 { 1 } else { 0 };
                    last_flags = (last_flags & !0xC5) | (zf << 6)
                        | (if diff & 0x80 != 0 { 0x80 } else { 0 })
                        | (if a < b { 1 } else { 0 });
                    if is_repnz && zf == 1 { break; }
                    if is_repz && zf == 0 { break; }
                }
            }
            // CMPSW: compare DS:SI word with ES:DI word
            0xA7 => {
                while remaining > 0 {
                    let src = ((ds * 16 + si) & 0xFFFFF) as usize;
                    let dst = ((es * 16 + di) & 0xFFFFF) as usize;
                    let a = if src < mem_len { state.memory[src] as i32 } else { 0 };
                    let a_hi = if src + 1 < mem_len { state.memory[src + 1] as i32 } else { 0 };
                    let b = if dst < mem_len { state.memory[dst] as i32 } else { 0 };
                    let b_hi = if dst + 1 < mem_len { state.memory[dst + 1] as i32 } else { 0 };
                    let a_word = a + a_hi * 256;
                    let b_word = b + b_hi * 256;
                    si = if df == 0 { (si + 2) & 0xFFFF } else { (si - 2) & 0xFFFF };
                    di = if df == 0 { (di + 2) & 0xFFFF } else { (di - 2) & 0xFFFF };
                    remaining -= 1;
                    let diff = (a_word - b_word) & 0xFFFF;
                    let zf = if diff == 0 { 1 } else { 0 };
                    last_flags = (last_flags & !0xC5) | (zf << 6)
                        | (if diff & 0x8000 != 0 { 0x80 } else { 0 })
                        | (if a_word < b_word { 1 } else { 0 });
                    if is_repnz && zf == 1 { break; }
                    if is_repz && zf == 0 { break; }
                }
            }
            // LODSB: load byte from DS:SI into AL
            0xAC => {
                for _ in 0..remaining {
                    let src = ((ds * 16 + si) & 0xFFFFF) as usize;
                    let byte = if src < mem_len { state.memory[src] as i32 } else { 0 };
                    state.registers[crate::state::reg::AX] =
                        (state.registers[crate::state::reg::AX] & 0xFF00) | (byte & 0xFF);
                    si = if df == 0 { (si + 1) & 0xFFFF } else { (si - 1) & 0xFFFF };
                }
                remaining = 0;
            }
            // LODSW: load word from DS:SI into AX
            0xAD => {
                for _ in 0..remaining {
                    let src = ((ds * 16 + si) & 0xFFFFF) as usize;
                    let lo = if src < mem_len { state.memory[src] as i32 } else { 0 };
                    let hi = if src + 1 < mem_len { state.memory[src + 1] as i32 } else { 0 };
                    state.registers[crate::state::reg::AX] = lo + hi * 256;
                    si = if df == 0 { (si + 2) & 0xFFFF } else { (si - 2) & 0xFFFF };
                }
                remaining = 0;
            }
            // 0xA8/0xA9 are TEST, not string ops — shouldn't match
            _ => return 0,
        }

        state.registers[crate::state::reg::CX] = remaining;
        state.registers[crate::state::reg::SI] = si;
        state.registers[crate::state::reg::DI] = di;
        state.registers[crate::state::reg::FLAGS] = last_flags;
        state.registers[crate::state::reg::IP] = ip + 2;
        state.frame_counter += 1;
        1
    }

    /// Apply computed property values to state and return the changes.
    ///
    /// Only writes canonical (non-prefixed) properties to state.
    /// Buffer copies (`--__0AX`, `--__1AX`, `--__2AX`) are skipped —
    /// they exist for x86CSS's triple-buffer pipeline but carry stale values
    /// that would nondeterministically overwrite the current tick's result.
    fn apply_state(&self, state: &mut State) -> Vec<(String, String)> {
        let mut changes = Vec::new();

        for (name, &value) in &self.properties {
            // Skip buffer copies — only the canonical name should write to state
            if name.starts_with("--__0") || name.starts_with("--__1") || name.starts_with("--__2") {
                continue;
            }
            // Skip byte-half properties (AL/AH/BL/BH/CL/CH/DL/DH).
            // In CSS, these are read-only views of the full register: e.g.
            //   --AL: --lowerBytes(var(--__1AX), 8)
            //   --AH: --rightShift(var(--__1AX), 8)
            // The full register formulas (--AX, etc.) already handle byte
            // merging when a byte write occurs: e.g. destA==-31 (AL) triggers
            //   --AX: floor(AX/256)*256 + lowerBytes(valA, 8)
            // Writing the byte halves back would clobber the updated full
            // register with stale values from the OLD tick.
            if is_byte_half(name) {
                continue;
            }
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

/// Reorder assignments so `--modRm*` properties are computed before `--instId`.
///
/// CSS evaluates all custom properties simultaneously, but our sequential evaluator
/// processes them in declaration order. The `--getInstId()` function references
/// `--modRm_reg` in compound conditions to distinguish instruction group encodings
/// (e.g., opcode 0xFF can be INC, DEC, CALL, JMP, or PUSH depending on the ModR/M
/// reg field). Without reordering, `--modRm_reg` defaults to 0 and all 0xFF group
/// instructions are incorrectly decoded as INC.
fn reorder_modrm_before_instid(mut assignments: Vec<Assignment>) -> Vec<Assignment> {
    // Find the position of --instId
    let inst_id_pos = assignments.iter().position(|a| a.property == "--instId");
    if inst_id_pos.is_none() {
        return assignments;
    }
    let inst_id_pos = inst_id_pos.unwrap();

    // Find all --modRm* assignments that come AFTER --instId
    let modrm_names = ["--modRm", "--modRm_rm", "--modRm_reg", "--modRm_mod"];
    let mut to_move = Vec::new();
    let mut i = assignments.len();
    while i > inst_id_pos + 1 {
        i -= 1;
        if modrm_names.contains(&assignments[i].property.as_str()) {
            to_move.push(assignments.remove(i));
        }
    }
    if to_move.is_empty() {
        return assignments;
    }

    // Re-find instId position (may have shifted after removals)
    let inst_id_pos = assignments
        .iter()
        .position(|a| a.property == "--instId")
        .unwrap();

    // Insert the modRm assignments before --instId, in their original relative order
    to_move.reverse();
    for (offset, a) in to_move.into_iter().enumerate() {
        log::info!("Reordering {} before --instId", a.property);
        assignments.insert(inst_id_pos + offset, a);
    }

    assignments
}

/// Check if a property names a byte-half register (AL, AH, BL, BH, etc.).
///
/// In CSS, these are read-only views computed from the full register each tick.
/// The full register formula (e.g. `--AX`) handles byte merging on writes,
/// so byte halves must NOT write back to state (they'd clobber with stale values).
fn is_byte_half(name: &str) -> bool {
    let bare = to_bare_name(name);
    matches!(bare, "AL" | "AH" | "BL" | "BH" | "CL" | "CH" | "DL" | "DH")
}

/// Check if a property is a triple-buffer copy (`--__0*`, `--__1*`, `--__2*`).
///
/// These assignments exist for x86CSS's animation pipeline but are no-ops
/// in calcify's mutable-state model — they just copy the canonical value
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
/// Uses x86CSS's naming convention:
/// - `--AX`, `--CX`, ..., `--flags` → register addresses
/// - `--m0`, `--m1`, ... → memory addresses
///
/// Automatically strips triple-buffer prefixes (`--__0`, `--__1`, `--__2`).
pub fn property_to_address(name: &str) -> Option<i32> {
    use crate::state::addr;
    let canonical = to_bare_name(name);
    match canonical {
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
        _ if canonical.starts_with('m') => parse_mem_address(&canonical[1..]),
        _ => None,
    }
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

const MAX_CALL_DEPTH: usize = 64;

// --- Evaluation methods ---
//
// These methods need &mut self to write to self.properties/call_depth, but also
// read self.functions/dispatch_tables. We use raw pointers in eval_function_call
// and eval_dispatch_raw to read from immutable program data while mutating
// properties. This is safe because functions/dispatch_tables are never modified
// during evaluation.

impl Evaluator {
    /// Evaluate an expression to a numeric value.
    fn eval_expr(&mut self, expr: &Expr, state: &State) -> f64 {
        match expr {
            Expr::Literal(v) => *v,

            Expr::Var { name, fallback } => {
                let v = self.resolve_property(name, state);
                if v != 0.0 {
                    return v;
                }
                // Property might genuinely be 0, or might not exist.
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
                0.0
            }

            Expr::StringLiteral(_) => 0.0,

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
    ///
    /// For buffer-prefixed names (`--__1AX`), also checks the canonical name (`--AX`)
    /// in computed properties before falling back to state.
    fn resolve_property(&self, name: &str, state: &State) -> f64 {
        if let Some(&v) = self.properties.get(name) {
            return v;
        }
        if name.starts_with("--__") && name.len() > 5 {
            let suffix = &name[5..];
            if !self.properties.is_empty() {
                let canonical = format!("--{suffix}");
                if let Some(&v) = self.properties.get(&canonical) {
                    return v;
                }
            }
        }
        if let Some(addr) = property_to_address(name) {
            return state.read_mem(addr) as f64;
        }
        // Special I/O properties not mapped to the register/memory address space
        if name == "--keyboard" || name == "--__1keyboard" || name == "--__2keyboard" {
            return state.keyboard as f64;
        }
        0.0
    }

    /// Evaluate a style test (condition inside an `if()` branch).
    fn eval_style_test(&mut self, test: &StyleTest, state: &State) -> bool {
        match test {
            StyleTest::Single { property, value } => {
                let prop_val = self.resolve_property(property, state) as i64;
                let test_val = self.eval_expr(value, state) as i64;
                prop_val == test_val
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

        // Fast paths for common x86CSS helper functions.
        // These replace dispatch table lookups and CSS expression evaluation
        // with direct native operations.
        match name {
            // --readMem(addr) → direct state memory/register read.
            "--readMem" => {
                if let Some(arg) = args.first() {
                    let addr = self.eval_expr(arg, state) as i32;
                    return state.read_mem(addr) as f64;
                }
                return 0.0;
            }
            // --read2(addr) → 16-bit little-endian word read.
            "--read2" => {
                if let Some(arg) = args.first() {
                    let addr = self.eval_expr(arg, state) as i32;
                    if addr < 0 {
                        // Negative addresses are registers — read as single value
                        return state.read_mem(addr) as f64;
                    }
                    return state.read_mem16(addr) as f64;
                }
                return 0.0;
            }
            // --lowerBytes(a, b) → a % 2^b = a & ((1 << b) - 1)
            "--lowerBytes" => {
                if args.len() >= 2 {
                    let a = self.eval_expr(&args[0], state) as i64;
                    let b = self.eval_expr(&args[1], state) as u32;
                    if b >= 64 { return a as f64; }
                    return (a & ((1i64 << b) - 1)) as f64;
                }
                return 0.0;
            }
            // --rightShift(a, b) → floor(a / 2^b)
            "--rightShift" => {
                if args.len() >= 2 {
                    let a = self.eval_expr(&args[0], state) as i64;
                    let b = self.eval_expr(&args[1], state) as u32;
                    if b >= 64 { return 0.0; }
                    return (a >> b) as f64;
                }
                return 0.0;
            }
            // --leftShift(a, b) → a * 2^b
            "--leftShift" => {
                if args.len() >= 2 {
                    let a = self.eval_expr(&args[0], state) as i64;
                    let b = self.eval_expr(&args[1], state) as u32;
                    if b >= 64 { return 0.0; }
                    return (a << b) as f64;
                }
                return 0.0;
            }
            // --bit(val, idx) → (val >> idx) & 1
            "--bit" => {
                if args.len() >= 2 {
                    let val = self.eval_expr(&args[0], state) as i64;
                    let idx = self.eval_expr(&args[1], state) as u32;
                    if idx >= 64 { return 0.0; }
                    return ((val >> idx) & 1) as f64;
                }
                return 0.0;
            }
            // --int(i) → identity (no-op)
            "--int" => {
                if let Some(arg) = args.first() {
                    return self.eval_expr(arg, state);
                }
                return 0.0;
            }
            _ => {}
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
                return 0.0;
            }
        };
        // SAFETY: functions is not modified during evaluation.
        let func = unsafe { &*func };

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
    ) -> f64 {
        let func = self.functions.get(name).map(|f| f as *const FunctionDef);
        let old_props: Vec<(String, Option<f64>)> = if let Some(func_ptr) = func {
            let func = &*func_ptr;
            func.parameters
                .iter()
                .enumerate()
                .map(|(i, param)| {
                    let old = self.properties.get(&param.name).copied();
                    let val = args.get(i).map(|a| self.eval_expr(a, state)).unwrap_or(0.0);
                    self.properties.insert(param.name.clone(), val);
                    (param.name.clone(), old)
                })
                .collect()
        } else {
            Vec::new()
        };

        let key = self.resolve_property(key_property, state) as i64;

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

    /// Helper: create a minimal Evaluator for unit tests (no assignments/patterns).
    fn test_evaluator(
        functions: HashMap<String, FunctionDef>,
        dispatch_tables: HashMap<String, DispatchTable>,
    ) -> Evaluator {
        Evaluator {
            functions,
            assignments: vec![],
            dispatch_tables,
            broadcast_writes: vec![],
            properties: HashMap::with_capacity(16),
            call_depth: 0,
        }
    }

    #[test]
    fn eval_literal() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let state = State::default();
        assert_eq!(eval.eval_expr(&Expr::Literal(42.0), &state), 42.0);
    }

    #[test]
    fn eval_calc_operations() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let state = State::default();

        let expr = Expr::Calc(CalcOp::Add(
            Box::new(Expr::Literal(10.0)),
            Box::new(Expr::Literal(20.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state), 30.0);

        let expr = Expr::Calc(CalcOp::Mul(
            Box::new(Expr::Literal(3.0)),
            Box::new(Expr::Literal(7.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state), 21.0);

        let expr = Expr::Calc(CalcOp::Mod(
            Box::new(Expr::Literal(17.0)),
            Box::new(Expr::Literal(5.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state), 2.0);
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
        assert_eq!(eval.eval_expr(&expr, &state), 0x1234 as f64);
    }

    #[test]
    fn eval_var_fallback() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let state = State::default();

        let expr = Expr::Var {
            name: "--nonexistent".to_string(),
            fallback: Some(Box::new(Expr::Literal(99.0))),
        };
        assert_eq!(eval.eval_expr(&expr, &state), 99.0);
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

        assert_eq!(eval.eval_expr(&expr, &state), 200.0);
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
        assert_eq!(eval.eval_expr(&expr, &state), 7.0);
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
        assert_eq!(eval.eval_expr(&expr, &state), 42.0);
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
        assert_eq!(eval.eval_expr(&expr, &state), 999.0);

        let expr = Expr::FunctionCall {
            name: "--lookup".to_string(),
            args: vec![Expr::Literal(99.0)],
        };
        assert_eq!(eval.eval_expr(&expr, &state), 0.0);
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
        assert_eq!(eval.eval_expr(&expr, &state), 0.0);
    }

    #[test]
    fn eval_mod_by_zero() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let state = State::default();

        let expr = Expr::Calc(CalcOp::Mod(
            Box::new(Expr::Literal(17.0)),
            Box::new(Expr::Literal(0.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state), 0.0);
    }

    #[test]
    fn eval_negate() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let state = State::default();

        let expr = Expr::Calc(CalcOp::Negate(Box::new(Expr::Literal(42.0))));
        assert_eq!(eval.eval_expr(&expr, &state), -42.0);

        let expr = Expr::Calc(CalcOp::Negate(Box::new(Expr::Literal(-7.0))));
        assert_eq!(eval.eval_expr(&expr, &state), 7.0);
    }

    #[test]
    fn eval_sign_and_abs() {
        let mut eval = test_evaluator(HashMap::new(), HashMap::new());
        let state = State::default();

        assert_eq!(
            eval.eval_expr(
                &Expr::Calc(CalcOp::Sign(Box::new(Expr::Literal(42.0)))),
                &state
            ),
            1.0
        );
        assert_eq!(
            eval.eval_expr(
                &Expr::Calc(CalcOp::Sign(Box::new(Expr::Literal(-5.0)))),
                &state
            ),
            -1.0
        );
        assert_eq!(
            eval.eval_expr(
                &Expr::Calc(CalcOp::Sign(Box::new(Expr::Literal(0.0)))),
                &state
            ),
            0.0
        );
        assert_eq!(
            eval.eval_expr(
                &Expr::Calc(CalcOp::Abs(Box::new(Expr::Literal(-99.0)))),
                &state
            ),
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
        assert_eq!(eval.eval_expr(&expr, &state), 50.0);

        // Value below min
        let expr = Expr::Calc(CalcOp::Clamp(
            Box::new(Expr::Literal(10.0)),
            Box::new(Expr::Literal(5.0)),
            Box::new(Expr::Literal(100.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state), 10.0);

        // Value above max
        let expr = Expr::Calc(CalcOp::Clamp(
            Box::new(Expr::Literal(0.0)),
            Box::new(Expr::Literal(200.0)),
            Box::new(Expr::Literal(100.0)),
        ));
        assert_eq!(eval.eval_expr(&expr, &state), 100.0);
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
        assert_eq!(eval.eval_expr(&expr, &state), 0.0);
    }
}
