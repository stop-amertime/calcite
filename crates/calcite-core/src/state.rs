//! Emulator state — flat representation of the CSS machine.
//!
//! State variables are discovered from CSS `@property` declarations at load time.
//! Calcite has no hardcoded knowledge of what those variables represent — they
//! could be x86 registers, halt flags, micro-op counters, or anything else the
//! CSS program defines. The negative address convention (slot 0 → -1, slot 1 → -2,
//! etc.) simply separates state variables from memory bytes in the unified
//! address space.

/// Default memory size for x86CSS (0x600 bytes = 1,536).
pub const DEFAULT_MEM_SIZE: usize = 0x600;

/// 16-color CGA palette used by [`State::render_framebuffer`].
///
/// CSS-DOS shadows the VGA DAC palette (768 bytes: 256 entries × RGB) at
/// this out-of-1MB linear address. OUT 0x3C9 writes populate it via the
/// kiln port-decode machinery (kiln/patterns/misc.mjs). Values are 6-bit
/// (0..63), matching real VGA hardware; `read_framebuffer_rgba` expands
/// them to 8-bit when producing canvas pixels. Keep in sync with
/// `DAC_LINEAR` in CSS-DOS's kiln/memory.mjs.
pub const VGA_DAC_LINEAR: i32 = 0x100000;

/// Standard IRGB CGA colors: the low 8 entries are the dark variants,
/// the high 8 are the bright variants. These also form the first 16
/// entries of the full VGA palette, so a future 256-entry upgrade is
/// backward-compatible with programs that only use indices 0-15.
pub const CGA_PALETTE: [(u8, u8, u8); 16] = [
    (0x00, 0x00, 0x00), // 0  black
    (0x00, 0x00, 0xAA), // 1  blue
    (0x00, 0xAA, 0x00), // 2  green
    (0x00, 0xAA, 0xAA), // 3  cyan
    (0xAA, 0x00, 0x00), // 4  red
    (0xAA, 0x00, 0xAA), // 5  magenta
    (0xAA, 0x55, 0x00), // 6  brown
    (0xAA, 0xAA, 0xAA), // 7  light gray
    (0x55, 0x55, 0x55), // 8  dark gray
    (0x55, 0x55, 0xFF), // 9  light blue
    (0x55, 0xFF, 0x55), // 10 light green
    (0x55, 0xFF, 0xFF), // 11 light cyan
    (0xFF, 0x55, 0x55), // 12 light red
    (0xFF, 0x55, 0xFF), // 13 light magenta
    (0xFF, 0xFF, 0x55), // 14 yellow
    (0xFF, 0xFF, 0xFF), // 15 white
];

/// The flat machine state that replaces CSS's triple-buffered custom properties.
#[derive(Debug, Clone)]
pub struct State {
    /// State variables discovered from CSS `@property` declarations.
    /// Addressed via negative numbers: slot 0 → addr -1, slot 1 → addr -2, etc.
    pub state_vars: Vec<i32>,
    /// State variable names, in slot order (for display/debugging).
    pub state_var_names: Vec<String>,
    /// Name → slot index lookup.
    pub(crate) state_var_index: std::collections::HashMap<String, usize>,
    /// Flat memory (byte-addressable, default 1,536 bytes).
    pub memory: Vec<u8>,
    /// Extended properties: full-width i32 storage for addresses above the byte
    /// memory range (e.g., file I/O counters at 0xFFFF0+). Reads/writes bypass
    /// the u8 truncation of the memory array.
    ///
    /// FxHashMap (not std::HashMap) — `read_mem` falls through here for the
    /// >0xF0000 region on every miss, and SipHash showed up as ~9% of worker
    /// CPU on the doom8088 flamegraph. FxHash is faster on small integer keys.
    pub extended: rustc_hash::FxHashMap<i32, i32>,
    /// String property values (e.g., `--textBuffer` for text output).
    pub string_properties: std::collections::HashMap<String, String>,
    /// Tick counter (incremented each evaluation cycle).
    pub frame_counter: u32,
    /// Optional per-tick read log. When the inner `Option` is `Some`, every
    /// `read_mem`/`read_mem16` call appends `(addr, value)`. `None` = no
    /// overhead beyond a single `is_some()` check. Interior mutability so
    /// `read_mem(&self)` can log without changing its signature.
    pub read_log: std::cell::RefCell<Option<Vec<(i32, i32)>>>,
    /// Optional per-tick write log. When `Some`, every `write_mem` call
    /// appends `(addr, value)`. Used by the memoisation-viability probe.
    pub write_log: Option<Vec<(i32, i32)>>,
    /// Packed-memory cell table: `packed_cell_table[cell_idx] = state_var_addr`
    /// (always negative; 0 means "no cell at this index"). When non-empty, it's
    /// installed at compile time by `CalciteEngine::new` from the compiled
    /// program's `packed_broadcast_writes[0].cell_table`. Reads through
    /// `read_mem` at positive linear addresses consult this table first so
    /// consumers (renderer, debugger, video-mode probes) see packed bytes as
    /// if they lived at their natural linear addresses.
    pub packed_cell_table: Vec<i32>,
    /// Bytes per packed cell (2 in current CSS-DOS). 0 means "no packed layout".
    pub packed_cell_size: u8,
    /// Optional pattern-recognised "windowed byte array" descriptor — when
    /// present, `read_mem` services addresses inside `window_base..window_end`
    /// by reading a key from a state-var cell, multiplying by `stride`,
    /// adding the in-window offset, and indexing `byte_array`. This is the
    /// shape the CSS-DOS rom-disk dispatch happens to take, but the descriptor
    /// is named in calcite's own vocabulary (window, key cell, byte array)
    /// not x86's (LBA, sector, disk) so calcite stays free of x86 knowledge.
    /// All fields are filled in at compile time by
    /// `Evaluator::wire_state_for_windowed_byte_array` from data the recogniser
    /// extracts out of the cabinet's `--readMem` dispatch entries — there is
    /// no CLI-only loading path; calcite-cli and calcite-wasm both use this
    /// path identically.
    pub windowed_byte_array: Option<WindowedByteArray>,
    /// Host-supplied pseudo-class state. Keyed by `(pseudo, selector)`
    /// (e.g. `("active", "kb-1")`); presence means the host has
    /// reported the gate as active. Read by gated assignments synthesised
    /// from `ParsedProgram::input_edges`. The host calls
    /// `set_pseudo_class_active` to flip an entry. Sparse — only edges
    /// the host actually drives appear here.
    ///
    /// Stored as a `HashSet` so the per-tick fast path
    /// (`apply_input_edges` in the evaluator) can short-circuit on
    /// `pseudo_active.is_empty()` without allocating lookup keys.
    pub pseudo_active: std::collections::HashSet<(String, String)>,
    /// Resolved input-edge groups, one per gated state-var slot.
    /// Installed by `Evaluator::wire_state_for_input_edges` once at
    /// engine construction (after state vars are loaded). Empty for
    /// cabinets without `&:has(…:pseudo)` rules — the entire smoke
    /// set except doom8088.
    ///
    /// With this installed, `set_pseudo_class_active` writes the gated
    /// slot directly at the moment of mutation, eliminating the
    /// per-tick gate + apply path entirely. The per-tick cost for a
    /// cabinet with input edges is the same as for one without: zero.
    pub input_edge_groups: Vec<InputEdgeGroup>,
}

/// One gated state-var slot's worth of input edges. The slot is
/// resolved from the binding's bare property name via
/// `state.var_slot()` at wiring time; the `edges` list carries the
/// `(pseudo, selector) → value` mapping for every edge that targets
/// this slot. `set_pseudo_class_active` recomputes the slot value by
/// summing the values of edges whose `(pseudo, selector)` is currently
/// active.
#[derive(Debug, Clone)]
pub struct InputEdgeGroup {
    /// The state-var slot all `edges` in this group write to.
    pub slot: usize,
    /// Edges whose `bare(property)` resolved to `slot`.
    pub edges: Vec<InputEdgeGroupEntry>,
}

/// A single `&:has(#SELECTOR:PSEUDO) { --PROP: VAL }` rule, resolved to
/// the bare strings (no `#` / no `--`) and the literal value extracted
/// at compile time.
#[derive(Debug, Clone)]
pub struct InputEdgeGroupEntry {
    pub pseudo: String,
    pub selector: String,
    pub value: i32,
}

/// A "window of bytes addressed by an in-memory key" — a CSS shape where a
/// contiguous address window is serviced by reading a key from a state-var
/// cell, multiplying by a stride, adding the in-window offset, and indexing a
/// flat byte array. Lives on `State` so the read fast path can resolve
/// in-window addresses without going back through the compiled CSS expression
/// for every byte.
///
/// All fields are derived at compile time from the cabinet's own dispatch
/// entries; nothing here reflects any upstream concept — calcite sees a
/// window, a key cell, and a byte array.
#[derive(Debug, Clone)]
pub struct WindowedByteArray {
    /// First linear address inside the window (inclusive).
    pub window_base: i32,
    /// One past the last linear address inside the window.
    pub window_end: i32,
    /// State-var slot index holding the lookup key (a u16 cell — low byte
    /// and high byte combined). Resolved from the property name extracted
    /// from the dispatch entry by `wire_state_for_windowed_byte_array`.
    pub key_cell_slot: usize,
    /// Multiplier applied to the cell value before adding in-window offset.
    pub stride: i32,
    /// First key represented by `byte_array[0]`. Subtract this from the
    /// computed key before indexing. (The cabinet's `--readDiskByte` flat
    /// array is base-keyed at the first non-zero entry, not at zero.)
    pub byte_array_base_key: i32,
    /// The flat byte array `--readDiskByte`'s dispatch compiles to. Shared
    /// with `CompiledProgram::flat_dispatch_arrays` so we don't duplicate
    /// the (potentially multi-MB) byte data.
    pub byte_array: std::sync::Arc<Vec<i32>>,
}

impl State {
    /// Create a new state with the given memory size.
    pub fn new(mem_size: usize) -> Self {
        Self {
            state_vars: Vec::new(),
            state_var_names: Vec::new(),
            state_var_index: std::collections::HashMap::new(),
            memory: vec![0; mem_size],
            extended: rustc_hash::FxHashMap::default(),
            string_properties: std::collections::HashMap::new(),
            frame_counter: 0,
            read_log: std::cell::RefCell::new(None),
            write_log: None,
            packed_cell_table: Vec::new(),
            packed_cell_size: 0,
            windowed_byte_array: None,
            pseudo_active: std::collections::HashSet::new(),
            input_edge_groups: Vec::new(),
        }
    }

    /// Apply a delta forward — overwrites channels with `new_*` values.
    /// Panics in debug if slot/addr is out of range (caller must apply
    /// deltas against the same State shape that produced them).
    pub fn apply_delta(&mut self, d: &StateDelta) {
        for (slot, _, new) in &d.state_vars {
            self.state_vars[*slot as usize] = *new;
        }
        for (addr, _, new) in &d.memory {
            self.memory[*addr as usize] = *new;
        }
        for (addr, _, new) in &d.extended {
            match new {
                Some(v) => {
                    self.extended.insert(*addr, *v);
                }
                None => {
                    self.extended.remove(addr);
                }
            }
        }
        for (name, _, new) in &d.string_properties {
            match new {
                Some(v) => {
                    self.string_properties.insert(name.clone(), v.clone());
                }
                None => {
                    self.string_properties.remove(name);
                }
            }
        }
        self.frame_counter = d.frame_counter_after;
    }

    /// Apply a delta in reverse — overwrites channels with `old_*` values.
    /// This is what makes reverse seek possible without re-execution.
    pub fn revert_delta(&mut self, d: &StateDelta) {
        for (slot, old, _) in &d.state_vars {
            self.state_vars[*slot as usize] = *old;
        }
        for (addr, old, _) in &d.memory {
            self.memory[*addr as usize] = *old;
        }
        for (addr, old, _) in &d.extended {
            match old {
                Some(v) => {
                    self.extended.insert(*addr, *v);
                }
                None => {
                    self.extended.remove(addr);
                }
            }
        }
        for (name, old, _) in &d.string_properties {
            match old {
                Some(v) => {
                    self.string_properties.insert(name.clone(), v.clone());
                }
                None => {
                    self.string_properties.remove(name);
                }
            }
        }
        self.frame_counter = d.frame_counter_before;
    }

    /// Look up a state variable by name. Returns None if not found.
    pub fn get_var(&self, name: &str) -> Option<i32> {
        self.state_var_index.get(name).map(|&i| self.state_vars[i])
    }

    /// Set a state variable by name. Returns false if not found.
    pub fn set_var(&mut self, name: &str, value: i32) -> bool {
        if let Some(&i) = self.state_var_index.get(name) {
            self.state_vars[i] = value;
            true
        } else {
            false
        }
    }

    /// Look up the slot index for a state variable name.
    pub fn var_slot(&self, name: &str) -> Option<usize> {
        self.state_var_index.get(name).copied()
    }

    /// Report a pseudo-class match edge as active or inactive. The
    /// (pseudo, selector) pair must match an InputEdge the cabinet's
    /// CSS declared via `&:has(#SELECTOR:PSEUDO) { ... }`. Any gated
    /// state-var slot whose value depends on this edge is **recomputed
    /// in this call**, so by the time the next tick runs the slot
    /// already reflects the new active set.
    ///
    /// `value=true` means "the host considers this pseudo-class active
    /// on this element right now"; `value=false` means inactive. The
    /// host is responsible for sending matching false edges (release).
    ///
    /// Apply-on-transition is cheap because mutations are rare (one
    /// per real key event; sub-200 mutations across a 34M-tick
    /// doom-loading run). Each call sums values from
    /// `input_edge_groups`, which is small (1-3 groups on doom-style
    /// cabinets) — orders of magnitude less work than the previous
    /// per-tick gate + apply path it replaces.
    pub fn set_pseudo_class_active(&mut self, pseudo: &str, selector: &str, value: bool) {
        let changed = if value {
            // HashSet's API forces an owned key on insert; lookups below
            // can avoid this with a tuple-of-refs query but inserts can't.
            self.pseudo_active.insert((pseudo.to_string(), selector.to_string()))
        } else {
            self.pseudo_active.remove(&(pseudo.to_string(), selector.to_string()))
        };
        if !changed {
            return;
        }
        // Recompute every gated slot from scratch. Each slot's value is
        // the saturating sum of its active edges' literal values.
        //
        // Why recompute from scratch rather than delta-add/subtract?
        // Multiple `&:has(#X:active) { --P: V }` rules can target the
        // same slot with the same key but different values (cabinet
        // authoring quirk). CSS semantics for such cases are "last rule
        // wins" via cascade, but the recogniser collects all of them.
        // Summing-from-scratch preserves the pre-existing
        // saturating-add behaviour of `apply_input_edges` without
        // needing to enforce uniqueness.
        for group in &self.input_edge_groups {
            let slot = group.slot;
            if slot >= self.state_vars.len() {
                continue;
            }
            let mut sum: i32 = 0;
            // Inverted iteration: walk active set (small — 0-2 entries
            // in steady state), match against this group's edges.
            // For each (pseudo, selector) currently active, sum the
            // values of edges that match. String compares by
            // reference; no allocations.
            for (active_pseudo, active_selector) in &self.pseudo_active {
                for entry in &group.edges {
                    if entry.pseudo == *active_pseudo
                        && entry.selector == *active_selector
                    {
                        sum = sum.saturating_add(entry.value);
                    }
                }
            }
            self.state_vars[slot] = sum;
        }
    }

    /// Read whether a pseudo-class match edge is currently active.
    ///
    /// Hot path: called once per input edge per tick during
    /// `apply_input_edges`. Avoid allocating the lookup key — the
    /// caller passes pre-owned `String`s and we look up by reference.
    pub fn pseudo_class_active(&self, pseudo: &str, selector: &str) -> bool {
        // Borrow trait on `(String, String)` lets `&(&str, &str)` query
        // without allocating. The HashSet stores owned tuples.
        // Note: HashSet::contains takes `&Q` where `(String, String): Borrow<Q>`,
        // and the standard library implements `Borrow<(str, str)>` for
        // `(String, String)` only in very recent Rust; safer to construct
        // the lookup pair explicitly with refs through `get`. Use the
        // by-ref helper.
        self.pseudo_class_active_pair(pseudo, selector)
    }

    /// Like `pseudo_class_active` but a fast inlined path used internally
    /// by `apply_input_edges`. Allocates if the Borrow trait route can't
    /// avoid it; in practice the standard library does support `Borrow<(str, str)>`
    /// for `(String, String)` since 1.84 — this helper papers over MSRV
    /// without a feature gate.
    #[inline]
    pub fn pseudo_class_active_pair(&self, pseudo: &str, selector: &str) -> bool {
        // Empty-set fast path.
        if self.pseudo_active.is_empty() {
            return false;
        }
        // We pay one allocation per lookup here (same as before). The
        // big win comes from the empty-set fast path above: 99 % of
        // ticks never reach this line.
        self.pseudo_active.contains(&(pseudo.to_string(), selector.to_string()))
    }

    /// Number of state variables.
    pub fn state_var_count(&self) -> usize {
        self.state_vars.len()
    }

    /// Read from the unified address space.
    ///
    /// Address conventions:
    /// - Negative: state variables (slot index = -addr - 1)
    /// - `0..`: memory bytes
    ///
    /// Packed cabinets store guest memory in state-var cells, not in
    /// `self.memory[]`. For positive guest addresses we consult the packed
    /// cell table first — that is the authoritative storage that the CPU's
    /// compiled LoadPackedByte op reads. Only on a miss (no cell declared
    /// for this byte) do we fall back to the flat `self.memory[]` shadow.
    /// On unpacked cabinets `packed_cell_table` is empty and the shadow is
    /// authoritative.
    #[inline]
    pub fn read_mem(&self, addr: i32) -> i32 {
        let v = if addr < 0 {
            let idx = (-addr - 1) as usize;
            if idx < self.state_vars.len() {
                // SAFETY: idx < state_vars.len() checked on the line above.
                unsafe { *self.state_vars.get_unchecked(idx) }
            } else {
                0
            }
        } else {
            let addr_u = addr as usize;
            // Packed-cell path first, for any positive address. Packed
            // cells cover conventional RAM (0..0xA0000) on most cabinets
            // and also out-of-range regions like the VGA DAC at 0x100000
            // that live above 1 MB; those cells are emitted by kiln for
            // both, so a single lookup covers both cases. The hot path
            // (~96% of bytecode ops are LoadStateAndBranchIfNotEqLit on
            // normal RAM) returns from inside this block; everything below
            // is for misses (BIOS, disk window, MMIO).
            if !self.packed_cell_table.is_empty() && self.packed_cell_size > 0 {
                let pack = self.packed_cell_size as usize;
                let cell_idx = addr_u / pack;
                if cell_idx < self.packed_cell_table.len() {
                    // SAFETY: cell_idx < packed_cell_table.len() checked above.
                    let cell_addr = unsafe { *self.packed_cell_table.get_unchecked(cell_idx) };
                    if cell_addr < 0 {
                        let sidx = (-cell_addr - 1) as usize;
                        if sidx < self.state_vars.len() {
                            // SAFETY: sidx < state_vars.len() checked above.
                            let cell = unsafe { *self.state_vars.get_unchecked(sidx) };
                            let off = addr_u % pack;
                            return ((cell >> (8 * off as u32)) & 0xFF) as i32;
                        }
                    }
                }
                // Cell table exists but no entry for this address — fall
                // through to the legacy paths below (extended HashMap for
                // out-of-1MB regions; flat shadow for BIOS ROM literals).
            }
            // Pattern-recognised "windowed byte array" — CSS-DOS uses this
            // shape for the rom-disk window, but the descriptor is in
            // calcite's own vocabulary (window/key cell/byte array).
            // Resolves the window read in O(1): one state-var read for the
            // key, one flat-array index. Same byte the compiled CSS path
            // would compute via the inline-exception chain → `--readDiskByte`
            // flat-array dispatch.
            //
            // Placed AFTER the packed-cell check because the windowed byte
            // array's addresses (e.g. 0xD0000-0xD01FF) sit well above the
            // packed cell table's range, so packed reads never fall in here.
            // Each byte of conventional RAM (the 96% hot path) would otherwise
            // pay for a windowed-byte-array check on every read_mem call —
            // measured as a ~30% web slowdown when the check came first.
            if let Some(ref dw) = self.windowed_byte_array {
                if addr >= dw.window_base && addr < dw.window_end {
                    let key = self.state_vars
                        .get(dw.key_cell_slot)
                        .copied()
                        .unwrap_or(0);
                    let computed_key = key.wrapping_mul(dw.stride)
                        .wrapping_add(addr - dw.window_base);
                    let idx = computed_key.wrapping_sub(dw.byte_array_base_key);
                    if idx >= 0 {
                        if let Some(&v) = dw.byte_array.get(idx as usize) {
                            if let Ok(mut borrow) = self.read_log.try_borrow_mut() {
                                if let Some(ref mut log) = *borrow { log.push((addr, v)); }
                            }
                            return v;
                        }
                    }
                    // Out of array → byte was 0 in the cabinet (the flat
                    // array's `default`).
                    if let Ok(mut borrow) = self.read_log.try_borrow_mut() {
                        if let Some(ref mut log) = *borrow { log.push((addr, 0)); }
                    }
                    return 0;
                }
            }
            if addr_u >= 0xF0000 {
                self.extended.get(&addr).copied().unwrap_or(0)
            } else if addr_u < self.memory.len() {
                self.memory[addr_u] as i32
            } else {
                0
            }
        };
        // Optional read log for memoisation-viability probing.
        if let Ok(mut borrow) = self.read_log.try_borrow_mut() {
            if let Some(ref mut log) = *borrow {
                log.push((addr, v));
            }
        }
        v
    }

    /// Read a 16-bit little-endian word from memory.
    #[inline]
    pub fn read_mem16(&self, addr: i32) -> i32 {
        // Fast path: when both bytes land in the same packed cell, one
        // state-var read + one 16-bit extract replaces two full read_mem
        // traversals. This requires addr >= 0, an active packed layout,
        // pack == 2 (the only value emitted today), and addr aligned to
        // the cell boundary so addr and addr+1 share a cell.
        if addr >= 0 && self.packed_cell_size == 2 && (addr & 1) == 0 {
            let addr_u = addr as usize;
            let cell_idx = addr_u >> 1;
            if cell_idx < self.packed_cell_table.len() {
                // SAFETY: cell_idx < packed_cell_table.len() checked above.
                let cell_addr = unsafe { *self.packed_cell_table.get_unchecked(cell_idx) };
                if cell_addr < 0 {
                    let sidx = (-cell_addr - 1) as usize;
                    if sidx < self.state_vars.len() {
                        // SAFETY: sidx < state_vars.len() checked above.
                        let cell = unsafe { *self.state_vars.get_unchecked(sidx) };
                        let lo_hi = cell & 0xFFFF;
                        // Optional read log: append both bytes to mirror the
                        // two-call path's behaviour for memoisation probes.
                        if let Ok(mut borrow) = self.read_log.try_borrow_mut() {
                            if let Some(ref mut log) = *borrow {
                                log.push((addr, lo_hi & 0xFF));
                                log.push((addr + 1, (lo_hi >> 8) & 0xFF));
                            }
                        }
                        return lo_hi;
                    }
                }
            }
            // Cell-table miss for this packed-aligned address — fall through
            // to the two-call path so the windowed-byte-array / extended /
            // shadow-memory fallbacks all run as before.
        }
        let lo = self.read_mem(addr);
        let hi = self.read_mem(addr + 1);
        lo + hi * 256
    }

    /// Write a value to the unified address space.
    ///
    /// Positive addresses below 0xF0000 are treated as single-byte guest
    /// memory stores. On a packed cabinet (where guest memory lives in
    /// `state_vars[]` indexed via `packed_cell_table`) the write is routed
    /// into the appropriate cell. The flat `self.memory[]` shadow is kept
    /// updated in parallel so read paths that bypass packing (e.g., the
    /// renderer staging in `read_framebuffer_rgba`, or uninitialised BIOS
    /// populate) still see the latest byte.
    pub fn write_mem(&mut self, addr: i32, value: i32) {
        if let Some(ref mut log) = self.write_log {
            log.push((addr, value));
        }
        if addr < 0 {
            let idx = (-addr - 1) as usize;
            if idx < self.state_vars.len() {
                self.state_vars[idx] = value;
            }
            return;
        }
        let addr_u = addr as usize;
        if addr_u >= 0xF0000 {
            self.extended.insert(addr, value);
            return;
        }
        let byte = (value & 0xFF) as u8;
        self.write_byte_packed_aware(addr_u, byte);
    }

    /// Internal: store a single byte at a positive linear address,
    /// honouring the packed-cell layout when active.
    #[inline]
    fn write_byte_packed_aware(&mut self, addr: usize, byte: u8) {
        // Always update the flat memory shadow so code paths that read it
        // directly see the same value. (Renderer/debugger tooling uses it.)
        if addr < self.memory.len() {
            self.memory[addr] = byte;
        }
        // If a packed cell layout is active, also splice the byte into the
        // state-var backing the cell. Without this, packed cabinets ignore
        // writes coming from runtime helpers (REP fast-forward, BDA
        // keyboard pushes, etc.) since the CPU's CSS read path consults
        // state_vars, not self.memory.
        if !self.packed_cell_table.is_empty() && self.packed_cell_size > 0 {
            let pack = self.packed_cell_size as usize;
            let cell_idx = addr / pack;
            if cell_idx < self.packed_cell_table.len() {
                let cell_addr = self.packed_cell_table[cell_idx];
                if cell_addr < 0 {
                    let sidx = (-cell_addr - 1) as usize;
                    if sidx < self.state_vars.len() {
                        let off = addr % pack;
                        let cell = self.state_vars[sidx];
                        let shift = 8 * off as u32;
                        let mask = !(0xFFi32 << shift);
                        self.state_vars[sidx] =
                            (cell & mask) | ((byte as i32) << shift);
                    }
                }
            }
        }
    }

    /// Fill a byte range through the packed-aware write path. Semantically
    /// equivalent to calling `write_mem(addr, byte)` for each byte, but
    /// avoids the per-byte boundary re-checks and keeps the flat
    /// `self.memory[]` in lock-step with the state-var cells.
    pub fn bulk_fill_byte(&mut self, addr: usize, count: usize, byte: u8) {
        if count == 0 { return; }
        let end = addr.saturating_add(count);
        let mem_end = self.memory.len();
        // Update the flat shadow first (cheap, keeps raw readers consistent).
        if addr < mem_end {
            let hi = end.min(mem_end);
            self.memory[addr..hi].fill(byte);
        }
        if self.packed_cell_table.is_empty() || self.packed_cell_size == 0 {
            return;
        }
        let pack = self.packed_cell_size as usize;
        let table_len = self.packed_cell_table.len();
        for i in addr..end {
            let cell_idx = i / pack;
            if cell_idx >= table_len { break; }
            let cell_addr = self.packed_cell_table[cell_idx];
            if cell_addr >= 0 { continue; }
            let sidx = (-cell_addr - 1) as usize;
            if sidx >= self.state_vars.len() { continue; }
            let off = i % pack;
            let cell = self.state_vars[sidx];
            let shift = 8 * off as u32;
            let mask = !(0xFFi32 << shift);
            self.state_vars[sidx] =
                (cell & mask) | ((byte as i32) << shift);
        }
    }

    /// Copy a byte range through the packed-aware write path, matching the
    /// per-byte read-then-write semantics of REP MOVSB on x86 (forward
    /// direction — the only direction this is called from, since
    /// `rep_fast_forward` bails on DF=1).
    ///
    /// **Overlap matters here.** If `dst > src` and the ranges overlap,
    /// later iterations read bytes that were *just written* by earlier
    /// iterations. That is exactly the LZ77 self-reference UPX (and any
    /// LZ-style decompressor) relies on: `MOV SI,dst-N; MOV CX,len;
    /// REP MOVSB` to repeat the last N bytes. A `memmove`-style snapshot
    /// would break that — the copy would pull stale source bytes and
    /// produce garbage. Symptom: UPX-packed COMMAND.COM exits via INT 20h
    /// because its decompression output is corrupt, kernel reads it as
    /// "shell exited cleanly" and prints "Bad or missing command interpreter".
    ///
    /// Forward-overlap with `src > dst` is fine either way (writes don't
    /// touch unread source bytes); backward-overlap (REP MOVSB with DF=1)
    /// would be wrong with this implementation, but the fast-forward bails
    /// before getting here in that case.
    pub fn bulk_copy_bytes(&mut self, src: usize, dst: usize, count: usize) {
        if count == 0 { return; }
        for i in 0..count {
            let byte = (self.read_mem((src + i) as i32) & 0xFF) as u8;
            self.write_byte_packed_aware(dst + i, byte);
        }
    }

    /// Push a key into the BDA keyboard ring buffer.
    ///
    /// `key` is packed as `(scancode << 8) | ascii`, matching the IBM PC
    /// BIOS convention. This is what the 8042 keyboard controller / INT 9
    /// handler does on real hardware.
    ///
    /// The buffer lives at segment 0x40 (linear 0x400 + BDA offset):
    /// - 0x41A: head pointer (offset into buffer)
    /// - 0x41C: tail pointer
    /// - 0x480: buffer start offset
    /// - 0x482: buffer end offset
    ///
    /// If the buffer is full the key is silently dropped (matching real hardware).
    pub fn bda_push_key(&mut self, key: i32) {
        let tail      = self.read_mem16(0x41C);
        let buf_end   = self.read_mem16(0x482);
        let buf_start = self.read_mem16(0x480);
        let linear    = 0x400 + tail;
        self.write_mem(linear,     key & 0xFF);
        self.write_mem(linear + 1, (key >> 8) & 0xFF);
        let mut new_tail = tail + 2;
        if new_tail >= buf_end { new_tail = buf_start; }
        let head = self.read_mem16(0x41A);
        if new_tail != head {
            // Not full — advance tail.
            self.write_mem(0x41C, new_tail & 0xFF);
            self.write_mem(0x41D, (new_tail >> 8) & 0xFF);
        }
    }

    /// Get the low byte of a 16-bit register value.
    pub fn lo8(value: i32) -> i32 {
        value & 0xFF
    }

    /// Get the high byte of a 16-bit register value.
    pub fn hi8(value: i32) -> i32 {
        (value >> 8) & 0xFF
    }

    /// Render text-mode video memory as a string.
    ///
    /// Reads `width * height` character cells from `base_addr` in text-mode
    /// format (2 bytes per cell: character byte + attribute byte). Returns
    /// the screen contents as a string with newline-separated rows.
    ///
    /// For DOS text mode at 0xB8000, call:
    /// `state.render_screen(0xB8000, 40, 25)`
    pub fn render_screen(&self, base_addr: usize, width: usize, height: usize) -> String {
        let mut lines = Vec::with_capacity(height);
        for y in 0..height {
            let mut row = String::with_capacity(width);
            for x in 0..width {
                let addr = base_addr + (y * width + x) * 2;
                let ch = if addr < self.memory.len() {
                    self.memory[addr]
                } else {
                    b' '
                };
                row.push(cp437_to_unicode(ch));
            }
            lines.push(row);
        }
        lines.join("\n")
    }

    /// Render text-mode video memory with ANSI 24-bit color escapes.
    ///
    /// Reads char+attribute pairs; the attribute byte is
    /// `BBBFFFFF` where the low nibble is foreground (0-15) and the high
    /// nibble's low 3 bits are background (0-7). Bit 7 is treated as the
    /// bright-background (no blink). Only emits a color-change escape when
    /// the attribute changes, so the output stays compact.
    pub fn render_screen_ansi(&self, base_addr: usize, width: usize, height: usize) -> String {
        let mut out = String::with_capacity(width * height * 4);
        let mut last_attr: Option<u8> = None;
        for y in 0..height {
            for x in 0..width {
                let addr = base_addr + (y * width + x) * 2;
                let (ch, attr) = if addr + 1 < self.memory.len() {
                    (self.memory[addr], self.memory[addr + 1])
                } else {
                    (b' ', 0x07)
                };
                if last_attr != Some(attr) {
                    let fg_idx = (attr & 0x0F) as usize;
                    // High bit often means "blink"; most programs really want
                    // bright BG, so treat bits 4-7 as a 4-bit BG index.
                    let bg_idx = ((attr >> 4) & 0x0F) as usize;
                    let (fr, fg, fb) = CGA_PALETTE[fg_idx];
                    let (br, bg_, bb) = CGA_PALETTE[bg_idx];
                    use std::fmt::Write;
                    let _ = write!(
                        out,
                        "\x1b[38;2;{};{};{};48;2;{};{};{}m",
                        fr, fg, fb, br, bg_, bb
                    );
                    last_attr = Some(attr);
                }
                out.push(cp437_to_unicode(ch));
            }
            // Reset before newline so the trailing background doesn't bleed
            // across the whole terminal line.
            out.push_str("\x1b[0m");
            last_attr = None;
            if y + 1 < height {
                out.push('\n');
            }
        }
        out
    }

    /// Render text-mode video memory as HTML with inline color spans.
    ///
    /// Same attribute decoding as [`Self::render_screen_ansi`]; emits a
    /// `<span style="color:#rrggbb;background:#rrggbb">` run whenever the
    /// attribute changes. Characters are CP437→Unicode translated, with
    /// `<`, `>`, `&` HTML-escaped. Rows are separated by `<br>`.
    pub fn render_screen_html(&self, base_addr: usize, width: usize, height: usize) -> String {
        use std::fmt::Write;
        let mut out = String::with_capacity(width * height * 8);
        let mut last_attr: Option<u8> = None;
        for y in 0..height {
            for x in 0..width {
                let addr = base_addr + (y * width + x) * 2;
                let (ch, attr) = if addr + 1 < self.memory.len() {
                    (self.memory[addr], self.memory[addr + 1])
                } else {
                    (b' ', 0x07)
                };
                if last_attr != Some(attr) {
                    if last_attr.is_some() {
                        out.push_str("</span>");
                    }
                    let fg_idx = (attr & 0x0F) as usize;
                    let bg_idx = ((attr >> 4) & 0x0F) as usize;
                    let (fr, fg, fb) = CGA_PALETTE[fg_idx];
                    let (br, bg_, bb) = CGA_PALETTE[bg_idx];
                    let _ = write!(
                        out,
                        "<span style=\"color:#{:02x}{:02x}{:02x};background:#{:02x}{:02x}{:02x}\">",
                        fr, fg, fb, br, bg_, bb
                    );
                    last_attr = Some(attr);
                }
                let c = cp437_to_unicode(ch);
                match c {
                    '<' => out.push_str("&lt;"),
                    '>' => out.push_str("&gt;"),
                    '&' => out.push_str("&amp;"),
                    _ => out.push(c),
                }
            }
            if last_attr.is_some() {
                out.push_str("</span>");
                last_attr = None;
            }
            if y + 1 < height {
                out.push_str("<br>");
            }
        }
        out
    }

    /// Render a graphics-mode framebuffer as a PPM P6 image.
    ///
    /// Reads `width * height` bytes from `base_addr` where each byte is a
    /// palette index (byte-per-pixel, like VGA Mode 13h). Indices are looked
    /// up in [`CGA_PALETTE`] modulo 16 (palette is 16 colors for now; upgrade
    /// to a 256-entry VGA palette is a drop-in change). Returns a complete
    /// PPM P6 byte buffer that can be written to a file.
    ///
    /// For Mode 13h at 0xA0000, call:
    /// `state.render_framebuffer(0xA0000, 320, 200)`
    pub fn render_framebuffer(&self, base_addr: usize, width: usize, height: usize) -> Vec<u8> {
        let header = format!("P6\n{} {}\n255\n", width, height);
        let mut out = Vec::with_capacity(header.len() + width * height * 3);
        out.extend_from_slice(header.as_bytes());
        for i in 0..(width * height) {
            let addr = base_addr + i;
            let byte = if addr < self.memory.len() {
                self.memory[addr]
            } else {
                0
            };
            let (r, g, b) = CGA_PALETTE[(byte as usize) & 0x0F];
            out.push(r);
            out.push(g);
            out.push(b);
        }
        out
    }

    /// Read a graphics-mode framebuffer as raw RGBA bytes for canvas blit.
    ///
    /// Returns `width * height * 4` bytes (R, G, B, A per pixel) suitable
    /// for `new ImageData(new Uint8ClampedArray(bytes), width, height)`
    /// in the browser. Alpha is always 255.
    ///
    /// This is the same pixel data as [`render_framebuffer`] but without
    /// the PPM header and with an alpha channel, so the browser can stuff
    /// it straight into a canvas without reformatting.
    pub fn read_framebuffer_rgba(
        &self,
        base_addr: usize,
        width: usize,
        height: usize,
    ) -> Vec<u8> {
        // Snapshot the VGA DAC palette once per frame. CSS-DOS shadows OUT
        // 0x3C9 writes to 768 bytes starting at linear 0x100000, stored as
        // 6-bit values (0..63) just like real VGA hardware. If the cabinet
        // doesn't declare a DAC zone (old builds, hack carts without gfx
        // pruning), every byte comes back 0 (all-black palette); we fall
        // back to the hardcoded CGA palette in that case so existing
        // cabinets don't regress.
        let mut dac = [0u8; 768];
        let mut dac_populated = false;
        for i in 0..768 {
            let v = self
                .extended
                .get(&(VGA_DAC_LINEAR + i as i32))
                .copied()
                .unwrap_or(0);
            if v != 0 {
                dac_populated = true;
            }
            dac[i] = (v & 0x3F) as u8; // truncate to 6 bits, hardware does this
        }

        let mut out = vec![0u8; width * height * 4];
        for i in 0..(width * height) {
            let addr = base_addr + i;
            let byte = if addr < self.memory.len() {
                self.memory[addr]
            } else {
                0
            };
            let (r, g, b) = if dac_populated {
                // Expand 6-bit DAC value (0..63) to 8-bit (0..255) the way
                // real VGA hardware did: replicate the top 2 bits into the
                // bottom 2. Keeps 0→0 and 63→255 without a divide.
                let idx = byte as usize * 3;
                let r6 = dac[idx];
                let g6 = dac[idx + 1];
                let b6 = dac[idx + 2];
                (
                    (r6 << 2) | (r6 >> 4),
                    (g6 << 2) | (g6 >> 4),
                    (b6 << 2) | (b6 >> 4),
                )
            } else {
                CGA_PALETTE[(byte as usize) & 0x0F]
            };
            let o = i * 4;
            out[o] = r;
            out[o + 1] = g;
            out[o + 2] = b;
            out[o + 3] = 255;
        }
        out
    }

    /// Read raw video memory bytes (character bytes only, no attributes).
    ///
    /// Returns `width * height` bytes from text-mode video memory.
    /// Useful for WASM/browser rendering where JS handles display.
    pub fn read_video_memory(&self, base_addr: usize, width: usize, height: usize) -> Vec<u8> {
        let mut buf = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let addr = base_addr + (y * width + x) * 2;
                buf[y * width + x] = if addr < self.memory.len() {
                    self.memory[addr]
                } else {
                    0
                };
            }
        }
        buf
    }

    /// Discover state variables and load initial values from `@property` declarations.
    ///
    /// Any `@property --X` where X is not `m{digits}` and not `clock` is a state
    /// variable. State variables get auto-assigned negative addresses: the first
    /// one gets -1, the second -2, etc. Memory bytes (`--m{N}`) are stored in
    /// the flat byte array at address N.
    ///
    /// This method populates the state var map AND installs the address map into
    /// the eval module's thread-local so that `property_to_address()` works for
    /// all subsequent operations.
    pub fn load_properties(&mut self, properties: &[crate::types::PropertyDef]) {
        use crate::types::CssValue;

        // Classify properties: state vars vs memory vs other
        let mut state_var_defs: Vec<(&str, i32)> = Vec::new(); // (bare_name, initial_value)
        let mut memory_defs: Vec<(usize, u8)> = Vec::new();    // (addr, value)
        let mut max_mem_addr: usize = 0;

        for prop in properties {
            let value = match &prop.initial_value {
                Some(CssValue::Integer(v)) => *v as i32,
                _ => 0,
            };

            // Strip leading "--"
            let bare = if prop.name.starts_with("--") {
                &prop.name[2..]
            } else {
                &prop.name
            };

            // Skip clock — it's animation plumbing, not state
            if bare == "clock" {
                continue;
            }

            // Check if it's a memory byte: m{digits}
            if let Some(rest) = bare.strip_prefix('m') {
                if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
                    if let Ok(addr) = rest.parse::<usize>() {
                        if addr < 0xF0000 {
                            max_mem_addr = max_mem_addr.max(addr + 1);
                        }
                        memory_defs.push((addr, (value & 0xFF) as u8));
                        continue;
                    }
                }
            }

            // It's a state variable
            state_var_defs.push((bare, value));
        }

        // Set up state vars
        self.state_vars.clear();
        self.state_var_names.clear();
        self.state_var_index.clear();

        let mut address_map = std::collections::HashMap::new();
        for (i, &(name, init)) in state_var_defs.iter().enumerate() {
            let addr = -(i as i32 + 1);
            self.state_vars.push(init);
            self.state_var_names.push(name.to_string());
            self.state_var_index.insert(name.to_string(), i);
            address_map.insert(name.to_string(), addr);
        }

        log::info!(
            "Discovered {} state variables from CSS: {:?}",
            self.state_vars.len(),
            self.state_var_names,
        );

        // Install the state var address map so property_to_address() works.
        // This REPLACES any existing map — call load_properties BEFORE
        // Evaluator::from_parsed() so that dispatch-table entries are merged
        // on top of state var entries (not the other way around).
        super::eval::set_address_map(address_map);

        // Set up memory
        if max_mem_addr > self.memory.len() {
            log::info!("Auto-sizing memory: {} → {} bytes", self.memory.len(), max_mem_addr);
            self.memory.resize(max_mem_addr, 0);
        }
        for (addr, value) in &memory_defs {
            if *addr >= 0xF0000 {
                self.extended.insert(*addr as i32, *value as i32);
            } else if *addr < self.memory.len() {
                self.memory[*addr] = *value;
            }
        }
    }

    /// Populate memory from a `--readMem`-shaped dispatch table's literal entries.
    ///
    /// CSS-DOS emits BIOS ROM bytes as literal branches inside the
    /// `--readMem` function (`style(--at: 0xF0055): 1;` etc.) rather than as
    /// `--mN` @property declarations. Those literals live in the evaluator's
    /// dispatch table, not in `load_properties`'s memory_defs, so without this
    /// pass `read_mem()` returns 0 for every BIOS address — the CPU reads the
    /// right bytes via the CSS function, but the debugger / tooling that calls
    /// `read_mem` sees an empty ROM.
    ///
    /// Only `Literal` entries are copied. Identity entries (`Var("--mN")`) are
    /// already covered by `load_properties`, and non-literal/non-identity
    /// entries (like `--readDiskByte(...)` for the rom-disk window) are
    /// ignored — reading those requires the evaluator.
    pub fn populate_memory_from_readmem(&mut self, table: &crate::pattern::dispatch_table::DispatchTable) {
        use crate::types::Expr;
        for (&key, expr) in &table.entries {
            let Expr::Literal(val) = expr else { continue };
            let addr = key as i32;
            if addr < 0 {
                continue;
            }
            let addr_u = addr as usize;
            let val_i32 = *val as i32;
            if addr_u >= 0xF0000 {
                self.extended.insert(addr, val_i32);
            } else if addr_u < self.memory.len() {
                self.memory[addr_u] = (val_i32 & 0xFF) as u8;
            }
        }
    }

    /// Serialise the runtime-mutable parts of the state to a byte blob.
    ///
    /// What goes in: `state_vars`, `memory`, `extended`, `string_properties`,
    /// `frame_counter` — everything the tick loop touches.
    ///
    /// What stays out: `state_var_names`/`state_var_index` (set by
    /// `load_properties` from the parsed cabinet), `packed_cell_table` /
    /// `packed_cell_size` / `windowed_byte_array` (wired at compile time from
    /// `CompiledProgram`). A snapshot is therefore portable across runs of
    /// the **same cabinet**: load the cabinet, restore the snapshot, the
    /// machine resumes mid-execution. It is NOT portable across cabinets;
    /// slot ordering depends on parse order.
    ///
    /// Format (little-endian throughout):
    ///   "CSNP"                     4 bytes — magic
    ///   u32 version = 1
    ///   u32 state_vars_len, then n × i32     state_vars
    ///   u32 memory_len,    then n × u8       memory
    ///   u32 extended_count, then n × (i32 addr, i32 val)
    ///   u32 string_count, then n × (u32 name_len, name_bytes,
    ///                               u32 val_len,  val_bytes)
    ///   u32 frame_counter
    pub fn snapshot(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            16 + self.state_vars.len() * 4 + self.memory.len() + self.extended.len() * 8,
        );
        out.extend_from_slice(b"CSNP");
        out.extend_from_slice(&1u32.to_le_bytes());

        out.extend_from_slice(&(self.state_vars.len() as u32).to_le_bytes());
        for &v in &self.state_vars {
            out.extend_from_slice(&v.to_le_bytes());
        }

        out.extend_from_slice(&(self.memory.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.memory);

        out.extend_from_slice(&(self.extended.len() as u32).to_le_bytes());
        for (&addr, &val) in &self.extended {
            out.extend_from_slice(&addr.to_le_bytes());
            out.extend_from_slice(&val.to_le_bytes());
        }

        out.extend_from_slice(&(self.string_properties.len() as u32).to_le_bytes());
        for (name, val) in &self.string_properties {
            let nb = name.as_bytes();
            let vb = val.as_bytes();
            out.extend_from_slice(&(nb.len() as u32).to_le_bytes());
            out.extend_from_slice(nb);
            out.extend_from_slice(&(vb.len() as u32).to_le_bytes());
            out.extend_from_slice(vb);
        }

        out.extend_from_slice(&self.frame_counter.to_le_bytes());
        out
    }

    /// Inverse of [`snapshot`](Self::snapshot). Replaces the runtime-mutable
    /// fields in `self`. The compile-time fields (state_var_names, packed
    /// cell table, disk window) are left alone — caller must be operating
    /// against the same cabinet that produced the snapshot.
    ///
    /// Returns `Err(&str)` on bad magic, unsupported version, truncation,
    /// invalid utf-8, or a state-vars/memory length mismatch with `self`.
    pub fn restore(&mut self, blob: &[u8]) -> Result<(), &'static str> {
        let mut p = 0usize;
        fn take<'a>(blob: &'a [u8], p: &mut usize, n: usize) -> Result<&'a [u8], &'static str> {
            if *p + n > blob.len() {
                return Err("snapshot truncated");
            }
            let s = &blob[*p..*p + n];
            *p += n;
            Ok(s)
        }
        fn read_u32(blob: &[u8], p: &mut usize) -> Result<u32, &'static str> {
            let s = take(blob, p, 4)?;
            Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        }
        fn read_i32(blob: &[u8], p: &mut usize) -> Result<i32, &'static str> {
            let s = take(blob, p, 4)?;
            Ok(i32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        }

        let magic = take(blob, &mut p, 4)?;
        if magic != b"CSNP" {
            return Err("snapshot: bad magic (expected CSNP)");
        }
        let version = read_u32(blob, &mut p)?;
        if version != 1 {
            return Err("snapshot: unsupported version");
        }

        let svlen = read_u32(blob, &mut p)? as usize;
        if svlen != self.state_vars.len() {
            return Err("snapshot: state_vars length mismatch (different cabinet?)");
        }
        for slot in 0..svlen {
            self.state_vars[slot] = read_i32(blob, &mut p)?;
        }

        let mlen = read_u32(blob, &mut p)? as usize;
        if mlen != self.memory.len() {
            return Err("snapshot: memory length mismatch (different cabinet?)");
        }
        let mbytes = take(blob, &mut p, mlen)?;
        self.memory.copy_from_slice(mbytes);

        let ext_count = read_u32(blob, &mut p)? as usize;
        self.extended.clear();
        for _ in 0..ext_count {
            let addr = read_i32(blob, &mut p)?;
            let val = read_i32(blob, &mut p)?;
            self.extended.insert(addr, val);
        }

        let str_count = read_u32(blob, &mut p)? as usize;
        self.string_properties.clear();
        for _ in 0..str_count {
            let nlen = read_u32(blob, &mut p)? as usize;
            let nb = take(blob, &mut p, nlen)?.to_vec();
            let vlen = read_u32(blob, &mut p)? as usize;
            let vb = take(blob, &mut p, vlen)?.to_vec();
            let name = String::from_utf8(nb).map_err(|_| "snapshot: bad utf-8 in name")?;
            let val = String::from_utf8(vb).map_err(|_| "snapshot: bad utf-8 in value")?;
            self.string_properties.insert(name, val);
        }

        self.frame_counter = read_u32(blob, &mut p)?;
        Ok(())
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new(DEFAULT_MEM_SIZE)
    }
}

/// Map a CP437 byte to a Unicode character for terminal rendering.
fn cp437_to_unicode(b: u8) -> char {
    // CP437 low control range (0x00-0x1F) and 0x7F mapped to common glyphs
    const LOW: [char; 32] = [
        ' ', '\u{263A}', '\u{263B}', '\u{2665}', '\u{2666}', '\u{2663}', '\u{2660}', '\u{2022}',
        '\u{25D8}', '\u{25CB}', '\u{25D9}', '\u{2642}', '\u{2640}', '\u{266A}', '\u{266B}', '\u{263C}',
        '\u{25BA}', '\u{25C4}', '\u{2195}', '\u{203C}', '\u{00B6}', '\u{00A7}', '\u{25AC}', '\u{21A8}',
        '\u{2191}', '\u{2193}', '\u{2192}', '\u{2190}', '\u{221F}', '\u{2194}', '\u{25B2}', '\u{25BC}',
    ];
    // CP437 high range (0x80-0xFF)
    const HIGH: [char; 128] = [
        '\u{00C7}', '\u{00FC}', '\u{00E9}', '\u{00E2}', '\u{00E4}', '\u{00E0}', '\u{00E5}', '\u{00E7}',
        '\u{00EA}', '\u{00EB}', '\u{00E8}', '\u{00EF}', '\u{00EE}', '\u{00EC}', '\u{00C4}', '\u{00C5}',
        '\u{00C9}', '\u{00E6}', '\u{00C6}', '\u{00F4}', '\u{00F6}', '\u{00F2}', '\u{00FB}', '\u{00F9}',
        '\u{00FF}', '\u{00D6}', '\u{00DC}', '\u{00A2}', '\u{00A3}', '\u{00A5}', '\u{20A7}', '\u{0192}',
        '\u{00E1}', '\u{00ED}', '\u{00F3}', '\u{00FA}', '\u{00F1}', '\u{00D1}', '\u{00AA}', '\u{00BA}',
        '\u{00BF}', '\u{2310}', '\u{00AC}', '\u{00BD}', '\u{00BC}', '\u{00A1}', '\u{00AB}', '\u{00BB}',
        '\u{2591}', '\u{2592}', '\u{2593}', '\u{2502}', '\u{2524}', '\u{2561}', '\u{2562}', '\u{2556}',
        '\u{2555}', '\u{2563}', '\u{2551}', '\u{2557}', '\u{255D}', '\u{255C}', '\u{255B}', '\u{2510}',
        '\u{2514}', '\u{2534}', '\u{252C}', '\u{251C}', '\u{2500}', '\u{253C}', '\u{255E}', '\u{255F}',
        '\u{255A}', '\u{2554}', '\u{2569}', '\u{2566}', '\u{2560}', '\u{2550}', '\u{256C}', '\u{2567}',
        '\u{2568}', '\u{2564}', '\u{2565}', '\u{2559}', '\u{2558}', '\u{2552}', '\u{2553}', '\u{256B}',
        '\u{256A}', '\u{2518}', '\u{250C}', '\u{2588}', '\u{2584}', '\u{258C}', '\u{2590}', '\u{2580}',
        '\u{03B1}', '\u{00DF}', '\u{0393}', '\u{03C0}', '\u{03A3}', '\u{03C3}', '\u{00B5}', '\u{03C4}',
        '\u{03A6}', '\u{0398}', '\u{03A9}', '\u{03B4}', '\u{221E}', '\u{03C6}', '\u{03B5}', '\u{2229}',
        '\u{2261}', '\u{00B1}', '\u{2265}', '\u{2264}', '\u{2320}', '\u{2321}', '\u{00F7}', '\u{2248}',
        '\u{00B0}', '\u{2219}', '\u{00B7}', '\u{221A}', '\u{207F}', '\u{00B2}', '\u{25A0}', ' ',
    ];
    match b {
        0x00..=0x1F => LOW[b as usize],
        0x20..=0x7E => b as char,
        0x7F => '\u{2302}',
        _ => HIGH[(b - 0x80) as usize],
    }
}

/// A per-tick change record over [`State`]. Stores both the pre- and post-tick
/// value for every mutated channel so deltas can be replayed forward OR
/// backward without re-executing the tick.
///
/// Channels:
/// - `state_vars`: (slot_index, old, new) — state_var_names/state_var_index
///   are program-static and not recorded.
/// - `memory`: (byte_addr, old, new) — only for byte memory. Extended/string
///   channels handled separately.
/// - `extended`: (addr, old, new) where `None` means "entry absent".
/// - `string_properties`: (name, old, new) where `None` means "entry absent".
///
/// Empty-delta ticks are still produced (frame_counter always changes).
#[derive(Debug, Clone, Default)]
pub struct StateDelta {
    /// Tick at the START of this tick (what frame_counter equalled before tick).
    pub frame_counter_before: u32,
    /// Tick at the END of this tick (frame_counter_before + 1 in practice).
    pub frame_counter_after: u32,
    pub state_vars: Vec<(u32, i32, i32)>,
    pub memory: Vec<(u32, u8, u8)>,
    pub extended: Vec<(i32, Option<i32>, Option<i32>)>,
    pub string_properties: Vec<(String, Option<String>, Option<String>)>,
}

impl StateDelta {
    pub fn total_changes(&self) -> usize {
        self.state_vars.len()
            + self.memory.len()
            + self.extended.len()
            + self.string_properties.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a state with some state variables for testing.
    fn test_state() -> State {
        let mut state = State::new(DEFAULT_MEM_SIZE);
        // Simulate discovering state vars from CSS
        let names = vec!["AX", "CX", "DX", "BX"];
        for (i, name) in names.iter().enumerate() {
            state.state_vars.push(0);
            state.state_var_names.push(name.to_string());
            state.state_var_index.insert(name.to_string(), i);
        }
        state
    }

    #[test]
    fn state_var_addressing() {
        let mut state = test_state();
        // AX is slot 0 → address -1
        state.state_vars[0] = 0x1234;
        assert_eq!(state.read_mem(-1), 0x1234);

        state.write_mem(-1, 0xABCD);
        assert_eq!(state.state_vars[0], 0xABCD);
    }

    #[test]
    fn state_var_by_name() {
        let mut state = test_state();
        state.set_var("AX", 42);
        assert_eq!(state.get_var("AX"), Some(42));
        assert_eq!(state.get_var("nonexistent"), None);
    }

    #[test]
    fn memory_addressing() {
        let mut state = State::default();
        state.write_mem(0x100, 0x42);
        assert_eq!(state.read_mem(0x100), 0x42);
    }

    #[test]
    fn read16_little_endian() {
        let mut state = State::default();
        state.write_mem(0x10, 0x34); // lo
        state.write_mem(0x11, 0x12); // hi
        assert_eq!(state.read_mem16(0x10), 0x1234);
    }

    #[test]
    fn byte_extraction() {
        assert_eq!(State::lo8(0x1234), 0x34);
        assert_eq!(State::hi8(0x1234), 0x12);
    }

    #[test]
    fn out_of_range_state_var() {
        let state = test_state();
        // Address -100 is beyond the 4 state vars
        assert_eq!(state.read_mem(-100), 0);
    }

    #[test]
    fn state_var_stores_full_value() {
        let mut state = test_state();
        // No masking — state vars store whatever the CSS computes
        state.write_mem(-1, 0x1_ABCD);
        assert_eq!(state.state_vars[0], 0x1_ABCD);

        state.write_mem(-2, -2);
        assert_eq!(state.state_vars[1], -2);
    }

    /// REP MOVSB with forward-overlap is the LZ77 self-reference pattern
    /// that UPX (and any LZ-style decompressor) depends on. Each iteration
    /// reads then writes one byte, so when SI catches up to the bytes DI
    /// just wrote, those new bytes get re-copied — repeating the run.
    /// `bulk_copy_bytes` must mirror this; a `memmove`-style snapshot
    /// would silently break LZ77 and any UPX-packed program would produce
    /// garbage on decompression. Concrete consequence we hit in the wild:
    /// UPX-packed COMMAND.COM exiting via INT 20h instead of running its
    /// shell loop, kernel reporting "Bad or missing command interpreter".
    #[test]
    fn bulk_copy_bytes_lz77_repeat() {
        let mut state = State::default();
        state.write_mem(0x100, b'A' as i32);
        state.write_mem(0x101, b'B' as i32);
        state.write_mem(0x102, b'C' as i32);
        // Copy 6 bytes from 0x100 to 0x103 — LZ-style "repeat last 3".
        // Per-iteration semantics: write A→0x103, B→0x104, C→0x105,
        // then read 0x103 (now 'A') → write 0x106, etc.
        state.bulk_copy_bytes(0x100, 0x103, 6);
        assert_eq!(state.read_mem(0x103), b'A' as i32);
        assert_eq!(state.read_mem(0x104), b'B' as i32);
        assert_eq!(state.read_mem(0x105), b'C' as i32);
        assert_eq!(state.read_mem(0x106), b'A' as i32);
        assert_eq!(state.read_mem(0x107), b'B' as i32);
        assert_eq!(state.read_mem(0x108), b'C' as i32);
    }

    /// Sister case: non-overlapping copy must still work — and copying
    /// with `src > dst` (no overlap into source) must not be perturbed
    /// by the per-byte loop.
    #[test]
    fn bulk_copy_bytes_disjoint() {
        let mut state = State::default();
        for i in 0..4 { state.write_mem(0x200 + i, (i + 1) as i32); }
        state.bulk_copy_bytes(0x200, 0x300, 4);
        for i in 0..4 {
            assert_eq!(state.read_mem(0x300 + i), (i + 1) as i32);
            // Source unchanged.
            assert_eq!(state.read_mem(0x200 + i), (i + 1) as i32);
        }
    }
}
