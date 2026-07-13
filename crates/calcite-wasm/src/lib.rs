//! WASM bindings for calc(ite).
//!
//! This crate compiles to a WASM module that runs inside a Web Worker.
//! The main thread sends CSS text, the worker parses/compiles/evaluates,
//! and sends back property diffs for DOM application.

use wasm_bindgen::prelude::*;

use calcite_core::script::MeasurementEvent;
use calcite_core::script_spec::parse_watch;

/// Initialise the WASM module (sets up logging, etc.).
#[wasm_bindgen(start)]
pub fn init() {
    std::panic::set_hook(Box::new(|info| {
        let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        };
        let loc = info.location().map(|l| format!(" at {}:{}", l.file(), l.line())).unwrap_or_default();
        web_sys::console::error_1(&format!("WASM panic: {msg}{loc}").into());
    }));
    console_log::init_with_level(log::Level::Info).ok();
    log::info!("calc(ite) WASM module initialised");
}

/// The main engine handle exposed to JavaScript.
#[wasm_bindgen]
pub struct CalciteEngine {
    state: calcite_core::State,
    evaluator: calcite_core::Evaluator,
    // Initial property defs, cached at construction. reset() rebuilds
    // `state` from this without rebuilding `evaluator` (which would
    // require reparsing + recompiling the CSS — expensive for large
    // cabinets).
    initial_properties: Vec<calcite_core::types::PropertyDef>,
    /// Optional script-primitive registry the host configures via
    /// `register_watch`. When non-empty, `tick_batch` / `run_batch_silent`
    /// will additionally poll the registry at chunk boundaries (every
    /// `WATCH_CHUNK_TICKS` ticks) and the host drains events via
    /// `drain_measurements`.
    watch_registry: calcite_core::script::WatchRegistry,
}

#[wasm_bindgen]
impl CalciteEngine {
    /// Create a new engine instance from CSS source as raw UTF-8 bytes.
    /// Use this for large files that exceed JS string limits.
    pub fn new_from_bytes(css_bytes: &[u8]) -> Result<CalciteEngine, JsError> {
        let css = std::str::from_utf8(css_bytes)
            .map_err(|e| JsError::new(&format!("Invalid UTF-8: {e}")))?;
        Self::new(css)
    }

    /// Create a new engine instance from CSS source text.
    #[wasm_bindgen(constructor)]
    pub fn new(css: &str) -> Result<CalciteEngine, JsError> {
        log::info!("Parsing {} bytes of CSS", css.len());

        let parsed =
            calcite_core::parser::parse_css(css).map_err(|e| JsError::new(&e.to_string()))?;

        log::info!(
            "Parsed: {} @property, {} @function, {} assignments",
            parsed.properties.len(),
            parsed.functions.len(),
            parsed.assignments.len(),
        );

        log::info!("Loading properties...");
        let mut state = calcite_core::State::default();
        state.load_properties(&parsed.properties);
        log::info!("Properties loaded, memory size: {} bytes", state.memory.len());

        log::info!("Creating evaluator...");
        // Owned form: moves the parsed ASTs into the evaluator instead of
        // cloning them (the borrow form deep-copies the whole program —
        // seconds of wasm time on big cabinets). Returns the @property
        // defs, which the engine keeps for reset().
        let (evaluator, initial_properties) =
            calcite_core::Evaluator::from_parsed_owned(parsed);
        log::info!("Evaluator created");

        // Copy literal BIOS-byte entries out of the --readMem dispatch table
        // into state memory so read_mem() / debugger tooling can see ROM. Without
        // this, BIOS bytes exist only as CSS literal branches and read_mem()
        // returns 0 for every F0000+ address.
        if let Some(table) = evaluator.dispatch_tables.get("--readMem") {
            state.populate_memory_from_readmem(table);
        }

        // Wire the packed-cell table into State so positive-addr reads
        // (video mode byte at 0x449, framebuffer at 0xA0000, text VRAM at
        // 0xB8000, stack at top-of-RAM) transparently consult the `--mc{N}`
        // state vars the packed-broadcast writer populates. Without this,
        // every read sees an all-zero `state.memory[]` because the packed
        // path bypasses the shadow. The helper picks the best table
        // available (read-side `packed_cell_tables[0]`, or the merged
        // write-port tables as a fallback).
        evaluator.wire_state_for_packed_memory(&mut state);
        // Install the windowed-byte-array descriptor so reads inside the
        // window collapse to one state-var read + one flat-array index
        // instead of walking the inline-exception chain and re-evaluating
        // the inner flat-array function on every byte. The byte data comes
        // from the cabinet's compiled flat-array dispatch — no separate
        // sidecar load is required.
        evaluator.wire_state_for_windowed_byte_array(&mut state);
        // Install input-edge groups on State. After this,
        // `state.set_pseudo_class_active(...)` writes the gated state-var
        // slot(s) directly — no per-tick apply path is needed.
        evaluator.wire_state_for_input_edges(&mut state);

        // Wasm builds use the in-memory dump sink because the SW /
        // worker has no filesystem access; payloads ride out on
        // MeasurementEvents that the host drains.
        let mut watch_registry = calcite_core::script::WatchRegistry::new();
        watch_registry.set_dump_sink(calcite_core::script::DumpSink::Memory);
        Ok(CalciteEngine { state, evaluator, initial_properties, watch_registry })
    }

    /// Diagnostic: per-phase compile timing as a JSON array
    /// (`[{"phase":"parse.cssparser","secs":12.3}, …]`). Recorded during
    /// the constructor's parse+compile; wasm has no stderr so this is the
    /// only way hosts can see the breakdown.
    pub fn compile_phase_report(&self) -> String {
        calcite_core::compile_stats::report_json()
    }

    /// Diagnostic: number of recognised packed-broadcast ports.
    /// 0 = recogniser didn't fire (cabinet falls back to slow per-cell eval).
    /// 3 = current 3-slot scheme; 6 = legacy byte-slot scheme.
    pub fn packed_broadcast_port_count(&self) -> u32 {
        self.evaluator.compiled().packed_broadcast_writes.len() as u32
    }

    /// Diagnostic: total compiled-program op count.
    pub fn compiled_op_count(&self) -> u32 {
        self.evaluator.compiled().ops.len() as u32
    }

    /// Diagnostic: total compacted slot count.
    pub fn compiled_slot_count(&self) -> u32 {
        self.evaluator.compiled().slot_count
    }

    /// Reset the engine's runtime state without recompiling the CSS.
    /// Equivalent to `new CalciteEngine(css)` but skips the parse +
    /// compile steps, which are the expensive ones for large cabinets.
    /// Used by the bridge worker to restart the machine on each
    /// viewer-connect without paying multi-second compile cost.
    pub fn reset(&mut self) {
        self.state = calcite_core::State::default();
        self.state.load_properties(&self.initial_properties);
        if let Some(table) = self.evaluator.dispatch_tables.get("--readMem") {
            self.state.populate_memory_from_readmem(table);
        }
        self.evaluator.wire_state_for_packed_memory(&mut self.state);
        self.evaluator.wire_state_for_windowed_byte_array(&mut self.state);
        self.evaluator.wire_state_for_input_edges(&mut self.state);
        // Watch registry: drop every registered watch and any pending
        // events. Hosts re-register after reset() if they want them.
        let mut new_reg = calcite_core::script::WatchRegistry::new();
        new_reg.set_dump_sink(calcite_core::script::DumpSink::Memory);
        self.watch_registry = new_reg;
    }

    /// Run a batch of ticks and return the property changes as a JSON string.
    ///
    /// Returns `[[name, value], ...]` pairs.
    pub fn tick_batch(&mut self, count: u32) -> Result<String, JsError> {
        let result = self.evaluator.run_batch(&mut self.state, count);

        // Serialize changes as JSON array of [name, value] pairs
        let json_parts: Vec<String> = result
            .changes
            .iter()
            .map(|(name, value)| format!("[\"{name}\",\"{value}\"]"))
            .collect();
        Ok(format!("[{}]", json_parts.join(",")))
    }

    /// Run a batch of ticks WITHOUT computing the per-batch state-var diff.
    ///
    /// `tick_batch` returns the changed state vars as a JSON string for
    /// callers (debug-server, conformance harness) that consume them.
    /// The web bridge ignores the returned JSON — it observes state via
    /// `get_state_var` / `read_framebuffer_rgba` after each batch instead.
    /// On large cabinets the diff is non-trivial (doom8088: ~10K state
    /// vars, full O(N) sweep each batch), and the JSON formatting on top
    /// of that adds another ~5-10% per batch.
    ///
    /// This entry point skips both. Same correctness contract as
    /// `tick_batch` for state advancement; just doesn't synthesize the
    /// JSON return.
    pub fn run_batch_silent(&mut self, count: u32) {
        self.evaluator.run_batch_silent(&mut self.state, count);
    }

    /// Serialise the runtime-mutable state (state_vars + memory + extended
    /// + string_properties + frame_counter) to a byte blob. Pair with
    /// [`restore`](Self::restore) on the same cabinet to resume execution
    /// from this exact tick — useful for benchmarking a level-load window
    /// without re-running boot/title/menu first.
    ///
    /// The blob is portable across calcite-cli and calcite-wasm and across
    /// runs of the same cabinet, but NOT across cabinets (slot ordering is
    /// per-parse).
    pub fn snapshot(&self) -> Vec<u8> {
        self.state.snapshot()
    }

    /// Restore a snapshot blob produced by [`snapshot`](Self::snapshot)
    /// against the same cabinet. Returns an error if the blob is malformed
    /// or the cabinet's state shape doesn't match the snapshot.
    pub fn restore(&mut self, blob: &[u8]) -> Result<(), JsError> {
        self.state.restore(blob).map_err(JsError::new)
    }

    /// Report a pseudo-class match edge as active or inactive.
    ///
    /// This is the principled input surface: the cabinet's CSS uses
    /// `&:has(#SELECTOR:PSEUDO) { --PROP: VALUE; }` rules to bind the
    /// match state to a custom property. The host calls this to flip
    /// the (pseudo, selector) edge; the next tick's evaluation
    /// produces VALUE on PROP via calcite's input-edge recogniser.
    ///
    /// `pseudo` is the pseudo-class name without the leading `:`
    /// (e.g. `"active"`). `selector` is the id-selector text without
    /// the leading `#` (e.g. `"kb-1"`). `value` is the boolean state.
    ///
    /// The host is responsible for sending matching false edges
    /// (release) — calcite does not synthesise key-up automatically.
    pub fn set_pseudo_class_active(&mut self, pseudo: &str, selector: &str, value: bool) {
        self.state.set_pseudo_class_active(pseudo, selector, value);
    }

    /// Returns the input edges the recogniser found in this cabinet's
    /// CSS, as a JS array of `{property, pseudo, selector, value}`
    /// objects. Useful for the host to know which edges to drive.
    pub fn input_edges(&self) -> JsValue {
        let edges = self.evaluator.input_edges_for_host();
        let arr = js_sys::Array::new();
        for (property, pseudo, selector, value) in edges {
            let obj = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&obj, &"property".into(), &property.into());
            let _ = js_sys::Reflect::set(&obj, &"pseudo".into(), &pseudo.into());
            let _ = js_sys::Reflect::set(&obj, &"selector".into(), &selector.into());
            let _ = js_sys::Reflect::set(&obj, &"value".into(), &value.into());
            arr.push(&obj);
        }
        arr.into()
    }

    /// Register a watch (script-primitive) with the engine. The spec
    /// uses the same string format as calcite-cli's `--watch` flag —
    /// see `crates/calcite-core/src/script.rs` for the full grammar.
    /// Examples (`spec`):
    ///
    /// * `"poll:stride:every=50000"` — a cheap stride that ticks every
    ///   50K engine ticks. Suitable as a `gate=` target for expensive
    ///   watches.
    /// * `"ingame:cond:0x3a3c4=0:gate=poll:then=emit+halt"` — fire +
    ///   halt when the byte at linear 0x3a3c4 is zero, evaluated only
    ///   on `poll` fires.
    /// * `"vram_dump:at:tick=2000000:then=dump=0xb8000,4000"` — capture
    ///   text VRAM at tick 2M; the bytes attach to the emitted event.
    ///
    /// Returns the watch's index on success. Throws on parse error.
    pub fn register_watch(&mut self, spec: &str) -> Result<u32, JsError> {
        let parsed = parse_watch(spec).map_err(|e| JsError::new(&e))?;
        Ok(self.watch_registry.register(parsed) as u32)
    }

    /// Number of registered watches.
    pub fn watch_count(&self) -> u32 {
        self.watch_registry.len() as u32
    }

    /// Discard all registered watches (and pending events) without
    /// touching the engine state. Call this between bench profile
    /// stages to start fresh; `reset()` does this implicitly along
    /// with rebuilding state.
    pub fn clear_watches(&mut self) {
        let mut new_reg = calcite_core::script::WatchRegistry::new();
        new_reg.set_dump_sink(calcite_core::script::DumpSink::Memory);
        self.watch_registry = new_reg;
    }

    /// Run `count` ticks while polling the watch registry every
    /// `chunk_ticks` ticks. Returns true if a watch requested halt
    /// before `count` ticks elapsed; false otherwise. The host drains
    /// events via `drain_measurements` after this returns.
    ///
    /// Use `chunk_ticks = 0` to disable chunking (single big batch with
    /// one final poll).
    pub fn run_batch_watched(&mut self, count: u32, chunk_ticks: u32) -> bool {
        if self.watch_registry.is_empty() {
            // No watches: degenerate to a plain run_batch.
            self.evaluator.run_batch_silent(&mut self.state, count);
            return false;
        }
        let chunk = if chunk_ticks == 0 { count.max(1) } else { chunk_ticks };
        // `At` watches fire on exact tick equality — a poll grid that
        // steps over the tick never fires them. Shrink the batch so a
        // poll lands exactly on each target, the same way calcite-cli's
        // watch loop does (cli/wasm parity).
        let at_targets = self.watch_registry.at_targets();
        let mut remaining = count;
        let mut tick_now: u32 = self.state.frame_counter;
        while remaining > 0 {
            let mut n = remaining.min(chunk);
            if let Some(&t) = at_targets.iter().find(|&&t| t > tick_now) {
                n = n.min(t - tick_now);
            }
            self.evaluator.run_batch_silent(&mut self.state, n);
            tick_now = tick_now.saturating_add(n);
            calcite_core::script_eval::poll(
                &mut self.watch_registry,
                &mut self.state,
                tick_now,
            );
            if self.watch_registry.halt_requested() {
                return true;
            }
            remaining = remaining.saturating_sub(n);
        }
        false
    }

    /// Drain pending measurement events, returning them as a JSON
    /// string. Each event is a JSON object with shape
    /// `{tick, watch, halted, vars:[[name,val]...], dumps:[{tag,addr,len,bytes}]}`.
    /// `bytes` is a base64 string when the dump sink is Memory.
    pub fn drain_measurements(&mut self) -> String {
        let events = self.watch_registry.drain_events();
        let parts: Vec<String> = events.iter().map(measurement_to_json).collect();
        format!("[{}]", parts.join(","))
    }

    /// Has any watch requested a halt during the most recent poll?
    pub fn watch_halt_requested(&self) -> bool {
        self.watch_registry.halt_requested()
    }

    /// Get the current value of a register (for debugging).
    pub fn get_register(&self, index: usize) -> i32 {
        if index < self.state.state_vars.len() {
            self.state.state_vars[index]
        } else {
            0
        }
    }

    /// Get a state variable by name (e.g. "cycleCount", "IP", "AX").
    /// Returns 0 if the variable doesn't exist.
    pub fn get_state_var(&self, name: &str) -> i32 {
        self.state.get_var(name).unwrap_or(0)
    }

    /// Number of ticks evaluated since reset. Equivalent to the
    /// `tick` count `calcite-cli` prints. Used by bench harnesses that
    /// want a stable "calcite work units done" metric independent of
    /// guest CPU frequency.
    pub fn get_tick(&self) -> u32 {
        self.state.frame_counter
    }

    /// Read text-mode video memory (character bytes only).
    ///
    /// Returns `width * height` bytes from video memory at `base_addr`.
    /// Default for DOS text mode: `read_video_memory(0xB8000, 40, 25)`.
    pub fn read_video_memory(&self, base_addr: usize, width: usize, height: usize) -> Vec<u8> {
        // Packed cabinets need unified reads; unpacked passes through.
        if self.state.packed_cell_table.is_empty() {
            return self.state.read_video_memory(base_addr, width, height);
        }
        let n = width * height;
        let mut buf = vec![0u8; n];
        for i in 0..n {
            buf[i] = self.read_byte_unified((base_addr + 2 * i) as i32); // char byte only, skip attribute
        }
        buf
    }

    /// Read a contiguous byte range from memory. Returns `len` bytes starting
    /// at `base_addr`. Out-of-range reads return 0.
    ///
    /// Used by the browser renderer when it needs the raw VGA/CGA framebuffer
    /// bytes (char+attr pairs for text mode, 2-bpp packed scanlines for CGA
    /// mode 0x04). Calcite stays x86-ignorant: it just hands over the bytes,
    /// the caller decodes.
    pub fn read_memory_range(&self, base_addr: usize, len: usize) -> Vec<u8> {
        let mut buf = vec![0u8; len];
        self.fill_range_unified(&mut buf, base_addr);
        buf
    }

    /// Bulk-fill `buf` with `buf.len()` bytes starting at linear `base_addr`,
    /// resolving both the legacy per-byte `state.memory[]` layout and the
    /// packed `--mc{N}` cell layout.
    ///
    /// For packed cabinets, walks cells linearly so each `state_vars[idx]`
    /// load yields two output bytes — the hot inner loop is a tight shift +
    /// store rather than the per-byte cell-table lookup you'd get from calling
    /// a byte-granularity helper in a loop. Non-packed paths (addr < 0,
    /// addr >= 0xF0000, or no packed table) fall through to the slow
    /// `state.read_mem` for correctness.
    fn fill_range_unified(&self, buf: &mut [u8], base_addr: usize) {
        let len = buf.len();
        if len == 0 { return; }
        let packed = !self.state.packed_cell_table.is_empty() && self.state.packed_cell_size > 0;
        if !packed {
            for i in 0..len {
                let addr = (base_addr + i) as i32;
                buf[i] = (self.state.read_mem(addr) & 0xFF) as u8;
            }
            return;
        }
        let pack = self.state.packed_cell_size as usize;
        // `pack == 2` is the only configuration kiln currently emits; the
        // shift-based extraction below assumes 2 bytes per cell.
        debug_assert_eq!(pack, 2);
        let table = &self.state.packed_cell_table;
        let state_vars = &self.state.state_vars;
        let mem_len = self.state.memory.len();
        let mem = &self.state.memory;
        let mut out_i = 0usize;
        let mut addr = base_addr;
        // Handle unaligned start: first byte might be the high half of a cell.
        if addr % pack != 0 && out_i < len {
            buf[out_i] = byte_at(addr, pack, table, state_vars, mem, mem_len);
            out_i += 1;
            addr += 1;
        }
        // Aligned bulk: two bytes per cell.
        while out_i + 1 < len {
            let cell_idx = addr / pack;
            let (lo, hi) = if cell_idx < table.len() {
                let sa = table[cell_idx];
                if sa < 0 {
                    let sidx = (-sa - 1) as usize;
                    if sidx < state_vars.len() {
                        let cell = state_vars[sidx];
                        ((cell & 0xFF) as u8, ((cell >> 8) & 0xFF) as u8)
                    } else {
                        fallback_pair(addr, mem, mem_len)
                    }
                } else {
                    fallback_pair(addr, mem, mem_len)
                }
            } else {
                fallback_pair(addr, mem, mem_len)
            };
            buf[out_i]     = lo;
            buf[out_i + 1] = hi;
            out_i += 2;
            addr += 2;
        }
        // Unaligned tail: one last low byte.
        if out_i < len {
            buf[out_i] = byte_at(addr, pack, table, state_vars, mem, mem_len);
        }
    }

    /// Single-byte convenience wrapper used by `get_video_mode` and friends.
    /// Delegates to the bulk fill to keep one source of truth for the
    /// packed/unpacked branching.
    #[inline]
    fn read_byte_unified(&self, addr: i32) -> u8 {
        if addr < 0 {
            return (self.state.read_mem(addr) & 0xFF) as u8;
        }
        let mut b = [0u8; 1];
        self.fill_range_unified(&mut b, addr as usize);
        b[0]
    }

    /// Render text-mode video memory as a string (for debugging).
    pub fn render_screen(&self, base_addr: usize, width: usize, height: usize) -> String {
        self.state.render_screen(base_addr, width, height)
    }

    /// Render text-mode video memory as HTML with CGA color spans.
    pub fn render_screen_html(&self, base_addr: usize, width: usize, height: usize) -> String {
        self.state.render_screen_html(base_addr, width, height)
    }

    /// Render a graphics-mode framebuffer as a PPM P6 image.
    ///
    /// Each byte at `base_addr + i` is a palette index; the returned buffer
    /// is a complete PPM P6 file including header. For VGA Mode 13h:
    /// `render_framebuffer(0xA0000, 320, 200)`.
    pub fn render_framebuffer(&self, base_addr: usize, width: usize, height: usize) -> Vec<u8> {
        self.state.render_framebuffer(base_addr, width, height)
    }

    /// Read a graphics-mode framebuffer as raw RGBA bytes.
    ///
    /// Returns `width * height * 4` bytes suitable for direct use with
    /// `new ImageData(new Uint8ClampedArray(bytes), width, height)` and
    /// `ctx.putImageData()` in the browser.
    pub fn read_framebuffer_rgba(
        &self,
        base_addr: usize,
        width: usize,
        height: usize,
    ) -> Vec<u8> {
        // Packed cabinets need the VRAM materialised from state vars before
        // the palette-resolving code runs (State::read_framebuffer_rgba indexes
        // `self.memory[]` directly). For unpacked cabinets the packed table is
        // empty and this unification is a no-op pass-through.
        if self.state.packed_cell_table.is_empty() {
            return self.state.read_framebuffer_rgba(base_addr, width, height);
        }
        let pixels = width * height;
        // Borrow state mutably only long enough to stage the bytes, then
        // call through the palette logic. Keeping a scratch buffer inside
        // State avoids an allocation per frame.
        let mut vram = vec![0u8; pixels];
        for i in 0..pixels {
            vram[i] = self.read_byte_unified((base_addr + i) as i32);
        }
        // Build a mini State view whose `memory` covers just the framebuffer
        // at base_addr, then call the existing palette resolver. DAC entries
        // are materialised into `scratch.extended` via the unified read path:
        // on packed cabinets the OUT 0x3C9 writes are routed through the
        // packed broadcast-write port directly into `state_vars` — they do
        // NOT go through `write_mem` and so never land in `self.extended`.
        // Just cloning `self.extended` here misses every DAC byte and the
        // palette-resolver falls back to CGA, producing the magenta/cyan
        // garbage we saw on Doom8088 / Prince of Persia in the web player.
        let mut scratch = calcite_core::State::default();
        if scratch.memory.len() < base_addr + pixels {
            scratch.memory.resize(base_addr + pixels, 0);
        }
        scratch.memory[base_addr..base_addr + pixels].copy_from_slice(&vram);
        scratch.extended = self.state.extended.clone();
        // Populate DAC palette via the unified path so packed cabinets work.
        // 768 bytes = 256 entries x RGB (matches CSS-DOS kiln/memory.mjs DAC_BYTES).
        for i in 0..768i32 {
            let addr = calcite_core::state::VGA_DAC_LINEAR + i;
            let v = self.read_byte_unified(addr) as i32;
            if v != 0 {
                scratch.extended.insert(addr, v);
            }
        }
        scratch.read_framebuffer_rgba(base_addr, width, height)
    }

    /// Read the current video mode from the BDA (0x0449).
    ///
    /// Returns the byte at flat address 0x0449 (BDA segment 0x0040, offset 0x49).
    /// This is written by INT 10h AH=00h (set mode) and read by AH=0Fh (get mode).
    /// Common values: 0x03 = 80x25 text, 0x13 = VGA Mode 13h (320x200x256).
    pub fn get_video_mode(&self) -> u8 {
        self.read_byte_unified(0x0449)
    }

    /// Read the last video mode the program REQUESTED, before any silent
    /// remap. Written by the corduroy BIOS to linear 0x04F2 on every
    /// INT 10h AH=00h call. If this differs from `get_video_mode()` the
    /// program asked for a mode CSS-DOS doesn't implement (EGA/VGA planar,
    /// CGA, Hercules, etc.) and was silently remapped to text mode 0x03.
    pub fn get_requested_video_mode(&self) -> u8 {
        self.read_byte_unified(0x04F2)
    }

    /// Read the sticky "unknown opcode" latch. 0 means none seen yet.
    /// A non-zero value is the opcode byte of the first instruction the
    /// CPU hit that has no dispatch entry — typically a 286/386/486
    /// instruction the 8086 core doesn't implement. Execution is
    /// effectively halted because IP can't advance through it.
    pub fn get_halt_code(&self) -> u8 {
        self.state.get_var("haltCode").unwrap_or(0) as u8
    }

    /// Return string properties as a JSON object string, e.g. `{"textBuffer":"Hello"}`.
    pub fn get_string_properties(&self) -> String {
        let pairs: Vec<String> = self
            .state
            .string_properties
            .iter()
            .map(|(k, v)| {
                let escaped = v
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n")
                    .replace('\r', "\\r")
                    .replace('\t', "\\t");
                format!("\"{k}\":\"{escaped}\"")
            })
            .collect();
        format!("{{{}}}", pairs.join(","))
    }
}

/// Serialise a [`MeasurementEvent`] to a JSON object string. `bytes`
/// payloads (from in-memory `DumpMemRange` / `Snapshot` actions) are
/// base64-encoded so the payload survives the JS string boundary; the
/// host decodes if it cares.
fn measurement_to_json(ev: &MeasurementEvent) -> String {
    let mut s = String::with_capacity(128);
    s.push('{');
    s.push_str(&format!("\"tick\":{}", ev.tick));
    let escaped_name = json_escape(&ev.watch_name);
    s.push_str(&format!(",\"watch\":\"{escaped_name}\""));
    s.push_str(&format!(",\"halted\":{}", ev.halted));
    s.push_str(",\"vars\":[");
    for (i, (n, v)) in ev.sampled_vars.iter().enumerate() {
        if i > 0 { s.push(','); }
        let n = json_escape(n);
        s.push_str(&format!("[\"{n}\",{v}]"));
    }
    s.push_str("],\"dumps\":[");
    for (i, d) in ev.dumps.iter().enumerate() {
        if i > 0 { s.push(','); }
        let path = match &d.path {
            Some(p) => format!("\"{}\"", json_escape(p)),
            None => "null".to_string(),
        };
        let b64 = if d.bytes.is_empty() {
            "\"\"".to_string()
        } else {
            format!("\"{}\"", base64_encode(&d.bytes))
        };
        s.push_str(&format!(
            "{{\"tag\":\"{}\",\"addr\":{},\"path\":{},\"bytes\":{}}}",
            d.tag, d.addr, path, b64
        ));
    }
    s.push_str("]}");
    s
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Minimal base64 encoder (no external dep). Used to ferry dump bytes
/// out via the JSON measurement stream.
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let b = &bytes[i..i + 3];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | (b[2] as u32);
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        out.push(ALPHABET[(n & 0x3F) as usize] as char);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let n = (bytes[i] as u32) << 16;
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8);
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        out.push('=');
    }
    out
}

/// Resolve a single byte at `addr` through the packed cell table, falling
/// back to `memory[]` at the end. Used for unaligned edges in
/// `fill_range_unified`.
#[inline]
fn byte_at(addr: usize, pack: usize, table: &[i32], state_vars: &[i32], mem: &[u8], mem_len: usize) -> u8 {
    let cell_idx = addr / pack;
    if cell_idx < table.len() {
        let sa = table[cell_idx];
        if sa < 0 {
            let sidx = (-sa - 1) as usize;
            if sidx < state_vars.len() {
                let cell = state_vars[sidx];
                let byte = if addr % pack == 0 { cell & 0xFF } else { (cell >> 8) & 0xFF };
                return byte as u8;
            }
        }
    }
    if addr < mem_len { mem[addr] } else { 0 }
}

/// Emit two bytes (low, high) from `memory[]` at `addr`. Used as the fallback
/// when the packed table has no coverage for the enclosing cell.
#[inline]
fn fallback_pair(addr: usize, mem: &[u8], mem_len: usize) -> (u8, u8) {
    let a = if addr < mem_len { mem[addr] } else { 0 };
    let b = if addr + 1 < mem_len { mem[addr + 1] } else { 0 };
    (a, b)
}
