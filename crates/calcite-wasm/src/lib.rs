//! WASM bindings for calc(ite).
//!
//! This crate compiles to a WASM module that runs inside a Web Worker.
//! The main thread sends CSS text, the worker parses/compiles/evaluates,
//! and sends back property diffs for DOM application.

use wasm_bindgen::prelude::*;

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

        log::info!("Creating evaluator...");
        let evaluator = calcite_core::Evaluator::from_parsed(&parsed);
        log::info!("Evaluator created, loading properties...");
        let mut state = calcite_core::State::default();
        state.load_properties(&parsed.properties);
        log::info!("Properties loaded, memory size: {} bytes", state.memory.len());

        Ok(CalciteEngine { state, evaluator })
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

    /// Set the keyboard input state.
    /// Pass (scancode << 8 | ascii), or 0 for no key.
    /// Writes to memory 0x500 (ascii) and 0x501 (scancode) — the address
    /// that the CSS-DOS BIOS polls for keystrokes via INT 16h.
    pub fn set_keyboard(&mut self, key: i32) {
        self.state.write_mem(0x500, key & 0xFF);
        self.state.write_mem(0x501, (key >> 8) & 0xFF);
    }

    /// Get the current value of a register (for debugging).
    pub fn get_register(&self, index: usize) -> i32 {
        if index < self.state.registers.len() {
            self.state.registers[index]
        } else {
            0
        }
    }

    /// Read text-mode video memory (character bytes only).
    ///
    /// Returns `width * height` bytes from video memory at `base_addr`.
    /// Default for DOS text mode: `read_video_memory(0xB8000, 40, 25)`.
    pub fn read_video_memory(&self, base_addr: usize, width: usize, height: usize) -> Vec<u8> {
        self.state.read_video_memory(base_addr, width, height)
    }

    /// Render text-mode video memory as a string (for debugging).
    pub fn render_screen(&self, base_addr: usize, width: usize, height: usize) -> String {
        self.state.render_screen(base_addr, width, height)
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
        self.state.read_framebuffer_rgba(base_addr, width, height)
    }

    /// Detect VGA memory regions (text and/or graphics) from the CSS.
    ///
    /// Returns a JSON object:
    /// ```json
    /// {
    ///   "text": {"addr": 753664, "size": 4000, "width": 80, "height": 25},
    ///   "gfx":  {"addr": 655360, "size": 64000, "width": 320, "height": 200}
    /// }
    /// ```
    /// Either field can be `null` if that mode isn't present. Both can
    /// be present simultaneously for programs that use both text and gfx
    /// memory regions.
    pub fn detect_video(&self) -> String {
        let regions = calcite_core::detect_video_regions();
        let text_json = match regions.text {
            Some((addr, size)) => {
                // Text mode: size/2 cells (char+attr pairs)
                let cells = size / 2;
                let (w, h) = if cells == 2000 {
                    (80, 25)
                } else if cells == 4000 {
                    (80, 50)
                } else if cells == 1000 {
                    (40, 25)
                } else {
                    (80, cells / 80)
                };
                format!("{{\"addr\":{addr},\"size\":{size},\"width\":{w},\"height\":{h}}}")
            }
            None => "null".to_string(),
        };
        let gfx_json = match regions.gfx {
            Some((addr, size)) => {
                // Graphics mode: 1 byte per pixel
                let (w, h) = if size == 64000 {
                    (320, 200)
                } else {
                    // Unknown size — assume 320 wide and derive height
                    (320, size / 320)
                };
                format!("{{\"addr\":{addr},\"size\":{size},\"width\":{w},\"height\":{h}}}")
            }
            None => "null".to_string(),
        };
        format!("{{\"text\":{text_json},\"gfx\":{gfx_json}}}")
    }

    /// Read the current video mode from the BDA (0x0449).
    ///
    /// Returns the byte at flat address 0x0449 (BDA segment 0x0040, offset 0x49).
    /// This is written by INT 10h AH=00h (set mode) and read by AH=0Fh (get mode).
    /// Common values: 0x03 = 80x25 text, 0x13 = VGA Mode 13h (320x200x256).
    pub fn get_video_mode(&self) -> u8 {
        self.state.read_mem(0x0449) as u8
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
