//! Emulator state — flat representation of the x86CSS machine.
//!
//! Replaces CSS's triple-buffered custom properties with direct mutable state.

/// CPU register indices into `State::registers`.
pub mod reg {
    /// Accumulator.
    pub const AX: usize = 0;
    /// Counter.
    pub const CX: usize = 1;
    /// Data.
    pub const DX: usize = 2;
    /// Base.
    pub const BX: usize = 3;
    /// Stack pointer.
    pub const SP: usize = 4;
    /// Base pointer.
    pub const BP: usize = 5;
    /// Source index.
    pub const SI: usize = 6;
    /// Destination index.
    pub const DI: usize = 7;
    /// Instruction pointer.
    pub const IP: usize = 8;
    /// Extra segment.
    pub const ES: usize = 9;
    /// Code segment.
    pub const CS: usize = 10;
    /// Stack segment.
    pub const SS: usize = 11;
    /// Data segment.
    pub const DS: usize = 12;
    /// Flags register.
    pub const FLAGS: usize = 13;
    /// Total number of registers.
    pub const COUNT: usize = 14;
}

/// x86CSS unified address space mapping.
///
/// x86CSS uses negative addresses for registers and split register halves.
/// These constants match the convention used in `base_template.html`.
pub mod addr {
    /// AX register (full 16-bit).
    pub const AX: i32 = -1;
    /// CX register (full 16-bit).
    pub const CX: i32 = -2;
    /// DX register (full 16-bit).
    pub const DX: i32 = -3;
    /// BX register (full 16-bit).
    pub const BX: i32 = -4;
    /// SP register (full 16-bit).
    pub const SP: i32 = -5;
    /// BP register (full 16-bit).
    pub const BP: i32 = -6;
    /// SI register (full 16-bit).
    pub const SI: i32 = -7;
    /// DI register (full 16-bit).
    pub const DI: i32 = -8;
    /// IP register (full 16-bit).
    pub const IP: i32 = -9;
    /// ES register (full 16-bit).
    pub const ES: i32 = -10;
    /// CS register (full 16-bit).
    pub const CS: i32 = -11;
    /// SS register (full 16-bit).
    pub const SS: i32 = -12;
    /// DS register (full 16-bit).
    pub const DS: i32 = -13;
    /// FLAGS register (full 16-bit).
    pub const FLAGS: i32 = -14;

    /// AH — high byte of AX. Address = -(reg_index + 20).
    pub const AH: i32 = -21;
    /// CH — high byte of CX.
    pub const CH: i32 = -22;
    /// DH — high byte of DX.
    pub const DH: i32 = -23;
    /// BH — high byte of BX.
    pub const BH: i32 = -24;

    /// AL — low byte of AX. Address = -(reg_index + 30).
    pub const AL: i32 = -31;
    /// CL — low byte of CX.
    pub const CL: i32 = -32;
    /// DL — low byte of DX.
    pub const DL: i32 = -33;
    /// BL — low byte of BX.
    pub const BL: i32 = -34;

}

/// Default memory size for x86CSS (0x600 bytes = 1,536).
pub const DEFAULT_MEM_SIZE: usize = 0x600;

/// The flat machine state that replaces CSS's triple-buffered custom properties.
#[derive(Debug, Clone)]
pub struct State {
    /// CPU registers (AX, CX, DX, BX, SP, BP, SI, DI, IP, ES, CS, SS, DS, FLAGS).
    pub registers: [i32; reg::COUNT],
    /// Flat memory (byte-addressable, default 1,536 bytes).
    pub memory: Vec<u8>,
    /// Text display buffer for BIOS INT 10h output.
    pub text_buffer: String,
    /// Last keyboard input (for INT 16h emulation).
    pub keyboard: u8,
    /// Tick counter (incremented each evaluation cycle).
    pub frame_counter: u32,
}

impl State {
    /// Create a new state with the given memory size.
    pub fn new(mem_size: usize) -> Self {
        Self {
            registers: [0; reg::COUNT],
            memory: vec![0; mem_size],
            text_buffer: String::new(),
            keyboard: 0,
            frame_counter: 0,
        }
    }

    /// Read from the unified address space.
    ///
    /// Address conventions (matching x86CSS's `readMem` / `base_template.html`):
    /// - `-1..-14`: full 16-bit registers (AX, CX, ..., FLAGS)
    /// - `-21..-24`: high byte halves (AH, CH, DH, BH)
    /// - `-31..-34`: low byte halves (AL, CL, DL, BL)
    /// - `0..`: memory bytes
    pub fn read_mem(&self, addr: i32) -> i32 {
        match addr {
            // Low byte halves: AL=-31, CL=-32, DL=-33, BL=-34
            -34..=-31 => {
                let reg_idx = (-addr - 31) as usize;
                Self::lo8(self.registers[reg_idx])
            }
            // High byte halves: AH=-21, CH=-22, DH=-23, BH=-24
            -24..=-21 => {
                let reg_idx = (-addr - 21) as usize;
                Self::hi8(self.registers[reg_idx])
            }
            // Full 16-bit registers: AX=-1, CX=-2, ..., FLAGS=-14
            -14..=-1 => {
                let reg_idx = (-addr - 1) as usize;
                self.registers[reg_idx]
            }
            _ => {
                let addr = addr as usize;
                if addr < self.memory.len() {
                    self.memory[addr] as i32
                } else {
                    0
                }
            }
        }
    }

    /// Read a 16-bit little-endian word from memory.
    pub fn read_mem16(&self, addr: i32) -> i32 {
        let lo = self.read_mem(addr);
        let hi = self.read_mem(addr + 1);
        lo + hi * 256
    }

    /// Write a value to the unified address space.
    ///
    /// Handles split register writes (AH/AL merge into AX, etc.) matching
    /// x86CSS's broadcast write pattern for split registers.
    pub fn write_mem(&mut self, addr: i32, value: i32) {
        match addr {
            // Write to low byte: AL=-31, CL=-32, DL=-33, BL=-34
            // Merges: keep high byte, replace low byte
            -34..=-31 => {
                let reg_idx = (-addr - 31) as usize;
                let hi = Self::hi8(self.registers[reg_idx]);
                self.registers[reg_idx] = hi * 256 + (value & 0xFF);
            }
            // Write to high byte: AH=-21, CH=-22, DH=-23, BH=-24
            // Merges: replace high byte, keep low byte
            -24..=-21 => {
                let reg_idx = (-addr - 21) as usize;
                let lo = Self::lo8(self.registers[reg_idx]);
                self.registers[reg_idx] = (value & 0xFF) * 256 + lo;
            }
            // Write to full 16-bit register — mask to 16 bits.
            // x86 registers are 16-bit unsigned; CSS uses --lowerBytes() for
            // truncation but intermediate values can overflow. Masking here
            // ensures register values stay in 0..65535 range.
            -14..=-1 => {
                let reg_idx = (-addr - 1) as usize;
                self.registers[reg_idx] = value & 0xFFFF;
            }
            _ => {
                let addr = addr as usize;
                if addr < self.memory.len() {
                    self.memory[addr] = (value & 0xFF) as u8;
                }
            }
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
                // Map printable ASCII; replace control chars with '.'
                row.push(if (0x20..0x7F).contains(&ch) {
                    ch as char
                } else if ch == 0 {
                    ' '
                } else {
                    '.'
                });
            }
            lines.push(row);
        }
        lines.join("\n")
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

    /// Initialize state from `@property` initial values.
    ///
    /// This loads the program binary and register defaults from the CSS —
    /// without it, the engine runs against empty memory.
    pub fn load_properties(&mut self, properties: &[crate::types::PropertyDef]) {
        use crate::types::CssValue;

        // First pass: find the maximum memory address to size the array
        let mut max_addr: usize = 0;
        for prop in properties {
            if let Some(addr) = super::eval::property_to_address(&prop.name) {
                if addr >= 0 {
                    max_addr = max_addr.max(addr as usize + 1);
                }
            }
        }
        if max_addr > self.memory.len() {
            self.memory.resize(max_addr, 0);
        }

        // Second pass: load values
        for prop in properties {
            let value = match &prop.initial_value {
                Some(CssValue::Integer(v)) => *v as i32,
                _ => continue,
            };

            let name = &prop.name;
            if let Some(addr) = super::eval::property_to_address(name) {
                self.write_mem(addr, value);
            }
        }
    }
}

/// Create a pre-tick hook that handles x86CSS external function stubs.
///
/// When IP is at a stub address, this hook captures text output by reading
/// characters from the stack/memory. The hook addresses and calling convention
/// are specific to x86CSS's generated CSS.
///
/// This function exists so callers (CLI, WASM, tests) can opt into x86CSS
/// text output handling. The evaluator core has no x86 knowledge.
pub fn x86css_text_output_hook() -> Box<dyn Fn(&mut State)> {
    Box::new(|state: &mut State| {
        let ip = state.registers[reg::IP];
        match ip {
            0x2000 => {
                // writeChar1: output 1 char from stack argument at SP+2
                let ch = state.read_mem(state.registers[reg::SP] + 2);
                if ch > 0 && ch < 128 {
                    state.text_buffer.push(ch as u8 as char);
                }
            }
            0x2002 => {
                // writeChar4: output 4 chars from string pointer at SP+2
                let ptr = state.read_mem16(state.registers[reg::SP] + 2);
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
                let ptr = state.read_mem16(state.registers[reg::SP] + 2);
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
    })
}

impl Default for State {
    fn default() -> Self {
        Self::new(DEFAULT_MEM_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_addressing() {
        let mut state = State::default();
        state.registers[reg::AX] = 0x1234;
        assert_eq!(state.read_mem(-1), 0x1234);

        state.write_mem(-1, 0xABCD);
        assert_eq!(state.registers[reg::AX], 0xABCD);
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
    fn split_register_read() {
        let mut state = State::default();
        state.registers[reg::AX] = 0x1234;

        // AL (low byte) at address -31
        assert_eq!(state.read_mem(addr::AL), 0x34);
        // AH (high byte) at address -21
        assert_eq!(state.read_mem(addr::AH), 0x12);
        // Full AX at address -1
        assert_eq!(state.read_mem(addr::AX), 0x1234);
    }

    #[test]
    fn split_register_write_low() {
        let mut state = State::default();
        state.registers[reg::AX] = 0x1200;

        // Write 0x34 to AL — should merge to 0x1234
        state.write_mem(addr::AL, 0x34);
        assert_eq!(state.registers[reg::AX], 0x1234);
    }

    #[test]
    fn split_register_write_high() {
        let mut state = State::default();
        state.registers[reg::AX] = 0x0034;

        // Write 0x12 to AH — should merge to 0x1234
        state.write_mem(addr::AH, 0x12);
        assert_eq!(state.registers[reg::AX], 0x1234);
    }

    #[test]
    fn all_split_registers() {
        let mut state = State::default();
        state.registers[reg::BX] = 0xABCD;

        assert_eq!(state.read_mem(addr::BH), 0xAB);
        assert_eq!(state.read_mem(addr::BL), 0xCD);

        state.write_mem(addr::BL, 0xEF);
        assert_eq!(state.registers[reg::BX], 0xABEF);

        state.write_mem(addr::BH, 0x00);
        assert_eq!(state.registers[reg::BX], 0x00EF);
    }

    #[test]
    fn register_write_stores_full_value() {
        let mut state = State::default();
        // Full register writes are masked to 16 bits (x86 registers are 16-bit).
        state.write_mem(addr::AX, 0x1_ABCD);
        assert_eq!(state.registers[reg::AX], 0xABCD);

        state.write_mem(addr::SP, -2);
        assert_eq!(state.registers[reg::SP], 0xFFFE);

        state.write_mem(addr::FLAGS, 0x10000);
        assert_eq!(state.registers[reg::FLAGS], 0); // masked to 16 bits
    }
}
