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
    /// Extended properties: full-width i32 storage for addresses above the byte
    /// memory range (e.g., file I/O counters at 0xFFFF0+). Reads/writes bypass
    /// the u8 truncation of the memory array.
    pub extended: std::collections::HashMap<i32, i32>,
    /// String property values (e.g., `--textBuffer` for text output).
    pub string_properties: std::collections::HashMap<String, String>,
    /// Last keyboard input (for INT 16h / INT 21h).
    /// Packed as (scancode << 8 | ascii) matching DOS conventions.
    pub keyboard: i32,
    /// Tick counter (incremented each evaluation cycle).
    pub frame_counter: u32,
}

impl State {
    /// Create a new state with the given memory size.
    pub fn new(mem_size: usize) -> Self {
        Self {
            registers: [0; reg::COUNT],
            memory: vec![0; mem_size],
            extended: std::collections::HashMap::new(),
            string_properties: std::collections::HashMap::new(),
            keyboard: 0i32,
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
                // High addresses (>= 0xF0000) use extended map for full-width i32 storage
                if addr >= 0xF0000 {
                    return self.extended.get(&(addr as i32)).copied().unwrap_or(0);
                }
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
            // Write to register — mask to 16 bits, except IP which stores
            // flat addresses (CS*16 + offset) that can exceed 0xFFFF.
            -14..=-1 => {
                let reg_idx = (-addr - 1) as usize;
                if addr == crate::state::addr::IP {
                    self.registers[reg_idx] = value;
                } else {
                    self.registers[reg_idx] = value & 0xFFFF;
                }
            }
            _ => {
                let addr_u = addr as usize;
                // High addresses (>= 0xF0000) use extended map for full-width i32 storage
                if addr_u >= 0xF0000 {
                    self.extended.insert(addr, value);
                    return;
                }
                if addr_u < self.memory.len() {
                    self.memory[addr_u] = (value & 0xFF) as u8;
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
                row.push(cp437_to_unicode(ch));
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

        // First pass: find the maximum memory address to size the array.
        // Skip extended addresses (>= 0xF0000) — they use the HashMap, not the byte array.
        let mut max_addr: usize = 0;
        for prop in properties {
            if let Some(addr) = super::eval::property_to_address(&prop.name) {
                let addr_u = addr as usize;
                if addr >= 0 && addr_u < 0xF0000 {
                    max_addr = max_addr.max(addr_u + 1);
                }
            }
        }
        if max_addr > self.memory.len() {
            log::info!("Auto-sizing memory: {} → {} bytes", self.memory.len(), max_addr);
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
