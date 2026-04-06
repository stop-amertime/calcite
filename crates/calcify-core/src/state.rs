//! Emulator state — flat representation of the x86CSS machine.
//!
//! Replaces CSS's triple-buffered custom properties with direct mutable state.

/// CPU register indices into `State::registers`.
pub mod reg {
    pub const AX: usize = 0;
    pub const CX: usize = 1;
    pub const DX: usize = 2;
    pub const BX: usize = 3;
    pub const SP: usize = 4;
    pub const BP: usize = 5;
    pub const SI: usize = 6;
    pub const DI: usize = 7;
    pub const IP: usize = 8;
    pub const ES: usize = 9;
    pub const CS: usize = 10;
    pub const SS: usize = 11;
    pub const DS: usize = 12;
    pub const FLAGS: usize = 13;
    pub const COUNT: usize = 14;
}

/// x86CSS unified address space mapping.
///
/// x86CSS uses negative addresses for registers and split register halves.
/// These constants match the convention used in `base_template.html`.
pub mod addr {
    // Full 16-bit registers (negative addresses used by readMem/broadcast write)
    pub const AX: i32 = -1;
    pub const CX: i32 = -2;
    pub const DX: i32 = -3;
    pub const BX: i32 = -4;
    pub const SP: i32 = -5;
    pub const BP: i32 = -6;
    pub const SI: i32 = -7;
    pub const DI: i32 = -8;
    pub const IP: i32 = -9;
    pub const ES: i32 = -10;
    pub const CS: i32 = -11;
    pub const SS: i32 = -12;
    pub const DS: i32 = -13;
    pub const FLAGS: i32 = -14;

    // High byte halves (AH, CH, DH, BH) — address = -(reg_index + 20)
    pub const AH: i32 = -21;
    pub const CH: i32 = -22;
    pub const DH: i32 = -23;
    pub const BH: i32 = -24;

    // Low byte halves (AL, CL, DL, BL) — address = -(reg_index + 30)
    pub const AL: i32 = -31;
    pub const CL: i32 = -32;
    pub const DL: i32 = -33;
    pub const BL: i32 = -34;

    // External function addresses (0x2000–0x200F)
    pub const EXT_BASE: i32 = 0x2000;
    pub const EXT_WRITE_CHAR: i32 = 0x2006;

    // External I/O addresses (0x2100–0x210F)
    pub const EXT_IO_BASE: i32 = 0x2100;
}

/// Default memory size for x86CSS (0x600 bytes = 1,536).
pub const DEFAULT_MEM_SIZE: usize = 0x600;

/// The flat machine state that replaces CSS's triple-buffered custom properties.
#[derive(Debug, Clone)]
pub struct State {
    pub registers: [i32; reg::COUNT],
    pub memory: Vec<u8>,
    pub text_buffer: String,
    pub keyboard: u8,
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
            // Write to full register (mask to 16 bits — x86 8088 registers are 16-bit)
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

    /// Initialize state from `@property` initial values.
    ///
    /// This loads the program binary and register defaults from the CSS —
    /// without it, the engine runs against empty memory.
    pub fn load_properties(&mut self, properties: &[crate::types::PropertyDef]) {
        use crate::types::CssValue;

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
    fn register_write_masks_to_16_bits() {
        let mut state = State::default();
        // Values exceeding 16 bits should be masked
        state.write_mem(addr::AX, 0x1_ABCD);
        assert_eq!(state.registers[reg::AX], 0xABCD);

        state.write_mem(addr::SP, 0xFFFF_FFFF_u32 as i32);
        assert_eq!(state.registers[reg::SP], 0xFFFF);

        state.write_mem(addr::FLAGS, 0x10000);
        assert_eq!(state.registers[reg::FLAGS], 0);
    }
}
