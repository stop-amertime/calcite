# x86css-mw (multi-write fork)

A complete CSS-only 8086 emulator. Transpiles x86 binaries into CSS custom
properties and `@function` definitions that execute in a browser — or via
[calcify](https://github.com/nicholasgasior/calcify), a JIT compiler for
computational CSS.

Forked from [rebane2001/x86css](https://github.com/rebane2001/x86css) with
significant extensions for multi-write instructions, full ISA coverage, and
segmented memory support.

## Status

### 8086 ISA — complete

All 106 8086 instructions are implemented:

| Category | Instructions |
|----------|-------------|
| Arithmetic | ADD, ADC, SUB, SBB, INC, DEC, NEG, MUL, IMUL, DIV, IDIV, CBW, CWD |
| Logic | AND, OR, XOR, NOT, TEST, SHL, SHR, SAR, ROL, ROR, RCL, RCR |
| Data movement | MOV, XCHG, LEA, LES, LDS, XLAT, PUSH, POP |
| String ops | MOVSB/W, STOSB/W, LODSB/W, CMPSB/W, SCASB/W |
| Control flow | JMP, CALL, RET, RETF, IRET, INT, INTO, LOOP, LOOPZ, LOOPNZ, JCXZ |
| Conditional jumps | JZ, JNZ, JB, JNB, JBE, JA, JS, JNS, JL, JGE, JLE, JG, JO, JNO, JPE, JPO |
| Flags | CLC, STC, CMC, CLD, STD, CLI, STI, PUSHF, POPF, SAHF, LAHF |
| Prefixes | REP/REPZ, REPNZ, LOCK, segment overrides (ES:, CS:, SS:, DS:) |
| BCD | DAA, DAS, AAA, AAS, AAM, AAD |
| Far calls | CALL FAR, JMP FAR, RETF, IRET |
| I/O | IN, OUT, HLT, WAIT, NOP |

### Segmented memory — working

- ModR/M address calculation applies `segment * 16 + offset` with correct
  default segment selection (SS for BP-based, DS for others)
- String instructions use `DS:SI` and `ES:DI` per the 8086 spec
- LES/LDS load far pointers (offset + segment)
- Far CALL/JMP/RETF/IRET push/pop CS correctly
- Segment override prefixes are recognised but currently **no-ops** (the
  prefix doesn't change the default segment for the next instruction)

### Multi-write support

The original x86css could only write one value per tick. This fork supports
two write slots per tick (`addrDestA`/`addrDestB`), enabling instructions that
modify multiple destinations (e.g., XCHG, MUL/DIV writing DX:AX, string ops
updating both data and index registers).

Side channels handle additional implicit writes (SI/DI deltas for string ops,
SP for PUSH/POP, flags).

### DOS services (INT 21h) — minimal

Currently stubbed:

| AH | Function | Status |
|----|----------|--------|
| 30h | Get DOS version | Returns DOS 5.0 |
| 4Ch | Exit program | Halts (IP = IP) |

All other INT 21h functions return no-op. To run real DOS programs (text I/O,
file access), additional stubs are needed.

### Other interrupts

| Interrupt | Status |
|-----------|--------|
| INT 3 | No-op (breakpoint) |
| INT 10h | Not stubbed (BIOS video) |
| INT 16h | Keyboard via memory address 0x2100 |
| All others | No-op (advance IP) |

### REP prefixes

REP/REPZ/REPNZ are recognised as instructions in the CSS but execute as
no-ops. The [calcify evaluator](../crates/calcify-core/) handles REP
natively, decrementing CX and repeating the following string instruction in a
loop within a single tick.

## Building

### From assembly

Place your 8086 binary in `program.bin` and the `_start` offset in
`program.start` (as a decimal number). Then:

```sh
python3 build_css.py
# Output: x86css.html
```

### From C

Requires [gcc-ia16](https://gitlab.com/tkchia/build-ia16):

```sh
python3 build_c.py
python3 build_css.py
```

### Configuration

Edit the top of `build_css.py`:

```python
MEM_SIZE = 0x600       # Memory size in bytes (default 1.5KB)
PROG_OFFSET = 0x100    # Program load address (.COM convention)
```

Increase `MEM_SIZE` for larger programs. Each byte becomes a CSS custom
property, so large memory = large CSS output.

### Custom I/O

| Address | Function |
|---------|----------|
| 0x2000 | writeChar1 — write single byte to screen |
| 0x2002 | writeChar4 — write 4 bytes to screen |
| 0x2004 | writeChar8 — write 8 bytes to screen |
| 0x2006 | readInput — read keyboard input |
| 0x2100 | SHOW_KEYBOARD — toggle on-screen keyboard (0=off, 1=numeric, 2=alpha) |

## Running with calcify

The generated CSS can be executed directly by calcify for much higher
throughput than browser rendering:

```sh
# Parse and run the CSS
cargo run -p calcify-cli -- path/to/x86css.html --ticks 1000000
```

calcify compiles the CSS expressions to bytecode, achieving ~230K ticks/sec
with pattern recognition for dispatch tables, broadcast writes, and bitwise
operations.

## Credits

- [rebane2001](https://github.com/rebane2001) for the original x86css
- Jane Ori for the original [CPU Hack](https://dev.to/janeori/expert-css-the-cpu-hack-4ddj)
- Soo-Young Lee for the [8086 instruction set reference](https://www.eng.auburn.edu/~sylee/ee2220/8086_instruction_set.html)
- mlsite.net for the [8086 opcode map](http://www.mlsite.net/8086/)
- crtc-demos && tkchia for [gcc-ia16](https://gitlab.com/tkchia/build-ia16)

_Originally Feb 2026 by rebane2001. Multi-write fork Apr 2026._
