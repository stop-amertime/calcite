# Conformance Testing

Calcite's correctness is verified by comparing its tick-by-tick register output against a reference 8086 emulator running the same BIOS + program binary.

## Architecture

```
                  ┌─────────────┐
  fib.com ───────►│ ref-emu.mjs │──► reference register trace
  bios.bin ──────►│ (js8086)    │
                  └─────────────┘
                                     compare.mjs ──► divergence report
                  ┌─────────────┐
  fib.css ───────►│ calcite-cli │──► calcite register trace
                  │ (--verbose) │
                  └─────────────┘
```

The reference emulator (`tools/js8086.js`) is a vendored copy of the [emu8](https://github.com/alex-code1234/emu8) 8086 CPU core. It executes real x86 machine code instruction-by-instruction with no CSS involvement, so it serves as ground truth.

## Quick start

```bash
# 1. Build the CSS from a .COM binary (in the i8086-css repo)
cd ../i8086-css
node web/build-cli.mjs examples/fib.com -o /tmp/fib.css

# 2. Run the comparison
cd ../calcite
node tools/compare.mjs \
  ../i8086-css/examples/fib.com \
  ../i8086-css/bios.bin \
  /tmp/fib.css \
  --ticks=500
```

Output looks like:

```
============================================================
CONFORMANCE REPORT: ../i8086-css/examples/fib.com
============================================================
Ticks compared: 500
Matching ticks: 24

FIRST DIVERGENCE at tick 24:
────────────────────────────────────────
    Tick 22:
     REF: AX=3654 CX=0 DX=70 BX=64 ... DS=0
     CAL: AX=3654 CX=0 DX=70 BX=64 ... DS=0
    Tick 23:
     ...
>>> Tick 24:
     REF: AX=3654 CX=0 DX=0 BX=64 ... DS=64
     CAL: AX=3654 CX=0 DX=70 BX=64 ... DS=64

Divergent registers:
  DX: ref=0 (0x0)  calcite=70 (0x46)

Instruction context:
  Previous IP: 983056 (0xf0010) = BIOS+0x10
  Bytes at IP: 8a 16 50 00 8a 36 (BIOS+0x10)
============================================================
```

## Tools

### `tools/compare.mjs`

Automated tick-by-tick comparison. Runs both emulators and reports the first divergence.

```
node tools/compare.mjs <program.com> <bios.bin> <program.css> [options]

Options:
  --ticks=N      Number of ticks to compare (default: 500)
```

The report includes:
- 3 ticks of context before the divergence
- Which registers differ and by how much
- The instruction bytes at the previous IP (to identify which x86 instruction caused the divergence)

### `tools/ref-emu.mjs`

Standalone reference emulator. Outputs one line per tick in the same format as `calcite-cli --verbose`.

```
node tools/ref-emu.mjs <program.com> <bios.bin> <ticks> [--json]

# Text output (matches calcite --verbose format):
Tick 0: AX=0 CX=0 DX=0 BX=0 SP=1528 ... flags=2

# JSON output:
[{"tick":0,"AX":0,"CX":0,...}, ...]
```

### `calcite-cli --dump-tick N`

Runs calcite to tick N, then dumps all computed CSS property values. Use this to diagnose WHY a divergence occurred — not just which register is wrong, but which intermediate property computed the wrong value.

```bash
RUST_LOG=error cargo run --release -p calcite-cli -- \
  --input /tmp/fib.css --dump-tick 24
```

Output:

```
=== Slot dump at tick 24 ===
Registers: AX=3654 CX=0 DX=70 ...

Computed properties:
  --addrDestA: 991 (0x3DF)      ← wrong, should be -33
  --addrValA: 0 (0x0)           ← correct
  --instId: 150 (0x96)
  --modRm_addr2: 991 (0x3DF)    ← wrong, should be 1104
  ...
```

### `calcite-cli --trace-json`

Outputs a JSON array of register states, one per tick. Useful for programmatic analysis.

```bash
RUST_LOG=error cargo run --release -p calcite-cli -- \
  --input /tmp/fib.css --ticks 100 --trace-json > trace.json
```

### `Evaluator::get_slot_value(name)`

Rust API for reading a compiled property's value after a tick. Used by `--dump-tick` internally, also available for integration tests.

```rust
evaluator.tick(&mut state);
let dest = evaluator.get_slot_value("--addrDestA"); // Some(-33)
let val = evaluator.get_slot_value("--addrValA");   // Some(0)
```

## Debugging workflow

When a program produces wrong output:

1. **Find the divergence**: `node tools/compare.mjs prog.com bios.bin prog.css --ticks=1000`
2. **Dump the tick**: `cargo run -p calcite-cli -- --input prog.css --dump-tick N`
3. **Trace the bad value**: Look at which property is wrong in the dump. Follow its dependencies (e.g., `--addrDestA` depends on `--getDest(0)` which depends on `--instArgDest1` which depends on `--modRm_addr2`).
4. **Check the CSS expression**: Find the expression in the CSS file and verify calcite evaluates it correctly.
5. **Fix and re-run**: After fixing, re-run `compare.mjs` to confirm the divergence is gone and no new ones appear.

## Known limitations

- **FLAGS register**: The reference emulator tracks all x86 flags (CF, PF, AF, ZF, SF, OF, etc.) while calcite only computes the subset the CSS tracks (currently ZF and SF). FLAGS comparisons are skipped in `compare.mjs`.
- **Segment overrides**: The reference emulator supports segment override prefixes; the i8086-css emulator treats them as no-ops. Instructions using segment overrides will diverge.
- **Keyboard/timer**: The reference emulator doesn't simulate keyboard input or timer interrupts. Programs that block on `INT 16h AH=00h` (read key) will loop forever in the reference emulator.

## BIOS setup

Both emulators load the same BIOS binary at F000:0000 with these IVT entries:

| INT | Handler offset | Function |
|-----|---------------|----------|
| 10h | 0x0000 | Video services |
| 16h | 0x0155 | Keyboard |
| 1Ah | 0x0190 | Timer |
| 20h | 0x0232 | Program terminate |
| 21h | 0x01A9 | DOS services |

Initial register state matches i8086-css .COM file conventions: CS=0, IP=0x100, SP=0x5F8, all others zero.
