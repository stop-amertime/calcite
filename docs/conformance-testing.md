# Conformance Testing

Calcite's correctness is verified by comparing its tick-by-tick register output
against a reference 8086 emulator (`tools/js8086.js`) running the same binary.
The reference emulator is authoritative — when calcite disagrees, calcite (or
the CSS generator) is wrong.

## Architecture

There are two modes: **DOS boot** (the primary workflow) and **simple BIOS**
(for standalone .COM programs).

### DOS boot mode

```
  BIOS + DOS kernel ──► js8086 reference emulator
                                                     fulldiff.mjs ──► first divergence
  CSS (from transpiler) ──► calcite-debugger (HTTP)
```

The debugger runs calcite as an HTTP server. `fulldiff.mjs` steps both
emulators in lockstep, comparing registers after every instruction. This is
the primary debugging workflow.

### Simple BIOS mode

```
  program.com + bios.bin ──► ref-emu.mjs (js8086)
                                                     compare.mjs ──► divergence report
  program.css ──► calcite-cli (--verbose)
```

For standalone .COM programs loaded at CS:IP = 0:0x100.

## The reference emulator

`tools/js8086.js` is a vendored copy of the [emu8](https://github.com/alex-code1234/emu8)
8086 CPU core. It executes real x86 machine code instruction-by-instruction
with no CSS involvement. It serves as ground truth for every register, every
flag bit, and every memory write.

**Always diff carefully against the reference.** Never assume calcite is
correct because it "looks right". The reference is the only source of truth.

## Quick start (DOS boot)

```bash
# 1. Generate CSS from a DOS program (in the i8086-css repo)
cd ../i8086-css
node transpiler/generate-dos.mjs programs/fib.com -o /tmp/fib.css

# 2. Start the debugger
cd ../calcite
cargo run --release -p calcite-debugger -- -i /tmp/fib.css

# 3. Find the first divergence (in another terminal)
node tools/fulldiff.mjs --ticks=5000
```

Or use `run.bat diagnose` for an interactive menu that automates steps 1-3.

## Tools

### `tools/fulldiff.mjs` — Primary divergence finder

The main tool for finding bugs. Requires the calcite-debugger running on
localhost:3333. Steps both emulators tick-by-tick and stops at the first
divergence.

```
node tools/fulldiff.mjs [--ticks=N] [--skip=N] [--port=3333]
```

Features:
- Compares ALL 16 bits of FLAGS (no masking — CF, PF, AF, ZF, SF, TF, IF, DF, OF)
- Handles REP instruction sync (JS ref does entire REP in one step, calcite
  expands per-tick, so fulldiff advances calcite by CX iterations to match)
- `--skip=N` to skip past known-good ticks (useful when debugging at tick 272K+)
- On divergence: prints previous state, instruction bytes, full register
  comparison with flag bit names, memory write diffs, and calcite CSS properties

Output looks like:

```
==============================================================================
FIRST DIVERGENCE at ref tick 272162 (calcite tick 285001)
==============================================================================

  Before: 0202:8991 (0a991)  [REP a5, CX: 128→0, calcite +128]
  Instruction bytes: f3 a5 cb 00 ...
  Pre-FLAGS: ref=0246 [PF|ZF|IF]

  Register        Reference    Calcite      Match
  ────────────────────────────────────────────────────────────
  AX              0000         0000           ✓
  FLAGS           0246         0242           ✗ DIFF  (diff bits: PF)
  ...

  Ref FLAGS:     0246 = 0000001001000110 [PF|ZF|IF]
  Calcite FLAGS: 0242 = 0000001001000010 [ZF|IF]
==============================================================================
```

### `tools/diagnose.mjs` — Property-level root cause analysis

Once you've found *which tick* diverges, this tool shows *why* — it
cross-references every CSS property against what the reference emulator
expects, pinpointing the exact property that computed wrong.

Requires the calcite-debugger running.

```
# For simple .COM programs:
node tools/diagnose.mjs <program.com> <bios.bin> [--ticks=N] [--port=3333]

# For DOS boot:
node tools/diagnose.mjs --dos [--ticks=N] [--port=3333]
```

### `tools/ref-dos.mjs` — Standalone DOS reference emulator

Runs the JS reference emulator in DOS boot mode with no calcite involvement.
Useful for understanding what the CPU *should* be doing.

```
node tools/ref-dos.mjs [--ticks=N] [--vga] [--trace] [--trace-from=N] [--halt-detect]
```

Options:
- `--ticks=N` — max instruction ticks (default 1000000)
- `--vga` — dump VGA text buffer at end
- `--trace` — print register state every tick
- `--trace-from=N` — start tracing at tick N (useful for large boot sequences)
- `--halt-detect` — stop when IP loops or HLT flag set (default on)

### `tools/ref-emu.mjs` — Standalone BIOS reference emulator

Runs the JS reference emulator for simple .COM programs (non-DOS).

```
node tools/ref-emu.mjs <program.com> <bios.bin> <ticks> [--json]
```

### `tools/compare.mjs` — Simple BIOS comparison

Tick-by-tick comparison for simple .COM programs. Runs both emulators and
reports the first divergence. Does not require the debugger.

```
node tools/compare.mjs <program.com> <bios.bin> <program.css> [--ticks=N]
```

### `../i8086-css/tools/compare-dos.mjs` — DOS boot comparison

Older DOS boot comparison tool that lives in the i8086-css repo. Runs both
emulators and reports divergences. Does not require the debugger (uses
calcite-cli directly).

```
cd ../i8086-css
node tools/compare-dos.mjs [--ticks=N]
```

### `calcite-cli --dump-tick N`

Runs calcite to tick N, then dumps all computed CSS property values.

```bash
RUST_LOG=error cargo run --release -p calcite-cli -- \
  --input program.css --dump-tick 24
```

### `calcite-cli --trace-json`

Outputs a JSON array of register states, one per tick.

```bash
RUST_LOG=error cargo run --release -p calcite-cli -- \
  --input program.css --ticks 100 --trace-json > trace.json
```

## Debugging workflow

### Standard workflow: find and fix a divergence

1. **Find it**: Start debugger + run `fulldiff.mjs`. Note the tick number.
2. **Diagnose it**: Run `diagnose.mjs` to see which CSS property is wrong.
3. **Understand it**: Use the debugger's `/state` and `/memory` endpoints to
   inspect calcite's internal state. Run `ref-dos.mjs --trace-from=N` to see
   what the reference does around that tick.
4. **Fix it**: Determine if the bug is in calcite or in the CSS generator:
   - If compiled and interpreted paths agree but both diverge from reference →
     CSS bug (fix in i8086-css transpiler)
   - If compiled and interpreted disagree → calcite compiler/evaluator bug
   - Use the debugger's `/compare-paths` endpoint to check this
5. **Verify**: Re-run `fulldiff.mjs` to confirm the fix and find the next divergence.

### Checking compiled vs interpreted paths

The debugger can run the same tick through both compiled (bytecode) and
interpreted (Expr tree) paths and diff the results:

```sh
curl -sX POST localhost:3333/seek -d '{"tick":272162}'
curl -s localhost:3333/compare-paths | python3 -m json.tool
```

If these disagree, the bug is in calcite's compiler. If they agree but diverge
from the reference, the bug is in the CSS.

### Using `run.bat` for interactive testing

```
run.bat              Interactive menu — pick a program, run it in calcite
run.bat diagnose     Pick a program and run full conformance diagnosis
```

Programs go in `programs/` (`.com` or `.exe` files, or subdirectories with
companion data files). CSS is auto-generated and cached in `programs/.cache/`.

## REP instruction sync

The JS reference emulator executes an entire REP string instruction (MOVSW,
STOSB, CMPSB, etc.) in a single step, decrementing CX to 0 (or until a
condition fails for REPE/REPNE). Calcite's CSS expands REP into individual
ticks, one per CX iteration.

`fulldiff.mjs` handles this automatically: when it detects a REP prefix before
a string operation, it reads CX before and after the reference step, then
advances calcite by that many ticks to stay in sync.

## Known considerations

- **All FLAGS bits are compared.** Earlier tools masked FLAGS to only check a
  subset (CF, PF, ZF, SF, DF, OF). `fulldiff.mjs` compares all 16 bits
  including AF, TF, IF — these matter for correct DOS operation.
- **Trap Flag (TF)**: The CSS emulator supports TF with proper INT 1 dispatch.
  The BIOS initializes a default INT 1 handler. The CSS transpiler auto-clears
  TF on interrupt stack push, matching real 8086 behaviour.
- **Timer (INT 1Ah)**: The BIOS auto-increments the timer tick counter on read,
  allowing DOS timeout loops to expire.
