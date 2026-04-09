NOTE: this workspace covers: 
./ calcite (usually the cwd)
../i8086-css (the next folder over)
more subfolders/repos for individual other games like doom.css eventually

As an agent, you are working in all three. If a fix needs to go in i8086-css (hopefully soon to be CSS-DOS) put it there. Just because you are in the calcite cwd doesn't mean you can only work on calcite. 

# calc(ite)

A JIT compiler for computational CSS.

## Quick reference

```sh
cargo test --workspace          # tests
cargo clippy --workspace        # lint
cargo fmt --all                 # format
just check                      # all three

cargo bench -p calcite-core     # criterion benchmarks (needs fixture)
wasm-pack build crates/calcite-wasm --target web --out-dir ../../web/pkg

# Debug server: parse once, step/inspect/compare via HTTP
cargo run --release -p calcite-debugger -- -i program.css
# Then: curl localhost:3333/state, /tick, /seek, /compare-paths, etc.
# See docs/debugger.md for full API
```

## Project layout

```
crates/
  calcite-core/      Core engine: parser, pattern compiler, evaluator, state
  calcite-cli/       CLI tool for running CSS through the engine
  calcite-debugger/  HTTP debug server — see docs/debugger.md
  calcite-wasm/      WASM bindings (wasm-bindgen) for browser Web Worker
web/
  index.html           Browser UI
  calcite-worker.js    Web Worker bridge
site/
  index.html           CSS-DOS showcase site
  programs/            Pre-compiled .css.gz program files for the site
programs/
  *.com, *.exe         DOS programs to run
  .cache/              Auto-generated CSS (via generate-dos.mjs)
tools/
  js8086.js            Reference 8086 emulator (vendored from emu8)
  fulldiff.mjs         Primary: first-divergence finder (REP-aware, full FLAGS)
  diagnose.mjs         Property-level CSS diagnosis at divergence point
  compare.mjs          Tick-by-tick comparison for simple BIOS .COM programs
  ref-emu.mjs          Standalone reference emulator (simple BIOS programs)
  ref-dos.mjs          Standalone reference emulator (DOS boot mode)
tests/
  fixtures/            Pre-compiled CSS from i8086-css
run.bat                Interactive menu to run/diagnose DOS programs
```

## Architecture

### Pipeline

```
CSS text → parse → pattern recognition → compile → bytecode → evaluate (tick loop)
```

1. **Parser** (`parser/`): Tokenises via `cssparser` crate. Parses `@property`,
   `@function`, `if(style())`, `calc()`, `var()`, CSS math functions, string
   literals. Output: `ParsedProgram` (properties + functions + assignments).

2. **Pattern recognition** (`pattern/`): Detects optimisable structures:
   - Dispatch tables: `if(style(--prop: N))` chains (≥4 branches) → HashMap
   - Broadcast writes: `if(style(--dest: N): val; else: keep)` (≥10 entries)
     → direct store, with word-write spillover support

3. **Compiler** (`compile.rs`): Flattens `Expr` trees → flat `Op` bytecode
   with indexed slots. Function body patterns (identity, bitmask, shifts,
   bit extraction) detected for interpreter fast-paths.

4. **Evaluator** (`eval.rs`): Runs the tick loop. Two paths:
   - Compiled: linear bytecode against slot array (fast path)
   - Interpreted: recursive Expr walking (fallback)
   Topological sort on assignments. Pre-tick hooks for side effects.

5. **State** (`state.rs`): Flat mutable replacement for CSS triple-buffer.
   Unified address space (negative = registers, non-negative = memory).
   Split-register merging (AH/AL ↔ AX). Auto-sized from @property decls.

### Key types (`types.rs`)

- `Expr` — expression tree (Literal, Var, Calc, StyleCondition, FunctionCall, etc.)
- `ParsedProgram` — parser output (properties, functions, assignments)
- `Assignment` — property name → Expr
- `FunctionDef` — name, parameters, locals, result expression
- `PropertyDef` — name, syntax, initial value, inheritance
- `CssValue` — Integer | String

### The cardinal rule

The entire point of this project is to see what pure CSS can do. The CSS
is a working program that runs in Chrome — no JavaScript, no WebAssembly,
just CSS custom properties and `calc()` evaluating an 8086 CPU. That's the
joke and the demo: "Doom runs in a stylesheet."

The CSS must be completely self-contained and functional on its own.
Chrome is the reference implementation. If you open the HTML file in Chrome,
it works. Slowly — maybe one frame per year — but it works. Every memory
cell, every register, every instruction decode is a CSS expression.

Calcite exists to make that CSS fast enough to be usable. It is a JIT
compiler for CSS, analogous to V8 for JavaScript. It parses the CSS, finds
patterns it can execute more efficiently, and produces the same results
Chrome would — just orders of magnitude faster.

What this means in practice:

- **Calcite must NEVER have x86 knowledge.** No opcode reading, no
  instruction semantics, no emulation. It evaluates CSS expressions — it
  doesn't know or care what they compute.
- **The CSS dictates everything, not calcite.** If the CSS has 6000 memory
  cell properties, calcite evaluates 6000 memory cell properties. If it
  needs a million properties for 1MB of RAM, calcite handles a million
  properties. It can recognise patterns and optimise (broadcast writes,
  dispatch tables), but it cannot skip, remove, or alter what the CSS
  expresses — just like V8 can't skip your JavaScript, only run it faster.
- **Never suggest CSS changes to help calcite.** The CSS is written to work
  in Chrome. Telling i8086-css to "not emit properties" or restructure
  things for calcite's benefit is backwards — like telling a JS developer
  to write worse code so V8 can optimise it.
- **No features that don't exist in CSS.** If Chrome can't evaluate it,
  calcite can't rely on it. Calcite can be smarter about evaluating CSS
  patterns, but it can't invent new semantics.
- **If calcite disagrees with Chrome, calcite is wrong.**
- **Pattern recognition is the whole game.** Recognising that 6000
  assignments all check the same property and converting them to a HashMap
  lookup is exactly what a JIT does. Same results, faster. That's the job.

### Relationship to CSS-DOS (formerly i8086-css)

[CSS-DOS](../i8086-css) is a sibling repo that generates 8086 CSS. It is
undergoing an architecture pivot from a JSON instruction database approach
(now in `legacy/`) to a JS→CSS transpiler (see `transpiler/`). See its
`CLAUDE.md` for details.

Calcite's only interface with CSS-DOS is the `.css` output — test fixtures in
`tests/fixtures/` are pre-compiled CSS. There is no crate dependency and must
never be one.

### Conformance testing — the main debugging workflow

The ground truth is `tools/js8086.js`, a reference 8086 emulator in JavaScript.
Every bug fix must be validated by diffing calcite against this reference
implementation tick-by-tick. The reference emulator runs real x86 machine code
with no CSS involvement, so any divergence means calcite (or the CSS generator)
is computing something wrong.

**The key principle: always diff carefully against the reference.** The
reference emulator is authoritative. When calcite and the reference disagree,
calcite is wrong (or the CSS is wrong). Never assume calcite is correct just
because it "looks right" — always verify with the reference.

The workflow for finding and fixing bugs is:

1. **Find the first divergence.** Start the HTTP debugger, then run
   `fulldiff.mjs` — it steps both emulators tick-by-tick, comparing ALL
   registers including all 16 bits of FLAGS (no masking). It handles REP
   instruction sync (the JS reference executes an entire REP in one step,
   while calcite expands it per-tick). It stops at the first mismatch with
   full context: previous state, instruction bytes, register comparison,
   and FLAG bit diffs.

   ```sh
   # Start the debugger
   cargo run --release -p calcite-debugger -- -i program.css
   # Find first divergence (in another terminal)
   node tools/fulldiff.mjs --ticks=5000
   # Skip past known-good ticks to search further
   node tools/fulldiff.mjs --ticks=5000 --skip=10000
   ```

2. **Diagnose the root cause.** Once you know the divergent tick,
   `diagnose.mjs` digs into CSS property-level diagnostics — it
   cross-references every CSS property against what the reference emulator
   expects, showing exactly which property computed the wrong value and why.

   ```sh
   # For simple .COM programs (debugger must be running):
   node tools/diagnose.mjs program.com bios.bin --ticks=5000
   # For DOS boot:
   node tools/diagnose.mjs --dos --ticks=5000
   ```

3. **Fix the bug.** Bugs are OFTEN in the CSS generator (i8086-css transpiler),
   not in calcite. When compiled and interpreted paths agree but diverge from
   the reference emulator, it's a CSS bug — fix it in i8086-css regardless of
   which repo you're working in. Fix, regenerate the CSS, re-run the
   comparison, repeat.

4. **Run the reference standalone to understand expected behaviour.** When you
   need to understand what the CPU *should* be doing without involving calcite:

   ```sh
   # DOS boot mode (same BIOS + DOS kernel):
   node tools/ref-dos.mjs --ticks=50000 --vga --trace-from=272000
   # Simple .COM program:
   node tools/ref-emu.mjs program.com bios.bin 5000 --json
   ```

Key tools:
- `tools/fulldiff.mjs` — **primary**: find first divergence, REP-aware, full FLAGS
- `tools/diagnose.mjs` — property-level CSS diagnosis (needs debugger running)
- `tools/ref-dos.mjs` — standalone reference emulator (DOS boot mode)
- `tools/ref-emu.mjs` — standalone reference emulator (simple BIOS programs)
- `tools/compare.mjs` — tick-by-tick comparison for simple BIOS .COM programs
- `../i8086-css/tools/compare-dos.mjs` — older DOS boot comparison (runs in i8086-css repo)

### Running programs

Drop any `.com` file into `programs/` and use `run.bat`:

```
run.bat              Interactive menu — pick a program by number
run.bat diagnose     Conformance diagnosis menu
```

CSS is auto-generated via the DOS transpiler pipeline and cached in
`programs/.cache/`. The generator lives at `../i8086-css/transpiler/generate-dos.mjs`.
