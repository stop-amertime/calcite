# calc(ify): A JIT Compiler for Computational CSS

## Project Summary

A WASM-based JIT compiler that replaces Chrome's style resolver for computational CSS, running inside the browser as a Web Worker. It parses real CSS files, recognises computational patterns, and compiles them into efficient native operations — then writes the computed property values back to the DOM for Chrome to render normally. The primary benchmark is running Lyra Rebane's x86CSS emulator (https://lyra.horse/x86css/) faster than Chrome's native style resolver can. The ultimate demo is **doom.css** — Doom compiled to 8086 machine code, embedded in CSS, running in a browser at playable speed.

The CSS is the source of truth. The browser still renders the output. The engine just evaluates the computational properties faster than Chrome does, the same way V8 JIT-compiles JavaScript faster than an interpreter would — the observable behaviour is identical, the internals are optimised.

## Motivation

CSS is accidentally Turing complete. People have built x86 CPUs, computers, and state machines in pure CSS. But the execution speed is terrible — not primarily because of rendering overhead (as previously assumed), but because CSS's evaluation model forces computational patterns into pathologically inefficient data structures.

The x86CSS emulator encodes memory as a 1,600-branch `if(style())` chain. Every memory read scans this chain linearly. It encodes register writes as a broadcast comparison where all 1,583 state variables check whether they're the write target. It implements bitwise XOR by decomposing 16-bit integers into individual bits, operating on each, and reconstructing. These patterns are correct CSS, but they're expressing O(1) operations as O(N) evaluations because CSS has no arrays, no addressed writes, and no bitwise operators.

Chrome's style engine evaluates this CSS faithfully and slowly. It has no reason to optimise for these patterns — `if()` and `@function` are brand new features, and nobody at Google was anticipating 1,600-branch conditional chains. The engine recognises these patterns and replaces them with efficient equivalents: hash-map dispatch for `if(style())` chains, direct-addressed writes instead of broadcast comparisons, and (optionally) native bitwise operations instead of bit decomposition.

## What This Is

A smarter style resolver for computational CSS. It sits between the CSS source and the rendered output, doing what Chrome's engine does but with pattern-aware optimisation. The contract is: given the same CSS, produce the same computed property values. How it gets there internally is its own business.

The line we draw: **the input is real CSS, the output is computed property values written back to the DOM, and the browser renders the result**. Everything between input and output is fair game for optimisation. This is the same contract any browser engine operates under.

## What This Is Not

- Not a hand-compiled reimplementation of x86CSS (the engine reads the CSS, it doesn't know it's an x86 emulator)
- Not a standalone runtime that replaces the browser (the browser still renders)
- Not trying to be spec-complete for all of CSS (only the computational subset needs to be correct)
- Not tied to x86CSS specifically — any computational CSS using the same feature set should work

## How x86CSS Actually Works (Source Analysis)

*This section is based on reading `base_template.html`, `build_css.py`, and `x86css.html` from the x86CSS repository.*

### DOM Structure

The structure is flat — far simpler than originally assumed:

```
.clock          ← container element, drives the clock animation
  └── .cpu      ← ALL computation happens here, as property declarations
        ├── .screen       ← display output (content via ::after)
        ├── key-board     ← input (button :hover:active → --keyboard property)
        └── table-container ← debug register display
```

There are only **3 functional `@container` rules**, all checking `--clock` on `.clock`:

```css
@container style(--clock: 1) { animation-play-state: running, paused }
@container style(--clock: 3) { animation-play-state: paused, running }
```

There is no container query hierarchy, no multi-element scoping, no deep nesting. The entire CPU — decode, execute, register update, memory update — is property declarations on a single element.

### The 4-Phase Triple Buffer

CSS can't do mutable assignment: you can't write `--x: calc(var(--x) + 1)` because that's a cycle. x86CSS solves this with a triple-buffer scheme using two CSS animations (`store` and `execute`) toggled by the clock:

Every variable exists in four copies:
- `--{name}` — the computed new value (the actual logic)
- `--__0{name}` — captured by the `execute` keyframe
- `--__1{name}` — the current read value, set from `--__2`
- `--__2{name}` — captured by the `store` keyframe

The clock cycles through 4 phases per instruction. Only phases 1 and 3 do real work:
- Phase 1: `store` animation runs → copies `--__0` (last execution output) into `--__2`
- Phase 3: `execute` animation runs → `.cpu` reads `--__1` from `--__2`, computes new `--{name}`, execute keyframe captures into `--__0`

The JS clock shortcut bypasses the CSS animation by setting `--clock` directly via `element.style` with `!important`, forcing synchronous style recalculation via `getComputedStyle()`.

**The engine eliminates this entirely.** Since the engine can do mutable assignment, it just reads state, computes, and writes — one pass, not four.

### State Variables

| Category | Count | Notes |
|----------|-------|-------|
| CPU registers | 14 | AX, CX, DX, BX, SP, BP, SI, DI, IP, ES, CS, SS, DS, flags |
| Frame counter | 1 | |
| Memory (0x000–0x5FF) | 1,536 | 0x600 bytes |
| External functions (0x2000–0x200F) | 16 | writeChar, readInput, etc. |
| External I/O (0x2100–0x210F) | 16 | keyboard visibility, etc. |
| **Total** | **1,583** | |

Of these, 1,024 are read-only (the program binary, loaded at 0x100) and 559 are read-write (triple-buffered in the CSS). Total custom properties in the CSS: ~3,340.

### The Execution Pipeline (Per Tick)

All the following are property declarations on `.cpu`, evaluated in declaration order:

```
1. READ STATE:     --__1{name} = var(--__2{name})           for each variable
2. DERIVE BYTES:   --AL = mod(--__1AX, 256)                 8 byte registers
3. DECODE:         --instId = readMem(IP) → dispatch        opcode lookup
4. PARSE MODRM:    --modRm = readMem(IP+1)                  bit extraction
5. RESOLVE ARGS:   --instArg1 = getInstArg(type, offset)    may call readMem
6. EXECUTE:        --addrDestA = D-{instruction}(0)         destination address
                   --addrValA  = V-{instruction}(0)         computed value
7. BROADCAST WRITE: for each variable:
                   --{name} = if(
                     style(--addrDestA: {my_addr}): addrValA;
                     style(--addrDestB: {my_addr}): addrValB;
                     else: --__1{name}                       keep old value
                   )
```

### The Three Bottleneck Patterns

**1. `readMem()` — O(N) linear scan, called ~10x per tick**

```css
@function --readMem(--at <integer>) returns <integer> {
  result: if(
    style(--at:-1): var(--__1AX);
    style(--at:-2): var(--__1CX);
    ... /* 34 register aliases */
    style(--at:0): var(--__1m0);
    style(--at:1): var(--__1m1);
    ... /* 1,536 memory bytes */
    style(--at:8192): var(--__1m8192);
    ... /* 32 external addresses */
  );
}
```

~1,602 branches. Called for: opcode fetch, ModRM byte, `read2()` (two calls), instruction argument resolution, memory-operand instructions. Estimated 10+ calls per tick = **~16,000 `style()` evaluations** for memory access alone.

**2. Broadcast write — O(N) per store**

After computing a result, all 1,583 variables each check `style(--addrDestA: {their_address})`. Only 1–2 match. The rest evaluate the comparison and discard. That's **~3,166 `style()` evaluations** per tick, of which ~3,164 are wasted.

**3. Bitwise operations — 32 intermediate variables per call**

XOR/AND/OR/NOT decompose 16-bit values into individual bits (32 `mod`/`round` operations), operate bit-by-bit, and reconstruct (16 multiplications + additions). Native equivalent: one CPU instruction.

### Total Per-Tick Cost

| Operation | style() evals | Notes |
|-----------|--------------|-------|
| readMem() | ~16,000 | 10 calls × 1,600 branches |
| Broadcast write | ~3,166 | 1,583 vars × 2 destinations |
| Instruction dispatch | ~360 | getDest/getVal/argTypes over ~60 instructions |
| ModRM decode | ~80 | Address mode resolution |
| Misc conditionals | ~200 | Various if() in functions |
| **Total** | **~20,000** | Per style recalculation |

Chrome does 4 style recalculations per instruction (4 clock phases). Total: **~80,000 `style()` evaluations per x86 instruction**.

Plus ~3,200 `var()` resolutions, ~1,800 `calc()` evaluations, and ~50-100 `@function` calls per recalculation.

## Where the Speedup Comes From

The speedup is not primarily from eliminating rendering (layout/paint/compositing are cheap for this DOM). It's from **replacing CSS's O(N) evaluation patterns with O(1) data structures**:

| CSS pattern | Chrome cost | Engine cost | Speedup |
|-------------|------------|-------------|---------|
| `readMem()` 1,600-branch if | ~800 comparisons/call (avg) | Array index: 1 lookup | ~800x per call |
| Broadcast write (1,583 vars) | 3,166 comparisons | Direct store: 1 write | ~1,500x per write |
| 4-phase triple buffer | 4 full recalculations | 1 evaluation pass | 4x |
| `@function` call overhead | CSS function resolution | Native function call | 10-50x per call |
| Bitwise ops (XOR etc.) | 32 mod/round + 16 mul | 1 native instruction | ~100x per call |

These don't multiply — the total speedup depends on which operations dominate wall-clock time in Chrome. The `readMem()` optimisation dominates because it's the most-called and most-expensive operation.

### Estimated Speedup Range

- **Conservative (10-20x):** Chrome has some internal optimisation for `if(style())` (short-circuit, partial caching). readMem is still the bottleneck but not as catastrophically expensive as worst-case analysis suggests.
- **Expected (50-200x):** Chrome evaluates `if(style())` naively (linear scan, no hash dispatch). The engine's data structure replacement provides full theoretical speedup on the dominant operations.
- **Optimistic (500x+):** Chrome's `@function` implementation has significant per-call overhead (scope creation, parameter binding) that compounds with the deep call nesting (readMem → read2 → getInstArg → ...).

The 10-50x target from the original spec is achievable with high confidence.

## Architecture

### High-Level Design

```
Browser (Chrome)
├── Loads x86css.html normally
├── CSS parsed by Chrome (for rendering)
├── JS extracts CSS text, sends to Worker
│
Web Worker (WASM)
├── calc(ify) — JIT Compiler (Rust → WASM)
│   ├── Parser: extract @function, @property, if(), calc() from CSS
│   ├── Compiler: recognise patterns, build optimised evaluator
│   │   ├── if(style()) chains → hash-map dispatch tables
│   │   ├── Broadcast write pattern → direct-addressed state store
│   │   ├── readMem → flat memory array
│   │   ├── Bitwise decomposition → native bitwise ops (optional)
│   │   └── @function bodies → compiled Rust closures
│   ├── Evaluator: run ticks against state
│   └── State: registers + memory + flags (flat arrays)
│
├── Returns: changed property values (diff only)
│
Browser Main Thread
├── Applies property values to DOM: element.style.setProperty(...)
├── Chrome renders the visual state (once per batch)
└── Requests next batch of ticks from Worker
```

### The Fairness Contract

The engine operates under the same contract as any browser engine optimisation:

1. **Input:** Standard CSS (no modifications, no annotations)
2. **Output:** Computed property values identical to what Chrome would produce
3. **Rendering:** Done by Chrome from the computed values — the visual result is real CSS rendering
4. **Optimisation boundary:** Everything between parsing and computed values is internal

The engine is not "cheating" any more than V8's JIT compiler is cheating compared to an interpreter. It produces the same results faster by recognising patterns and using better data structures.

### Parser

Built on Servo's `cssparser` crate for tokenisation. Custom parsing layer on top for the computational subset:

**Must parse:**
- `@property` declarations (type, initial value, inheritance)
- `@function` definitions (parameters, local variables, `result` descriptor)
- `if(style(...): ...; else: ...)` conditional values
- `calc()`, `mod()`, `round()`, `min()`, `max()`, `clamp()`, `pow()`, `sign()`, `abs()`
- `var()` references with fallbacks
- `@keyframes` (for clock detection)
- `@container style(...)` rules (for clock mechanism)
- `content` property and `counter()` (for display output)

**Does not need to parse:**
- Selectors (beyond the handful used for `.cpu`, `.clock`, `.screen`)
- Layout/visual properties
- Media queries
- Pseudo-elements/pseudo-classes (except `:hover:active` for keyboard)

### Pattern Compiler

This is the core innovation. After parsing, the compiler recognises computational patterns and replaces them with efficient equivalents:

**Pattern 1: Large `if(style())` dispatch → hash map**

Detects: an `if()` expression with many `style(--param: N)` branches where `--param` is a function parameter or computed property.

Replaces: the linear condition scan with a hash-map lookup keyed on the parameter value. For integer-valued parameters (which is all of them in x86CSS), this is an array index.

This single optimisation transforms `readMem()` from O(1602) to O(1).

**Pattern 2: Broadcast write → direct store**

Detects: a set of property declarations that all have the form `if(style(--dest: {addr}): value; else: previous)` where `--dest` is the same computed property across all declarations and `{addr}` is unique per declaration.

Replaces: evaluate `--dest` once, write `value` to `state[dest]` directly.

This transforms the per-tick state update from O(1583) comparisons to O(1) write.

**Pattern 3: Bit decomposition → native bitwise (optional)**

Detects: a function that extracts individual bits via `mod(round(down, x / 2^n), 2)`, operates on each bit independently, and reconstructs via weighted sum.

Replaces: native bitwise operation (XOR, AND, OR, NOT).

This is the most aggressive optimisation and the most debatable for "fairness." It's optional — the engine should work without it, just slower. Including it is equivalent to a browser engine recognising that a chain of arithmetic operations implements a bitwise operation and emitting a single instruction.

**Pattern 4: @function inlining (optional)**

Detects: small functions called frequently (e.g., `--int()`, `--lowerBytes()`, `--rightShift()`).

Replaces: inline the function body at the call site, eliminating function call overhead.

### Evaluator

The evaluator runs a compiled tick function against a flat state representation:

```rust
struct State {
    registers: [i32; 14],    // AX, CX, DX, BX, SP, BP, SI, DI, IP, ES, CS, SS, DS, flags
    memory: [u8; MEM_SIZE],  // flat byte array
    text_buffer: String,     // display output
    keyboard: u8,            // current key input
}

fn tick(state: &mut State, program: &CompiledProgram) {
    // 1. Decode
    let opcode = state.read_mem(state.registers[IP]);
    let inst_id = program.decode_table[opcode];

    // 2. Parse ModRM if needed
    let modrm = if program.instructions[inst_id].has_modrm {
        state.read_mem(state.registers[IP] + 1)
    } else { 0 };

    // 3. Resolve arguments
    let (arg1, arg2) = resolve_args(state, program, inst_id, modrm);

    // 4. Execute instruction (calls compiled D-xxx and V-xxx functions)
    let (dest, value) = program.instructions[inst_id].execute(state, arg1, arg2);

    // 5. Write result
    state.write(dest, value);

    // 6. Advance IP (unless jump)
    if !jumped { state.registers[IP] += inst_len; }
}
```

This is what the CSS *does*, expressed as what it *means*. Every line corresponds to a section of the CSS evaluation pipeline. The engine isn't bypassing the CSS — it's evaluating it efficiently.

### Browser Integration (Primary Deliverable)

```javascript
// Main thread
const worker = new Worker('calcify-worker.js');
const cpu = document.querySelector('.cpu');

// Extract CSS from the page
const cssText = [...document.styleSheets]
  .flatMap(s => [...s.cssRules].map(r => r.cssText))
  .join('\n');

worker.postMessage({ type: 'init', css: cssText });

worker.onmessage = ({ data }) => {
  if (data.type === 'tick-result') {
    // Apply changed properties to DOM
    for (const [name, value] of data.changes) {
      cpu.style.setProperty(name, value);
    }
    // Chrome renders from these values
    // Request next batch
    worker.postMessage({ type: 'tick', count: BATCH_SIZE });
  }
};
```

The key performance parameter is `BATCH_SIZE` — how many ticks to run before updating the DOM. Higher batches = faster computation, choppier display. For interactive programs: 5-10. For benchmarking: 1000+.

**Alternative: Native process over WebSocket.** For maximum performance, the engine runs as a native Rust binary communicating with the browser via WebSocket or Chrome extension native messaging. Same interface, no WASM overhead. This is a deployment option, not a different architecture.

### Conformance Testing

The engine must produce identical computed values to Chrome. Testing approach:

1. **Chrome baseline capture:** Use Puppeteer to run x86CSS in Chrome. After each tick, extract all register values and changed memory via `getComputedStyle()`. Save as JSON snapshots.

2. **Engine comparison:** Run the same CSS through the engine for the same number of ticks. Compare state after each tick against Chrome snapshots.

3. **Divergence debugging:** When states differ, binary-search to find the first divergent tick, then the first divergent property, then trace the evaluation chain to find the semantic mismatch.

This is the first thing to build, even before the engine, because it defines the correctness criterion and validates that state extraction from Chrome actually works.

## The Doom Target: doom.css

Doom8088 (https://github.com/FrenkelS/Doom8088) is a port of Doom for 16-bit processors — the 8088 and 286. It uses `gcc-ia16`, the same compiler x86CSS uses. It targets the same instruction set. It has a text-mode display option (40×25, 80×50, 16 colours) that maps directly to x86CSS's character-based output.

This means Doom can run in CSS without extending x86CSS to a 386. The path is: compile Doom8088 → feed the binary to `build_css.py` → run it through the compute engine. Two pieces of work, two headlines: "Doom runs in CSS" and then "...and now it's playable."

### What Doom8088 Needs

**Memory:** ~560KB conventional minimum. That's ~573,000 custom properties — the generated CSS would be enormous. Chrome will never evaluate this natively (the `readMem()` if-chain would have 573,000 branches). The compute engine replaces this with `memory[addr]`, so it doesn't care.

**WAD file:** Doom needs its game data. Doom8088 uses a preprocessed WAD (DOOM16DT.WAD) generated by jWadUtil from the shareware DOOM1.WAD. The text mode WAD is **1,357,504 bytes** (~1.3MB) — sprites and walls are reduced-detail, textures converted to 16-colour dithered text mode format. This is loaded at runtime via DOS file I/O (`fopen`/`fread`), not embedded in the program binary. The engine intercepts DOS file I/O interrupts and serves the WAD from its own buffer — the WAD lives outside the CSS and the emulated address space, just like a hard drive is separate from the CPU.

**Display:** Text mode writes character+attribute pairs to video memory at 0xB800:0000. For x86CSS: map this region to display properties. 40×25 = 2,000 bytes. The engine reads the framebuffer region and pushes display updates only when it changes — the smart frameskip optimisation.

**Missing instructions:** x86CSS doesn't implement all 8086 instructions — Rebane only implemented what her demo programs needed. Doom8088 compiled with `gcc-ia16 -march=i8088 -mcmodel=medium` requires 17 unimplemented instructions across 1,137 occurrences:

| Instruction | Occurrences | What it does | Difficulty |
|------------|-------------|-------------|-----------|
| RETF | 370 | Far return (pop IP and CS) | Easy but critical — every function return |
| SAR | 123 | Arithmetic right shift (preserves sign) | Easy — like SHR but sign-extending |
| RCL | 117 | Rotate left through carry | Moderate — needs carry flag |
| LES | 111 | Load far pointer into ES:reg | Moderate — segment register write |
| LDS | 106 | Load far pointer into DS:reg | Moderate — segment register write |
| NEG | 77 | Two's complement negate | Easy — `0 - value` |
| CLC | 65 | Clear carry flag | Trivial |
| RCR | 58 | Rotate right through carry | Moderate — needs carry flag |
| XLAT | 52 | Table lookup: AL = [BX+AL] | Easy |
| STOSB | 25 | Store byte AL to [ES:DI], increment DI | Easy |
| LAHF | 16 | Load flags into AH | Easy |
| STI/CLI | 12 | Enable/disable interrupts | Stub (no hardware interrupts) |
| IRET | 2 | Return from interrupt | Easy — pop IP, CS, flags |
| PUSHF/POPF | 2 | Push/pop flags register | Easy |
| JCXZ | 1 | Jump if CX is zero | Trivial |

Additionally, `lcall` (far CALL) appears 1,195 times. x86CSS implements CALL but only as a near call (pushes IP only). Far calls must push CS:IP (4 bytes). This is the most critical fix — every function call in Doom8088 is a far call due to the medium memory model. Similarly, `ljmp` (far JMP) appears 72 times.

Port I/O instructions IN (6) and OUT (18) also appear for hardware access (display page flipping, timer). These need handling in the engine's I/O layer.

No REP/REPZ/REPNZ prefixes appear anywhere in the Doom8088 disassembly — this is a nice surprise, as string repeat operations were expected to be needed.

**BIOS/DOS interface:** Doom8088 uses `int 10h` (BIOS video — mode set and cursor hiding at startup only), `int 21h` (DOS file I/O for WAD loading), `int 16h` (BIOS keyboard). Port I/O appears 24 times in the disassembly: `OUT` (18) and `IN` (6) for CRT register page flipping (ports 0x3D4/0x3D5), CGA palette (0x3D9), and timer. The engine stubs BIOS calls, intercepts DOS file I/O, and handles the small set of port addresses.

**Far CALL/RET — the critical fix:** x86CSS implements CALL as a near call (pushes only IP, 2 bytes). Doom8088 has 1,195 far calls (`lcall`) which push CS:IP (4 bytes), and 370 far returns (RETF) which pop both. This is the most impactful missing feature — every function call in Doom8088 is a far call. x86CSS's CALL implementation must be extended to detect far vs near calls and handle CS accordingly.

### Why calc(ify) Is Essential for doom.css

Without calc(ify), doom.css is purely theoretical. Chrome would need to evaluate a `readMem()` with ~573,000 branches, ~10 times per tick. Even if Chrome short-circuits, that's millions of `style()` evaluations per tick. A single frame of Doom might take hours.

With calc(ify): `memory[addr]` is a single array index lookup regardless of memory size. calc(ify) makes doom.css go from "provably possible but never observable" to "here's a video of me playing it."

### Estimated Framerate

Doom8088 achieves ~10-15fps on a real 286 at 12MHz. The compute engine on a modern CPU should far exceed a 286's throughput.

| Configuration | Estimated fps | Notes |
|--------------|--------------|-------|
| Native engine, interpreted (5700X3D) | 10-25 fps | Baseline, single-core |
| Native engine + background JIT | 20-60 fps | 5-15x speedup on hot rendering loops |
| WASM in-browser, interpreted | 2-10 fps | Recognisable slideshow |
| WASM in-browser + JIT | 5-20 fps | Potentially playable in-browser |
| Chrome native (no engine) | <0.001 fps | Hours per frame, if it evaluates at all |

The text mode display helps enormously — 40×25 characters is trivial to render, unlike a pixel framebuffer.

## Development Phases

The project has two parallel tracks that converge: the **calc(ify) track** (make x86CSS fast) and the **doom.css track** (get Doom running in x86CSS). calc(ify) is the technical contribution. doom.css is the demo that makes people care.

### Phase 0: Baselines

- Run x86CSS's Fibonacci demo in Chrome with the JS clock. Measure ticks/second.
- Profile Chrome DevTools: time breakdown per style recalculation
- Build the Puppeteer state-extraction harness for conformance testing
- Capture baseline state snapshots (1000+ ticks)
- Compile Doom8088 in text mode with `gcc-ia16`, check the resulting binary size and instruction usage
- Identify exact instruction gaps between Doom8088's needs and x86CSS's implementation

**Deliverable:** Chrome baseline number + Doom8088 binary + instruction gap analysis.

### Phase 1: Parser and Pattern Recognition

- Parse x86CSS with `cssparser` + custom computational-subset parser
- Extract all `@function` definitions, `@property` declarations, `if()` expressions
- Identify and catalogue the computational patterns (readMem dispatch, broadcast write, bitwise decomposition, triple-buffer clock)
- Output: intermediate representation + pattern analysis report

**Deliverable:** A tool that parses x86CSS and prints: "found readMem() with 1,602 branches, found broadcast write over 1,583 variables, found 4 bitwise decomposition functions, found 4-phase clock."

### Phase 2: Core Evaluator

- Implement the compiled evaluator (pattern → efficient data structure)
- readMem → array-indexed lookup
- Broadcast write → direct `state[addr] = value`
- `@function` evaluation with parameter binding
- `calc()` / `mod()` / `round()` / `min()` / `max()` / `pow()` evaluation
- `if(style())` evaluation (hash dispatch for large chains, linear for small ones)
- Build the tick loop
- Verify output matches Chrome snapshots from Phase 0

**Deliverable:** A CLI tool that takes x86CSS's CSS, runs N ticks, prints register state. Output matches Chrome snapshots for 1000+ ticks of Fibonacci.

### Phase 3: WASM + Browser Integration

- Compile the evaluator to WASM
- Build the Web Worker wrapper and JS bridge
- Implement property-diff protocol (only send changed values to DOM)
- Smart frameskip: only push DOM updates when display-relevant properties change
- Batch-size configuration (ticks per DOM update)
- Keyboard input forwarding
- Benchmark: in-browser WASM vs Chrome native

**Deliverable:** x86CSS Fibonacci running in Chrome, powered by the WASM engine. Side-by-side speed comparison with native Chrome evaluation.

### Phase 4: Doom8088 Port

*This can start in parallel with Phases 1-3.*

- Fork Doom8088, configure for text mode 40×25, no sound, no EMS/XMS
- Replace BIOS `int 10h` video calls with stubs (mode already set by engine)
- Replace DOS `int 21h` file I/O with x86CSS I/O address conventions (or handle in engine via interrupt interception)
- Replace `int 16h` keyboard reads with x86CSS keyboard I/O address
- Remove hand-written ASM files (`m_fixed.asm`, `i_vtexta.asm`, `z_xms.asm`), use C equivalents
- Compile with `gcc-ia16 -march=i8088 -mcmodel=medium`
- Implement the 17 missing instructions in x86CSS, prioritised by occurrence count:
  - **Critical (1,565 occurrences):** Far CALL (extend existing CALL to push CS:IP), RETF (370)
  - **High (417 occurrences):** SAR (123), RCL (117), LES (111), LDS (106)
  - **Medium (252 occurrences):** NEG (77), CLC (65), RCR (58), XLAT (52)
  - **Low (63 occurrences):** STOSB (25), LAHF (16), STI/CLI (12), IRET (2), PUSHF/POPF (2), JCXZ (1)
- Handle port I/O: OUT (18 occurrences) and IN (6) for CRT registers and timer
- Extend `build_css.py`: bump memory to ~640KB conventional + 32KB video RAM at 0xB8000, add segment register initialisation
- Generate the CSS: `build_css.py` → `doom-x86css.html` (~100-150MB)
- Run through the compute engine, verify Doom boots to title screen

**Deliverable:** Doom's title screen rendering in CSS via the compute engine. Screenshot or it didn't happen.

### Phase 5: Playable Doom Demo

- Optimise the engine for the Doom-scale CSS (~573K memory properties)
- Implement native bitwise operations (XOR/AND/OR/NOT pattern recognition)
- Implement `@function` inlining for hot paths
- Add text-mode display rendering to the browser integration (character grid with colour attributes)
- Add keyboard input mapping (WASD/arrow keys → Doom controls)
- Tune batch size for best framerate vs display responsiveness
- Build native-over-WebSocket mode for maximum performance
- Record playthrough video

**Deliverable:** A video of someone playing Doom Episode 1, Level 1 in CSS in a browser. Playable means: responsive to keyboard input, recognisable 3D rendering, enemies visible and killable. Framerate is whatever the engine achieves — even 2fps counts.

### Phase 6: Multi-Core Performance Engineering

Profile the evaluator. If Doom is under target framerate, the bottleneck is single-threaded tick throughput. The evaluator is pinned to one core; the other cores are idle. Use them.

**Core allocation:**

| Core | Role | What it does |
|------|------|-------------|
| 0 | Evaluator | The tick loop. Fully sequential, runs flat out. Untouchable. |
| 1 | Display thread | Reads video memory region from shared state, converts to character+colour data, packages for browser via SharedArrayBuffer. Only wakes when video memory is dirty. |
| 2 | Background JIT compiler | Scans ahead in program ROM, identifies basic blocks (branch-free instruction sequences), pre-compiles them into optimised native functions. When the evaluator's IP enters a pre-compiled block, it runs the compiled version instead of interpreting. |
| 3 | Input thread | Reads keyboard state from the browser via SharedArrayBuffer, writes to keyboard I/O address in shared state. Keeps the evaluator's hot loop free of any browser API calls. |

The display and input threads are straightforward (SharedArrayBuffer + Atomics for synchronisation). The background JIT is the ambitious one.

**Background JIT design:**

The JIT thread has read access to program ROM. It scans from common entry points (the main game loop, the renderer entry, known hot functions) and identifies basic blocks — sequences of instructions ending at a branch, call, or return. For each block, it compiles an optimised native function that:

- Eliminates per-instruction opcode lookup (the block's instructions are known)
- Eliminates per-instruction handler dispatch (operations are inlined)
- Keeps intermediate values in local variables (which the Rust compiler maps to CPU registers) instead of writing back to emulated registers between instructions
- Folds sequential memory accesses (e.g., `read2` = two byte reads = one 16-bit read in the compiled version)
- Replaces bitwise decomposition patterns with native bitwise ops within the block

The evaluator checks on each tick: is the current IP in a compiled block? If yes, call the compiled function, advance IP past the block, continue. If no, interpret normally. The JIT thread never blocks the evaluator — if a block isn't compiled yet, interpretation continues. Compiled blocks are stored in a lock-free hash map keyed by IP address.

**What this gains:**

For Doom's inner rendering loops (drawing wall columns, filling spans), a basic block might be 5-15 instructions. Interpreted, each instruction costs ~50-60ns (decode + execute + writeback). Compiled into a single native function, the whole block might cost 30-50ns total — the overhead is gone and the Rust compiler optimises across instruction boundaries. That's a 5-15x speedup on hot loops. Since rendering dominates Doom's frame time, this could be the difference between 5fps and 20fps.

**What this doesn't help:**

Branchy code (game logic, AI, menu systems) has short basic blocks and benefits less. But that code is also a small fraction of frame time.

**Invalidation:**

Doom doesn't do self-modifying code, so compiled blocks are valid forever. For correctness in the general case, the engine would watch for writes to code regions and invalidate compiled blocks — but this can be a future concern, not a launch requirement.

**Implementation complexity:**

This is the most complex part of the engine and should only be attempted after the interpreted evaluator is working and profiled. If the interpreter already achieves target framerate, skip this entirely. Build it only if profiling shows the tick loop is the bottleneck and basic block compilation would help.

**WASM SIMD:**

No REP MOVSB/STOSB appears in the Doom8088 disassembly, so the originally planned SIMD optimisation for block memory operations is unnecessary. SIMD could still help with display buffer conversion (converting 2,000 char+attr byte pairs to colours in bulk), but this is a minor optimisation on an already-cheap operation.

### Phase 7: Polish and Release

- Package as a standalone webpage — one URL, no install, Doom plays in CSS in Chrome
- Write-up explaining the full pipeline (Doom → gcc-ia16 → CSS → compute engine → browser)
- Performance comparison: Chrome native vs WASM engine vs native engine
- Open-source calc(ify), doom.css (the Doom8088 fork), and the generated CSS

**Deployment options:**

Option A: Serve the full ~100MB CSS file. Slower to load but funnier — the loading screen shows a progress bar with literary comparisons as milestones:

> "Downloading 100MB of CSS to run Doom..."
>
> ████░░░░░░░░░░░░ 5MB — You've downloaded 1 Shakespeare
> ████████░░░░░░░░ 28MB — 5 complete works of Shakespeare
> ██████████████░░ 83MB — You've now downloaded more CSS than 15 complete Harry Potter series
> ████████████████ 100MB — 18 Shakespeares. 42 copies of the original Doom executable. 83 copies of Doom's own source code. 375,000 tweets. 70 reams of paper if printed. 73 days to read aloud.
>
> Doom is ready.

Option B: Generate the CSS on-the-fly in the browser from the small source files (binary ~50KB + base template ~69KB + instruction table ~84KB + WAD ~1.3MB). Total download ~2MB. More practical, less theatrical.

Option A is the demo. Option B is for anyone who actually wants to use it.

## Open Questions (Updated)

### Resolved by Source Analysis

1. ~~Container query scoping complexity~~ → Only 3 container queries, all trivial. Non-issue.
2. ~~Memory representation~~ → One `@property` per byte, named `--m{N}`. One property per register, named `--{REG}`. Registers use negative addresses (-1 for AX, -2 for CX, etc.). Split registers (AH/AL) have additional addresses (-21/-31).
3. ~~Multi-byte values~~ → Registers store 16-bit integers in a single property. `read2()` reads two consecutive bytes and combines: `low + high * 256`.
4. ~~Clock mechanism~~ → 4-phase triple buffer with two animations. JS clock overrides via `!important`. Fully understood.
5. ~~Parallelism profile~~ → The instruction stream is fundamentally sequential — no instruction-level parallelism within the tick loop. However, display rendering, input polling, and background JIT compilation can run on separate cores. GPU acceleration is not viable.

### Resolved by Doom8088 Compilation & Research

6. ~~Doom8088 gcc-ia16 output~~ → Compiled and disassembled. 17 unimplemented instructions, 1,137 occurrences. RETF (370), SAR (123), RCL (117), LES (111), LDS (106) are the big ones. No REP prefix appears anywhere. Far CALL (`lcall`, 1,195 occurrences) is the most critical gap — x86CSS's CALL only does near calls.
7. ~~WAD size~~ → DOOM16DT.WAD (text mode) is exactly 1,357,504 bytes (~1.3MB). Generated by jWadUtil from shareware DOOM1.WAD. Small enough for the engine to load separately and serve through intercepted DOS file I/O.
8. ~~Segment addressing~~ → Confirmed: x86CSS uses flat addressing. Segment registers exist as storage but are never used in address calculation. Engine does `segment * 16 + offset` internally before the array index. One multiply, one add. No CSS changes needed.
9. ~~cssparser crate support~~ → v0.37.0 (March 2026) tokenises `if()` as `Token::Function("if")` and `@function` as `Token::AtKeyword("function")` correctly. No semantic parsing — we build all feature-specific parsing on top using the `AtRuleParser` trait and `parse_nested_block()`. The crate also has `look_for_arbitrary_substitution_functions()` (v0.36.0+) for detecting `if()` in values.

### Still Open

1. **Chrome's actual ticks/second.** The spec claims ~1/sec. This needs measurement. If it's actually 10-30/sec, the achievable speedup is proportionally less dramatic (but still worthwhile).

2. **How does Chrome implement `if(style())` internally?** Linear scan? Short-circuit? Some caching? This determines the baseline cost and therefore the realistic speedup. We've inferred it's likely naive, but profiling would confirm.

3. **`@function` call overhead in Chrome.** Is there measurable per-call overhead (scope creation, parameter binding)? Or does Chrome inline/optimise small functions? With ~50-100 function calls per tick and some calls nested 4-5 deep, this could be a significant fraction of Chrome's total cost.

4. **Spec stability.** `if()` and `@function` are new. If Chrome changes their semantics, x86CSS may update to match, and the engine would need to follow. Pragmatic approach: pin to a specific Chrome version and x86CSS commit.

5. **SharedArrayBuffer for Worker communication.** Could the state live in shared memory between the Worker and main thread, avoiding the serialisation cost of `postMessage`? This would make small batch sizes (1-5 ticks) much more efficient, improving display smoothness. Requires COOP/COEP headers.

## Success Criteria

1. **Correctness:** calc(ify) produces identical computed property values to Chrome for x86CSS, verified by automated comparison over 1000+ ticks
2. **Speed:** At least 10x faster than Chrome's native style resolver, measured in-browser (WASM Worker vs Chrome's `getComputedStyle()` loop)
3. **In-browser:** The demo runs as a normal webpage — CSS renders visually, keyboard input works, calc(ify) is invisible to the user except for speed
4. **Generality:** The parser and pattern compiler are not hardcoded to x86CSS — they would work on other computational CSS using the same feature set
5. **doom.css:** Doom8088 runs in CSS via calc(ify), playable at ≥1fps with keyboard input, rendering recognisable 3D gameplay in text mode

## Prior Art and References

- Lyra Rebane's x86CSS: https://lyra.horse/x86css/ (primary benchmark)
- x86CSS source: https://github.com/rebane2001/x86CSS
- Jane Ori's CSS CPU Hack: https://dev.to/janeori/expert-css-the-cpu-hack-4ddj (the foundational technique)
- Doom8088: https://github.com/FrenkelS/Doom8088 (16-bit Doom port, same toolchain as x86CSS)
- RealDOOM: https://github.com/sqpat/RealDOOM (alternative 16-bit Doom port)
- CSS `if()` spec: https://drafts.csswg.org/css-conditional-values/
- CSS `@function` spec: https://drafts.csswg.org/css-mixins/
- Servo's cssparser crate: https://github.com/servo/rust-cssparser
- Clement Cherlin's CSS Turing completeness proof: https://github.com/Mooninaut/css-is-turing-complete

## Appendix A: Jane Ori's CPU Hack

Jane Ori's 2023 blog post "Expert CSS: The CPU Hack" established the core technique that makes CSS computation possible:

1. **Animation state trumps selector state** — property values set by a running animation override all other sources
2. **Keyframe `var()` references are cached** — resolved when the animation starts or properties change, not when the source property changes
3. **Changing animation properties breaks the cache** — toggling `animation-play-state` forces re-resolution
4. **A paused animation's values are frozen** — reading from a paused animation doesn't trigger recomputation

From these observations: use two animations (`store` and `execute`). One captures computed state, the other reads it as input. Toggle them alternately. This creates a feedback loop that circumvents CSS's cycle detection: output → cache → input → compute → output. Each toggle = one clock tick.

Ori's version required `:hover` to drive the clock. Rebane replaced this with a self-sustaining CSS animation + container query clock that requires no user interaction.

## Appendix B: x86CSS Source Structure

```
base_template.html    69KB   1,670 lines   Handwritten CSS: CPU logic, all @functions
build_css.py          13KB     362 lines   Generates repetitive CSS: memory, instruction tables
x86css.html          809KB  13,889 lines   Generated output (the thing that runs)
x86-instructions-rebane.json  84KB        Instruction set reference
c/main.c              4.5KB               Demo program (Fibonacci/Horsle)
```

Key statistics for the generated `x86css.html`:
- 1,585 `@property` declarations
- 130 `@function` definitions
- ~10,576 `style()` conditions
- ~16,918 `var()` references
- ~1,732 `calc()` expressions
- 3 `@keyframes` rules
- 3 functional `@container` rules

## Appendix C: The Broadcast Write Pattern (Detailed)

The most unusual pattern in x86CSS. After the CPU computes a destination address and value:

```css
/* build_css.py generates this for EVERY state variable: */
--m0:   if(style(--addrDestA:0):var(--addrValA1);
           style(--addrDestB:0):var(--addrValB);
           else:var(--__1m0));
--m1:   if(style(--addrDestA:1):var(--addrValA1);
           style(--addrDestB:1):var(--addrValB);
           else:var(--__1m1));
/* ... 1,581 more variables ... */
--m1535: if(style(--addrDestA:1535):var(--addrValA1);
            style(--addrDestB:1535):var(--addrValB);
            else:var(--__1m1535));
```

This is an addressed write implemented as a broadcast. The CPU says "write value X to address Y" and every memory location independently asks "am I address Y?" It's O(N) where N is the total number of state variables, and exactly 1-2 of them match.

The engine replaces this with: `state.memory[dest_addr] = value` — O(1).

Split registers (AX with AH/AL) have additional logic:

```css
--AX: if(
  style(--__1IP:0x2006): var(--keyboard, 0);     /* special: keyboard input */
  style(--addrDestA:-1): var(--addrValA);          /* write to AX */
  style(--addrDestB:-1): var(--addrValB);
  style(--addrDestA:-21): calc(addrValA*256 + AL); /* write to AH: merge */
  style(--addrDestB:-21): calc(addrValB*256 + AL);
  style(--addrDestA:-31): calc(AH*256 + lowerBytes(addrValA, 8)); /* write to AL */
  style(--addrDestB:-31): calc(AH*256 + lowerBytes(addrValB, 8));
  else: var(--__1AX)                               /* no write: keep old */
);
```

The engine handles this with explicit register accessors that manage byte/word views.
