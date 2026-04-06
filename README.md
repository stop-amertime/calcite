# calc(ify)

A JIT compiler for computational CSS. Parses real CSS, recognises computational
patterns, and compiles them into efficient native operations.

Primary target: running [x86CSS](https://lyra.horse/x86css/) (a CSS-based x86
8088 emulator that runs Doom) faster than Chrome's native style resolver.

## How it works

x86CSS encodes an entire x86 8088 CPU in CSS custom properties. Each "tick" of
the CPU is one recalculation of ~3000 CSS properties. Chrome's style engine
treats these as opaque strings and evaluates them via O(n) pattern matching.

Calcify parses the CSS once, recognises the computational patterns, and replaces
them with efficient native operations:

| Pattern | CSS cost | Calcify cost |
|---|---|---|
| `if(style(--key: 0): ...; style(--key: 1): ...; ...)` (1500+ branches) | O(n) linear scan | O(1) HashMap lookup |
| Per-cell memory writes (`--m0: if(style(--dest: 0): val; else: keep)`) | O(cells) per tick | O(1) direct store |
| Triple-buffer pipeline (`--__0AX`, `--__1AX`, `--__2AX`) | 3x property copies | Eliminated (mutable state) |

## Project layout

```
crates/
  calcify-core/    Core engine: parser, pattern compiler, evaluator, state
  calcify-cli/     CLI tool for running CSS through the engine
  calcify-wasm/    WASM bindings for browser Web Worker
web/
  index.html           Browser UI
  calcify-worker.js    JS Web Worker bridge
```

## Building and testing

```sh
cargo check --workspace         # typecheck
cargo test --workspace          # run tests
cargo clippy --workspace        # lint
cargo fmt --all                 # format
```

### WASM build (requires wasm-pack)

```sh
wasm-pack build crates/calcify-wasm --target web --out-dir ../../web/pkg
```

### Benchmarks (requires fixture file)

```sh
cargo bench --bench x86css
```

Benchmarks require `tests/fixtures/x86css-main.css` (the compiled x86CSS
stylesheet). See `docs/benchmarking.md` for details on Chrome comparison
methodology.

## Architecture

### Parser (`parser/`)

CSS tokenisation via the `cssparser` crate. Custom parsing for `@function`,
`@property`, `if(style())`, `calc()`, `var()`, and all CSS math functions.

### Pattern recognition (`pattern/`)

Detects computational patterns at compile time:

- **Dispatch tables**: Large `if(style(--prop: N))` chains in `@function`
  results are converted to `HashMap<i64, Expr>` for O(1) lookup.
- **Broadcast writes**: Per-cell memory assignments
  (`--mN: if(style(--dest: N): val)`) are converted to direct state writes.
  Supports word-write spillover (16-bit writes across adjacent bytes).

### Evaluator (`eval.rs`)

The tick loop. Evaluates compiled programs against flat mutable state.
Reuses allocations across ticks for minimal overhead.

### State (`state.rs`)

Flat machine state replacing CSS's triple-buffered custom properties.
Uses x86CSS's unified address space: negative addresses = registers,
positive addresses = memory bytes.

## License

MIT
