use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use std::path::PathBuf;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod calcite_logo;
mod cssdos_logo;
mod menu;

use calcite_core::script_spec as watch_spec;

/// calc(ite) — JIT compiler for computational CSS.
///
/// Parses CSS files, recognises computational patterns, and evaluates
/// them efficiently. Primary target: running x86CSS faster than Chrome.
#[derive(Parser, Debug)]
#[command(name = "calcite", version, about)]
struct Cli {
    /// Path to the CSS file to evaluate.
    ///
    /// If omitted, launches the interactive CSS-DOS program picker.
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Number of ticks to run. Omit for unlimited (interactive).
    #[arg(short = 'n', long)]
    ticks: Option<u32>,

    /// Print register state after each tick.
    #[arg(short, long)]
    verbose: bool,

    /// Only parse and analyse patterns (don't evaluate).
    #[arg(long)]
    parse_only: bool,

    /// Render the screen every N ticks.
    ///
    /// Reads BDA 0x0449 to determine the current video mode and renders
    /// accordingly: text modes show an in-terminal ANSI screen; Mode 13h
    /// shows a status line. Default: 500. Set to 0 for render-on-exit only.
    #[arg(long, value_name = "N", default_value = "500")]
    screen_interval: u32,

    /// Ticks per batch inside the interactive loop.
    ///
    /// The interactive loop polls keyboard, checks halt, and re-renders between
    /// batches. At 4.77 MHz 8086 timing, 50000 ticks ≈ 10.5 ms ≈ 95 fps, so
    /// keyboard latency at this batch size is imperceptible. Smaller values
    /// give finer-grained input at the cost of throughput.
    #[arg(long, value_name = "N", default_value = "50000")]
    interactive_batch: u32,

    /// Throttle execution to this multiple of real 8086 speed (4.77 MHz).
    ///
    /// 1.0 = real 8086 timing (default). 2.0 = double speed. 0 = unlimited
    /// (run as fast as the machine can). Throttling uses the 8086 cycleCount,
    /// not wall-ticks, so programs with variable cycles-per-instruction feel
    /// correct.
    #[arg(long, value_name = "MULT", default_value = "1.0")]
    speed: f64,

    /// Dump all computed slot values at a specific tick for debugging.
    ///
    /// Runs to the specified tick, then prints every property name and its
    /// computed value. Useful for diagnosing divergences found by compare.mjs.
    #[arg(long, value_name = "TICK")]
    dump_tick: Option<u32>,

    /// Dump state at multiple ticks in a single run.
    ///
    /// Comma-separated list of ticks (e.g. `--dump-ticks=100000,200000,300000`).
    /// Compiles once, runs forward, dumps at each tick. Much faster than
    /// invoking the CLI multiple times for sampling a property over time.
    /// Combine with --sample-cells for compact per-tick output.
    #[arg(long, value_name = "LIST", value_delimiter = ',')]
    dump_ticks: Vec<u32>,

    /// Only print these memory cells in dumps (comma-separated packed-cell indices).
    ///
    /// Example: `--sample-cells=2344,632` prints just those cells. Combine with
    /// --dump-tick or --dump-ticks to track specific memory cells across time.
    /// Without this flag, dumps include all state-var and cell values.
    #[arg(long, value_name = "LIST", value_delimiter = ',')]
    sample_cells: Vec<u32>,

    /// Log every tick when a specified cell changes value.
    ///
    /// Comma-separated list of packed-cell indices to watch. Runs the program
    /// tick-by-tick (no batching) and prints a line whenever any watched cell
    /// changes: `tick=N mcIDX: OLD -> NEW (CS:IP)`. Use with --ticks N to bound
    /// the run. Slower than batched execution but finer-grained than sampling.
    #[arg(long, value_name = "LIST", value_delimiter = ',')]
    watch_cell: Vec<u32>,

    /// Output a JSON register trace for conformance comparison.
    ///
    /// Each tick produces a JSON object with register values.
    /// Combine with compare.mjs for automated divergence detection.
    #[arg(long)]
    trace_json: bool,

    /// Stop execution when memory at ADDR becomes non-zero.
    ///
    /// The address is checked after each tick. Useful for halt flags
    /// written by BIOS/DOS exit handlers.
    #[arg(long, value_name = "ADDR")]
    halt: Option<String>,

    /// After execution, write the Mode 13h framebuffer to a PPM file.
    ///
    /// Only written when the final video mode is 0x13 (320x200x256).
    /// Useful for capturing the last frame of a graphics program.
    #[arg(long, value_name = "PATH")]
    framebuffer_out: Option<String>,

    /// Inject pseudo-class press/release events at specific ticks.
    ///
    /// Format: TICK:[+|-]SELECTOR,TICK:[+|-]SELECTOR,...
    /// `+` (or no prefix) presses; `-` releases. SELECTOR is the id-
    /// selector text without the leading `#` (e.g. `kb-a`). The event
    /// flips the (`active`, SELECTOR) pseudo-class match edge on
    /// `state.pseudo_active`; the cabinet's
    /// `&:has(#SELECTOR:active) { --PROP: V }` rule produces V via
    /// calcite's input-edge recogniser.
    ///
    /// Example: `--press-events=50:kb-a,100:-kb-a`
    /// Presses kb-a at tick 50, releases at tick 100.
    #[arg(long, value_name = "EVENTS")]
    press_events: Option<String>,

    /// Dump a flat byte range of guest memory to a binary file at end of run.
    ///
    /// Format: `ADDR:LEN:PATH`. ADDR and LEN may be decimal or 0xHEX. Writes
    /// `LEN` bytes starting at linear address `ADDR` to `PATH`. Repeat the
    /// flag to dump multiple regions in one run.
    ///
    /// Designed as the cheap data-source for an external screenshot tool:
    /// run `calcite-cli` to a target tick, dump VRAM (and the BDA video-mode
    /// byte) to disk, then let a Node-side rasteriser turn the bytes into
    /// pixels. Avoids the per-tick chunked-IPC cost of going through
    /// `calcite-debugger` and avoids re-implementing the rasterisers in Rust.
    ///
    /// Example: `--dump-mem-range=0xB8000:4000:vram.bin --dump-mem-range=0x449:1:mode.bin`
    #[arg(long = "dump-mem-range", value_name = "ADDR:LEN:PATH")]
    dump_mem_ranges: Vec<String>,

    /// Trace the moment haltCode goes 0->nonzero.
    ///
    /// Runs tick-by-tick, keeps a small ring buffer of (CS, IP, AX, BX, CX, DX,
    /// SS, SP, opcode) values, and prints the last N entries when haltCode
    /// transitions to non-zero. Useful for finding the exact instruction that
    /// branched the CPU into garbage. Stops at the first halt; combine with
    /// --ticks N as an upper bound.
    #[arg(long)]
    trace_halt: bool,

    /// Fast-forward (run_batch) to this tick before starting tick-by-tick tracing.
    /// Useful with --trace-halt when you already know the halt is far in.
    #[arg(long, value_name = "N")]
    trace_halt_skip: Option<u32>,

    /// Restore a snapshot blob (produced by `--snapshot-out`) into the
    /// engine **before** the run begins. The cabinet must be the same one
    /// that produced the snapshot — slot ordering is parse-dependent.
    /// Pairs naturally with `--ticks N` to measure a precise window.
    #[arg(long = "restore", value_name = "PATH")]
    restore_path: Option<PathBuf>,

    /// Write a snapshot blob (state_vars + memory + extended +
    /// string_properties + frame_counter) to PATH **after** the run
    /// completes. Combine with `--script-event` and `--ticks N` to land on
    /// a specific moment (e.g. just-reached-in-game) and freeze it for
    /// later restore. The blob is independent of `--dump-mem-range` —
    /// snapshots are for resuming execution; mem-range dumps are for
    /// rendering.
    #[arg(long = "snapshot-out", value_name = "PATH")]
    snapshot_out: Option<PathBuf>,

    /// Sample CS:IP at fixed and bursty intervals for offline analysis.
    ///
    /// Format: `STRIDE,BURST,EVERY,PATH`. STRIDE/BURST/EVERY may be decimal
    /// or 0xHEX.
    ///   STRIDE — single-tick samples are taken every STRIDE ticks (the
    ///            wide axis: an unbiased CS:IP heatmap).
    ///   BURST  — every EVERY-th sample is replaced by BURST consecutive
    ///            single-tick samples (the local-shape axis: lets you
    ///            reconstruct loop bodies, REP bails, call depth).
    ///   EVERY  — bursts fire on sample 0, EVERY, 2*EVERY, … (set EVERY=0
    ///            to disable bursts entirely).
    ///   PATH   — output CSV: tick,cs,ip,sp,burst_id  (burst_id=-1 for
    ///            singles). Header line written first.
    ///
    /// Designed for offline hot-spot analysis of long-running workloads
    /// (Doom8088 level-load, etc). Pairs with `--restore` to skip boot
    /// and only sample a window of interest.
    ///
    /// Example: `--sample-cs-ip=600,1000,100,samples.csv` (≈50K wide
    /// samples + 50 bursts of 1000 over a 30M-tick window).
    #[arg(long = "sample-cs-ip", value_name = "STRIDE,BURST,EVERY,PATH")]
    sample_cs_ip: Option<String>,

    /// Op-adjacency profile. Records (prev_kind, curr_kind) counts for
    /// every op dispatched (including inside dispatch entries, function
    /// bodies, broadcast-write value_ops). Pair with --restore and --ticks
    /// to profile a specific window. Writes a CSV at PATH and prints a
    /// summary to stderr at end-of-run. ~10ns/op overhead when enabled.
    #[arg(long = "op-profile", value_name = "PATH")]
    op_profile_path: Option<PathBuf>,

    /// Generic measurement watch. Repeatable.
    ///
    /// Format: `NAME:KIND:SPEC[:gate=NAME][:sample=VAR1,VAR2][:then=ACTION1+ACTION2+...]`
    ///
    /// KIND:SPEC variants:
    ///   `stride:every=N`           — fire every N ticks.
    ///   `burst:every=N,count=K`    — fire on K consecutive ticks every N.
    ///   `at:tick=T`                — fire exactly once at tick T (replaces
    ///                                old --script-event scheduling).
    ///   `edge:addr=A`              — fire on byte transitions at A.
    ///   `cond:TEST1[,TEST2...][,repeat]`
    ///                              — fire when all tests hold. Default
    ///                                fires once, then disables. Add
    ///                                `,repeat` for sustain mode: fires
    ///                                on every gated poll while the
    ///                                predicate holds (pair with
    ///                                `setvar_pulse` to spam a key while
    ///                                a stage is up).
    ///   `halt:addr=A`              — fire and stop the run when byte at A
    ///                                is non-zero.
    ///
    /// TEST: `ADDR=VAL` | `ADDR!=VAL` | `pattern@BASE:STRIDE:WINDOW=NEEDLE`
    ///
    /// ACTION (within `then=`, separated by `+`):
    ///   `emit`                     — push a measurement event.
    ///   `halt`                     — stop the run.
    ///   `setvar=NAME,VALUE`        — write VALUE to state var NAME (e.g.
    ///                                `setvar=keyboard,0x1c0d`).
    ///   `setvar_pulse=NAME,VALUE,HOLD_TICKS`  — write VALUE now, then 0
    ///                                after HOLD_TICKS ticks. The make/break
    ///                                edge pair lets cabinet keyboard
    ///                                handlers register a press
    ///                                (e.g. `setvar_pulse=keyboard,0x1c0d,50000`).
    ///   `dump=ADDR,LEN[,PATH]`     — write LEN bytes from ADDR to PATH.
    ///                                PATH supports `{tick}` and `{name}`.
    ///   `snapshot[=PATH]`          — write a snapshot blob to PATH.
    ///
    /// Cheap-vs-expensive gating: `cond` and `edge` read guest memory and
    /// are too expensive to evaluate per-tick on hot loops. Each watch
    /// may name a `gate` (the name of another registered watch) — the
    /// gated watch only evaluates on ticks where its gate fired.
    ///
    /// Examples:
    ///   --watch poll:stride:every=50000
    ///   --watch ingame:cond:0x3a3c4=0:gate=poll:then=emit+halt
    ///   --watch text_drdos:cond:0x449=0x03,pattern@0xb8000:2:4000=DR-DOS:gate=poll:then=emit
    ///   --watch enter_at_2m:at:tick=2000000:then=setvar_pulse=keyboard,0x1c0d,50000
    ///   --watch dump_vram:stride:every=1000000:then=dump=0xb8000,4000,vram_{tick}.bin
    ///
    /// Events go to stderr (one line each) and, if `--measure-out=PATH`
    /// is given, to PATH as JSON Lines.
    #[arg(long = "watch", value_name = "NAME:KIND:SPEC[:OPTS]")]
    watches: Vec<String>,

    /// Write measurement events to PATH as JSON Lines (one event per line).
    ///
    /// Each event is a `{"tick":T, "watch":"NAME", "vars":[[NAME,VAL],...],
    /// "halted":bool, "dumps":[{tag, addr, path}]}` object. `--watch ...:then=emit`
    /// (or any action that produces an event) populates this stream.
    #[arg(long = "measure-out", value_name = "PATH")]
    measure_out: Option<PathBuf>,
}


/// Map a crossterm key event to a CSS-DOS-style id selector
/// (`kb-X`) without the leading `#`. Returns `None` for unmapped
/// keys. Mirrors the selector set kiln emits via
/// `template.mjs::KEYBOARD_KEYS`. The cabinet's
/// `&:has(#SEL:active) { --keyboard: V }` rules supply the
/// (scancode<<8)|ascii value when calcite flips the pseudo edge.
fn key_to_selector(key: &KeyEvent) -> Option<&'static str> {
    use KeyCode::*;
    Some(match key.code {
        Esc => "kb-esc",
        Char('1') | Char('!') => "kb-1",
        Char('2') | Char('@') => "kb-2",
        Char('3') | Char('#') => "kb-3",
        Char('4') | Char('$') => "kb-4",
        Char('5') | Char('%') => "kb-5",
        Char('6') | Char('^') => "kb-6",
        Char('7') | Char('&') => "kb-7",
        Char('8') | Char('*') => "kb-8",
        Char('9') | Char('(') => "kb-9",
        Char('0') | Char(')') => "kb-0",
        Backspace => "kb-bksp",
        Tab => "kb-tab",
        Char('q') | Char('Q') => "kb-q",
        Char('w') | Char('W') => "kb-w",
        Char('e') | Char('E') => "kb-e",
        Char('r') | Char('R') => "kb-r",
        Char('t') | Char('T') => "kb-t",
        Char('y') | Char('Y') => "kb-y",
        Char('u') | Char('U') => "kb-u",
        Char('i') | Char('I') => "kb-i",
        Char('o') | Char('O') => "kb-o",
        Char('p') | Char('P') => "kb-p",
        Enter => "kb-enter",
        Char('a') | Char('A') => "kb-a",
        Char('s') | Char('S') => "kb-s",
        Char('d') | Char('D') => "kb-d",
        Char('f') | Char('F') => "kb-f",
        Char('g') | Char('G') => "kb-g",
        Char('h') | Char('H') => "kb-h",
        Char('j') | Char('J') => "kb-j",
        Char('k') | Char('K') => "kb-k",
        Char('l') | Char('L') => "kb-l",
        Char('z') | Char('Z') => "kb-z",
        Char('x') | Char('X') => "kb-x",
        Char('c') | Char('C') => "kb-c",
        Char('v') | Char('V') => "kb-v",
        Char('b') | Char('B') => "kb-b",
        Char('n') | Char('N') => "kb-n",
        Char('m') | Char('M') => "kb-m",
        Char(',') | Char('<') => "kb-comma",
        Char('.') | Char('>') => "kb-period",
        Char('/') | Char('?') => "kb-slash",
        Char(' ') => "kb-space",
        Up => "kb-up",
        Down => "kb-down",
        Left => "kb-left",
        Right => "kb-right",
        F(1) => "kb-f1",
        F(2) => "kb-f2",
        F(3) => "kb-f3",
        F(4) => "kb-f4",
        F(5) => "kb-f5",
        F(6) => "kb-f6",
        F(7) => "kb-f7",
        F(8) => "kb-f8",
        F(9) => "kb-f9",
        F(10) => "kb-f10",
        _ => return None,
    })
}

/// Map a crossterm key event to DOS keyboard value: (scancode << 8) | ascii.
/// Returns 0 for unmapped keys.
fn key_to_dos(key: &KeyEvent) -> i32 {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

    let (scan, ascii): (u8, u8) = match key.code {
        KeyCode::Esc => (0x01, 0x1B),
        KeyCode::Char('1') | KeyCode::Char('!') => (0x02, key.code.into_ascii()),
        KeyCode::Char('2') | KeyCode::Char('@') => (0x03, key.code.into_ascii()),
        KeyCode::Char('3') | KeyCode::Char('#') => (0x04, key.code.into_ascii()),
        KeyCode::Char('4') | KeyCode::Char('$') => (0x05, key.code.into_ascii()),
        KeyCode::Char('5') | KeyCode::Char('%') => (0x06, key.code.into_ascii()),
        KeyCode::Char('6') | KeyCode::Char('^') => (0x07, key.code.into_ascii()),
        KeyCode::Char('7') | KeyCode::Char('&') => (0x08, key.code.into_ascii()),
        KeyCode::Char('8') | KeyCode::Char('*') => (0x09, key.code.into_ascii()),
        KeyCode::Char('9') | KeyCode::Char('(') => (0x0A, key.code.into_ascii()),
        KeyCode::Char('0') | KeyCode::Char(')') => (0x0B, key.code.into_ascii()),
        KeyCode::Char('-') | KeyCode::Char('_') => (0x0C, key.code.into_ascii()),
        KeyCode::Char('=') | KeyCode::Char('+') => (0x0D, key.code.into_ascii()),
        KeyCode::Backspace => (0x0E, 0x08),
        KeyCode::Tab => (0x0F, 0x09),
        KeyCode::Char('q') | KeyCode::Char('Q') => (0x10, key.code.into_ascii()),
        KeyCode::Char('w') | KeyCode::Char('W') => (0x11, key.code.into_ascii()),
        KeyCode::Char('e') | KeyCode::Char('E') => (0x12, key.code.into_ascii()),
        KeyCode::Char('r') | KeyCode::Char('R') => (0x13, key.code.into_ascii()),
        KeyCode::Char('t') | KeyCode::Char('T') => (0x14, key.code.into_ascii()),
        KeyCode::Char('y') | KeyCode::Char('Y') => (0x15, key.code.into_ascii()),
        KeyCode::Char('u') | KeyCode::Char('U') => (0x16, key.code.into_ascii()),
        KeyCode::Char('i') | KeyCode::Char('I') => (0x17, key.code.into_ascii()),
        KeyCode::Char('o') | KeyCode::Char('O') => (0x18, key.code.into_ascii()),
        KeyCode::Char('p') | KeyCode::Char('P') => (0x19, key.code.into_ascii()),
        KeyCode::Char('[') | KeyCode::Char('{') => (0x1A, key.code.into_ascii()),
        KeyCode::Char(']') | KeyCode::Char('}') => (0x1B, key.code.into_ascii()),
        KeyCode::Enter => (0x1C, 0x0D),
        KeyCode::Char('a') | KeyCode::Char('A') => (0x1E, key.code.into_ascii()),
        KeyCode::Char('s') | KeyCode::Char('S') => (0x1F, key.code.into_ascii()),
        KeyCode::Char('d') | KeyCode::Char('D') => (0x20, key.code.into_ascii()),
        KeyCode::Char('f') | KeyCode::Char('F') => (0x21, key.code.into_ascii()),
        KeyCode::Char('g') | KeyCode::Char('G') => (0x22, key.code.into_ascii()),
        KeyCode::Char('h') | KeyCode::Char('H') => (0x23, key.code.into_ascii()),
        KeyCode::Char('j') | KeyCode::Char('J') => (0x24, key.code.into_ascii()),
        KeyCode::Char('k') | KeyCode::Char('K') => (0x25, key.code.into_ascii()),
        KeyCode::Char('l') | KeyCode::Char('L') => (0x26, key.code.into_ascii()),
        KeyCode::Char(';') | KeyCode::Char(':') => (0x27, key.code.into_ascii()),
        KeyCode::Char('\'') | KeyCode::Char('"') => (0x28, key.code.into_ascii()),
        KeyCode::Char('`') | KeyCode::Char('~') => (0x29, key.code.into_ascii()),
        KeyCode::Char('\\') | KeyCode::Char('|') => (0x2B, key.code.into_ascii()),
        KeyCode::Char('z') | KeyCode::Char('Z') => (0x2C, key.code.into_ascii()),
        KeyCode::Char('x') | KeyCode::Char('X') => (0x2D, key.code.into_ascii()),
        KeyCode::Char('c') | KeyCode::Char('C') => (0x2E, key.code.into_ascii()),
        KeyCode::Char('v') | KeyCode::Char('V') => (0x2F, key.code.into_ascii()),
        KeyCode::Char('b') | KeyCode::Char('B') => (0x30, key.code.into_ascii()),
        KeyCode::Char('n') | KeyCode::Char('N') => (0x31, key.code.into_ascii()),
        KeyCode::Char('m') | KeyCode::Char('M') => (0x32, key.code.into_ascii()),
        KeyCode::Char(',') | KeyCode::Char('<') => (0x33, key.code.into_ascii()),
        KeyCode::Char('.') | KeyCode::Char('>') => (0x34, key.code.into_ascii()),
        KeyCode::Char('/') | KeyCode::Char('?') => (0x35, key.code.into_ascii()),
        KeyCode::Char(' ') => (0x39, 0x20),
        // Extended keys: ascii=0
        KeyCode::Up => (0x48, 0),
        KeyCode::Down => (0x50, 0),
        KeyCode::Left => (0x4B, 0),
        KeyCode::Right => (0x4D, 0),
        KeyCode::Home => (0x47, 0),
        KeyCode::End => (0x4F, 0),
        KeyCode::PageUp => (0x49, 0),
        KeyCode::PageDown => (0x51, 0),
        KeyCode::Insert => (0x52, 0),
        KeyCode::Delete => (0x53, 0),
        KeyCode::F(1) => (0x3B, 0),
        KeyCode::F(2) => (0x3C, 0),
        KeyCode::F(3) => (0x3D, 0),
        KeyCode::F(4) => (0x3E, 0),
        KeyCode::F(5) => (0x3F, 0),
        KeyCode::F(6) => (0x40, 0),
        KeyCode::F(7) => (0x41, 0),
        KeyCode::F(8) => (0x42, 0),
        KeyCode::F(9) => (0x43, 0),
        KeyCode::F(10) => (0x44, 0),
        _ => return 0,
    };

    // Ctrl+letter: ascii becomes 1-26
    let ascii = if ctrl {
        match key.code {
            KeyCode::Char(c) if c.is_ascii_alphabetic() => (c.to_ascii_lowercase() as u8) - b'a' + 1,
            _ => ascii,
        }
    } else {
        ascii
    };

    ((scan as i32) << 8) | (ascii as i32)
}

/// Helper trait to extract ASCII value from KeyCode.
trait KeyCodeAscii {
    fn into_ascii(self) -> u8;
}

impl KeyCodeAscii for KeyCode {
    fn into_ascii(self) -> u8 {
        match self {
            KeyCode::Char(c) => c as u8,
            _ => 0,
        }
    }
}

/// Parse and execute a `--dump-mem-range=ADDR:LEN:PATH` spec: write `LEN`
/// bytes from guest linear address `ADDR` to `PATH`. Returns a stringified
/// error on parse or write failure. Numbers may be decimal or `0x`-prefixed.
///
/// Out-of-range reads return zeros (matching `state.read_mem` semantics) so a
/// dump request that overshoots the cabinet's allocated memory still produces
/// a file of the requested length — the caller can decide whether all-zeros
/// means "VRAM not yet written" or "wrong address".
fn dump_mem_range(spec: &str, state: &calcite_core::State) -> Result<(), String> {
    let parts: Vec<&str> = spec.splitn(3, ':').collect();
    if parts.len() != 3 {
        return Err(format!("expected ADDR:LEN:PATH, got {spec:?}"));
    }
    let parse_num = |s: &str, what: &str| -> Result<i64, String> {
        let s = s.trim();
        let v = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            i64::from_str_radix(hex, 16)
        } else {
            s.parse::<i64>()
        };
        v.map_err(|e| format!("invalid {what} {s:?}: {e}"))
    };
    let addr = parse_num(parts[0], "ADDR")?;
    let len = parse_num(parts[1], "LEN")?;
    if len < 0 {
        return Err(format!("LEN must be non-negative, got {len}"));
    }
    let path = parts[2];
    let mut out = vec![0u8; len as usize];
    for i in 0..(len as usize) {
        out[i] = (state.read_mem(addr as i32 + i as i32) & 0xFF) as u8;
    }
    std::fs::write(path, &out).map_err(|e| format!("write {path:?}: {e}"))?;
    eprintln!("dump-mem-range: {} bytes from 0x{:X} -> {}", len, addr, path);
    Ok(())
}


fn main() {
    env_logger::init();
    let cli = Cli::parse();

    // Resolve input path: direct --input, or interactive menu.
    let input_path: PathBuf = match cli.input.clone() {
        Some(p) => p,
        None => {
            // Interactive menu mode. Root is the current working directory
            // (run.bat already cd's to calcite/; direct users should invoke
            // from the calcite repo root).
            let root = PathBuf::from(r"C:\Users\AdmT9N0CX01V65438A\Documents\src\calcite");
            let entries = menu::discover(&root);
            let selected = match menu::run(&entries) {
                Ok(Some(i)) => i,
                Ok(None) => {
                    eprintln!();
                    std::process::exit(0);
                }
                Err(e) => {
                    eprintln!("Menu error: {e}");
                    std::process::exit(1);
                }
            };
            // Clear after the menu exits so the runner output starts fresh.
            print!("\x1b[2J\x1b[H");
            cssdos_logo::print();
            menu::resolve_to_css(&entries[selected])
        }
    };

    // Transition into the calcite half of the experience: fresh screen,
    // calcite banner, parse/compile bars, video.
    print!("\x1b[2J\x1b[H");
    calcite_logo::print();

    let css = match std::fs::read_to_string(&input_path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading {}: {e}", input_path.display());
            std::process::exit(1);
        }
    };

    log::info!(
        "Read {} bytes of CSS from {}",
        css.len(),
        input_path.display()
    );

    let t0 = std::time::Instant::now();
    match calcite_core::parser::parse_css(&css) {
        Ok(parsed) => {
            let parse_time = t0.elapsed();
            eprintln!(
                "Parsed: {} @property, {} @function, {} assignments ({:.2}s)",
                parsed.properties.len(),
                parsed.functions.len(),
                parsed.assignments.len(),
                parse_time.as_secs_f64(),
            );

            if cli.parse_only {
                return;
            }

            // load_properties MUST be called before from_parsed so that state
            // variable addresses are in the address map before compilation.
            let mut state = calcite_core::State::default();
            state.load_properties(&parsed.properties);

            let t1 = std::time::Instant::now();
            let mut evaluator = calcite_core::Evaluator::from_parsed(&parsed);
            let compile_time = t1.elapsed();

            // --op-profile: enable adjacency tracking before the run starts.
            // The CSV + summary print happens at end-of-run.
            if cli.op_profile_path.is_some() {
                calcite_core::pattern::op_profile::op_profile_enable();
            }

            // Packed-cell cabinets (PACK_SIZE > 1) keep guest memory in
            // `mcN` state vars, not in `state.memory[]`. Without this call
            // every positive-address read returns 0 even though the CPU
            // wrote the value — see Evaluator::wire_state_for_packed_memory.
            evaluator.wire_state_for_packed_memory(&mut state);
            // The cabinet's windowed-byte-array dispatch is recognised at
            // compile time and the descriptor is installed on State here. No
            // sidecar load: the bytes already live in the cabinet's compiled
            // flat-array dispatch — see
            // Evaluator::wire_state_for_windowed_byte_array.
            evaluator.wire_state_for_windowed_byte_array(&mut state);
            // Input-edge groups: resolve `&:has(#X:pseudo) { --P: V }`
            // bindings to their target state-var slots so
            // `state.set_pseudo_class_active(...)` can write the slot
            // directly. No per-tick apply path needed after this.
            evaluator.wire_state_for_input_edges(&mut state);

            // --restore: replay a previously-captured execution state into
            // the freshly-wired engine. This must run AFTER load_properties
            // and the wire_state_for_* calls so the slot count is finalised
            // (the snapshot's state_vars length must match self.state_vars).
            if let Some(path) = &cli.restore_path {
                match std::fs::read(path) {
                    Ok(bytes) => match state.restore(&bytes) {
                        Ok(()) => eprintln!(
                            "restored snapshot from {} ({} bytes)",
                            path.display(),
                            bytes.len()
                        ),
                        Err(e) => {
                            eprintln!("--restore={}: {}", path.display(), e);
                            std::process::exit(2);
                        }
                    },
                    Err(e) => {
                        eprintln!("--restore={}: {}", path.display(), e);
                        std::process::exit(2);
                    }
                }
            }

            // Interactive iff --ticks is omitted; in that case run unlimited.
            let interactive = cli.ticks.is_none();
            let ticks_limit: u32 = cli.ticks.unwrap_or(u32::MAX);
            eprintln!(
                "Compiled: {:.2}s (parse {:.2}s + compile {:.2}s), memory: {} KB",
                (parse_time + compile_time).as_secs_f64(),
                parse_time.as_secs_f64(),
                compile_time.as_secs_f64(),
                state.memory.len() / 1024,
            );
            if interactive {
                eprintln!("Starting... (Ctrl+C to quit)");
            }

            let halt_addr: Option<i32> = cli.halt.as_ref().map(|s| {
                if s.starts_with("0x") || s.starts_with("0X") {
                    i32::from_str_radix(&s[2..], 16).expect("Invalid halt address")
                } else if s.starts_with('-') || s.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                    s.parse().expect("Invalid halt address")
                } else {
                    // Treat as a state variable name — look up its slot address
                    if let Some(slot) = state.var_slot(s) {
                        -(slot as i32) - 1
                    } else {
                        panic!("Unknown state variable for --halt: {}", s);
                    }
                }
            });

            /// Apply any press-events scheduled for `tick` to `state`.
            /// Each event is (tick, selector, active); only events
            /// whose tick matches fire.
            fn apply_press_events(
                events: &[(u32, String, bool)],
                tick: u32,
                state: &mut calcite_core::State,
            ) {
                for (ev_tick, sel, active) in events {
                    if *ev_tick == tick {
                        state.set_pseudo_class_active("active", sel, *active);
                    }
                }
            }

            // Parse --press-events=TICK:[+|-]SELECTOR,TICK:[+|-]SELECTOR,...
            // Each event is (tick, selector, active). `+` or no prefix
            // means press; `-` means release. The pseudo-class is
            // always "active".
            let press_events: Vec<(u32, String, bool)> = cli.press_events.as_ref().map(|s| {
                s.split(',')
                    .filter(|part| !part.is_empty())
                    .map(|part| {
                        let (tick_str, sel_part) = part.split_once(':')
                            .unwrap_or_else(|| panic!("Invalid press-event format: {part:?}"));
                        let tick: u32 = tick_str.parse()
                            .unwrap_or_else(|_| panic!("Invalid tick in press-event: {tick_str:?}"));
                        let (active, sel) = if let Some(rest) = sel_part.strip_prefix('-') {
                            (false, rest.to_string())
                        } else if let Some(rest) = sel_part.strip_prefix('+') {
                            (true, rest.to_string())
                        } else {
                            (true, sel_part.to_string())
                        };
                        if sel.is_empty() {
                            panic!("Empty selector in press-event: {part:?}");
                        }
                        (tick, sel, active)
                    })
                    .collect()
            }).unwrap_or_default();

            // Parse --watch specs into a WatchRegistry. The native CLI
            // writes dump/snapshot payloads to disk via the templated
            // path; events flow to stderr (one line each) and to
            // --measure-out as JSON Lines.
            let mut watch_registry = calcite_core::script::WatchRegistry::new();
            watch_registry.set_dump_sink(calcite_core::script::DumpSink::File);
            for s in &cli.watches {
                match watch_spec::parse_watch(s) {
                    Ok(w) => { watch_registry.register(w); }
                    Err(e) => {
                        eprintln!("--watch {s:?}: {e}");
                        std::process::exit(1);
                    }
                }
            }
            // Validate gate references.
            {
                let names: std::collections::HashSet<String> =
                    watch_registry.watch_names().map(|s| s.to_string()).collect();
                for s in &cli.watches {
                    if let Ok(w) = watch_spec::parse_watch(s) {
                        if let Some(g) = &w.gate {
                            if !names.contains(g) {
                                eprintln!("--watch {:?}: gate {:?} is not a registered watch", w.name, g);
                                std::process::exit(1);
                            }
                        }
                    }
                }
            }
            if !watch_registry.is_empty() {
                eprintln!("registered {} watch(es)", watch_registry.len());
            }

            // --measure-out: open the JSON-lines stream up-front so we
            // can append per emitted event.
            let mut measure_out: Option<std::io::BufWriter<std::fs::File>> =
                cli.measure_out.as_ref().and_then(|p| {
                    match std::fs::File::create(p) {
                        Ok(f) => Some(std::io::BufWriter::new(f)),
                        Err(e) => {
                            eprintln!("--measure-out={}: {}", p.display(), e);
                            std::process::exit(1);
                        }
                    }
                });

            let t2 = std::time::Instant::now();

            // Real 8086 clock: 4.77 MHz. We derive CPU speed from cycleCount
            // (accumulated 8086 cycle costs per instruction) divided by wall time.
            const REAL_8086_HZ: f64 = 4_772_727.0;

            // Format a frequency at a fixed width so the status line doesn't
            // glitch back and forth. Always MHz with 2 decimals for the rolling
            // status display (1.00 KHz == 0.00 MHz, still readable).
            fn format_hz(hz: f64) -> String {
                if hz >= 1_000_000.0 {
                    format!("{:.2} MHz", hz / 1_000_000.0)
                } else if hz >= 1_000.0 {
                    format!("{:.1} KHz", hz / 1_000.0)
                } else {
                    format!("{:.0} Hz", hz)
                }
            }
            fn format_hz_fixed(hz: f64) -> String {
                // Always "X.XX MHz" — 8 chars wide. At low speeds this reads
                // "0.04 MHz" etc., which stays aligned column-wise across
                // repaints.
                format!("{:5.2} MHz", hz / 1_000_000.0)
            }

            // Render the screen based on BDA video mode (0x0449).
            // Text modes: render in-terminal using ANSI. Mode 13h: status line.
            //
            // ticks = CSS ticks (= instructions in V4 single-cycle arch)
            // cycles = accumulated 8086 clock cycles from cycleCount
            //
            // Status line throttling: the speed readout only refreshes every
            // ~500ms of wall time so the digits don't thrash between repaints.
            // We also EMA-smooth ticks/s to make it calmer.
            let mut smoothed_tps: f64 = 0.0;
            let mut last_status_update = std::time::Instant::now();
            let mut cached_status = String::new();
            let mut render_screen = |state: &calcite_core::State, ticks: u32,
                                  delta_ticks: u32, delta_secs: f64,
                                  first: bool| {
                let cycles = state.get_var("cycleCount").unwrap_or(0) as u64;
                let instant_tps = if delta_secs > 0.0 { delta_ticks as f64 / delta_secs } else { 0.0 };
                // Exponential moving average — alpha 0.3 feels smooth but still
                // responsive.
                if smoothed_tps == 0.0 {
                    smoothed_tps = instant_tps;
                } else {
                    smoothed_tps = 0.3 * instant_tps + 0.7 * smoothed_tps;
                }
                let now = std::time::Instant::now();
                let since_update = now.duration_since(last_status_update).as_secs_f64();
                if first || since_update >= 0.5 {
                    let avg_cpt = if ticks > 0 { cycles as f64 / ticks as f64 } else { 10.0 };
                    let cycles_per_sec = smoothed_tps * avg_cpt;
                    let pct = cycles_per_sec / REAL_8086_HZ * 100.0;
                    // Fixed-width fields so the line doesn't jitter.
                    cached_status = format!(
                        " {:>10} ticks | {} | {:>5.1}% of 8086 | {:>9.0} ticks/s",
                        ticks, format_hz_fixed(cycles_per_sec), pct, smoothed_tps
                    );
                    last_status_update = now;
                }
                let status = &cached_status;
                let video_mode = state.read_mem(0x0449) as u8;
                if video_mode == 0x13 {
                    // Render 320x200 mode 13h via ANSI half-block chars.
                    // Each '▀' cell shows two vertical pixels: fg=upper, bg=lower.
                    // Downsample horizontally 2:1 so 320-wide fits typical terminals.
                    if first {
                        eprint!("\x1b[2J\x1b[H\x1b[?25l");
                    } else {
                        eprint!("\x1b[H");
                    }
                    let mut out = String::with_capacity(320 * 100 * 24);
                    for y in (0..200).step_by(2) {
                        for x in 0..320 {
                            let top = state.read_mem(0xA0000 + y * 320 + x) as usize & 0x0F;
                            let bot = state.read_mem(0xA0000 + (y + 1) * 320 + x) as usize & 0x0F;
                            let (tr, tg, tb) = calcite_core::state::CGA_PALETTE[top];
                            let (br, bg_, bb) = calcite_core::state::CGA_PALETTE[bot];
                            out.push_str(&format!(
                                "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀",
                                tr, tg, tb, br, bg_, bb
                            ));
                        }
                        out.push_str("\x1b[0m\r\n");
                    }
                    eprint!("{out}{status}\r\n");
                } else {
                    let cols = state.read_mem(0x044A) as usize;
                    let (width, height) = if cols == 40 { (40, 25) } else { (80, 25) };
                    if first {
                        eprint!("\x1b[2J\x1b[H\x1b[?25l");
                    } else {
                        eprint!("\x1b[H");
                    }
                    let screen = state.render_screen_ansi(0xB8000, width, height);
                    eprint!("┌{}┐\r\n", "─".repeat(width));
                    for line in screen.lines() {
                        eprint!("│{line}\x1b[0m│\r\n");
                    }
                    eprint!("└{}┘\r\n", "─".repeat(width));
                    eprint!("{status}\r\n");
                }
            };

            // --dump-tick / --dump-ticks: run to specific tick(s) and dump state
            // Build a sorted list of targets: either the single --dump-tick value,
            // or all values from --dump-ticks (combined if both given).
            let mut dump_targets: Vec<u32> = cli.dump_ticks.clone();
            if let Some(t) = cli.dump_tick {
                dump_targets.push(t);
            }
            if !dump_targets.is_empty() {
                dump_targets.sort_unstable();
                dump_targets.dedup();
                let compact = !cli.sample_cells.is_empty();

                // Print a header so callers can tell this is the dump output.
                if compact {
                    // Header: tick + register names + sampled cell indices.
                    print!("# tick");
                    for name in ["AX","BX","CX","DX","SP","BP","SI","DI","IP","CS","DS","ES","SS","flags","cycleCount"] {
                        print!(" {}", name);
                    }
                    for &idx in &cli.sample_cells {
                        print!(" mc{}", idx);
                    }
                    println!();
                }

                // Advance to each target in order, dumping along the way.
                let mut cursor: u32 = 0;
                for &target in &dump_targets {
                    // Advance from cursor to target. Run tick-by-tick when key events
                    // need injection; otherwise use run_batch + a final tick.
                    if !press_events.is_empty() {
                        for tick in cursor..target {
                            apply_press_events(&press_events, tick, &mut state);
                            evaluator.tick(&mut state);
                        }
                    } else if target > cursor {
                        let batch = target - cursor;
                        if batch > 1 {
                            evaluator.run_batch(&mut state, batch - 1);
                        }
                        evaluator.tick(&mut state);
                    }
                    cursor = target;

                    if compact {
                        // One line per tick: tick + core regs + sampled cells.
                        print!("{}", target);
                        for name in ["AX","BX","CX","DX","SP","BP","SI","DI","IP","CS","DS","ES","SS","flags","cycleCount"] {
                            let v = state.state_var_names.iter().position(|n| n == name)
                                .map(|i| state.state_vars[i]).unwrap_or(0);
                            print!(" {}", v);
                        }
                        // Sampled cells — look up by "mcN" state-var name (how kiln
                        // emits packed cells) and fall back to 0 for unknown cells.
                        for &idx in &cli.sample_cells {
                            let name = format!("mc{}", idx);
                            let v = state.state_var_names.iter().position(|n| n == &name)
                                .map(|i| state.state_vars[i]).unwrap_or(0);
                            print!(" {}", v);
                        }
                        println!();
                    } else {
                        // Full slot dump (existing behavior, now usable for each target)
                        println!("=== Slot dump at tick {} ===", target);
                        print!("Registers:");
                        for (i, name) in state.state_var_names.iter().enumerate() {
                            print!(" {}={}", name, state.state_vars[i]);
                        }
                        println!();
                        let props = [
                            "--csBase", "--ipAddr", "--q0", "--q1", "--q2", "--q3",
                            "--opcode", "--mod", "--reg", "--rm",
                            "--modrmExtra", "--ea", "--eaSeg", "--eaOff",
                            "--immOff", "--immByte", "--immWord", "--imm8", "--imm16",
                            "--rmVal8", "--rmVal16", "--regVal8", "--regVal16",
                            "--memAddr", "--memVal",
                            "--memAddr0", "--memVal0", "--memAddr1", "--memVal1",
                            "--memAddr2", "--memVal2",
                            "--addrDestA", "--addrDestB", "--addrDestC",
                            "--addrValA", "--addrValA1", "--addrValA2", "--addrValB", "--addrValC",
                            "--isWordWrite", "--instId", "--instLen",
                            "--modRm", "--modRm_mod", "--modRm_reg", "--modRm_rm",
                            "--moveStack", "--moveSI", "--moveDI", "--jumpCS", "--addrJump",
                            "--AX", "--CX", "--DX", "--BX",
                            "--AL", "--CL", "--DL", "--BL",
                            "--AH", "--CH", "--DH", "--BH",
                            "--SP", "--BP", "--SI", "--DI", "--IP",
                            "--ES", "--CS", "--SS", "--DS", "--flags",
                            "--halt", "--uOp",
                        ];
                        println!("\nComputed properties:");
                        for name in &props {
                            if let Some(val) = evaluator.get_slot_value(name) {
                                println!("  {}: {} (0x{:X})", name, val, val as u32);
                            }
                        }
                    }
                }
                // --dump-mem-range pairs naturally with --dump-tick — same
                // "stop at this tick and report what's there" model. Run them
                // here too so callers can `--dump-tick=N --dump-mem-range=...`
                // without also passing --ticks (which the eval block below
                // requires).
                for spec in &cli.dump_mem_ranges {
                    if let Err(e) = dump_mem_range(spec, &state) {
                        eprintln!("--dump-mem-range={spec}: {e}");
                        std::process::exit(1);
                    }
                }
                return;
            }

            // --sample-cs-ip: hybrid wide+bursty CS:IP sampler for offline
            // hot-spot analysis. Writes a CSV to disk; prints nothing to
            // stdout except summary stats at end.
            if let Some(spec) = cli.sample_cs_ip.as_ref() {
                let parts: Vec<&str> = spec.splitn(4, ',').collect();
                if parts.len() != 4 {
                    eprintln!("--sample-cs-ip: expected STRIDE,BURST,EVERY,PATH; got {:?}", spec);
                    std::process::exit(2);
                }
                let parse_u = |s: &str, name: &str| -> u32 {
                    let s = s.trim();
                    let r = if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                        u32::from_str_radix(h, 16)
                    } else { s.parse::<u32>() };
                    r.unwrap_or_else(|_| {
                        eprintln!("--sample-cs-ip: bad {}: {:?}", name, s);
                        std::process::exit(2);
                    })
                };
                let stride = parse_u(parts[0], "STRIDE").max(1);
                let burst = parse_u(parts[1], "BURST");
                let every = parse_u(parts[2], "EVERY"); // 0 disables bursts
                let path = parts[3].to_string();

                let out_file = std::fs::File::create(&path).unwrap_or_else(|e| {
                    eprintln!("--sample-cs-ip: cannot create {}: {}", path, e);
                    std::process::exit(2);
                });
                let mut out = std::io::BufWriter::new(out_file);
                use std::io::Write;
                writeln!(out, "tick,cs,ip,sp,burst_id").ok();

                let cs_slot = state.state_var_names.iter().position(|n| n == "CS");
                let ip_slot = state.state_var_names.iter().position(|n| n == "IP");
                let sp_slot = state.state_var_names.iter().position(|n| n == "SP");
                let read = |state: &calcite_core::State, slot: Option<usize>| -> i32 {
                    slot.map(|i| state.state_vars[i]).unwrap_or(0)
                };

                let mut tick: u32 = 0;
                let mut sample_count: u32 = 0;
                let mut burst_count: u32 = 0;
                let mut singles: u32 = 0;
                let t_start = std::time::Instant::now();

                while tick < ticks_limit {
                    // Decide: is this a burst sample, or a wide single?
                    let is_burst = every > 0 && burst > 0
                        && (sample_count % every == 0);

                    if is_burst {
                        // Tick-by-tick for BURST consecutive ticks. Cap at
                        // remaining budget so we don't overshoot ticks_limit.
                        let n = burst.min(ticks_limit - tick);
                        let bid = burst_count as i32;
                        for _ in 0..n {
                            evaluator.tick(&mut state);
                            tick += 1;
                            let cs = read(&state, cs_slot);
                            let ip = read(&state, ip_slot);
                            let sp = read(&state, sp_slot);
                            writeln!(out, "{},{},{},{},{}", tick, cs, ip, sp, bid).ok();
                        }
                        burst_count += 1;
                    } else {
                        // Wide pass: run STRIDE-1 ticks via run_batch (fast),
                        // then one tick to sample on. This gives us "the IP
                        // value at exactly tick T" rather than "somewhere
                        // inside the batch".
                        let advance = stride.min(ticks_limit - tick);
                        if advance > 1 {
                            evaluator.run_batch(&mut state, advance - 1);
                        }
                        if advance >= 1 {
                            evaluator.tick(&mut state);
                        }
                        tick += advance;
                        let cs = read(&state, cs_slot);
                        let ip = read(&state, ip_slot);
                        let sp = read(&state, sp_slot);
                        writeln!(out, "{},{},{},{},{}", tick, cs, ip, sp, -1).ok();
                        singles += 1;
                    }
                    sample_count += 1;

                    if let Some(addr) = halt_addr {
                        if state.read_mem(addr) != 0 {
                            eprintln!("Halt: memory 0x{:X} set at tick {}", addr, tick);
                            break;
                        }
                    }
                }
                out.flush().ok();
                drop(out);
                let elapsed = t_start.elapsed().as_secs_f64();
                let tps = if elapsed > 0.0 { tick as f64 / elapsed } else { 0.0 };
                eprintln!(
                    "sample-cs-ip: {} ticks in {:.1}s ({:.0} ticks/s); {} singles + {} bursts ({} samples) -> {}",
                    tick, elapsed, tps, singles, burst_count, singles + burst_count * burst, path,
                );
                return;
            }

            // --watch-cell: tick-by-tick monitor, print only transitions.
            if !cli.watch_cell.is_empty() {
                // Resolve each watched cell-idx to its state-var slot index up front.
                let watched: Vec<(u32, Option<usize>, i32)> = cli.watch_cell
                    .iter()
                    .map(|&idx| {
                        let name = format!("mc{}", idx);
                        let slot = state.state_var_names.iter().position(|n| n == &name);
                        let val = slot.map(|i| state.state_vars[i]).unwrap_or(0);
                        (idx, slot, val)
                    })
                    .collect();
                let mut last: Vec<i32> = watched.iter().map(|(_, _, v)| *v).collect();
                // Header: name the columns so the output is self-describing.
                print!("# watch-cell: tick");
                for (idx, _, _) in &watched { print!(" mc{}", idx); }
                println!(" | CS IP");
                // Initial values at tick 0.
                print!("0");
                for v in &last { print!(" {}", v); }
                let cs0 = state.get_var("CS").unwrap_or(0);
                let ip0 = state.get_var("IP").unwrap_or(0);
                println!(" | {} {}", cs0, ip0);

                for tick in 1..=ticks_limit {
                    apply_press_events(&press_events, tick, &mut state);
                    evaluator.tick(&mut state);
                    let mut any_change = false;
                    let mut now: Vec<i32> = Vec::with_capacity(watched.len());
                    for (_, slot, _) in &watched {
                        let v = slot.map(|i| state.state_vars[i]).unwrap_or(0);
                        now.push(v);
                    }
                    for i in 0..watched.len() {
                        if now[i] != last[i] { any_change = true; break; }
                    }
                    if any_change {
                        print!("{}", tick);
                        for v in &now { print!(" {}", v); }
                        let cs = state.get_var("CS").unwrap_or(0);
                        let ip = state.get_var("IP").unwrap_or(0);
                        println!(" | {} {}", cs, ip);
                        last = now;
                    }
                    if let Some(addr) = halt_addr {
                        if state.read_mem(addr) != 0 {
                            eprintln!("Halt: memory 0x{:X} set at tick {}", addr, tick);
                            break;
                        }
                    }
                }
                return;
            }

            // --trace-halt: tick-by-tick monitor that prints a ring buffer of
            // recent CPU state when haltCode goes 0 -> nonzero. Designed for
            // finding the exact instruction that branches the CPU into garbage
            // (the symptom is "haltCode=110/OUTSB" suddenly latching after
            // millions of clean ticks). Slow — runs unbatched — so cap with
            // --ticks N when looking near a known halt window.
            if cli.trace_halt {
                let names = ["CS","IP","AX","BX","CX","DX","SI","DI","SS","SP","BP","DS","ES","flags","cycleCount","haltCode","_irqActive","_tf","picPending","picMask","picInService","_irqBit","_pitFired","_kbdEdge","picVector","prevKeyboard","keyboard","pitCounter"];
                let slots: Vec<Option<usize>> = names.iter()
                    .map(|n| state.state_var_names.iter().position(|x| x == n))
                    .collect();
                let halt_idx = names.iter().position(|n| *n == "haltCode").unwrap();
                let halt_slot = slots[halt_idx];
                let cs_slot   = slots[0];
                let ip_slot   = slots[1];
                const RING: usize = 2048;
                let mut ring: Vec<(u32, Vec<i32>, u8, u8, u8, u8)> = Vec::with_capacity(RING + 4);
                // Fast-forward via run_batch if --trace-halt-skip was given.
                let skip = cli.trace_halt_skip.unwrap_or(0);
                let mut start_tick: u32 = 1;
                if skip > 0 && skip <= ticks_limit {
                    eprintln!("# trace-halt: fast-forwarding to tick {}", skip);
                    evaluator.run_batch(&mut state, skip);
                    let halt_now = halt_slot.map(|i| state.state_vars[i]).unwrap_or(0);
                    eprintln!("# trace-halt: post-skip haltCode={}", halt_now);
                    start_tick = skip + 1;
                }
                let mut prev_halt: i32 = halt_slot.map(|i| state.state_vars[i]).unwrap_or(0);
                eprintln!("# trace-halt: running tick-by-tick from {} to {}", start_tick, ticks_limit);
                for tick in start_tick..=ticks_limit {
                    apply_press_events(&press_events, tick, &mut state);
                    evaluator.tick(&mut state);
                    // Read each watched state-var.
                    let vals: Vec<i32> = slots.iter()
                        .map(|s| s.map(|i| state.state_vars[i]).unwrap_or(0))
                        .collect();
                    // Read 4 bytes at CS:IP for instruction context.
                    let cs = cs_slot.map(|i| state.state_vars[i]).unwrap_or(0);
                    let ip = ip_slot.map(|i| state.state_vars[i]).unwrap_or(0);
                    let lin = (cs as i32) * 16 + (ip as i32);
                    let b0 = (state.read_mem(lin)     & 0xFF) as u8;
                    let b1 = (state.read_mem(lin + 1) & 0xFF) as u8;
                    let b2 = (state.read_mem(lin + 2) & 0xFF) as u8;
                    let b3 = (state.read_mem(lin + 3) & 0xFF) as u8;
                    if ring.len() == RING { ring.remove(0); }
                    ring.push((tick, vals.clone(), b0, b1, b2, b3));
                    let halt = vals[halt_idx];
                    if prev_halt == 0 && halt != 0 {
                        eprintln!("\n=== haltCode 0 -> {} (0x{:02X}) at tick {} ===", halt, halt & 0xFF, tick);
                        eprintln!("(last {} ticks; columns: {})", ring.len(), names.join(","));
                        for (t, v, a, b, c, d) in &ring {
                            print!("tick={} ", t);
                            for (i, name) in names.iter().enumerate() {
                                print!("{}={} ", name, v[i]);
                            }
                            println!("bytes_at_csip={:02x} {:02x} {:02x} {:02x}", a, b, c, d);
                        }
                        return;
                    }
                    prev_halt = halt;
                }
                eprintln!("# trace-halt: ticks_limit reached ({}), no halt detected", ticks_limit);
                return;
            }

            // --trace-json: output JSON register trace
            if cli.trace_json {
                let halt_check = halt_addr;
                print!("[");
                for tick in 0..ticks_limit {
                    apply_press_events(&press_events, tick, &mut state);
                    evaluator.tick(&mut state);
                    if tick > 0 { print!(","); }
                    print!("{{\"tick\":{}", tick);
                    for (i, name) in state.state_var_names.iter().enumerate() {
                        print!(",\"{}\":{}", name, state.state_vars[i]);
                    }
                    print!("}}");
                    // Halt check for trace mode too
                    if let Some(addr) = halt_check {
                        if state.read_mem(addr) != 0 {
                            break;
                        }
                    }
                }
                println!("]");
                return;
            }



            let mut ticks_run: u32 = 0;
            let mut first_frame = true;
            let screen_interval = cli.screen_interval;
            let needs_per_tick = cli.verbose || halt_addr.is_some() || (interactive && screen_interval > 0) || !press_events.is_empty();

            // Rolling speed measurement: track ticks at the last render
            let mut last_render_time = t2;
            let mut last_render_ticks: u32 = 0;

            // Enable raw terminal mode for keyboard input
            if interactive {
                crossterm::terminal::enable_raw_mode().ok();
            }

            // Keyboard model: one BDA buffer push per physical key press event.
            // The OS already fires key-press events repeatedly while a key is
            // held (Windows autorepeat), so the terminal event stream handles
            // "hold to repeat" for us. We don't simulate any additional repeat
            // inside the loop — that caused games to receive multiple moves per
            // press. A future improvement is a proper simulated autorepeat
            // (500 ms initial delay, 100 ms interval) that watches for release
            // via a last-seen timer, but the naïve version is fine for now.

            if needs_per_tick {
                // Verbose mode prints every tick so it must stay per-tick; batch
                // early and show the tail.
                let verbose_skip = if cli.verbose && ticks_limit > 100 {
                    ticks_limit.saturating_sub(20)
                } else {
                    0
                };
                if verbose_skip > 0 {
                    evaluator.run_batch(&mut state, verbose_skip);
                    ticks_run = verbose_skip;
                    let ip = state.get_var("IP").unwrap_or(0);
                    eprintln!("(batch: {} ticks, IP={})", verbose_skip, ip);
                }

                // Batch size between keyboard polls / renders / halt checks.
                // At 4.77 MHz, 50K ticks ≈ 10.5 ms — well under human perception
                // for input latency. Clamp to screen_interval so renders still
                // happen when requested.
                let base_batch = if cli.interactive_batch == 0 { 50_000 } else { cli.interactive_batch };
                let batch_cap = if screen_interval > 0 {
                    base_batch.min(screen_interval)
                } else {
                    base_batch
                }.max(1);

                // Speed throttling: pace execution to cli.speed × 4.77 MHz using
                // the accumulated cycleCount as the "sim time" clock. After each
                // batch, sleep if we've outrun the schedule. speed=0 disables.
                let throttle = cli.speed > 0.0;
                let throttle_start_wall = std::time::Instant::now();
                let throttle_start_cycles = state.get_var("cycleCount").unwrap_or(0) as u64;
                let sim_hz = REAL_8086_HZ * cli.speed;

                let mut quit = false;
                let mut tick = verbose_skip;
                // Interactive keyboard: write to --keyboard so the press
                // edge fires IRQ 1, then schedule a release a few thousand
                // ticks later so the break code reaches programs that read
                // port 0x60 directly (DOOM hooks INT 9, reads `inp(0x60)`,
                // and tracks held keys via the high bit of the scancode —
                // it never looks at the BDA ring buffer). bda_push_key is
                // not enough on its own.
                const KBD_HOLD_TICKS: u32 = 5000;
                let mut pending_release_tick: Option<u32> = None;
                let mut held_selector: Option<&'static str> = None;
                while tick < ticks_limit {
                    if interactive {
                        // Poll keyboard non-blocking.
                        while event::poll(std::time::Duration::ZERO).unwrap_or(false) {
                            if let Ok(Event::Key(key_event)) = event::read() {
                                if key_event.kind == crossterm::event::KeyEventKind::Press {
                                    if key_event.modifiers.contains(KeyModifiers::CONTROL)
                                        && key_event.code == KeyCode::Char('c')
                                    {
                                        quit = true;
                                        break;
                                    }
                                    if let Some(sel) = key_to_selector(&key_event) {
                                        // Press: flip the (active, kb-X) edge.
                                        // The cabinet's
                                        // `&:has(#kb-X:active) { --keyboard: V }`
                                        // rule produces V via calcite's
                                        // input-edge recogniser; the engine's
                                        // edge detector then fires IRQ 1, etc.
                                        // BDA ring push covers INT 16h-only
                                        // programs that don't hook IRQ 1.
                                        state.set_pseudo_class_active("active", sel, true);
                                        let dos_key = key_to_dos(&key_event);
                                        if dos_key != 0 {
                                            state.bda_push_key(dos_key);
                                        }
                                        held_selector = Some(sel);
                                        pending_release_tick = Some(tick.saturating_add(KBD_HOLD_TICKS));
                                    }
                                }
                            }
                        }
                        if quit { break; }
                        // Fire scheduled release: pseudo edge inactive so the
                        // cabinet sees --keyboard return to 0 and
                        // `--_kbdRelease` fires.
                        if let Some(rt) = pending_release_tick {
                            if tick >= rt {
                                if let Some(sel) = held_selector.take() {
                                    state.set_pseudo_class_active("active", sel, false);
                                }
                                pending_release_tick = None;
                            }
                        }
                    }

                    // Determine this batch's size: capped by remaining ticks and
                    // by the next scripted press-event (if any).
                    let mut batch = batch_cap.min(ticks_limit - tick);
                    for (ev_tick, _, _) in &press_events {
                        if *ev_tick >= tick && *ev_tick < tick + batch {
                            batch = ev_tick - tick;
                            if batch == 0 { batch = 1; break; } // fire this tick immediately
                        }
                    }
                    // Cap on the pending interactive release tick too, so the
                    // release fires at the right time rather than after a
                    // 50K-tick batch races past it.
                    if let Some(rt) = pending_release_tick {
                        if rt >= tick && rt < tick + batch {
                            batch = (rt - tick).max(1);
                        }
                    }
                    if batch == 0 { batch = 1; }

                    // Fire any scripted press-events scheduled for this tick.
                    apply_press_events(&press_events, tick, &mut state);

                    // We push keys exactly once on press (in the event loop
                    // above). Re-pushing here would look like a repeat —
                    // rogue's INT 16h consumes the key each poll, so any refill
                    // registers as another keypress. Games handle their own key
                    // repeat; the CLI just delivers one press per physical key
                    // event.

                    evaluator.run_batch(&mut state, batch);
                    tick += batch;
                    ticks_run = tick;

                    if throttle {
                        let cycles_now = state.get_var("cycleCount").unwrap_or(0) as u64;
                        let sim_cycles = cycles_now.saturating_sub(throttle_start_cycles);
                        let target_wall = sim_cycles as f64 / sim_hz;
                        let actual_wall = throttle_start_wall.elapsed().as_secs_f64();
                        if target_wall > actual_wall {
                            let sleep_secs = target_wall - actual_wall;
                            std::thread::sleep(std::time::Duration::from_secs_f64(sleep_secs));
                        }
                    }

                    if cli.verbose {
                        print!("Tick {}:", tick - 1);
                        for (i, name) in state.state_var_names.iter().enumerate() {
                            print!(" {}={}", name, state.state_vars[i]);
                        }
                        println!();
                    }

                    if interactive && screen_interval > 0 && ticks_run % screen_interval == 0 {
                        let now = std::time::Instant::now();
                        let delta_secs = now.duration_since(last_render_time).as_secs_f64();
                        let delta_ticks = ticks_run - last_render_ticks;
                        render_screen(&state, ticks_run,
                                      delta_ticks, delta_secs, first_frame);
                        last_render_time = now;
                        last_render_ticks = ticks_run;
                        first_frame = false;
                    }

                    if let Some(addr) = halt_addr {
                        if state.read_mem(addr) != 0 {
                            eprintln!("Halt: memory 0x{:X} set at tick {}", addr, ticks_run);
                            break;
                        }
                    }
                }
            } else if !watch_registry.is_empty() {
                // Watch-driven runner. Advance in chunks; after each
                // chunk poll the registry against the current state.
                // Cheap watches (Stride/Burst/At/Halt) gate the
                // expensive ones (Cond/Edge) per the registry's two-
                // phase poll.
                //
                // We poll once per CHUNK_TICKS, not per individual
                // tick, to keep the per-tick overhead at zero. This
                // means stride watches with `every` smaller than
                // CHUNK_TICKS only fire at chunk boundaries  not the
                // exact stride boundary  but they still fire with
                // the right cadence on average. For exact-tick fires
                // (At, Halt) we shrink the next chunk to land on the
                // target tick. CHUNK_TICKS picked at 50_000  matches
                // the old --poll-stride default; ~10ms of guest time
                // at 8086 4.77 MHz, ~150ms wall on the slower wasm
                // runtime.
                const CHUNK_TICKS: u32 = 50_000;

                // Pre-compute exact-tick targets (from At watches) and
                // the smallest stride across stride-kind watches. The
                // chunk size shrinks to whichever target lands soonest
                // so cursor never overshoots a watch's exact firing
                // tick.
                let mut at_targets: Vec<u32> = Vec::new();
                let mut min_stride: u32 = u32::MAX;
                for spec_str in &cli.watches {
                    if let Ok(w) = watch_spec::parse_watch(spec_str) {
                        match w.kind {
                            calcite_core::script::WatchKind::At { tick: t } => {
                                at_targets.push(t);
                            }
                            calcite_core::script::WatchKind::Stride { every, .. } => {
                                if every > 0 && every < min_stride {
                                    min_stride = every;
                                }
                            }
                            calcite_core::script::WatchKind::Burst { every, .. } => {
                                if every > 0 && every < min_stride {
                                    min_stride = every;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                at_targets.sort_unstable();

                let mut cursor: u32 = 0;
                let mut halted = false;
                while cursor < ticks_limit && !halted {
                    let mut chunk = CHUNK_TICKS.min(ticks_limit - cursor);
                    if min_stride < chunk {
                        chunk = min_stride;
                    }
                    // Land exactly on the next At target if one is
                    // closer than the chunk size.
                    if let Ok(idx) = at_targets.binary_search(&(cursor + 1)) {
                        let t = at_targets[idx];
                        if t > cursor && t - cursor < chunk {
                            chunk = t - cursor;
                        }
                    } else if let Some(t) = at_targets.iter().find(|&&t| t > cursor) {
                        if *t - cursor < chunk {
                            chunk = *t - cursor;
                        }
                    }
                    if chunk == 0 { chunk = 1; }
                    evaluator.run_batch(&mut state, chunk);
                    cursor = cursor.saturating_add(chunk);
                    calcite_core::script_eval::poll(&mut watch_registry, &mut state, cursor);

                    let elapsed_ms = t2.elapsed().as_millis() as u64;
                    let cycles = state.get_var("cycleCount").unwrap_or(0);
                    for ev in watch_registry.drain_events() {
                        // Stderr summary line.
                        eprintln!(
                            "watch={} tick={} t_ms={} cycles={}{}{}",
                            ev.watch_name,
                            ev.tick,
                            elapsed_ms,
                            cycles,
                            if ev.halted { " halted=1" } else { "" },
                            if ev.sampled_vars.is_empty() {
                                String::new()
                            } else {
                                let pairs: Vec<String> = ev
                                    .sampled_vars
                                    .iter()
                                    .map(|(n, v)| format!(" {}={}", n, v))
                                    .collect();
                                pairs.join("")
                            },
                        );
                        // JSONL stream to --measure-out.
                        if let Some(out) = measure_out.as_mut() {
                            use std::io::Write;
                            let vars: Vec<String> = ev
                                .sampled_vars
                                .iter()
                                .map(|(n, v)| format!("[\"{}\",{}]", n, v))
                                .collect();
                            let dumps: Vec<String> = ev
                                .dumps
                                .iter()
                                .map(|d| {
                                    let path = d
                                        .path
                                        .as_deref()
                                        .map(|p| format!("\"{}\"", p.replace('\\', "\\\\")))
                                        .unwrap_or_else(|| "null".to_string());
                                    format!(
                                        "{{\"tag\":\"{}\",\"addr\":{},\"path\":{},\"len\":{}}}",
                                        d.tag,
                                        d.addr,
                                        path,
                                        d.bytes.len(),
                                    )
                                })
                                .collect();
                            let _ = writeln!(
                                out,
                                "{{\"tick\":{},\"watch\":\"{}\",\"halted\":{},\"vars\":[{}],\"dumps\":[{}]}}",
                                ev.tick,
                                ev.watch_name,
                                ev.halted,
                                vars.join(","),
                                dumps.join(","),
                            );
                        }
                    }

                    if watch_registry.halt_requested() {
                        halted = true;
                    }
                    if let Some(addr) = halt_addr {
                        if state.read_mem(addr) != 0 {
                            eprintln!("Halt: memory 0x{:X} set at tick {}", addr, cursor);
                            halted = true;
                        }
                    }
                }
                ticks_run = cursor;
                if let Some(out) = measure_out.as_mut() {
                    use std::io::Write;
                    let _ = out.flush();
                }
            } else {
                evaluator.run_batch(&mut state, ticks_limit);
                ticks_run = ticks_limit;
            }

            // Restore terminal
            if interactive {
                crossterm::terminal::disable_raw_mode().ok();
            }
            let tick_time = t2.elapsed();

            if !cli.verbose && cli.dump_tick.is_none() && !cli.trace_json {
                let ip = state.get_var("IP").unwrap_or(0);
                let cycles = state.get_var("cycleCount").unwrap_or(0) as u64;
                let elapsed = tick_time.as_secs_f64();
                let tps = if elapsed > 0.0 { ticks_run as f64 / elapsed } else { 0.0 };
                let cps = if elapsed > 0.0 { cycles as f64 / elapsed } else { 0.0 };
                let pct = cps / REAL_8086_HZ * 100.0;
                println!(
                    "{} ticks | {} cycles | {} ({:.1}% of 4.77 MHz) | {:.0} ticks/s | IP={}",
                    ticks_run,
                    cycles,
                    format_hz(cps),
                    pct,
                    tps,
                    ip,
                );
            }
            // Diagnostic: dump REP fast-forward fire/bail counts when
            // CALCITE_REP_DIAG is set. Cheap (~free) when not enabled.
            if std::env::var("CALCITE_REP_DIAG").is_ok() {
                eprint!("{}", calcite_core::compile::rep_diag_report());
            }
            eprintln!(
                "Elapsed: {:.3}s ({} ticks, {:.0} ticks/sec)",
                tick_time.as_secs_f64(),
                ticks_run,
                ticks_run as f64 / tick_time.as_secs_f64(),
            );
            // --op-profile: dump matrix CSV + print summary to stderr.
            if let Some(path) = &cli.op_profile_path {
                let snap = calcite_core::pattern::op_profile::op_profile_snapshot();
                if let Err(e) = snap.write_csv(path) {
                    eprintln!("--op-profile write {}: {}", path.display(), e);
                } else {
                    eprintln!("op-profile CSV written to {}", path.display());
                }
                eprint!("{}", snap.report_summary());
                calcite_core::pattern::op_profile::op_profile_disable();
            }

            // Display string property output (e.g., --textBuffer)
            for (name, value) in &state.string_properties {
                if !value.is_empty() {
                    println!("\n--{name}:\n{value}");
                }
            }

            // Final screen render (always, unless we just rendered this exact tick)
            if interactive {
                let just_rendered = screen_interval > 0 && ticks_run % screen_interval == 0;
                if !just_rendered {
                    let now = std::time::Instant::now();
                    let delta_secs = now.duration_since(last_render_time).as_secs_f64();
                    let delta_ticks = ticks_run - last_render_ticks;
                    render_screen(&state, ticks_run,
                                  delta_ticks, delta_secs, first_frame);
                }
                // Restore cursor visibility
                eprint!("\x1b[?25h");
            }

            // Write Mode 13h framebuffer to PPM if requested.
            if let Some(path) = cli.framebuffer_out.as_ref() {
                let video_mode = state.read_mem(0x0449) as u8;
                if video_mode == 0x13 {
                    let ppm = state.render_framebuffer(0xA0000, 320, 200);
                    match std::fs::write(path, &ppm) {
                        Ok(()) => eprintln!("Framebuffer: wrote {} bytes to {}", ppm.len(), path),
                        Err(e) => {
                            eprintln!("Failed to write framebuffer to {}: {}", path, e);
                            std::process::exit(1);
                        }
                    }
                } else {
                    eprintln!("--framebuffer-out: current video mode is 0x{video_mode:02X}, not Mode 13h — nothing written");
                }
            }

            // --dump-mem-range: write raw byte regions from guest memory to
            // disk so an external tool (e.g. tests/harness/fast-shoot.mjs)
            // can render screenshots without re-implementing video decode in
            // Rust or paying per-tick IPC cost through calcite-debugger.
            for spec in &cli.dump_mem_ranges {
                if let Err(e) = dump_mem_range(spec, &state) {
                    eprintln!("--dump-mem-range={spec}: {e}");
                    std::process::exit(1);
                }
            }

            // --snapshot-out: freeze the runtime-mutable state to disk so a
            // later --restore can resume at this exact tick. Pair with
            // --ticks N or a `:halt` script-event to land on a precise
            // moment (e.g. just-reached-in-game) before snapshotting.
            if let Some(path) = &cli.snapshot_out {
                let blob = state.snapshot();
                match std::fs::write(path, &blob) {
                    Ok(()) => eprintln!(
                        "snapshot: wrote {} bytes to {}",
                        blob.len(),
                        path.display()
                    ),
                    Err(e) => {
                        eprintln!("--snapshot-out={}: {}", path.display(), e);
                        std::process::exit(1);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Parse error: {e}");
            std::process::exit(1);
        }
    }
}
