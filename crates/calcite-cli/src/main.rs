use clap::Parser;
use std::path::PathBuf;

/// calc(ite) — JIT compiler for computational CSS.
///
/// Parses CSS files, recognises computational patterns, and evaluates
/// them efficiently. Primary target: running x86CSS faster than Chrome.
#[derive(Parser, Debug)]
#[command(name = "calcite", version, about)]
struct Cli {
    /// Path to the CSS file to evaluate.
    #[arg(short, long)]
    input: PathBuf,

    /// Number of ticks to run.
    #[arg(short = 'n', long, default_value = "1")]
    ticks: u32,

    /// Print register state after each tick.
    #[arg(short, long)]
    verbose: bool,

    /// Only parse and analyse patterns (don't evaluate).
    #[arg(long)]
    parse_only: bool,

    /// Render a region of memory as a text-mode screen after execution.
    ///
    /// Format: ADDR WIDTHxHEIGHT (e.g. "0xB8000 40x25").
    /// Memory is read in text-mode format (char+attribute byte pairs).
    #[arg(long, value_name = "ADDR WxH", num_args = 2)]
    screen: Option<Vec<String>>,

    /// Dump all computed slot values at a specific tick for debugging.
    ///
    /// Runs to the specified tick, then prints every property name and its
    /// computed value. Useful for diagnosing divergences found by compare.mjs.
    #[arg(long, value_name = "TICK")]
    dump_tick: Option<u32>,

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
}

fn parse_screen_args(args: &[String]) -> (i32, usize, usize) {
    if args.len() != 2 {
        eprintln!("--screen requires ADDR WxH (e.g. --screen 0xB8000 40x25)");
        std::process::exit(1);
    }
    let addr = if args[0].starts_with("0x") || args[0].starts_with("0X") {
        i32::from_str_radix(&args[0][2..], 16).unwrap_or_else(|_| {
            eprintln!("Invalid address '{}', expected hex (e.g. 0xB8000)", args[0]);
            std::process::exit(1);
        })
    } else {
        args[0].parse().unwrap_or_else(|_| {
            eprintln!("Invalid address '{}', expected integer or hex", args[0]);
            std::process::exit(1);
        })
    };
    if let Some((w, h)) = args[1].split_once('x') {
        if let (Ok(w), Ok(h)) = (w.parse(), h.parse()) {
            return (addr, w, h);
        }
    }
    eprintln!(
        "Invalid screen dimensions '{}', expected WxH (e.g. 40x25)",
        args[1]
    );
    std::process::exit(1);
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    let css = match std::fs::read_to_string(&cli.input) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading {}: {e}", cli.input.display());
            std::process::exit(1);
        }
    };

    log::info!(
        "Read {} bytes of CSS from {}",
        css.len(),
        cli.input.display()
    );

    let t0 = std::time::Instant::now();
    match calcite_core::parser::parse_css(&css) {
        Ok(parsed) => {
            let parse_time = t0.elapsed();
            println!(
                "Parsed: {} @property, {} @function, {} assignments ({:.2}s)",
                parsed.properties.len(),
                parsed.functions.len(),
                parsed.assignments.len(),
                parse_time.as_secs_f64(),
            );

            if cli.parse_only {
                return;
            }

            let t1 = std::time::Instant::now();
            let mut evaluator = calcite_core::Evaluator::from_parsed(&parsed);
            let compile_time = t1.elapsed();

            let mut state = calcite_core::State::default();
            state.load_properties(&parsed.properties);

            eprintln!(
                "Compiled: {:.2}s (parse {:.2}s + compile {:.2}s)",
                (parse_time + compile_time).as_secs_f64(),
                parse_time.as_secs_f64(),
                compile_time.as_secs_f64(),
            );

            let halt_addr: Option<i32> = cli.halt.as_ref().map(|s| {
                if s.starts_with("0x") || s.starts_with("0X") {
                    i32::from_str_radix(&s[2..], 16).expect("Invalid halt address")
                } else {
                    s.parse().expect("Invalid halt address")
                }
            });

            let t2 = std::time::Instant::now();

            // --dump-tick: run to specific tick and dump all computed properties
            if let Some(target_tick) = cli.dump_tick {
                if target_tick > 0 {
                    evaluator.run_batch(&mut state, target_tick);
                }
                evaluator.tick(&mut state);
                println!("=== Slot dump at tick {} ===", target_tick);
                println!(
                    "Registers: AX={} CX={} DX={} BX={} SP={} BP={} SI={} DI={} IP={} ES={} CS={} SS={} DS={} FLAGS={}",
                    state.registers[calcite_core::state::reg::AX],
                    state.registers[calcite_core::state::reg::CX],
                    state.registers[calcite_core::state::reg::DX],
                    state.registers[calcite_core::state::reg::BX],
                    state.registers[calcite_core::state::reg::SP],
                    state.registers[calcite_core::state::reg::BP],
                    state.registers[calcite_core::state::reg::SI],
                    state.registers[calcite_core::state::reg::DI],
                    state.registers[calcite_core::state::reg::IP],
                    state.registers[calcite_core::state::reg::ES],
                    state.registers[calcite_core::state::reg::CS],
                    state.registers[calcite_core::state::reg::SS],
                    state.registers[calcite_core::state::reg::DS],
                    state.registers[calcite_core::state::reg::FLAGS],
                );
                // Dump key computed properties
                let props = [
                    // v2 decode properties
                    "--csBase", "--ipAddr", "--q0", "--q1", "--q2", "--q3",
                    "--opcode", "--mod", "--reg", "--rm",
                    "--modrmExtra", "--ea", "--eaSeg", "--eaOff",
                    "--immOff", "--immByte", "--immWord", "--imm8", "--imm16",
                    "--rmVal8", "--rmVal16", "--regVal8", "--regVal16",
                    "--memAddr0", "--memVal0", "--memAddr1", "--memVal1",
                    "--memAddr2", "--memVal2",
                    // v1 legacy properties
                    "--addrDestA", "--addrDestB", "--addrDestC",
                    "--addrValA", "--addrValA1", "--addrValA2", "--addrValB", "--addrValC",
                    "--isWordWrite", "--instId", "--instLen",
                    "--modRm", "--modRm_mod", "--modRm_reg", "--modRm_rm",
                    "--moveStack", "--moveSI", "--moveDI", "--jumpCS",
                    // registers
                    "--AX", "--CX", "--DX", "--BX",
                    "--AL", "--CL", "--DL", "--BL",
                    "--AH", "--CH", "--DH", "--BH",
                    "--SP", "--BP", "--SI", "--DI", "--IP",
                    "--ES", "--CS", "--SS", "--DS", "--flags", "--halt",
                ];
                println!("\nComputed properties:");
                for name in &props {
                    if let Some(val) = evaluator.get_slot_value(name) {
                        println!("  {}: {} (0x{:X})", name, val, val as u32);
                    }
                }
                return;
            }

            // --trace-json: output JSON register trace
            if cli.trace_json {
                print!("[");
                for tick in 0..cli.ticks {
                    evaluator.tick(&mut state);
                    if tick > 0 { print!(","); }
                    print!(
                        "{{\"tick\":{},\"AX\":{},\"CX\":{},\"DX\":{},\"BX\":{},\"SP\":{},\"BP\":{},\"SI\":{},\"DI\":{},\"IP\":{},\"ES\":{},\"CS\":{},\"SS\":{},\"DS\":{},\"FLAGS\":{}}}",
                        tick,
                        state.registers[calcite_core::state::reg::AX],
                        state.registers[calcite_core::state::reg::CX],
                        state.registers[calcite_core::state::reg::DX],
                        state.registers[calcite_core::state::reg::BX],
                        state.registers[calcite_core::state::reg::SP],
                        state.registers[calcite_core::state::reg::BP],
                        state.registers[calcite_core::state::reg::SI],
                        state.registers[calcite_core::state::reg::DI],
                        state.registers[calcite_core::state::reg::IP],
                        state.registers[calcite_core::state::reg::ES],
                        state.registers[calcite_core::state::reg::CS],
                        state.registers[calcite_core::state::reg::SS],
                        state.registers[calcite_core::state::reg::DS],
                        state.registers[calcite_core::state::reg::FLAGS],
                    );
                }
                println!("]");
                return;
            }

            let mut ticks_run: u32 = 0;
            if cli.verbose {
                // In verbose mode, batch the early ticks, then show
                // per-tick output for the tail. Small counts show all.
                let batch_count = if cli.ticks <= 100 { 0 } else { cli.ticks.saturating_sub(20) };
                if batch_count > 0 {
                    evaluator.run_batch(&mut state, batch_count);
                    ticks_run = batch_count;
                    eprintln!(
                        "(batch: {} ticks, IP={})",
                        batch_count,
                        state.registers[calcite_core::state::reg::IP]
                    );
                }
                for tick in batch_count..cli.ticks {
                    let result = evaluator.tick(&mut state);
                    ticks_run = tick + 1;
                    println!(
                        "Tick {tick}: {} changes | AX={} CX={} DX={} BX={} SP={} BP={} SI={} DI={} IP={} ES={} CS={} SS={} DS={} flags={}",
                        result.changes.len(),
                        state.registers[calcite_core::state::reg::AX],
                        state.registers[calcite_core::state::reg::CX],
                        state.registers[calcite_core::state::reg::DX],
                        state.registers[calcite_core::state::reg::BX],
                        state.registers[calcite_core::state::reg::SP],
                        state.registers[calcite_core::state::reg::BP],
                        state.registers[calcite_core::state::reg::SI],
                        state.registers[calcite_core::state::reg::DI],
                        state.registers[calcite_core::state::reg::IP],
                        state.registers[calcite_core::state::reg::ES],
                        state.registers[calcite_core::state::reg::CS],
                        state.registers[calcite_core::state::reg::SS],
                        state.registers[calcite_core::state::reg::DS],
                        state.registers[calcite_core::state::reg::FLAGS],
                    );
                    if let Some(addr) = halt_addr {
                        if state.read_mem(addr) != 0 {
                            eprintln!("Halt: memory 0x{:X} set at tick {}", addr, tick);
                            break;
                        }
                    }
                }
            } else if let Some(addr) = halt_addr {
                for tick in 0..cli.ticks {
                    evaluator.tick(&mut state);
                    ticks_run = tick + 1;
                    if state.read_mem(addr) != 0 {
                        eprintln!("Halt: memory 0x{:X} set at tick {}", addr, tick);
                        break;
                    }
                }
            } else {
                evaluator.run_batch(&mut state, cli.ticks);
                ticks_run = cli.ticks;
            }
            let tick_time = t2.elapsed();

            if !cli.verbose && cli.dump_tick.is_none() && !cli.trace_json {
                println!(
                    "Ran {} ticks | AX={} CX={} IP={}",
                    ticks_run,
                    state.registers[calcite_core::state::reg::AX],
                    state.registers[calcite_core::state::reg::CX],
                    state.registers[calcite_core::state::reg::IP],
                );
            }
            eprintln!(
                "Ticks: {:.3}s ({:.0} ticks/sec)",
                tick_time.as_secs_f64(),
                ticks_run as f64 / tick_time.as_secs_f64(),
            );

            // Display string property output (e.g., --textBuffer)
            for (name, value) in &state.string_properties {
                if !value.is_empty() {
                    println!("\n--{name}:\n{value}");
                }
            }

            if let Some(ref args) = cli.screen {
                let (addr, width, height) = parse_screen_args(args);
                let screen = state.render_screen(addr as usize, width, height);
                println!("\n┌{}┐", "─".repeat(width));
                for line in screen.lines() {
                    println!("│{line}│");
                }
                println!("└{}┘", "─".repeat(width));
            }
        }
        Err(e) => {
            eprintln!("Parse error: {e}");
            std::process::exit(1);
        }
    }
}
