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

            let t2 = std::time::Instant::now();
            if cli.verbose {
                // In verbose mode, use run_batch for the first 90% of ticks,
                // then switch to per-tick output for the last 10%.
                let batch_count = cli.ticks.saturating_sub(20);
                if batch_count > 0 {
                    evaluator.run_batch(&mut state, batch_count);
                    eprintln!(
                        "(batch: {} ticks, IP={})",
                        batch_count,
                        state.registers[calcite_core::state::reg::IP]
                    );
                }
                for tick in batch_count..cli.ticks {
                    let result = evaluator.tick(&mut state);
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
                }
            } else {
                evaluator.run_batch(&mut state, cli.ticks);
            }
            let tick_time = t2.elapsed();

            if !cli.verbose {
                println!(
                    "Ran {} ticks | AX={} CX={} IP={}",
                    cli.ticks,
                    state.registers[calcite_core::state::reg::AX],
                    state.registers[calcite_core::state::reg::CX],
                    state.registers[calcite_core::state::reg::IP],
                );
            }
            eprintln!(
                "Ticks: {:.3}s ({:.0} ticks/sec)",
                tick_time.as_secs_f64(),
                cli.ticks as f64 / tick_time.as_secs_f64(),
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
