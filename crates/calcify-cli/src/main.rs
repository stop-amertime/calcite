use clap::Parser;
use std::path::PathBuf;

/// calc(ify) — JIT compiler for computational CSS.
///
/// Parses CSS files, recognises computational patterns, and evaluates
/// them efficiently. Primary target: running x86CSS faster than Chrome.
#[derive(Parser, Debug)]
#[command(name = "calcify", version, about)]
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
    match calcify_core::parser::parse_css(&css) {
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
            let mut evaluator = calcify_core::Evaluator::from_parsed(&parsed);
            let compile_time = t1.elapsed();

            let mut state = calcify_core::State::default();
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
                    eprintln!("(batch: {} ticks, IP={})", batch_count, state.registers[calcify_core::state::reg::IP]);
                }
                for tick in batch_count..cli.ticks {
                    let result = evaluator.tick(&mut state);
                    println!(
                        "Tick {tick}: {} changes | AX={} CX={} DX={} BX={} SP={} BP={} SI={} DI={} IP={} ES={} CS={} SS={} DS={} flags={}",
                        result.changes.len(),
                        state.registers[calcify_core::state::reg::AX],
                        state.registers[calcify_core::state::reg::CX],
                        state.registers[calcify_core::state::reg::DX],
                        state.registers[calcify_core::state::reg::BX],
                        state.registers[calcify_core::state::reg::SP],
                        state.registers[calcify_core::state::reg::BP],
                        state.registers[calcify_core::state::reg::SI],
                        state.registers[calcify_core::state::reg::DI],
                        state.registers[calcify_core::state::reg::IP],
                        state.registers[calcify_core::state::reg::ES],
                        state.registers[calcify_core::state::reg::CS],
                        state.registers[calcify_core::state::reg::SS],
                        state.registers[calcify_core::state::reg::DS],
                        state.registers[calcify_core::state::reg::FLAGS],
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
                    state.registers[calcify_core::state::reg::AX],
                    state.registers[calcify_core::state::reg::CX],
                    state.registers[calcify_core::state::reg::IP],
                );
            }
            eprintln!(
                "Ticks: {:.3}s ({:.0} ticks/sec)",
                tick_time.as_secs_f64(),
                cli.ticks as f64 / tick_time.as_secs_f64(),
            );
        }
        Err(e) => {
            eprintln!("Parse error: {e}");
            std::process::exit(1);
        }
    }
}
