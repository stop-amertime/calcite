//! calcite-bench — headless benchmark runner for measuring evaluation speed.
//!
//! Runs a CSS program for N ticks with no rendering, keyboard, or terminal overhead.
//! Reports: ticks/sec, cycles/sec, time breakdown per phase, and per-tick histogram.
//!
//! Usage:
//!   cargo run --release --bin calcite-bench -- -i program.css -n 10000
//!   cargo run --release --bin calcite-bench -- -i program.css -n 10000 --phase-breakdown
//!   cargo run --release --bin calcite-bench -- -i program.css -n 10000 --warmup 500

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug)]
#[command(name = "calcite-bench", version, about = "Headless benchmark runner for calc(ite)")]
struct Args {
    /// Path to the CSS file to evaluate.
    #[arg(short, long)]
    input: PathBuf,

    /// Number of ticks to benchmark.
    #[arg(short = 'n', long, default_value = "10000")]
    ticks: u32,

    /// Number of warmup ticks before measurement starts.
    #[arg(long, default_value = "0")]
    warmup: u32,

    /// Show per-phase time breakdown (parse, compile, execute).
    #[arg(long)]
    phase_breakdown: bool,

    /// Show per-tick timing histogram (buckets of tick durations).
    #[arg(long)]
    histogram: bool,

    /// Profile per-phase time breakdown within each tick.
    ///
    /// Shows where time is spent: bytecode execution, state snapshot,
    /// string eval, change detection, hooks.
    #[arg(long)]
    profile: bool,

    /// Halt when memory at ADDR becomes non-zero.
    #[arg(long, value_name = "ADDR")]
    halt: Option<String>,

    /// Run multiple iterations and show statistics.
    #[arg(long, default_value = "1")]
    iterations: u32,

    /// Batch size for run_batch (0 = single-tick loop for profiling).
    #[arg(long, default_value = "0")]
    batch: u32,

    /// Restore a snapshot blob (produced by `calcite-cli --snapshot-out`)
    /// into the engine **before** the benchmark window. Use this to skip
    /// boot/menu/loading and only measure a specific stage. Cabinet must
    /// match the one used to produce the snapshot.
    #[arg(long, value_name = "PATH")]
    restore: Option<PathBuf>,
}

const REAL_8086_HZ: f64 = 4_772_727.0;

fn format_hz(hz: f64) -> String {
    if hz >= 1_000_000.0 {
        format!("{:.2} MHz", hz / 1_000_000.0)
    } else if hz >= 1_000.0 {
        format!("{:.1} KHz", hz / 1_000.0)
    } else {
        format!("{:.0} Hz", hz)
    }
}

fn format_duration(secs: f64) -> String {
    if secs >= 1.0 {
        format!("{:.3}s", secs)
    } else if secs >= 0.001 {
        format!("{:.2}ms", secs * 1_000.0)
    } else {
        format!("{:.1}us", secs * 1_000_000.0)
    }
}

struct BenchResult {
    ticks: u32,
    cycles: u64,
    elapsed: f64,
    tick_durations: Option<Vec<f64>>, // per-tick durations in seconds, if histogram mode
    profile: Option<AggregateProfile>,
}

/// Aggregated per-phase timing across many ticks.
#[derive(Default)]
struct AggregateProfile {
    // Top-level
    hooks: f64,
    snapshot: f64,
    change_detect: f64,
    string_eval: f64,
    // Inside execute
    slot_reset: f64,
    main_ops: f64,
    dispatch_time: f64,
    linear_ops: f64,
    writeback: f64,
    broadcast: f64,
    // Counters
    main_ops_count: u64,
    dispatch_count: u64,
    dispatch_sub_ops_count: u64,
    branches_taken: u64,
    branches_not_taken: u64,
    broadcast_fire_count: u64,
    broadcast_total_count: u64,
    writeback_count: u64,
    writeback_changed_count: u64,
    op_counts: std::collections::HashMap<&'static str, u64>,
    count: u32,
}

impl AggregateProfile {
    fn add(&mut self, p: &calcite_core::TickProfile) {
        self.hooks += p.hooks;
        self.snapshot += p.snapshot;
        self.change_detect += p.change_detect;
        self.string_eval += p.string_eval;
        self.slot_reset += p.slot_reset;
        self.main_ops += p.main_ops;
        self.dispatch_time += p.dispatch_time;
        self.linear_ops += p.linear_ops;
        self.writeback += p.writeback;
        self.broadcast += p.broadcast;
        self.main_ops_count += p.main_ops_count;
        self.dispatch_count += p.dispatch_count;
        self.dispatch_sub_ops_count += p.dispatch_sub_ops_count;
        self.branches_taken += p.branches_taken;
        self.branches_not_taken += p.branches_not_taken;
        self.broadcast_fire_count += p.broadcast_fire_count;
        self.broadcast_total_count += p.broadcast_total_count;
        self.writeback_count += p.writeback_count;
        self.writeback_changed_count += p.writeback_changed_count;
        for (k, v) in &p.op_counts {
            *self.op_counts.entry(k).or_insert(0) += v;
        }
        self.count += 1;
    }

    fn total(&self) -> f64 {
        self.hooks + self.snapshot + self.change_detect + self.string_eval
            + self.slot_reset + self.main_ops + self.writeback + self.broadcast
    }
}

fn run_bench(
    evaluator: &mut calcite_core::Evaluator,
    state: &mut calcite_core::State,
    ticks: u32,
    halt_addr: Option<i32>,
    batch: u32,
    collect_histogram: bool,
    collect_profile: bool,
) -> BenchResult {
    let mut tick_durations = if collect_histogram {
        Some(Vec::with_capacity(ticks as usize))
    } else {
        None
    };
    let mut profile = if collect_profile {
        Some(AggregateProfile::default())
    } else {
        None
    };

    let start = Instant::now();
    let mut ticks_run: u32 = 0;

    if batch > 0 && !collect_histogram && !collect_profile {
        // Batch mode: minimal overhead. Halt (if set) is checked between
        // batches — still near full speed, small overshoot bounded by `batch`.
        let full_batches = ticks / batch;
        let remainder = ticks % batch;
        let mut halted = false;
        for _ in 0..full_batches {
            evaluator.run_batch(state, batch);
            ticks_run += batch;
            if let Some(addr) = halt_addr {
                if state.read_mem(addr) != 0 {
                    halted = true;
                    break;
                }
            }
        }
        if !halted && remainder > 0 {
            evaluator.run_batch(state, remainder);
            ticks_run += remainder;
        }
    } else if collect_profile {
        // Profile mode: use tick_profiled for per-phase timing
        for _ in 0..ticks {
            let (_result, tick_profile) = evaluator.tick_profiled(state);
            if let Some(ref mut agg) = profile {
                agg.add(&tick_profile);
            }
            ticks_run += 1;

            if let Some(addr) = halt_addr {
                if state.read_mem(addr) != 0 {
                    break;
                }
            }
        }
    } else {
        // Single-tick mode: best for profiling (every tick is visible in flamegraphs)
        for _ in 0..ticks {
            let tick_start = if collect_histogram {
                Some(Instant::now())
            } else {
                None
            };

            evaluator.tick(state);
            ticks_run += 1;

            if let Some(durations) = &mut tick_durations {
                durations.push(tick_start.unwrap().elapsed().as_secs_f64());
            }

            if let Some(addr) = halt_addr {
                if state.read_mem(addr) != 0 {
                    break;
                }
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let cycles = state.get_var("cycleCount").unwrap_or(0) as u64;

    BenchResult {
        ticks: ticks_run,
        cycles,
        elapsed,
        tick_durations,
        profile,
    }
}

fn print_histogram(durations: &[f64]) {
    if durations.is_empty() {
        return;
    }

    let mut sorted = durations.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = sorted[sorted.len() / 2];
    let p90 = sorted[(sorted.len() as f64 * 0.9) as usize];
    let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let mean = durations.iter().sum::<f64>() / durations.len() as f64;

    eprintln!("\n--- Per-tick timing ---");
    eprintln!(
        "  min: {}  mean: {}  p50: {}  p90: {}  p99: {}  max: {}",
        format_duration(min),
        format_duration(mean),
        format_duration(p50),
        format_duration(p90),
        format_duration(p99),
        format_duration(max),
    );

    // Simple ASCII histogram with 10 buckets
    let bucket_count = 10;
    let range = max - min;
    if range == 0.0 {
        eprintln!("  (all ticks identical)");
        return;
    }
    let bucket_width = range / bucket_count as f64;
    let mut buckets = vec![0u32; bucket_count];
    for &d in durations {
        let idx = ((d - min) / bucket_width).floor() as usize;
        let idx = idx.min(bucket_count - 1);
        buckets[idx] += 1;
    }
    let max_count = *buckets.iter().max().unwrap();
    let bar_width = 40;

    eprintln!();
    for (i, &count) in buckets.iter().enumerate() {
        let lo = min + i as f64 * bucket_width;
        let hi = lo + bucket_width;
        let bar_len = if max_count > 0 {
            (count as f64 / max_count as f64 * bar_width as f64) as usize
        } else {
            0
        };
        eprintln!(
            "  {:>8} - {:>8} | {:<width$} {} ({:.0}%)",
            format_duration(lo),
            format_duration(hi),
            "#".repeat(bar_len),
            count,
            count as f64 / durations.len() as f64 * 100.0,
            width = bar_width,
        );
    }
}

fn print_profile(prof: &AggregateProfile) {
    let total = prof.total();
    if total == 0.0 || prof.count == 0 {
        return;
    }

    let n = prof.count as f64;
    let pct = |v: f64| v / total * 100.0;
    let avg = |v: f64| v / n;

    eprintln!("\n--- Per-phase breakdown ({} ticks) ---", prof.count);
    eprintln!(
        "  {:20} {:>10} {:>10} {:>6}",
        "Phase", "Total", "Avg/tick", "%"
    );
    eprintln!("  {}", "-".repeat(52));

    // Time breakdown table
    let phases: &[(&str, f64)] = &[
        ("  Dispatch lookups", prof.dispatch_time),
        ("  Linear ops", prof.linear_ops),
        ("Writeback", prof.writeback),
        ("Broadcast writes", prof.broadcast),
        ("Slot reset", prof.slot_reset),
        ("Change detect", prof.change_detect),
        ("Snapshot", prof.snapshot),
        ("String eval", prof.string_eval),
        ("Hooks", prof.hooks),
    ];

    // Sort by time descending
    let mut phases: Vec<_> = phases.to_vec();
    phases.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (name, time) in &phases {
        if pct(*time) < 0.05 && *time < 0.001 {
            continue; // skip noise
        }
        eprintln!(
            "  {:22} {:>10} {:>10} {:>5.1}%",
            name,
            format_duration(*time),
            format_duration(avg(*time)),
            pct(*time),
        );
    }
    eprintln!("  {}", "-".repeat(54));
    eprintln!(
        "  {:22} {:>10} {:>10}",
        "TOTAL",
        format_duration(total),
        format_duration(avg(total)),
    );

    // Counters
    let total_branches = prof.branches_taken + prof.branches_not_taken;
    eprintln!("\n--- Op counters (per tick avg) ---");
    eprintln!(
        "  Main-stream ops:      {:.0}",
        prof.main_ops_count as f64 / n
    );
    eprintln!(
        "  Dispatch lookups:     {:.0} (avg {:.0} sub-ops each)",
        prof.dispatch_count as f64 / n,
        if prof.dispatch_count > 0 {
            prof.dispatch_sub_ops_count as f64 / prof.dispatch_count as f64
        } else {
            0.0
        },
    );
    eprintln!(
        "  Branches:             {:.0} taken / {:.0} not taken ({:.0}% taken)",
        prof.branches_taken as f64 / n,
        prof.branches_not_taken as f64 / n,
        if total_branches > 0 {
            prof.branches_taken as f64 / total_branches as f64 * 100.0
        } else {
            0.0
        },
    );
    eprintln!(
        "  Writeback entries:    {:.0} ({:.0} changed)",
        prof.writeback_count as f64 / n,
        prof.writeback_changed_count as f64 / n,
    );
    eprintln!(
        "  Broadcast writes:     {:.0} checked, {:.0} fired",
        prof.broadcast_total_count as f64 / n,
        prof.broadcast_fire_count as f64 / n,
    );

    // Op-type frequency table
    if !prof.op_counts.is_empty() {
        eprintln!("\n--- Op frequency (per tick avg, sorted) ---");
        let mut ops: Vec<_> = prof.op_counts.iter().collect();
        ops.sort_by(|a, b| b.1.cmp(a.1));
        let total_ops: u64 = ops.iter().map(|(_, v)| **v).sum();
        for (name, count) in &ops {
            let per_tick = **count as f64 / n;
            let pct_of_ops = **count as f64 / total_ops as f64 * 100.0;
            if per_tick < 1.0 {
                continue;
            }
            eprintln!(
                "  {:16} {:>8.0} {:>5.1}%",
                name, per_tick, pct_of_ops,
            );
        }
    }

    // Derived metrics
    eprintln!("\n--- Derived ---");
    if prof.main_ops_count > 0 {
        let ns_per_linear_op = if prof.main_ops_count > prof.dispatch_count {
            prof.linear_ops / (prof.main_ops_count - prof.dispatch_count) as f64
                * 1_000_000_000.0
        } else {
            0.0
        };
        eprintln!("  ns/linear-op:         {:.1}", ns_per_linear_op);
    }
    if prof.dispatch_count > 0 {
        let us_per_dispatch = prof.dispatch_time / prof.dispatch_count as f64 * 1_000_000.0;
        eprintln!("  us/dispatch:          {:.1}", us_per_dispatch);
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    // --- Parse ---
    let t_parse_start = Instant::now();
    let css = match std::fs::read_to_string(&args.input) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading {}: {e}", args.input.display());
            std::process::exit(1);
        }
    };
    let parsed = match calcite_core::parser::parse_css(&css) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Parse error: {e}");
            std::process::exit(1);
        }
    };
    let parse_time = t_parse_start.elapsed().as_secs_f64();

    // --- Compile ---
    let t_compile_start = Instant::now();
    let mut state = calcite_core::State::default();
    state.load_properties(&parsed.properties);
    let mut evaluator = calcite_core::Evaluator::from_parsed(&parsed);
    evaluator.wire_state_for_packed_memory(&mut state);
    evaluator.wire_state_for_windowed_byte_array(&mut state);
    evaluator.wire_state_for_input_edges(&mut state);
    if let Some(path) = &args.restore {
        match std::fs::read(path) {
            Ok(bytes) => match state.restore(&bytes) {
                Ok(()) => eprintln!("restored snapshot from {} ({} bytes)", path.display(), bytes.len()),
                Err(e) => { eprintln!("--restore={}: {}", path.display(), e); std::process::exit(2); }
            },
            Err(e) => { eprintln!("--restore={}: {}", path.display(), e); std::process::exit(2); }
        }
    }
    let compile_time = t_compile_start.elapsed().as_secs_f64();

    let halt_addr: Option<i32> = args.halt.as_ref().map(|s| {
        if s.starts_with("0x") || s.starts_with("0X") {
            i32::from_str_radix(&s[2..], 16).expect("Invalid halt address")
        } else {
            s.parse().expect("Invalid halt address")
        }
    });

    if args.phase_breakdown {
        eprintln!("--- Phase breakdown ---");
        eprintln!("  Parse:   {} ({} props, {} fns, {} assignments)",
            format_duration(parse_time),
            parsed.properties.len(),
            parsed.functions.len(),
            parsed.assignments.len(),
        );
        eprintln!("  Compile: {} (memory: {} KB)",
            format_duration(compile_time),
            state.memory.len() / 1024,
        );
    }

    // --- Warmup ---
    // Use run_batch (no per-tick state-vars clone). The slow tick() path
    // is ~100x slower than run_batch on dense cabinets like doom8088, so
    // a 5M warmup that should take ~13s would otherwise take ~22 minutes.
    if args.warmup > 0 {
        let t_warmup = Instant::now();
        let warmup_batch = args.batch.max(1);
        let mut remaining = args.warmup;
        while remaining > 0 {
            let n = remaining.min(warmup_batch);
            evaluator.run_batch(&mut state, n);
            remaining -= n;
        }
        let warmup_time = t_warmup.elapsed().as_secs_f64();
        if args.phase_breakdown {
            eprintln!("  Warmup:  {} ({} ticks, {:.0} ticks/s)",
                format_duration(warmup_time),
                args.warmup,
                args.warmup as f64 / warmup_time,
            );
        }
    }

    // --- Benchmark ---
    let mut results: Vec<BenchResult> = Vec::new();

    for iter in 0..args.iterations {
        // For iterations > 1, reset state each time
        if iter > 0 {
            state = calcite_core::State::default();
            state.load_properties(&parsed.properties);
            evaluator = calcite_core::Evaluator::from_parsed(&parsed);
            evaluator.wire_state_for_packed_memory(&mut state);
            evaluator.wire_state_for_windowed_byte_array(&mut state);
            evaluator.wire_state_for_input_edges(&mut state);
            if let Some(path) = &args.restore {
                if let Ok(bytes) = std::fs::read(path) {
                    state.restore(&bytes).ok();
                }
            }
            if args.warmup > 0 {
                let warmup_batch = args.batch.max(1);
                let mut remaining = args.warmup;
                while remaining > 0 {
                    let n = remaining.min(warmup_batch);
                    evaluator.run_batch(&mut state, n);
                    remaining -= n;
                }
            }
        }

        let result = run_bench(
            &mut evaluator,
            &mut state,
            args.ticks,
            halt_addr,
            args.batch,
            args.histogram && iter == 0, // only collect histogram on first iteration
            args.profile && iter == 0,   // only collect profile on first iteration
        );
        results.push(result);
    }

    // --- Report ---
    eprintln!();
    if results.len() == 1 {
        let r = &results[0];
        let tps = r.ticks as f64 / r.elapsed;
        let avg_cpt = if r.ticks > 0 { r.cycles as f64 / r.ticks as f64 } else { 1.0 };
        let cps = tps * avg_cpt;
        let pct = cps / REAL_8086_HZ * 100.0;

        eprintln!("=== Benchmark Results ===");
        eprintln!("  Ticks:      {}", r.ticks);
        eprintln!("  Cycles:     {}", r.cycles);
        eprintln!("  Elapsed:    {}", format_duration(r.elapsed));
        eprintln!("  Ticks/sec:  {:.0}", tps);
        eprintln!("  Cycles/sec: {} ({:.1}% of 4.77 MHz 8086)", format_hz(cps), pct);
        eprintln!("  Avg cyc/tick: {:.1}", avg_cpt);
        eprintln!("  Avg us/tick:  {:.1}", r.elapsed / r.ticks as f64 * 1_000_000.0);

        // Machine-readable one-liner on stdout
        println!(
            "{} ticks | {} cycles | {} ({:.1}% of 4.77 MHz) | {:.0} ticks/s | {}/tick",
            r.ticks,
            r.cycles,
            format_hz(cps),
            pct,
            tps,
            format_duration(r.elapsed / r.ticks as f64),
        );

        if let Some(ref durations) = r.tick_durations {
            print_histogram(durations);
        }
        if let Some(ref prof) = r.profile {
            print_profile(prof);
        }
    } else {
        let tps_values: Vec<f64> = results
            .iter()
            .map(|r| r.ticks as f64 / r.elapsed)
            .collect();
        let mean_tps = tps_values.iter().sum::<f64>() / tps_values.len() as f64;
        let variance = tps_values
            .iter()
            .map(|v| (v - mean_tps).powi(2))
            .sum::<f64>()
            / tps_values.len() as f64;
        let stddev = variance.sqrt();

        let r0 = &results[0];
        let avg_cpt = if r0.ticks > 0 {
            r0.cycles as f64 / r0.ticks as f64
        } else {
            1.0
        };
        let mean_cps = mean_tps * avg_cpt;
        let pct = mean_cps / REAL_8086_HZ * 100.0;

        eprintln!("=== Benchmark Results ({} iterations) ===", results.len());
        eprintln!("  Ticks/iter: {}", r0.ticks);
        eprintln!(
            "  Ticks/sec:  {:.0} +/- {:.0} ({:.1}% CV)",
            mean_tps,
            stddev,
            stddev / mean_tps * 100.0,
        );
        eprintln!("  Cycles/sec: {} ({:.1}% of 4.77 MHz)", format_hz(mean_cps), pct);

        println!(
            "{:.0} +/- {:.0} ticks/s | {} ({:.1}% of 4.77 MHz)",
            mean_tps,
            stddev,
            format_hz(mean_cps),
            pct,
        );

        if let Some(ref durations) = results[0].tick_durations {
            print_histogram(durations);
        }
        if let Some(ref prof) = results[0].profile {
            print_profile(prof);
        }
    }
}
