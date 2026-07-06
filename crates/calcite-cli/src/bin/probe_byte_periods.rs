//! Scan a cabinet's rom-disk byte array for periodic regions and report
//! findings. The first cut of the byte-pattern matcher leg of the
//! load-time fusion plan: confirm that real cart bytes contain the
//! periodic regions we expect (e.g., doom8088's 21-byte × 16-rep
//! column drawer body).
//!
//! Usage:
//!   probe-byte-periods [--input PATH] [--min-period N] [--max-period N]
//!                      [--min-reps N] [--top N] [--needle HEX]
//!
//! Defaults to doom8088 cabinet.

use std::path::PathBuf;

use calcite_core::Evaluator;
use calcite_core::pattern::byte_period::{detect_byte_periods, ByteRegion};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "probe-byte-periods")]
struct Args {
    #[arg(short, long)]
    input: Option<PathBuf>,

    #[arg(long, default_value = "4")]
    min_period: usize,

    #[arg(long, default_value = "256")]
    max_period: usize,

    #[arg(long, default_value = "4")]
    min_reps: usize,

    /// Print top N regions (by total span). 0 = print all.
    #[arg(long, default_value = "20")]
    top: usize,

    /// Optional hex-string needle to search for. If provided, the
    /// probe also reports every offset the needle is found at, and
    /// notes which detected region (if any) starts there. Useful for
    /// confirming the doom column-drawer body offset against
    /// independent ground truth.
    #[arg(long)]
    needle: Option<String>,
}

fn parse_hex(s: &str) -> Vec<u8> {
    let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(cleaned.len() % 2 == 0, "needle must have even hex length");
    (0..cleaned.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&cleaned[i..i + 2], 16).expect("hex byte"))
        .collect()
}

fn find_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return Vec::new();
    }
    (0..=haystack.len() - needle.len())
        .filter(|&i| &haystack[i..i + needle.len()] == needle)
        .collect()
}

fn main() {
    let args = Args::parse();
    let css_path = args.input.unwrap_or_else(|| {
        PathBuf::from("C:/Users/AdmT9N0CX01V65438A/Documents/src/CSS-DOS/doom8088.css")
    });

    eprintln!("probe-byte-periods");
    eprintln!("  input: {}", css_path.display());

    let css = std::fs::read_to_string(&css_path).expect("read css");
    eprintln!("  css size: {} bytes", css.len());

    eprintln!("Parsing...");
    let parsed = calcite_core::parser::parse_css(&css).expect("parse");
    eprintln!(
        "  {} assignments, {} properties, {} functions",
        parsed.assignments.len(),
        parsed.properties.len(),
        parsed.functions.len()
    );

    eprintln!("Compiling (Evaluator::from_parsed)...");
    let evaluator = Evaluator::from_parsed(&parsed);
    let program = evaluator.compiled();

    let dw = match &program.windowed_byte_array {
        Some(dw) => dw,
        None => {
            eprintln!("ERROR: no windowed_byte_array recognised in cabinet — nothing to scan.");
            std::process::exit(1);
        }
    };

    let (base_key, values) = match &dw.backing {
        calcite_core::compile::CompiledWindowBacking::Literal { base_key, values } => {
            (*base_key, values)
        }
        calcite_core::compile::CompiledWindowBacking::PackedCells { .. } => {
            eprintln!("ERROR: windowed_byte_array is packed-cell-backed — no literal byte array to scan.");
            std::process::exit(1);
        }
    };
    let bytes: Vec<u8> = values
        .iter()
        .map(|&v| (v & 0xFF) as u8)
        .collect();
    eprintln!(
        "  windowed_byte_array: window=[{:#x}..{:#x}), stride={}, base_key={:#x}, byte_array.len={}",
        dw.window_base,
        dw.window_end,
        dw.stride,
        base_key,
        bytes.len()
    );

    let t0 = std::time::Instant::now();
    let regions: Vec<ByteRegion> =
        detect_byte_periods(&bytes, args.min_period, args.max_period, args.min_reps);
    let elapsed = t0.elapsed();
    eprintln!(
        "Scan complete in {:.2}ms: {} regions",
        elapsed.as_secs_f64() * 1000.0,
        regions.len()
    );

    let mut sorted = regions.clone();
    sorted.sort_by(|a, b| b.span().cmp(&a.span()));
    let take = if args.top == 0 { sorted.len() } else { args.top.min(sorted.len()) };
    println!("== top {} regions by span ==", take);
    println!("    {:>10}  {:>6}  {:>6}  {:>8}", "start", "period", "reps", "span");
    for r in sorted.iter().take(take) {
        println!(
            "    {:>10}  {:>6}  {:>6}  {:>8}",
            r.start,
            r.period,
            r.reps,
            r.span()
        );
    }

    if let Some(hex) = args.needle.as_deref() {
        let needle = parse_hex(hex);
        let hits = find_all(&bytes, &needle);
        println!();
        println!(
            "== needle ({} bytes) hits: {} ==",
            needle.len(),
            hits.len()
        );
        for &offset in hits.iter().take(32) {
            let region = regions
                .iter()
                .find(|r| r.start == offset)
                .map(|r| format!(" -> region period={} reps={}", r.period, r.reps))
                .unwrap_or_default();
            println!("    offset {} (0x{:x}){}", offset, offset, region);
        }
        if hits.len() > 32 {
            println!("    ... ({} more)", hits.len() - 32);
        }
    }

    // Always print summary stats.
    println!();
    let total_span: usize = regions.iter().map(|r| r.span()).sum();
    let max_reps = regions.iter().map(|r| r.reps).max().unwrap_or(0);
    let max_span = regions.iter().map(|r| r.span()).max().unwrap_or(0);
    println!(
        "summary: {} regions, total span {} bytes ({:.2}% of {} byte image), max reps {}, max span {}",
        regions.len(),
        total_span,
        100.0 * (total_span as f64) / (bytes.len() as f64),
        bytes.len(),
        max_reps,
        max_span
    );
}
