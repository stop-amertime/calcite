use criterion::{criterion_group, criterion_main, Criterion};

use calcify_core::eval::Evaluator;
use calcify_core::parser::parse_css;
use calcify_core::state::State;

fn load_css() -> String {
    std::fs::read_to_string("../../tests/fixtures/x86css-main.css")
        .expect("x86css-main.css fixture required for benchmarks")
}

fn bench_parse(c: &mut Criterion) {
    let css = load_css();
    c.bench_function("parse_x86css", |b| {
        b.iter(|| parse_css(&css).unwrap());
    });
}

fn bench_setup(c: &mut Criterion) {
    let css = load_css();
    let parsed = parse_css(&css).unwrap();
    c.bench_function("evaluator_from_parsed", |b| {
        b.iter(|| Evaluator::from_parsed(&parsed));
    });
}

fn bench_tick(c: &mut Criterion) {
    let css = load_css();
    let parsed = parse_css(&css).unwrap();
    let mut evaluator = Evaluator::from_parsed(&parsed);
    let mut state = State::default();
    state.load_properties(&parsed.properties);
    // Warm up past BIOS init
    for _ in 0..50 {
        evaluator.tick(&mut state);
    }

    c.bench_function("single_tick", |b| {
        b.iter(|| evaluator.tick(&mut state));
    });
}

fn bench_batch(c: &mut Criterion) {
    let css = load_css();
    let parsed = parse_css(&css).unwrap();
    let mut evaluator = Evaluator::from_parsed(&parsed);

    let mut group = c.benchmark_group("batch_ticks");
    for &count in &[10, 100] {
        group.bench_function(format!("{count}_ticks"), |b| {
            b.iter_with_setup(
                || {
                    let mut state = State::default();
                    state.load_properties(&parsed.properties);
                    state
                },
                |mut state| evaluator.run_batch(&mut state, count),
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_parse, bench_setup, bench_tick, bench_batch);
criterion_main!(benches);
