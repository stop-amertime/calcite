//! Integration tests exercising the full pipeline: parse → analyse → evaluate.
//!
//! CSS snippets are modelled after x86CSS patterns.

use calcite_core::eval::Evaluator;
use calcite_core::parser::parse_css;
use calcite_core::State;

/// Standard @property declarations for test state variables.
/// Tests that assign to --AX, --BX, etc. need these so that the state var
/// map is populated (no hardcoded register names in calcite).
const TEST_PROPERTIES: &str = r#"
@property --AX { syntax: "<integer>"; inherits: true; initial-value: 0; }
@property --CX { syntax: "<integer>"; inherits: true; initial-value: 0; }
@property --DX { syntax: "<integer>"; inherits: true; initial-value: 0; }
@property --BX { syntax: "<integer>"; inherits: true; initial-value: 0; }
@property --SP { syntax: "<integer>"; inherits: true; initial-value: 0; }
@property --BP { syntax: "<integer>"; inherits: true; initial-value: 0; }
@property --SI { syntax: "<integer>"; inherits: true; initial-value: 0; }
@property --DI { syntax: "<integer>"; inherits: true; initial-value: 0; }
@property --IP { syntax: "<integer>"; inherits: true; initial-value: 0; }
@property --ES { syntax: "<integer>"; inherits: true; initial-value: 0; }
@property --CS { syntax: "<integer>"; inherits: true; initial-value: 0; }
@property --SS { syntax: "<integer>"; inherits: true; initial-value: 0; }
@property --DS { syntax: "<integer>"; inherits: true; initial-value: 0; }
@property --flags { syntax: "<integer>"; inherits: true; initial-value: 0; }
"#;

/// Parse CSS and build an evaluator. Prepends standard @property declarations
/// if the CSS doesn't already contain any (so tests don't need boilerplate).
fn setup(css: &str) -> (Evaluator, State) {
    let full_css = if css.contains("@property") {
        css.to_string()
    } else {
        format!("{}\n{}", TEST_PROPERTIES, css)
    };
    let parsed = parse_css(&full_css).expect("CSS should parse");
    // load_properties MUST be called before from_parsed so that state var
    // addresses are in the address map before compilation resolves them.
    let mut state = State::default();
    state.load_properties(&parsed.properties);
    let evaluator = Evaluator::from_parsed(&parsed);
    // Wire input-edge groups onto State so set_pseudo_class_active writes
    // gated slots directly. No-op for CSS without `&:has(…:pseudo)` rules.
    evaluator.wire_state_for_input_edges(&mut state);
    (evaluator, state)
}

#[test]
fn simple_assignment() {
    let css = r#"
        .cpu {
            --AX: 42;
            --CX: 7;
        }
    "#;

    let (mut evaluator, mut state) = setup(css);
    evaluator.tick(&mut state);

    assert_eq!(state.get_var("AX").unwrap(), 42);
    assert_eq!(state.get_var("CX").unwrap(), 7);
}

#[test]
fn calc_expression() {
    let css = r#"
        .cpu {
            --AX: calc(10 + 32);
            --CX: calc(var(--AX) * 2);
        }
    "#;

    let (mut evaluator, mut state) = setup(css);
    evaluator.tick(&mut state);

    assert_eq!(state.get_var("AX").unwrap(), 42);
    // --CX depends on --AX being computed first (declaration order)
    assert_eq!(state.get_var("CX").unwrap(), 84);
}

#[test]
fn var_references_between_properties() {
    let css = r#"
        .cpu {
            --AX: 100;
            --BX: calc(var(--AX) + 50);
            --CX: calc(var(--BX) - var(--AX));
        }
    "#;

    let (mut evaluator, mut state) = setup(css);
    evaluator.tick(&mut state);

    assert_eq!(state.get_var("AX").unwrap(), 100);
    assert_eq!(state.get_var("BX").unwrap(), 150);
    assert_eq!(state.get_var("CX").unwrap(), 50);
}

#[test]
fn if_style_condition() {
    let css = r#"
        .cpu {
            --AX: 2;
            --BX: if(style(--AX: 1): 10; style(--AX: 2): 20; style(--AX: 3): 30; else: 0);
        }
    "#;

    let (mut evaluator, mut state) = setup(css);
    evaluator.tick(&mut state);

    assert_eq!(state.get_var("BX").unwrap(), 20);
}

#[test]
fn mod_and_round() {
    // 4660 = 0x1234 (CSS doesn't have hex number literals)
    let css = r#"
        .cpu {
            --AX: 4660;
            --BX: mod(var(--AX), 256);
            --CX: round(down, calc(var(--AX) / 256), 1);
        }
    "#;

    let (mut evaluator, mut state) = setup(css);
    evaluator.tick(&mut state);

    assert_eq!(state.get_var("AX").unwrap(), 4660);
    // AL = AX mod 256 = 0x34 = 52
    assert_eq!(state.get_var("BX").unwrap(), 52);
    // AH = floor(AX / 256) = 0x12 = 18
    assert_eq!(state.get_var("CX").unwrap(), 18);
}

#[test]
fn memory_writes() {
    let css = r#"
        .cpu {
            --m0: 255;
            --m1: 66;
            --m256: 99;
        }
    "#;

    let (mut evaluator, mut state) = setup(css);
    evaluator.tick(&mut state);

    assert_eq!(state.memory[0], 255);
    assert_eq!(state.memory[1], 66);
    assert_eq!(state.memory[256], 99);
}

#[test]
fn function_call() {
    // A simplified readMem-like function
    let css = r#"
        @function --double(--x <integer>) returns <integer> {
            result: calc(var(--x) * 2);
        }

        .cpu {
            --AX: --double(21);
        }
    "#;

    let (mut evaluator, mut state) = setup(css);
    evaluator.tick(&mut state);

    assert_eq!(state.get_var("AX").unwrap(), 42);
}

#[test]
fn multiple_ticks_accumulate() {
    // Each tick adds 1 to AX (by reading its previous value)
    let css = r#"
        .cpu {
            --AX: calc(var(--AX) + 1);
        }
    "#;

    let (mut evaluator, mut state) = setup(css);

    for _ in 0..10 {
        evaluator.tick(&mut state);
    }

    assert_eq!(state.get_var("AX").unwrap(), 10);
}

#[test]
fn dispatch_table_in_function() {
    // A function with enough branches to trigger dispatch table recognition
    let css = r#"
        @function --lookup(--key <integer>) returns <integer> {
            result: if(
                style(--key: 0): 100;
                style(--key: 1): 200;
                style(--key: 2): 300;
                style(--key: 3): 400;
                style(--key: 4): 500;
                else: 0
            );
        }

        .cpu {
            --AX: --lookup(3);
        }
    "#;

    let (mut evaluator, mut state) = setup(css);
    evaluator.tick(&mut state);

    assert_eq!(state.get_var("AX").unwrap(), 400);
}

#[test]
fn property_declarations() {
    let css = r#"
        @property --counter {
            syntax: "<integer>";
            inherits: false;
            initial-value: 0;
        }

        .cpu {
            --AX: 1;
        }
    "#;

    let parsed = parse_css(css).expect("should parse");
    assert_eq!(parsed.properties.len(), 1);
    assert_eq!(parsed.properties[0].name, "--counter");
    assert_eq!(parsed.functions.len(), 0);
    assert_eq!(parsed.assignments.len(), 1);
}

#[test]
fn complex_nested_expression() {
    // Mimics x86CSS's byte extraction pattern: split and reconstruct
    // 4660 = 0x1234, lo=52 (0x34), hi=18 (0x12)
    let css = r#"
        .cpu {
            --AX: 4660;
            --BX: calc(mod(var(--AX), 256) + round(down, calc(var(--AX) / 256), 1) * 256);
        }
    "#;

    let (mut evaluator, mut state) = setup(css);
    evaluator.tick(&mut state);

    // BX should reconstruct AX from its bytes: AL + AH*256 = AX
    assert_eq!(state.get_var("AX").unwrap(), 4660);
    assert_eq!(state.get_var("BX").unwrap(), 4660);
}

#[test]
fn min_max_clamp() {
    let css = r#"
        .cpu {
            --AX: min(100, 200, 50);
            --BX: max(10, 30, 20);
            --CX: clamp(0, 500, 255);
        }
    "#;

    let (mut evaluator, mut state) = setup(css);
    evaluator.tick(&mut state);

    assert_eq!(state.get_var("AX").unwrap(), 50);
    assert_eq!(state.get_var("BX").unwrap(), 30);
    assert_eq!(state.get_var("CX").unwrap(), 255);
}

#[test]
fn empty_css() {
    let css = "";
    let parsed = parse_css(css).expect("empty CSS should parse");
    assert!(parsed.properties.is_empty());
    assert!(parsed.functions.is_empty());
    assert!(parsed.assignments.is_empty());
}

#[test]
fn and_or_conditions() {
    let css = r#"
        .cpu {
            --AX: 5;
            --BX: 10;
            --CX: if(style(--AX: 5) and style(--BX: 10): 1; else: 0);
            --DX: if(style(--AX: 99) or style(--BX: 10): 1; else: 0);
        }
    "#;

    let (mut evaluator, mut state) = setup(css);
    evaluator.tick(&mut state);

    // Both conditions true → 1
    assert_eq!(state.get_var("CX").unwrap(), 1);
    // First false, second true → 1 (or)
    assert_eq!(state.get_var("DX").unwrap(), 1);
}

#[test]
fn if_without_else() {
    let css = r#"
        .cpu {
            --AX: 42;
            --BX: if(style(--AX: 42): 100);
        }
    "#;

    let (mut evaluator, mut state) = setup(css);
    evaluator.tick(&mut state);

    assert_eq!(state.get_var("BX").unwrap(), 100);
}

#[test]
fn string_literal_in_if() {
    // Strings have numeric value 0 but should parse without error
    let css = r#"
        .cpu {
            --AX: if(style(--BX: 0): 1; else: 0);
        }
    "#;

    let (mut evaluator, mut state) = setup(css);
    evaluator.tick(&mut state);

    assert_eq!(state.get_var("AX").unwrap(), 1);
}

/// Parse the real x86CSS generated CSS and validate key metrics.
///
/// This test requires the fixture file from the x86CSS repo.
/// It validates that our parser can handle the full complexity of real-world
/// computational CSS.
#[test]
#[ignore = "requires x86css-main.css fixture"]
fn parse_real_x86css() {
    let css = std::fs::read_to_string("../../tests/fixtures/x86css-main.css")
        .expect("fixture not found at tests/fixtures/x86css-main.css");

    let parsed = parse_css(&css).expect("x86CSS should parse");

    // Validate key metrics — counts updated after multi-write instruction support
    assert_eq!(parsed.properties.len(), 1585, "@property count");
    assert!(
        parsed.functions.len() >= 128,
        "should have >= 128 @functions, got {}",
        parsed.functions.len()
    );
    assert!(
        parsed.assignments.len() > 3000,
        "should have >3000 assignments, got {}",
        parsed.assignments.len()
    );

    // Build evaluator and check pattern recognition
    let evaluator = Evaluator::from_parsed(&parsed);

    // readMem should be recognised as a dispatch table
    assert!(
        evaluator.dispatch_tables.contains_key("--readMem"),
        "readMem should be a dispatch table"
    );
    let readmem = &evaluator.dispatch_tables["--readMem"];
    assert!(
        readmem.entries.len() > 1500,
        "readMem should have >1500 entries, got {}",
        readmem.entries.len()
    );

    // Should have multiple dispatch tables
    assert!(
        evaluator.dispatch_tables.len() >= 8,
        "should have >= 8 dispatch tables, got {}",
        evaluator.dispatch_tables.len()
    );

    // Should detect broadcast write pattern
    assert!(
        !evaluator.broadcast_writes.is_empty(),
        "should detect broadcast write pattern"
    );

    // Verify key functions were parsed with correct parameters
    let readmem_func = evaluator
        .functions
        .get("--readMem")
        .expect("--readMem should exist");
    assert_eq!(readmem_func.parameters.len(), 1);
    assert_eq!(readmem_func.parameters[0].name, "--at");

    let xor_func = evaluator
        .functions
        .get("--xor")
        .expect("--xor should exist");
    assert_eq!(xor_func.parameters.len(), 2);
    assert!(
        xor_func.locals.len() >= 30,
        "--xor should have many locals (bit decomposition)"
    );
}

/// Conformance test: verify calcite's steady-state matches Chrome's baseline.
///
/// Chrome baseline was captured from the live x86CSS page (lyra.horse/x86css/)
/// after BIOS initialization. Rather than tick-for-tick comparison (impossible
/// without CSS animation synchronization), we verify that calcite reaches the
/// same main loop and register value ranges as Chrome.
#[test]
#[ignore = "requires x86css-main.css fixture"]
fn chrome_conformance_steady_state() {
    let css = std::fs::read_to_string("../../tests/fixtures/x86css-main.css")
        .expect("fixture not found at tests/fixtures/x86css-main.css");

    let parsed = parse_css(&css).expect("should parse");
    let mut evaluator = Evaluator::from_parsed(&parsed);
    let mut calcite_state = State::default();
    calcite_state.load_properties(&parsed.properties);

    // Run 1000 ticks — enough to get through BIOS init and into the main loop
    for _ in 0..1000 {
        evaluator.tick(&mut calcite_state);
    }

    // Sample calcite's register values over 200 additional ticks
    let mut calcite_ip_values = std::collections::HashSet::new();
    let mut calcite_ax_values = std::collections::HashSet::new();
    let mut calcite_si_values = std::collections::HashSet::new();
    let mut calcite_sp_values = std::collections::HashSet::new();
    let mut calcite_bp_values = std::collections::HashSet::new();
    let mut calcite_flags_values = std::collections::HashSet::new();

    for _ in 0..200 {
        evaluator.tick(&mut calcite_state);
        let regs = calcite_core::conformance::RegisterSnapshot::from_state(&calcite_state);
        calcite_ip_values.insert(regs.values.get("IP").copied().unwrap_or(0));
        calcite_ax_values.insert(regs.values.get("AX").copied().unwrap_or(0));
        calcite_si_values.insert(regs.values.get("SI").copied().unwrap_or(0));
        calcite_sp_values.insert(regs.values.get("SP").copied().unwrap_or(0));
        calcite_bp_values.insert(regs.values.get("BP").copied().unwrap_or(0));
        calcite_flags_values.insert(regs.values.get("flags").copied().unwrap_or(0));
    }

    // Chrome baseline ranges (captured from lyra.horse/x86css/ via Playwright).
    // See tests/fixtures/chrome-baseline.json for the full captured data.
    // Note: baseline was captured with old CSS (before multi-write support).
    // The new CSS has different instruction implementations, so exact register
    // values may differ. We check structural properties instead.

    /// IP addresses the program visits during steady-state execution.
    /// After the ModR/M reorder fix, the program correctly executes CALL indirect
    /// and reaches external function stubs + the readInput handler.
    const EXPECTED_IPS: &[i32] = &[
        292, 295, 302, 303, 307, 310, 313, // inner loop
        901, 907, 911, 913, 915, // outer loop
        0x2004, 0x2006, // external functions (writeChar8, readInput)
    ];

    // Print comparison for diagnostics
    eprintln!("=== Chrome vs Calcite steady-state comparison ===");
    eprintln!("AX  Calcite: {:?}", calcite_ax_values);
    eprintln!("IP  Calcite: {:?}", {
        let mut v: Vec<_> = calcite_ip_values.iter().copied().collect();
        v.sort();
        v
    });
    eprintln!("SP  Calcite: {:?}", {
        let mut v: Vec<_> = calcite_sp_values.iter().copied().collect();
        v.sort();
        v
    });
    eprintln!("BP  Calcite: {:?}", {
        let mut v: Vec<_> = calcite_bp_values.iter().copied().collect();
        v.sort();
        v
    });
    eprintln!("SI  Calcite: {:?}", {
        let mut v: Vec<_> = calcite_si_values.iter().copied().collect();
        v.sort();
        v
    });
    eprintln!("flags Calcite: {:?}", {
        let mut v: Vec<_> = calcite_flags_values.iter().copied().collect();
        v.sort();
        v
    });

    // IP should hit expected addresses (steady-state loop + external functions)
    let calcite_hits_expected = EXPECTED_IPS.iter().any(|ip| calcite_ip_values.contains(ip));
    assert!(
        calcite_hits_expected,
        "IP should hit expected addresses, got: {:?}",
        calcite_ip_values
    );

    // SP should be in a reasonable stack range (around 1500-1530)
    let sp_in_range = calcite_sp_values
        .iter()
        .all(|&sp| (1500..=1530).contains(&sp));
    assert!(
        sp_in_range,
        "SP should be in stack range 1500-1530, got: {:?}",
        calcite_sp_values
    );

    // Flags should include at least one of the Chrome baseline values (0 or 192)
    assert!(
        calcite_flags_values.contains(&0) || calcite_flags_values.contains(&192),
        "flags should include 0 or 192, got: {:?}",
        calcite_flags_values
    );

    // SI should be in a reasonable range (Chrome shows 1120-1152)
    assert!(
        calcite_si_values
            .iter()
            .all(|&si| (1100..=1300).contains(&si)),
        "SI should be in range 1100-1300, got: {:?}",
        calcite_si_values
    );
}

/// Execution trace diagnostic: dumps tick-by-tick instruction decode signals
/// to understand what the CPU is actually doing.
///
/// Execution trace: dump per-tick CPU state for diagnosis.
/// Run with: cargo test --ignored execution_trace -- --nocapture
#[test]
#[ignore = "requires x86css-demo.css fixture"]
fn execution_trace() {
    let css = std::fs::read_to_string("../../tests/fixtures/x86css-demo.css")
        .expect("fixture not found at tests/fixtures/x86css-demo.css");

    let parsed = parse_css(&css).expect("should parse demo CSS");
    let mut evaluator = Evaluator::from_parsed(&parsed);
    let mut state = State::default();
    state.load_properties(&parsed.properties);

    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .try_init();

    eprintln!(
        "tick | IP   | instId | instLen | destA | destB | valA  | valB  | AX   | SP   | flags"
    );
    eprintln!(
        "-----|------|--------|---------|-------|-------|-------|-------|------|------|------"
    );

    for tick in 0..50 {
        evaluator.tick(&mut state);
        let ip = state.get_var("IP").unwrap_or(0);
        let inst_id = evaluator.get_property("--instId").unwrap_or(-1.0) as i32;
        let inst_len = evaluator.get_property("--instLen").unwrap_or(0.0) as i32;
        let dest_a = evaluator.get_property("--addrDestA").unwrap_or(-100.0) as i32;
        let dest_b = evaluator.get_property("--addrDestB").unwrap_or(-100.0) as i32;
        let val_a = evaluator.get_property("--addrValA").unwrap_or(0.0) as i32;
        let val_b = evaluator.get_property("--addrValB").unwrap_or(0.0) as i32;

        eprintln!(
            "{:4} | {:4} | {:6} | {:7} | {:5} | {:5} | {:5} | {:5} | {:4} | {:4} | {:5}",
            tick,
            ip,
            inst_id,
            inst_len,
            dest_a,
            dest_b,
            val_a,
            val_b,
            state.get_var("AX").unwrap_or(0),
            state.get_var("SP").unwrap_or(0),
            state.get_var("flags").unwrap_or(0),
        );
    }
}

/// Fibonacci benchmark: run the demo's Fibonacci program and verify output.
///
/// The x86CSS demo program starts with BIOS init, displays a menu, and waits for
/// keyboard input. Pressing '1' runs the Fibonacci function which outputs:
///   "Fibonacci sequence:\n0 1 1 2 3 5 8 13 21 34 55 89"
///
/// Benchmark compiled vs interpreted: run 10K ticks on the demo fixture with each path.
/// Run with: cargo test --ignored compiled_vs_interpreted_benchmark -- --nocapture
#[test]
#[ignore = "requires x86css-demo.css fixture"]
fn compiled_vs_interpreted_benchmark() {
    let css = std::fs::read_to_string("../../tests/fixtures/x86css-demo.css")
        .expect("fixture not found at tests/fixtures/x86css-demo.css");

    let parsed = parse_css(&css).expect("should parse demo CSS");
    let ticks = 10_000u32;

    // Interpreted path
    let mut eval_interp = Evaluator::from_parsed(&parsed);
    let mut state_interp = State::default();
    state_interp.load_properties(&parsed.properties);

    let start = std::time::Instant::now();
    for _ in 0..ticks {
        eval_interp.tick_interpreted(&mut state_interp);
    }
    let interp_elapsed = start.elapsed();

    // Compiled path
    let mut eval_compiled = Evaluator::from_parsed(&parsed);
    let mut state_compiled = State::default();
    state_compiled.load_properties(&parsed.properties);

    let start = std::time::Instant::now();
    for _ in 0..ticks {
        eval_compiled.tick(&mut state_compiled);
    }
    let compiled_elapsed = start.elapsed();

    let interp_tps = ticks as f64 / interp_elapsed.as_secs_f64();
    let compiled_tps = ticks as f64 / compiled_elapsed.as_secs_f64();
    let speedup = interp_elapsed.as_secs_f64() / compiled_elapsed.as_secs_f64();

    println!("\n=== Compiled vs Interpreted Benchmark ({ticks} ticks) ===");
    println!(
        "  Interpreted: {:.1}ms ({:.0} ticks/sec)",
        interp_elapsed.as_secs_f64() * 1000.0,
        interp_tps
    );
    println!(
        "  Compiled:    {:.1}ms ({:.0} ticks/sec)",
        compiled_elapsed.as_secs_f64() * 1000.0,
        compiled_tps
    );
    println!("  Speedup:     {:.1}x", speedup);

    // Verify both paths reach same state
    for i in 0..state_compiled.state_var_count() {
        assert_eq!(
            state_compiled.state_vars[i], state_interp.state_vars[i],
            "state var {} diverged after {ticks} ticks: compiled={}, interpreted={}",
            state_compiled.state_var_names.get(i).map(|s| s.as_str()).unwrap_or("?"),
            state_compiled.state_vars[i], state_interp.state_vars[i]
        );
    }
}

/// Run with: cargo test --ignored fibonacci_benchmark -- --nocapture
#[test]
#[ignore = "requires x86css-demo.css fixture"]
fn fibonacci_benchmark() {
    let css = std::fs::read_to_string("../../tests/fixtures/x86css-demo.css")
        .expect("fixture not found at tests/fixtures/x86css-demo.css");

    let parsed = parse_css(&css).expect("should parse demo CSS");
    let mut evaluator = Evaluator::from_parsed(&parsed);
    let mut state = State::default();
    state.load_properties(&parsed.properties);

    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .try_init();

    let max_ticks: u32 = 50_000;
    let mut keyboard_set = false;

    let start = std::time::Instant::now();

    for tick in 0..max_ticks {
        let ip = state.get_var("IP").unwrap_or(0);

        // When the program reaches readInput (IP=0x2006), provide keyboard input.
        // NOTE: x86css-demo.css uses the legacy keyboard convention (0x500 polling),
        // not the gossamer-dos BDA ring buffer. New programs should use the BDA.
        if !keyboard_set && ip == 0x2006 {
            state.write_mem(0x500, 49); // ASCII '1' = select Fibonacci
            state.write_mem(0x501, 0x02); // scancode for '1'
            keyboard_set = true;
        }

        evaluator.tick(&mut state);

        // Check if text output contains the full Fibonacci sequence
        if keyboard_set
            && state
                .string_properties
                .get("textBuffer")
                .cloned()
                .unwrap_or_default()
                .contains("89")
        {
            let elapsed = start.elapsed();
            eprintln!(
                "Fibonacci completed: {} ticks in {elapsed:?} ({:.0} ticks/sec)",
                tick + 1,
                (tick + 1) as f64 / elapsed.as_secs_f64()
            );
            eprintln!(
                "Output: {:?}",
                state
                    .string_properties
                    .get("textBuffer")
                    .cloned()
                    .unwrap_or_default()
            );

            assert!(
                state
                    .string_properties
                    .get("textBuffer")
                    .cloned()
                    .unwrap_or_default()
                    .contains("Fibonacci sequence:\n0 1 1 2 3 5 8 13 21 34 55 89"),
                "Expected Fibonacci sequence in output, got: {:?}",
                state
                    .string_properties
                    .get("textBuffer")
                    .cloned()
                    .unwrap_or_default()
            );
            return;
        }
    }

    panic!(
        "Fibonacci did not complete in {max_ticks} ticks. Output so far: {:?}",
        state
            .string_properties
            .get("textBuffer")
            .cloned()
            .unwrap_or_default()
    );
}

// --- Parser negative / error path tests ---

#[test]
fn completely_invalid_css() {
    // Random garbage should parse without crashing, producing empty output
    let parsed = parse_css("}{}{}{not css at all!!!").expect("should not error");
    assert!(parsed.assignments.is_empty());
    assert!(parsed.functions.is_empty());
    assert!(parsed.properties.is_empty());
}

#[test]
fn unclosed_rule_block() {
    let css = ".cpu { --AX: 42;";
    // cssparser is lenient with unclosed blocks
    let parsed = parse_css(css).expect("should not error");
    assert_eq!(parsed.assignments.len(), 1);
    assert_eq!(parsed.assignments[0].property, "--AX");
}

#[test]
fn missing_semicolons() {
    // Missing semicolons between declarations — parser should recover
    let css = r#"
        .cpu {
            --AX: 10
            --BX: 20;
        }
    "#;
    let parsed = parse_css(css).expect("should not error");
    // At minimum BX (after the semicolon) should parse. AX may or may not
    // depending on how the parser recovers.
    assert!(!parsed.assignments.is_empty());
}

#[test]
fn non_custom_properties_ignored() {
    let css = r#"
        .cpu {
            color: red;
            font-size: 12px;
            --AX: 42;
        }
    "#;
    let parsed = parse_css(css).expect("should not error");
    // Only --AX should be captured
    assert_eq!(parsed.assignments.len(), 1);
    assert_eq!(parsed.assignments[0].property, "--AX");
}

#[test]
fn empty_function_body() {
    let css = r#"
        @function --empty(--x <integer>) returns <integer> {
        }
    "#;
    let parsed = parse_css(css).expect("should not error");
    assert_eq!(parsed.functions.len(), 1);
    assert_eq!(parsed.functions[0].name, "--empty");
}

#[test]
fn function_without_result() {
    let css = r#"
        @function --noResult(--x <integer>) returns <integer> {
            --local: calc(var(--x) + 1);
        }
    "#;
    let parsed = parse_css(css).expect("should not error");
    // Function should still parse (result defaults to 0)
    assert_eq!(parsed.functions.len(), 1);
    assert_eq!(parsed.functions[0].locals.len(), 1);
}

#[test]
fn nested_calc_missing_operand() {
    // calc(1 +) — missing right operand
    let css = r#"
        .cpu {
            --AX: calc(1 +);
            --BX: 42;
        }
    "#;
    let parsed = parse_css(css).expect("should not error");
    // AX may fail to parse; BX should succeed
    let bx = parsed.assignments.iter().find(|a| a.property == "--BX");
    assert!(bx.is_some(), "BX should still parse after bad AX");
}

#[test]
fn unknown_at_rules_skipped() {
    let css = r#"
        @keyframes spin { from { } to { } }
        @media screen { }
        .cpu {
            --AX: 1;
        }
    "#;
    let parsed = parse_css(css).expect("should not error");
    assert_eq!(parsed.assignments.len(), 1);
}

#[test]
fn deeply_nested_expressions() {
    // Deeply nested calc — should not stack overflow
    let mut expr = "1".to_string();
    for _ in 0..50 {
        expr = format!("calc({expr} + 1)");
    }
    let css = format!(".cpu {{ --AX: {expr}; }}");
    let parsed = parse_css(&css).expect("should not error");
    assert_eq!(parsed.assignments.len(), 1);

    let (mut evaluator, mut state) = setup(&format!(".cpu {{ --AX: {expr}; }}"));
    evaluator.tick(&mut state);
    assert_eq!(state.get_var("AX").unwrap(), 51);
}

/// Run the same CSS through both compiled and interpreted paths and verify
/// they produce identical state.
#[test]
fn compiled_vs_interpreted() {
    let css = r#"
        @property --AX { syntax: "<integer>"; inherits: false; initial-value: 0; }
        @property --CX { syntax: "<integer>"; inherits: false; initial-value: 0; }
        @property --DX { syntax: "<integer>"; inherits: false; initial-value: 0; }
        @property --BX { syntax: "<integer>"; inherits: false; initial-value: 100; }
        @property --IP { syntax: "<integer>"; inherits: false; initial-value: 5; }

        @function --double {
            --x: <integer>;
            result: calc(var(--x) * 2);
        }

        .cpu {
            --AX: calc(var(--BX) + 1);
            --CX: --double(var(--AX));
            --DX: if(
                style(--BX: 100): calc(var(--CX) + 10);
                else: 0;
            );
            --IP: calc(var(--IP) + 1);
        }
    "#;

    let parsed = parse_css(css).expect("CSS should parse");

    // Compiled path
    let mut state_compiled = State::default();
    state_compiled.load_properties(&parsed.properties);
    let mut eval_compiled = Evaluator::from_parsed(&parsed);
    for _ in 0..5 {
        eval_compiled.tick(&mut state_compiled);
    }

    // Interpreted path
    let mut state_interp = State::default();
    state_interp.load_properties(&parsed.properties);
    let mut eval_interp = Evaluator::from_parsed(&parsed);
    for _ in 0..5 {
        eval_interp.tick_interpreted(&mut state_interp);
    }

    // Compare all state variables
    for i in 0..state_compiled.state_var_count() {
        assert_eq!(
            state_compiled.state_vars[i], state_interp.state_vars[i],
            "state var {} diverged: compiled={}, interpreted={}",
            state_compiled.state_var_names.get(i).map(|s| s.as_str()).unwrap_or("?"),
            state_compiled.state_vars[i], state_interp.state_vars[i]
        );
    }
}

/// Test that function calls inside nested style() conditions work in the compiled path.
/// This is a regression test for a bug where --read2(calc(...)) returned 0 when called
/// inside if(style(--opcode:X): if(style(--uOp:Y): func(...))).
#[test]
fn nested_style_function_call() {
    // Minimal reproduction: --CS should be set to --double(var(--q1)) when both
    // opcode=205 and uOp=5. The compiled path was returning 0 instead.
    let css = r#"
        @property --CS { syntax: "<integer>"; inherits: false; initial-value: 0; }
        @property --opcode { syntax: "<integer>"; inherits: false; initial-value: 205; }
        @property --uOp { syntax: "<integer>"; inherits: false; initial-value: 5; }
        @property --q1 { syntax: "<integer>"; inherits: false; initial-value: 42; }

        @function --double(--x <integer>) returns <integer> {
            result: calc(var(--x) * 2);
        }

        .cpu {
            --CS: if(
                style(--opcode: 205): if(
                    style(--uOp: 5): --double(var(--q1));
                    else: 0;
                );
                else: 0;
            );
        }
    "#;

    // Enable debug logging to see parser errors
    let parsed = parse_css(css).expect("CSS should parse");

    // Compiled path
    let mut state_compiled = State::default();
    state_compiled.load_properties(&parsed.properties);
    let mut eval_compiled = Evaluator::from_parsed(&parsed);
    eval_compiled.tick(&mut state_compiled);

    // Interpreted path
    let mut state_interp = State::default();
    state_interp.load_properties(&parsed.properties);
    let mut eval_interp = Evaluator::from_parsed(&parsed);
    eval_interp.tick_interpreted(&mut state_interp);

    let cs_slot = state_compiled.state_var_names.iter().position(|n| n == "--CS" || n == "CS").unwrap_or_else(|| {
        panic!("--CS not found in state_var_names: {:?}", state_compiled.state_var_names)
    });
    let compiled_cs = state_compiled.state_vars[cs_slot];
    let interp_cs = state_interp.state_vars[cs_slot];

    assert_eq!(
        compiled_cs, interp_cs,
        "nested style function call: compiled CS={} vs interpreted CS={} (expected 84)",
        compiled_cs, interp_cs
    );
    assert_eq!(compiled_cs, 84, "expected --double(42) = 84, got {}", compiled_cs);
}

/// End-to-end input-edge test: parse a tiny cabinet with two
/// `&:has(#kb-X:active) { --keyboard: V; }` rules, drive press/release
/// via `set_pseudo_class_active`, and verify `--keyboard` reflects the
/// gated values via the recogniser. No host-side `set_var` calls.
///
/// Mirrors the calcite-wasm e2e probe (pseudo-active-api-probe.html) but
/// runs in plain Rust so it shows up in `cargo test`.
#[test]
fn input_edges_drive_keyboard_via_set_pseudo_class_active() {
    let css = r#"
        @property --keyboard { syntax: "<integer>"; initial-value: 0; inherits: true; }
        @property --opcode   { syntax: "<integer>"; initial-value: 0; inherits: true; }
        .cpu {
            &:has(#kb-1:active) { --keyboard: 561; }
            &:has(#kb-a:active) { --keyboard: 7777; }
        }
    "#;
    let (mut eval, mut state) = setup(css);

    // Sanity: parser found two input edges.
    let edges = eval.input_edges_for_host();
    assert_eq!(edges.len(), 2);
    assert!(edges.iter().any(|(p, ps, sel, v)| p == "--keyboard" && ps == "active" && sel == "kb-1" && *v == 561));
    assert!(edges.iter().any(|(p, ps, sel, v)| p == "--keyboard" && ps == "active" && sel == "kb-a" && *v == 7777));

    // No press → 0.
    eval.tick(&mut state);
    assert_eq!(state.get_var("keyboard").unwrap(), 0);

    // Press kb-1 → 561.
    state.set_pseudo_class_active("active", "kb-1", true);
    eval.tick(&mut state);
    assert_eq!(state.get_var("keyboard").unwrap(), 561);

    // Add kb-a → sum to 8338.
    state.set_pseudo_class_active("active", "kb-a", true);
    eval.tick(&mut state);
    assert_eq!(state.get_var("keyboard").unwrap(), 8338);

    // Release kb-1 → just kb-a → 7777.
    state.set_pseudo_class_active("active", "kb-1", false);
    eval.tick(&mut state);
    assert_eq!(state.get_var("keyboard").unwrap(), 7777);

    // Release kb-a → 0.
    state.set_pseudo_class_active("active", "kb-a", false);
    eval.tick(&mut state);
    assert_eq!(state.get_var("keyboard").unwrap(), 0);
}
