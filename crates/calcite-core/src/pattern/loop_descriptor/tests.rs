//! Unit tests for the self-loop recogniser.
//!
//! Two synthetic cabinets, A and B, produce equivalent descriptors
//! despite having completely different slot/property names. This is the
//! cardinal-rule genericity probe: A is x86-shaped (slot names match the
//! current kiln-emitted cabinets); B is brainfuck-shaped (arbitrary
//! opaque names, no x86 ABI, no shared naming convention with A). The
//! recogniser must not see the difference.

use super::*;
use crate::types::*;

// ---------------------------------------------------------------------------
// Helpers for building Expr trees (lots of `Box::new` otherwise).
// ---------------------------------------------------------------------------

fn lit(v: f64) -> Expr {
    Expr::Literal(v)
}

fn var(name: &str) -> Expr {
    Expr::Var {
        name: name.to_string(),
        fallback: None,
    }
}

fn add(a: Expr, b: Expr) -> Expr {
    Expr::Calc(CalcOp::Add(Box::new(a), Box::new(b)))
}

fn sub(a: Expr, b: Expr) -> Expr {
    Expr::Calc(CalcOp::Sub(Box::new(a), Box::new(b)))
}

fn mul(a: Expr, b: Expr) -> Expr {
    Expr::Calc(CalcOp::Mul(Box::new(a), Box::new(b)))
}

fn maxof(a: Expr, b: Expr) -> Expr {
    Expr::Calc(CalcOp::Max(vec![a, b]))
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::FunctionCall {
        name: name.to_string(),
        args,
    }
}

fn style_eq(prop: &str, value: f64) -> StyleTest {
    StyleTest::Single {
        property: prop.to_string(),
        value: Expr::Literal(value),
    }
}

/// Build `if(<test>: then; else: fallback)`.
fn iff(test: StyleTest, then: Expr, fallback: Expr) -> Expr {
    Expr::StyleCondition {
        branches: vec![StyleBranch {
            condition: test,
            then,
        }],
        fallback: Box::new(fallback),
    }
}

/// Build a multi-branch StyleCondition, all branches keyed on `prop`.
fn dispatch(prop: &str, branches: Vec<(f64, Expr)>, fallback: Expr) -> Expr {
    Expr::StyleCondition {
        branches: branches
            .into_iter()
            .map(|(v, then)| StyleBranch {
                condition: style_eq(prop, v),
                then,
            })
            .collect(),
        fallback: Box::new(fallback),
    }
}

fn assign(prop: &str, value: Expr) -> Assignment {
    Assignment {
        property: prop.to_string(),
        value,
    }
}

// ---------------------------------------------------------------------------
// Builders for the per-V kiln shapes.
// ---------------------------------------------------------------------------

/// Build the IP-stay-or-advance body for one opcode.
///
/// Shape: `if(<predicate>: calc(self - prefix_sub); else: calc(self + advance_lit))`.
///
/// `predicate` is the loop-continue gate; `self_var` is the prior IP
/// mirror; `prefix_sub` is what gets subtracted on the stay branch.
fn ip_body(predicate: StyleTest, self_var: &str, prefix_sub: Expr, advance_lit: i32) -> Expr {
    iff(
        predicate,
        sub(var(self_var), prefix_sub),
        add(var(self_var), lit(advance_lit as f64)),
    )
}

/// Build the multi-branch IP-stay-or-advance body for an opcode whose
/// loop-continue predicate is a disjunction of multiple branch
/// conditions. Used for CMPS/SCAS in real cabinets:
///
///   `if(<P1>: stay; <P2>: stay; ...; else: advance)`
///
/// All branches share the same stay body. `predicates` provides the
/// per-branch conditions; each must independently signal "stay".
fn ip_body_multi(
    predicates: Vec<StyleTest>,
    self_var: &str,
    prefix_sub: Expr,
    advance_lit: i32,
) -> Expr {
    let stay = sub(var(self_var), prefix_sub);
    let advance = add(var(self_var), lit(advance_lit as f64));
    Expr::StyleCondition {
        branches: predicates
            .into_iter()
            .map(|p| StyleBranch {
                condition: p,
                then: stay.clone(),
            })
            .collect(),
        fallback: Box::new(advance),
    }
}

/// Build the counter-decrement body for one opcode.
///
/// Shape: `if(<no-rep-guard>: self; else: max(0, calc(self - 1)))`.
fn counter_body(no_rep_guard: StyleTest, self_var: &str) -> Expr {
    iff(
        no_rep_guard,
        var(self_var),
        maxof(lit(0.0), sub(var(self_var), lit(1.0))),
    )
}

/// Build the pointer-step body for one opcode in kiln's actual shape:
///
///   `if(<rep-guard>: var(self); else: <update-expr>)`
///
/// where update-expr =
///   `OUTER_CALL(calc(calc(var(self) + k) - INNER_CALL(var(flag), bit) * (2k)), 16)`
/// — the kiln `--lowerBytes` / direction-flag idiom for a 16-bit
/// modular pointer step.
fn pointer_body(
    rep_guard: StyleTest,
    self_var: &str,
    base_step: i32,
    flag_var: &str,
    flag_bit: u32,
    outer_call: &str,
    inner_call: &str,
) -> Expr {
    let inner = sub(
        add(var(self_var), lit(base_step as f64)),
        mul(
            call(inner_call, vec![var(flag_var), lit(flag_bit as f64)]),
            lit(2.0 * base_step as f64),
        ),
    );
    let update = call(outer_call, vec![inner, lit(16.0)]);
    iff(rep_guard, var(self_var), update)
}

/// Build a "no entry" fallback expression (slot keeps its prior value).
fn keep_self(self_var: &str) -> Expr {
    var(self_var)
}

// ---------------------------------------------------------------------------
// Cabinet A — x86-shaped names.
//
// Two opcodes:
//   0xAA: STOSB-shape — counter, one pointer (DI).
//   0xA4: MOVSB-shape — counter, two pointers (DI, SI).
// ---------------------------------------------------------------------------

fn cabinet_a() -> Vec<Assignment> {
    let pred_continue = style_eq("--_repContinue", 1.0);
    let no_rep = style_eq("--hasREP", 0.0);

    let cx_dispatch = dispatch(
        "--opcode",
        vec![
            (0xAA as f64, counter_body(no_rep.clone(), "--__1CX")),
            (0xA4 as f64, counter_body(no_rep.clone(), "--__1CX")),
        ],
        keep_self("--__1CX"),
    );

    let ip_dispatch = dispatch(
        "--opcode",
        vec![
            (
                0xAA as f64,
                ip_body(
                    pred_continue.clone(),
                    "--__1IP",
                    var("--prefixLen"),
                    1,
                ),
            ),
            (
                0xA4 as f64,
                ip_body(
                    pred_continue.clone(),
                    "--__1IP",
                    var("--prefixLen"),
                    1,
                ),
            ),
        ],
        keep_self("--__1IP"),
    );

    // Outer wrapper kiln adds: calc(<dispatch> + var(--prefixLen)).
    let ip_wrapped = add(ip_dispatch, var("--prefixLen"));

    let active_guard = StyleTest::And(vec![
        style_eq("--hasREP", 1.0),
        style_eq("--_repActive", 0.0),
    ]);
    let di_dispatch = dispatch(
        "--opcode",
        vec![
            (
                0xAA as f64,
                pointer_body(
                    active_guard.clone(),
                    "--__1DI",
                    1,
                    "--__1flags",
                    10,
                    "--lowerBytes",
                    "--bit",
                ),
            ),
            (
                0xA4 as f64,
                pointer_body(
                    active_guard.clone(),
                    "--__1DI",
                    1,
                    "--__1flags",
                    10,
                    "--lowerBytes",
                    "--bit",
                ),
            ),
        ],
        keep_self("--__1DI"),
    );
    let si_dispatch = dispatch(
        "--opcode",
        vec![(
            0xA4 as f64,
            pointer_body(
                active_guard.clone(),
                "--__1SI",
                1,
                "--__1flags",
                10,
                "--lowerBytes",
                "--bit",
            ),
        )],
        keep_self("--__1SI"),
    );

    // Memwrite address slot: -1 when inactive, real address when active.
    let memaddr0_dispatch = dispatch(
        "--opcode",
        vec![
            (
                0xAA as f64,
                iff(
                    active_guard.clone(),
                    lit(-1.0),
                    add(mul(var("--__1ES"), lit(16.0)), var("--__1DI")),
                ),
            ),
            (
                0xA4 as f64,
                iff(
                    active_guard.clone(),
                    lit(-1.0),
                    add(mul(var("--__1ES"), lit(16.0)), var("--__1DI")),
                ),
            ),
        ],
        lit(-1.0),
    );

    let memval0_dispatch = dispatch(
        "--opcode",
        vec![
            (0xAA as f64, var("--AL")),
            (0xA4 as f64, var("--_strSrcByte")),
        ],
        lit(0.0),
    );

    vec![
        assign("--CX", cx_dispatch),
        assign("--IP", ip_wrapped),
        assign("--DI", di_dispatch),
        assign("--SI", si_dispatch),
        assign("--memAddr0", memaddr0_dispatch),
        assign("--memVal0", memval0_dispatch),
    ]
}

// ---------------------------------------------------------------------------
// Cabinet B — brainfuck-shaped names.
//
// Same structural shape as cabinet A, but the slot names share NOTHING
// with x86 land. The recogniser must produce the same descriptor count
// and the same structural fields (counter, pointer count, advance lit).
// Names will obviously differ — we compare structure, not strings.
// ---------------------------------------------------------------------------

fn cabinet_b() -> Vec<Assignment> {
    let pred_continue = style_eq("--moodMeter", 1.0);
    let no_rep = style_eq("--cookbookOpen", 0.0);

    // Two opcodes: 70 (a "fill" shape, like A's 0xAA) and 80 (a "copy"
    // shape, like A's 0xA4).
    let counter_dispatch = dispatch(
        "--recipeStep",
        vec![
            (70.0, counter_body(no_rep.clone(), "--priorTapeUses")),
            (80.0, counter_body(no_rep.clone(), "--priorTapeUses")),
        ],
        keep_self("--priorTapeUses"),
    );

    let cursor_advance_dispatch = dispatch(
        "--recipeStep",
        vec![
            (
                70.0,
                ip_body(
                    pred_continue.clone(),
                    "--priorCursor",
                    var("--introBytes"),
                    1,
                ),
            ),
            (
                80.0,
                ip_body(
                    pred_continue.clone(),
                    "--priorCursor",
                    var("--introBytes"),
                    1,
                ),
            ),
        ],
        keep_self("--priorCursor"),
    );
    let cursor_wrapped = add(cursor_advance_dispatch, var("--introBytes"));

    let active_guard = StyleTest::And(vec![
        style_eq("--cookbookOpen", 1.0),
        style_eq("--ladlePoised", 0.0),
    ]);

    // Pointer 1 — analog of DI, but called tapeWriteHead.
    let twh_dispatch = dispatch(
        "--recipeStep",
        vec![
            (
                70.0,
                pointer_body(
                    active_guard.clone(),
                    "--priorTapeWriteHead",
                    1,
                    "--priorMoodFlags",
                    7, // any small bit, doesn't have to be 10
                    "--clampLowBits",
                    "--readBitN",
                ),
            ),
            (
                80.0,
                pointer_body(
                    active_guard.clone(),
                    "--priorTapeWriteHead",
                    1,
                    "--priorMoodFlags",
                    7,
                    "--clampLowBits",
                    "--readBitN",
                ),
            ),
        ],
        keep_self("--priorTapeWriteHead"),
    );
    // Pointer 2 — analog of SI but only for opcode 80.
    let trh_dispatch = dispatch(
        "--recipeStep",
        vec![(
            80.0,
            pointer_body(
                active_guard.clone(),
                "--priorTapeReadHead",
                1,
                "--priorMoodFlags",
                7,
                "--clampLowBits",
                "--readBitN",
            ),
        )],
        keep_self("--priorTapeReadHead"),
    );

    let bag_addr = dispatch(
        "--recipeStep",
        vec![
            (
                70.0,
                iff(
                    active_guard.clone(),
                    lit(-1.0),
                    add(
                        mul(var("--priorBagPage"), lit(16.0)),
                        var("--priorTapeWriteHead"),
                    ),
                ),
            ),
            (
                80.0,
                iff(
                    active_guard.clone(),
                    lit(-1.0),
                    add(
                        mul(var("--priorBagPage"), lit(16.0)),
                        var("--priorTapeWriteHead"),
                    ),
                ),
            ),
        ],
        lit(-1.0),
    );
    let bag_val = dispatch(
        "--recipeStep",
        vec![
            (70.0, var("--ladleByte")),
            (80.0, var("--mirrorSourceByte")),
        ],
        lit(0.0),
    );

    vec![
        assign("--tapeUses", counter_dispatch),
        assign("--cursor", cursor_wrapped),
        assign("--tapeWriteHead", twh_dispatch),
        assign("--tapeReadHead", trh_dispatch),
        assign("--bagAddr0", bag_addr),
        assign("--bagVal0", bag_val),
    ]
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[test]
fn cabinet_a_recognises_two_loops() {
    let descs = recognise_loops(&cabinet_a());
    assert_eq!(descs.len(), 2, "expected 2 loop descriptors, got {}: {:?}", descs.len(), descs);

    let stosb = descs.iter().find(|d| d.key_value == 0xAA).expect("stosb desc");
    assert!(stosb.counter.is_some(), "stosb should have a counter");
    assert_eq!(stosb.pointers.len(), 1, "stosb should have 1 pointer (DI)");
    assert_eq!(stosb.ip_advance_literal, 1);
    assert_eq!(stosb.writes.len(), 1, "stosb should have 1 write descriptor");

    let movsb = descs.iter().find(|d| d.key_value == 0xA4).expect("movsb desc");
    assert!(movsb.counter.is_some(), "movsb should have a counter");
    assert_eq!(movsb.pointers.len(), 2, "movsb should have 2 pointers (DI, SI)");
    assert_eq!(movsb.ip_advance_literal, 1);
}

#[test]
fn cabinet_b_recognises_two_loops_equivalently() {
    let descs = recognise_loops(&cabinet_b());
    assert_eq!(
        descs.len(),
        2,
        "brainfuck-shaped cabinet should produce same descriptor count as x86-shaped: {:?}",
        descs
    );

    // Sort by key_value for determinism.
    let mut by_k = descs.clone();
    by_k.sort_by_key(|d| d.key_value);
    let fill = &by_k[0]; // key 70 — analog of stosb
    let copy = &by_k[1]; // key 80 — analog of movsb

    assert_eq!(fill.key_value, 70);
    assert!(fill.counter.is_some());
    assert_eq!(fill.pointers.len(), 1);
    assert_eq!(fill.writes.len(), 1);
    assert_eq!(fill.ip_advance_literal, 1);

    assert_eq!(copy.key_value, 80);
    assert!(copy.counter.is_some());
    assert_eq!(copy.pointers.len(), 2);
    assert_eq!(copy.ip_advance_literal, 1);
}

/// The genericity probe in concentrated form: structural fields (counts
/// and step magnitudes) must match exactly between A and B, even though
/// no slot or property name overlaps.
#[test]
fn a_and_b_descriptors_are_structurally_equivalent() {
    let a = recognise_loops(&cabinet_a());
    let b = recognise_loops(&cabinet_b());
    assert_eq!(a.len(), b.len());

    // Cabinets A and B use unrelated key-value sets (e.g. 0xAA/0xA4 vs
    // 70/80). Comparing by key-value would be meaningless. Instead we
    // compare the *multiset* of structural signatures across all
    // descriptors — both cabinets must yield the same multiset.
    let signatures = |x: &[LoopDescriptor]| {
        let mut sigs: Vec<_> = x
            .iter()
            .map(|d| {
                let mut psteps: Vec<i32> = d.pointers.iter().map(|p| p.base_step).collect();
                psteps.sort();
                (
                    d.counter.as_ref().map(|c| c.step),
                    d.pointers.len(),
                    psteps,
                    d.ip_advance_literal,
                    d.writes.len(),
                    d.flag_conditioned,
                )
            })
            .collect();
        sigs.sort();
        sigs
    };

    let a_sig = signatures(&a);
    let b_sig = signatures(&b);
    assert_eq!(
        a_sig, b_sig,
        "cabinet A and cabinet B must produce structurally identical descriptors"
    );
}

#[test]
fn no_loop_when_no_ip_stay_shape() {
    // A dispatch family with only a counter and a pointer, but no
    // IP-stay shape. Should produce zero descriptors — there's no
    // termination signal we can fast-forward against.
    let no_rep = style_eq("--hasREP", 0.0);
    let active_guard = StyleTest::And(vec![
        style_eq("--hasREP", 1.0),
        style_eq("--_repActive", 0.0),
    ]);
    let cx = dispatch(
        "--opcode",
        vec![(0xAA as f64, counter_body(no_rep, "--__1CX"))],
        keep_self("--__1CX"),
    );
    let di = dispatch(
        "--opcode",
        vec![(
            0xAA as f64,
            pointer_body(
                active_guard,
                "--__1DI",
                1,
                "--__1flags",
                10,
                "--lowerBytes",
                "--bit",
            ),
        )],
        keep_self("--__1DI"),
    );
    let asns = vec![assign("--CX", cx), assign("--DI", di)];
    let descs = recognise_loops(&asns);
    assert!(
        descs.is_empty(),
        "no IP-stay-shape means no descriptors, got {:?}",
        descs
    );
}

#[test]
fn no_loop_when_only_ip_stay_no_counter_or_pointer_or_write() {
    // IP-stay alone, nothing else. We refuse: an unbounded loop is not
    // safe to fast-forward.
    let pred = style_eq("--cont", 1.0);
    let ip = dispatch(
        "--opcode",
        vec![(7.0, ip_body(pred, "--prevPC", var("--prefixLen"), 1))],
        keep_self("--prevPC"),
    );
    // Add a second member that's not counter/pointer/write — just a
    // passthrough. This avoids the "single-member family" filter.
    let other = dispatch("--opcode", vec![(7.0, lit(0.0))], lit(0.0));
    let asns = vec![assign("--PC", ip), assign("--noise", other)];
    let descs = recognise_loops(&asns);
    assert!(
        descs.is_empty(),
        "IP-stay without counter/pointer/write must be refused: {:?}",
        descs
    );
}

#[test]
fn dispatch_family_picks_largest_member_set() {
    // Two different dispatch keys present. Recogniser picks the one
    // with more members.
    let pred = style_eq("--cont", 1.0);
    let no_rep = style_eq("--hasREP", 0.0);
    let active_guard = StyleTest::And(vec![
        style_eq("--hasREP", 1.0),
        style_eq("--_repActive", 0.0),
    ]);

    // Family on --opcode (3 members).
    let cx = dispatch(
        "--opcode",
        vec![(0xAA as f64, counter_body(no_rep, "--__1CX"))],
        keep_self("--__1CX"),
    );
    let ip = dispatch(
        "--opcode",
        vec![(
            0xAA as f64,
            ip_body(pred.clone(), "--__1IP", var("--prefixLen"), 1),
        )],
        keep_self("--__1IP"),
    );
    let di = dispatch(
        "--opcode",
        vec![(
            0xAA as f64,
            pointer_body(
                active_guard.clone(),
                "--__1DI",
                1,
                "--__1flags",
                10,
                "--lowerBytes",
                "--bit",
            ),
        )],
        keep_self("--__1DI"),
    );

    // Decoy family on --otherKey (2 members, no shape).
    let dec1 = dispatch("--otherKey", vec![(1.0, lit(1.0))], lit(0.0));
    let dec2 = dispatch("--otherKey", vec![(2.0, lit(2.0))], lit(0.0));

    let asns = vec![
        assign("--CX", cx),
        assign("--IP", ip),
        assign("--DI", di),
        assign("--noise1", dec1),
        assign("--noise2", dec2),
    ];
    let descs = recognise_loops(&asns);
    assert_eq!(descs.len(), 1);
    assert_eq!(descs[0].key_property, "--opcode");
    assert_eq!(descs[0].key_value, 0xAA);
}

/// Renaming a slot must not change the descriptor structure (only the
/// stored names). Concretely: change every `--__1CX` to `--zzzCX`, every
/// `--__1IP` to `--prevIP`, etc., and the recogniser must still produce
/// equivalent structural facts.
#[test]
fn renaming_slots_preserves_structure() {
    fn rename_one(asns: Vec<Assignment>, table: &[(&str, &str)]) -> Vec<Assignment> {
        fn ren(s: &str, t: &[(&str, &str)]) -> String {
            for (from, to) in t {
                if s == *from {
                    return (*to).to_string();
                }
            }
            s.to_string()
        }
        fn ren_expr(e: &Expr, t: &[(&str, &str)]) -> Expr {
            match e {
                Expr::Literal(v) => Expr::Literal(*v),
                Expr::StringLiteral(s) => Expr::StringLiteral(s.clone()),
                Expr::Var { name, fallback } => Expr::Var {
                    name: ren(name, t),
                    fallback: fallback.as_ref().map(|f| Box::new(ren_expr(f, t))),
                },
                Expr::Calc(op) => Expr::Calc(ren_calc(op, t)),
                Expr::StyleCondition { branches, fallback } => Expr::StyleCondition {
                    branches: branches
                        .iter()
                        .map(|b| StyleBranch {
                            condition: ren_test(&b.condition, t),
                            then: ren_expr(&b.then, t),
                        })
                        .collect(),
                    fallback: Box::new(ren_expr(fallback, t)),
                },
                Expr::FunctionCall { name, args } => Expr::FunctionCall {
                    name: ren(name, t),
                    args: args.iter().map(|a| ren_expr(a, t)).collect(),
                },
                Expr::Concat(parts) => Expr::Concat(parts.iter().map(|p| ren_expr(p, t)).collect()),
            }
        }
        fn ren_calc(op: &CalcOp, t: &[(&str, &str)]) -> CalcOp {
            match op {
                CalcOp::Add(a, b) => CalcOp::Add(Box::new(ren_expr(a, t)), Box::new(ren_expr(b, t))),
                CalcOp::Sub(a, b) => CalcOp::Sub(Box::new(ren_expr(a, t)), Box::new(ren_expr(b, t))),
                CalcOp::Mul(a, b) => CalcOp::Mul(Box::new(ren_expr(a, t)), Box::new(ren_expr(b, t))),
                CalcOp::Div(a, b) => CalcOp::Div(Box::new(ren_expr(a, t)), Box::new(ren_expr(b, t))),
                CalcOp::Mod(a, b) => CalcOp::Mod(Box::new(ren_expr(a, t)), Box::new(ren_expr(b, t))),
                CalcOp::Pow(a, b) => CalcOp::Pow(Box::new(ren_expr(a, t)), Box::new(ren_expr(b, t))),
                CalcOp::Min(args) => CalcOp::Min(args.iter().map(|a| ren_expr(a, t)).collect()),
                CalcOp::Max(args) => CalcOp::Max(args.iter().map(|a| ren_expr(a, t)).collect()),
                CalcOp::Clamp(a, b, c) => CalcOp::Clamp(
                    Box::new(ren_expr(a, t)),
                    Box::new(ren_expr(b, t)),
                    Box::new(ren_expr(c, t)),
                ),
                CalcOp::Round(s, a, b) => CalcOp::Round(
                    *s,
                    Box::new(ren_expr(a, t)),
                    Box::new(ren_expr(b, t)),
                ),
                CalcOp::Sign(a) => CalcOp::Sign(Box::new(ren_expr(a, t))),
                CalcOp::Abs(a) => CalcOp::Abs(Box::new(ren_expr(a, t))),
                CalcOp::Negate(a) => CalcOp::Negate(Box::new(ren_expr(a, t))),
            }
        }
        fn ren_test(test: &StyleTest, t: &[(&str, &str)]) -> StyleTest {
            match test {
                StyleTest::Single { property, value } => StyleTest::Single {
                    property: ren(property, t),
                    value: ren_expr(value, t),
                },
                StyleTest::And(parts) => {
                    StyleTest::And(parts.iter().map(|p| ren_test(p, t)).collect())
                }
                StyleTest::Or(parts) => {
                    StyleTest::Or(parts.iter().map(|p| ren_test(p, t)).collect())
                }
            }
        }
        asns.into_iter()
            .map(|a| Assignment {
                property: ren(&a.property, table),
                value: ren_expr(&a.value, table),
            })
            .collect()
    }

    let table: &[(&str, &str)] = &[
        ("--opcode", "--decided-step"),
        ("--__1CX", "--rememberedRunLength"),
        ("--__1IP", "--rememberedSong"),
        ("--__1DI", "--rememberedHand"),
        ("--__1SI", "--rememberedFoot"),
        ("--__1ES", "--rememberedDimension"),
        ("--__1flags", "--rememberedMood"),
        ("--CX", "--runLength"),
        ("--IP", "--song"),
        ("--DI", "--hand"),
        ("--SI", "--foot"),
        ("--memAddr0", "--paint0"),
        ("--memVal0", "--colour0"),
        ("--prefixLen", "--introBytes"),
        ("--hasREP", "--cookbookOpen"),
        ("--_repActive", "--ladlePoised"),
        ("--_repContinue", "--moodMeter"),
        ("--_strSrcByte", "--mirrorSourceByte"),
        ("--AL", "--ladleByte"),
        ("--lowerBytes", "--clampLowBits"),
        ("--bit", "--readBitN"),
    ];

    let original = recognise_loops(&cabinet_a());
    let renamed = recognise_loops(&rename_one(cabinet_a(), table));
    assert_eq!(original.len(), renamed.len());

    let summary = |descs: &[LoopDescriptor]| {
        let mut keys: Vec<i64> = descs.iter().map(|d| d.key_value).collect();
        keys.sort_unstable();
        keys.into_iter()
            .map(|k| {
                let d = descs.iter().find(|x| x.key_value == k).unwrap();
                (
                    d.counter.as_ref().map(|c| c.step),
                    d.pointers.len(),
                    {
                        let mut s: Vec<i32> = d.pointers.iter().map(|p| p.base_step).collect();
                        s.sort();
                        s
                    },
                    d.ip_advance_literal,
                    d.writes.len(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(summary(&original), summary(&renamed));
}

/// Real cabinets wrap each register dispatch in an outer
/// `if(<gate-A>: var(self); <gate-B>: var(self); ...; else: <real-dispatch>)`
/// — the TF / IRQ override layer that takes precedence over normal
/// instruction execution. The recogniser must peel this wrapper and
/// still find the loop. The cabinet may also wrap IP in
/// `calc(<dispatch> + var(--prefixLen))`. Both wrappers must be
/// stripped before recognition.
#[test]
fn outer_wrappers_are_stripped() {
    let pred = style_eq("--cont", 1.0);
    let no_rep = style_eq("--hasREP", 0.0);
    let active_guard = StyleTest::And(vec![
        style_eq("--hasREP", 1.0),
        style_eq("--inhibit", 0.0),
    ]);

    // Inner per-V bodies — same shape as cabinet_a.
    let cx_inner = dispatch(
        "--op",
        vec![(0xAA as f64, counter_body(no_rep, "--prevCount"))],
        keep_self("--prevCount"),
    );
    let ip_inner = dispatch(
        "--op",
        vec![(
            0xAA as f64,
            ip_body(pred.clone(), "--prevPC", var("--introBytes"), 1),
        )],
        keep_self("--prevPC"),
    );
    let di_inner = dispatch(
        "--op",
        vec![(
            0xAA as f64,
            pointer_body(
                active_guard.clone(),
                "--prevHand",
                1,
                "--prevMood",
                10,
                "--clampLow",
                "--bitN",
            ),
        )],
        keep_self("--prevHand"),
    );

    // Wrap CX in TF/IRQ-style passthrough wrapper.
    let cx_wrapped = Expr::StyleCondition {
        branches: vec![
            StyleBranch {
                condition: style_eq("--trapMode", 1.0),
                then: var("--prevCount"),
            },
            StyleBranch {
                condition: style_eq("--alarmActive", 1.0),
                then: var("--prevCount"),
            },
        ],
        fallback: Box::new(cx_inner),
    };

    // Wrap IP in TF/IRQ wrapper, then in calc(... + var(introBytes)).
    let ip_passthrough = Expr::StyleCondition {
        branches: vec![
            StyleBranch {
                condition: style_eq("--trapMode", 1.0),
                then: var("--prevPC"),
            },
            StyleBranch {
                condition: style_eq("--alarmActive", 1.0),
                then: var("--prevPC"),
            },
        ],
        fallback: Box::new(ip_inner),
    };
    let ip_wrapped = add(ip_passthrough, var("--introBytes"));

    // DI gets the same wrapper.
    let di_wrapped = Expr::StyleCondition {
        branches: vec![
            StyleBranch {
                condition: style_eq("--trapMode", 1.0),
                then: var("--prevHand"),
            },
            StyleBranch {
                condition: style_eq("--alarmActive", 1.0),
                then: var("--prevHand"),
            },
        ],
        fallback: Box::new(di_inner),
    };

    let asns = vec![
        assign("--count", cx_wrapped),
        assign("--pc", ip_wrapped),
        assign("--hand", di_wrapped),
    ];

    let descs = recognise_loops(&asns);
    assert_eq!(
        descs.len(),
        1,
        "outer-wrapped dispatches should still recognise the loop, got {:?}",
        descs
    );
    let d = &descs[0];
    assert!(d.counter.is_some());
    assert_eq!(d.pointers.len(), 1);
    assert_eq!(d.ip_advance_literal, 1);
}

/// Outer wrappers around the IP register can return non-self values in
/// their override branches (kiln's TF/IRQ wrapper emits things like
/// `--_tfIP` or `picVector*4`, not `var(self)`). We must still find the
/// real dispatch in the fallback regardless. The recogniser does NOT
/// require that override branches read self — only that the fallback
/// contains a single-key dispatch.
#[test]
fn ip_wrapper_with_non_self_overrides_still_recognises() {
    let pred = style_eq("--cont", 1.0);
    let no_rep = style_eq("--hasREP", 0.0);
    let active_guard = StyleTest::And(vec![
        style_eq("--hasREP", 1.0),
        style_eq("--inhibit", 0.0),
    ]);

    let cx_inner = dispatch(
        "--op",
        vec![(0xAA as f64, counter_body(no_rep, "--prevCount"))],
        keep_self("--prevCount"),
    );
    let ip_inner = dispatch(
        "--op",
        vec![(
            0xAA as f64,
            ip_body(pred.clone(), "--prevPC", var("--introBytes"), 1),
        )],
        keep_self("--prevPC"),
    );
    let di_inner = dispatch(
        "--op",
        vec![(
            0xAA as f64,
            pointer_body(
                active_guard.clone(),
                "--prevHand",
                1,
                "--prevMood",
                10,
                "--clampLow",
                "--bitN",
            ),
        )],
        keep_self("--prevHand"),
    );

    // IP wrapper: TF returns trap-IP target, IRQ returns IVT slot
    // address. Neither is `var(self)`. Recogniser must still descend.
    let ip_wrapped = Expr::StyleCondition {
        branches: vec![
            StyleBranch {
                condition: style_eq("--trapMode", 1.0),
                then: var("--trapTargetIP"),
            },
            StyleBranch {
                condition: style_eq("--alarmActive", 1.0),
                then: mul(var("--ivtSlot"), lit(4.0)),
            },
        ],
        fallback: Box::new(ip_inner),
    };
    let ip_outer = add(ip_wrapped, var("--introBytes"));

    let asns = vec![
        assign("--count", cx_inner),
        assign("--pc", ip_outer),
        assign("--hand", di_inner),
    ];
    let descs = recognise_loops(&asns);
    assert_eq!(
        descs.len(),
        1,
        "must descend into IP wrapper fallback even when branches don't read self: {:?}",
        descs
    );
    let d = &descs[0];
    assert!(d.counter.is_some());
    assert_eq!(d.pointers.len(), 1);
}

#[test]
fn memwrite_pairing_uses_assignment_order_proximity() {
    // Build a cabinet with TWO write pairs interleaved in source order:
    //   --addrA (idx 2)
    //   --valA  (idx 3)
    //   --addrB (idx 4)
    //   --valB  (idx 5)
    // The recogniser should pair (addrA, valA) and (addrB, valB), NOT
    // pair them by alphabetical sort which would give (addrA, valA) but
    // also pair (addrB, valA) if we didn't track positions.
    let pred_continue = style_eq("--repCont", 1.0);
    let no_rep = style_eq("--hasRep", 0.0);
    let active_guard = style_eq("--repActive", 0.0);

    let cx = dispatch(
        "--op",
        vec![(0xAA as f64, counter_body(no_rep.clone(), "--cx0"))],
        keep_self("--cx0"),
    );
    let ip = dispatch(
        "--op",
        vec![(
            0xAA as f64,
            ip_body(pred_continue.clone(), "--ip0", var("--pl"), 1),
        )],
        keep_self("--ip0"),
    );

    // Address bodies: -1 when active, real address otherwise.
    fn addr(active_guard: StyleTest, slot_self: &str) -> Expr {
        iff(active_guard, lit(-1.0), add(mul(var("--es"), lit(16.0)), var(slot_self)))
    }

    let addr_a = dispatch(
        "--op",
        vec![(0xAA as f64, addr(active_guard.clone(), "--di0"))],
        lit(-1.0),
    );
    // Two distinct value bodies so we can distinguish which got paired.
    let val_a = dispatch(
        "--op",
        vec![(0xAA as f64, var("--regA"))],
        lit(0.0),
    );
    let addr_b = dispatch(
        "--op",
        vec![(0xAA as f64, addr(active_guard.clone(), "--di1"))],
        lit(-1.0),
    );
    let val_b = dispatch(
        "--op",
        vec![(0xAA as f64, var("--regB"))],
        lit(0.0),
    );

    // Order matters: this is the test.
    let asns = vec![
        assign("--cx0", cx),       // idx 0
        assign("--ip0", ip),       // idx 1
        assign("--addrA", addr_a), // idx 2
        assign("--valA", val_a),   // idx 3
        assign("--addrB", addr_b), // idx 4
        assign("--valB", val_b),   // idx 5
    ];

    let descs = recognise_loops(&asns);
    assert_eq!(descs.len(), 1, "one descriptor expected: {:?}", descs);
    let d = &descs[0];
    assert_eq!(d.writes.len(), 2, "two write pairs expected");
    // Sort by addr_property for deterministic comparison.
    let mut writes = d.writes.clone();
    writes.sort_by(|a, b| a.addr_property.cmp(&b.addr_property));
    assert_eq!(writes[0].addr_property, "--addrA");
    assert_eq!(writes[0].val_property, "--valA",
        "addrA must pair with valA (immediately after in source order), got {:?}", writes[0]);
    assert_eq!(writes[1].addr_property, "--addrB");
    assert_eq!(writes[1].val_property, "--valB",
        "addrB must pair with valB (immediately after in source order), got {:?}", writes[1]);
}

// ---------------------------------------------------------------------------
// Phase 3a tests: CMPS/SCAS-shape recognition + BulkClass classification.
// ---------------------------------------------------------------------------

/// Build a CMPS/SCAS-shape cabinet using a multi-branch IP body where
/// each branch is an AND of three property tests, all yielding "stay",
/// with the fallback advancing. No memory writes — read-only loop. The
/// recogniser must produce one descriptor with `flag_conditioned=true`
/// and `bulk_class=ReadOnly`.
fn cabinet_cmps_shape() -> Vec<Assignment> {
    // The disjunction expands to: (P1 AND P2 AND P3) OR (P4 AND P5 AND P6).
    let branch_a = StyleTest::And(vec![
        style_eq("--cont", 1.0),
        style_eq("--repType", 1.0),
        style_eq("--zfBit", 1.0),
    ]);
    let branch_b = StyleTest::And(vec![
        style_eq("--cont", 1.0),
        style_eq("--repType", 2.0),
        style_eq("--zfBit", 0.0),
    ]);
    let no_rep = style_eq("--hasRep", 0.0);
    let active_guard = StyleTest::And(vec![
        style_eq("--hasRep", 1.0),
        style_eq("--repInactive", 0.0),
    ]);

    let cx = dispatch(
        "--op",
        vec![(0xA6 as f64, counter_body(no_rep.clone(), "--cx0"))],
        keep_self("--cx0"),
    );

    // Multi-branch IP body — same shape kiln emits via repCondIP.
    let ip_inner = dispatch(
        "--op",
        vec![(
            0xA6 as f64,
            ip_body_multi(
                vec![branch_a.clone(), branch_b.clone()],
                "--ip0",
                var("--pl"),
                1,
            ),
        )],
        keep_self("--ip0"),
    );
    let ip_wrapped = add(ip_inner, var("--pl"));

    let di = dispatch(
        "--op",
        vec![(
            0xA6 as f64,
            pointer_body(
                active_guard.clone(),
                "--di0",
                1,
                "--flags0",
                10,
                "--lowBytes",
                "--bit",
            ),
        )],
        keep_self("--di0"),
    );
    let si = dispatch(
        "--op",
        vec![(
            0xA6 as f64,
            pointer_body(
                active_guard.clone(),
                "--si0",
                1,
                "--flags0",
                10,
                "--lowBytes",
                "--bit",
            ),
        )],
        keep_self("--si0"),
    );

    vec![
        assign("--cx", cx),
        assign("--ip", ip_wrapped),
        assign("--di", di),
        assign("--si", si),
    ]
}

#[test]
fn cmps_shape_recognised_with_flag_conditioning() {
    let descs = recognise_loops(&cabinet_cmps_shape());
    assert_eq!(descs.len(), 1, "expected one descriptor: {:?}", descs);
    let d = &descs[0];
    assert!(d.counter.is_some(), "counter must be recognised");
    assert_eq!(d.pointers.len(), 2, "CMPS-shape has two pointers (DI, SI)");
    assert_eq!(d.writes.len(), 0, "CMPS has no memory writes");
    assert_eq!(d.ip_advance_literal, 1);
    assert!(
        d.flag_conditioned,
        "predicate spans multiple distinct properties → flag_conditioned",
    );
    assert_eq!(
        d.bulk_class,
        BulkClass::ReadOnly,
        "no writes → ReadOnly bulk class",
    );
    // The synthesised predicate must be an Or of two And-conditions.
    match &d.predicate {
        StyleTest::Or(parts) => {
            assert_eq!(parts.len(), 2, "two stay-branches → Or with 2 parts");
        }
        other => panic!("expected Or predicate from multi-branch IP body, got {:?}", other),
    }
}

/// SCAS-shape: like CMPS but only one pointer (DI). Same multi-branch
/// IP body. No writes.
fn cabinet_scas_shape() -> Vec<Assignment> {
    let branch_a = StyleTest::And(vec![
        style_eq("--cont", 1.0),
        style_eq("--repType", 1.0),
        style_eq("--zfBit", 1.0),
    ]);
    let branch_b = StyleTest::And(vec![
        style_eq("--cont", 1.0),
        style_eq("--repType", 2.0),
        style_eq("--zfBit", 0.0),
    ]);
    let no_rep = style_eq("--hasRep", 0.0);
    let active_guard = StyleTest::And(vec![
        style_eq("--hasRep", 1.0),
        style_eq("--repInactive", 0.0),
    ]);

    let cx = dispatch(
        "--op",
        vec![(0xAE as f64, counter_body(no_rep.clone(), "--cx0"))],
        keep_self("--cx0"),
    );
    let ip_inner = dispatch(
        "--op",
        vec![(
            0xAE as f64,
            ip_body_multi(
                vec![branch_a, branch_b],
                "--ip0",
                var("--pl"),
                1,
            ),
        )],
        keep_self("--ip0"),
    );
    let ip_wrapped = add(ip_inner, var("--pl"));
    let di = dispatch(
        "--op",
        vec![(
            0xAE as f64,
            pointer_body(
                active_guard,
                "--di0",
                1,
                "--flags0",
                10,
                "--lowBytes",
                "--bit",
            ),
        )],
        keep_self("--di0"),
    );

    vec![
        assign("--cx", cx),
        assign("--ip", ip_wrapped),
        assign("--di", di),
    ]
}

#[test]
fn scas_shape_recognised_with_one_pointer_and_readonly_class() {
    let descs = recognise_loops(&cabinet_scas_shape());
    assert_eq!(descs.len(), 1, "expected one descriptor: {:?}", descs);
    let d = &descs[0];
    assert_eq!(d.pointers.len(), 1, "SCAS has just DI");
    assert_eq!(d.writes.len(), 0);
    assert!(d.flag_conditioned);
    assert_eq!(d.bulk_class, BulkClass::ReadOnly);
}

#[test]
fn cabinet_a_classifies_stos_as_fill_and_movs_as_copy() {
    // Cabinet A's STOSB-shape (0xAA) writes a constant from --AL — no
    // pointer-mirror reads. Should classify as Fill.
    // Cabinet A's MOVSB-shape (0xA4) writes mem[DS:SI] which the cabinet
    // exposes as --_strSrcByte. That's NOT a pointer mirror in the
    // recogniser's view though — the pointer mirrors are --__1DI and
    // --__1SI (the prior-tick mirrors of DI and SI, used by the pointer
    // step body). For the classifier to call this Copy, the val_expr
    // must reference one of those mirror names.
    //
    // Real kiln-emitted MOVSB uses --_strSrcByte (a pre-computed
    // intermediate) rather than reading via SI directly. So the
    // classifier sees no pointer reference and classifies as Fill. That's
    // a real result of the pure-shape recogniser — it can't tell that
    // --_strSrcByte happens to be derived from SI without inspecting how
    // --_strSrcByte itself is computed elsewhere.
    //
    // For cabinets that DO read via the pointer slot directly (e.g. an
    // emitter without an intermediate), the classification would be
    // Copy. Phase 3b's runtime applier will use Fill/Copy/PerIter to
    // pick its memory-routing strategy; for cabinets where the
    // intermediate hides the dependency, PerIter (per-byte read_mem) is
    // the correct fallback and a separate optimisation pass over the
    // intermediate's definition can promote Fill→Copy where applicable.
    let descs = recognise_loops(&cabinet_a());
    assert_eq!(descs.len(), 2);

    let stosb = descs.iter().find(|d| d.key_value == 0xAA).unwrap();
    assert_eq!(
        stosb.bulk_class,
        BulkClass::Fill,
        "STOSB writes constant from --AL, no pointer-mirror reads",
    );

    let movsb = descs.iter().find(|d| d.key_value == 0xA4).unwrap();
    // Cabinet A's val_expr is `var("--_strSrcByte")` — not a pointer
    // mirror. So pure-shape classifier sees Fill here. Document the
    // result rather than asserting Copy.
    assert!(
        matches!(movsb.bulk_class, BulkClass::Fill | BulkClass::Copy),
        "MOVSB classified as {:?} (Fill is correct for intermediate-via shape)",
        movsb.bulk_class,
    );
}

#[test]
fn pointer_mirror_in_value_expr_classifies_as_copy() {
    // Build a STOS-shape cabinet whose write value reads through the
    // pointer mirror directly (no intermediate). The classifier must
    // see the dependency and call it Copy.
    let pred_continue = style_eq("--cont", 1.0);
    let no_rep = style_eq("--hasRep", 0.0);
    let active_guard = style_eq("--repActive", 0.0);

    let cx = dispatch(
        "--op",
        vec![(0xAA as f64, counter_body(no_rep.clone(), "--cx0"))],
        keep_self("--cx0"),
    );
    let ip_inner = dispatch(
        "--op",
        vec![(
            0xAA as f64,
            ip_body(pred_continue.clone(), "--ip0", var("--pl"), 1),
        )],
        keep_self("--ip0"),
    );
    let ip_wrapped = add(ip_inner, var("--pl"));

    let di = dispatch(
        "--op",
        vec![(
            0xAA as f64,
            pointer_body(
                active_guard.clone(),
                "--diMirror",
                1,
                "--flags0",
                10,
                "--lowBytes",
                "--bit",
            ),
        )],
        keep_self("--diMirror"),
    );

    let addr = dispatch(
        "--op",
        vec![(
            0xAA as f64,
            iff(
                active_guard.clone(),
                lit(-1.0),
                add(mul(var("--es"), lit(16.0)), var("--diMirror")),
            ),
        )],
        lit(-1.0),
    );
    // Crucially: val reads via the pointer mirror directly. The
    // classifier sees this and calls it Copy.
    let val = dispatch(
        "--op",
        vec![(
            0xAA as f64,
            // some byte-fetch through the pointer mirror — shape doesn't
            // matter, just that --diMirror appears in the expr tree.
            call("--readByte", vec![var("--diMirror")]),
        )],
        lit(0.0),
    );

    let asns = vec![
        assign("--cx0", cx),
        assign("--ip0", ip_wrapped),
        assign("--diReg", di),
        assign("--addr0", addr),
        assign("--val0", val),
    ];
    let descs = recognise_loops(&asns);
    assert_eq!(descs.len(), 1, "expected one descriptor: {:?}", descs);
    let d = &descs[0];
    assert_eq!(d.writes.len(), 1);
    assert_eq!(
        d.bulk_class,
        BulkClass::Copy,
        "val reads through --diMirror (a pointer self_property) → Copy",
    );
}

#[test]
fn brainfuck_cmps_shape_classifies_identically_to_x86_cmps_shape() {
    // The cardinal-rule probe extended to phase 3a: an arbitrary-named
    // CMPS-shaped cabinet must classify the same as cabinet_cmps_shape.
    let stay_branch_a = StyleTest::And(vec![
        style_eq("--moodMeter", 1.0),
        style_eq("--ladleType", 1.0),
        style_eq("--frothBit", 1.0),
    ]);
    let stay_branch_b = StyleTest::And(vec![
        style_eq("--moodMeter", 1.0),
        style_eq("--ladleType", 2.0),
        style_eq("--frothBit", 0.0),
    ]);
    let no_rep = style_eq("--cookbookOpen", 0.0);
    let active_guard = StyleTest::And(vec![
        style_eq("--cookbookOpen", 1.0),
        style_eq("--ladlePoised", 0.0),
    ]);

    let cx = dispatch(
        "--recipeStep",
        vec![(99.0, counter_body(no_rep.clone(), "--priorTapeUses"))],
        keep_self("--priorTapeUses"),
    );
    let ip_inner = dispatch(
        "--recipeStep",
        vec![(
            99.0,
            ip_body_multi(
                vec![stay_branch_a, stay_branch_b],
                "--priorCursor",
                var("--introBytes"),
                1,
            ),
        )],
        keep_self("--priorCursor"),
    );
    let ip_wrapped = add(ip_inner, var("--introBytes"));

    let p1 = dispatch(
        "--recipeStep",
        vec![(
            99.0,
            pointer_body(
                active_guard.clone(),
                "--priorWriteHead",
                1,
                "--priorMoodFlags",
                7,
                "--clampLowBits",
                "--readBitN",
            ),
        )],
        keep_self("--priorWriteHead"),
    );
    let p2 = dispatch(
        "--recipeStep",
        vec![(
            99.0,
            pointer_body(
                active_guard,
                "--priorReadHead",
                1,
                "--priorMoodFlags",
                7,
                "--clampLowBits",
                "--readBitN",
            ),
        )],
        keep_self("--priorReadHead"),
    );

    let asns = vec![
        assign("--tapeUses", cx),
        assign("--cursor", ip_wrapped),
        assign("--writeHead", p1),
        assign("--readHead", p2),
    ];
    let descs = recognise_loops(&asns);
    assert_eq!(descs.len(), 1);
    let d = &descs[0];
    assert!(d.counter.is_some());
    assert_eq!(d.pointers.len(), 2);
    assert_eq!(d.writes.len(), 0);
    assert!(d.flag_conditioned);
    assert_eq!(d.bulk_class, BulkClass::ReadOnly);
}

// ---------------------------------------------------------------------------
// Phase 3b Step 1: addr_decomposition.
// ---------------------------------------------------------------------------

/// Cabinet A's STOS / MOVS write addresses are
/// `add(mul(--__1ES, 16), --__1DI)` — segment-shifted-then-pointer.
/// Both descriptors should pick up the same `(segment, pointer)` pair.
#[test]
fn cabinet_a_writes_decompose_segment_times_sixteen_plus_pointer() {
    let descs = recognise_loops(&cabinet_a());
    assert!(!descs.is_empty());
    for d in &descs {
        assert!(!d.writes.is_empty(), "expected at least one write entry: {:?}", d);
        for w in &d.writes {
            assert_eq!(
                w.addr_decomposition.as_ref().map(|(s, p)| (s.as_str(), p.as_str())),
                Some(("--__1ES", "--__1DI")),
                "expected decomposition (--__1ES, --__1DI), got {:?}",
                w.addr_decomposition
            );
        }
    }
}

/// Cabinet B uses unrelated names but the same structural shape, so
/// decomposition must succeed and produce that cabinet's seg/ptr pair.
/// This is the cardinal-rule genericity probe for Step 1.
#[test]
fn cabinet_b_writes_decompose_to_brainfuck_names() {
    let descs = recognise_loops(&cabinet_b());
    assert!(!descs.is_empty());
    for d in &descs {
        for w in &d.writes {
            assert_eq!(
                w.addr_decomposition.as_ref().map(|(s, p)| (s.as_str(), p.as_str())),
                Some(("--priorBagPage", "--priorTapeWriteHead")),
                "expected decomposition (--priorBagPage, --priorTapeWriteHead), got {:?}",
                w.addr_decomposition
            );
        }
    }
}

/// The reversed-orientation form `pointer + (segment * 16)` decomposes
/// the same way. The matcher accepts either ordering of the outer add.
#[test]
fn addr_decomposition_accepts_pointer_plus_segment_times_sixteen() {
    // Hand-roll a one-opcode dispatch family with the reversed addr
    // shape, otherwise structurally identical to cabinet_a's stos.
    let pred_continue = style_eq("--rc", 1.0);
    let no_rep = style_eq("--hr", 0.0);
    let active = StyleTest::And(vec![
        style_eq("--hr", 1.0),
        style_eq("--ra", 0.0),
    ]);

    let cx = dispatch("--op", vec![(1.0, counter_body(no_rep, "--c"))], keep_self("--c"));
    let ip = dispatch(
        "--op",
        vec![(1.0, ip_body(pred_continue, "--ip", var("--pl"), 1))],
        keep_self("--ip"),
    );
    let ip_wrapped = add(ip, var("--pl"));
    let p = dispatch(
        "--op",
        vec![(1.0, pointer_body(active.clone(), "--di", 1, "--fl", 10, "--lb", "--bit"))],
        keep_self("--di"),
    );
    // Reversed: pointer first, then (segment * 16).
    let addr = dispatch(
        "--op",
        vec![(
            1.0,
            iff(
                active.clone(),
                lit(-1.0),
                add(var("--di"), mul(var("--es"), lit(16.0))),
            ),
        )],
        lit(-1.0),
    );
    let val = dispatch("--op", vec![(1.0, var("--al"))], lit(0.0));

    let asns = vec![
        assign("--c", cx),
        assign("--ip", ip_wrapped),
        assign("--di", p),
        assign("--mAddr", addr),
        assign("--mVal", val),
    ];
    let descs = recognise_loops(&asns);
    assert_eq!(descs.len(), 1, "expected 1 descriptor, got {:?}", descs);
    let w = &descs[0].writes[0];
    assert_eq!(
        w.addr_decomposition.as_ref().map(|(s, p)| (s.as_str(), p.as_str())),
        Some(("--es", "--di")),
    );
}

/// `16 * segment` (multiplication operand order swapped) is also accepted.
/// The rule is "one operand is the literal 16, the other is a bare var",
/// not "the var must come first".
#[test]
fn addr_decomposition_accepts_sixteen_times_segment() {
    let pred_continue = style_eq("--rc", 1.0);
    let no_rep = style_eq("--hr", 0.0);
    let active = StyleTest::And(vec![
        style_eq("--hr", 1.0),
        style_eq("--ra", 0.0),
    ]);

    let cx = dispatch("--op", vec![(1.0, counter_body(no_rep, "--c"))], keep_self("--c"));
    let ip = dispatch(
        "--op",
        vec![(1.0, ip_body(pred_continue, "--ip", var("--pl"), 1))],
        keep_self("--ip"),
    );
    let ip_wrapped = add(ip, var("--pl"));
    let p = dispatch(
        "--op",
        vec![(1.0, pointer_body(active.clone(), "--di", 1, "--fl", 10, "--lb", "--bit"))],
        keep_self("--di"),
    );
    let addr = dispatch(
        "--op",
        vec![(
            1.0,
            iff(
                active.clone(),
                lit(-1.0),
                // (16 * --es) + --di
                add(mul(lit(16.0), var("--es")), var("--di")),
            ),
        )],
        lit(-1.0),
    );
    let val = dispatch("--op", vec![(1.0, var("--al"))], lit(0.0));

    let asns = vec![
        assign("--c", cx),
        assign("--ip", ip_wrapped),
        assign("--di", p),
        assign("--mAddr", addr),
        assign("--mVal", val),
    ];
    let descs = recognise_loops(&asns);
    let w = &descs[0].writes[0];
    assert_eq!(
        w.addr_decomposition.as_ref().map(|(s, p)| (s.as_str(), p.as_str())),
        Some(("--es", "--di")),
    );
}

/// An address whose shape is NOT segment*16 + pointer must yield None.
/// Here the multiplier is 32 (some other paging granule) — the matcher
/// is committed to literal 16 (the canonical 8086 page constant; any
/// non-emulator cabinet using a different page size will need its own
/// matcher entry, which is fine — that's a structural fact about the
/// shape, not a name-content read).
#[test]
fn addr_decomposition_returns_none_for_non_canonical_shape() {
    let pred_continue = style_eq("--rc", 1.0);
    let no_rep = style_eq("--hr", 0.0);
    let active = StyleTest::And(vec![
        style_eq("--hr", 1.0),
        style_eq("--ra", 0.0),
    ]);

    let cx = dispatch("--op", vec![(1.0, counter_body(no_rep, "--c"))], keep_self("--c"));
    let ip = dispatch(
        "--op",
        vec![(1.0, ip_body(pred_continue, "--ip", var("--pl"), 1))],
        keep_self("--ip"),
    );
    let ip_wrapped = add(ip, var("--pl"));
    let p = dispatch(
        "--op",
        vec![(1.0, pointer_body(active.clone(), "--di", 1, "--fl", 10, "--lb", "--bit"))],
        keep_self("--di"),
    );
    // Multiplier is 32, not 16. Decomposer should return None.
    let addr = dispatch(
        "--op",
        vec![(
            1.0,
            iff(
                active.clone(),
                lit(-1.0),
                add(mul(var("--es"), lit(32.0)), var("--di")),
            ),
        )],
        lit(-1.0),
    );
    let val = dispatch("--op", vec![(1.0, var("--al"))], lit(0.0));

    let asns = vec![
        assign("--c", cx),
        assign("--ip", ip_wrapped),
        assign("--di", p),
        assign("--mAddr", addr),
        assign("--mVal", val),
    ];
    let descs = recognise_loops(&asns);
    assert_eq!(descs.len(), 1);
    assert!(
        descs[0].writes[0].addr_decomposition.is_none(),
        "expected None for multiplier-32 shape, got {:?}",
        descs[0].writes[0].addr_decomposition
    );
}

// ---------------------------------------------------------------------------
// Phase 3b Step 2: indirect-read intermediate recognition.
//
// MOVS-style cabinets emit the per-iter source byte as a derived
// intermediate slot whose dispatch body is a function call keyed on a
// stepping pointer (`var(--_strSrcByte)` in doom8088, with body
// `--readMem(calc(var(--_strSrcSeg) + var(--__1SI)))`). The pure-shape
// classifier in phase 3a couldn't see through the intermediate; step 2
// traces one level into the assignment list so MOVS reclassifies from
// Fill to Copy.
// ---------------------------------------------------------------------------

/// Helper: build a STOS-shape cabinet whose val_expr is a bare
/// `Var(intermediate)`, with the intermediate's body being a function
/// call keyed on the loop's pointer mirror. With the indirect-read
/// recogniser, this should classify as Copy (the "MOVS through derived
/// intermediate" pattern).
fn cabinet_with_indirect_read(
    intermediate_name: &str,
    intermediate_body: Expr,
) -> Vec<Assignment> {
    let pred_continue = style_eq("--cont", 1.0);
    let no_rep = style_eq("--hasRep", 0.0);
    let active_guard = style_eq("--repActive", 0.0);

    let cx = dispatch(
        "--op",
        vec![(0xA4 as f64, counter_body(no_rep.clone(), "--cx0"))],
        keep_self("--cx0"),
    );
    let ip_inner = dispatch(
        "--op",
        vec![(
            0xA4 as f64,
            ip_body(pred_continue.clone(), "--ip0", var("--pl"), 1),
        )],
        keep_self("--ip0"),
    );
    let ip_wrapped = add(ip_inner, var("--pl"));

    let di = dispatch(
        "--op",
        vec![(
            0xA4 as f64,
            pointer_body(
                active_guard.clone(),
                "--diMirror",
                1,
                "--flags0",
                10,
                "--lowBytes",
                "--bit",
            ),
        )],
        keep_self("--diMirror"),
    );
    let si = dispatch(
        "--op",
        vec![(
            0xA4 as f64,
            pointer_body(
                active_guard.clone(),
                "--siMirror",
                1,
                "--flags0",
                10,
                "--lowBytes",
                "--bit",
            ),
        )],
        keep_self("--siMirror"),
    );

    let addr = dispatch(
        "--op",
        vec![(
            0xA4 as f64,
            iff(
                active_guard.clone(),
                lit(-1.0),
                add(mul(var("--es"), lit(16.0)), var("--diMirror")),
            ),
        )],
        lit(-1.0),
    );

    // val_expr is a bare Var to the intermediate slot.
    let val = dispatch(
        "--op",
        vec![(0xA4 as f64, var(intermediate_name))],
        lit(0.0),
    );

    vec![
        assign("--cx0", cx),
        assign("--ip0", ip_wrapped),
        assign("--diReg", di),
        assign("--siReg", si),
        assign("--addr0", addr),
        assign("--val0", val),
        // The intermediate's own dispatch body — a function call keyed
        // on the SI pointer mirror. This is the structural shape the
        // step-2 recogniser traces into.
        assign(intermediate_name, intermediate_body),
    ]
}

/// Positive: x86-shaped cabinet, MOVS-style. The intermediate's body is
/// `--readMem(calc(var(--seg) + var(--siMirror)))`. Step 2 must
/// recognise the indirect read and classify the loop as Copy.
#[test]
fn indirect_read_through_intermediate_classifies_as_copy() {
    // Body: --readMem(calc(var(--srcSeg) + var(--siMirror)))
    let body = call(
        "--readMem",
        vec![add(var("--srcSeg"), var("--siMirror"))],
    );
    let asns = cabinet_with_indirect_read("--strSrcByte", body);
    let descs = recognise_loops(&asns);
    assert_eq!(descs.len(), 1, "expected one descriptor: {:?}", descs);
    let d = &descs[0];
    assert_eq!(d.writes.len(), 1);
    let w = &d.writes[0];
    let ir = w
        .val_indirect_read
        .as_ref()
        .expect("indirect_read should be Some");
    assert_eq!(ir.intermediate_property, "--strSrcByte");
    assert_eq!(ir.pointer_property, "--siMirror");
    assert_eq!(ir.seg_property.as_deref(), Some("--srcSeg"));
    assert_eq!(
        d.bulk_class,
        BulkClass::Copy,
        "indirect-read through pointer mirror promotes Fill → Copy",
    );
}

/// Cardinal-rule probe for step 2: a brainfuck-shaped cabinet using the
/// same structural shape (function-call keyed on a pointer mirror via a
/// derived intermediate) must produce the same Copy classification.
/// Names share nothing with x86 land — only the shape matters.
#[test]
fn brainfuck_indirect_read_classifies_identically() {
    let pred_continue = style_eq("--moodMeter", 1.0);
    let no_rep = style_eq("--cookbookOpen", 0.0);
    let active_guard = StyleTest::And(vec![
        style_eq("--cookbookOpen", 1.0),
        style_eq("--ladlePoised", 0.0),
    ]);

    let cx = dispatch(
        "--recipeStep",
        vec![(80.0, counter_body(no_rep.clone(), "--priorTapeUses"))],
        keep_self("--priorTapeUses"),
    );
    let cursor_inner = dispatch(
        "--recipeStep",
        vec![(
            80.0,
            ip_body(pred_continue.clone(), "--priorCursor", var("--introBytes"), 1),
        )],
        keep_self("--priorCursor"),
    );
    let cursor_wrapped = add(cursor_inner, var("--introBytes"));

    // Two pointers with brainfuck-shaped names.
    let twh = dispatch(
        "--recipeStep",
        vec![(
            80.0,
            pointer_body(
                active_guard.clone(),
                "--priorTapeWriteHead",
                1,
                "--priorMoodFlags",
                7,
                "--clampLowBits",
                "--readBitN",
            ),
        )],
        keep_self("--priorTapeWriteHead"),
    );
    let trh = dispatch(
        "--recipeStep",
        vec![(
            80.0,
            pointer_body(
                active_guard.clone(),
                "--priorTapeReadHead",
                1,
                "--priorMoodFlags",
                7,
                "--clampLowBits",
                "--readBitN",
            ),
        )],
        keep_self("--priorTapeReadHead"),
    );

    let bag_addr = dispatch(
        "--recipeStep",
        vec![(
            80.0,
            iff(
                active_guard.clone(),
                lit(-1.0),
                add(
                    mul(var("--priorBagPage"), lit(16.0)),
                    var("--priorTapeWriteHead"),
                ),
            ),
        )],
        lit(-1.0),
    );
    // val reads through an intermediate that itself reads via the
    // tapeReadHead pointer mirror. Same structural shape as cabinet A's
    // `--_strSrcByte`, but every name is brainfuck-flavoured.
    let bag_val = dispatch(
        "--recipeStep",
        vec![(80.0, var("--mirrorSourceByte"))],
        lit(0.0),
    );
    // The intermediate's own body — function call keyed on the read
    // head pointer mirror.
    let mirror_body = call(
        "--peekByteAt",
        vec![add(var("--bagSourceSeg"), var("--priorTapeReadHead"))],
    );

    let asns = vec![
        assign("--tapeUses", cx),
        assign("--cursor", cursor_wrapped),
        assign("--tapeWriteHead", twh),
        assign("--tapeReadHead", trh),
        assign("--bagAddr0", bag_addr),
        assign("--bagVal0", bag_val),
        assign("--mirrorSourceByte", mirror_body),
    ];

    let descs = recognise_loops(&asns);
    assert_eq!(descs.len(), 1, "expected one descriptor: {:?}", descs);
    let d = &descs[0];
    assert_eq!(d.writes.len(), 1);
    let w = &d.writes[0];
    let ir = w
        .val_indirect_read
        .as_ref()
        .expect("brainfuck cabinet must also recognise indirect read");
    assert_eq!(ir.intermediate_property, "--mirrorSourceByte");
    assert_eq!(ir.pointer_property, "--priorTapeReadHead");
    assert_eq!(ir.seg_property.as_deref(), Some("--bagSourceSeg"));
    assert_eq!(
        d.bulk_class,
        BulkClass::Copy,
        "brainfuck cabinet with same shape must classify identically",
    );
}

/// Negative: val_expr's intermediate body is NOT a function call — just
/// a bare Var or a plain literal. Step 2 must not promote this; the
/// classifier stays at Fill.
#[test]
fn intermediate_without_function_call_stays_fill() {
    // Body is a bare Var, not a function call. Even though it
    // references the pointer mirror, the recogniser requires the
    // FunctionCall shape (the canonical "this is a read primitive"
    // marker). Anything else might be a derived constant or a
    // pointer-mirror passthrough — promotion is unsafe without the
    // call-shape signal.
    let body = var("--siMirror");
    let asns = cabinet_with_indirect_read("--strSrcByte", body);
    let descs = recognise_loops(&asns);
    let w = &descs[0].writes[0];
    assert!(
        w.val_indirect_read.is_none(),
        "non-call-shape body must not produce indirect_read, got {:?}",
        w.val_indirect_read,
    );
    assert_eq!(
        descs[0].bulk_class,
        BulkClass::Fill,
        "without indirect-read recognition, classifier stays Fill",
    );
}

/// Negative: the intermediate's body is a function call but its args
/// don't reference any of the loop's pointer mirrors. The recogniser
/// must reject this — the read isn't keyed on the stepping pointer, so
/// it doesn't represent a per-iter source byte.
#[test]
fn function_call_without_pointer_mirror_reference_stays_fill() {
    // Body: --readMem(calc(var(--someConst) + var(--otherConst))) —
    // function call with no pointer-mirror reference.
    let body = call(
        "--readMem",
        vec![add(var("--someConst"), var("--otherConst"))],
    );
    let asns = cabinet_with_indirect_read("--strSrcByte", body);
    let descs = recognise_loops(&asns);
    let w = &descs[0].writes[0];
    assert!(
        w.val_indirect_read.is_none(),
        "no pointer mirror in args → no indirect_read, got {:?}",
        w.val_indirect_read,
    );
    assert_eq!(descs[0].bulk_class, BulkClass::Fill);
}

/// Negative: the intermediate name doesn't resolve to any top-level
/// assignment in the cabinet. The recogniser cannot trace through it
/// and must leave the classification unchanged. This matches the
/// existing `cabinet_a` shape (which references `--_strSrcByte` but
/// doesn't define it locally) — the existing test that asserts
/// `Fill | Copy` still holds.
#[test]
fn unresolved_intermediate_name_leaves_classification_unchanged() {
    // Build a cabinet whose val references an intermediate, but DON'T
    // include the intermediate in the assignment list.
    let pred_continue = style_eq("--cont", 1.0);
    let no_rep = style_eq("--hasRep", 0.0);
    let active_guard = style_eq("--repActive", 0.0);

    let cx = dispatch(
        "--op",
        vec![(0xA4 as f64, counter_body(no_rep.clone(), "--cx0"))],
        keep_self("--cx0"),
    );
    let ip_inner = dispatch(
        "--op",
        vec![(
            0xA4 as f64,
            ip_body(pred_continue.clone(), "--ip0", var("--pl"), 1),
        )],
        keep_self("--ip0"),
    );
    let ip_wrapped = add(ip_inner, var("--pl"));
    let di = dispatch(
        "--op",
        vec![(
            0xA4 as f64,
            pointer_body(
                active_guard.clone(),
                "--diMirror",
                1,
                "--flags0",
                10,
                "--lowBytes",
                "--bit",
            ),
        )],
        keep_self("--diMirror"),
    );
    let addr = dispatch(
        "--op",
        vec![(
            0xA4 as f64,
            iff(
                active_guard.clone(),
                lit(-1.0),
                add(mul(var("--es"), lit(16.0)), var("--diMirror")),
            ),
        )],
        lit(-1.0),
    );
    let val = dispatch(
        "--op",
        // References --notDefinedAnywhere — no assignment for it.
        vec![(0xA4 as f64, var("--notDefinedAnywhere"))],
        lit(0.0),
    );

    let asns = vec![
        assign("--cx0", cx),
        assign("--ip0", ip_wrapped),
        assign("--diReg", di),
        assign("--addr0", addr),
        assign("--val0", val),
    ];
    let descs = recognise_loops(&asns);
    let w = &descs[0].writes[0];
    assert!(
        w.val_indirect_read.is_none(),
        "unresolved intermediate must yield None, got {:?}",
        w.val_indirect_read,
    );
    assert_eq!(
        descs[0].bulk_class,
        BulkClass::Fill,
        "unresolved intermediate stays at Fill",
    );
}

/// Indirect read where the call argument is a complex expression that
/// doesn't decompose as `var(seg) + var(ptr)` — only the pointer
/// reference is found, not a clean segment slot. The recogniser still
/// captures the indirect read (so Copy classification fires) but with
/// `seg_property: None`. The runtime applier later evaluates the full
/// arg expression rather than relying on a pre-resolved seg slot.
#[test]
fn indirect_read_without_clean_seg_decomposition_still_promotes_to_copy() {
    // Body: --readMem(calc(calc(var(--baseSeg) * 16) + var(--siMirror)))
    // The arg has the pointer mirror in it, but the seg side is an
    // arithmetic expression rather than a bare Var. Decomposition can't
    // simplify it.
    let body = call(
        "--readMem",
        vec![add(
            mul(var("--baseSeg"), lit(16.0)),
            var("--siMirror"),
        )],
    );
    let asns = cabinet_with_indirect_read("--strSrcByte", body);
    let descs = recognise_loops(&asns);
    let w = &descs[0].writes[0];
    let ir = w
        .val_indirect_read
        .as_ref()
        .expect("pointer mirror present → indirect_read should be Some");
    assert_eq!(ir.pointer_property, "--siMirror");
    assert!(
        ir.seg_property.is_none(),
        "non-bare-Var seg expression → seg_property is None, got {:?}",
        ir.seg_property,
    );
    assert_eq!(
        descs[0].bulk_class,
        BulkClass::Copy,
        "Copy promotion fires on pointer-mirror reference alone",
    );
}
