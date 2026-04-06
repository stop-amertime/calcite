//! Parsing for CSS math functions: calc(), mod(), round(), min(), max(), etc.
//!
//! These are all parsed into `Expr::Calc(CalcOp::...)` nodes.
//! Also handles `var()`, `if()`, literal numbers, and `@function` calls.

use cssparser::{Parser, Token};

use crate::error::Result;
use crate::types::*;

type CssParseError<'i> = cssparser::ParseError<'i, crate::CalcifyError>;

/// Helper: run parse_nested_block with CalcifyError as the custom error.
fn nested_block<'i, 't, F, T>(input: &mut Parser<'i, 't>, f: F) -> Result<T>
where
    F: for<'tt> FnOnce(&mut Parser<'i, 'tt>) -> std::result::Result<T, CssParseError<'i>>,
{
    input.parse_nested_block(f).map_err(css_err)
}

/// Convert cssparser::ParseError<CalcifyError> into CalcifyError.
fn css_err(e: CssParseError<'_>) -> crate::CalcifyError {
    match e.kind {
        cssparser::ParseErrorKind::Custom(e) => e,
        other => crate::CalcifyError::Parse(format!("{other:?}")),
    }
}

/// Convert a BasicParseError to CalcifyError (for use inside closures).
fn basic_err(e: cssparser::BasicParseError<'_>) -> crate::CalcifyError {
    crate::CalcifyError::Parse(format!("{e:?}"))
}

/// Wrap a CalcifyError into a CssParseError for use inside parse_nested_block closures.
fn wrap_err<'i>(e: crate::CalcifyError, loc: cssparser::SourceLocation) -> CssParseError<'i> {
    cssparser::ParseError {
        kind: cssparser::ParseErrorKind::Custom(e),
        location: loc,
    }
}

/// Parse a CSS value expression from a cssparser `Parser`.
///
/// Recursive descent with precedence: `+`/`-` lowest, `*`/`/` higher.
pub fn parse_expr<'i, 't>(input: &mut Parser<'i, 't>) -> Result<Expr> {
    let left = parse_product(input)?;
    parse_additive(input, left)
}

/// Same as parse_expr but returns CssParseError — for use inside parse_nested_block.
fn parse_expr_css<'i, 't>(
    input: &mut Parser<'i, 't>,
) -> std::result::Result<Expr, CssParseError<'i>> {
    parse_expr(input).map_err(|e| wrap_err(e, input.current_source_location()))
}

/// Parse additive operators (`+`, `-`) at the lowest precedence level.
fn parse_additive<'i, 't>(input: &mut Parser<'i, 't>, mut left: Expr) -> Result<Expr> {
    loop {
        let state = input.state();
        // In CSS calc(), `+` and `-` must be whitespace-separated.
        // We check for whitespace, then the operator.
        match input.next_including_whitespace() {
            Ok(&Token::WhiteSpace(_)) => {}
            _ => {
                input.reset(&state);
                return Ok(left);
            }
        }

        match input.next() {
            Ok(&Token::Delim('+')) => {
                let right = parse_product(input)?;
                left = Expr::Calc(CalcOp::Add(Box::new(left), Box::new(right)));
            }
            Ok(&Token::Delim('-')) => {
                let right = parse_product(input)?;
                left = Expr::Calc(CalcOp::Sub(Box::new(left), Box::new(right)));
            }
            _ => {
                input.reset(&state);
                return Ok(left);
            }
        }
    }
}

/// Parse a single term, then multiplicative operators (`*`, `/`).
fn parse_product<'i, 't>(input: &mut Parser<'i, 't>) -> Result<Expr> {
    let left = parse_unary(input)?;
    parse_multiplicative(input, left)
}

/// Parse multiplicative operators (`*`, `/`) at higher precedence.
fn parse_multiplicative<'i, 't>(input: &mut Parser<'i, 't>, mut left: Expr) -> Result<Expr> {
    loop {
        let state = input.state();
        match input.next() {
            Ok(&Token::Delim('*')) => {
                let right = parse_unary(input)?;
                left = Expr::Calc(CalcOp::Mul(Box::new(left), Box::new(right)));
            }
            Ok(&Token::Delim('/')) => {
                let right = parse_unary(input)?;
                left = Expr::Calc(CalcOp::Div(Box::new(left), Box::new(right)));
            }
            _ => {
                input.reset(&state);
                return Ok(left);
            }
        }
    }
}

/// Parse a unary expression (optional negation, then an atom).
fn parse_unary<'i, 't>(input: &mut Parser<'i, 't>) -> Result<Expr> {
    let state = input.state();
    match input.next() {
        Ok(&Token::Delim('-')) => {
            let inner = parse_atom(input)?;
            Ok(Expr::Calc(CalcOp::Negate(Box::new(inner))))
        }
        Ok(&Token::Delim('+')) => parse_atom(input),
        _ => {
            input.reset(&state);
            parse_atom(input)
        }
    }
}

/// Parse an atomic expression: number, var(), calc(), if(), function call, or parenthesised expr.
fn parse_atom<'i, 't>(input: &mut Parser<'i, 't>) -> Result<Expr> {
    let state = input.state();
    let token = input
        .next()
        .map_err(|e| crate::CalcifyError::Parse(format!("unexpected end of input: {e:?}")))?
        .clone();

    match token {
        Token::Number { value, .. } => Ok(Expr::Literal(value as f64)),

        Token::Function(ref name) => {
            let name = name.to_string();
            parse_function_call(input, &name)
        }

        Token::ParenthesisBlock => nested_block(input, parse_expr_css),

        // String literals (used by display functions like --i2char, --getInstStr)
        Token::QuotedString(ref s) => Ok(Expr::StringLiteral(s.to_string())),

        // A bare custom property ident — treat as var reference.
        Token::Ident(ref name) if name.starts_with("--") => Ok(Expr::Var {
            name: name.to_string(),
            fallback: None,
        }),

        other => {
            input.reset(&state);
            Err(crate::CalcifyError::Parse(format!(
                "unexpected token in expression: {other:?}"
            )))
        }
    }
}

/// Parse a function call after the `Function` token has been consumed.
fn parse_function_call<'i, 't>(input: &mut Parser<'i, 't>, name: &str) -> Result<Expr> {
    nested_block(input, |inner| {
        let result = match name {
            "var" => parse_var(inner),
            "calc" => parse_expr(inner),
            "if" => parse_if(inner),
            "mod" => parse_mod(inner),
            "round" => parse_round(inner),
            "min" => parse_min_max(inner, true),
            "max" => parse_min_max(inner, false),
            "clamp" => parse_clamp(inner),
            "pow" => parse_two_arg(inner, CalcOp::Pow),
            "sign" => parse_expr(inner).map(|arg| Expr::Calc(CalcOp::Sign(Box::new(arg)))),
            "abs" => parse_expr(inner).map(|arg| Expr::Calc(CalcOp::Abs(Box::new(arg)))),
            _ if name.starts_with("--") => parse_custom_function_call(inner, name),
            _ => Err(crate::CalcifyError::Parse(format!(
                "unknown function: {name}"
            ))),
        };
        result.map_err(|e| wrap_err(e, inner.current_source_location()))
    })
}

/// Parse `var(--name)` or `var(--name, fallback)`.
fn parse_var<'i, 't>(input: &mut Parser<'i, 't>) -> Result<Expr> {
    let name = input.expect_ident().map_err(basic_err)?.to_string();

    if !name.starts_with("--") {
        return Err(crate::CalcifyError::Parse(format!(
            "var() argument must be a custom property (got {name})"
        )));
    }

    let fallback = if input.try_parse(|i| i.expect_comma()).is_ok() {
        Some(Box::new(parse_expr(input)?))
    } else {
        None
    };

    Ok(Expr::Var { name, fallback })
}

/// Parse `if(style(--prop: val): then; style(--a:1) and style(--b:2): then; ...; else: fallback)`.
fn parse_if<'i, 't>(input: &mut Parser<'i, 't>) -> Result<Expr> {
    let mut branches = Vec::new();

    loop {
        let state = input.state();

        // Check for `else:` — ends the branch list
        if input.try_parse(|i| i.expect_ident_matching("else")).is_ok() {
            input.expect_colon().map_err(basic_err)?;
            let fallback = parse_expr(input)?;
            return Ok(Expr::StyleCondition {
                branches,
                fallback: Box::new(fallback),
            });
        }

        // Try to parse a condition (possibly compound with and/or) then `: then_expr`
        match parse_style_condition(input) {
            Ok(condition) => {
                input.expect_colon().map_err(basic_err)?;
                let then = parse_expr(input)?;
                branches.push(StyleBranch { condition, then });
                let _ = input.try_parse(|i| i.expect_semicolon());
            }
            Err(_) => {
                input.reset(&state);
                if branches.is_empty() {
                    return Err(crate::CalcifyError::Parse(
                        "expected style() or else in if()".to_string(),
                    ));
                }
                // Try to parse an implicit fallback expression; if exhausted, use 0
                let fallback = if input.is_exhausted() {
                    Expr::Literal(0.0)
                } else {
                    parse_expr(input).unwrap_or(Expr::Literal(0.0))
                };
                return Ok(Expr::StyleCondition {
                    branches,
                    fallback: Box::new(fallback),
                });
            }
        }
    }
}

/// Parse a style condition: `style(--prop: val)` possibly followed by `and`/`or` chains.
fn parse_style_condition<'i, 't>(input: &mut Parser<'i, 't>) -> Result<StyleTest> {
    let first = parse_single_style_test(input)?;

    // Check for `and`/`or` chaining
    let state = input.state();
    match input.next() {
        Ok(Token::Ident(ref kw)) if &**kw == "and" => {
            let mut tests = vec![first];
            tests.push(parse_single_style_test(input)?);
            // Continue consuming `and style(...)` pairs
            loop {
                let s = input.state();
                match input.next() {
                    Ok(Token::Ident(ref kw)) if &**kw == "and" => {
                        tests.push(parse_single_style_test(input)?);
                    }
                    _ => {
                        input.reset(&s);
                        break;
                    }
                }
            }
            Ok(StyleTest::And(tests))
        }
        Ok(Token::Ident(ref kw)) if &**kw == "or" => {
            let mut tests = vec![first];
            tests.push(parse_single_style_test(input)?);
            loop {
                let s = input.state();
                match input.next() {
                    Ok(Token::Ident(ref kw)) if &**kw == "or" => {
                        tests.push(parse_single_style_test(input)?);
                    }
                    _ => {
                        input.reset(&s);
                        break;
                    }
                }
            }
            Ok(StyleTest::Or(tests))
        }
        _ => {
            input.reset(&state);
            Ok(first)
        }
    }
}

/// Parse a single `style(--prop: val)` test.
fn parse_single_style_test<'i, 't>(input: &mut Parser<'i, 't>) -> Result<StyleTest> {
    // Expect `style` function token
    let state = input.state();
    match input.next().cloned() {
        Ok(Token::Function(ref name)) if &**name == "style" => nested_block(input, |inner| {
            let prop = inner
                .expect_ident_cloned()
                .map_err(|e| wrap_err(basic_err(e), inner.current_source_location()))?;
            inner
                .expect_colon()
                .map_err(|e| wrap_err(basic_err(e), inner.current_source_location()))?;
            let val = parse_expr_css(inner)?;
            Ok(StyleTest::Single {
                property: prop.to_string(),
                value: val,
            })
        }),
        _ => {
            input.reset(&state);
            Err(crate::CalcifyError::Parse(
                "expected style() in condition".to_string(),
            ))
        }
    }
}

/// Parse `mod(a, b)`.
fn parse_mod<'i, 't>(input: &mut Parser<'i, 't>) -> Result<Expr> {
    let a = parse_expr(input)?;
    input.expect_comma().map_err(basic_err)?;
    let b = parse_expr(input)?;
    Ok(Expr::Calc(CalcOp::Mod(Box::new(a), Box::new(b))))
}

/// Parse `round(strategy, value, interval)` or `round(value, interval)`.
fn parse_round<'i, 't>(input: &mut Parser<'i, 't>) -> Result<Expr> {
    let strategy = input
        .try_parse(|i| {
            let ident = i.expect_ident_cloned().map_err(|_| ())?;
            let strat = match &*ident {
                "nearest" => RoundStrategy::Nearest,
                "up" => RoundStrategy::Up,
                "down" => RoundStrategy::Down,
                "to-zero" => RoundStrategy::ToZero,
                _ => return Err(()),
            };
            i.expect_comma().map_err(|_| ())?;
            Ok(strat)
        })
        .unwrap_or(RoundStrategy::Nearest);

    let value = parse_expr(input)?;
    // Interval is optional — defaults to 1 if not provided
    let interval = if input.try_parse(|i| i.expect_comma()).is_ok() {
        parse_expr(input)?
    } else {
        Expr::Literal(1.0)
    };

    Ok(Expr::Calc(CalcOp::Round(
        strategy,
        Box::new(value),
        Box::new(interval),
    )))
}

/// Parse `min(a, b, ...)` or `max(a, b, ...)`.
fn parse_min_max<'i, 't>(input: &mut Parser<'i, 't>, is_min: bool) -> Result<Expr> {
    let mut args = vec![parse_expr(input)?];
    while input.try_parse(|i| i.expect_comma()).is_ok() {
        args.push(parse_expr(input)?);
    }
    Ok(Expr::Calc(if is_min {
        CalcOp::Min(args)
    } else {
        CalcOp::Max(args)
    }))
}

/// Parse `clamp(min, val, max)`.
fn parse_clamp<'i, 't>(input: &mut Parser<'i, 't>) -> Result<Expr> {
    let min = parse_expr(input)?;
    input.expect_comma().map_err(basic_err)?;
    let val = parse_expr(input)?;
    input.expect_comma().map_err(basic_err)?;
    let max = parse_expr(input)?;
    Ok(Expr::Calc(CalcOp::Clamp(
        Box::new(min),
        Box::new(val),
        Box::new(max),
    )))
}

/// Parse a two-argument math function like `pow(a, b)`.
fn parse_two_arg<'i, 't, F>(input: &mut Parser<'i, 't>, make_op: F) -> Result<Expr>
where
    F: FnOnce(Box<Expr>, Box<Expr>) -> CalcOp,
{
    let a = parse_expr(input)?;
    input.expect_comma().map_err(basic_err)?;
    let b = parse_expr(input)?;
    Ok(Expr::Calc(make_op(Box::new(a), Box::new(b))))
}

/// Parse a custom @function call: `--funcName(arg1, arg2, ...)`.
fn parse_custom_function_call<'i, 't>(input: &mut Parser<'i, 't>, name: &str) -> Result<Expr> {
    let mut args = Vec::new();
    if !input.is_exhausted() {
        args.push(parse_expr(input)?);
        while input.try_parse(|i| i.expect_comma()).is_ok() {
            args.push(parse_expr(input)?);
        }
    }
    Ok(Expr::FunctionCall {
        name: name.to_string(),
        args,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cssparser::ParserInput;

    fn parse(s: &str) -> Result<Expr> {
        let mut input = ParserInput::new(s);
        let mut parser = Parser::new(&mut input);
        parse_expr(&mut parser)
    }

    #[test]
    fn literal_number() {
        let expr = parse("42").unwrap();
        assert!(matches!(expr, Expr::Literal(v) if (v - 42.0).abs() < f64::EPSILON));
    }

    #[test]
    fn negative_number() {
        let expr = parse("-42").unwrap();
        match expr {
            Expr::Literal(v) => assert!((v - -42.0).abs() < f64::EPSILON),
            Expr::Calc(CalcOp::Negate(inner)) => {
                assert!(matches!(*inner, Expr::Literal(v) if (v - 42.0).abs() < f64::EPSILON));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn var_simple() {
        let expr = parse("var(--foo)").unwrap();
        assert!(matches!(expr, Expr::Var { ref name, fallback: None } if name == "--foo"));
    }

    #[test]
    fn var_with_fallback() {
        let expr = parse("var(--foo, 0)").unwrap();
        match expr {
            Expr::Var {
                ref name,
                fallback: Some(_),
            } => assert_eq!(name, "--foo"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn calc_addition() {
        let expr = parse("calc(1 + 2)").unwrap();
        assert!(matches!(expr, Expr::Calc(CalcOp::Add(_, _))));
    }

    #[test]
    fn calc_multiplication() {
        let expr = parse("calc(3 * 4)").unwrap();
        assert!(matches!(expr, Expr::Calc(CalcOp::Mul(_, _))));
    }

    #[test]
    fn mod_function() {
        let expr = parse("mod(var(--x), 256)").unwrap();
        assert!(matches!(expr, Expr::Calc(CalcOp::Mod(_, _))));
    }

    #[test]
    fn round_with_strategy() {
        let expr = parse("round(down, var(--x), 1)").unwrap();
        match expr {
            Expr::Calc(CalcOp::Round(RoundStrategy::Down, _, _)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn min_function() {
        let expr = parse("min(var(--a), var(--b), 100)").unwrap();
        match expr {
            Expr::Calc(CalcOp::Min(args)) => assert_eq!(args.len(), 3),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn custom_function_call() {
        let expr = parse("--readMem(var(--at))").unwrap();
        match expr {
            Expr::FunctionCall { ref name, ref args } => {
                assert_eq!(name, "--readMem");
                assert_eq!(args.len(), 1);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn if_style_simple() {
        let expr = parse("if(style(--x: 1): 10; style(--x: 2): 20; else: 0)").unwrap();
        match expr {
            Expr::StyleCondition {
                ref branches,
                fallback: _,
            } => {
                assert_eq!(branches.len(), 2);
                match &branches[0].condition {
                    StyleTest::Single { property, .. } => assert_eq!(property, "--x"),
                    other => panic!("expected Single, got: {other:?}"),
                }
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn nested_calc() {
        let expr = parse("calc(var(--a) * 256 + var(--b))").unwrap();
        // Should parse as: (var(--a) * 256) + var(--b) due to precedence
        assert!(matches!(expr, Expr::Calc(CalcOp::Add(_, _))));
    }
}
