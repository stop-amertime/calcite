//! Top-level stylesheet parsing.
//!
//! Iterates through CSS rules using `cssparser::StyleSheetParser`, dispatching to
//! specialised parsers for `@property`, `@function`, and style rules.

use cssparser::{
    AtRuleParser, CowRcStr, ParseError, Parser, ParserState, QualifiedRuleParser, StyleSheetParser,
};

use crate::error::Result;
use crate::types::*;

use super::css_functions::parse_expr_list;
use super::property::parse_property_body;

/// A top-level CSS rule we care about.
#[derive(Debug)]
pub enum CssRule {
    /// `@property` declaration.
    Property(PropertyDef),
    /// `@function` definition.
    Function(FunctionDef),
    /// Style rule (`.cpu { ... }`) containing property assignments and
    /// nested input-edge rules.
    Style(StyleBlock),
}

/// Our rule parser, passed to `StyleSheetParser`.
pub struct CalciteRuleParser;

/// Prelude for at-rules (saved between parse_prelude and parse_block).
pub enum AtRulePrelude {
    /// `@property --name` prelude.
    Property(String),
    /// `@function --name(params)` prelude.
    Function(String, Vec<FunctionParam>),
    /// An at-rule we don't care about — skip it.
    Unknown,
}

impl<'i> AtRuleParser<'i> for CalciteRuleParser {
    type Prelude = AtRulePrelude;
    type AtRule = CssRule;
    type Error = ();

    fn parse_prelude<'t>(
        &mut self,
        name: CowRcStr<'i>,
        input: &mut Parser<'i, 't>,
    ) -> std::result::Result<Self::Prelude, ParseError<'i, ()>> {
        match &*name {
            "property" => {
                let prop_name = input
                    .expect_ident_cloned()
                    .map_err(|_| input.new_custom_error(()))?;
                Ok(AtRulePrelude::Property(prop_name.to_string()))
            }
            "function" => {
                let (func_name, params) = parse_function_prelude(input)?;
                Ok(AtRulePrelude::Function(func_name, params))
            }
            // Skip @keyframes, @container, @media, etc. — we just consume them.
            _ => Ok(AtRulePrelude::Unknown),
        }
    }

    fn parse_block<'t>(
        &mut self,
        prelude: Self::Prelude,
        _start: &ParserState,
        input: &mut Parser<'i, 't>,
    ) -> std::result::Result<Self::AtRule, ParseError<'i, ()>> {
        match prelude {
            AtRulePrelude::Property(name) => {
                let prop =
                    parse_property_body(&name, input).map_err(|_| input.new_custom_error(()))?;
                Ok(CssRule::Property(prop))
            }
            AtRulePrelude::Function(name, params) => {
                let func = parse_function_body(&name, params, input)
                    .map_err(|_| input.new_custom_error(()))?;
                Ok(CssRule::Function(func))
            }
            AtRulePrelude::Unknown => {
                // Consume and discard the block contents
                while input.next().is_ok() {}
                // Return a dummy rule that we'll filter out
                Ok(CssRule::Style(StyleBlock::default()))
            }
        }
    }

    fn rule_without_block(
        &mut self,
        prelude: Self::Prelude,
        _start: &ParserState,
    ) -> std::result::Result<Self::AtRule, ()> {
        match prelude {
            AtRulePrelude::Unknown => Ok(CssRule::Style(StyleBlock::default())),
            _ => Err(()),
        }
    }
}

impl<'i> QualifiedRuleParser<'i> for CalciteRuleParser {
    type Prelude = ();
    type QualifiedRule = CssRule;
    type Error = ();

    fn parse_prelude<'t>(
        &mut self,
        input: &mut Parser<'i, 't>,
    ) -> std::result::Result<Self::Prelude, ParseError<'i, ()>> {
        // Consume the selector — we don't parse it, just skip to the block.
        // We treat all style rules as potentially containing property assignments.
        while input.next().is_ok() {}
        Ok(())
    }

    fn parse_block<'t>(
        &mut self,
        _prelude: Self::Prelude,
        _start: &ParserState,
        input: &mut Parser<'i, 't>,
    ) -> std::result::Result<Self::QualifiedRule, ParseError<'i, ()>> {
        let mut input_edges = Vec::new();
        let assignments = parse_declarations(input, &mut input_edges);
        Ok(CssRule::Style(StyleBlock {
            assignments,
            input_edges,
        }))
    }
}

/// Returned for a `Style` rule: the flat assignments plus any input
/// edges recognised inside nested `&:has(#ID:pseudo) { ... }` blocks.
#[derive(Debug, Default)]
pub struct StyleBlock {
    pub assignments: Vec<Assignment>,
    pub input_edges: Vec<InputEdge>,
}

/// Parse the `@function` prelude: name, parameters, and optional return type.
///
/// Format: `--funcName(--param1 <type>, --param2 <type>) returns <type>`
fn parse_function_prelude<'i, 't>(
    input: &mut Parser<'i, 't>,
) -> std::result::Result<(String, Vec<FunctionParam>), ParseError<'i, ()>> {
    let state = input.state();
    match input.next().cloned() {
        Ok(cssparser::Token::Function(name)) => {
            // Parse parameters inside the parens
            let params = input
                .parse_nested_block(|inner| {
                    let mut params = Vec::new();
                    while !inner.is_exhausted() {
                        // Each param: --name <type>
                        if let Ok(param_name) = inner.expect_ident_cloned() {
                            let param_name = param_name.to_string();
                            // Try to read the type (e.g., `<integer>`)
                            let syntax = inner
                                .try_parse(|i| {
                                    i.expect_delim('<').map_err(|_| ())?;
                                    let type_name = i.expect_ident_cloned().map_err(|_| ())?;
                                    i.expect_delim('>').map_err(|_| ())?;
                                    Ok::<_, ()>(match &*type_name {
                                        "integer" => PropertySyntax::Integer,
                                        "number" => PropertySyntax::Number,
                                        "length" => PropertySyntax::Length,
                                        _ => PropertySyntax::Custom(type_name.to_string()),
                                    })
                                })
                                .unwrap_or(PropertySyntax::Any);
                            params.push(FunctionParam {
                                name: param_name,
                                syntax,
                            });
                        }
                        // Skip comma separators
                        let _ = inner.try_parse(|i| i.expect_comma());
                    }
                    Ok::<_, ParseError<'_, ()>>(params)
                })
                .unwrap_or_default();

            // Consume optional `returns <type>` clause
            let _ = input.try_parse(|i| {
                i.expect_ident_matching("returns").map_err(|_| ())?;
                while i.next().is_ok() {}
                Ok::<_, ()>(())
            });

            Ok((name.to_string(), params))
        }
        Ok(cssparser::Token::Ident(name)) => Ok((name.to_string(), Vec::new())),
        _ => {
            input.reset(&state);
            Err(input.new_custom_error(()))
        }
    }
}

/// Parse the body of an `@function` rule.
///
/// ```css
/// @function --readMem(--at <integer>) returns <integer> {
///   --local1: calc(var(--at) + 1);
///   result: if(style(--at: 0): var(--m0); ...);
/// }
/// ```
///
/// We parse local variable definitions (`--name: expr;`) and the `result` descriptor.
fn parse_function_body<'i, 't>(
    name: &str,
    parameters: Vec<FunctionParam>,
    input: &mut Parser<'i, 't>,
) -> Result<FunctionDef> {
    let mut locals = Vec::new();
    let mut result_expr = None;

    while !input.is_exhausted() {
        let state = input.state();
        let ident = match input.expect_ident_cloned() {
            Ok(id) => id.to_string(),
            Err(_) => {
                input.reset(&state);
                // Try to skip this token
                if input.next().is_err() {
                    break;
                }
                continue;
            }
        };

        if input.expect_colon().is_err() {
            // Not a declaration, skip
            let _ = input.try_parse(|i| i.expect_semicolon());
            continue;
        }

        if ident == "result" {
            match parse_expr_list(input) {
                Ok(expr) => result_expr = Some(expr),
                Err(e) => {
                    log::warn!("failed to parse result in @function {name}: {e}");
                }
            }
        } else if ident.starts_with("--") {
            match parse_expr_list(input) {
                Ok(expr) => {
                    locals.push(LocalVarDef {
                        name: ident,
                        value: expr,
                    });
                }
                Err(e) => {
                    log::warn!("failed to parse local {ident} in @function {name}: {e}");
                }
            }
        }

        let _ = input.try_parse(|i| i.expect_semicolon());
    }

    // If no explicit result was found, use a literal 0 as placeholder.
    let result = result_expr.unwrap_or(Expr::Literal(0.0));

    Ok(FunctionDef {
        name: name.to_string(),
        parameters,
        locals,
        result,
    })
}

/// Parse declarations (property assignments) from a rule body.
///
/// Extracts `--name: <expr>` declarations that represent computational state,
/// and recognises nested `&:has(#ID:pseudo) { --PROP: <expr>; }` rules whose
/// gated assignments become `InputEdge` entries.
fn parse_declarations<'i, 't>(
    input: &mut Parser<'i, 't>,
    input_edges: &mut Vec<InputEdge>,
) -> Vec<Assignment> {
    let mut assignments = Vec::new();

    while !input.is_exhausted() {
        let state = input.state();

        // Probe the next token. We accept three shapes here:
        //   `--ident: expr;`        normal declaration
        //   `&:has(#ID:pseudo) {…}` nested input-edge rule
        //   anything else           skip
        let next = match input.next().cloned() {
            Ok(t) => t,
            Err(_) => break,
        };

        if matches!(&next, cssparser::Token::Delim('&')) {
            // Try to parse a nested input-edge rule. If it doesn't match
            // the shape exactly, fall through and skip the rest of this
            // construct.
            if try_parse_input_edge_rule(input, input_edges).is_err() {
                // Skip until we find a `}` closing the (failed) nested
                // block, or end of input.
                skip_to_block_end(input);
            }
            continue;
        }

        let name = match next {
            cssparser::Token::Ident(name) => name.to_string(),
            _ => {
                // Unrecognised token — keep scanning.
                continue;
            }
        };

        // Re-position so cssparser sees the colon next.
        // (We've already consumed the ident; just expect colon.)
        if input.expect_colon().is_err() {
            // Not a declaration
            let _ = input.try_parse(|i| i.expect_semicolon());
            continue;
        }

        // Only capture custom property declarations (--name)
        if name.starts_with("--") {
            match parse_expr_list(input) {
                Ok(expr) => {
                    assignments.push(Assignment {
                        property: name,
                        value: expr,
                    });
                }
                Err(e) => {
                    log::debug!("skipping declaration {name}: {e}");
                }
            }
        } else {
            // Non-custom property — skip the value
            while input.try_parse(|i| i.expect_semicolon()).is_err() {
                if input.next().is_err() {
                    break;
                }
            }
            continue;
        }

        let _ = input.try_parse(|i| i.expect_semicolon());

        // Suppress unused-variable warning on `state` for now; reserved
        // for future error-recovery resets.
        let _ = state;
    }

    assignments
}

/// After consuming the leading `&`, try to parse the rest of a
/// `:has(#IDENT:IDENT) { --PROP: <expr>; }` rule. Pushes one InputEdge
/// per gated assignment found inside the block. Returns Err if the
/// shape doesn't match — caller is responsible for skipping the failed
/// construct.
fn try_parse_input_edge_rule<'i, 't>(
    input: &mut Parser<'i, 't>,
    input_edges: &mut Vec<InputEdge>,
) -> std::result::Result<(), ParseError<'i, ()>> {
    // After `&`, expect `:has(`
    input.expect_colon()?;
    input.expect_function_matching("has")?;

    let (selector, pseudo) = input.parse_nested_block(|inner| {
        // Inside has(): expect `#IDENT : IDENT`
        let tok = inner.next().cloned().map_err(|_| inner.new_custom_error::<(), ()>(()))?;
        let id = match tok {
            cssparser::Token::IDHash(s) => s.to_string(),
            cssparser::Token::Hash(s) => s.to_string(),
            _ => return Err(inner.new_custom_error::<(), ()>(())),
        };
        inner.expect_colon()?;
        let pseudo = inner.expect_ident_cloned()?.to_string();
        // Allow trailing tokens but don't require any specific shape.
        while inner.next().is_ok() {}
        Ok::<_, ParseError<'_, ()>>((id, pseudo))
    })?;

    // Then `{ --PROP: <expr>; … }`. We must consume the curly-block
    // token before parse_nested_block can descend into it.
    let next_tok = match input.next() {
        Ok(t) => t.clone(),
        Err(_) => return Err(input.new_custom_error(())),
    };
    if !matches!(next_tok, cssparser::Token::CurlyBracketBlock) {
        return Err(input.new_custom_error(()));
    }
    input.parse_nested_block(|inner| {
        while !inner.is_exhausted() {
            let prop = match inner.expect_ident_cloned() {
                Ok(n) => n.to_string(),
                Err(_) => {
                    // Skip stray tokens
                    let _ = inner.next();
                    continue;
                }
            };
            if inner.expect_colon().is_err() {
                let _ = inner.try_parse(|i| i.expect_semicolon());
                continue;
            }
            if !prop.starts_with("--") {
                while inner.try_parse(|i| i.expect_semicolon()).is_err() {
                    if inner.next().is_err() {
                        break;
                    }
                }
                continue;
            }
            match parse_expr_list(inner) {
                Ok(value) => {
                    input_edges.push(InputEdge {
                        property: prop,
                        pseudo: pseudo.clone(),
                        selector: selector.clone(),
                        value,
                    });
                }
                Err(e) => {
                    log::debug!("skipping gated declaration {prop}: {e}");
                }
            }
            let _ = inner.try_parse(|i| i.expect_semicolon());
        }
        Ok::<_, ParseError<'_, ()>>(())
    })?;

    Ok(())
}

/// After a parse failure inside a nested-rule attempt, consume tokens
/// until we balance back out of the current curly block (or hit EOF).
fn skip_to_block_end<'i, 't>(input: &mut Parser<'i, 't>) {
    // Scan forward looking for the `{` … `}` that the failed nested rule
    // would have produced. If we hit a curly block, descend into it and
    // discard. If we just see junk, drop tokens until we hit `;` or EOF.
    while let Ok(tok) = input.next().cloned() {
        match tok {
            cssparser::Token::CurlyBracketBlock => {
                let _ = input.parse_nested_block(|inner| {
                    while inner.next().is_ok() {}
                    Ok::<_, ParseError<'_, ()>>(())
                });
                return;
            }
            cssparser::Token::Semicolon => return,
            _ => {}
        }
    }
}

/// Parse a full CSS stylesheet into a `ParsedProgram`.
pub fn parse_stylesheet(css: &str) -> Result<ParsedProgram> {
    let total_bytes = css.len();
    let show_progress = std::env::var_os("CALCITE_NO_PROGRESS").is_none()
        && total_bytes >= 1_000_000;
    let progress_start = web_time::Instant::now();
    let mut last_render = web_time::Instant::now();

    // ------------------------------------------------------------------
    // Fast-path pre-scan.
    //
    // On large programs (≥ 1 MB) we scan the raw bytes for dense,
    // byte-templated regions (repeated `--mN:` assignments and
    // `@property --mN` blocks). Anything we can absorb is emitted as
    // pre-built `PropertyDef`/`BroadcastWrite` entries, and the source
    // bytes are blanked with spaces/newlines so cssparser still sees an
    // input of the same length (preserving error-message line numbers)
    // but skips over the dense region as pure whitespace.
    //
    // Falls back cleanly: if nothing matches, we pass the original `css`
    // through to cssparser unchanged.
    // ------------------------------------------------------------------
    let t_fast = web_time::Instant::now();
    let fast = if total_bytes >= 1_000_000 {
        super::fast_path::recognise(css)
    } else {
        // Empty result — skip the scan overhead on small inputs.
        super::fast_path::FastPathResult::empty_pub()
    };
    let fast_elapsed = t_fast.elapsed().as_secs_f64();
    let cssparser_input: std::borrow::Cow<str> = if fast.blank_ranges.is_empty() {
        std::borrow::Cow::Borrowed(css)
    } else {
        std::borrow::Cow::Owned(super::fast_path::apply_blank_ranges(css, &fast.blank_ranges))
    };
    if !fast.blank_ranges.is_empty() {
        let blanked: usize = fast.blank_ranges.iter().map(|&(s, e)| e - s).sum();
        log::info!(
            "[parse fast-path] recognised {} properties + {} broadcast writes + {} packed ports, blanked {:.1} MB ({:.1}% of input) in {:.2}s",
            fast.properties.len(),
            fast.broadcast_writes.len(),
            fast.packed_broadcast_ports.len(),
            blanked as f64 / 1_048_576.0,
            100.0 * blanked as f64 / total_bytes as f64,
            fast_elapsed,
        );
    }

    let css_for_parser: &str = &cssparser_input;
    let mut input = cssparser::ParserInput::new(css_for_parser);
    let mut parser = Parser::new(&mut input);
    let mut rule_parser = CalciteRuleParser;

    let mut properties = fast.properties;
    let mut functions = Vec::new();
    let mut assignments = Vec::new();
    let mut input_edges: Vec<InputEdge> = Vec::new();

    let mut sheet = StyleSheetParser::new(&mut parser, &mut rule_parser);
    while let Some(result) = sheet.next() {
        match result {
            Ok(CssRule::Property(prop)) => {
                properties.push(prop);
            }
            Ok(CssRule::Function(func)) => {
                functions.push(func);
            }
            Ok(CssRule::Style(block)) => {
                assignments.extend(block.assignments);
                input_edges.extend(block.input_edges);
            }
            Err((err, _slice)) => {
                log::debug!("skipping unparseable rule: {err:?}");
            }
        }
        if show_progress && last_render.elapsed().as_millis() >= 100 {
            let pos = sheet.input.position().byte_index();
            crate::compile::render_progress(
                "Parsing",
                pos,
                total_bytes,
                progress_start.elapsed().as_secs_f64(),
            );
            last_render = web_time::Instant::now();
        }
    }
    if show_progress {
        crate::compile::render_progress(
            "Parsing",
            total_bytes,
            total_bytes,
            progress_start.elapsed().as_secs_f64(),
        );
        eprintln!();
    }

    Ok(ParsedProgram {
        properties,
        functions,
        assignments,
        prebuilt_broadcast_writes: fast.broadcast_writes,
        prebuilt_packed_broadcast_ports: fast.packed_broadcast_ports,
        fast_path_absorbed: fast.absorbed_properties,
        input_edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_edges_recognised_from_nested_rules() {
        let css = r#"
            .cpu {
              &:has(#kb-1:active) { --keyboard: 561; }
              &:has(#kb-a:active) { --keyboard: 7777; }
              --foo: 7;
            }
        "#;
        let prog = parse_stylesheet(css).unwrap();

        // Flat declaration survives.
        assert_eq!(prog.assignments.len(), 1);
        assert_eq!(prog.assignments[0].property, "--foo");

        // Two input edges recognised.
        assert_eq!(prog.input_edges.len(), 2);
        let e0 = &prog.input_edges[0];
        assert_eq!(e0.property, "--keyboard");
        assert_eq!(e0.pseudo, "active");
        assert_eq!(e0.selector, "kb-1");
        assert_eq!(e0.value, Expr::Literal(561.0));

        let e1 = &prog.input_edges[1];
        assert_eq!(e1.selector, "kb-a");
        assert_eq!(e1.value, Expr::Literal(7777.0));
    }

    #[test]
    fn input_edges_handle_complex_value_expression() {
        let css = ".cpu { &:has(#x:active) { --y: calc(1 + 2); } }";
        let prog = parse_stylesheet(css).unwrap();
        assert_eq!(prog.input_edges.len(), 1);
        assert_eq!(prog.input_edges[0].selector, "x");
        // Just verify it parsed something non-trivial.
        assert!(!matches!(prog.input_edges[0].value, Expr::Literal(_)));
    }

    #[test]
    fn nonsense_nested_rules_are_skipped_without_breaking_following_decls() {
        let css = r#"
            .cpu {
              &:nth-child(2) { --bogus: 99; }
              --good: 42;
            }
        "#;
        let prog = parse_stylesheet(css).unwrap();
        assert_eq!(prog.input_edges.len(), 0);
        assert_eq!(prog.assignments.len(), 1);
        assert_eq!(prog.assignments[0].property, "--good");
    }
}
