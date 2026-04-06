//! Top-level stylesheet parsing.
//!
//! Iterates through CSS rules using `cssparser::StyleSheetParser`, dispatching to
//! specialised parsers for `@property`, `@function`, and style rules.

use cssparser::{
    AtRuleParser, CowRcStr, ParseError, Parser, ParserState, QualifiedRuleParser, StyleSheetParser,
};

use crate::error::Result;
use crate::types::*;

use super::css_functions::parse_expr;
use super::property::parse_property_body;

/// A top-level CSS rule we care about.
#[derive(Debug)]
pub enum CssRule {
    /// `@property` declaration.
    Property(PropertyDef),
    /// `@function` definition.
    Function(FunctionDef),
    /// Style rule (`.cpu { ... }`) containing property assignments.
    Style(Vec<Assignment>),
}

/// Our rule parser, passed to `StyleSheetParser`.
pub struct CalcifyRuleParser;

/// Prelude for at-rules (saved between parse_prelude and parse_block).
pub enum AtRulePrelude {
    /// `@property --name` prelude.
    Property(String),
    /// `@function --name(params)` prelude.
    Function(String, Vec<FunctionParam>),
    /// An at-rule we don't care about — skip it.
    Unknown,
}

impl<'i> AtRuleParser<'i> for CalcifyRuleParser {
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
                Ok(CssRule::Style(Vec::new()))
            }
        }
    }

    fn rule_without_block(
        &mut self,
        prelude: Self::Prelude,
        _start: &ParserState,
    ) -> std::result::Result<Self::AtRule, ()> {
        match prelude {
            AtRulePrelude::Unknown => Ok(CssRule::Style(Vec::new())),
            _ => Err(()),
        }
    }
}

impl<'i> QualifiedRuleParser<'i> for CalcifyRuleParser {
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
        let assignments = parse_declarations(input);
        Ok(CssRule::Style(assignments))
    }
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
            match parse_expr(input) {
                Ok(expr) => result_expr = Some(expr),
                Err(e) => {
                    log::warn!("failed to parse result in @function {name}: {e}");
                }
            }
        } else if ident.starts_with("--") {
            match parse_expr(input) {
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
/// Extracts `--name: <expr>` declarations that represent computational state.
fn parse_declarations<'i, 't>(input: &mut Parser<'i, 't>) -> Vec<Assignment> {
    let mut assignments = Vec::new();

    while !input.is_exhausted() {
        let state = input.state();

        // Try to read an ident (property name)
        let name = match input.expect_ident_cloned() {
            Ok(name) => name.to_string(),
            Err(_) => {
                input.reset(&state);
                // Skip this token and try the next
                if input.next().is_err() {
                    break;
                }
                continue;
            }
        };

        // Expect colon
        if input.expect_colon().is_err() {
            // Not a declaration
            let _ = input.try_parse(|i| i.expect_semicolon());
            continue;
        }

        // Only capture custom property declarations (--name)
        if name.starts_with("--") {
            match parse_expr(input) {
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
    }

    assignments
}

/// Parse a full CSS stylesheet into a `ParsedProgram`.
pub fn parse_stylesheet(css: &str) -> Result<ParsedProgram> {
    let mut input = cssparser::ParserInput::new(css);
    let mut parser = Parser::new(&mut input);
    let mut rule_parser = CalcifyRuleParser;

    let mut properties = Vec::new();
    let mut functions = Vec::new();
    let mut assignments = Vec::new();

    let sheet = StyleSheetParser::new(&mut parser, &mut rule_parser);
    for result in sheet {
        match result {
            Ok(CssRule::Property(prop)) => {
                properties.push(prop);
            }
            Ok(CssRule::Function(func)) => {
                functions.push(func);
            }
            Ok(CssRule::Style(decls)) => {
                assignments.extend(decls);
            }
            Err((err, _slice)) => {
                log::debug!("skipping unparseable rule: {err:?}");
            }
        }
    }

    Ok(ParsedProgram {
        properties,
        functions,
        assignments,
    })
}
