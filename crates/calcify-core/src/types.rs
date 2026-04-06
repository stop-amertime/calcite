//! Core type definitions for the calc(ify) intermediate representation.
//!
//! These types represent the parsed and compiled forms of computational CSS constructs.

/// A raw CSS value (before expression parsing).
#[derive(Debug, Clone)]
pub enum CssValue {
    Integer(i64),
    Number(f64),
    String(String),
}

/// A parsed `@property` declaration.
#[derive(Debug, Clone)]
pub struct PropertyDef {
    pub name: String,
    pub syntax: PropertySyntax,
    pub inherits: bool,
    pub initial_value: Option<CssValue>,
}

/// The `syntax` descriptor of an `@property` rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropertySyntax {
    Integer,
    Number,
    Length,
    Custom(String),
    Any,
}

/// A parsed `@function` definition.
#[derive(Debug, Clone)]
pub struct FunctionDef {
    pub name: String,
    pub parameters: Vec<FunctionParam>,
    pub locals: Vec<LocalVarDef>,
    pub result: Expr,
}

/// A parameter of a `@function`.
#[derive(Debug, Clone)]
pub struct FunctionParam {
    pub name: String,
    pub syntax: PropertySyntax,
}

/// A local variable within a `@function`.
#[derive(Debug, Clone)]
pub struct LocalVarDef {
    pub name: String,
    pub value: Expr,
}

/// Expression tree — the core IR for CSS computational values.
#[derive(Debug, Clone)]
pub enum Expr {
    /// A literal numeric value.
    Literal(f64),

    /// A literal string value (for display functions like --i2char).
    StringLiteral(String),

    /// A `var(--name)` reference, with optional fallback.
    Var {
        name: String,
        fallback: Option<Box<Expr>>,
    },

    /// `calc()` and other math functions.
    Calc(CalcOp),

    /// `if(style(--prop: val): then; else: otherwise)` conditional.
    StyleCondition {
        branches: Vec<StyleBranch>,
        fallback: Box<Expr>,
    },

    /// A `@function` call: `--funcName(arg1, arg2)`.
    FunctionCall { name: String, args: Vec<Expr> },
}

/// A single branch in an `if(style(...): ...)` chain.
#[derive(Debug, Clone)]
pub struct StyleBranch {
    /// The condition — a single `style()` test or a compound `and`/`or`.
    pub condition: StyleTest,
    pub then: Expr,
}

/// A style condition test inside `if()`.
#[derive(Debug, Clone)]
pub enum StyleTest {
    /// A single `style(--prop: val)` test.
    Single { property: String, value: Expr },
    /// `condition1 and condition2` — all must match.
    And(Vec<StyleTest>),
    /// `condition1 or condition2` — any must match.
    Or(Vec<StyleTest>),
}

/// Arithmetic / math operations from `calc()`, `mod()`, `round()`, etc.
#[derive(Debug, Clone)]
pub enum CalcOp {
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Mod(Box<Expr>, Box<Expr>),
    Min(Vec<Expr>),
    Max(Vec<Expr>),
    Clamp(Box<Expr>, Box<Expr>, Box<Expr>),
    Round(RoundStrategy, Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Sign(Box<Expr>),
    Abs(Box<Expr>),
    Negate(Box<Expr>),
}

/// The rounding strategy for `round()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundStrategy {
    Nearest,
    Up,
    Down,
    ToZero,
}

/// A complete parsed CSS program ready for pattern compilation.
#[derive(Debug, Clone)]
pub struct ParsedProgram {
    pub properties: Vec<PropertyDef>,
    pub functions: Vec<FunctionDef>,
    /// Property assignments on `.cpu`, in declaration order.
    pub assignments: Vec<Assignment>,
}

/// A single property assignment: `--name: <expr>`.
#[derive(Debug, Clone)]
pub struct Assignment {
    pub property: String,
    pub value: Expr,
}

/// A compiled program after pattern recognition and optimisation.
#[derive(Debug)]
pub struct CompiledProgram {
    /// Opcode → instruction handler index.
    pub decode_table: std::collections::HashMap<u16, usize>,
    /// Compiled instruction handlers.
    pub instructions: Vec<CompiledInstruction>,
}

/// A single compiled instruction (post-pattern-recognition).
#[derive(Debug)]
pub struct CompiledInstruction {
    pub name: String,
    pub has_modrm: bool,
    // The actual execution logic will be filled in during Phase 2.
}
