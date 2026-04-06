//! Core type definitions for the calc(ify) intermediate representation.
//!
//! These types represent the parsed and compiled forms of computational CSS constructs.

use std::hash::{Hash, Hasher};

/// Hash an f64 by its bit representation. This treats -0.0 and +0.0 as different
/// values (matching f64::to_bits semantics), which is fine for CSS integer values.
fn hash_f64<H: Hasher>(v: &f64, state: &mut H) {
    v.to_bits().hash(state);
}

/// A raw CSS value (before expression parsing).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CssValue {
    Integer(i64),
    String(String),
}

/// A parsed `@property` declaration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PropertyDef {
    pub name: String,
    pub syntax: PropertySyntax,
    pub inherits: bool,
    pub initial_value: Option<CssValue>,
}

/// The `syntax` descriptor of an `@property` rule.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PropertySyntax {
    Integer,
    Number,
    Length,
    Custom(String),
    Any,
}

/// A parsed `@function` definition.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionDef {
    pub name: String,
    pub parameters: Vec<FunctionParam>,
    pub locals: Vec<LocalVarDef>,
    pub result: Expr,
}

/// A parameter of a `@function`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionParam {
    pub name: String,
    pub syntax: PropertySyntax,
}

/// A local variable within a `@function`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocalVarDef {
    pub name: String,
    pub value: Expr,
}

/// Expression tree — the core IR for CSS computational values.
///
/// Implements `Eq` and `Hash` via `f64::to_bits()` for the `Literal` variant.
/// This is sound for CSS integer values where NaN/signed-zero are not expected.
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

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Expr::Literal(a), Expr::Literal(b)) => a.to_bits() == b.to_bits(),
            (Expr::StringLiteral(a), Expr::StringLiteral(b)) => a == b,
            (
                Expr::Var {
                    name: n1,
                    fallback: f1,
                },
                Expr::Var {
                    name: n2,
                    fallback: f2,
                },
            ) => n1 == n2 && f1 == f2,
            (Expr::Calc(a), Expr::Calc(b)) => a == b,
            (
                Expr::StyleCondition {
                    branches: b1,
                    fallback: f1,
                },
                Expr::StyleCondition {
                    branches: b2,
                    fallback: f2,
                },
            ) => b1 == b2 && f1 == f2,
            (
                Expr::FunctionCall {
                    name: n1,
                    args: a1,
                },
                Expr::FunctionCall {
                    name: n2,
                    args: a2,
                },
            ) => n1 == n2 && a1 == a2,
            _ => false,
        }
    }
}

impl Eq for Expr {}

impl Hash for Expr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Expr::Literal(v) => hash_f64(v, state),
            Expr::StringLiteral(s) => s.hash(state),
            Expr::Var { name, fallback } => {
                name.hash(state);
                fallback.hash(state);
            }
            Expr::Calc(op) => op.hash(state),
            Expr::StyleCondition { branches, fallback } => {
                branches.hash(state);
                fallback.hash(state);
            }
            Expr::FunctionCall { name, args } => {
                name.hash(state);
                args.hash(state);
            }
        }
    }
}

/// A single branch in an `if(style(...): ...)` chain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StyleBranch {
    /// The condition — a single `style()` test or a compound `and`/`or`.
    pub condition: StyleTest,
    pub then: Expr,
}

/// A style condition test inside `if()`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StyleTest {
    /// A single `style(--prop: val)` test.
    Single { property: String, value: Expr },
    /// `condition1 and condition2` — all must match.
    And(Vec<StyleTest>),
    /// `condition1 or condition2` — any must match.
    Or(Vec<StyleTest>),
}

/// Arithmetic / math operations from `calc()`, `mod()`, `round()`, etc.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoundStrategy {
    Nearest,
    Up,
    Down,
    ToZero,
}

/// A complete parsed CSS program ready for pattern compilation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParsedProgram {
    pub properties: Vec<PropertyDef>,
    pub functions: Vec<FunctionDef>,
    /// Property assignments on `.cpu`, in declaration order.
    pub assignments: Vec<Assignment>,
}

/// A single property assignment: `--name: <expr>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Assignment {
    pub property: String,
    pub value: Expr,
}
