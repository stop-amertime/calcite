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
    /// An integer value from `initial-value`.
    Integer(i64),
    /// A string value from `initial-value`.
    String(String),
}

/// A parsed `@property` declaration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PropertyDef {
    /// Property name (e.g., `"--AX"`).
    pub name: String,
    /// The `syntax` descriptor.
    pub syntax: PropertySyntax,
    /// Whether the property inherits.
    pub inherits: bool,
    /// The `initial-value` descriptor.
    pub initial_value: Option<CssValue>,
}

/// The `syntax` descriptor of an `@property` rule.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PropertySyntax {
    /// `<integer>`.
    Integer,
    /// `<number>`.
    Number,
    /// `<length>`.
    Length,
    /// Any other syntax string.
    Custom(String),
    /// `*` (universal).
    Any,
}

/// A parsed `@function` definition.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionDef {
    /// Function name (e.g., `"--readMem"`).
    pub name: String,
    /// Declared parameters.
    pub parameters: Vec<FunctionParam>,
    /// Local variable definitions (evaluated on call).
    pub locals: Vec<LocalVarDef>,
    /// The `result:` expression.
    pub result: Expr,
}

/// A parameter of a `@function`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionParam {
    /// Parameter name (e.g., `"--at"`).
    pub name: String,
    /// Declared syntax type.
    pub syntax: PropertySyntax,
}

/// A local variable within a `@function`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocalVarDef {
    /// Variable name.
    pub name: String,
    /// Initializer expression.
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
        /// The property name being referenced.
        name: String,
        /// Fallback expression if the variable is undefined.
        fallback: Option<Box<Expr>>,
    },

    /// `calc()` and other math functions.
    Calc(CalcOp),

    /// `if(style(--prop: val): then; else: otherwise)` conditional.
    StyleCondition {
        /// Ordered condition branches to test.
        branches: Vec<StyleBranch>,
        /// Default value when no branch matches.
        fallback: Box<Expr>,
    },

    /// A `@function` call: `--funcName(arg1, arg2)`.
    FunctionCall {
        /// The function name.
        name: String,
        /// Argument expressions.
        args: Vec<Expr>,
    },
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
                Expr::FunctionCall { name: n1, args: a1 },
                Expr::FunctionCall { name: n2, args: a2 },
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
    /// Expression to evaluate when the condition matches.
    pub then: Expr,
}

/// A style condition test inside `if()`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StyleTest {
    /// A single `style(--prop: val)` test.
    Single {
        /// The property being tested.
        property: String,
        /// The value to compare against.
        value: Expr,
    },
    /// `condition1 and condition2` — all must match.
    And(Vec<StyleTest>),
    /// `condition1 or condition2` — any must match.
    Or(Vec<StyleTest>),
}

/// Arithmetic / math operations from `calc()`, `mod()`, `round()`, etc.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CalcOp {
    /// `a + b`.
    Add(Box<Expr>, Box<Expr>),
    /// `a - b`.
    Sub(Box<Expr>, Box<Expr>),
    /// `a * b`.
    Mul(Box<Expr>, Box<Expr>),
    /// `a / b` (returns 0 for division by zero).
    Div(Box<Expr>, Box<Expr>),
    /// `mod(a, b)`.
    Mod(Box<Expr>, Box<Expr>),
    /// `min(a, b, ...)`.
    Min(Vec<Expr>),
    /// `max(a, b, ...)`.
    Max(Vec<Expr>),
    /// `clamp(min, val, max)`.
    Clamp(Box<Expr>, Box<Expr>, Box<Expr>),
    /// `round(strategy, val, interval)`.
    Round(RoundStrategy, Box<Expr>, Box<Expr>),
    /// `pow(base, exp)`.
    Pow(Box<Expr>, Box<Expr>),
    /// `sign(val)` — returns -1, 0, or 1.
    Sign(Box<Expr>),
    /// `abs(val)`.
    Abs(Box<Expr>),
    /// Unary negation (`-val`).
    Negate(Box<Expr>),
}

/// The rounding strategy for `round()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoundStrategy {
    /// Round to nearest (ties to even).
    Nearest,
    /// Round toward positive infinity.
    Up,
    /// Round toward negative infinity.
    Down,
    /// Round toward zero.
    ToZero,
}

/// A complete parsed CSS program ready for pattern compilation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParsedProgram {
    /// `@property` declarations (used to initialise state).
    pub properties: Vec<PropertyDef>,
    /// `@function` definitions.
    pub functions: Vec<FunctionDef>,
    /// Property assignments on `.cpu`, in declaration order.
    pub assignments: Vec<Assignment>,
}

/// A single property assignment: `--name: <expr>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Assignment {
    /// The property being assigned (e.g., `"--AX"`).
    pub property: String,
    /// The value expression.
    pub value: Expr,
}
