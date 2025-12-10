/* src/rune/ast.rs */
//! RUNE Abstract Syntax Tree (AST) definitions.
//!
//! # TOON-RUNE – RUNE AST Module
//!▫~•◦---------------------------‣
//!
//! This module defines the core expression tree structures for RUNE:
//! identifiers, literals, terms, and expressions built on `RuneOp`.
//! It also includes statement-level constructs for TOON blocks and root declarations.
//!
//! The AST is intentionally minimal and expression-centric. Higher-level
//! constructs (definitions, constraints, blocks) can be layered on top
//! without changing the fundamental expression nodes.
//!
//! ### Key Types
//! - [`Literal`] – Numeric values.
//! - [`Ident`]   – Symbolic names (types, tensors, nodes, roots).
//! - [`Term`]    – Basic units: identifiers, literals, grouped expressions.
//! - [`Expr`]    – Recursive expression tree parameterized by [`RuneOp`].
//! - [`Stmt`]    – Top-level statements: root declarations, TOON blocks, expressions.
//!
//! ### Example
//! ```rust
//! use rune_format::rune::{Stmt, Expr};
//! use rune_format::rune::RuneOp;
//!
//! let root_stmt = Stmt::root("continuum");
//! let expr_stmt = Stmt::expr(
//!     Expr::binary(
//!         Expr::ident("users"),
//!         RuneOp::Descendant,
//!         Expr::ident("0"),
//!     )
//! );
//! ```
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::rune::ops::{MathOp, RuneOp};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Basic type system for RUNE expressions.
/// This is intentionally small to bootstrap typed AST and inference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuneType {
    Scalar,
    String,
    Gf8,
    PointCloud,
    Array,
    Bool,
    Unknown,
}

impl fmt::Display for RuneType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuneType::Scalar => write!(f, "Scalar"),
            RuneType::String => write!(f, "String"),
            RuneType::Gf8 => write!(f, "Gf8"),
            RuneType::PointCloud => write!(f, "PointCloud"),
            RuneType::Array => write!(f, "Array"),
            RuneType::Bool => write!(f, "Bool"),
            RuneType::Unknown => write!(f, "Unknown"),
        }
    }
}

/// A parameter in a kernel archetype declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelParam {
    pub name: Ident,
    pub typ: Ident,
}

/// Information about a single archetype in a kernel declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchetypeInfo {
    pub name: Ident,
    pub params: Vec<KernelParam>,
}

/// A kernel archetype definition with parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelArchetype {
    pub name: Ident,
    pub params: Vec<(Ident, Literal)>,
}

/// A symbolic identifier in RUNE.
///
/// This covers type symbols (`T`, `Gf8`, `XUID`), nodes, roots,
/// fields, and any named entities.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Ident(pub String);

impl Ident {
    pub fn new<S: Into<String>>(s: S) -> Self {
        Ident(s.into())
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for Ident {
    fn from(s: String) -> Self {
        Ident::new(s)
    }
}

impl From<&str> for Ident {
    fn from(s: &str) -> Self {
        Ident::new(s)
    }
}

impl Into<String> for Ident {
    fn into(self) -> String {
        self.0
    }
}

/// A semantic identifier with a single-letter namespace prefix.
///
/// Examples: T:Gf8, V:vector, R:continuum, Q:e32l
/// The prefix is always a single uppercase letter (A-Z).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SemanticIdent {
    /// The semantic prefix (A-Z)
    pub prefix: char,
    /// The identifier name
    pub name: Ident,
}

impl SemanticIdent {
    pub fn new(prefix: char, name: impl Into<String>) -> Self {
        SemanticIdent {
            prefix,
            name: Ident::new(name),
        }
    }
}

impl fmt::Display for SemanticIdent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.prefix, self.name)
    }
}

/// Literal values in RUNE expressions.
///
/// Supports numeric literals, strings, arrays, booleans, and objects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    /// Numeric literal (parsed as f64).
    Number(f64),
    /// String literal.
    String(String),
    /// Boolean literal: B:t (true) or B:f (false)
    Bool(bool),
    /// Array literal: [1,2,3] or [a,b,c]
    Array(Vec<Expr>),
    /// Object literal: { key: val, ... }
    Object(Vec<(String, Expr)>),
}

impl Literal {
    pub fn number<N: Into<f64>>(n: N) -> Self {
        Literal::Number(n.into())
    }

    pub fn string<S: Into<String>>(s: S) -> Self {
        Literal::String(s.into())
    }

    pub fn bool<B: Into<bool>>(b: B) -> Self {
        Literal::Bool(b.into())
    }

    pub fn array(elements: Vec<Expr>) -> Self {
        Literal::Array(elements)
    }

    pub fn object(entries: Vec<(String, Expr)>) -> Self {
        Literal::Object(entries)
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Number(n) => write!(f, "{}", n),
            Literal::String(s) => write!(f, "\"{}\"", s),
            Literal::Bool(b) => write!(f, "B:{}", if *b { "t" } else { "f" }),
            Literal::Array(elements) => {
                write!(f, "[")?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, "]")
            }
            Literal::Object(entries) => {
                write!(f, "{{")?;
                for (i, (key, val)) in entries.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", key, val)?;
                }
                write!(f, "}}")
            }
        }
    }
}

/// Arithmetic expressions within `[...]` value blocks.
///
/// These support traditional math with operators: `+ - * /`.
/// Isolated from glyph operators for clean separation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MathExpr {
    /// A single math atom (identifier, number, or grouped math).
    Atom(MathAtom),

    /// A binary math operation `lhs op rhs`.
    Binary {
        left: Box<MathExpr>,
        op: MathOp,
        right: Box<MathExpr>,
    },

    /// A unary math operation `op expr` (e.g., `-x`, `+5`).
    Unary {
        op: MathUnaryOp,
        operand: Box<MathExpr>,
    },
}

/// Unary operators in arithmetic expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MathUnaryOp {
    /// Negation `-x`.
    Negate,
    /// Positive `+x` (typically a no-op).
    Plus,
}

impl MathUnaryOp {
    pub fn as_str(self) -> &'static str {
        match self {
            MathUnaryOp::Negate => "-",
            MathUnaryOp::Plus => "+",
        }
    }
}

/// Atoms in arithmetic expressions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MathAtom {
    /// Numeric literal.
    Number(f64),
    /// Variable identifier.
    Ident(Ident),
    /// Grouped sub-expression `(math)` (for precedence).
    Group(Box<MathExpr>),
    /// Array literal inside math block `[expr, expr, ...]`.
    Array(Vec<MathExpr>),
}

impl MathExpr {
    /// Create a math atom expression.
    pub fn atom(atom: MathAtom) -> Self {
        MathExpr::Atom(atom)
    }

    /// Create a binary math expression.
    pub fn binary(left: MathExpr, op: MathOp, right: MathExpr) -> Self {
        MathExpr::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    /// Create a unary math expression.
    pub fn unary(op: MathUnaryOp, operand: MathExpr) -> Self {
        MathExpr::Unary {
            op,
            operand: Box::new(operand),
        }
    }
}

impl fmt::Display for MathExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathExpr::Atom(atom) => match atom {
                MathAtom::Number(n) => write!(f, "{}", n),
                MathAtom::Ident(id) => write!(f, "{}", id),
                MathAtom::Group(inner) => write!(f, "({})", inner),
                MathAtom::Array(elements) => {
                    write!(f, "[")?;
                    for (i, elem) in elements.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", elem)?;
                    }
                    write!(f, "]")
                }
            },
            MathExpr::Binary { left, op, right } => {
                // Add parens for clarity in nested operations
                write!(f, "{} {} {}", left, op, right)
            }
            MathExpr::Unary { op, operand } => {
                write!(f, "{}{}", op.as_str(), operand)
            }
        }
    }
}

/// Atomic terms in a RUNE expression.
///
/// These are the building blocks that operators connect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Term {
    /// A named symbol (identifier).
    Ident(Ident),
    /// A semantic identifier with namespace prefix (e.g., T:Gf8, V:vector).
    SemanticIdent(SemanticIdent),
    /// A literal value.
    Literal(Literal),
    /// A grouped sub-expression `(expr)`.
    Group(Box<Expr>),
    /// Arithmetic within `[...]` value blocks.
    Math(Box<MathExpr>),
    /// A function call: name(arg1, arg2, ...)
    FunctionCall { name: Ident, args: Vec<Expr> },
}

impl Term {
    pub fn ident<S: Into<String>>(s: S) -> Self {
        Term::Ident(Ident::new(s))
    }

    pub fn semantic_ident(prefix: char, name: impl Into<String>) -> Self {
        Term::SemanticIdent(SemanticIdent::new(prefix, name))
    }

    pub fn literal<N: Into<f64>>(n: N) -> Self {
        Term::Literal(Literal::number(n))
    }

    pub fn group(expr: Expr) -> Self {
        Term::Group(Box::new(expr))
    }

    pub fn math(math: MathExpr) -> Self {
        Term::Math(Box::new(math))
    }
}

/// A full RUNE expression.
///
/// This is the node-level representation that a Pratt parser will
/// construct from a token stream (`Term`s and `RuneOp`s).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// A single term (identifier, literal, or grouped expression).
    Term(Term),

    /// A binary expression `lhs op rhs`.
    Binary {
        left: Box<Expr>,
        op: RuneOp,
        right: Box<Expr>,
    },
}

impl Expr {
    /// Construct a term expression from an identifier.
    pub fn ident<S: Into<String>>(s: S) -> Self {
        Expr::Term(Term::ident(s))
    }

    /// Construct a term expression from a numeric literal.
    pub fn literal<N: Into<f64>>(n: N) -> Self {
        Expr::Term(Term::literal(n))
    }

    /// Construct a grouped expression `(expr)`.
    pub fn group(expr: Expr) -> Self {
        Expr::Term(Term::group(expr))
    }

    /// Construct a binary expression `left op right`.
    pub fn binary(left: Expr, op: RuneOp, right: Expr) -> Self {
        Expr::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }
}

/// A typed expression wrapper used for annotating an `Expr` node
/// with a best-effort type computed during parsing/type inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedExpr {
    pub expr: Expr,
    pub r#type: RuneType,
}

impl TypedExpr {
    pub fn new(expr: Expr, r#type: RuneType) -> Self {
        Self { expr, r#type }
    }

    /// Infer a type for a given expression node using shallow heuristics.
    /// This is intentionally conservative: only literal math/strings/arrays
    /// and some semantic hints are recognized. More advanced inference is
    /// left for a future pass.
    pub fn infer(expr: &Expr) -> Self {
        // NOTE: we avoid glob-importing Expr::* to reduce ambiguity with the Term enum
        let r#type = match expr {
            Expr::Term(term) => match term {
                crate::rune::ast::Term::Literal(crate::rune::ast::Literal::Number(_)) => {
                    RuneType::Scalar
                }
                crate::rune::ast::Term::Literal(crate::rune::ast::Literal::String(_)) => {
                    RuneType::String
                }
                crate::rune::ast::Term::Literal(crate::rune::ast::Literal::Bool(_)) => {
                    RuneType::Bool
                }
                crate::rune::ast::Term::Literal(crate::rune::ast::Literal::Array(_)) => {
                    RuneType::Array
                }
                crate::rune::ast::Term::Literal(crate::rune::ast::Literal::Object(_)) => {
                    RuneType::Unknown
                }
                crate::rune::ast::Term::Math(_) => RuneType::Scalar,
                crate::rune::ast::Term::Ident(_) => RuneType::Unknown,
                crate::rune::ast::Term::SemanticIdent(s) => {
                    // Heuristic: common semantic prefix T:Gf8 -> Gf8 type
                    match s.prefix {
                        'T' => {
                            if s.name.0.to_lowercase().contains("gf8") {
                                RuneType::Gf8
                            } else {
                                RuneType::Unknown
                            }
                        }
                        _ => RuneType::Unknown,
                    }
                }
                crate::rune::ast::Term::Group(inner) => TypedExpr::infer(inner).r#type.clone(),
                crate::rune::ast::Term::FunctionCall { name, args: _ } => {
                    // Heuristic: function name suggests type
                    if name.0.contains("Quat")
                        || name.0.contains("quaternion")
                        || name.0.contains("Gf8")
                    {
                        RuneType::Gf8 // quaternions are 4D, represented as scalars for now
                    } else {
                        RuneType::Unknown
                    }
                }
            },
            Expr::Binary {
                left,
                op: _,
                right: _,
            } => {
                // Binary expression type inference: prefer left-side for now
                TypedExpr::infer(left).r#type.clone()
            }
        };

        TypedExpr::new(expr.clone(), r#type)
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Term(t) => match t {
                Term::Ident(id) => write!(f, "{}", id),
                Term::SemanticIdent(sid) => write!(f, "{}", sid),
                Term::Literal(Literal::Number(n)) => write!(f, "{}", n),
                Term::Literal(Literal::String(s)) => write!(f, "\"{}\"", s),
                Term::Literal(Literal::Bool(b)) => write!(f, "B:{}", if *b { "t" } else { "f" }),
                Term::Literal(Literal::Array(elements)) => {
                    write!(f, "[")?;
                    for (i, elem) in elements.iter().enumerate() {
                        if i > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "{}", elem)?;
                    }
                    write!(f, "]")
                }
                Term::Literal(Literal::Object(entries)) => {
                    write!(f, "{{")?;
                    for (i, (key, val)) in entries.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}: {}", key, val)?;
                    }
                    write!(f, "}}")
                }
                Term::Group(inner) => write!(f, "({})", inner),
                Term::Math(math) => write!(f, "[{}]", math),
                Term::FunctionCall { name, args } => {
                    write!(f, "{}(", name)?;
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", arg)?;
                    }
                    write!(f, ")")
                }
            },
            Expr::Binary { left, op, right } => {
                // Don't add spaces around :: (namespace operator)
                if *op == RuneOp::Namespace {
                    write!(f, "{}::{}", left, right)
                } else {
                    write!(f, "{} {} {}", left, op, right)
                }
            }
        }
    }
}

/// Top-level RUNE statements.
///
/// These are the syntactic units parsed from RUNE files:
/// root declarations anchor contexts, TOON/RUNE blocks provide raw data,
/// kernel declarations define computational archetypes,
/// and expressions allow symbolic computations over that data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    /// A root declaration: `root: name`
    /// Anchors the semantic context of the document.
    RootDecl(Ident),

    /// A TOON block: `name ~TOON:\n  content`
    /// Raw TOON data preserved verbatim for later parsing by the TOON library.
    ToonBlock { name: Ident, content: String },

    /// A RUNE block: `name ~RUNE:\n  content`
    /// Preferred raw data blocks that can be interpreted/executed by Rune.
    RuneBlock { name: Ident, content: String },

    /// A kernel declaration: `Kernel:QDot := CUDA:Archetype:RowDot(D: 8)`
    /// Defines a kernel from a CUDA archetype template.
    KernelDecl {
        name: SemanticIdent,
        archetype: KernelArchetype,
    },

    /// A RUNE expression statement.
    /// Typically constraints, definitions, or relations over TOON data.
    Expr(Expr),
}

impl Stmt {
    /// Create a root declaration statement.
    pub fn root<S: Into<String>>(name: S) -> Self {
        Stmt::RootDecl(Ident::new(name))
    }

    /// Create a TOON block statement.
    pub fn toon_block<S: Into<String>>(name: S, content: String) -> Self {
        Stmt::ToonBlock {
            name: Ident::new(name),
            content,
        }
    }

    /// Create a RUNE block statement.
    pub fn rune_block<S: Into<String>>(name: S, content: String) -> Self {
        Stmt::RuneBlock {
            name: Ident::new(name),
            content,
        }
    }

    /// Create an expression statement.
    pub fn expr(expr: Expr) -> Self {
        Stmt::Expr(expr)
    }
}

impl fmt::Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stmt::RootDecl(name) => write!(f, "root: {}", name),
            Stmt::ToonBlock { name, content } => {
                writeln!(f, "{} ~TOON:", name)?;
                for line in content.lines() {
                    writeln!(f, "  {}", line)?;
                }
                Ok(())
            }
            Stmt::RuneBlock { name, content } => {
                writeln!(f, "{} ~RUNE:", name)?;
                for line in content.lines() {
                    writeln!(f, "  {}", line)?;
                }
                Ok(())
            }
            Stmt::KernelDecl { name, archetype } => {
                write!(f, "{} := {}", name, archetype.name)?;
                if !archetype.params.is_empty() {
                    write!(f, "(")?;
                    for (i, (param_name, param_value)) in archetype.params.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}: {}", param_name, param_value)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            Stmt::Expr(expr) => write!(f, "{}", expr),
        }
    }
}

/// Typed form of top-level statements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StmtTyped {
    RootDecl(Ident),
    ToonBlock {
        name: Ident,
        content: String,
    },
    RuneBlock {
        name: Ident,
        content: String,
    },
    KernelDecl {
        name: SemanticIdent,
        archetype: KernelArchetype,
    },
    Expr(TypedExpr),
}

impl StmtTyped {
    pub fn root<S: Into<String>>(name: S) -> Self {
        StmtTyped::RootDecl(Ident::new(name))
    }

    pub fn toon_block<S: Into<String>>(name: S, content: String) -> Self {
        StmtTyped::ToonBlock {
            name: Ident::new(name),
            content,
        }
    }

    pub fn rune_block<S: Into<String>>(name: S, content: String) -> Self {
        StmtTyped::RuneBlock {
            name: Ident::new(name),
            content,
        }
    }

    pub fn expr(expr: TypedExpr) -> Self {
        StmtTyped::Expr(expr)
    }
}

impl fmt::Display for StmtTyped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StmtTyped::RootDecl(name) => write!(f, "root: {}", name),
            StmtTyped::ToonBlock { name, content } => {
                writeln!(f, "{} ~TOON:", name)?;
                for line in content.lines() {
                    writeln!(f, "  {}", line)?;
                }
                Ok(())
            }
            StmtTyped::RuneBlock { name, content } => {
                writeln!(f, "{} ~RUNE:", name)?;
                for line in content.lines() {
                    writeln!(f, "  {}", line)?;
                }
                Ok(())
            }
            StmtTyped::KernelDecl { name, archetype } => {
                write!(f, "{} := {}", name, archetype.name)?;
                if !archetype.params.is_empty() {
                    write!(f, "(")?;
                    for (i, (param_name, param_value)) in archetype.params.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}: {}", param_name, param_value)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            StmtTyped::Expr(te) => write!(f, "{} :: {:?}", te.expr, te.r#type),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rune::ops::RuneOp;

    #[test]
    fn test_expr_binary() {
        let left = Expr::ident("users");
        let right = Expr::literal(0.0);
        let expr = Expr::binary(left, RuneOp::Descendant, right);
        assert_eq!(format!("{}", expr), "users / 0");
    }

    #[test]
    fn test_stmt_root() {
        let stmt = Stmt::root("continuum");
        assert_eq!(format!("{}", stmt), "root: continuum");
    }

    #[test]
    fn test_stmt_toon_block() {
        let content = "items[2]{id,name}:\n  1,hello\n  2,world".to_string();
        let stmt = Stmt::toon_block("data", content);
        let output = format!("{}", stmt);
        assert!(output.contains("data ~TOON:"));
        assert!(output.contains("  items[2]"));
    }
}
