/* src/ast/mod.rs */
//! RUNE Abstract Syntax Tree (AST) definitions.
//!
//! # Rune-Xero – Zero-Copy AST
//!▫~•◦--------------------------‣
//!
//! This module defines the zero-copy expression tree structures for RUNE.
//! All identifiers, literals, and block contents are stored as borrowed
//! slices (`&'a str`) bound to the original input lifetime.
//!
//! ### Key Types
//! - [`Literal<'a>`] – Zero-copy values (raw slices for strings).
//! - [`Ident<'a>`]   – borrowed symbolic names.
//! - [`Term<'a>`]    – Basic units: identifiers, literals, grouped expressions.
//! - [`Expr<'a>`]    – Recursive expression tree parameterized by [`RuneOp`].
//! - [`Stmt<'a>`]    – Top-level statements holding raw block slices.
//!
//! ### Zero-Copy Notes
//! - No `String` allocations.
//! - No `serde` dependency (pure AST).
//! - `TypedExpr` computes types without cloning string data.
//!
//! ### Example
//! ```rust
//! use rune_xero::ast::{Stmt, Expr};
//! use rune_xero::ops::RuneOp;
//!
//! let root_stmt = Stmt::root("continuum");
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::operator::{MathOp, RuneOp};
use std::fmt;

/// Basic type system for RUNE expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// A kernel archetype definition with parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelArchetype<'a> {
    pub name: Ident<'a>,
    pub params: Vec<(Ident<'a>, Literal<'a>)>,
}

/// A symbolic identifier in RUNE.
///
/// Wraps a borrowed slice `&'a str`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ident<'a>(pub &'a str);

impl<'a> Ident<'a> {
    pub const fn new(s: &'a str) -> Self {
        Ident(s)
    }

    pub fn as_str(&self) -> &'a str {
        self.0
    }
}

impl<'a> fmt::Display for Ident<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl<'a> From<&'a str> for Ident<'a> {
    fn from(s: &'a str) -> Self {
        Ident::new(s)
    }
}

/// A semantic identifier with a single-letter namespace prefix.
///
/// Examples: T:Gf8, V:vector, R:continuum.
/// Zero-copy: holds references to the prefix char (by value) and name slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SemanticIdent<'a> {
    /// The semantic prefix (A-Z)
    pub prefix: char,
    /// The identifier name
    pub name: &'a str,
}

impl<'a> SemanticIdent<'a> {
    pub const fn new(prefix: char, name: &'a str) -> Self {
        SemanticIdent { prefix, name }
    }
}

impl<'a> fmt::Display for SemanticIdent<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.prefix, self.name)
    }
}

/// Literal values in RUNE expressions.
///
/// Supports numeric literals, raw string slices, boolean flags.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal<'a> {
    /// Numeric literal (parsed as f64).
    Number(f64),
    /// Raw string slice (content within quotes).
    Str(&'a str),
    /// Boolean literal: B:t (true) or B:f (false)
    Bool(bool),
    /// Array literal: [1,2,3]
    Array(Vec<Expr<'a>>),
    /// Object literal: { key: val, ... }
    Object(Vec<(&'a str, Expr<'a>)>),
}

impl<'a> Literal<'a> {
    pub fn number(n: f64) -> Self {
        Literal::Number(n)
    }

    pub fn string(s: &'a str) -> Self {
        Literal::Str(s)
    }

    pub fn bool(b: bool) -> Self {
        Literal::Bool(b)
    }
}

impl<'a> fmt::Display for Literal<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Number(n) => write!(f, "{}", n),
            Literal::Str(s) => write!(f, "\"{}\"", s), // Re-quote for display
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
#[derive(Debug, Clone, PartialEq)]
pub enum MathExpr<'a> {
    Atom(MathAtom<'a>),
    Binary {
        left: Box<MathExpr<'a>>,
        op: MathOp,
        right: Box<MathExpr<'a>>,
    },
    Unary {
        op: MathUnaryOp,
        operand: Box<MathExpr<'a>>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathUnaryOp {
    Negate,
    Plus,
}

impl MathUnaryOp {
    pub const fn as_str(self) -> &'static str {
        match self {
            MathUnaryOp::Negate => "-",
            MathUnaryOp::Plus => "+",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MathAtom<'a> {
    Number(f64),
    Ident(Ident<'a>),
    Group(Box<MathExpr<'a>>),
    Array(Vec<MathExpr<'a>>),
}

impl<'a> MathExpr<'a> {
    pub fn atom(atom: MathAtom<'a>) -> Self {
        MathExpr::Atom(atom)
    }

    pub fn binary(left: MathExpr<'a>, op: MathOp, right: MathExpr<'a>) -> Self {
        MathExpr::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }
}

impl<'a> fmt::Display for MathExpr<'a> {
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
                write!(f, "{} {} {}", left, op, right)
            }
            MathExpr::Unary { op, operand } => {
                write!(f, "{}{}", op.as_str(), operand)
            }
        }
    }
}

/// Atomic terms in a RUNE expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Term<'a> {
    Ident(Ident<'a>),
    SemanticIdent(SemanticIdent<'a>),
    Literal(Literal<'a>),
    Group(Box<Expr<'a>>),
    Math(Box<MathExpr<'a>>),
    FunctionCall { name: Ident<'a>, args: Vec<Expr<'a>> },
}

impl<'a> Term<'a> {
    pub fn ident(s: &'a str) -> Self {
        Term::Ident(Ident::new(s))
    }
}

/// A full RUNE expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr<'a> {
    Term(Term<'a>),
    Binary {
        left: Box<Expr<'a>>,
        op: RuneOp,
        right: Box<Expr<'a>>,
    },
}

impl<'a> Expr<'a> {
    pub fn ident(s: &'a str) -> Self {
        Expr::Term(Term::ident(s))
    }

    pub fn literal(n: f64) -> Self {
        Expr::Term(Term::Literal(Literal::number(n)))
    }

    pub fn binary(left: Expr<'a>, op: RuneOp, right: Expr<'a>) -> Self {
        Expr::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }
}

/// A typed expression wrapper.
///
/// In the zero-copy version, we avoid cloning the Expr.
/// Use this struct to pass an Expr reference alongside its inferred type.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedExpr<'a> {
    pub expr: Expr<'a>, // Cloning Expr is cheap (Box<T> copy), data is zero-copy
    pub r#type: RuneType,
}

impl<'a> TypedExpr<'a> {
    pub fn new(expr: Expr<'a>, r#type: RuneType) -> Self {
        Self { expr, r#type }
    }

    /// Infer a type for a given expression node using shallow heuristics.
    pub fn infer(expr: &Expr<'a>) -> Self {
        let r#type = match expr {
            Expr::Term(term) => match term {
                Term::Literal(Literal::Number(_)) => RuneType::Scalar,
                Term::Literal(Literal::Str(_)) => RuneType::String,
                Term::Literal(Literal::Bool(_)) => RuneType::Bool,
                Term::Literal(Literal::Array(_)) => RuneType::Array,
                Term::Literal(Literal::Object(_)) => RuneType::Unknown,
                Term::Math(_) => RuneType::Scalar,
                Term::Ident(_) => RuneType::Unknown,
                Term::SemanticIdent(s) => {
                    match s.prefix {
                        'T' => {
                            // Check raw slice for "Gf8" (case-insensitive check manual or simplified)
                            // For zero-copy, we might just check exact match or manual loop
                            if s.name.eq_ignore_ascii_case("Gf8") {
                                RuneType::Gf8
                            } else {
                                RuneType::Unknown
                            }
                        }
                        _ => RuneType::Unknown,
                    }
                }
                Term::Group(inner) => TypedExpr::infer(inner).r#type,
                Term::FunctionCall { name, .. } => {
                    let n = name.0;
                    if n.contains("Quat") || n.contains("Gf8") {
                        RuneType::Gf8
                    } else {
                        RuneType::Unknown
                    }
                }
            },
            Expr::Binary { left, .. } => {
                TypedExpr::infer(left).r#type
            }
        };

        TypedExpr::new(expr.clone(), r#type)
    }
}

impl<'a> fmt::Display for Expr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Term(t) => match t {
                Term::Ident(id) => write!(f, "{}", id),
                Term::SemanticIdent(sid) => write!(f, "{}", sid),
                Term::Literal(lit) => write!(f, "{}", lit),
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
                if *op == RuneOp::Namespace {
                    write!(f, "{}::{}", left, right)
                } else {
                    write!(f, "{} {} {}", left, op, right)
                }
            }
        }
    }
}

/// Top-level RUNE statements (Zero-Copy).
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt<'a> {
    RootDecl(Ident<'a>),
    ToonBlock { name: Ident<'a>, content: &'a str },
    RuneBlock { name: Ident<'a>, content: &'a str },
    KernelDecl {
        name: SemanticIdent<'a>,
        archetype: KernelArchetype<'a>,
    },
    Expr(Expr<'a>),
}

impl<'a> Stmt<'a> {
    pub fn root(name: &'a str) -> Self {
        Stmt::RootDecl(Ident::new(name))
    }

    pub fn toon_block(name: &'a str, content: &'a str) -> Self {
        Stmt::ToonBlock {
            name: Ident::new(name),
            content,
        }
    }

    pub fn rune_block(name: &'a str, content: &'a str) -> Self {
        Stmt::RuneBlock {
            name: Ident::new(name),
            content,
        }
    }

    pub fn expr(expr: Expr<'a>) -> Self {
        Stmt::Expr(expr)
    }
}

impl<'a> fmt::Display for Stmt<'a> {
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
#[derive(Debug, Clone, PartialEq)]
pub enum StmtTyped<'a> {
    RootDecl(Ident<'a>),
    ToonBlock {
        name: Ident<'a>,
        content: &'a str,
    },
    RuneBlock {
        name: Ident<'a>,
        content: &'a str,
    },
    KernelDecl {
        name: SemanticIdent<'a>,
        archetype: KernelArchetype<'a>,
    },
    Expr(TypedExpr<'a>),
}

impl<'a> StmtTyped<'a> {
    pub fn root(name: &'a str) -> Self {
        StmtTyped::RootDecl(Ident::new(name))
    }

    pub fn toon_block(name: &'a str, content: &'a str) -> Self {
        StmtTyped::ToonBlock {
            name: Ident::new(name),
            content,
        }
    }

    pub fn expr(expr: TypedExpr<'a>) -> Self {
        StmtTyped::Expr(expr)
    }
}

impl<'a> fmt::Display for StmtTyped<'a> {
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
    use crate::operator::RuneOp;

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
        let content = "1,hello\n2,world";
        let stmt = Stmt::toon_block("data", content);
        let output = format!("{}", stmt);
        assert!(output.contains("data ~TOON:"));
    }
}
