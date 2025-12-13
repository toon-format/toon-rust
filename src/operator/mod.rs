/* src/operator/mod.rs */
//! Core Operator Registry and Definitions for RUNE.
//!
//! # Rune-Xero – Operator Registry
//!▫~•◦-----------------------------‣
//!
//! This module defines the strict, closed registry of valid RUNE operators.
//! It maps text representations to strongly-typed, `Copy`-enabled Rust enums.
//!
//! ### Key Capabilities
//! - **Zero-Allocation Parsing:** `FromStr` implementation never allocates, even on error.
//! - **Closed Registry:** `RuneOp` exhaustively lists every valid operator.
//! - **Precedence Logic:** Pratt parsing binding powers defined as `(u8, u8)` constants.
//! - **Category Safety:** Classification of operators into Glyph, Relation, and Math.
//!
//! ### Example
//! ```rust
//! use rune_xero::operator::RuneOp;
//! use std::str::FromStr;
//!
//! let op = RuneOp::from_str("->").unwrap();
//! assert_eq!(op, RuneOp::FlowRight);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::fmt;
use std::str::FromStr;

/// Categories of operators in RUNE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCategory {
    /// Topological shapes (e.g., `/\`, `\|/`).
    Glyph,
    /// Structural relations (e.g., `->`, `:`, `:=`).
    Relation,
    /// Value comparisons (e.g., `<`, `>`).
    Compare,
    /// Arithmetic operations (e.g., `+`, `*`).
    Math,
}

/// The Closed Registry of all valid RUNE operators.
///
/// Any sequence of characters not matching one of these variants
/// is syntactically invalid in RUNE.
///
/// This enum is `Copy` and `Eq`, suitable for use as a lightweight token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuneOp {
    // --- 1. Glyph Operators (Topology/Shape) ---
    /// `/\` : Branch then converge (Split -> Join).
    SplitJoin,
    /// `\/` : Converge then branch (Join -> Split).
    JoinSplit,
    /// `|/` : Stable lineage then branch away (Descend from Anchor).
    AnchorDescend,
    /// `/|` : Branch away then stabilize (Branch -> Stabilize).
    BranchStabilize,
    /// `\|` : Converge to root then stabilize.
    RootStabilize,
    /// `|\` : Stabilize then converge to root.
    StabilizeRoot,
    /// `\|/` : Symmetric split from a stable center.
    SymmetricSplit,
    /// `/|\` : Branch, Anchor, Branch (Composite).
    BranchAnchorBranch,

    // --- 2. Token Operators (Relations) ---
    /// `:` : Bind / Key-Value / Annotation.
    Bind,
    /// `=:` : Specializes / Instance of / Emergent from.
    Specializes,
    /// `::` : Namespace / Type Tag.
    Namespace,
    /// `:=` : Definition / Assignment.
    Define,
    /// `:=:` : Match / Pattern recognition.
    Match,
    /// `=:=` : Unify / Structural isomorphism.
    Unify,
    /// `=` : Equality / Constraint (Invariant).
    Equal,
    /// `->` : Directed Edge (Flow Right / Rootwards).
    FlowRight,
    /// `<-` : Reverse Edge (Flow Left).
    FlowLeft,
    /// `<->` : Bidirectional flow / Oscillation / Exchange.
    FlowBidirectional,
    /// `>-<` : Convergent flow / Transformation focus.
    FlowConvergent,
    /// `/` : Descendant / Under (Structural Context).
    Descendant,
    /// `\` : Ancestor / Parent (Sugar for `->` in some contexts).
    Ancestor,
    /// `|` : Alias / Equivalence.
    Alias,
    /// `||` : Parallel / Siblings.
    Parallel,
    /// `~` : Transform / View.
    Transform,
    /// `|>` : Pipeline Right / Function composition (left-to-right).
    PipelineRight,
    /// `<|` : Pipeline Left / Reverse function composition (right-to-left).
    PipelineLeft,
    /// `:>` : Output / Produces / Generates (context yields output).
    Output,
    /// `<:` : Input / Requires / Accepts (context needs input).
    Input,

    // --- 4. Comparison ---
    /// `<` : Less / Precedes / Deeper.
    Less,
    /// `<=` : Less than or equal.
    LessEqual,
    /// `>` : Greater / Succeeds / Higher.
    Greater,
    /// `>=` : Greater than or equal.
    GreaterEqual,
}

impl RuneOp {
    /// Returns the semantic category of the operator.
    pub const fn category(&self) -> OpCategory {
        match self {
            Self::SplitJoin
            | Self::JoinSplit
            | Self::AnchorDescend
            | Self::BranchStabilize
            | Self::RootStabilize
            | Self::StabilizeRoot
            | Self::SymmetricSplit
            | Self::BranchAnchorBranch => OpCategory::Glyph,

            Self::Bind
            | Self::Specializes
            | Self::Namespace
            | Self::Define
            | Self::Match
            | Self::Unify
            | Self::Equal
            | Self::FlowRight
            | Self::FlowLeft
            | Self::FlowBidirectional
            | Self::FlowConvergent
            | Self::Descendant
            | Self::Ancestor
            | Self::Alias
            | Self::Parallel
            | Self::Transform
            | Self::PipelineRight
            | Self::PipelineLeft
            | Self::Output
            | Self::Input => OpCategory::Relation,

            Self::Less | Self::LessEqual | Self::Greater | Self::GreaterEqual => {
                OpCategory::Compare
            }
        }
    }

    /// Returns the textual representation of the operator.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::SplitJoin => "/\\",
            Self::JoinSplit => "\\/",
            Self::AnchorDescend => "|/",
            Self::BranchStabilize => "/|",
            Self::RootStabilize => "\\|",
            Self::StabilizeRoot => "|\\",
            Self::SymmetricSplit => "\\|/",
            Self::BranchAnchorBranch => "/|\\",

            Self::Bind => ":",
            Self::Specializes => "=:",
            Self::Namespace => "::",
            Self::Define => ":=",
            Self::Match => ":=:",
            Self::Unify => "=:=",
            Self::Equal => "=",
            Self::FlowRight => "->",
            Self::FlowLeft => "<-",
            Self::FlowBidirectional => "<->",
            Self::FlowConvergent => ">-<",
            Self::Descendant => "/",
            Self::Ancestor => "\\",
            Self::Alias => "|",
            Self::Parallel => "||",
            Self::Transform => "~",
            Self::PipelineRight => "|>",
            Self::PipelineLeft => "<|",
            Self::Output => ":>",
            Self::Input => "<:",

            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
        }
    }

    /// Binding Power for Pratt Parsing (Precedence).
    ///
    /// Returns `(left_binding_power, right_binding_power)`.
    /// Higher numbers bind tighter.
    pub const fn binding_power(&self) -> (u8, u8) {
        match self {
            // Namespace / Path / Hierarchy
            Self::Namespace => (70, 71),
            Self::Descendant | Self::Ancestor => (60, 61),

            // Flow / Graph Edges / Glyphs / Transform
            Self::FlowRight
            | Self::FlowLeft
            | Self::FlowBidirectional
            | Self::FlowConvergent
            | Self::SplitJoin
            | Self::JoinSplit
            | Self::SymmetricSplit
            | Self::BranchAnchorBranch
            | Self::Transform
            | Self::AnchorDescend
            | Self::BranchStabilize
            | Self::RootStabilize
            | Self::StabilizeRoot => (50, 51),

            // Comparison
            Self::Less | Self::LessEqual | Self::Greater | Self::GreaterEqual | Self::Equal => {
                (40, 41)
            }

            // Loose Structure
            Self::Parallel | Self::Alias => (30, 31),

            // Additional relation operators
            Self::Specializes
            | Self::Match
            | Self::Unify
            | Self::PipelineRight
            | Self::PipelineLeft
            | Self::Output
            | Self::Input => (35, 36),

            // Definition / Assignment / Bind: Lowest
            Self::Bind | Self::Define => (10, 11),
        }
    }
}

/// Zero-Allocation parsing error.
/// 
/// Does not hold the invalid string to prevent allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidOpError;

impl fmt::Display for InvalidOpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid RUNE operator literal")
    }
}

impl std::error::Error for InvalidOpError {}

impl FromStr for RuneOp {
    type Err = InvalidOpError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            // Glyphs (3-char)
            "\\|/" => Ok(Self::SymmetricSplit),
            "/|\\" => Ok(Self::BranchAnchorBranch),

            // Glyphs (2-char)
            "/\\" => Ok(Self::SplitJoin),
            "\\/" => Ok(Self::JoinSplit),
            "|/" => Ok(Self::AnchorDescend),
            "/|" => Ok(Self::BranchStabilize),
            "\\|" => Ok(Self::RootStabilize),
            "|\\" => Ok(Self::StabilizeRoot),

            // Tokens (3-char)
            "=:=" => Ok(Self::Unify),
            ":=:" => Ok(Self::Match),

            // Tokens (3-char)
            "|>" => Ok(Self::PipelineRight),
            "<|" => Ok(Self::PipelineLeft),
            ":>" => Ok(Self::Output),
            "<:" => Ok(Self::Input),

            // Tokens (3-char) - Flow
            "<->" => Ok(Self::FlowBidirectional),
            ">-<" => Ok(Self::FlowConvergent),

            // Tokens (2-char)
            "=:" => Ok(Self::Specializes),
            "::" => Ok(Self::Namespace),
            ":=" => Ok(Self::Define),
            "->" => Ok(Self::FlowRight),
            "<-" => Ok(Self::FlowLeft),
            "<=" => Ok(Self::LessEqual),
            ">=" => Ok(Self::GreaterEqual),
            "||" => Ok(Self::Parallel),

            // Tokens (1-char)
            ":" => Ok(Self::Bind),
            "=" => Ok(Self::Equal),
            "<" => Ok(Self::Less),
            ">" => Ok(Self::Greater),
            "/" => Ok(Self::Descendant),
            "\\" => Ok(Self::Ancestor),
            "|" => Ok(Self::Alias),
            "~" => Ok(Self::Transform),

            _ => Err(InvalidOpError),
        }
    }
}

/// Arithmetic operators within math blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MathOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power, // ^ operator
    Modulo,
    Root, // R operator: n-th root
}

impl MathOp {
    pub const fn precedence(self) -> u8 {
        match self {
            MathOp::Add | MathOp::Subtract => 1,                     // + -
            MathOp::Multiply | MathOp::Divide | MathOp::Modulo => 2, // * / %
            MathOp::Power | MathOp::Root => 3,                       // ^ R
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            MathOp::Add => "+",
            MathOp::Subtract => "-",
            MathOp::Multiply => "*",
            MathOp::Divide => "/",
            MathOp::Power => "^",
            MathOp::Modulo => "%",
            MathOp::Root => "R",
        }
    }
}

impl fmt::Display for MathOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Display for RuneOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_from_str() {
        assert_eq!(RuneOp::from_str("->").unwrap(), RuneOp::FlowRight);
        assert_eq!(RuneOp::from_str("<-").unwrap(), RuneOp::FlowLeft);
        assert_eq!(RuneOp::from_str("<->").unwrap(), RuneOp::FlowBidirectional);
        assert_eq!(RuneOp::from_str(">-<").unwrap(), RuneOp::FlowConvergent);
        assert_eq!(RuneOp::from_str("/\\").unwrap(), RuneOp::SplitJoin);
        assert_eq!(RuneOp::from_str(":=").unwrap(), RuneOp::Define);
        assert_eq!(RuneOp::from_str("=:=").unwrap(), RuneOp::Unify);
        assert_eq!(RuneOp::from_str(":=:").unwrap(), RuneOp::Match);
        assert_eq!(RuneOp::from_str("=:").unwrap(), RuneOp::Specializes);
        assert_eq!(RuneOp::from_str("|>").unwrap(), RuneOp::PipelineRight);
        assert_eq!(RuneOp::from_str("<|").unwrap(), RuneOp::PipelineLeft);
        assert_eq!(RuneOp::from_str(":>").unwrap(), RuneOp::Output);
        assert_eq!(RuneOp::from_str("<:").unwrap(), RuneOp::Input);
    }

    #[test]
    fn test_invalid_operator() {
        assert!(RuneOp::from_str("=>").is_err());
        assert!(RuneOp::from_str("/->").is_err());
        assert!(RuneOp::from_str(":|").is_err());
    }

    #[test]
    fn test_binding_power() {
        assert!(RuneOp::FlowRight.binding_power().0 > RuneOp::Define.binding_power().0);
    }
}