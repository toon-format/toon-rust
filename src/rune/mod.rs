/* src/rune/hydron/mod.rs */
//! RUNE (Root-Unified Notation Encoding) is a semantic extension built on top of TOON.
//! Where TOON provides token-efficient data serialization, RUNE adds:
//!
//! - **Root-oriented semantics**: Everything revolves around hierarchical roots
//! - **Operator calculus**: Glyphs and tokens for describing relationships, flow, and structure
//! - **E8-awareness**: Geometric and identity-aware operators
//! - **Composability**: Mix RUNE semantics with TOON data blocks seamlessly
//!
//! ## Overview
//!
//! RUNE files can contain:
//! - **TOON blocks**: Raw TOON data (preserved verbatim)
//! - **RUNE operators**: Relations, constraints, transformations over TOON data
//! - **Root declarations**: Anchor points in your E8 ecosystem
//!
//! ## Example RUNE File
//! ```rune
//! root: continuum
//!
//! data ~TOON:
//!   users[3]{id,name,role}:
//!     1,Ada,admin
//!     2,Bob,user
//!     3,Eve,viewer
//!
//! # RUNE semantics over TOON data
//! users / 0 -> role := admin
//! users / * -> name ~ ValidString()
//! ```
//!
//! This crate leverages the TOON format as foundational data representation
//! while adding symbolic operator layers for E8 ecosystems.
//!
//! TOKEN_FORMAT is Copyright (c) 2025-PRESENT Shreyas S Bhat, Johann Schopplich
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

mod ast;
mod ops;
mod parser;
pub mod parts;

#[cfg(feature = "hydron")]
pub mod hydron;

pub use ast::*;
pub use ops::*;
pub use parser::*;

// Re-export common types for convenience
pub type RuneParser = parser::ParseError;

/// Parse a RUNE source string into a list of statements.
pub fn parse_rune(input: &str) -> Result<Vec<Stmt>, ParseError> {
    parser::parse(input)
}

/// Encode TOON data blocks within RUNE files as raw strings.
pub fn encode_rune(statements: &[Stmt]) -> String {
    let mut output = String::new();
    for stmt in statements {
        output.push_str(&format!("{}\n", stmt));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_toon_block() {
        let input = r#"
root: test_root

data ~TOON:
  items[2]{id,name}:
    1,hello
    2,world
"#;
        let stmts = parse_rune(input).unwrap();
        assert_eq!(stmts.len(), 2);
        // First statement is root declaration
        if let Stmt::RootDecl(root) = &stmts[0] {
            assert_eq!(root.0.as_str(), "test_root");
        } else {
            panic!("Expected root declaration");
        }
        // Second is TOON block
        if let Stmt::ToonBlock { name, content } = &stmts[1] {
            assert_eq!(name.0.as_str(), "data");
            assert!(content.contains("items[2]"));
        } else {
            panic!("Expected TOON block");
        }
    }

    #[test]
    fn test_operator_expression() {
        let input = r#"
items / 0 -> name := hello
"#;
        let stmts = parse_rune(input).unwrap();
        assert_eq!(stmts.len(), 1);
        if let Stmt::Expr(expr) = &stmts[0] {
            // Check it's a binary expression with -> operator (lower precedence)
            // Parses as: items / 0 -> (name := hello)
            if let Expr::Binary { op, left, right } = expr {
                assert_eq!(*op, RuneOp::FlowRight);
                assert_eq!(format!("{}", left), "items / 0");
                assert_eq!(format!("{}", right), "name := hello");
            } else {
                panic!("Expected binary expression");
            }
        }
    }
}
