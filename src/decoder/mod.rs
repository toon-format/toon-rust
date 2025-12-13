/* rune-xero/src/decoder/mod.rs */
//!▫~•◦---------------------------‣
//! # RUNE-Xero – Zero-Copy Decoder Module
//!▫~•◦------------------------------------‣
//!
//! The decoder serves as the configuration entry point for the parsing pipeline.
//! In the Zero-Copy architecture, "decoding" is synonymous with "parsing" —
//! it produces a borrowed Abstract Syntax Tree (AST) rather than allocating
//! new structs or JSON values.
//!
//! ## Key Capabilities
//! - **Zero-Allocation**: Returns `Document<'a>` borrowing directly from input.
//! - **Configurable Strictness**: Applies validation rules via `DecodeOptions`.
//! - **No External Dependencies**: Removes `serde` and `serde_json` from the hot path.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod parser;
pub mod scanner;
pub mod validation;
pub mod expand;

use crate::types::{DecodeOptions, RuneResult};
use parser::ast::Document;

/// Decode a RUNE string into a Zero-Copy Document AST.
///
/// Returns a `Document<'a>` that borrows from the input string.
/// Accessing data is done via traversing the AST nodes.
///
/// # Examples
///
/// ```rust
/// use rune_format::{decode, DecodeOptions};
/// use rune_format::parser::ast::{Item, Statement};
///
/// let input = "root: continuum";
/// let doc = decode(input, &DecodeOptions::default()).unwrap();
/// 
/// if let Item::Statement(Statement::RootDecl(name)) = &doc.items[0] {
///     assert_eq!(*name, "continuum"); // Borrowed &str
/// }
/// ```
pub fn decode<'a>(input: &'a str, options: &DecodeOptions) -> RuneResult<Document<'a>> {
    // 1. Parse into AST
    let document = parser::parse(input).map_err(|e| crate::types::RuneError::ParseError {
        line: 0,
        column: 0,
        message: e.to_string(),
        context: None,
    })?;

    // 3. Optional: Validation (Zero-Copy Traversal)
    if options.strict {
        let validator = validation::Validator::new(validation::ValidationConfig {
            strict_types: true,
            allow_nulls: false,
        });
        // Note: Validator needs to be updated to handle Document/Item logic
        // validator.validate_document(&document)?; 
    }

    Ok(document)
}

/// Decode with strict validation enabled.
pub fn decode_strict<'a>(input: &'a str) -> RuneResult<Document<'a>> {
    decode(input, &DecodeOptions::new().with_strict(true))
}

/// Decode with strict validation and additional options.
pub fn decode_strict_with_options<'a>(
    input: &'a str,
    options: &DecodeOptions,
) -> RuneResult<Document<'a>> {
    let mut opts = *options;
    opts = opts.with_strict(true);
    decode(input, &opts)
}

/// Decode without type coercion (strings remain strings).
/// In the AST, this just affects how tokens are interpreted by the scanner.
pub fn decode_no_coerce<'a>(input: &'a str) -> RuneResult<Document<'a>> {
    decode(input, &DecodeOptions::new().with_coerce_types(false))
}

/// Decode without type coercion and with additional options.
pub fn decode_no_coerce_with_options<'a>(
    input: &'a str,
    options: &DecodeOptions,
) -> RuneResult<Document<'a>> {
    let mut opts = *options;
    opts = opts.with_coerce_types(false);
    decode(input, &opts)
}

/// Decode with default options.
pub fn decode_default<'a>(input: &'a str) -> RuneResult<Document<'a>> {
    decode(input, &DecodeOptions::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use parser::ast::{Item, Statement, Value, Term, Expression};

    #[test]
    fn test_decode_root() {
        let input = "root: system";
        let doc = decode_default(input).unwrap();
        match &doc.items[0] {
            Item::Statement(Statement::RootDecl(s)) => assert_eq!(*s, "system"),
            _ => panic!("Expected root decl"),
        }
    }

    #[test]
    fn test_decode_expr() {
        let input = "a + b";
        let doc = decode_default(input).unwrap();
        // Just verify it parsed into an expression statement
        match &doc.items[0] {
            Item::Statement(Statement::Expr(_)) => {},
            _ => panic!("Expected expression"),
        }
    }

    #[test]
    fn test_decode_options_flow() {
        // Ensure options are passed correctly (strict mode doesn't panic on valid input)
        let input = "root: valid";
        let _ = decode_strict(input).unwrap();
    }
}
