/* rune-xero/src/utils/string.rs */
//!▫~•◦-------------------------------‣
//! # High-performance, zero-copy string utilities for the RUNE protocol.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module provides functions for escaping, unescaping, quoting, and splitting
//! strings according to RUNE specifications, with a strong focus on minimizing
//! or eliminating memory allocations and copies.
//!
//! ## Key Capabilities
//! - **Conditional Allocation:** Escaping and unescaping functions use `Cow<'a, str>`
//!   to avoid heap allocations when the input string requires no changes.
//! - **Zero-Copy Splitting:** The delimiter splitting function returns a vector of
//!   borrowed slices (`&'a str`), preventing allocations for each substring.
//! - **Key Validation:** Provides fast, allocation-free validation for unquoted keys.
//!
//! ### Architectural Notes
//! The functions in this module are foundational for the RUNE parser and serializer.
//! By using lifetimes and borrowed types, they ensure high performance and low
//! memory overhead when processing RUNE data streams.
//!
//! #### Example
//! ```rust
//! use crate::utils::string::{split_by_delimiter, unescape_string};
//! use crate::types::Delimiter;
//! use std::borrow::Cow;
//!
//! let input = "field1,\"a\\\\b,c\",field3";
//! let parts = split_by_delimiter(input, Delimiter::Comma);
//! assert_eq!(parts, vec!["field1", "\"a\\\\b,c\"", "field3"]);
//!
//! // unescape_string only allocates if an escape sequence is present.
//! let unescaped: Cow<str> = unescape_string(parts).unwrap();
//! assert_eq!(unescaped, r#""a\b,c""#);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::{types::Delimiter, utils::literal};
use std::borrow::Cow;

/// Escape special characters in a string for quoted output.
///
/// This version is zero-copy if no escaping is needed. It returns a `Cow<str>`,
/// which will be a borrowed slice of the original string if no special
/// characters are present, or an owned `String` if allocation was necessary.
pub fn escape_string(s: &str) -> Cow<str> {
    // Find the first character that needs to be escaped.
    let first_escape = s.find(|c| matches!(c, '\n' | '\r' | '\t' | '"' | '\\'));

    match first_escape {
        // If no such character exists, we can return the original slice. Zero-copy.
        None => Cow::Borrowed(s),
        // Otherwise, we must allocate and build the escaped string.
        Some(start_pos) => {
            let mut result = String::with_capacity(s.len());
            // Append the part of the string before the first escaped character.
            result.push_str(&s[..start_pos]);

            // Iterate through the rest of the string and handle escapes.
            for ch in s[start_pos..].chars() {
                match ch {
                    '\n' => result.push_str("\\n"),
                    '\r' => result.push_str("\\r"),
                    '\t' => result.push_str("\\t"),
                    '"' => result.push_str("\\\""),
                    '\\' => result.push_str("\\\\"),
                    _ => result.push(ch),
                }
            }
            Cow::Owned(result)
        }
    }
}

/// Unescape special characters in a quoted string.
///
/// This version is zero-copy if no escape sequences are present. It returns
/// a `Cow<str>`, which will be a borrowed slice of the original string if no
/// backslashes are found, or an owned `String` if unescaping was performed.
///
/// Per RUNE spec §7.1, only these escape sequences are valid:
/// - `\\` → `\`
/// - `\"` → `"`
/// - `\n` → newline
/// - `\r` → carriage return
/// - `\t` → tab
///
/// Any other escape sequence MUST cause an error.
///
/// # Errors
///
/// Returns an error if the string contains an invalid or unterminated escape sequence.
pub fn unescape_string(s: &str) -> Result<Cow<str>, String> {
    // Find the first potential escape sequence.
    let first_escape = s.find('\\');

    match first_escape {
        // If no backslash exists, no unescaping is needed. Zero-copy.
        None => Ok(Cow::Borrowed(s)),
        // If a backslash is found, we must process the string and may need to allocate.
        Some(start_pos) => {
            let mut result = String::with_capacity(s.len());
            // Append the prefix that had no escapes.
            result.push_str(&s[..start_pos]);

            let mut chars = s[start_pos..].chars().peekable();
            let mut position = start_pos;

            while let Some(ch) = chars.next() {
                position += ch.len_utf8();

                if ch == '\\' {
                    if let Some(&next) = chars.peek() {
                        let consumed = match next {
                            'n' => Some('\n'),
                            'r' => Some('\r'),
                            't' => Some('\t'),
                            '"' => Some('"'),
                            '\\' => Some('\\'),
                            _ => {
                                return Err(format!(
                                    "Invalid escape sequence '\\{next}' at position {position}. \
                                     Only \\\\, \\\", \\n, \\r, \\t are valid",
                                ));
                            }
                        };
                        result.push(consumed.unwrap());
                        chars.next(); // consume the escaped character
                        position += next.len_utf8();
                    } else {
                        return Err(format!(
                            "Unterminated escape sequence at end of string (position {position})",
                        ));
                    }
                } else {
                    result.push(ch);
                }
            }
            Ok(Cow::Owned(result))
        }
    }
}

/// Check if a key can be written without quotes (alphanumeric, underscore,
/// dot). This function is already zero-copy.
pub fn is_valid_unquoted_key(key: &str) -> bool {
    if key.is_empty() {
        return false;
    }

    let mut chars = key.chars();
    // Safe to unwrap as we've checked for empty.
    let first = chars.next().unwrap();

    if !first.is_alphabetic() && first != '_' {
        return false;
    }

    chars.all(|c| c.is_alphanumeric() || c == '_' || c == '.')
}

/// Determine if a string needs quoting based on content and delimiter.
/// This function is already zero-copy.
pub fn needs_quoting(s: &str, delimiter: char) -> bool {
    if s.is_empty() {
        return true;
    }

    if literal::is_literal_like(s) {
        return true;
    }

    if s.chars().any(literal::is_structural_char) {
        return true;
    }

    if s.contains(['\\', '"', delimiter, '\n', '\r', '\t']) {
        return true;
    }

    if s.starts_with(char::is_whitespace) || s.ends_with(char::is_whitespace) {
        return true;
    }

    if s.starts_with('-') {
        return true;
    }

    // Check for leading zeros (e.g., "05", "007", "0123")
    // Numbers with leading zeros must be quoted
    if s.starts_with('0') && s.len() > 1 && s.chars().nth(1).is_some_and(|c| c.is_ascii_digit()) {
        return true;
    }

    false
}

/// Quote and escape a string.
/// This function still allocates a `String` because quotes must be added,
/// but it now uses the zero-copy `escape_string` internally, avoiding a
/// second allocation when the string does not need escaping.
pub fn quote_string(s: &str) -> String {
    format!("\"{}\"", escape_string(s))
}

/// Splits a string by a delimiter, honoring quotes.
///
/// This is a zero-copy implementation that returns a `Vec` of borrowed slices
/// (`&str`) pointing into the original input string. It avoids all intermediate
/// string allocations.
pub fn split_by_delimiter<'a>(s: &'a str, delimiter: Delimiter) -> Vec<&'a str> {
    if s.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut start = 0;
    let mut in_quotes = false;
    let delim_char = delimiter.as_char() as u8;
    let bytes = s.as_bytes();

    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b'"' {
            // A simple quote toggle; assumes quotes are not escaped within fields for splitting.
            in_quotes = !in_quotes;
        } else if byte == delim_char && !in_quotes {
            // Found a delimiter, push the preceding slice.
            result.push(s[start..i].trim());
            start = i + 1; // Start the next slice after the delimiter.
        }
    }

    // Push the final segment after the last delimiter.
    if start <= s.len() {
        result.push(s[start..].trim());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[test]
    fn test_escape_string() {
        // Zero-copy case
        assert_eq!(escape_string("hello"), Cow::Borrowed("hello"));
        // Allocation case
        assert_eq!(
            escape_string("hello\nworld"),
            Cow::Owned::<String>("hello\\nworld".to_string())
        );
        assert_eq!(
            escape_string("say \"hi\""),
            Cow::Owned::<String>("say \\\"hi\\\"".to_string())
        );
        assert_eq!(
            escape_string("back\\slash"),
            Cow::Owned::<String>("back\\\\slash".to_string())
        );
    }

    #[test]
    fn test_unescape_string() {
        // Zero-copy case
        assert_eq!(unescape_string("hello").unwrap(), Cow::Borrowed("hello"));
        // Allocation cases
        assert_eq!(
            unescape_string("hello\\nworld").unwrap(),
            Cow::Owned::<String>("hello\nworld".to_string())
        );
        assert_eq!(
            unescape_string("say \\\"hi\\\"").unwrap(),
            Cow::Owned::<String>("say \"hi\"".to_string())
        );
        assert_eq!(
            unescape_string("back\\\\slash").unwrap(),
            Cow::Owned::<String>("back\\slash".to_string())
        );
        assert_eq!(
            unescape_string("tab\\there").unwrap(),
            Cow::Owned::<String>("tab\there".to_string())
        );
        assert_eq!(
            unescape_string("return\\rhere").unwrap(),
            Cow::Owned::<String>("return\rhere".to_string())
        );
    }

    #[test]
    fn test_unescape_string_invalid_escapes() {
        // Invalid escape sequences should error
        assert!(unescape_string("bad\\xescape").is_err());
        assert!(unescape_string("bad\\uescape").is_err());
        assert!(unescape_string("bad\\0escape").is_err());
        assert!(unescape_string("bad\\aescape").is_err());

        // Unterminated escape at end
        assert!(unescape_string("ends\\").is_err());
    }

    #[test]
    fn test_unescape_string_error_messages() {
        let result = unescape_string("bad\\x");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Invalid escape sequence"));
        assert!(err.contains("\\x"));
    }

    #[test]
    fn test_needs_quoting() {
        let comma = Delimiter::Comma.as_char();

        assert!(needs_quoting("", comma));
        assert!(needs_quoting("true", comma));
        assert!(needs_quoting("false", comma));
        assert!(needs_quoting("null", comma));
        assert!(needs_quoting("123", comma));
        assert!(needs_quoting("hello[world]", comma));
        assert!(needs_quoting("key:value", comma));
        assert!(needs_quoting("a,b", comma));
        assert!(!needs_quoting("a,b", Delimiter::Pipe.as_char()));
        assert!(!needs_quoting("hello world", comma));
        assert!(needs_quoting(" hello", comma));
        assert!(needs_quoting("hello ", comma));
        assert!(!needs_quoting("hello", comma));
        assert!(!needs_quoting("world", comma));
        assert!(!needs_quoting("helloworld", comma));
        assert!(needs_quoting("-123", comma));
        assert!(needs_quoting("007", comma));
    }

    #[test]
    fn test_quote_string() {
        assert_eq!(quote_string("hello"), "\"hello\"");
        assert_eq!(quote_string("hello\nworld"), "\"hello\\nworld\"");
    }

    #[test]
    fn test_split_by_delimiter() {
        let comma = Delimiter::Comma;

        let input1 = "a,b,c";
        let expected1: Vec<&str> = vec!["a", "b", "c"];
        assert_eq!(split_by_delimiter(input1, comma), expected1);

        let input2 = "a, b, c";
        let expected2: Vec<&str> = vec!["a", "b", "c"];
        assert_eq!(split_by_delimiter(input2, comma), expected2);

        let input3 = "\"a,b\",c";
        let expected3: Vec<&str> = vec!["\"a,b\"", "c"];
        assert_eq!(split_by_delimiter(input3, comma), expected3);

        let input4 = "a,\"b, c\", d";
        let expected4: Vec<&str> = vec!["a", "\"b, c\"", "d"];
        assert_eq!(split_by_delimiter(input4, comma), expected4);
        
        let input5 = "";
        let expected5: Vec<&str> = vec![];
        assert_eq!(split_by_delimiter(input5, comma), expected5);
    }

    #[test]
    fn test_is_valid_unquoted_key() {
        // Valid keys (should return true)
        assert!(is_valid_unquoted_key("normal_key"));
        assert!(is_valid_unquoted_key("key123"));
        assert!(is_valid_unquoted_key("key.value"));
        assert!(is_valid_unquoted_key("_private"));
        assert!(is_valid_unquoted_key("KeyName"));
        assert!(is_valid_unquoted_key("key_name"));
        assert!(is_valid_unquoted_key("key.name.sub"));
        assert!(is_valid_unquoted_key("a"));
        assert!(is_valid_unquoted_key("_"));
        assert!(is_valid_unquoted_key("key_123.value"));

        // Invalid keys (should return false)
        assert!(!is_valid_unquoted_key(""));
        assert!(!is_valid_unquoted_key("123"));
        assert!(!is_valid_unquoted_key("key:value"));
        assert!(!is_valid_unquoted_key("key-value"));
        assert!(!is_valid_unquoted_key("key value"));
        assert!(!is_valid_unquoted_key(".key"));
        assert!(!is_valid_unquoted_key("key[value]"));
        assert!(!is_valid_unquoted_key("key{value}"));
    }
}