/* rune-xero/src/constants.rs */
//!▫~•◦-----------------------------‣
//! # RUNE-Xero – Constants
//!▫~•◦-----------------------------‣
//!
//! System-wide constants and static lookup tables.
//! Optimized for compile-time evaluation where possible.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::types::Delimiter;

/// Characters that have structural meaning in RUNE format.
pub const STRUCTURAL_CHARS: &[char] = &['[', ']', '{', '}', ':', '-'];

/// RUNE keywords that must be quoted when used as strings.
pub const KEYWORDS: &[&str] = &["null", "true", "false"];

/// Default indentation size (2 spaces).
pub const DEFAULT_INDENT: usize = 2;

/// Default delimiter (comma).
pub const DEFAULT_DELIMITER: Delimiter = Delimiter::Comma;

/// Maximum nesting depth to prevent stack overflow.
pub const MAX_DEPTH: usize = 256;

/// Internal marker prefix for quoted keys containing dots.
pub(crate) const QUOTED_KEY_MARKER: &str = "\x00";

/// Check if a character is structural.
/// Const-compatible implementation using match.
#[inline]
pub const fn is_structural_char(ch: char) -> bool {
    match ch {
        '[' | ']' | '{' | '}' | ':' | '-' => true,
        _ => false,
    }
}

/// Check if a string is a reserved keyword.
#[inline]
pub fn is_keyword(s: &str) -> bool {
    // Linear search is optimal for very small set (N=3)
    KEYWORDS.contains(&s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_structural_char() {
        assert!(is_structural_char('['));
        assert!(is_structural_char(']'));
        assert!(is_structural_char('{'));
        assert!(is_structural_char('}'));
        assert!(is_structural_char(':'));
        assert!(is_structural_char('-'));
        assert!(!is_structural_char('a'));
        assert!(!is_structural_char(','));
    }

    #[test]
    fn test_is_keyword() {
        assert!(is_keyword("null"));
        assert!(is_keyword("true"));
        assert!(is_keyword("false"));
        assert!(!is_keyword("hello"));
        assert!(!is_keyword("TRUE"));
    }
}