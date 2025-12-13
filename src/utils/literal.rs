/* rune-xero/src/utils/literal.rs */
//!▫~•◦-----------------------------‣
//! # RUNE-Xero – Literal Utilities
//!▫~•◦-----------------------------‣
//!
//! Helper functions for identifying literal values (numbers, keywords).
//! Optimized for zero-allocation byte-level inspection.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::constants;

/// Check if a string looks like a keyword or number (needs quoting).
pub fn is_literal_like(s: &str) -> bool {
    is_keyword(s) || is_numeric_like(s)
}

#[inline]
pub fn is_keyword(s: &str) -> bool {
    constants::is_keyword(s)
}

#[inline]
pub fn is_structural_char(ch: char) -> bool {
    constants::is_structural_char(ch)
}

/// Check if a string looks like a number (starts with digit, no leading zeros).
///
/// Zero-copy implementation: inspects bytes directly without allocation.
pub fn is_numeric_like(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let bytes = s.as_bytes();
    let mut i = 0;

    // Optional sign
    if bytes[i] == b'-' {
        i += 1;
    }

    // Must have at least one digit after optional sign
    if i >= bytes.len() {
        return false;
    }

    let first_digit = bytes[i];
    if !first_digit.is_ascii_digit() {
        return false;
    }

    // Leading zero check: "01" is invalid (likely string), "0" or "0.5" is valid.
    // If starts with '0', next char (if exists) cannot be a digit.
    if first_digit == b'0' {
        if let Some(&next) = bytes.get(i + 1) {
            if next.is_ascii_digit() {
                return false;
            }
        }
    }

    // Check remainder for valid numeric characters
    // Note: This loose check allows "1.2.3", relying on parser to strict validate later.
    // It strictly replicates the original semantic equivalence.
    bytes[i..].iter().all(|&c| {
        c.is_ascii_digit() 
            || c == b'.' 
            || c == b'e' || c == b'E' 
            || c == b'+' || c == b'-'
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_literal_like() {
        assert!(is_literal_like("null"));
        assert!(is_literal_like("true"));
        assert!(is_literal_like("false"));
        assert!(is_literal_like("123"));
        assert!(is_literal_like("-456"));
        assert!(is_literal_like("3.14"));
        assert!(!is_literal_like("hello"));
        assert!(!is_literal_like(""));
    }

    #[test]
    fn test_is_keyword() {
        assert!(is_keyword("null"));
        assert!(is_keyword("true"));
        assert!(is_keyword("false"));
        assert!(!is_keyword("TRUE"));
        assert!(!is_keyword("hello"));
    }

    #[test]
    fn test_is_structural_char() {
        assert!(is_structural_char('['));
        assert!(is_structural_char('{'));
        assert!(is_structural_char(':'));
        assert!(!is_structural_char('a'));
    }

    #[test]
    fn test_is_numeric_like() {
        assert!(is_numeric_like("123"));
        assert!(is_numeric_like("-456"));
        assert!(is_numeric_like("0"));
        assert!(is_numeric_like("3.14"));
        assert!(is_numeric_like("1e10"));
        assert!(is_numeric_like("1.5e-3"));

        assert!(!is_numeric_like(""));
        assert!(!is_numeric_like("-"));
        assert!(!is_numeric_like("abc"));
        assert!(!is_numeric_like("01"));
        assert!(!is_numeric_like("00"));
    }
}