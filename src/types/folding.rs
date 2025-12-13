/* rune-xero/src/types/folding.rs */
//!▫~•◦-----------------------------‣
//! # RUNE-Xero – Key Folding Types
//!▫~•◦-----------------------------‣
//!
//! Enums and validation logic for key folding and path expansion.
//! Zero-allocation implementation operating purely on stack types and string slices.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// Configuration for key folding behavior during encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeyFoldingMode {
    /// No folding performed. All objects use standard nesting.
    #[default]
    Off,
    /// Fold eligible chains according to safety rules.
    Safe,
}

/// Configuration for path expansion behavior during decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathExpansionMode {
    /// Dotted keys are treated as literal keys. No expansion.
    #[default]
    Off,
    /// Expand eligible dotted keys according to safety rules.
    Safe,
}

/// Check if a key segment is a valid IdentifierSegment.
///
/// Strict rules for segments involved in folding/expansion:
/// - Must start with alphabetic char or `_`.
/// - Must contain only alphanumeric chars or `_`.
/// - NO dots allowed (dot is the separator).
///
/// Operates on borrowed string slice without allocation.
pub fn is_identifier_segment(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let mut chars = s.chars();

    // First character must be letter or underscore
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };

    if !first.is_alphabetic() && first != '_' {
        return false;
    }

    // Remaining characters: letters, digits, or underscore (NO dots)
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_folding_mode_default() {
        assert_eq!(KeyFoldingMode::default(), KeyFoldingMode::Off);
    }

    #[test]
    fn test_path_expansion_mode_default() {
        assert_eq!(PathExpansionMode::default(), PathExpansionMode::Off);
    }

    #[test]
    fn test_is_identifier_segment() {
        assert!(is_identifier_segment("a"));
        assert!(is_identifier_segment("_private"));
        assert!(is_identifier_segment("userName"));
        assert!(is_identifier_segment("user_name"));
        assert!(is_identifier_segment("user123"));
        assert!(is_identifier_segment("_123"));

        assert!(!is_identifier_segment(""));
        assert!(!is_identifier_segment("123"));
        assert!(!is_identifier_segment("user-name"));
        assert!(!is_identifier_segment("user.name"));
        assert!(!is_identifier_segment("user name"));
        assert!(!is_identifier_segment("user:name"));
        assert!(!is_identifier_segment(".name"));
    }
}