/* rune-xero/src/types/delimiter.rs */
//!▫~•◦-------------------------------‣
//! # RUNE-Xero – Delimiter Module
//!▫~•◦-------------------------------------------------------------------‣
//! 
//! Defines the supported delimiters for RUNE arrays (Comma, Pipe, Tab).
//! Provides zero-cost conversion and inspection utilities for parser tokens.
//!
//! ## Key Capabilities
//! - **Zero-Allocation:** All methods operate on stack-allocated `char` or borrowed `&str`.
//! - **Const-Evaluation:** Mapping methods are `const fn` for compile-time optimization.
//! - **Iterator Efficiency:** String checks use iterators to avoid temporary allocations.
//!
//! ### Architectural Notes
//! This module is a core primitive with no external dependencies.
//! It is used by both the `Scanner` (decoder) and `Writer` (encoder).
//!
//! #### Example
//! ```rust
//! use rune_xero::types::Delimiter;
//!
//! let d = Delimiter::Pipe;
//! assert_eq!(d.as_char(), '|');
//! assert!(d.contains_in("a|b"));
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::fmt;

/// Delimiter character used to separate array elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Delimiter {
    #[default]
    Comma,
    Tab,
    Pipe,
}

impl Delimiter {
    /// Get the character representation of this delimiter.
    #[inline]
    pub const fn as_char(&self) -> char {
        match self {
            Delimiter::Comma => ',',
            Delimiter::Tab => '\t',
            Delimiter::Pipe => '|',
        }
    }

    /// Get the string representation for metadata.
    #[inline]
    pub const fn as_metadata_str(&self) -> &'static str {
        match self {
            Delimiter::Comma => "",
            Delimiter::Tab => "\t",
            Delimiter::Pipe => "|",
        }
    }

    /// Parse a delimiter from a character.
    #[inline]
    pub const fn from_char(c: char) -> Option<Self> {
        match c {
            ',' => Some(Delimiter::Comma),
            '\t' => Some(Delimiter::Tab),
            '|' => Some(Delimiter::Pipe),
            _ => None,
        }
    }

    /// Check if the delimiter character appears in the string.
    /// Zero-allocation check using iterator.
    pub fn contains_in(&self, s: &str) -> bool {
        s.contains(self.as_char())
    }
}

impl fmt::Display for Delimiter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use std::fmt::Write;
        f.write_char(self.as_char())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delimiter_conversion() {
        assert_eq!(Delimiter::Comma.as_char(), ',');
        assert_eq!(Delimiter::Tab.as_char(), '\t');
        assert_eq!(Delimiter::Pipe.as_char(), '|');
    }

    #[test]
    fn test_delimiter_from_char() {
        assert_eq!(Delimiter::from_char(','), Some(Delimiter::Comma));
        assert_eq!(Delimiter::from_char('\t'), Some(Delimiter::Tab));
        assert_eq!(Delimiter::from_char('|'), Some(Delimiter::Pipe));
        assert_eq!(Delimiter::from_char('x'), None);
    }

    #[test]
    fn test_delimiter_contains() {
        assert!(Delimiter::Comma.contains_in("a,b,c"));
        assert!(Delimiter::Tab.contains_in("a\tb\tc"));
        assert!(Delimiter::Pipe.contains_in("a|b|c"));
        assert!(!Delimiter::Comma.contains_in("abc"));
    }
}