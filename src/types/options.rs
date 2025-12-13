/* rune-xero/src/types/options.rs */
//!▫~•◦-----------------------------‣
//! # RUNE-Xero – Options Module
//!▫~•◦-----------------------------‣
//!
//! Configuration structs for Encoding and Decoding.
//! Optimized for Zero-Copy usage (Copy types, no internal allocations).
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::{
    constants::DEFAULT_INDENT,
    types::{Delimiter, KeyFoldingMode, PathExpansionMode},
};

/// Indentation style configuration.
///
/// Designed to be `Copy` so options can be passed by value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Indent {
    pub size: usize,
    pub use_tabs: bool,
}

impl Default for Indent {
    fn default() -> Self {
        Self {
            size: DEFAULT_INDENT,
            use_tabs: false,
        }
    }
}

impl Indent {
    /// Create a spaces-based indentation.
    pub const fn spaces(count: usize) -> Self {
        Self { size: count, use_tabs: false }
    }

    /// Create a tabs-based indentation.
    pub const fn tabs() -> Self {
        Self { size: 1, use_tabs: true }
    }
    
    // Note: `get_string()` removed to prevent allocation. 
    // Writers should use `size` and `use_tabs` to stream output directly.
}

/// Options for encoding JSON values to RUNE format.
/// Marked `Copy` for efficient passing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeOptions {
    pub delimiter: Delimiter,
    pub indent: Indent,
    pub key_folding: KeyFoldingMode,
    pub flatten_depth: usize,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            delimiter: Delimiter::Comma,
            indent: Indent::default(),
            key_folding: KeyFoldingMode::Off,
            flatten_depth: usize::MAX,
        }
    }
}

impl EncodeOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_delimiter(mut self, delimiter: Delimiter) -> Self {
        self.delimiter = delimiter;
        self
    }

    pub fn with_indent(mut self, style: Indent) -> Self {
        self.indent = style;
        self
    }

    pub fn with_spaces(mut self, count: usize) -> Self {
        self.indent = Indent::spaces(count);
        self
    }

    pub fn with_key_folding(mut self, mode: KeyFoldingMode) -> Self {
        self.key_folding = mode;
        self
    }

    pub fn with_flatten_depth(mut self, depth: usize) -> Self {
        self.flatten_depth = depth;
        self
    }
}

/// Options for decoding RUNE format.
/// Marked `Copy` for efficient passing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeOptions {
    pub delimiter: Option<Delimiter>,
    pub strict: bool,
    pub coerce_types: bool,
    pub indent: Indent,
    pub expand_paths: PathExpansionMode,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            delimiter: None,
            strict: true,
            coerce_types: true,
            indent: Indent::default(),
            expand_paths: PathExpansionMode::Off,
        }
    }
}

impl DecodeOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    pub fn with_delimiter(mut self, delimiter: Delimiter) -> Self {
        self.delimiter = Some(delimiter);
        self
    }

    pub fn with_coerce_types(mut self, coerce: bool) -> Self {
        self.coerce_types = coerce;
        self
    }

    pub fn with_indent(mut self, style: Indent) -> Self {
        self.indent = style;
        self
    }

    pub fn with_expand_paths(mut self, mode: PathExpansionMode) -> Self {
        self.expand_paths = mode;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_options_indent() {
        let opts = EncodeOptions::new().with_spaces(4);
        assert_eq!(opts.indent.size, 4);
        assert!(!opts.indent.use_tabs);

        let opts = EncodeOptions::new().with_indent(Indent::spaces(2));
        assert_eq!(opts.indent.size, 2);
    }

    #[test]
    fn test_decode_options_coerce_types() {
        let opts = DecodeOptions::new();
        assert!(opts.coerce_types);

        let opts = DecodeOptions::new().with_coerce_types(false);
        assert!(!opts.coerce_types);
    }
}