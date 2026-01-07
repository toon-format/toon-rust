use crate::{
    constants::DEFAULT_INDENT,
    types::{Delimiter, KeyFoldingMode, PathExpansionMode},
};

/// Indentation style used for nested structures.
///
/// # Examples
/// ```
/// use toon_format::Indent;
///
/// let indent = Indent::Spaces(2);
/// let _ = indent;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Indent {
    Spaces(usize),
}

impl Default for Indent {
    fn default() -> Self {
        Indent::Spaces(DEFAULT_INDENT)
    }
}

impl Indent {
    /// Return the indentation string for a given depth.
    ///
    /// # Examples
    /// ```
    /// use toon_format::Indent;
    ///
    /// let indent = Indent::Spaces(2);
    /// assert_eq!(indent.get_string(2), "    ");
    /// ```
    pub fn get_string(&self, depth: usize) -> String {
        if depth == 0 {
            return String::new();
        }

        match self {
            Indent::Spaces(count) => {
                if *count > 0 {
                    " ".repeat(*count * depth)
                } else {
                    String::new()
                }
            }
        }
    }

    /// Return the number of spaces used for indentation.
    ///
    /// # Examples
    /// ```
    /// use toon_format::Indent;
    ///
    /// let indent = Indent::Spaces(4);
    /// assert_eq!(indent.get_spaces(), 4);
    /// ```
    pub fn get_spaces(&self) -> usize {
        match self {
            Indent::Spaces(count) => *count,
        }
    }
}

/// Options for encoding JSON values to TOON format.
///
/// # Examples
/// ```
/// use toon_format::{Delimiter, EncodeOptions};
///
/// let opts = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
/// let _ = opts;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Create new encoding options with defaults.
    ///
    /// # Examples
    /// ```
    /// use toon_format::EncodeOptions;
    ///
    /// let opts = EncodeOptions::new();
    /// let _ = opts;
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the delimiter for array elements.
    ///
    /// # Examples
    /// ```
    /// use toon_format::{Delimiter, EncodeOptions};
    ///
    /// let opts = EncodeOptions::new().with_delimiter(Delimiter::Tab);
    /// let _ = opts;
    /// ```
    pub fn with_delimiter(mut self, delimiter: Delimiter) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set the indentation string for nested structures.
    ///
    /// # Examples
    /// ```
    /// use toon_format::{EncodeOptions, Indent};
    ///
    /// let opts = EncodeOptions::new().with_indent(Indent::Spaces(4));
    /// let _ = opts;
    /// ```
    pub fn with_indent(mut self, style: Indent) -> Self {
        self.indent = style;
        self
    }

    /// Set indentation to a specific number of spaces.
    ///
    /// # Examples
    /// ```
    /// use toon_format::EncodeOptions;
    ///
    /// let opts = EncodeOptions::new().with_spaces(2);
    /// let _ = opts;
    /// ```
    pub fn with_spaces(mut self, count: usize) -> Self {
        self.indent = Indent::Spaces(count);
        self
    }

    /// Enable key folding (v1.5 feature).
    ///
    /// When set to `Safe`, single-key object chains will be folded into
    /// dotted-path notation if all safety requirements are met.
    ///
    /// Default: `Off`
    ///
    /// # Examples
    /// ```
    /// use toon_format::EncodeOptions;
    /// use toon_format::types::KeyFoldingMode;
    ///
    /// let opts = EncodeOptions::new().with_key_folding(KeyFoldingMode::Safe);
    /// let _ = opts;
    /// ```
    pub fn with_key_folding(mut self, mode: KeyFoldingMode) -> Self {
        self.key_folding = mode;
        self
    }

    /// Set maximum depth for key folding.
    ///
    /// Controls how many segments will be folded. A value of 2 folds
    /// only two-segment chains: `{a: {b: val}}` → `a.b: val`.
    ///
    /// Default: `usize::MAX` (fold entire eligible chains)
    ///
    /// # Examples
    /// ```
    /// use toon_format::EncodeOptions;
    /// use toon_format::types::KeyFoldingMode;
    ///
    /// let opts = EncodeOptions::new()
    ///     .with_key_folding(KeyFoldingMode::Safe)
    ///     .with_flatten_depth(2);
    /// let _ = opts;
    /// ```
    pub fn with_flatten_depth(mut self, depth: usize) -> Self {
        self.flatten_depth = depth;
        self
    }
}

/// Options for decoding TOON format to JSON values.
///
/// # Examples
/// ```
/// use toon_format::DecodeOptions;
///
/// let opts = DecodeOptions::new().with_strict(false);
/// let _ = opts;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Create new decoding options with defaults (strict mode enabled).
    ///
    /// # Examples
    /// ```
    /// use toon_format::DecodeOptions;
    ///
    /// let opts = DecodeOptions::new();
    /// let _ = opts;
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable strict mode (validates array lengths, indentation,
    /// etc.).
    ///
    /// # Examples
    /// ```
    /// use toon_format::DecodeOptions;
    ///
    /// let opts = DecodeOptions::new().with_strict(false);
    /// let _ = opts;
    /// ```
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Set the expected delimiter (auto-detected if None).
    ///
    /// # Examples
    /// ```
    /// use toon_format::{DecodeOptions, Delimiter};
    ///
    /// let opts = DecodeOptions::new().with_delimiter(Delimiter::Pipe);
    /// let _ = opts;
    /// ```
    pub fn with_delimiter(mut self, delimiter: Delimiter) -> Self {
        self.delimiter = Some(delimiter);
        self
    }

    /// Enable or disable type coercion (strings like "123" -> numbers).
    ///
    /// # Examples
    /// ```
    /// use toon_format::DecodeOptions;
    ///
    /// let opts = DecodeOptions::new().with_coerce_types(false);
    /// let _ = opts;
    /// ```
    pub fn with_coerce_types(mut self, coerce: bool) -> Self {
        self.coerce_types = coerce;
        self
    }

    /// Set the indentation style for decode operations that require it.
    ///
    /// # Examples
    /// ```
    /// use toon_format::{DecodeOptions, Indent};
    ///
    /// let opts = DecodeOptions::new().with_indent(Indent::Spaces(2));
    /// let _ = opts;
    /// ```
    pub fn with_indent(mut self, style: Indent) -> Self {
        self.indent = style;
        self
    }

    /// Enable path expansion (v1.5 feature).
    ///
    /// When set to `Safe`, dotted keys will be expanded into nested objects
    /// if all segments are IdentifierSegments.
    ///
    /// Conflict handling:
    /// - `strict=true`: Errors on conflicts
    /// - `strict=false`: Last-write-wins
    ///
    /// Default: `Off`
    ///
    /// # Examples
    /// ```
    /// use toon_format::DecodeOptions;
    /// use toon_format::types::PathExpansionMode;
    ///
    /// let opts = DecodeOptions::new().with_expand_paths(PathExpansionMode::Safe);
    /// let _ = opts;
    /// ```
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
        assert_eq!(opts.indent, Indent::Spaces(4));

        let opts = EncodeOptions::new().with_indent(Indent::Spaces(2));
        assert_eq!(opts.indent, Indent::Spaces(2));
    }

    #[test]
    fn test_decode_options_coerce_types() {
        let opts = DecodeOptions::new();
        assert!(opts.coerce_types);

        let opts = DecodeOptions::new().with_coerce_types(false);
        assert!(!opts.coerce_types);

        let opts = DecodeOptions::new().with_coerce_types(true);
        assert!(opts.coerce_types);
    }
}
