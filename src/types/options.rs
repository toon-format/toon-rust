use crate::{
    constants::DEFAULT_INDENT,
    types::Delimiter,
};

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

    pub fn get_spaces(&self) -> usize {
        match self {
            Indent::Spaces(count) => *count,
        }
    }
}

/// Options for encoding JSON values to TOON format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeOptions {
    pub delimiter: Delimiter,
    pub indent: Indent,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            delimiter: Delimiter::Comma,
            indent: Indent::default(),
        }
    }
}

impl EncodeOptions {
    /// Create new encoding options with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the delimiter for array elements.
    pub fn with_delimiter(mut self, delimiter: Delimiter) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set the indentation string for nested structures.
    pub fn with_indent(mut self, style: Indent) -> Self {
        self.indent = style;
        self
    }

    /// Set indentation to a specific number of spaces.
    pub fn with_spaces(mut self, count: usize) -> Self {
        self.indent = Indent::Spaces(count);
        self
    }
}

/// Options for decoding TOON format to JSON values (§13: `strict` and
/// `indentSize`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeOptions {
    pub strict: bool,
    pub indent: Indent,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            strict: true,
            indent: Indent::default(),
        }
    }
}

impl DecodeOptions {
    /// Create new decoding options with defaults (strict mode enabled).
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable strict mode (validates array lengths, indentation,
    /// etc.).
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Set the number of spaces per indentation level.
    pub fn with_indent(mut self, style: Indent) -> Self {
        self.indent = style;
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
    fn test_decode_options_defaults() {
        let opts = DecodeOptions::new();
        assert!(opts.strict);
        assert_eq!(opts.indent, Indent::Spaces(2));

        let opts = DecodeOptions::new().with_strict(false);
        assert!(!opts.strict);
    }
}
