use crate::{
    constants::DEFAULT_INDENT,
    types::Delimiter,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Indent {
    Spaces(usize),
    Tabs,
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
            Indent::Tabs => "\t".repeat(depth),
        }
    }
}

/// Options for encoding JSON values to TOON format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeOptions {
    pub delimiter: Delimiter,
    pub length_marker: Option<char>,
    pub indent: Indent,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            delimiter: Delimiter::Comma,
            length_marker: None,
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

    /// Set a character prefix for array length markers (e.g., `#` for `[#3]`).
    pub fn with_length_marker(mut self, marker: char) -> Self {
        self.length_marker = Some(marker);
        self
    }

    /// Set the indentation string for nested structures.
    pub fn with_indent(mut self, style: Indent) -> Self {
        self.indent = style;
        self
    }

    /// Format an array length with optional marker prefix.
    pub fn format_length(&self, length: usize) -> String {
        if let Some(marker) = self.length_marker {
            format!("{}{}", marker, length)
        } else {
            length.to_string()
        }
    }

    /// Set indentation to a specific number of spaces.
    pub fn with_spaces(mut self, count: usize) -> Self {
        self.indent = Indent::Spaces(count);
        self
    }

    /// Set indentation to tabs.
    pub fn with_tabs(mut self) -> Self {
        self.indent = Indent::Tabs;
        self
    }
}

/// Options for decoding TOON format to JSON values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeOptions {
    pub delimiter: Option<Delimiter>,
    pub strict: bool,
    pub coerce_types: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            delimiter: None,
            strict: true,
            coerce_types: true,
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

    /// Set the expected delimiter (auto-detected if None).
    pub fn with_delimiter(mut self, delimiter: Delimiter) -> Self {
        self.delimiter = Some(delimiter);
        self
    }

    /// Enable or disable type coercion (strings like "123" -> numbers).
    pub fn with_coerce_types(mut self, coerce: bool) -> Self {
        self.coerce_types = coerce;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_encode_options_length_marker() {
        let opts = EncodeOptions::new().with_length_marker('#');
        assert_eq!(opts.format_length(5), "#5");

        let opts = EncodeOptions::new();
        assert_eq!(opts.format_length(5), "5");
    }

    #[test]
    fn test_encode_options_indent() {
        let opts = EncodeOptions::new().with_spaces(4);
        assert_eq!(opts.indent, Indent::Spaces(4));

        let opts = EncodeOptions::new().with_tabs();
        assert_eq!(opts.indent, Indent::Tabs);

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
