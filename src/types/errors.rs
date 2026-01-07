use std::sync::Arc;
use thiserror::Error;

/// Result type alias for TOON operations.
pub type ToonResult<T> = std::result::Result<T, ToonError>;

/// Errors that can occur during TOON encoding or decoding.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ToonError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Parse error at line {line}, column {column}: {message}")]
    ParseError {
        line: usize,
        column: usize,
        message: String,
        #[source]
        context: Option<Box<ErrorContext>>,
    },

    #[error("Invalid character '{char}' at position {position}")]
    InvalidCharacter { char: char, position: usize },

    #[error("Unexpected end of input")]
    UnexpectedEof,

    #[error("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    #[error("Invalid delimiter: {0}")]
    InvalidDelimiter(String),

    #[error("Array length mismatch: expected {expected}, found {found}")]
    LengthMismatch {
        expected: usize,
        found: usize,
        #[source]
        context: Option<Box<ErrorContext>>,
    },

    #[error("Invalid structure: {0}")]
    InvalidStructure(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),
}

/// Contextual information for error reporting, including source location
/// and suggestions.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ErrorContextSource {
    Inline {
        source_line: String,
        preceding_lines: Vec<String>,
        following_lines: Vec<String>,
        indicator: Option<String>,
    },
    Lazy {
        input: Arc<str>,
        line: usize,
        column: usize,
        context_lines: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorContext {
    source: ErrorContextSource,
    pub suggestion: Option<String>,
}

impl std::fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\nContext:")?;

        match &self.source {
            ErrorContextSource::Inline {
                source_line,
                preceding_lines,
                following_lines,
                indicator,
            } => {
                for line in preceding_lines {
                    writeln!(f, "  {line}")?;
                }

                writeln!(f, "> {source_line}")?;

                if let Some(indicator) = indicator {
                    writeln!(f, "  {indicator}")?;
                }

                for line in following_lines {
                    writeln!(f, "  {line}")?;
                }
            }
            ErrorContextSource::Lazy {
                input,
                line,
                column,
                context_lines,
            } => {
                let lines: Vec<&str> = input.lines().collect();
                if *line == 0 || *line > lines.len() {
                    return Ok(());
                }

                let line_idx = line - 1;
                let start_line = line_idx.saturating_sub(*context_lines);
                let end_line = (line_idx + context_lines + 1).min(lines.len());

                for line in &lines[start_line..line_idx] {
                    writeln!(f, "  {line}")?;
                }

                writeln!(f, "> {}", lines[line_idx])?;

                if *column > 0 {
                    let indicator = " ".repeat(column.saturating_sub(1));
                    writeln!(f, "  {indicator}^")?;
                }

                for line in &lines[(line_idx + 1)..end_line] {
                    writeln!(f, "  {line}")?;
                }
            }
        }

        if let Some(suggestion) = &self.suggestion {
            writeln!(f, "\nSuggestion: {suggestion}")?;
        }

        Ok(())
    }
}

impl std::error::Error for ErrorContext {}

fn extract_lines_to_strings(
    input: &str,
    line: usize,
    context_lines: usize,
) -> (String, Vec<String>, Vec<String>) {
    let lines: Vec<&str> = input.lines().collect();

    if line == 0 || line > lines.len() {
        return (String::new(), Vec::new(), Vec::new());
    }

    let line_idx = line - 1;
    let source_line = lines.get(line_idx).unwrap_or(&"").to_string();

    let start_line = line_idx.saturating_sub(context_lines);
    let end_line = (line_idx + context_lines + 1).min(lines.len());

    let preceding_lines = lines[start_line..line_idx]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let following_lines = lines[(line_idx + 1)..end_line]
        .iter()
        .map(|s| s.to_string())
        .collect();

    (source_line, preceding_lines, following_lines)
}

impl ErrorContext {
    /// Create a new error context with a source line.
    pub fn new(source_line: impl Into<String>) -> Self {
        Self {
            source: ErrorContextSource::Inline {
                source_line: source_line.into(),
                preceding_lines: Vec::new(),
                following_lines: Vec::new(),
                indicator: None,
            },
            suggestion: None,
        }
    }

    /// Add preceding context lines.
    pub fn with_preceding_lines(mut self, lines: Vec<String>) -> Self {
        self.ensure_inline();
        if let ErrorContextSource::Inline {
            preceding_lines, ..
        } = &mut self.source
        {
            *preceding_lines = lines;
        }
        self
    }

    /// Add following context lines.
    pub fn with_following_lines(mut self, lines: Vec<String>) -> Self {
        self.ensure_inline();
        if let ErrorContextSource::Inline {
            following_lines, ..
        } = &mut self.source
        {
            *following_lines = lines;
        }
        self
    }

    /// Add a suggestion message to help fix the error.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Add a column indicator (caret) pointing to the error position.
    pub fn with_indicator(mut self, column: usize) -> Self {
        self.ensure_inline();
        if let ErrorContextSource::Inline { indicator, .. } = &mut self.source {
            let indicator_value = format!("{}^", " ".repeat(column.saturating_sub(1)));
            *indicator = Some(indicator_value);
        }
        self
    }

    fn ensure_inline(&mut self) {
        if let ErrorContextSource::Lazy {
            input,
            line,
            column,
            context_lines,
        } = &self.source
        {
            let (source_line, preceding_lines, following_lines) =
                extract_lines_to_strings(input, *line, *context_lines);
            let indicator = if *column > 0 {
                Some(format!("{}^", " ".repeat(column.saturating_sub(1))))
            } else {
                None
            };

            self.source = ErrorContextSource::Inline {
                source_line,
                preceding_lines,
                following_lines,
                indicator,
            };
        }
    }

    /// Create error context from a shared input buffer with lazy extraction.
    pub fn from_shared_input(
        input: Arc<str>,
        line: usize,
        column: usize,
        context_lines: usize,
    ) -> Option<Self> {
        let line_count = input.lines().count();
        if line == 0 || line > line_count {
            return None;
        }

        Some(Self {
            source: ErrorContextSource::Lazy {
                input,
                line,
                column,
                context_lines,
            },
            suggestion: None,
        })
    }

    /// Create error context from input string with automatic context
    /// extraction.
    pub fn from_input(
        input: &str,
        line: usize,
        column: usize,
        context_lines: usize,
    ) -> Option<Self> {
        Self::from_shared_input(Arc::from(input), line, column, context_lines)
    }
}

impl ToonError {
    /// Create a parse error at the given position.
    pub fn parse_error(line: usize, column: usize, message: impl Into<String>) -> Self {
        ToonError::ParseError {
            line,
            column,
            message: message.into(),
            context: None,
        }
    }

    /// Create a parse error with additional context information.
    pub fn parse_error_with_context(
        line: usize,
        column: usize,
        message: impl Into<String>,
        context: ErrorContext,
    ) -> Self {
        ToonError::ParseError {
            line,
            column,
            message: message.into(),
            context: Some(Box::new(context)),
        }
    }

    /// Create an error for an invalid character.
    pub fn invalid_char(char: char, position: usize) -> Self {
        ToonError::InvalidCharacter { char, position }
    }

    /// Create an error for a type mismatch.
    pub fn type_mismatch(expected: impl Into<String>, found: impl Into<String>) -> Self {
        ToonError::TypeMismatch {
            expected: expected.into(),
            found: found.into(),
        }
    }

    /// Create an error for array length mismatch.
    pub fn length_mismatch(expected: usize, found: usize) -> Self {
        ToonError::LengthMismatch {
            expected,
            found,
            context: None,
        }
    }

    /// Create an array length mismatch error with context.
    pub fn length_mismatch_with_context(
        expected: usize,
        found: usize,
        context: ErrorContext,
    ) -> Self {
        ToonError::LengthMismatch {
            expected,
            found,
            context: Some(Box::new(context)),
        }
    }

    /// Add context to an error if it supports it.
    pub fn with_context(self, context: ErrorContext) -> Self {
        match self {
            ToonError::ParseError {
                line,
                column,
                message,
                ..
            } => ToonError::ParseError {
                line,
                column,
                message,
                context: Some(Box::new(context)),
            },
            ToonError::LengthMismatch {
                expected, found, ..
            } => ToonError::LengthMismatch {
                expected,
                found,
                context: Some(Box::new(context)),
            },
            other => other,
        }
    }

    /// Add a suggestion to help fix the error.
    pub fn with_suggestion(self, suggestion: impl Into<String>) -> Self {
        let suggestion = suggestion.into();
        match self {
            ToonError::ParseError {
                line,
                column,
                message,
                context,
            } => {
                let new_context = context
                    .map(|c| Box::new(c.with_suggestion(suggestion.clone())))
                    .or_else(|| Some(Box::new(ErrorContext::new("").with_suggestion(suggestion))));
                ToonError::ParseError {
                    line,
                    column,
                    message,
                    context: new_context,
                }
            }
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_creation() {
        let ctx = ErrorContext::new("test line")
            .with_suggestion("Try using quotes")
            .with_indicator(5);

        match &ctx.source {
            ErrorContextSource::Inline {
                source_line,
                indicator,
                ..
            } => {
                assert_eq!(source_line, "test line");
                assert!(indicator.is_some());
            }
            ErrorContextSource::Lazy { .. } => panic!("Expected inline context"),
        }
        assert_eq!(ctx.suggestion, Some("Try using quotes".to_string()));
    }

    #[test]
    fn test_error_context_from_input() {
        let input = "line 1\nline 2 with error\nline 3";
        let ctx = ErrorContext::from_input(input, 2, 6, 1);

        assert!(ctx.is_some());
        let ctx = ctx.unwrap();
        match &ctx.source {
            ErrorContextSource::Lazy {
                line,
                column,
                context_lines,
                ..
            } => {
                assert_eq!(*line, 2);
                assert_eq!(*column, 6);
                assert_eq!(*context_lines, 1);
            }
            ErrorContextSource::Inline { .. } => panic!("Expected lazy context"),
        }

        let rendered = format!("{ctx}");
        assert!(rendered.contains("line 2 with error"));
        assert!(rendered.contains("line 1"));
        assert!(rendered.contains("line 3"));
    }

    #[test]
    fn test_parse_error_with_context() {
        let ctx =
            ErrorContext::new("invalid: value").with_suggestion("Did you mean 'value: invalid'?");

        let err = ToonError::parse_error_with_context(1, 8, "Unexpected token", ctx);

        match err {
            ToonError::ParseError {
                line,
                column,
                message,
                context,
            } => {
                assert_eq!(line, 1);
                assert_eq!(column, 8);
                assert_eq!(message, "Unexpected token");
                assert!(context.is_some());
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_error_with_suggestion() {
        let err = ToonError::parse_error(1, 5, "Invalid syntax")
            .with_suggestion("Use quotes around string values");

        match err {
            ToonError::ParseError { context, .. } => {
                assert!(context.is_some());
                let ctx = context.unwrap();
                assert_eq!(
                    ctx.suggestion,
                    Some("Use quotes around string values".to_string())
                );
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_length_mismatch_with_context() {
        let ctx = ErrorContext::new("items[3]: a,b").with_suggestion(
            "Expected 3 items but found 2. Add another item or fix the length marker.",
        );

        let err = ToonError::length_mismatch_with_context(3, 2, ctx);

        match err {
            ToonError::LengthMismatch {
                expected,
                found,
                context,
            } => {
                assert_eq!(expected, 3);
                assert_eq!(found, 2);
                assert!(context.is_some());
            }
            _ => panic!("Wrong error type"),
        }
    }
}
