/* rune-xero/src/types/errors.rs */
//!▫~•◦-----------------------------‣
//! # RUNE-Xero – Error Types
//!▫~•◦-----------------------------‣
//!
//! Error definitions for encoding and decoding. Uses owned `String` to avoid
//! lifetime gymnastics while keeping zero-copy data paths untouched.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use yoshi::AnyError;

/// Result type alias for RUNE operations.
pub type RuneResult<T> = std::result::Result<T, RuneError>;

/// Errors that can occur during RUNE encoding or decoding.
#[derive(AnyError, Debug, Clone, PartialEq)]
pub enum RuneError {
    #[anyerror("Invalid input: {0}")]
    InvalidInput(String),

    #[anyerror("Parse error at line {line}, column {column}: {message}")]
    ParseError {
        line: usize,
        column: usize,
        message: String,
        context: Option<Box<ErrorContext>>,
    },

    #[anyerror("Invalid character '{char}' at position {position}")]
    InvalidCharacter { char: char, position: usize },

    #[anyerror("Unexpected end of input")]
    UnexpectedEof,

    #[anyerror("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    #[anyerror("Invalid delimiter: {0}")]
    InvalidDelimiter(String),

    #[anyerror("Array length mismatch: expected {expected}, found {found}")]
    LengthMismatch {
        expected: usize,
        found: usize,
        context: Option<Box<ErrorContext>>,
    },

    #[anyerror("Invalid structure: {0}")]
    InvalidStructure(String),

    #[anyerror("Serialization error: {0}")]
    SerializationError(String),

    #[anyerror("Deserialization error: {0}")]
    DeserializationError(String),

    // Mapping for io::Error which is always owned
    #[anyerror("IO error: {0}")]
    IoError(String),
}

/// Contextual information for error reporting, including source location
/// and suggestions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorContext {
    pub source_line: String,
    pub preceding_lines: Vec<String>,
    pub following_lines: Vec<String>,
    pub suggestion: Option<String>,
    pub indicator: Option<String>,
}

impl std::fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\nContext:")?;

        for line in &self.preceding_lines {
            writeln!(f, "  {line}")?;
        }

        writeln!(f, "> {}", self.source_line)?;

        if let Some(indicator) = &self.indicator {
            writeln!(f, "  {indicator}")?;
        }

        for line in &self.following_lines {
            writeln!(f, "  {line}")?;
        }

        if let Some(suggestion) = &self.suggestion {
            writeln!(f, "\nSuggestion: {suggestion}")?;
        }

        Ok(())
    }
}

impl ErrorContext {
    /// Create a new error context with a source line.
    pub fn new(source_line: impl Into<String>) -> Self {
        Self {
            source_line: source_line.into(),
            preceding_lines: Vec::new(),
            following_lines: Vec::new(),
            suggestion: None,
            indicator: None,
        }
    }

    /// Add preceding context lines.
    pub fn with_preceding_lines(mut self, lines: Vec<String>) -> Self {
        self.preceding_lines = lines;
        self
    }

    /// Add a suggestion message.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Add a column indicator (caret).
    pub fn with_indicator(mut self, column: usize) -> Self {
        let indicator = format!("{}^", " ".repeat(column));
        self.indicator = Some(indicator);
        self
    }

    /// Extract error context from input string.
    pub fn from_input(input: &str, line: usize, column: usize, context_lines: usize) -> Option<Self> {
        let lines: Vec<&str> = input.lines().collect();

        if line == 0 || line > lines.len() {
            return None;
        }

        let line_idx = line - 1;
        let source_line = lines.get(line_idx)?.to_string();

        let start_line = line_idx.saturating_sub(context_lines);
        let end_line = (line_idx + context_lines + 1).min(lines.len());

        let preceding_lines = lines[start_line..line_idx]
            .iter()
            .map(|&s| s.to_string())
            .collect();

        let following_lines = lines[(line_idx + 1)..end_line]
            .iter()
            .map(|&s| s.to_string())
            .collect();

        Some(Self {
            source_line,
            preceding_lines,
            following_lines,
            suggestion: None,
            indicator: Some(" ".repeat(column.saturating_sub(1)) + "^"),
        })
    }
}

impl RuneError {
    pub fn parse_error(line: usize, column: usize, message: impl Into<String>) -> Self {
        RuneError::ParseError {
            line,
            column,
            message: message.into(),
            context: None,
        }
    }

    pub fn parse_error_with_context(
        line: usize,
        column: usize,
        message: impl Into<String>,
        context: ErrorContext,
    ) -> Self {
        RuneError::ParseError {
            line,
            column,
            message: message.into(),
            context: Some(Box::new(context)),
        }
    }

    pub fn type_mismatch(expected: impl Into<String>, found: impl Into<String>) -> Self {
        RuneError::TypeMismatch {
            expected: expected.into(),
            found: found.into(),
        }
    }
}
