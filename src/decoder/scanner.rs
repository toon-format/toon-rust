/* rune-xero/src/decoder/scanner.rs */
//!▫~•◦-------------------------------‣
//! # RUNE-Xero – Lexical Scanner Module
//!▫~•◦----------------------------------‣
//!
//! The `Scanner` provides high-performance, zero-copy tokenization of RUNE source text.
//! It avoids all unnecessary allocations by returning borrowed slices (`&'a str`)
//! wherever possible, falling back to `Cow::Owned` only when unescaping is strictly required.
//!
//! ## Key Capabilities
//! - **Zero-Allocation Tokenization:** Tokens borrow directly from the source `input`.
//! - **Conditional Ownership:** Uses `Cow<'a, str>` to handle escaped strings transparently.
//! - **Indentation Sensitive:** Tracks column/line position for significant whitespace logic.
//!
//! ### Architectural Notes
//! This module is the foundational layer of the RUNE-Xero parser pipeline.
//! It depends on `crate::types` for token definitions and error handling.
//!
//! #### Example
//! ```rust
//! use crate::decoder::scanner::Scanner;
//!
//! let input = r#"key: "value""#;
//! let mut scanner = Scanner::new(input);
//! let token = scanner.scan_token().unwrap();
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::types::{Delimiter, RuneError, RuneResult};
use std::borrow::Cow;

/// Tokens produced by the scanner during lexical analysis.
/// Zero-copy variant: holds references to the original source where possible.
#[derive(Debug, Clone, PartialEq)]
pub enum Token<'a> {
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Colon,
    Dash,
    Newline,
    /// Content and is_quoted flag.
    /// Uses Cow to borrow from input unless unescaping occurred.
    String(Cow<'a, str>, bool),
    Number(f64),
    Integer(i64),
    Bool(bool),
    Null,
    Delimiter(Delimiter),
    Eof,
}

/// Scanner that tokenizes RUNE input into a sequence of tokens.
/// Operates on a borrowed string slice without copying the input.
pub struct Scanner<'a> {
    input: &'a str,
    cursor: usize, // Current byte offset in input
    line: usize,
    column: usize,
    active_delimiter: Option<Delimiter>,
    last_line_indent: usize,
}

impl<'a> Scanner<'a> {
    /// Create a new scanner for the given input string.
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            cursor: 0,
            line: 1,
            column: 1,
            active_delimiter: None,
            last_line_indent: 0,
        }
    }

    /// Set the active delimiter for tokenizing array elements.
    pub fn set_active_delimiter(&mut self, delimiter: Option<Delimiter>) {
        self.active_delimiter = delimiter;
    }

    /// Get the current position (line, column).
    pub fn current_position(&self) -> (usize, usize) {
        (self.line, self.column)
    }

    pub fn get_line(&self) -> usize {
        self.line
    }

    pub fn get_column(&self) -> usize {
        self.column
    }

    /// Returns the character at the current cursor position.
    pub fn peek(&self) -> Option<char> {
        self.input[self.cursor..].chars().next()
    }

    /// Count leading spaces from current position without advancing.
    pub fn count_leading_spaces(&self) -> usize {
        let mut chars = self.input[self.cursor..].chars();
        let mut count = 0;
        while let Some(ch) = chars.next() {
            if ch == ' ' {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    pub fn count_spaces_after_newline(&self) -> usize {
        let mut chars = self.input[self.cursor..].chars();
        // Check if current char is newline
        if chars.next() != Some('\n') {
            return 0;
        }
        
        let mut count = 0;
        while let Some(ch) = chars.next() {
            if ch == ' ' {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    pub fn peek_ahead(&self, offset: usize) -> Option<char> {
        self.input[self.cursor..].chars().nth(offset)
    }

    /// Advance the cursor by one character.
    pub fn advance(&mut self) -> Option<char> {
        if let Some(ch) = self.peek() {
            self.cursor += ch.len_utf8();
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            Some(ch)
        } else {
            None
        }
    }

    pub fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch == ' ' {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Scan the next token from the input.
    pub fn scan_token(&mut self) -> RuneResult<Token<'a>> {
        // Handle indentation checking at start of line
        if self.column == 1 {
            let mut count = 0;
            // Use an iterator to check indentation without mutating state yet
            let chars = self.input[self.cursor..].chars();
            
            for ch in chars {
                if ch == ' ' {
                    count += 1;
                } else {
                    if ch == '\t' {
                        let (line, col) = self.current_position();
                        return Err(RuneError::parse_error(
                            line,
                            col + count,
                            "Tabs are not allowed in indentation",
                        ));
                    }
                    break;
                }
            }
            self.last_line_indent = count;
        }

        self.skip_whitespace();

        match self.peek() {
            None => Ok(Token::Eof),
            Some('\n') => {
                self.advance();
                Ok(Token::Newline)
            }
            Some('[') => {
                self.advance();
                Ok(Token::LeftBracket)
            }
            Some(']') => {
                self.advance();
                Ok(Token::RightBracket)
            }
            Some('{') => {
                self.advance();
                Ok(Token::LeftBrace)
            }
            Some('}') => {
                self.advance();
                Ok(Token::RightBrace)
            }
            Some(':') => {
                self.advance();
                Ok(Token::Colon)
            }
            Some('-') => {
                self.advance();
                // Check if next char is digit (negative number)
                if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    let num_slice = self.scan_number_slice(true)?;
                    return self.parse_number(num_slice);
                }
                Ok(Token::Dash)
            }
            Some(',') => {
                if matches!(self.active_delimiter, Some(Delimiter::Comma)) {
                    self.advance();
                    Ok(Token::Delimiter(Delimiter::Comma))
                } else {
                    self.scan_unquoted_string()
                }
            }
            Some('|') => {
                if matches!(self.active_delimiter, Some(Delimiter::Pipe)) {
                    self.advance();
                    Ok(Token::Delimiter(Delimiter::Pipe))
                } else {
                    self.scan_unquoted_string()
                }
            }
            Some('\t') => {
                if matches!(self.active_delimiter, Some(Delimiter::Tab)) {
                    self.advance();
                    Ok(Token::Delimiter(Delimiter::Tab))
                } else {
                    self.scan_unquoted_string()
                }
            }
            Some('"') => self.scan_quoted_string(),
            Some(ch) if ch.is_ascii_digit() => {
                let num_slice = self.scan_number_slice(false)?;
                self.parse_number(num_slice)
            }
            Some(_) => self.scan_unquoted_string(),
        }
    }

    fn scan_quoted_string(&mut self) -> RuneResult<Token<'a>> {
        self.advance(); // consume opening quote

        let start_byte = self.cursor;
        let mut has_escapes = false;
        
        // First pass: scan until closing quote, check for escapes
        while let Some(ch) = self.peek() {
            if ch == '\\' {
                has_escapes = true;
                self.advance(); // skip backslash
                if self.advance().is_none() { // consume escaped char
                     return Err(RuneError::UnexpectedEof);
                }
            } else if ch == '"' {
                break;
            } else {
                self.advance();
            }
        }
        
        let end_byte = self.cursor;
        
        // Ensure we stopped on a quote
        if self.peek() != Some('"') {
             return Err(RuneError::UnexpectedEof);
        }
        self.advance(); // consume closing quote

        let raw_content = &self.input[start_byte..end_byte];

        if !has_escapes {
            // Zero-copy path: return slice directly
            Ok(Token::String(Cow::Borrowed(raw_content), true))
        } else {
            // Allocation path: escapes processing required
            let mut value = String::with_capacity(raw_content.len());
            let mut chars = raw_content.chars();
            
            while let Some(ch) = chars.next() {
                if ch == '\\' {
                    match chars.next() {
                        Some('n') => value.push('\n'),
                        Some('r') => value.push('\r'),
                        Some('t') => value.push('\t'),
                        Some('"') => value.push('"'),
                        Some('\\') => value.push('\\'),
                        Some(c) => {
                             let (line, col) = self.current_position();
                             // Approximation of col for error reporting
                             return Err(RuneError::parse_error(
                                line,
                                col, // Note: precise col tracking inside string is simplified here
                                format!("Invalid escape sequence: \\{c}"),
                            ));
                        }
                        None => return Err(RuneError::UnexpectedEof),
                    }
                } else {
                    value.push(ch);
                }
            }
            Ok(Token::String(Cow::Owned(value), true))
        }
    }

    fn scan_unquoted_string(&mut self) -> RuneResult<Token<'a>> {
        let start_byte = self.cursor;

        while let Some(ch) = self.peek() {
            if ch == '\n'
                || ch == ' '
                || ch == ':'
                || ch == '['
                || ch == ']'
                || ch == '{'
                || ch == '}'
            {
                break;
            }

            if matches!(
                (self.active_delimiter, ch),
                (Some(Delimiter::Comma), ',')
                    | (Some(Delimiter::Pipe), '|')
                    | (Some(Delimiter::Tab), '\t')
            ) {
                break;
            }
            self.advance();
        }

        let end_byte = self.cursor;
        let raw_slice = &self.input[start_byte..end_byte];
        
        // Optimization: single char delimiters kept as-is
        let value_slice = if raw_slice.len() == 1 && (raw_slice == "," || raw_slice == "|" || raw_slice == "\t") {
            raw_slice
        } else {
            raw_slice.trim_end()
        };

        match value_slice {
            "null" => Ok(Token::Null),
            "true" => Ok(Token::Bool(true)),
            "false" => Ok(Token::Bool(false)),
            _ => Ok(Token::String(Cow::Borrowed(value_slice), false)),
        }
    }

    pub fn get_last_line_indent(&self) -> usize {
        self.last_line_indent
    }

    /// Returns the slice corresponding to the number string.
    fn scan_number_slice(&mut self, negative: bool) -> RuneResult<&'a str> {
        let start_byte = if negative {
             // We advanced past '-' already, but we need to include it in the slice for parsing.
             // We need to look back. 
             self.cursor - 1 
        } else {
            self.cursor
        };

        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() || ch == '.' || ch == 'e' || ch == 'E' || ch == '+' || ch == '-' {
                self.advance();
            } else {
                break;
            }
        }
        
        Ok(&self.input[start_byte..self.cursor])
    }

    fn parse_number(&self, s: &'a str) -> RuneResult<Token<'a>> {
        // Number followed immediately by other chars like "0(f)" should be a string
        if let Some(next_ch) = self.peek() {
             if next_ch != ' '
                && next_ch != '\n'
                && next_ch != ':'
                && next_ch != '['
                && next_ch != ']'
                && next_ch != '{'
                && next_ch != '}'
                && !matches!(
                    (self.active_delimiter, next_ch),
                    (Some(Delimiter::Comma), ',')
                        | (Some(Delimiter::Pipe), '|')
                        | (Some(Delimiter::Tab), '\t')
                )
            {
                return Ok(Token::String(Cow::Borrowed(s), false));
            }
        }

        // Leading zeros like "05" are strings, but "0", "0.5", "-0" are numbers
        if s.starts_with('0') && s.len() > 1 {
            let second_char = s.as_bytes()[1] as char;
            if second_char.is_ascii_digit() {
                return Ok(Token::String(Cow::Borrowed(s), false));
            }
        }
        
        // Same check for "-05"
        if s.starts_with("-0") && s.len() > 2 {
             let third_char = s.as_bytes()[2] as char;
             if third_char.is_ascii_digit() {
                 return Ok(Token::String(Cow::Borrowed(s), false));
             }
        }

        if s.contains('.') || s.contains('e') || s.contains('E') {
            if let Ok(f) = s.parse::<f64>() {
                Ok(Token::Number(f))
            } else {
                Ok(Token::String(Cow::Borrowed(s), false))
            }
        } else if let Ok(i) = s.parse::<i64>() {
            Ok(Token::Integer(i))
        } else {
            Ok(Token::String(Cow::Borrowed(s), false))
        }
    }

    /// Read the rest of the current line (until newline or EOF).
    /// Returns the content slice and leading space flag.
    pub fn read_rest_of_line_with_space_info(&mut self) -> (Cow<'a, str>, bool) {
        let had_leading_space = matches!(self.peek(), Some(' '));
        self.skip_whitespace();

        let start_byte = self.cursor;
        
        while let Some(ch) = self.peek() {
            if ch == '\n' {
                break;
            }
            self.advance();
        }
        
        let end_byte = self.cursor;
        let slice = &self.input[start_byte..end_byte];
        let trimmed = slice.trim_end();
        
        (Cow::Borrowed(trimmed), had_leading_space)
    }

    /// Read the rest of the current line.
    pub fn read_rest_of_line(&mut self) -> Cow<'a, str> {
        self.read_rest_of_line_with_space_info().0
    }

    /// Utility to parse a value string (e.g., from a test or substring)
    /// This now requires the input 's' to live as long as the Token.
    pub fn parse_value_string<'b>(&self, s: &'b str) -> RuneResult<Token<'b>> {
        let trimmed = s.trim();

        if trimmed.is_empty() {
            return Ok(Token::String(Cow::Borrowed(""), false));
        }

        if trimmed.starts_with('"') {
            // Need to handle escapes here too.
            // Check if we can borrow (no escapes) or must own.
            // This mirrors scan_quoted_string logic but for an isolated string.
            
            // Basic validation
             if !trimmed.ends_with('"') || trimmed.len() < 2 {
                 return Err(RuneError::parse_error(self.line, self.column, "Unterminated string"));
             }
             
             let content = &trimmed[1..trimmed.len()-1];
             if content.contains('\\') {
                 // Must allocate to unescape
                 let mut value = String::with_capacity(content.len());
                 let mut chars = content.chars();
                 while let Some(ch) = chars.next() {
                     if ch == '\\' {
                         match chars.next() {
                             Some('n') => value.push('\n'),
                             Some('r') => value.push('\r'),
                             Some('t') => value.push('\t'),
                             Some('"') => value.push('"'),
                             Some('\\') => value.push('\\'),
                             _ => return Err(RuneError::parse_error(self.line, self.column, "Invalid escape")),
                         }
                     } else {
                         value.push(ch);
                     }
                 }
                 return Ok(Token::String(Cow::Owned(value), true));
             } else {
                 return Ok(Token::String(Cow::Borrowed(content), true));
             }
        }

        match trimmed {
            "true" => return Ok(Token::Bool(true)),
            "false" => return Ok(Token::Bool(false)),
            "null" => return Ok(Token::Null),
            _ => {}
        }
        
        // Number parsing logic
        if trimmed.starts_with('-') || trimmed.starts_with(|c: char| c.is_ascii_digit()) {
             if trimmed.starts_with('0') && trimmed.len() > 1 && trimmed.as_bytes()[1].is_ascii_digit() {
                 return Ok(Token::String(Cow::Borrowed(trimmed), false));
             }
             
             if trimmed.contains('.') || trimmed.contains('e') || trimmed.contains('E') {
                if let Ok(f) = trimmed.parse::<f64>() {
                    let normalized = if f == -0.0 { 0.0 } else { f };
                    return Ok(Token::Number(normalized));
                }
            } else if let Ok(i) = trimmed.parse::<i64>() {
                return Ok(Token::Integer(i));
            }
        }
        
        Ok(Token::String(Cow::Borrowed(trimmed), false))
    }

    pub fn detect_delimiter(&mut self) -> Option<Delimiter> {
        let saved_cursor = self.cursor;
        let saved_line = self.line;
        let saved_col = self.column;

        let mut delim = None;
        while let Some(ch) = self.peek() {
            match ch {
                ',' => { delim = Some(Delimiter::Comma); break; }
                '|' => { delim = Some(Delimiter::Pipe); break; }
                '\t' => { delim = Some(Delimiter::Tab); break; }
                '\n' | ':' | '[' | ']' | '{' | '}' => break,
                _ => {
                    self.advance();
                }
            }
        }

        // Restore state
        self.cursor = saved_cursor;
        self.line = saved_line;
        self.column = saved_col;
        
        delim
    }
}

#[cfg(test)]
mod tests {
    use core::f64;
    use super::*;

    #[test]
    fn test_scan_structural_tokens() {
        let mut scanner = Scanner::new("[]{}:-");
        assert_eq!(scanner.scan_token().unwrap(), Token::LeftBracket);
        assert_eq!(scanner.scan_token().unwrap(), Token::RightBracket);
        assert_eq!(scanner.scan_token().unwrap(), Token::LeftBrace);
        assert_eq!(scanner.scan_token().unwrap(), Token::RightBrace);
        assert_eq!(scanner.scan_token().unwrap(), Token::Colon);
        assert_eq!(scanner.scan_token().unwrap(), Token::Dash);
    }

    #[test]
    fn test_scan_numbers() {
        let mut scanner = Scanner::new("42 3.141592653589793 -5");
        assert_eq!(scanner.scan_token().unwrap(), Token::Integer(42));
        assert_eq!(
            scanner.scan_token().unwrap(),
            Token::Number(f64::consts::PI)
        );
        assert_eq!(scanner.scan_token().unwrap(), Token::Integer(-5));
    }

    #[test]
    fn test_scan_quoted_string() {
        let mut scanner = Scanner::new(r#""hello world""#);
        // Zero-copy: matches borrowed content
        match scanner.scan_token().unwrap() {
            Token::String(cow, quoted) => {
                assert_eq!(cow, "hello world");
                assert!(matches!(cow, Cow::Borrowed(_)));
                assert!(quoted);
            }
            _ => panic!("Expected string"),
        }
    }

    #[test]
    fn test_scan_escaped_string() {
        let mut scanner = Scanner::new(r#""hello\nworld""#);
        // Allocation: matches owned content
        match scanner.scan_token().unwrap() {
            Token::String(cow, quoted) => {
                assert_eq!(cow, "hello\nworld");
                assert!(matches!(cow, Cow::Owned(_))); // Escaped needs alloc
                assert!(quoted);
            }
            _ => panic!("Expected string"),
        }
    }

    #[test]
    fn test_read_rest_of_line_with_space_info() {
        let mut scanner = Scanner::new(" world");
        let (content, had_space) = scanner.read_rest_of_line_with_space_info();
        assert_eq!(content, "world");
        assert!(had_space);
    }
}