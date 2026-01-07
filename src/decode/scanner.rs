use crate::types::{Delimiter, ToonError, ToonResult};

/// Tokens produced by the scanner during lexical analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Colon,
    Dash,
    Newline,
    String(String, bool),
    Number(f64),
    Integer(i64),
    Bool(bool),
    Null,
    Delimiter(Delimiter),
    Eof,
}

/// Scanner that tokenizes TOON input into a sequence of tokens.
pub struct Scanner {
    input: Vec<char>,
    position: usize,
    line: usize,
    column: usize,
    active_delimiter: Option<Delimiter>,
    last_line_indent: usize,
    coerce_types: bool,
}

impl Scanner {
    /// Create a new scanner for the given input string.
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            position: 0,
            line: 1,
            column: 1,
            active_delimiter: None,
            last_line_indent: 0,
            coerce_types: true,
        }
    }

    /// Set the active delimiter for tokenizing array elements.
    pub fn set_active_delimiter(&mut self, delimiter: Option<Delimiter>) {
        self.active_delimiter = delimiter;
    }

    pub fn set_coerce_types(&mut self, coerce_types: bool) {
        self.coerce_types = coerce_types;
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

    pub fn peek(&self) -> Option<char> {
        self.input.get(self.position).copied()
    }

    pub fn count_leading_spaces(&self) -> usize {
        let mut idx = self.position;
        let mut count = 0;
        while let Some(&ch) = self.input.get(idx) {
            if ch == ' ' {
                count += 1;
                idx += 1;
            } else {
                break;
            }
        }
        count
    }

    pub fn count_spaces_after_newline(&self) -> usize {
        let mut idx = self.position;
        if self.input.get(idx) != Some(&'\n') {
            return 0;
        }
        idx += 1;
        let mut count = 0;
        while let Some(&ch) = self.input.get(idx) {
            if ch == ' ' {
                count += 1;
                idx += 1;
            } else {
                break;
            }
        }
        count
    }

    pub fn peek_ahead(&self, offset: usize) -> Option<char> {
        self.input.get(self.position + offset).copied()
    }

    pub fn advance(&mut self) -> Option<char> {
        if let Some(ch) = self.input.get(self.position) {
            self.position += 1;
            if *ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            Some(*ch)
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
    pub fn scan_token(&mut self) -> ToonResult<Token> {
        if self.column == 1 {
            let mut count = 0;
            let mut idx = self.position;

            while let Some(&ch) = self.input.get(idx) {
                if ch == ' ' {
                    count += 1;
                    idx += 1;
                } else {
                    if ch == '\t' {
                        let (line, col) = self.current_position();
                        return Err(ToonError::parse_error(
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
                if let Some(ch) = self.peek() {
                    if ch.is_ascii_digit() {
                        let num_str = self.scan_number_string(true)?;
                        return self.parse_number(&num_str);
                    }
                }
                Ok(Token::Dash)
            }
            Some(',') => {
                // Delimiter only when active, otherwise part of unquoted string
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
                let num_str = self.scan_number_string(false)?;
                self.parse_number(&num_str)
            }
            Some(_) => self.scan_unquoted_string(),
        }
    }

    fn scan_quoted_string(&mut self) -> ToonResult<Token> {
        self.advance();

        let mut value = String::new();
        let mut escaped = false;

        while let Some(ch) = self.advance() {
            if escaped {
                match ch {
                    'n' => value.push('\n'),
                    'r' => value.push('\r'),
                    't' => value.push('\t'),
                    '"' => value.push('"'),
                    '\\' => value.push('\\'),
                    _ => {
                        let (line, col) = self.current_position();
                        return Err(ToonError::parse_error(
                            line,
                            col - 1,
                            format!("Invalid escape sequence: \\{ch}"),
                        ));
                    }
                }
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                return Ok(Token::String(value, true));
            } else {
                value.push(ch);
            }
        }

        Err(ToonError::UnexpectedEof)
    }

    fn scan_unquoted_string(&mut self) -> ToonResult<Token> {
        let mut value = String::new();

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

            // Active delimiters stop the string; otherwise they're part of it
            if let Some(active) = self.active_delimiter {
                if (active == Delimiter::Comma && ch == ',')
                    || (active == Delimiter::Pipe && ch == '|')
                    || (active == Delimiter::Tab && ch == '\t')
                {
                    break;
                }
            }
            value.push(ch);
            self.advance();
        }

        // Single-char delimiters kept as-is, others trimmed
        let value = if value.len() == 1 && (value == "," || value == "|" || value == "\t") {
            value
        } else {
            value.trim_end().to_string()
        };

        if !self.coerce_types {
            return Ok(Token::String(value, false));
        }

        match value.as_str() {
            "null" => Ok(Token::Null),
            "true" => Ok(Token::Bool(true)),
            "false" => Ok(Token::Bool(false)),
            _ => Ok(Token::String(value, false)),
        }
    }

    pub fn get_last_line_indent(&self) -> usize {
        self.last_line_indent
    }

    fn scan_number_string(&mut self, negative: bool) -> ToonResult<String> {
        let mut num_str = if negative {
            String::from("-")
        } else {
            String::new()
        };

        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() || ch == '.' || ch == 'e' || ch == 'E' || ch == '+' || ch == '-'
            {
                num_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        Ok(num_str)
    }

    fn parse_number(&self, s: &str) -> ToonResult<Token> {
        if !self.coerce_types {
            return Ok(Token::String(s.to_string(), false));
        }

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
                return Ok(Token::String(s.to_string(), false));
            }
        }

        // Leading zeros like "05" are strings, but "0", "0.5", "-0" are numbers
        if s.starts_with('0') && s.len() > 1 {
            if let Some(second_char) = s.chars().nth(1) {
                if second_char.is_ascii_digit() {
                    return Ok(Token::String(s.to_string(), false));
                }
            }
        }
        if s.starts_with("-0") && s.len() > 2 {
            if let Some(third_char) = s.chars().nth(2) {
                if third_char.is_ascii_digit() {
                    return Ok(Token::String(s.to_string(), false));
                }
            }
        }

        if s.contains('.') || s.contains('e') || s.contains('E') {
            if let Ok(f) = s.parse::<f64>() {
                Ok(Token::Number(f))
            } else {
                Ok(Token::String(s.to_string(), false))
            }
        } else if let Ok(i) = s.parse::<i64>() {
            Ok(Token::Integer(i))
        } else {
            Ok(Token::String(s.to_string(), false))
        }
    }

    /// Read the rest of the current line (until newline or EOF).
    /// Returns the content and any leading spaces between the current token
    /// and the rest of the line.
    pub fn read_rest_of_line_with_space_info(&mut self) -> (String, String) {
        let mut leading_space = String::new();
        while matches!(self.peek(), Some(' ')) {
            leading_space.push(' ');
            self.advance();
        }

        let mut result = String::new();
        while let Some(ch) = self.peek() {
            if ch == '\n' {
                break;
            }
            result.push(ch);
            self.advance();
        }

        (result.trim_end().to_string(), leading_space)
    }

    /// Read the rest of the current line (until newline or EOF).
    pub fn read_rest_of_line(&mut self) -> String {
        self.read_rest_of_line_with_space_info().0
    }

    /// Parse a complete value string into a token.
    pub fn parse_value_string(&self, s: &str) -> ToonResult<Token> {
        let trimmed = s.trim();

        if trimmed.is_empty() {
            return Ok(Token::String(String::new(), false));
        }

        if trimmed.starts_with('"') {
            let mut value = String::new();
            let mut escaped = false;
            let chars: Vec<char> = trimmed.chars().collect();
            let mut i = 1;

            while i < chars.len() {
                let ch = chars[i];
                if escaped {
                    match ch {
                        'n' => value.push('\n'),
                        'r' => value.push('\r'),
                        't' => value.push('\t'),
                        '"' => value.push('"'),
                        '\\' => value.push('\\'),
                        _ => {
                            return Err(ToonError::parse_error(
                                self.line,
                                self.column,
                                format!("Invalid escape sequence: \\{ch}"),
                            ));
                        }
                    }
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    if i != chars.len() - 1 {
                        return Err(ToonError::parse_error(
                            self.line,
                            self.column,
                            "Unexpected characters after closing quote",
                        ));
                    }
                    return Ok(Token::String(value, true));
                } else {
                    value.push(ch);
                }
                i += 1;
            }

            return Err(ToonError::parse_error(
                self.line,
                self.column,
                "Unterminated string: missing closing quote",
            ));
        }

        if !self.coerce_types {
            return Ok(Token::String(trimmed.to_string(), false));
        }

        match trimmed {
            "true" => return Ok(Token::Bool(true)),
            "false" => return Ok(Token::Bool(false)),
            "null" => return Ok(Token::Null),
            _ => {}
        }

        if trimmed.starts_with('-') || trimmed.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            // Leading zeros like "05" or "-05" are strings
            if trimmed.starts_with('0') && trimmed.len() > 1 {
                if let Some(second_char) = trimmed.chars().nth(1) {
                    if second_char.is_ascii_digit() {
                        return Ok(Token::String(trimmed.to_string(), false));
                    }
                }
            }
            if trimmed.starts_with("-0") && trimmed.len() > 2 {
                if let Some(third_char) = trimmed.chars().nth(2) {
                    if third_char.is_ascii_digit() {
                        return Ok(Token::String(trimmed.to_string(), false));
                    }
                }
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

        Ok(Token::String(trimmed.to_string(), false))
    }

    pub fn detect_delimiter(&mut self) -> Option<Delimiter> {
        let saved_pos = self.position;

        while let Some(ch) = self.peek() {
            match ch {
                ',' => {
                    self.position = saved_pos;
                    return Some(Delimiter::Comma);
                }
                '|' => {
                    self.position = saved_pos;
                    return Some(Delimiter::Pipe);
                }
                '\t' => {
                    self.position = saved_pos;
                    return Some(Delimiter::Tab);
                }
                '\n' | ':' | '[' | ']' | '{' | '}' => break,
                _ => {
                    self.advance();
                }
            }
        }

        self.position = saved_pos;
        None
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
    fn test_scan_booleans() {
        let mut scanner = Scanner::new("true false");
        assert_eq!(scanner.scan_token().unwrap(), Token::Bool(true));
        assert_eq!(scanner.scan_token().unwrap(), Token::Bool(false));
    }

    #[test]
    fn test_scan_null() {
        let mut scanner = Scanner::new("null");
        assert_eq!(scanner.scan_token().unwrap(), Token::Null);
    }

    #[test]
    fn test_scan_quoted_string() {
        let mut scanner = Scanner::new(r#""hello world""#);
        assert_eq!(
            scanner.scan_token().unwrap(),
            Token::String("hello world".to_string(), true)
        );
    }

    #[test]
    fn test_scan_escaped_string() {
        let mut scanner = Scanner::new(r#""hello\nworld""#);
        assert_eq!(
            scanner.scan_token().unwrap(),
            Token::String("hello\nworld".to_string(), true)
        );
    }

    #[test]
    fn test_scan_unquoted_string() {
        let mut scanner = Scanner::new("hello");
        assert_eq!(
            scanner.scan_token().unwrap(),
            Token::String("hello".to_string(), false)
        );
    }

    #[test]
    fn test_detect_delimiter() {
        let mut scanner = Scanner::new("a,b,c");
        assert_eq!(scanner.detect_delimiter(), Some(Delimiter::Comma));

        let mut scanner = Scanner::new("a|b|c");
        assert_eq!(scanner.detect_delimiter(), Some(Delimiter::Pipe));

        let mut scanner = Scanner::new("a\tb\tc");
        assert_eq!(scanner.detect_delimiter(), Some(Delimiter::Tab));
    }

    #[test]
    fn test_read_rest_of_line_with_space_info() {
        let mut scanner = Scanner::new(" world");
        let (content, leading_space) = scanner.read_rest_of_line_with_space_info();
        assert_eq!(content, "world");
        assert_eq!(leading_space, " ");

        let mut scanner = Scanner::new("world");
        let (content, leading_space) = scanner.read_rest_of_line_with_space_info();
        assert_eq!(content, "world");
        assert!(leading_space.is_empty());

        let mut scanner = Scanner::new("(hello)");
        let (content, leading_space) = scanner.read_rest_of_line_with_space_info();
        assert_eq!(content, "(hello)");
        assert!(leading_space.is_empty());

        let mut scanner = Scanner::new("");
        let (content, leading_space) = scanner.read_rest_of_line_with_space_info();
        assert_eq!(content, "");
        assert!(leading_space.is_empty());
    }

    #[test]
    fn test_parse_value_string() {
        let scanner = Scanner::new("");
        assert_eq!(
            scanner.parse_value_string("hello").unwrap(),
            Token::String("hello".to_string(), false)
        );

        assert_eq!(
            scanner.parse_value_string("(hello)").unwrap(),
            Token::String("(hello)".to_string(), false)
        );

        assert_eq!(
            scanner
                .parse_value_string("Mostly Functions (3 of 3)")
                .unwrap(),
            Token::String("Mostly Functions (3 of 3)".to_string(), false)
        );
        assert_eq!(
            scanner.parse_value_string("0(f)").unwrap(),
            Token::String("0(f)".to_string(), false)
        );

        assert_eq!(
            scanner.parse_value_string("42").unwrap(),
            Token::Integer(42)
        );

        assert_eq!(
            scanner.parse_value_string("true").unwrap(),
            Token::Bool(true)
        );
        assert_eq!(
            scanner.parse_value_string("false").unwrap(),
            Token::Bool(false)
        );
        assert_eq!(scanner.parse_value_string("null").unwrap(), Token::Null);

        assert_eq!(
            scanner.parse_value_string(r#""hello world""#).unwrap(),
            Token::String("hello world".to_string(), true)
        );
    }

    #[test]
    fn test_number_followed_by_parenthesis() {
        let mut scanner = Scanner::new("0(f)");
        let num_token = scanner.scan_number_string(false).unwrap();
        let token = scanner.parse_number(&num_token).unwrap();

        assert_eq!(token, Token::String("0".to_string(), false));
    }
}
