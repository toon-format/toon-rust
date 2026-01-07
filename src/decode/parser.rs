use serde_json::{Map, Number, Value};
use std::sync::Arc;

use crate::{
    constants::{KEYWORDS, MAX_DEPTH, QUOTED_KEY_MARKER},
    decode::{
        scanner::{Scanner, Token},
        validation,
    },
    types::{DecodeOptions, Delimiter, ErrorContext, PathExpansionMode, ToonError, ToonResult},
    utils::{is_valid_unquoted_key, validation::validate_depth},
};

/// Context for parsing arrays to determine correct indentation depth.
///
/// Arrays as the first field of list-item objects require special indentation:
/// their content (rows for tabular, items for non-uniform) appears at depth +2
/// relative to the hyphen line, while arrays in other contexts use depth +1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrayParseContext {
    /// Normal array parsing context (content at depth +1)
    Normal,

    /// Array as first field of list-item object
    /// (content at depth +2 relative to hyphen line)
    ListItemFirstField,
}

/// Parser that builds JSON values from a sequence of tokens.
#[allow(unused)]
pub struct Parser {
    scanner: Scanner,
    current_token: Token,
    options: DecodeOptions,
    delimiter: Option<Delimiter>,
    delimiter_stack: Vec<Option<Delimiter>>,
    input: Arc<str>,
}

impl Parser {
    /// Create a new parser with the given input and options.
    pub fn new(input: &str, options: DecodeOptions) -> ToonResult<Self> {
        let input: Arc<str> = Arc::from(input);
        let mut scanner = Scanner::from_shared_input(input.clone());
        let chosen_delim = options.delimiter;
        scanner.set_active_delimiter(chosen_delim);
        scanner.set_coerce_types(options.coerce_types);
        scanner.configure_indentation(options.strict, options.indent.get_spaces());
        let current_token = scanner.scan_token()?;

        Ok(Self {
            scanner,
            current_token,
            delimiter: chosen_delim,
            delimiter_stack: Vec::new(),
            options,
            input,
        })
    }

    /// Parse the input into a JSON value.
    pub fn parse(&mut self) -> ToonResult<Value> {
        if self.options.strict {
            self.validate_indentation(self.scanner.get_last_line_indent())?;
        }
        let value = self.parse_value()?;

        // In strict mode, check for trailing content at root level
        if self.options.strict {
            self.skip_newlines()?;
            if !matches!(self.current_token, Token::Eof) {
                return Err(self
                    .parse_error_with_context(
                        "Multiple values at root level are not allowed in strict mode",
                    )
                    .with_suggestion("Wrap multiple values in an object or array"));
            }
        }

        Ok(value)
    }

    fn advance(&mut self) -> ToonResult<()> {
        self.current_token = self.scanner.scan_token()?;
        Ok(())
    }

    fn skip_newlines(&mut self) -> ToonResult<()> {
        while matches!(self.current_token, Token::Newline) {
            self.advance()?;
        }
        Ok(())
    }

    fn push_delimiter(&mut self, delimiter: Option<Delimiter>) {
        self.delimiter_stack.push(self.delimiter);
        self.delimiter = delimiter;
        self.scanner.set_active_delimiter(delimiter);
    }

    fn pop_delimiter(&mut self) {
        if let Some(previous) = self.delimiter_stack.pop() {
            self.delimiter = previous;
            self.scanner.set_active_delimiter(previous);
            if let (Some(delim), Token::String(value, was_quoted)) = (previous, &self.current_token)
            {
                if !*was_quoted && value.len() == 1 && value.starts_with(delim.as_char()) {
                    self.current_token = Token::Delimiter(delim);
                }
            }
        }
    }

    fn format_key(&self, key: &str, was_quoted: bool) -> String {
        if was_quoted && key.contains('.') {
            format!("{QUOTED_KEY_MARKER}{key}")
        } else {
            key.to_string()
        }
    }

    fn validate_unquoted_key(&self, key: &str, was_quoted: bool) -> ToonResult<()> {
        if self.options.strict && !was_quoted {
            if self.options.expand_paths != PathExpansionMode::Off && key.contains('.') {
                return Ok(());
            }

            if !is_valid_unquoted_key(key) {
                return Err(self
                    .parse_error_with_context(format!("Invalid unquoted key: '{key}'"))
                    .with_suggestion("Quote the key to use special characters"));
            }
        }
        Ok(())
    }

    fn validate_unquoted_string(&self, value: &str, was_quoted: bool) -> ToonResult<()> {
        if self.options.strict && !was_quoted && value.contains('\t') {
            return Err(self
                .parse_error_with_context("Unquoted tab characters are not allowed in strict mode")
                .with_suggestion("Quote the value to include tabs"));
        }
        Ok(())
    }

    fn is_key_token(&self) -> bool {
        matches!(
            self.current_token,
            Token::String(_, _) | Token::Bool(_) | Token::Null
        )
    }

    fn key_from_token(&self) -> Option<(String, bool)> {
        match &self.current_token {
            Token::String(s, was_quoted) => Some((self.format_key(s, *was_quoted), *was_quoted)),
            Token::Bool(b) => Some((
                if *b {
                    KEYWORDS[1].to_string()
                } else {
                    KEYWORDS[2].to_string()
                },
                false,
            )),
            Token::Null => Some((KEYWORDS[0].to_string(), false)),
            _ => None,
        }
    }

    fn find_unexpected_delimiter(
        &self,
        field: &str,
        expected: Option<Delimiter>,
    ) -> Option<Delimiter> {
        let expected = expected?;
        let delimiters = [Delimiter::Comma, Delimiter::Pipe, Delimiter::Tab];

        delimiters
            .into_iter()
            .find(|delim| *delim != expected && field.contains(delim.as_char()))
    }

    fn parse_value(&mut self) -> ToonResult<Value> {
        self.parse_value_with_depth(0)
    }

    fn parse_value_with_depth(&mut self, depth: usize) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        let had_newline = matches!(self.current_token, Token::Newline);
        self.skip_newlines()?;

        match &self.current_token {
            Token::Null => {
                // Peek ahead to see if this is a key (followed by ':') or a value
                let next_char_is_colon = matches!(self.scanner.peek(), Some(':'));
                if next_char_is_colon {
                    let key = KEYWORDS[0].to_string();
                    self.advance()?;
                    self.parse_object_with_initial_key(key, false, depth)
                } else {
                    self.advance()?;
                    Ok(Value::Null)
                }
            }
            Token::Bool(b) => {
                let next_char_is_colon = matches!(self.scanner.peek(), Some(':'));
                if next_char_is_colon {
                    let key = if *b {
                        KEYWORDS[1].to_string()
                    } else {
                        KEYWORDS[2].to_string()
                    };
                    self.advance()?;
                    self.parse_object_with_initial_key(key, false, depth)
                } else {
                    let val = *b;
                    self.advance()?;
                    Ok(Value::Bool(val))
                }
            }
            Token::Integer(i) => {
                let next_char_is_colon = matches!(self.scanner.peek(), Some(':'));
                if next_char_is_colon {
                    let key = i.to_string();
                    self.advance()?;
                    self.parse_object_with_initial_key(key, false, depth)
                } else {
                    let val = *i;
                    self.advance()?;
                    Ok(serde_json::Number::from(val).into())
                }
            }
            Token::Number(n) => {
                let next_char_is_colon = matches!(self.scanner.peek(), Some(':'));
                if next_char_is_colon {
                    let key = n.to_string();
                    self.advance()?;
                    self.parse_object_with_initial_key(key, false, depth)
                } else {
                    let val = *n;
                    self.advance()?;
                    // Normalize floats that are actually integers
                    if val.is_finite() && val.fract() == 0.0 && val.abs() <= i64::MAX as f64 {
                        Ok(serde_json::Number::from(val as i64).into())
                    } else {
                        Ok(serde_json::Number::from_f64(val)
                            .ok_or_else(|| {
                                ToonError::InvalidInput(format!("Invalid number: {val}"))
                            })?
                            .into())
                    }
                }
            }
            Token::String(s, was_quoted) => {
                let key_was_quoted = *was_quoted;
                let first = s.clone();
                self.advance()?;

                match &self.current_token {
                    Token::Colon | Token::LeftBracket => {
                        let key = self.format_key(&first, key_was_quoted);
                        self.parse_object_with_initial_key(key, key_was_quoted, depth)
                    }
                    _ => {
                        // Strings on new indented lines could be missing colons (keys) or values
                        // Only error in strict mode when we know it's a new line
                        if self.options.strict && depth > 0 && had_newline {
                            return Err(self
                                .parse_error_with_context(format!(
                                    "Expected ':' after '{first}' in object context"
                                ))
                                .with_suggestion(
                                    "Add ':' after the key, or place the value on the same line \
                                     as the parent key",
                                ));
                        }

                        // Root-level string value - join consecutive tokens
                        let mut accumulated = first;
                        while let Token::String(next, _) = &self.current_token {
                            if !accumulated.is_empty() {
                                accumulated.push(' ');
                            }
                            accumulated.push_str(next);
                            self.advance()?;
                        }
                        self.validate_unquoted_string(&accumulated, key_was_quoted)?;
                        Ok(Value::String(accumulated))
                    }
                }
            }
            Token::LeftBracket => self.parse_root_array(depth),
            Token::Eof => Ok(Value::Object(Map::new())),
            _ => self.parse_object(depth),
        }
    }

    fn parse_object(&mut self, depth: usize) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        let mut obj = Map::new();
        // Track the indentation of the first key to ensure all keys align
        let mut base_indent: Option<usize> = None;

        loop {
            while matches!(self.current_token, Token::Newline) {
                self.advance()?;
            }

            if matches!(self.current_token, Token::Eof) {
                break;
            }

            let current_indent = self.normalize_indent(self.scanner.get_last_line_indent());

            if self.options.strict {
                self.validate_indentation(current_indent)?;
            }

            // Once we've seen the first key, all subsequent keys must match its indent
            if let Some(expected) = base_indent {
                if current_indent != expected {
                    break;
                }
            } else {
                base_indent = Some(current_indent);
            }

            let (key, was_quoted) = match self.key_from_token() {
                Some(key) => key,
                None => {
                    return Err(self
                        .parse_error_with_context(format!(
                            "Expected key, found {:?}",
                            self.current_token
                        ))
                        .with_suggestion("Object keys must be strings"));
                }
            };
            self.validate_unquoted_key(&key, was_quoted)?;
            self.advance()?;

            let value = if matches!(self.current_token, Token::LeftBracket) {
                self.parse_array(depth)?
            } else {
                if !matches!(self.current_token, Token::Colon) {
                    return Err(self
                        .parse_error_with_context(format!(
                            "Expected ':' or '[', found {:?}",
                            self.current_token
                        ))
                        .with_suggestion("Use ':' for object values or '[' for arrays"));
                }
                self.advance()?;
                self.parse_field_value(depth)?
            };

            obj.insert(key, value);
        }

        Ok(Value::Object(obj))
    }

    fn parse_object_with_initial_key(
        &mut self,
        key: String,
        key_was_quoted: bool,
        depth: usize,
    ) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        let mut obj = Map::new();
        let mut base_indent: Option<usize> = None;

        // Validate indentation for the initial key if in strict mode
        if self.options.strict {
            let current_indent = self.normalize_indent(self.scanner.get_last_line_indent());
            self.validate_indentation(current_indent)?;
        }

        self.validate_unquoted_key(&key, key_was_quoted)?;

        if matches!(self.current_token, Token::LeftBracket) {
            let value = self.parse_array(depth)?;
            obj.insert(key, value);
        } else {
            if !matches!(self.current_token, Token::Colon) {
                return Err(self.parse_error_with_context(format!(
                    "Expected ':', found {:?}",
                    self.current_token
                )));
            }
            self.advance()?;

            let value = self.parse_field_value(depth)?;
            obj.insert(key, value);
        }

        loop {
            // Skip newlines and check if the next line belongs to this object
            while matches!(self.current_token, Token::Newline) {
                self.advance()?;

                if !self.options.strict {
                    while matches!(self.current_token, Token::Newline) {
                        self.advance()?;
                    }
                }

                if matches!(self.current_token, Token::Newline) {
                    continue;
                }

                let next_indent = self.normalize_indent(self.scanner.get_last_line_indent());

                // Check if the next line is at the right indentation level
                let should_continue = if let Some(expected) = base_indent {
                    next_indent == expected
                } else {
                    // First field: use depth-based expected indent
                    let current_depth_indent = self.options.indent.get_spaces() * depth;
                    next_indent == current_depth_indent
                };

                if !should_continue {
                    break;
                }
            }

            if matches!(self.current_token, Token::Eof) {
                break;
            }

            if !self.is_key_token() {
                break;
            }

            if matches!(self.current_token, Token::Eof) {
                break;
            }

            let current_indent = self.normalize_indent(self.scanner.get_last_line_indent());

            if let Some(expected) = base_indent {
                if current_indent != expected {
                    break;
                }
            } else {
                // verify first additional field matches expected depth
                let expected_depth_indent = self.options.indent.get_spaces() * depth;
                if current_indent != expected_depth_indent {
                    break;
                }
            }

            if self.options.strict {
                self.validate_indentation(current_indent)?;
            }

            if base_indent.is_none() {
                base_indent = Some(current_indent);
            }

            let (key, was_quoted) = match self.key_from_token() {
                Some(key) => key,
                None => break,
            };
            self.validate_unquoted_key(&key, was_quoted)?;
            self.advance()?;

            let value = if matches!(self.current_token, Token::LeftBracket) {
                self.parse_array(depth)?
            } else {
                if !matches!(self.current_token, Token::Colon) {
                    break;
                }
                self.advance()?;
                self.parse_field_value(depth)?
            };

            obj.insert(key, value);
        }

        Ok(Value::Object(obj))
    }

    fn parse_field_value(&mut self, depth: usize) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        if matches!(self.current_token, Token::Newline | Token::Eof) {
            let has_children = if matches!(self.current_token, Token::Newline) {
                let current_depth_indent = self.options.indent.get_spaces() * (depth + 1);
                let next_indent = self.scanner.count_leading_spaces();
                let next_indent = self.normalize_indent(next_indent);
                next_indent >= current_depth_indent
            } else {
                false
            };

            if has_children {
                self.parse_value_with_depth(depth + 1)
            } else {
                Ok(Value::Object(Map::new()))
            }
        } else if matches!(self.current_token, Token::LeftBracket) {
            self.parse_value_with_depth(depth + 1)
        } else {
            // Check if there's more content after the current token
            let (rest, leading_space) = self.scanner.read_rest_of_line_with_space_count();

            let result = if rest.is_empty() {
                // Single token - convert directly to avoid redundant parsing
                match &self.current_token {
                    Token::String(s, was_quoted) => {
                        self.validate_unquoted_string(s, *was_quoted)?;
                        Ok(Value::String(s.clone()))
                    }
                    Token::Integer(i) => Ok(serde_json::Number::from(*i).into()),
                    Token::Number(n) => {
                        let val = *n;
                        if val.is_finite() && val.fract() == 0.0 && val.abs() <= i64::MAX as f64 {
                            Ok(serde_json::Number::from(val as i64).into())
                        } else {
                            Ok(serde_json::Number::from_f64(val)
                                .ok_or_else(|| {
                                    ToonError::InvalidInput(format!("Invalid number: {val}"))
                                })?
                                .into())
                        }
                    }
                    Token::Bool(b) => Ok(Value::Bool(*b)),
                    Token::Null => Ok(Value::Null),
                    _ => Err(self.parse_error_with_context("Unexpected token after colon")),
                }
            } else {
                // Multi-token value - reconstruct and re-parse as complete string
                let token_len = match &self.current_token {
                    Token::String(s, was_quoted) => s.len() + if *was_quoted { 2 } else { 0 },
                    Token::Integer(_) => 20,
                    Token::Number(_) => 32,
                    Token::Bool(true) => 4,
                    Token::Bool(false) => 5,
                    Token::Null => 4,
                    _ => 0,
                };
                let mut value_str = String::with_capacity(token_len + leading_space + rest.len());

                match &self.current_token {
                    Token::String(s, true) => {
                        // Quoted strings need quotes preserved for re-parsing
                        value_str.push('"');
                        crate::utils::escape_string_into(&mut value_str, s);
                        value_str.push('"');
                    }
                    Token::String(s, false) => value_str.push_str(s),
                    Token::Integer(i) => value_str.push_str(&i.to_string()),
                    Token::Number(n) => value_str.push_str(&n.to_string()),
                    Token::Bool(b) => value_str.push_str(if *b { "true" } else { "false" }),
                    Token::Null => value_str.push_str("null"),
                    _ => {
                        return Err(self.parse_error_with_context("Unexpected token after colon"));
                    }
                }

                // Only add space if there was whitespace in the original input
                if !rest.is_empty() && leading_space > 0 {
                    value_str.extend(std::iter::repeat_n(' ', leading_space));
                }
                value_str.push_str(&rest);

                let token = self.scanner.parse_value_string(&value_str)?;
                match token {
                    Token::String(s, was_quoted) => {
                        if self.options.strict && !was_quoted && value_str.contains('\t') {
                            return Err(self.parse_error_with_context(
                                "Unquoted tab characters are not allowed in strict mode",
                            ));
                        }
                        self.validate_unquoted_string(&s, was_quoted)?;
                        Ok(Value::String(s))
                    }
                    Token::Integer(i) => Ok(serde_json::Number::from(i).into()),
                    Token::Number(n) => {
                        if n.is_finite() && n.fract() == 0.0 && n.abs() <= i64::MAX as f64 {
                            Ok(serde_json::Number::from(n as i64).into())
                        } else {
                            Ok(serde_json::Number::from_f64(n)
                                .ok_or_else(|| {
                                    ToonError::InvalidInput(format!("Invalid number: {n}"))
                                })?
                                .into())
                        }
                    }
                    Token::Bool(b) => Ok(Value::Bool(b)),
                    Token::Null => Ok(Value::Null),
                    _ => Err(ToonError::InvalidInput("Unexpected token type".to_string())),
                }
            }?;

            self.current_token = self.scanner.scan_token()?;
            Ok(result)
        }
    }

    fn parse_root_array(&mut self, depth: usize) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        if !matches!(self.current_token, Token::LeftBracket) {
            return Err(self.parse_error_with_context("Expected '[' at the start of root array"));
        }

        self.parse_array(depth)
    }

    fn parse_array_header(&mut self) -> ToonResult<(usize, Option<Delimiter>, bool)> {
        if !matches!(self.current_token, Token::LeftBracket) {
            return Err(self.parse_error_with_context("Expected '['"));
        }
        self.advance()?;

        // Parse array length (plain integer only)
        // Supports formats: [N], [N|], [N\t] (no # marker)
        let length = match &self.current_token {
            Token::Integer(n) => {
                validation::validate_array_length_non_negative(*n)?;
                *n as usize
            }
            Token::Number(_) => {
                return Err(self.parse_error_with_context("Array length must be an integer"));
            }
            Token::String(s, _) => {
                // Check if string starts with # - this marker is not supported
                if s.starts_with('#') {
                    return Err(self
                        .parse_error_with_context(
                            "Length marker '#' is not supported. Use [N] format instead of [#N]",
                        )
                        .with_suggestion("Remove the '#' prefix from the array length"));
                }

                if s.contains('.') || s.contains('e') || s.contains('E') {
                    return Err(self.parse_error_with_context("Array length must be an integer"));
                }

                let parsed = s.parse::<i64>().map_err(|_| {
                    self.parse_error_with_context(format!("Expected array length, found: {s}"))
                })?;
                validation::validate_array_length_non_negative(parsed)?;
                parsed as usize
            }
            _ => {
                return Err(self.parse_error_with_context(format!(
                    "Expected array length, found {:?}",
                    self.current_token
                )));
            }
        };

        self.advance()?;

        // Check for optional delimiter after length
        let detected_delim = match &self.current_token {
            Token::Delimiter(d) => {
                let delim = *d;
                self.advance()?;
                Some(delim)
            }
            Token::String(s, _) if s == "," => {
                self.advance()?;
                Some(Delimiter::Comma)
            }
            Token::String(s, _) if s == "|" => {
                self.advance()?;
                Some(Delimiter::Pipe)
            }
            Token::String(s, _) if s == "\t" => {
                self.advance()?;
                Some(Delimiter::Tab)
            }
            _ => None,
        };

        if !matches!(self.current_token, Token::RightBracket) {
            return Err(self.parse_error_with_context(format!(
                "Expected ']', found {:?}",
                self.current_token
            )));
        }
        self.advance()?;

        let has_fields = matches!(self.current_token, Token::LeftBrace);

        Ok((length, detected_delim, has_fields))
    }

    fn parse_field_list(&mut self, expected_delim: Option<Delimiter>) -> ToonResult<Vec<String>> {
        if !matches!(self.current_token, Token::LeftBrace) {
            return Err(self.parse_error_with_context("Expected '{' for field list"));
        }
        self.advance()?;

        let mut fields = Vec::new();
        let mut field_list_delim = None;

        loop {
            match &self.current_token {
                Token::String(s, was_quoted) => {
                    if self.options.strict {
                        if let Some(unexpected) = self.find_unexpected_delimiter(s, expected_delim)
                        {
                            return Err(self.parse_error_with_context(format!(
                                "Field list delimiter {unexpected} does not match expected {}",
                                expected_delim
                                    .map(|delim| delim.to_string())
                                    .unwrap_or_else(|| "none".to_string())
                            )));
                        }
                        self.validate_unquoted_key(s, *was_quoted)?;
                    }

                    fields.push(self.format_key(s, *was_quoted));
                    self.advance()?;

                    if matches!(self.current_token, Token::RightBrace) {
                        break;
                    }

                    if let Token::Delimiter(delim) = &self.current_token {
                        if self.options.strict {
                            validation::validate_delimiter_consistency(
                                Some(*delim),
                                expected_delim,
                            )?;
                        }
                        if field_list_delim.is_none() {
                            field_list_delim = Some(*delim);
                        }
                        self.advance()?;
                    } else {
                        return Err(self.parse_error_with_context(format!(
                            "Expected delimiter or '}}', found {:?}",
                            self.current_token
                        )));
                    }
                }
                Token::Bool(_) | Token::Null => {
                    let (field, was_quoted) = match self.key_from_token() {
                        Some(key) => key,
                        None => {
                            return Err(self.parse_error_with_context(format!(
                                "Expected field name, found {:?}",
                                self.current_token
                            )))
                        }
                    };
                    self.validate_unquoted_key(&field, was_quoted)?;
                    fields.push(field);
                    self.advance()?;

                    if matches!(self.current_token, Token::RightBrace) {
                        break;
                    }

                    if let Token::Delimiter(delim) = &self.current_token {
                        if self.options.strict {
                            validation::validate_delimiter_consistency(
                                Some(*delim),
                                expected_delim,
                            )?;
                        }
                        if field_list_delim.is_none() {
                            field_list_delim = Some(*delim);
                        }
                        self.advance()?;
                    } else {
                        return Err(self.parse_error_with_context(format!(
                            "Expected delimiter or '}}', found {:?}",
                            self.current_token
                        )));
                    }
                }
                Token::RightBrace => break,
                _ => {
                    return Err(self.parse_error_with_context(format!(
                        "Expected field name, found {:?}",
                        self.current_token
                    )))
                }
            }
        }

        self.advance()?;
        validation::validate_field_list(&fields)?;
        if self.options.strict {
            validation::validate_delimiter_consistency(field_list_delim, expected_delim)?;
        }

        Ok(fields)
    }

    fn parse_array(&mut self, depth: usize) -> ToonResult<Value> {
        self.parse_array_with_context(depth, ArrayParseContext::Normal)
    }

    fn parse_array_with_context(
        &mut self,
        depth: usize,
        context: ArrayParseContext,
    ) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        let (length, detected_delim, has_fields) = self.parse_array_header()?;

        if let (Some(detected), Some(expected)) = (detected_delim, self.options.delimiter) {
            if detected != expected {
                return Err(self.parse_error_with_context(format!(
                    "Detected delimiter {detected} but expected {expected}"
                )));
            }
        }

        let active_delim = detected_delim
            .or(self.options.delimiter)
            .or(Some(Delimiter::Comma));

        let mut pushed = false;
        let result = (|| -> ToonResult<Value> {
            self.push_delimiter(active_delim);
            pushed = true;

            let fields = if has_fields {
                Some(self.parse_field_list(active_delim)?)
            } else {
                None
            };

            if !matches!(self.current_token, Token::Colon) {
                return Err(self.parse_error_with_context("Expected ':' after array header"));
            }
            self.advance()?;

            if let Some(fields) = fields {
                self.parse_tabular_array(length, &fields, depth, context)
            } else {
                // Non-tabular arrays as first field of list items require depth adjustment
                // (items at depth +2 relative to hyphen, not the usual +1)
                let adjusted_depth = match context {
                    ArrayParseContext::Normal => depth,
                    ArrayParseContext::ListItemFirstField => depth + 1,
                };
                self.parse_regular_array(length, adjusted_depth)
            }
        })();

        if pushed {
            self.pop_delimiter();
        }

        result
    }

    fn parse_tabular_array(
        &mut self,
        length: usize,
        fields: &[String],
        depth: usize,
        context: ArrayParseContext,
    ) -> ToonResult<Value> {
        let mut rows = Vec::with_capacity(length);

        if !matches!(self.current_token, Token::Newline) {
            return Err(self
                .parse_error_with_context("Expected newline after tabular array header")
                .with_suggestion("Tabular arrays must have rows on separate lines"));
        }
        self.skip_newlines()?;

        // Tabular arrays as first field of list-item objects require rows at depth +2
        // (relative to hyphen), while normal tabular arrays use depth +1
        let row_depth_offset = match context {
            ArrayParseContext::Normal => 1,
            ArrayParseContext::ListItemFirstField => 2,
        };
        let indent_size = self.options.indent.get_spaces();
        let expected_indent = indent_size * (depth + row_depth_offset);

        let mut row_index = 0;
        loop {
            if matches!(self.current_token, Token::Eof) {
                if self.options.strict {
                    return Err(self.parse_error_with_context(format!(
                        "Expected {} rows, but got {} before EOF",
                        length,
                        rows.len()
                    )));
                }
                break;
            }

            let current_indent = self.normalize_indent(self.scanner.get_last_line_indent());

            if self.options.strict {
                self.validate_indentation(current_indent)?;

                if current_indent != expected_indent {
                    return Err(self.parse_error_with_context(format!(
                        "Invalid indentation for tabular row: expected {expected_indent} spaces, \
                         found {current_indent}"
                    )));
                }
            } else {
                let is_key_value = self.is_key_token() && matches!(self.scanner.peek(), Some(':'));
                if current_indent != expected_indent || is_key_value {
                    break;
                }
            }

            let mut row = Map::with_capacity(fields.len());

            for (field_index, field) in fields.iter().enumerate() {
                // Skip delimiter before each field except the first
                if field_index > 0 {
                    if matches!(self.current_token, Token::Delimiter(_)) {
                        self.advance()?;
                    } else {
                        return Err(self
                            .parse_error_with_context(format!(
                                "Expected delimiter, found {:?}",
                                self.current_token
                            ))
                            .with_suggestion(format!(
                                "Tabular row {} field {} needs a delimiter",
                                row_index + 1,
                                field_index + 1
                            )));
                    }
                }

                // Empty values show up as delimiters or newlines
                let value = if matches!(self.current_token, Token::Delimiter(_))
                    || matches!(self.current_token, Token::Newline | Token::Eof)
                {
                    Value::String(String::new())
                } else {
                    self.parse_tabular_field_value()?
                };

                row.insert(field.clone(), value);

                // Validate row completeness
                if field_index < fields.len() - 1 {
                    // Not the last field - shouldn't hit newline yet
                    if matches!(self.current_token, Token::Newline | Token::Eof) {
                        if self.options.strict {
                            return Err(self
                                .parse_error_with_context(format!(
                                    "Tabular row {}: expected {} values, but found only {}",
                                    row_index + 1,
                                    fields.len(),
                                    field_index + 1
                                ))
                                .with_suggestion(format!(
                                    "Row {} should have exactly {} values",
                                    row_index + 1,
                                    fields.len()
                                )));
                        } else {
                            // Fill remaining fields with null in non-strict mode
                            for field in fields.iter().skip(field_index + 1) {
                                row.insert(field.clone(), Value::Null);
                            }
                            break;
                        }
                    }
                } else if !matches!(self.current_token, Token::Newline | Token::Eof)
                    && matches!(self.current_token, Token::Delimiter(_))
                {
                    // Last field but there's another delimiter - too many values
                    return Err(self
                        .parse_error_with_context(format!(
                            "Tabular row {}: expected {} values, but found extra values",
                            row_index + 1,
                            fields.len()
                        ))
                        .with_suggestion(format!(
                            "Row {} should have exactly {} values",
                            row_index + 1,
                            fields.len()
                        )));
                }
            }

            if !self.options.strict && row.len() < fields.len() {
                for field in fields.iter().skip(row.len()) {
                    row.insert(field.clone(), Value::Null);
                }
            }

            rows.push(Value::Object(row));
            row_index += 1;

            if matches!(self.current_token, Token::Eof) {
                break;
            }

            if !matches!(self.current_token, Token::Newline) {
                if !self.options.strict {
                    while !matches!(self.current_token, Token::Newline | Token::Eof) {
                        self.advance()?;
                    }
                    if matches!(self.current_token, Token::Eof) {
                        break;
                    }
                } else {
                    return Err(self.parse_error_with_context(format!(
                        "Expected newline after tabular row {}",
                        row_index
                    )));
                }
            }

            if self.options.strict {
                if row_index < length {
                    self.advance()?;
                    if matches!(self.current_token, Token::Newline) {
                        return Err(self.parse_error_with_context(
                            "Blank lines are not allowed inside tabular arrays in strict mode",
                        ));
                    }

                    self.skip_newlines()?;
                } else if matches!(self.current_token, Token::Newline) {
                    // After the last row, check if there are extra rows
                    self.advance()?;
                    self.skip_newlines()?;

                    let actual_indent = self.normalize_indent(self.scanner.get_last_line_indent());

                    // If something at the same indent level, it might be a new row (error)
                    // unless it's a key-value pair (which belongs to parent)
                    if actual_indent == expected_indent && !matches!(self.current_token, Token::Eof)
                    {
                        let is_key_value =
                            self.is_key_token() && matches!(self.scanner.peek(), Some(':'));

                        if !is_key_value {
                            return Err(self.parse_error_with_context(format!(
                                "Array length mismatch: expected {length} rows, but more rows found",
                            )));
                        }
                    }
                }

                if row_index >= length {
                    break;
                }
            } else if matches!(self.current_token, Token::Newline) {
                self.advance()?;
                self.skip_newlines()?;
            }
        }

        if self.options.strict {
            validation::validate_array_length(length, rows.len())?;
        }

        Ok(Value::Array(rows))
    }

    fn parse_regular_array(&mut self, length: usize, depth: usize) -> ToonResult<Value> {
        let mut items = Vec::with_capacity(length);
        let indent_size = self.options.indent.get_spaces();

        match &self.current_token {
            Token::Newline => {
                self.skip_newlines()?;

                let expected_indent = indent_size * (depth + 1);

                if self.options.strict {
                    for i in 0..length {
                        let current_indent =
                            self.normalize_indent(self.scanner.get_last_line_indent());
                        self.validate_indentation(current_indent)?;

                        if current_indent != expected_indent {
                            return Err(self.parse_error_with_context(format!(
                                "Invalid indentation for list item: expected {expected_indent} \
                                 spaces, found {current_indent}"
                            )));
                        }
                        if !matches!(self.current_token, Token::Dash) {
                            return Err(self
                                .parse_error_with_context(format!(
                                    "Expected '-' for list item, found {:?}",
                                    self.current_token
                                ))
                                .with_suggestion(format!(
                                    "List arrays need '-' prefix for each item (item {} of {})",
                                    i + 1,
                                    length
                                )));
                        }
                        self.advance()?;

                        let value = if matches!(self.current_token, Token::Newline | Token::Eof) {
                            Value::Object(Map::new())
                        } else if matches!(self.current_token, Token::LeftBracket) {
                            self.parse_array(depth + 1)?
                        } else if self.is_key_token() {
                            let (key, key_was_quoted) = match self.key_from_token() {
                                Some(key) => key,
                                None => {
                                    return Err(self.parse_error_with_context(format!(
                                        "Expected key, found {:?}",
                                        self.current_token
                                    )));
                                }
                            };
                            self.validate_unquoted_key(&key, key_was_quoted)?;
                            self.advance()?;

                            if matches!(self.current_token, Token::Colon | Token::LeftBracket) {
                                // This is an object: key followed by colon or array bracket
                                // First field of list-item object may be an array requiring special
                                // indentation
                                let first_value =
                                    if matches!(self.current_token, Token::LeftBracket) {
                                        // Array directly after key (e.g., "- key[N]:")
                                        // Use ListItemFirstField context to apply correct indentation
                                        self.parse_array_with_context(
                                            depth + 1,
                                            ArrayParseContext::ListItemFirstField,
                                        )?
                                    } else {
                                        self.advance()?;
                                        // Handle nested arrays: "key: [2]: ..."
                                        if matches!(self.current_token, Token::LeftBracket) {
                                            // Array after colon - not directly on hyphen line, use normal
                                            // context
                                            self.parse_array(depth + 2)?
                                        } else {
                                            self.parse_field_value(depth + 2)?
                                        }
                                    };

                                let mut obj = Map::new();
                                obj.insert(key, first_value);

                                let field_indent = indent_size * (depth + 2);

                                // Check if there are more fields at the same indentation level
                                let should_parse_more_fields =
                                    if matches!(self.current_token, Token::Newline) {
                                        let next_indent = self.scanner.count_leading_spaces();
                                        let next_indent = self.normalize_indent(next_indent);

                                        if next_indent < field_indent {
                                            false
                                        } else {
                                            self.advance()?;

                                            if !self.options.strict {
                                                self.skip_newlines()?;
                                            }
                                            true
                                        }
                                    } else if self.is_key_token() {
                                        // When already positioned at a field key, check its indent
                                        let current_indent = self
                                            .normalize_indent(self.scanner.get_last_line_indent());
                                        current_indent == field_indent
                                    } else {
                                        false
                                    };

                                // Parse additional fields if they're at the right indentation
                                if should_parse_more_fields {
                                    while !matches!(self.current_token, Token::Eof) {
                                        let current_indent = self
                                            .normalize_indent(self.scanner.get_last_line_indent());

                                        if current_indent != field_indent {
                                            break;
                                        }

                                        // Stop if we hit the next list item
                                        if matches!(self.current_token, Token::Dash) {
                                            break;
                                        }

                                        let (field_key, field_key_was_quoted) =
                                            match self.key_from_token() {
                                                Some(key) => key,
                                                None => break,
                                            };
                                        self.validate_unquoted_key(
                                            &field_key,
                                            field_key_was_quoted,
                                        )?;
                                        self.advance()?;

                                        let field_value =
                                            if matches!(self.current_token, Token::LeftBracket) {
                                                self.parse_array(depth + 2)?
                                            } else if matches!(self.current_token, Token::Colon) {
                                                self.advance()?;
                                                if matches!(self.current_token, Token::LeftBracket)
                                                {
                                                    self.parse_array(depth + 2)?
                                                } else {
                                                    self.parse_field_value(depth + 2)?
                                                }
                                            } else {
                                                break;
                                            };

                                        obj.insert(field_key, field_value);

                                        if matches!(self.current_token, Token::Newline) {
                                            let next_indent = self.scanner.count_leading_spaces();
                                            let next_indent = self.normalize_indent(next_indent);
                                            if next_indent < field_indent {
                                                break;
                                            }
                                            self.advance()?;
                                            if !self.options.strict {
                                                self.skip_newlines()?;
                                            }
                                        } else {
                                            break;
                                        }
                                    }
                                }

                                Value::Object(obj)
                            } else if matches!(self.current_token, Token::LeftBracket) {
                                // Array as object value: "key[2]: ..."
                                let array_value = self.parse_array(depth + 1)?;
                                let mut obj = Map::new();
                                obj.insert(key, array_value);
                                Value::Object(obj)
                            } else {
                                // Plain string value
                                Value::String(key)
                            }
                        } else {
                            self.parse_primitive()?
                        };

                        items.push(value);

                        if items.len() < length {
                            if matches!(self.current_token, Token::Newline) {
                                self.advance()?;

                                if self.options.strict
                                    && matches!(self.current_token, Token::Newline)
                                {
                                    return Err(self.parse_error_with_context(
                                        "Blank lines are not allowed inside list arrays in strict mode",
                                    ));
                                }

                                self.skip_newlines()?;
                            } else if !matches!(self.current_token, Token::Dash) {
                                return Err(self.parse_error_with_context(format!(
                                    "Expected newline or next list item after list item {}",
                                    i + 1
                                )));
                            }
                        } else if matches!(self.current_token, Token::Newline) {
                            // After the last item, check for extra items
                            self.advance()?;
                            self.skip_newlines()?;

                            let list_indent = indent_size * (depth + 1);
                            let actual_indent =
                                self.normalize_indent(self.scanner.get_last_line_indent());
                            // If we see another dash at the same indent, there are too many items
                            if actual_indent == list_indent
                                && matches!(self.current_token, Token::Dash)
                            {
                                return Err(self.parse_error_with_context(format!(
                                    "Array length mismatch: expected {length} items, but more items \
                                     found",
                                )));
                            }
                        }
                    }
                } else {
                    loop {
                        if matches!(self.current_token, Token::Eof) {
                            break;
                        }

                        let current_indent =
                            self.normalize_indent(self.scanner.get_last_line_indent());
                        if current_indent != expected_indent {
                            break;
                        }

                        if !matches!(self.current_token, Token::Dash) {
                            break;
                        }
                        self.advance()?;

                        let value = if matches!(self.current_token, Token::Newline | Token::Eof) {
                            Value::Object(Map::new())
                        } else if matches!(self.current_token, Token::LeftBracket) {
                            self.parse_array(depth + 1)?
                        } else if self.is_key_token() {
                            let (key, key_was_quoted) = match self.key_from_token() {
                                Some(key) => key,
                                None => {
                                    return Err(self.parse_error_with_context(format!(
                                        "Expected key, found {:?}",
                                        self.current_token
                                    )));
                                }
                            };
                            self.validate_unquoted_key(&key, key_was_quoted)?;
                            self.advance()?;

                            if matches!(self.current_token, Token::Colon | Token::LeftBracket) {
                                let first_value =
                                    if matches!(self.current_token, Token::LeftBracket) {
                                        self.parse_array_with_context(
                                            depth + 1,
                                            ArrayParseContext::ListItemFirstField,
                                        )?
                                    } else {
                                        self.advance()?;
                                        if matches!(self.current_token, Token::LeftBracket) {
                                            self.parse_array(depth + 2)?
                                        } else {
                                            self.parse_field_value(depth + 2)?
                                        }
                                    };

                                let mut obj = Map::new();
                                obj.insert(key, first_value);

                                let field_indent = indent_size * (depth + 2);

                                let should_parse_more_fields =
                                    if matches!(self.current_token, Token::Newline) {
                                        let next_indent = self.scanner.count_leading_spaces();
                                        let next_indent = self.normalize_indent(next_indent);

                                        if next_indent < field_indent {
                                            false
                                        } else {
                                            self.advance()?;
                                            self.skip_newlines()?;
                                            true
                                        }
                                    } else if self.is_key_token() {
                                        let current_indent = self
                                            .normalize_indent(self.scanner.get_last_line_indent());
                                        current_indent == field_indent
                                    } else {
                                        false
                                    };

                                if should_parse_more_fields {
                                    while !matches!(self.current_token, Token::Eof) {
                                        let current_indent = self
                                            .normalize_indent(self.scanner.get_last_line_indent());
                                        if current_indent != field_indent {
                                            break;
                                        }

                                        if matches!(self.current_token, Token::Dash) {
                                            break;
                                        }

                                        let (field_key, field_key_was_quoted) =
                                            match self.key_from_token() {
                                                Some(key) => key,
                                                None => break,
                                            };
                                        self.validate_unquoted_key(
                                            &field_key,
                                            field_key_was_quoted,
                                        )?;
                                        self.advance()?;

                                        let field_value =
                                            if matches!(self.current_token, Token::LeftBracket) {
                                                self.parse_array(depth + 2)?
                                            } else if matches!(self.current_token, Token::Colon) {
                                                self.advance()?;
                                                if matches!(self.current_token, Token::LeftBracket)
                                                {
                                                    self.parse_array(depth + 2)?
                                                } else {
                                                    self.parse_field_value(depth + 2)?
                                                }
                                            } else {
                                                break;
                                            };

                                        obj.insert(field_key, field_value);

                                        if matches!(self.current_token, Token::Newline) {
                                            let next_indent = self.scanner.count_leading_spaces();
                                            let next_indent = self.normalize_indent(next_indent);
                                            if next_indent < field_indent {
                                                break;
                                            }
                                            self.advance()?;
                                            self.skip_newlines()?;
                                        } else {
                                            break;
                                        }
                                    }
                                }

                                Value::Object(obj)
                            } else if matches!(self.current_token, Token::LeftBracket) {
                                let array_value = self.parse_array(depth + 1)?;
                                let mut obj = Map::new();
                                obj.insert(key, array_value);
                                Value::Object(obj)
                            } else {
                                Value::String(key)
                            }
                        } else {
                            self.parse_primitive()?
                        };

                        items.push(value);

                        if matches!(self.current_token, Token::Newline) {
                            self.advance()?;
                            self.skip_newlines()?;
                        } else if matches!(self.current_token, Token::Eof) {
                            break;
                        } else if !matches!(self.current_token, Token::Dash) {
                            return Err(self.parse_error_with_context(format!(
                                "Expected newline or next list item after list item {}",
                                items.len()
                            )));
                        }
                    }
                }
            }
            _ => {
                if self.options.strict {
                    for i in 0..length {
                        if i > 0 {
                            if matches!(self.current_token, Token::Delimiter(_)) {
                                self.advance()?;
                            } else {
                                return Err(self
                                    .parse_error_with_context(format!(
                                        "Expected delimiter, found {:?}",
                                        self.current_token
                                    ))
                                    .with_suggestion(format!(
                                        "Expected delimiter between items (item {} of {})",
                                        i + 1,
                                        length
                                    )));
                            }
                        }

                        let value = if matches!(self.current_token, Token::Delimiter(_))
                            || (matches!(self.current_token, Token::Eof | Token::Newline)
                                && i < length)
                        {
                            Value::String(String::new())
                        } else if matches!(self.current_token, Token::LeftBracket) {
                            self.parse_array(depth + 1)?
                        } else {
                            self.parse_primitive()?
                        };

                        items.push(value);
                    }
                } else {
                    let mut i = 0;
                    loop {
                        if i == 0 && matches!(self.current_token, Token::Newline | Token::Eof) {
                            break;
                        }

                        if i > 0 {
                            if matches!(self.current_token, Token::Delimiter(_)) {
                                self.advance()?;
                            } else {
                                return Err(self.parse_error_with_context(format!(
                                    "Expected delimiter, found {:?}",
                                    self.current_token
                                )));
                            }
                        }

                        let value = if matches!(self.current_token, Token::Delimiter(_))
                            || matches!(self.current_token, Token::Eof | Token::Newline)
                        {
                            Value::String(String::new())
                        } else if matches!(self.current_token, Token::LeftBracket) {
                            self.parse_array(depth + 1)?
                        } else {
                            self.parse_primitive()?
                        };

                        items.push(value);
                        i += 1;

                        if matches!(self.current_token, Token::Newline | Token::Eof) {
                            break;
                        }
                    }
                }
            }
        }

        if self.options.strict {
            validation::validate_array_length(length, items.len())?;

            if matches!(self.current_token, Token::Delimiter(_)) {
                return Err(self.parse_error_with_context(format!(
                    "Array length mismatch: expected {length} items, but more items found",
                )));
            }
        }

        Ok(Value::Array(items))
    }

    fn parse_tabular_field_value(&mut self) -> ToonResult<Value> {
        match &self.current_token {
            Token::Null => {
                self.advance()?;
                Ok(Value::Null)
            }
            Token::Bool(b) => {
                let val = *b;
                self.advance()?;
                Ok(Value::Bool(val))
            }
            Token::Integer(i) => {
                let val = *i;
                self.advance()?;
                Ok(Number::from(val).into())
            }
            Token::Number(n) => {
                let val = *n;
                self.advance()?;
                // If the float is actually an integer, represent it as such
                if val.is_finite() && val.fract() == 0.0 && val.abs() <= i64::MAX as f64 {
                    Ok(Number::from(val as i64).into())
                } else {
                    Ok(Number::from_f64(val)
                        .ok_or_else(|| ToonError::InvalidInput(format!("Invalid number: {val}")))?
                        .into())
                }
            }
            Token::String(s, was_quoted) => {
                // Tabular fields can have multiple string tokens joined with spaces
                self.validate_unquoted_string(s, *was_quoted)?;
                let mut accumulated = s.clone();
                self.advance()?;

                while let Token::String(next, next_was_quoted) = &self.current_token {
                    self.validate_unquoted_string(next, *next_was_quoted)?;
                    if !accumulated.is_empty() {
                        accumulated.push(' ');
                    }
                    accumulated.push_str(next);
                    self.advance()?;
                }

                Ok(Value::String(accumulated))
            }
            _ => Err(self.parse_error_with_context(format!(
                "Expected primitive value, found {:?}",
                self.current_token
            ))),
        }
    }

    fn parse_primitive(&mut self) -> ToonResult<Value> {
        match &self.current_token {
            Token::Null => {
                self.advance()?;
                Ok(Value::Null)
            }
            Token::Bool(b) => {
                let val = *b;
                self.advance()?;
                Ok(Value::Bool(val))
            }
            Token::Integer(i) => {
                let val = *i;
                self.advance()?;
                Ok(Number::from(val).into())
            }
            Token::Number(n) => {
                let val = *n;
                self.advance()?;

                if val.is_finite() && val.fract() == 0.0 && val.abs() <= i64::MAX as f64 {
                    Ok(Number::from(val as i64).into())
                } else {
                    Ok(Number::from_f64(val)
                        .ok_or_else(|| ToonError::InvalidInput(format!("Invalid number: {val}")))?
                        .into())
                }
            }
            Token::String(s, was_quoted) => {
                self.validate_unquoted_string(s, *was_quoted)?;
                let val = s.clone();
                self.advance()?;
                Ok(Value::String(val))
            }
            _ => Err(self.parse_error_with_context(format!(
                "Expected primitive value, found {:?}",
                self.current_token
            ))),
        }
    }

    fn parse_error_with_context(&self, message: impl Into<String>) -> ToonError {
        let (line, column) = self.scanner.current_position();
        let message = message.into();

        let context = ErrorContext::from_shared_input(self.input.clone(), line, column, 2)
            .unwrap_or_else(|| ErrorContext::new(""));

        ToonError::ParseError {
            line,
            column,
            message,
            context: Some(Box::new(context)),
        }
    }

    fn validate_indentation(&self, indent_amount: usize) -> ToonResult<()> {
        if !self.options.strict {
            return Ok(());
        }

        let indent_size = self.options.indent.get_spaces();
        // In strict mode, indentation must be a multiple of the configured indent size
        if indent_size > 0 && indent_amount > 0 && !indent_amount.is_multiple_of(indent_size) {
            Err(self.parse_error_with_context(format!(
                "Invalid indentation: found {indent_amount} spaces, but must be a multiple of \
                 {indent_size}"
            )))
        } else {
            Ok(())
        }
    }

    fn normalize_indent(&self, indent_amount: usize) -> usize {
        if self.options.strict {
            return indent_amount;
        }

        let indent_size = self.options.indent.get_spaces();
        if indent_size == 0 {
            indent_amount
        } else {
            (indent_amount / indent_size) * indent_size
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64;

    use serde_json::json;

    use super::*;

    fn parse(input: &str) -> ToonResult<Value> {
        let mut parser = Parser::new(input, DecodeOptions::default())?;
        parser.parse()
    }

    #[test]
    fn test_parse_primitives() {
        assert_eq!(parse("null").unwrap(), json!(null));
        assert_eq!(parse("true").unwrap(), json!(true));
        assert_eq!(parse("false").unwrap(), json!(false));
        assert_eq!(parse("42").unwrap(), json!(42));
        assert_eq!(parse("3.141592653589793").unwrap(), json!(f64::consts::PI));
        assert_eq!(parse("hello").unwrap(), json!("hello"));
    }

    #[test]
    fn test_parse_simple_object() {
        let result = parse("name: Alice\nage: 30").unwrap();
        assert_eq!(result["name"], json!("Alice"));
        assert_eq!(result["age"], json!(30));
    }

    #[test]
    fn test_parse_primitive_array() {
        let result = parse("tags[3]: a,b,c").unwrap();
        assert_eq!(result["tags"], json!(["a", "b", "c"]));
    }

    #[test]
    fn test_parse_empty_array() {
        let result = parse("items[0]:").unwrap();
        assert_eq!(result["items"], json!([]));
    }

    #[test]
    fn test_parse_tabular_array() {
        let result = parse("users[2]{id,name}:\n  1,Alice\n  2,Bob").unwrap();
        assert_eq!(
            result["users"],
            json!([
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ])
        );
    }

    #[test]
    fn test_empty_tokens() {
        let result = parse("items[3]: a,,c").unwrap();
        assert_eq!(result["items"], json!(["a", "", "c"]));
    }

    #[test]
    fn test_empty_nested_object() {
        let result = parse("user:").unwrap();
        assert_eq!(result, json!({"user": {}}));
    }

    #[test]
    fn test_list_item_object() {
        let result =
            parse("items[2]:\n  - id: 1\n    name: First\n  - id: 2\n    name: Second").unwrap();
        assert_eq!(
            result["items"],
            json!([
                {"id": 1, "name": "First"},
                {"id": 2, "name": "Second"}
            ])
        );
    }

    #[test]
    fn test_nested_array_in_list_item() {
        let result = parse("items[1]:\n  - tags[3]: a,b,c").unwrap();
        assert_eq!(result["items"], json!([{"tags": ["a", "b", "c"]}]));
    }

    #[test]
    fn test_two_level_siblings() {
        let input = "x:\n  y: 1\n  z: 2";
        let opts = DecodeOptions::default();
        let mut parser = Parser::new(input, opts).unwrap();
        let result = parser.parse().unwrap();

        let x = result.as_object().unwrap().get("x").unwrap();
        let x_obj = x.as_object().unwrap();

        assert_eq!(x_obj.len(), 2, "x should have 2 keys");
        assert_eq!(x_obj.get("y").unwrap(), &serde_json::json!(1));
        assert_eq!(x_obj.get("z").unwrap(), &serde_json::json!(2));
    }

    #[test]
    fn test_nested_object_with_sibling() {
        let input = "a:\n  b:\n    c: 1\n  d: 2";
        let opts = DecodeOptions::default();
        let mut parser = Parser::new(input, opts).unwrap();
        let result = parser.parse().unwrap();

        let a = result.as_object().unwrap().get("a").unwrap();
        let a_obj = a.as_object().unwrap();

        assert_eq!(a_obj.len(), 2, "a should have 2 keys (b and d)");
        assert!(a_obj.contains_key("b"), "a should have key 'b'");
        assert!(a_obj.contains_key("d"), "a should have key 'd'");

        let b = a_obj.get("b").unwrap().as_object().unwrap();
        assert_eq!(b.len(), 1, "b should have only 1 key (c)");
        assert!(b.contains_key("c"), "b should have key 'c'");
        assert!(!b.contains_key("d"), "b should NOT have key 'd'");
    }

    #[test]
    fn test_field_value_with_parentheses() {
        let result = parse("msg: Mostly Functions (3 of 3)").unwrap();
        assert_eq!(result, json!({"msg": "Mostly Functions (3 of 3)"}));

        let result = parse("val: (hello)").unwrap();
        assert_eq!(result, json!({"val": "(hello)"}));

        let result = parse("test: a (b) c (d)").unwrap();
        assert_eq!(result, json!({"test": "a (b) c (d)"}));
    }

    #[test]
    fn test_field_value_number_with_parentheses() {
        let result = parse("code: 0(f)").unwrap();
        assert_eq!(result, json!({"code": "0(f)"}));

        let result = parse("val: 5(test)").unwrap();
        assert_eq!(result, json!({"val": "5(test)"}));

        let result = parse("msg: test 123)").unwrap();
        assert_eq!(result, json!({"msg": "test 123)"}));
    }

    #[test]
    fn test_field_value_single_token_optimization() {
        let result = parse("name: hello").unwrap();
        assert_eq!(result, json!({"name": "hello"}));

        let result = parse("age: 42").unwrap();
        assert_eq!(result, json!({"age": 42}));

        let result = parse("active: true").unwrap();
        assert_eq!(result, json!({"active": true}));

        let result = parse("value: null").unwrap();
        assert_eq!(result, json!({"value": null}));
    }

    #[test]
    fn test_field_value_multi_token() {
        let result = parse("msg: hello world").unwrap();
        assert_eq!(result, json!({"msg": "hello world"}));

        let result = parse("msg: test 123 end").unwrap();
        assert_eq!(result, json!({"msg": "test 123 end"}));
    }

    #[test]
    fn test_field_value_spacing_preserved() {
        let result = parse("val: hello world").unwrap();
        assert_eq!(result, json!({"val": "hello world"}));

        let result = parse("val: 0(f)").unwrap();
        assert_eq!(result, json!({"val": "0(f)"}));
    }

    #[test]
    fn test_round_trip_parentheses() {
        use crate::{decode::decode_default, encode::encode_default};

        let original = json!({
            "message": "Mostly Functions (3 of 3)",
            "code": "0(f)",
            "simple": "(hello)",
            "mixed": "test 123)"
        });

        let encoded = encode_default(&original).unwrap();
        let decoded: Value = decode_default(&encoded).unwrap();

        assert_eq!(original, decoded);
    }

    #[test]
    fn test_multiple_fields_with_edge_cases() {
        let input = r#"message: Mostly Functions (3 of 3)
sone: (hello)
hello: 0(f)"#;

        let result = parse(input).unwrap();
        assert_eq!(
            result,
            json!({
                "message": "Mostly Functions (3 of 3)",
                "sone": "(hello)",
                "hello": "0(f)"
            })
        );
    }

    #[test]
    fn test_decode_list_item_tabular_array_v3() {
        // Tabular arrays as first field of list items
        // Rows must be at depth +2 relative to hyphen (6 spaces from root)
        let input = r#"items[1]:
  - users[2]{id,name}:
      1,Ada
      2,Bob
    status: active"#;

        let result = parse(input).unwrap();

        assert_eq!(
            result,
            json!({
                "items": [
                    {
                        "users": [
                            {"id": 1, "name": "Ada"},
                            {"id": 2, "name": "Bob"}
                        ],
                        "status": "active"
                    }
                ]
            })
        );
    }

    #[test]
    fn test_decode_list_item_tabular_array_multiple_items() {
        // Multiple list items each with tabular array as first field
        let input = r#"data[2]:
  - records[1]{id,val}:
      1,x
    count: 1
  - records[1]{id,val}:
      2,y
    count: 1"#;

        let result = parse(input).unwrap();

        assert_eq!(
            result,
            json!({
                "data": [
                    {
                        "records": [{"id": 1, "val": "x"}],
                        "count": 1
                    },
                    {
                        "records": [{"id": 2, "val": "y"}],
                        "count": 1
                    }
                ]
            })
        );
    }

    #[test]
    fn test_decode_list_item_tabular_array_with_multiple_fields() {
        // List item with tabular array first and multiple sibling fields
        let input = r#"entries[1]:
  - people[2]{name,age}:
      Alice,30
      Bob,25
    total: 2
    category: staff"#;

        let result = parse(input).unwrap();

        assert_eq!(
            result,
            json!({
                "entries": [
                    {
                        "people": [
                            {"name": "Alice", "age": 30},
                            {"name": "Bob", "age": 25}
                        ],
                        "total": 2,
                        "category": "staff"
                    }
                ]
            })
        );
    }

    #[test]
    fn test_decode_list_item_non_tabular_array_unchanged() {
        // Non-tabular arrays as first field should work normally
        let input = r#"items[1]:
  - tags[3]: a,b,c
    name: test"#;

        let result = parse(input).unwrap();

        assert_eq!(
            result,
            json!({
                "items": [
                    {
                        "tags": ["a", "b", "c"],
                        "name": "test"
                    }
                ]
            })
        );
    }

    #[test]
    fn test_decode_strict_rejects_v2_tabular_indent() {
        use crate::decode::decode_strict;

        // Old format: rows at depth +1 (4 spaces from root)
        // Strict mode should reject this incorrect indentation
        let input_v2 = r#"items[1]:
  - users[2]{id,name}:
    1,Ada
    2,Bob"#;

        let result = decode_strict::<Value>(input_v2);

        // Should error due to incorrect indentation
        assert!(
            result.is_err(),
            "Old format with incorrect indentation should be rejected in strict mode"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("indentation") || err_msg.contains("Invalid indentation"),
            "Error should mention indentation. Got: {}",
            err_msg
        );
    }

    #[test]
    fn test_decode_tabular_array_not_in_list_item_unchanged() {
        // Regular tabular arrays (not in list items) should still use depth +1
        let input = r#"users[2]{id,name}:
  1,Ada
  2,Bob"#;

        let result = parse(input).unwrap();

        assert_eq!(
            result,
            json!({
                "users": [
                    {"id": 1, "name": "Ada"},
                    {"id": 2, "name": "Bob"}
                ]
            })
        );
    }

    #[test]
    fn test_decode_nested_tabular_not_first_field() {
        // Tabular array as a subsequent field (not first) should use normal depth
        let input = r#"items[1]:
  - name: test
    data[2]{id,val}:
      1,x
      2,y"#;

        let result = parse(input).unwrap();

        assert_eq!(
            result,
            json!({
                "items": [
                    {
                        "name": "test",
                        "data": [
                            {"id": 1, "val": "x"},
                            {"id": 2, "val": "y"}
                        ]
                    }
                ]
            })
        );
    }
}
