use serde_json::{
    Map,
    Number,
    Value,
};

use crate::{
    constants::{
        KEYWORDS,
        MAX_DEPTH,
    },
    decode::{
        scanner::{
            Scanner,
            Token,
        },
        validation,
    },
    types::{
        DecodeOptions,
        Delimiter,
        ErrorContext,
        ToonError,
        ToonResult,
    },
    utils::validation::validate_depth,
};

/// Parser that builds JSON values from a sequence of tokens.
pub struct Parser<'a> {
    scanner: Scanner,
    current_token: Token,
    options: DecodeOptions,
    delimiter: Option<Delimiter>,
    input: &'a str,
}

impl<'a> Parser<'a> {
    /// Create a new parser with the given input and options.
    pub fn new(input: &'a str, options: DecodeOptions) -> Self {
        let mut scanner = Scanner::new(input);
        let chosen_delim = options.delimiter;
        scanner.set_active_delimiter(chosen_delim);
        let current_token = scanner.scan_token().unwrap_or(Token::Eof);

        Self {
            scanner,
            current_token,
            delimiter: chosen_delim,
            options,
            input,
        }
    }

    /// Parse the input into a JSON value.
    pub fn parse(&mut self) -> ToonResult<Value> {
        self.parse_value()
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

    fn parse_value(&mut self) -> ToonResult<Value> {
        self.parse_value_with_depth(0)
    }

    fn parse_value_with_depth(&mut self, depth: usize) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        self.skip_newlines()?;

        match &self.current_token {
            Token::Null => {
                // "null:" indicates "null" is a key, not a value
                let next_char_is_colon = matches!(self.scanner.peek(), Some(':'));
                if next_char_is_colon {
                    let key = KEYWORDS[0].to_string();
                    self.advance()?;
                    self.parse_object_with_initial_key(key, depth)
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
                    self.parse_object_with_initial_key(key, depth)
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
                    self.parse_object_with_initial_key(key, depth)
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
                    self.parse_object_with_initial_key(key, depth)
                } else {
                    let val = *n;
                    self.advance()?;
                    Ok(serde_json::Number::from_f64(val)
                        .ok_or_else(|| ToonError::InvalidInput(format!("Invalid number: {}", val)))?
                        .into())
                }
            }
            Token::String(s, _) => {
                let first = s.clone();
                self.advance()?;

                match &self.current_token {
                    Token::Colon | Token::LeftBracket => {
                        self.parse_object_with_initial_key(first, depth)
                    }
                    _ => {
                        // Accumulate consecutive strings with spaces (e.g., "hello" "world" ->
                        // "hello world")
                        let mut accumulated = first;
                        loop {
                            match &self.current_token {
                                Token::String(next, _) => {
                                    if !accumulated.is_empty() {
                                        accumulated.push(' ');
                                    }
                                    accumulated.push_str(next);
                                    self.advance()?;
                                }
                                _ => break,
                            }
                        }
                        Ok(Value::String(accumulated))
                    }
                }
            }
            Token::LeftBracket => self.parse_root_array(depth),
            Token::Eof => Ok(Value::Null),
            _ => self.parse_object(depth),
        }
    }

    fn parse_object(&mut self, depth: usize) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        let mut obj = Map::new();
        let mut base_indent: Option<usize> = None;

        loop {
            while matches!(self.current_token, Token::Newline) {
                self.advance()?;
            }

            if matches!(self.current_token, Token::Eof) {
                break;
            }

            let current_indent = self.scanner.get_last_line_indent();
            if let Some(expected) = base_indent {
                if current_indent != expected {
                    break;
                }
            } else {
                base_indent = Some(current_indent);
            }

            let key = match &self.current_token {
                Token::String(s, _) => s.clone(),
                _ => {
                    return Err(self
                        .parse_error_with_context(format!(
                            "Expected key, found {:?}",
                            self.current_token
                        ))
                        .with_suggestion("Object keys must be strings"));
                }
            };
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

    fn parse_object_with_initial_key(&mut self, key: String, depth: usize) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        let mut obj = Map::new();

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

        self.skip_newlines()?;

        loop {
            if matches!(self.current_token, Token::Eof) {
                break;
            }

            let next_key = match &self.current_token {
                Token::String(s, _) => s.clone(),
                _ => break,
            };
            self.advance()?;

            let next_value = if matches!(self.current_token, Token::LeftBracket) {
                self.parse_array(depth)?
            } else {
                if !matches!(self.current_token, Token::Colon) {
                    break;
                }
                self.advance()?;
                self.parse_field_value(depth)?
            };

            obj.insert(next_key, next_value);
            self.skip_newlines()?;
        }

        Ok(Value::Object(obj))
    }

    fn parse_field_value(&mut self, depth: usize) -> ToonResult<Value> {
        match &self.current_token {
            Token::Newline => self.parse_indented_object(depth + 1),
            _ => self.parse_primitive(),
        }
    }

    fn parse_indented_object(&mut self, depth: usize) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        let mut obj = Map::new();

        loop {
            while matches!(self.current_token, Token::Newline) {
                self.advance()?;
            }

            if self.scanner.get_last_line_indent() == 0 || matches!(self.current_token, Token::Eof)
            {
                break;
            }

            let key = match &self.current_token {
                Token::String(s, _) => s.clone(),
                _ => {
                    return Err(self
                        .parse_error_with_context(format!(
                            "Expected key, found {:?}",
                            self.current_token
                        ))
                        .with_suggestion("Object keys must be strings"));
                }
            };

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
                        .with_suggestion("Use ':' after object keys"));
                }
                self.advance()?;
                self.parse_field_value(depth)?
            };

            obj.insert(key, value);
            while matches!(self.current_token, Token::Newline) {
                self.advance()?;
            }
        }

        Ok(Value::Object(obj))
    }

    fn parse_primitive(&mut self) -> ToonResult<Value> {
        match &self.current_token {
            Token::String(s, is_quoted) => {
                let value = if *is_quoted {
                    Value::String(s.clone())
                } else if self.options.coerce_types {
                    self.coerce_string_to_type(s)
                } else {
                    Value::String(s.clone())
                };
                self.advance()?;
                Ok(value)
            }
            Token::Integer(i) => {
                let value = Value::Number((*i).into());
                self.advance()?;
                Ok(value)
            }
            Token::Number(f) => {
                let value = Number::from_f64(*f)
                    .map(Value::Number)
                    .unwrap_or_else(|| Value::String(f.to_string()));
                self.advance()?;
                Ok(value)
            }
            Token::Bool(b) => {
                let value = *b;
                self.advance()?;
                Ok(Value::Bool(value))
            }
            Token::Null => {
                self.advance()?;
                Ok(Value::Null)
            }
            _ => Err(self
                .parse_error_with_context(format!(
                    "Expected primitive value, found {:?}",
                    self.current_token
                ))
                .with_suggestion("Expected a value (string, number, boolean, or null)")),
        }
    }

    fn create_error_context(&self) -> ErrorContext {
        let line = self.scanner.get_line();
        let column = self.scanner.get_column();

        ErrorContext::from_input(self.input, line, column, 2)
            .unwrap_or_else(|| ErrorContext::new("").with_indicator(column))
    }

    fn parse_error_with_context(&self, message: impl Into<String>) -> ToonError {
        let context = self.create_error_context();
        ToonError::parse_error_with_context(
            self.scanner.get_line(),
            self.scanner.get_column(),
            message,
            context,
        )
    }

    fn coerce_string_to_type(&self, s: &str) -> Value {
        if s == "null" {
            return Value::Null;
        }

        if s == "true" {
            return Value::Bool(true);
        }
        if s == "false" {
            return Value::Bool(false);
        }

        if let Ok(i) = s.parse::<i64>() {
            return Value::Number(i.into());
        }

        if let Ok(f) = s.parse::<f64>() {
            if let Some(num) = Number::from_f64(f) {
                return Value::Number(num);
            }
        }

        Value::String(s.to_string())
    }

    fn parse_array(&mut self, depth: usize) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        if !matches!(self.current_token, Token::LeftBracket) {
            return Err(self
                .parse_error_with_context("Expected '['")
                .with_suggestion("Arrays must start with '['"));
        }
        self.advance()?;

        let length = self.parse_array_length()?;

        self.detect_or_consume_delimiter()?;

        if !matches!(self.current_token, Token::RightBracket) {
            return Err(self
                .parse_error_with_context("Expected ']'")
                .with_suggestion("Close array length with ']'"));
        }
        self.advance()?;

        if self.delimiter.is_none() {
            self.delimiter = Some(Delimiter::Comma);
        }
        self.scanner.set_active_delimiter(self.delimiter);

        let fields = if matches!(self.current_token, Token::LeftBrace) {
            Some(self.parse_field_list()?)
        } else {
            None
        };

        if !matches!(self.current_token, Token::Colon) {
            return Err(self
                .parse_error_with_context("Expected ':'")
                .with_suggestion("Array header must end with ':'"));
        }
        self.advance()?;

        if length == 0 {
            return Ok(Value::Array(vec![]));
        }

        if let Some(fields) = fields {
            validation::validate_field_list(&fields)?;
            self.parse_tabular_array(length, fields, depth)
        } else {
            self.parse_regular_array(length, depth)
        }
    }

    fn parse_root_array(&mut self, depth: usize) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;
        self.parse_array(depth)
    }

    fn parse_array_length(&mut self) -> ToonResult<usize> {
        if let Some(length_str) = match &self.current_token {
            Token::String(s, _) if s.starts_with('#') => Some(s[1..].to_string()),
            _ => None,
        } {
            self.advance()?;
            return length_str.parse::<usize>().map_err(|_| {
                self.parse_error_with_context(format!("Invalid array length: {}", length_str))
                    .with_suggestion("Length must be a positive number")
            });
        }

        match &self.current_token {
            Token::Integer(i) => {
                let len = *i as usize;
                self.advance()?;
                Ok(len)
            }
            _ => Err(self
                .parse_error_with_context(format!(
                    "Expected array length, found {:?}",
                    self.current_token
                ))
                .with_suggestion("Array must have a length like [5] or #5")),
        }
    }

    fn detect_or_consume_delimiter(&mut self) -> ToonResult<()> {
        match &self.current_token {
            Token::Delimiter(delim) => {
                if self.delimiter.is_none() {
                    self.delimiter = Some(*delim);
                }
                self.advance()?;
            }
            Token::String(s, _) if s == "," || s == "|" || s == "\t" => {
                let delim = if s == "," {
                    Delimiter::Comma
                } else if s == "|" {
                    Delimiter::Pipe
                } else {
                    Delimiter::Tab
                };
                if self.delimiter.is_none() {
                    self.delimiter = Some(delim);
                }
                self.advance()?;
            }
            _ => {}
        }
        self.scanner.set_active_delimiter(self.delimiter);
        Ok(())
    }

    fn parse_field_list(&mut self) -> ToonResult<Vec<String>> {
        if !matches!(self.current_token, Token::LeftBrace) {
            return Err(self
                .parse_error_with_context("Expected '{'")
                .with_suggestion("Tabular arrays need field list like {id,name}"));
        }
        self.advance()?;

        let mut fields = Vec::new();

        loop {
            match &self.current_token {
                Token::String(s, _) => {
                    fields.push(s.clone());
                    self.advance()?;

                    if matches!(self.current_token, Token::Delimiter(_)) {
                        self.advance()?;
                    } else if matches!(self.current_token, Token::RightBrace) {
                        break;
                    }
                }
                Token::RightBrace => break,
                _ => {
                    return Err(self
                        .parse_error_with_context(format!(
                            "Expected field name, found {:?}",
                            self.current_token
                        ))
                        .with_suggestion("Field names must be strings separated by commas"));
                }
            }
        }

        if !matches!(self.current_token, Token::RightBrace) {
            return Err(self
                .parse_error_with_context("Expected '}'")
                .with_suggestion("Close field list with '}'"));
        }
        self.advance()?;

        Ok(fields)
    }

    fn parse_tabular_array(
        &mut self,
        length: usize,
        fields: Vec<String>,
        depth: usize,
    ) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        let mut rows = Vec::new();

        self.skip_newlines()?;

        self.scanner.set_active_delimiter(self.delimiter);

        for row_index in 0..length {
            let mut row = Map::new();

            for (i, field) in fields.iter().enumerate() {
                if i > 0 {
                    match &self.current_token {
                        Token::Delimiter(_) => {
                            self.advance()?;
                        }
                        Token::String(s, _) if s == "," || s == "|" || s == "\t" => {
                            self.advance()?;
                        }
                        _ => {
                            return Err(self
                                .parse_error_with_context(format!(
                                    "Expected delimiter in tabular row {}, got {:?}",
                                    row_index, self.current_token
                                ))
                                .with_suggestion(&format!(
                                    "Expected delimiter between fields in row {}",
                                    row_index + 1
                                )));
                        }
                    }
                }

                let value = self.parse_primitive()?;
                row.insert(field.clone(), value);
            }

            rows.push(Value::Object(row));

            if row_index < length - 1 {
                self.skip_newlines()?;
            }
        }

        validation::validate_array_length(length, rows.len(), self.options.strict)?;

        Ok(Value::Array(rows))
    }

    fn parse_regular_array(&mut self, length: usize, depth: usize) -> ToonResult<Value> {
        let mut items = Vec::new();

        match &self.current_token {
            Token::Newline => {
                self.skip_newlines()?;

                for i in 0..length {
                    if !matches!(self.current_token, Token::Dash) {
                        return Err(self
                            .parse_error_with_context(format!(
                                "Expected '-' for list item, found {:?}",
                                self.current_token
                            ))
                            .with_suggestion(&format!(
                                "List arrays need '-' prefix for each item (item {} of {})",
                                i + 1,
                                length
                            )));
                    }
                    self.advance()?;

                    let value = if matches!(self.current_token, Token::LeftBracket) {
                        self.parse_array(depth + 1)?
                    } else {
                        self.parse_primitive()?
                    };

                    items.push(value);

                    if items.len() < length {
                        self.skip_newlines()?;
                    }
                }
            }
            _ => {
                for i in 0..length {
                    if i > 0 {
                        match &self.current_token {
                            Token::Delimiter(_) => {
                                self.advance()?;
                            }
                            Token::String(s, _) if s == "," || s == "|" || s == "\t" => {
                                self.advance()?;
                            }
                            _ => {
                                return Err(self
                                    .parse_error_with_context(format!(
                                        "Expected delimiter, found {:?}",
                                        self.current_token
                                    ))
                                    .with_suggestion(&format!(
                                        "Expected delimiter between items (item {} of {})",
                                        i + 1,
                                        length
                                    )));
                            }
                        }
                    }

                    let value = if matches!(self.current_token, Token::LeftBracket) {
                        self.parse_array(depth + 1)?
                    } else {
                        self.parse_primitive()?
                    };

                    items.push(value);
                }
            }
        }

        validation::validate_array_length(length, items.len(), self.options.strict)?;

        Ok(Value::Array(items))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn parse(input: &str) -> ToonResult<Value> {
        let mut parser = Parser::new(input, DecodeOptions::default());
        parser.parse()
    }

    #[test]
    fn test_parse_primitives() {
        assert_eq!(parse("null").unwrap(), json!(null));
        assert_eq!(parse("true").unwrap(), json!(true));
        assert_eq!(parse("false").unwrap(), json!(false));
        assert_eq!(parse("42").unwrap(), json!(42));
        assert_eq!(parse("3.14").unwrap(), json!(3.14));
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
}
