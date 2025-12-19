use crate::{
    types::Delimiter,
    utils::literal,
};

/// Escape special characters in a string for quoted output.
pub fn escape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());

    for ch in s.chars() {
        match ch {
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            _ => result.push(ch),
        }
    }

    result
}

/// Unescape special characters in a quoted string.
///
/// Per TOON spec §7.1, only these escape sequences are valid:
/// - `\\` → `\`
/// - `\"` → `"`
/// - `\n` → newline
/// - `\r` → carriage return
/// - `\t` → tab
///
/// Any other escape sequence MUST cause an error.
///
/// # Errors
///
/// Returns an error if the string contains an invalid escape sequence
/// or if a backslash appears at the end of the string.
pub fn unescape_string(s: &str) -> Result<String, String> {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    let mut position = 0;

    while let Some(ch) = chars.next() {
        position += 1;

        if ch == '\\' {
            if let Some(&next) = chars.peek() {
                match next {
                    'n' => {
                        result.push('\n');
                        chars.next(); // consume the 'n'
                        position += 1;
                    }
                    'r' => {
                        result.push('\r');
                        chars.next();
                        position += 1;
                    }
                    't' => {
                        result.push('\t');
                        chars.next();
                        position += 1;
                    }
                    '"' => {
                        result.push('"');
                        chars.next();
                        position += 1;
                    }
                    '\\' => {
                        result.push('\\');
                        chars.next();
                        position += 1;
                    }
                    _ => {
                        return Err(format!(
                            "Invalid escape sequence '\\{next}' at position {position}. Only \
                             \\\\, \\\", \\n, \\r, \\t are valid",
                        ));
                    }
                }
            } else {
                return Err(format!(
                    "Unterminated escape sequence at end of string (position {position})",
                ));
            }
        } else {
            result.push(ch);
        }
    }

    Ok(result)
}

/// Check if a key can be written without quotes (alphanumeric, underscore,
/// dot).
pub fn is_valid_unquoted_key(key: &str) -> bool {
    if key.is_empty() {
        return false;
    }

    let mut chars = key.chars();
    let first = if let Some(c) = chars.next() {
        c
    } else {
        return false
    };

    if !first.is_alphabetic() && first != '_' {
        return false;
    }

    chars.all(|c| c.is_alphanumeric() || c == '_' || c == '.')
}

/// Determine if a string needs quoting based on content and delimiter.
pub fn needs_quoting(s: &str, delimiter: char) -> bool {
    if s.is_empty() {
        return true;
    }

    if literal::is_literal_like(s) {
        return true;
    }

    if s.chars().any(literal::is_structural_char) {
        return true;
    }

    if s.contains('\\') || s.contains('"') {
        return true;
    }

    if s.contains(delimiter) {
        return true;
    }

    if s.contains('\n') || s.contains('\r') || s.contains('\t') {
        return true;
    }

    if s.starts_with(char::is_whitespace) || s.ends_with(char::is_whitespace) {
        return true;
    }

    if s.starts_with('-') {
        return true;
    }

    // Check for leading zeros (e.g., "05", "007", "0123")
    // Numbers with leading zeros must be quoted
    if s.starts_with('0') && s.len() > 1 && s.chars().nth(1).is_some_and(|c| c.is_ascii_digit()) {
        return true;
    }

    false
}

/// Quote and escape a string.
pub fn quote_string(s: &str) -> String {
    format!("\"{}\"", escape_string(s))
}

pub fn split_by_delimiter(s: &str, delimiter: Delimiter) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let chars = s.chars().peekable();
    let delim_char = delimiter.as_char();

    for ch in chars {
        if ch == '"' && (current.is_empty() || !current.ends_with('\\')) {
            in_quotes = !in_quotes;
            current.push(ch);
        } else if ch == delim_char && !in_quotes {
            result.push(current.trim().to_string());
            current.clear();
        } else {
            current.push(ch);
        }
    }

    if !current.is_empty() {
        result.push(current.trim().to_string());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_string() {
        assert_eq!(escape_string("hello"), "hello");
        assert_eq!(escape_string("hello\nworld"), "hello\\nworld");
        assert_eq!(escape_string("say \"hi\""), "say \\\"hi\\\"");
        assert_eq!(escape_string("back\\slash"), "back\\\\slash");
    }

    #[test]
    fn test_unescape_string() {
        // Valid escapes
        assert_eq!(unescape_string("hello").unwrap(), "hello");
        assert_eq!(unescape_string("hello\\nworld").unwrap(), "hello\nworld");
        assert_eq!(unescape_string("say \\\"hi\\\"").unwrap(), "say \"hi\"");
        assert_eq!(unescape_string("back\\\\slash").unwrap(), "back\\slash");
        assert_eq!(unescape_string("tab\\there").unwrap(), "tab\there");
        assert_eq!(unescape_string("return\\rhere").unwrap(), "return\rhere");
    }

    #[test]
    fn test_unescape_string_invalid_escapes() {
        // Invalid escape sequences should error
        assert!(unescape_string("bad\\xescape").is_err());
        assert!(unescape_string("bad\\uescape").is_err());
        assert!(unescape_string("bad\\0escape").is_err());
        assert!(unescape_string("bad\\aescape").is_err());

        // Unterminated escape at end
        assert!(unescape_string("ends\\").is_err());
    }

    #[test]
    fn test_unescape_string_error_messages() {
        let result = unescape_string("bad\\x");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Invalid escape sequence"));
        assert!(err.contains("\\x"));
    }

    #[test]
    fn test_needs_quoting() {
        let comma = Delimiter::Comma.as_char();

        assert!(needs_quoting("", comma));

        assert!(needs_quoting("true", comma));
        assert!(needs_quoting("false", comma));
        assert!(needs_quoting("null", comma));
        assert!(needs_quoting("123", comma));

        assert!(needs_quoting("hello[world]", comma));
        assert!(needs_quoting("key:value", comma));

        assert!(needs_quoting("a,b", comma));
        assert!(!needs_quoting("a,b", Delimiter::Pipe.as_char()));

        assert!(!needs_quoting("hello world", comma));
        assert!(needs_quoting(" hello", comma));
        assert!(needs_quoting("hello ", comma));

        assert!(!needs_quoting("hello", comma));
        assert!(!needs_quoting("world", comma));
        assert!(!needs_quoting("helloworld", comma));
    }

    #[test]
    fn test_quote_string() {
        assert_eq!(quote_string("hello"), "\"hello\"");
        assert_eq!(quote_string("hello\nworld"), "\"hello\\nworld\"");
    }

    #[test]
    fn test_split_by_delimiter() {
        let comma = Delimiter::Comma;

        assert_eq!(split_by_delimiter("a,b,c", comma), vec!["a", "b", "c"]);

        assert_eq!(split_by_delimiter("a, b, c", comma), vec!["a", "b", "c"]);

        assert_eq!(split_by_delimiter("\"a,b\",c", comma), vec!["\"a,b\"", "c"]);
    }

    #[test]
    fn test_is_valid_unquoted_key() {
        // Valid keys (should return true)
        assert!(is_valid_unquoted_key("normal_key"));
        assert!(is_valid_unquoted_key("key123"));
        assert!(is_valid_unquoted_key("key.value"));
        assert!(is_valid_unquoted_key("_private"));
        assert!(is_valid_unquoted_key("KeyName"));
        assert!(is_valid_unquoted_key("key_name"));
        assert!(is_valid_unquoted_key("key.name.sub"));
        assert!(is_valid_unquoted_key("a"));
        assert!(is_valid_unquoted_key("_"));
        assert!(is_valid_unquoted_key("key_123.value"));

        assert!(!is_valid_unquoted_key(""));
        assert!(!is_valid_unquoted_key("123"));
        assert!(!is_valid_unquoted_key("key:value"));
        assert!(!is_valid_unquoted_key("key-value"));
        assert!(!is_valid_unquoted_key("key value"));
        assert!(!is_valid_unquoted_key(".key"));
        assert!(is_valid_unquoted_key("key.value.sub."));
        assert!(is_valid_unquoted_key("key."));
        assert!(!is_valid_unquoted_key("key[value]"));
        assert!(!is_valid_unquoted_key("key{value}"));
    }
}
