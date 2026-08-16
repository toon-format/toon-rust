use crate::utils::literal;

/// Escape special characters in a string for quoted output (§7.1).
///
/// Rows are matched top-to-bottom: backslash, quote, `\n`, `\r`, `\t`, then
/// other U+0000–U+001F controls as lowercase `\uXXXX`.
pub fn escape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());

    for ch in s.chars() {
        match ch {
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\u{0000}'..='\u{001F}' => {
                result.push_str(&format!("\\u{:04x}", ch as u32));
            }
            _ => result.push(ch),
        }
    }

    result
}

/// Unescape special characters in a quoted string (§7.1).
///
/// Valid escape sequences are `\\`, `\"`, `\n`, `\r`, `\t`, and `\uXXXX`
/// with exactly four case-insensitive hex digits. `\uXXXX` escapes encoding
/// surrogate code points (U+D800–U+DFFF) are rejected: supplementary code
/// points appear as literal UTF-8, never as surrogate escapes.
///
/// # Errors
///
/// Returns an error for any other escape sequence, a truncated `\u` escape,
/// or a backslash at the end of the string.
pub fn unescape_string(s: &str) -> Result<String, String> {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();

    while let Some(ch) = chars.next() {
        if ch != '\\' {
            result.push(ch);
            continue;
        }

        match chars.next() {
            Some('n') => result.push('\n'),
            Some('r') => result.push('\r'),
            Some('t') => result.push('\t'),
            Some('"') => result.push('"'),
            Some('\\') => result.push('\\'),
            Some('u') => {
                let mut code = 0u32;
                for _ in 0..4 {
                    let digit = chars
                        .next()
                        .and_then(|c| c.to_digit(16))
                        .ok_or_else(|| {
                            "Invalid escape sequence: \\u must be followed by 4 hex digits"
                                .to_string()
                        })?;
                    code = code * 16 + digit;
                }
                if (0xD800..=0xDFFF).contains(&code) {
                    return Err(format!(
                        "Invalid escape sequence: \\u{code:04x} is a lone surrogate. \
                         Supplementary code points MUST appear as literal UTF-8"
                    ));
                }
                result.push(char::from_u32(code).expect("non-surrogate BMP code point"));
            }
            Some(other) => {
                return Err(format!(
                    "Invalid escape sequence '\\{other}'. Only \\\\, \\\", \\n, \\r, \\t, and \
                     \\uXXXX are valid"
                ));
            }
            None => {
                return Err("Unterminated escape sequence at end of string".to_string());
            }
        }
    }

    Ok(result)
}

/// Check if a key can be written without quotes (§7.3):
/// `^[A-Za-z_][A-Za-z0-9_.]*$`. The pattern is ASCII-only, so every
/// non-ASCII key is quoted.
pub fn is_valid_unquoted_key(key: &str) -> bool {
    let bytes = key.as_bytes();
    let Some(&first) = bytes.first() else {
        return false;
    };

    if !first.is_ascii_alphabetic() && first != b'_' {
        return false;
    }

    bytes[1..]
        .iter()
        .all(|&b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.')
}

/// Determine if a string value needs quoting per §7.2.
///
/// `delimiter` is the relevant delimiter for the position (§11.1): the
/// active delimiter for inline array values, tabular row cells, and keyed
/// entry-row cells; the document delimiter for object field values.
pub fn needs_quoting(s: &str, delimiter: char) -> bool {
    if s.is_empty() {
        return true;
    }

    // Leading or trailing space or tab (exactly U+0020 / U+0009).
    if s.starts_with([' ', '\t']) || s.ends_with([' ', '\t']) {
        return true;
    }

    if literal::is_literal_like(s) {
        return true;
    }

    // Colons, quotes, backslashes, brackets, and braces (§7.2).
    if s.contains([':', '"', '\\', '[', ']', '{', '}']) {
        return true;
    }

    // Control characters U+0000–U+001F.
    if s.chars().any(|c| c <= '\u{001F}') {
        return true;
    }

    if s.contains(delimiter) {
        return true;
    }

    // A leading hyphen would read as a list-item marker; a leading number
    // sign would read as a comment line (§7.2).
    if s.starts_with('-') || s.starts_with('#') {
        return true;
    }

    false
}

/// Quote and escape a string.
pub fn quote_string(s: &str) -> String {
    format!("\"{}\"", escape_string(s))
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
        assert_eq!(escape_string("a\u{0004}b"), "a\\u0004b");
        assert_eq!(escape_string("x\u{001F}y"), "x\\u001fy");
    }

    #[test]
    fn test_unescape_string() {
        assert_eq!(unescape_string("hello").unwrap(), "hello");
        assert_eq!(unescape_string("hello\\nworld").unwrap(), "hello\nworld");
        assert_eq!(unescape_string("say \\\"hi\\\"").unwrap(), "say \"hi\"");
        assert_eq!(unescape_string("back\\\\slash").unwrap(), "back\\slash");
        assert_eq!(unescape_string("tab\\there").unwrap(), "tab\there");
        assert_eq!(unescape_string("return\\rhere").unwrap(), "return\rhere");
    }

    #[test]
    fn test_unescape_unicode_escapes() {
        assert_eq!(unescape_string("a\\u0004b").unwrap(), "a\u{0004}b");
        assert_eq!(unescape_string("a\\u00E9b").unwrap(), "aéb");
        assert_eq!(unescape_string("a\\u00e9b").unwrap(), "aéb");

        // Lone surrogates are rejected.
        assert!(unescape_string("\\ud800").is_err());
        assert!(unescape_string("\\uDFFF").is_err());

        // Truncated \u escapes are rejected.
        assert!(unescape_string("\\u12").is_err());
        assert!(unescape_string("\\u12g4").is_err());
    }

    #[test]
    fn test_unescape_string_invalid_escapes() {
        assert!(unescape_string("bad\\xescape").is_err());
        assert!(unescape_string("bad\\0escape").is_err());
        assert!(unescape_string("bad\\aescape").is_err());

        // Unterminated escape at end
        assert!(unescape_string("ends\\").is_err());
    }

    #[test]
    fn test_needs_quoting() {
        let comma = ',';

        assert!(needs_quoting("", comma));

        assert!(needs_quoting("true", comma));
        assert!(needs_quoting("false", comma));
        assert!(needs_quoting("null", comma));
        assert!(needs_quoting("123", comma));
        assert!(needs_quoting("+1", comma));

        assert!(needs_quoting("hello[world]", comma));
        assert!(needs_quoting("key:value", comma));

        assert!(needs_quoting("a,b", comma));
        assert!(!needs_quoting("a,b", '|'));

        assert!(!needs_quoting("hello world", comma));
        assert!(needs_quoting(" hello", comma));
        assert!(needs_quoting("hello ", comma));

        assert!(needs_quoting("#", comma));
        assert!(needs_quoting("#hello", comma));
        assert!(needs_quoting("-", comma));
        assert!(needs_quoting("-dash", comma));

        assert!(!needs_quoting("hello", comma));
    }

    #[test]
    fn test_quote_string() {
        assert_eq!(quote_string("hello"), "\"hello\"");
        assert_eq!(quote_string("hello\nworld"), "\"hello\\nworld\"");
    }

    #[test]
    fn test_is_valid_unquoted_key() {
        assert!(is_valid_unquoted_key("normal_key"));
        assert!(is_valid_unquoted_key("key123"));
        assert!(is_valid_unquoted_key("key.value"));
        assert!(is_valid_unquoted_key("_private"));
        assert!(is_valid_unquoted_key("KeyName"));
        assert!(is_valid_unquoted_key("a"));
        assert!(is_valid_unquoted_key("_"));
        assert!(is_valid_unquoted_key("key."));

        assert!(!is_valid_unquoted_key(""));
        assert!(!is_valid_unquoted_key("123"));
        assert!(!is_valid_unquoted_key("key:value"));
        assert!(!is_valid_unquoted_key("key-value"));
        assert!(!is_valid_unquoted_key("key value"));
        assert!(!is_valid_unquoted_key(".key"));
        assert!(!is_valid_unquoted_key("key[value]"));
        assert!(!is_valid_unquoted_key("key{value}"));

        // The §7.3 pattern is ASCII-only: non-ASCII keys are quoted.
        assert!(!is_valid_unquoted_key("café"));
        assert!(!is_valid_unquoted_key("名前"));
    }
}
