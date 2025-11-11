//! Decoder Implementation
pub mod parser;
pub mod scanner;
pub mod validation;

use serde_json::Value;

use crate::types::{
    DecodeOptions,
    ToonResult,
};

/// Decode a TOON string to a JSON value with custom options.
///
/// # Examples
///
/// ```
/// use serde_json::json;
/// use toon_format::{
///     decode,
///     DecodeOptions,
///     Delimiter,
/// };
///
/// let input = "name: Alice\nage: 30";
/// let options = DecodeOptions::new().with_strict(false);
/// let result = decode(input, &options)?;
/// assert_eq!(result["name"], json!("Alice"));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn decode(input: &str, options: &DecodeOptions) -> ToonResult<Value> {
    let mut parser = parser::Parser::new(input, options.clone())?;
    parser.parse()
}

/// Decode with strict validation enabled (validates array lengths,
/// indentation).
///
/// # Examples
///
/// ```
/// use serde_json::json;
/// use toon_format::decode_strict;
///
/// // Valid array length
/// let result = decode_strict("items[2]: a,b")?;
/// assert_eq!(result["items"], json!(["a", "b"]));
///
/// // Invalid array length (will error)
/// assert!(decode_strict("items[3]: a,b").is_err());
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn decode_strict(input: &str) -> ToonResult<Value> {
    decode(input, &DecodeOptions::new().with_strict(true))
}

/// Decode with strict validation and additional options.
///
/// # Examples
///
/// ```
/// use serde_json::json;
/// use toon_format::{
///     decode_strict_with_options,
///     DecodeOptions,
/// };
///
/// let options = DecodeOptions::new()
///     .with_strict(true)
///     .with_delimiter(toon_format::Delimiter::Pipe);
/// let result = decode_strict_with_options("items[2|]: a|b", &options)?;
/// assert_eq!(result["items"], json!(["a", "b"]));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn decode_strict_with_options(input: &str, options: &DecodeOptions) -> ToonResult<Value> {
    let opts = options.clone().with_strict(true);
    decode(input, &opts)
}

/// Decode without type coercion (strings remain strings).
///
/// # Examples
///
/// ```
/// use serde_json::json;
/// use toon_format::decode_no_coerce;
///
/// // Without coercion: quoted strings that look like numbers stay as strings
/// let result = decode_no_coerce("value: \"123\"")?;
/// assert_eq!(result["value"], json!("123"));
///
/// // With default coercion: unquoted "true" becomes boolean
/// let result = toon_format::decode_default("value: true")?;
/// assert_eq!(result["value"], json!(true));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn decode_no_coerce(input: &str) -> ToonResult<Value> {
    decode(input, &DecodeOptions::new().with_coerce_types(false))
}

/// Decode without type coercion and with additional options.
///
/// # Examples
///
/// ```
/// use serde_json::json;
/// use toon_format::{
///     decode_no_coerce_with_options,
///     DecodeOptions,
/// };
///
/// let options = DecodeOptions::new()
///     .with_coerce_types(false)
///     .with_strict(false);
/// let result = decode_no_coerce_with_options("value: \"123\"", &options)?;
/// assert_eq!(result["value"], json!("123"));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn decode_no_coerce_with_options(input: &str, options: &DecodeOptions) -> ToonResult<Value> {
    let opts = options.clone().with_coerce_types(false);
    decode(input, &opts)
}

/// Decode with default options (strict mode, type coercion enabled).
///
/// # Examples
///
/// ```
/// use serde_json::json;
/// use toon_format::decode_default;
///
/// // Simple object
/// let input = "name: Alice\nage: 30";
/// let result = decode_default(input)?;
/// assert_eq!(result["name"], json!("Alice"));
/// assert_eq!(result["age"], json!(30));
///
/// // Primitive array
/// let input = "tags[3]: reading,gaming,coding";
/// let result = decode_default(input)?;
/// assert_eq!(result["tags"], json!(["reading", "gaming", "coding"]));
///
/// // Tabular array
/// let input = "users[2]{id,name}:\n  1,Alice\n  2,Bob";
/// let result = decode_default(input)?;
/// assert_eq!(result["users"][0]["name"], json!("Alice"));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn decode_default(input: &str) -> ToonResult<Value> {
    decode(input, &DecodeOptions::default())
}

#[cfg(test)]
mod tests {
    use core::f64;

    use serde_json::json;

    use super::*;

    #[test]
    fn test_decode_null() {
        assert_eq!(decode_default("null").unwrap(), json!(null));
    }

    #[test]
    fn test_decode_bool() {
        assert_eq!(decode_default("true").unwrap(), json!(true));
        assert_eq!(decode_default("false").unwrap(), json!(false));
    }

    #[test]
    fn test_decode_number() {
        assert_eq!(decode_default("42").unwrap(), json!(42));
        assert_eq!(
            decode_default("3.141592653589793").unwrap(),
            json!(f64::consts::PI)
        );
        assert_eq!(decode_default("-5").unwrap(), json!(-5));
    }

    #[test]
    fn test_decode_string() {
        assert_eq!(decode_default("hello").unwrap(), json!("hello"));
        assert_eq!(
            decode_default("\"hello world\"").unwrap(),
            json!("hello world")
        );
    }

    #[test]
    fn test_decode_simple_object() {
        let input = "name: Alice\nage: 30";
        let result = decode_default(input).unwrap();
        assert_eq!(result["name"], json!("Alice"));
        assert_eq!(result["age"], json!(30));
    }

    #[test]
    fn test_decode_primitive_array() {
        let input = "tags[3]: reading,gaming,coding";
        let result = decode_default(input).unwrap();
        assert_eq!(result["tags"], json!(["reading", "gaming", "coding"]));
    }

    #[test]
    fn test_decode_tabular_array() {
        let input = "users[2]{id,name,role}:\n  1,Alice,admin\n  2,Bob,user";
        let result = decode_default(input).unwrap();
        assert_eq!(
            result["users"],
            json!([
                {"id": 1, "name": "Alice", "role": "admin"},
                {"id": 2, "name": "Bob", "role": "user"}
            ])
        );
    }

    #[test]
    fn test_decode_empty_array() {
        let input = "items[0]:";
        let result = decode_default(input).unwrap();
        assert_eq!(result["items"], json!([]));
    }

    #[test]
    fn test_decode_quoted_strings() {
        let input = "tags[3]: \"true\",\"42\",\"-3.14\"";
        let result = decode_default(input).unwrap();
        assert_eq!(result["tags"], json!(["true", "42", "-3.14"]));
    }
}
