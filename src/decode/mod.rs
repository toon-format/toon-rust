//! Decoder Implementation
pub mod expansion;
pub mod parser;
pub mod scanner;
pub mod validation;

use serde_json::Value;

use crate::types::{
    DecodeOptions,
    ToonResult,
};

/// Decode a TOON string into any deserializable type.
///
/// This function accepts any type implementing `serde::Deserialize`, including:
/// - Custom structs with `#[derive(Deserialize)]`
/// - `serde_json::Value`
/// - Built-in types (Vec, HashMap, etc.)
///
/// # Examples
///
/// **With custom structs:**
/// ```
/// use serde::Deserialize;
/// use toon_format::{
///     decode,
///     DecodeOptions,
/// };
///
/// #[derive(Deserialize, Debug, PartialEq)]
/// struct User {
///     name: String,
///     age: u32,
/// }
///
/// let toon = "name: Alice\nage: 30";
/// let user: User = decode(toon, &DecodeOptions::default())?;
/// assert_eq!(user.name, "Alice");
/// assert_eq!(user.age, 30);
/// # Ok::<(), toon_format::ToonError>(())
/// ```
///
/// **With JSON values:**
/// ```
/// use serde_json::{
///     json,
///     Value,
/// };
/// use toon_format::{
///     decode,
///     DecodeOptions,
/// };
///
/// let input = "name: Alice\nage: 30";
/// let result: Value = decode(input, &DecodeOptions::default())?;
/// assert_eq!(result["name"], json!("Alice"));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn decode<T: serde::de::DeserializeOwned>(
    input: &str,
    options: &DecodeOptions,
) -> ToonResult<T> {
    let mut parser = parser::Parser::new(input, options.clone())?;
    let value = parser.parse()?;

    // Apply path expansion if enabled (v1.5 feature)
    use crate::types::PathExpansionMode;
    let final_value = if options.expand_paths != PathExpansionMode::Off {
        let json_value = crate::types::JsonValue::from(value);
        let expanded =
            expansion::expand_paths_recursive(json_value, options.expand_paths, options.strict)?;
        Value::from(expanded)
    } else {
        value
    };

    serde_json::from_value(final_value)
        .map_err(|e| crate::types::ToonError::DeserializationError(e.to_string()))
}

/// Decode with strict validation enabled (validates array lengths,
/// indentation).
///
/// # Examples
///
/// ```
/// use serde_json::{
///     json,
///     Value,
/// };
/// use toon_format::decode_strict;
///
/// // Valid array length
/// let result: Value = decode_strict("items[2]: a,b")?;
/// assert_eq!(result["items"], json!(["a", "b"]));
///
/// // Invalid array length (will error)
/// assert!(decode_strict::<Value>("items[3]: a,b").is_err());
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn decode_strict<T: serde::de::DeserializeOwned>(input: &str) -> ToonResult<T> {
    decode(input, &DecodeOptions::new().with_strict(true))
}

/// Decode with strict validation and additional options.
///
/// # Examples
///
/// ```
/// use serde_json::{
///     json,
///     Value,
/// };
/// use toon_format::{
///     decode_strict_with_options,
///     DecodeOptions,
/// };
///
/// let options = DecodeOptions::new()
///     .with_strict(true)
///     .with_delimiter(toon_format::Delimiter::Pipe);
/// let result: Value = decode_strict_with_options("items[2|]: a|b", &options)?;
/// assert_eq!(result["items"], json!(["a", "b"]));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn decode_strict_with_options<T: serde::de::DeserializeOwned>(
    input: &str,
    options: &DecodeOptions,
) -> ToonResult<T> {
    let opts = options.clone().with_strict(true);
    decode(input, &opts)
}

/// Decode without type coercion (strings remain strings).
///
/// # Examples
///
/// ```
/// use serde_json::{
///     json,
///     Value,
/// };
/// use toon_format::decode_no_coerce;
///
/// // Without coercion: quoted strings that look like numbers stay as strings
/// let result: Value = decode_no_coerce("value: \"123\"")?;
/// assert_eq!(result["value"], json!("123"));
///
/// // With default coercion: unquoted "true" becomes boolean
/// let result: Value = toon_format::decode_default("value: true")?;
/// assert_eq!(result["value"], json!(true));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn decode_no_coerce<T: serde::de::DeserializeOwned>(input: &str) -> ToonResult<T> {
    decode(input, &DecodeOptions::new().with_coerce_types(false))
}

/// Decode without type coercion and with additional options.
///
/// # Examples
///
/// ```
/// use serde_json::{
///     json,
///     Value,
/// };
/// use toon_format::{
///     decode_no_coerce_with_options,
///     DecodeOptions,
/// };
///
/// let options = DecodeOptions::new()
///     .with_coerce_types(false)
///     .with_strict(false);
/// let result: Value = decode_no_coerce_with_options("value: \"123\"", &options)?;
/// assert_eq!(result["value"], json!("123"));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn decode_no_coerce_with_options<T: serde::de::DeserializeOwned>(
    input: &str,
    options: &DecodeOptions,
) -> ToonResult<T> {
    let opts = options.clone().with_coerce_types(false);
    decode(input, &opts)
}

/// Decode with default options (strict mode, type coercion enabled).
///
/// Works with any type implementing `serde::Deserialize`.
///
/// # Examples
///
/// **With structs:**
/// ```
/// use serde::Deserialize;
/// use toon_format::decode_default;
///
/// #[derive(Deserialize)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// let input = "name: Alice\nage: 30";
/// let person: Person = decode_default(input)?;
/// assert_eq!(person.name, "Alice");
/// # Ok::<(), toon_format::ToonError>(())
/// ```
///
/// **With JSON values:**
/// ```
/// use serde_json::{
///     json,
///     Value,
/// };
/// use toon_format::decode_default;
///
/// let input = "tags[3]: reading,gaming,coding";
/// let result: Value = decode_default(input)?;
/// assert_eq!(result["tags"], json!(["reading", "gaming", "coding"]));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn decode_default<T: serde::de::DeserializeOwned>(input: &str) -> ToonResult<T> {
    decode(input, &DecodeOptions::default())
}

#[cfg(test)]
mod tests {
    use core::f64;

    use serde_json::json;

    use super::*;

    #[test]
    fn test_decode_null() {
        assert_eq!(decode_default::<Value>("null").unwrap(), json!(null));
    }

    #[test]
    fn test_decode_bool() {
        assert_eq!(decode_default::<Value>("true").unwrap(), json!(true));
        assert_eq!(decode_default::<Value>("false").unwrap(), json!(false));
    }

    #[test]
    fn test_decode_number() {
        assert_eq!(decode_default::<Value>("42").unwrap(), json!(42));
        assert_eq!(
            decode_default::<Value>("3.141592653589793").unwrap(),
            json!(f64::consts::PI)
        );
        assert_eq!(decode_default::<Value>("-5").unwrap(), json!(-5));
    }

    #[test]
    fn test_decode_string() {
        assert_eq!(decode_default::<Value>("hello").unwrap(), json!("hello"));
        assert_eq!(
            decode_default::<Value>("\"hello world\"").unwrap(),
            json!("hello world")
        );
    }

    #[test]
    fn test_decode_simple_object() {
        let input = "name: Alice\nage: 30";
        let result: Value = decode_default(input).unwrap();
        assert_eq!(result["name"], json!("Alice"));
        assert_eq!(result["age"], json!(30));
    }

    #[test]
    fn test_decode_primitive_array() {
        let input = "tags[3]: reading,gaming,coding";
        let result: Value = decode_default(input).unwrap();
        assert_eq!(result["tags"], json!(["reading", "gaming", "coding"]));
    }

    #[test]
    fn test_decode_tabular_array() {
        let input = "users[2]{id,name,role}:\n  1,Alice,admin\n  2,Bob,user";
        let result: Value = decode_default(input).unwrap();
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
        let result: Value = decode_default(input).unwrap();
        assert_eq!(result["items"], json!([]));
    }

    #[test]
    fn test_decode_quoted_strings() {
        let input = "tags[3]: \"true\",\"42\",\"-3.14\"";
        let result: Value = decode_default(input).unwrap();
        assert_eq!(result["tags"], json!(["true", "42", "-3.14"]));
    }
}
