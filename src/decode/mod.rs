//! Decoder Implementation
#[cfg(feature = "layout")]
pub(crate) mod layout_builder;
pub(crate) mod line;
pub mod parser;

#[cfg(feature = "layout")]
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
    serde_json::from_value(value)
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
///     Indent,
/// };
///
/// let options = DecodeOptions::new()
///     .with_strict(true)
///     .with_indent(Indent::Spaces(4));
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

/// Decode with default options (strict mode enabled).
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

/// Decode a TOON document and return the value alongside layout metadata
/// describing how the document was actually written on the wire.
///
/// Available only when the `layout` cargo feature is enabled.
///
/// **Experimental.** This API supports independent exploration of schema
/// and tooling use cases and is not part of the TOON specification.
/// See [`crate::layout`] for the metadata types.
///
/// # Examples
///
/// ```
/// use toon_format::{
///     decode_with_layout,
///     DecodeOptions,
///     NodeLayout,
/// };
///
/// let toon = "users[2]{id,name}:\n  1,Alice\n  2,Bob";
/// let (_value, layout) = decode_with_layout(toon, &DecodeOptions::default())?;
///
/// assert!(matches!(
///     layout.get("/users"),
///     Some(NodeLayout::Tabular { .. })
/// ));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
#[cfg(feature = "layout")]
pub fn decode_with_layout(
    input: &str,
    options: &DecodeOptions,
) -> ToonResult<(Value, crate::layout::Layout)> {
    let mut parser = parser::Parser::new(input, options.clone())?.with_layout();
    let value = parser.parse()?;
    let layout = parser.take_layout().unwrap_or_default();
    Ok((value, layout))
}

#[cfg(test)]
mod tests {
    use serde_json::{
        json,
        Value,
    };

    use super::*;

    #[test]
    fn test_decode_simple_object() {
        let result: Value = decode_default("name: Alice\nage: 30").unwrap();
        assert_eq!(result, json!({"name": "Alice", "age": 30}));
    }

    #[test]
    fn test_decode_nested_object() {
        let input = "user:\n  name: Alice\n  age: 30";
        let result: Value = decode_default(input).unwrap();
        assert_eq!(result, json!({"user": {"name": "Alice", "age": 30}}));
    }

    #[test]
    fn test_decode_inline_array() {
        let result: Value = decode_default("tags[3]: a,b,c").unwrap();
        assert_eq!(result, json!({"tags": ["a", "b", "c"]}));
    }

    #[test]
    fn test_decode_tabular_array() {
        let input = "users[2]{id,name}:\n  1,Alice\n  2,Bob";
        let result: Value = decode_default(input).unwrap();
        assert_eq!(
            result,
            json!({"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]})
        );
    }

    #[test]
    fn test_decode_root_primitive() {
        let result: Value = decode_default("hello").unwrap();
        assert_eq!(result, json!("hello"));

        let result: Value = decode_default("42").unwrap();
        assert_eq!(result, json!(42));

        let result: Value = decode_default("true").unwrap();
        assert_eq!(result, json!(true));
    }

    #[test]
    fn test_decode_root_array() {
        let result: Value = decode_default("[3]: a,b,c").unwrap();
        assert_eq!(result, json!(["a", "b", "c"]));
    }

    #[test]
    fn test_decode_empty_document() {
        let result: Value = decode_default("").unwrap();
        assert_eq!(result, json!({}));
    }

    #[test]
    fn test_decode_strict_length_mismatch() {
        assert!(decode_strict::<Value>("items[3]: a,b").is_err());
    }

    #[test]
    fn test_decode_non_strict_length_mismatch() {
        let opts = DecodeOptions::new().with_strict(false);
        let result: Value = decode("items[3]: a,b", &opts).unwrap();
        assert_eq!(result, json!({"items": ["a", "b"]}));
    }

    #[test]
    fn test_decode_excludes_only_the_crlf_carriage_return() {
        // Lines are split on LF alone and exactly one trailing CR is dropped,
        // so a second CR is content rather than a line terminator.
        let result: Value = decode_default("k: 1\r\n").unwrap();
        assert_eq!(result, json!({"k": 1}));

        let result: Value = decode_default("k: 1\r\r\n").unwrap();
        assert_eq!(result, json!({"k": "1\r"}));

        // A CR inside a line is never a terminator.
        let result: Value = decode_default("k: a\rb").unwrap();
        assert_eq!(result, json!({"k": "a\rb"}));
    }

    #[test]
    fn test_decode_rejects_zero_indent_size() {
        use crate::types::{
            Indent,
            ToonError,
        };

        // Indentation depth is `indent / indent_size`, so a zero indent size
        // would divide by zero. The parser rejects it up front, before any
        // line is read, in both strict and non-strict mode.
        for strict in [true, false] {
            let opts = DecodeOptions::new()
                .with_strict(strict)
                .with_indent(Indent::Spaces(0));

            for input in ["name: Alice", "user:\n  name: Alice", ""] {
                let err = decode::<Value>(input, &opts)
                    .expect_err("zero indent size must be rejected, not panic");
                assert!(
                    matches!(err, ToonError::InvalidInput(_)),
                    "expected InvalidInput for strict={strict}, input={input:?}, got: {err:?}"
                );
            }
        }
    }
}
