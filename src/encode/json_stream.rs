//! Reader/writer conveniences for JSON → TOON encoding.
//!
//! Spec v4.1 selects the form from the value's shape (§1.4, §9): tabular and
//! keyed tabular headers depend on whole-subtree analysis, so no header can
//! be emitted before its entire subtree has been read. These functions
//! therefore parse the full JSON input and run the in-memory encoder; they
//! exist as I/O conveniences, not as bounded-memory streaming.

use std::io::{
    Read,
    Write,
};

use serde_json::Value as SerdeValue;

use crate::types::{
    EncodeOptions,
    JsonValue,
    ToonError,
    ToonResult,
};

/// Encode JSON from a reader into a TOON `String`.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "json_stream")]
/// # fn main() -> Result<(), toon_format::ToonError> {
/// use toon_format::{
///     encode_json_reader,
///     EncodeOptions,
/// };
///
/// let json = br#"{"name":"Alice","age":30}"#;
/// let toon = encode_json_reader(&json[..], &EncodeOptions::default())?;
/// assert_eq!(toon, "name: Alice\nage: 30");
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "json_stream"))]
/// # fn main() {}
/// ```
pub fn encode_json_reader<R: Read>(reader: R, options: &EncodeOptions) -> ToonResult<String> {
    let value: SerdeValue = serde_json::from_reader(reader)
        .map_err(|e| ToonError::SerializationError(e.to_string()))?;
    super::encode_impl(&JsonValue::from(value), options)
}

/// Encode JSON from a reader into a TOON `String` using default options.
pub fn encode_json_reader_default<R: Read>(reader: R) -> ToonResult<String> {
    encode_json_reader(reader, &EncodeOptions::default())
}

/// Encode JSON from a reader into TOON, writing the result to the supplied
/// writer.
pub fn encode_json_stream<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    options: &EncodeOptions,
) -> ToonResult<()> {
    let output = encode_json_reader(reader, options)?;
    writer
        .write_all(output.as_bytes())
        .map_err(|e| ToonError::SerializationError(e.to_string()))
}

/// Encode JSON from a reader into TOON, writing to the supplied writer using
/// default options.
pub fn encode_json_stream_default<R: Read, W: Write>(reader: R, writer: W) -> ToonResult<()> {
    encode_json_stream(reader, writer, &EncodeOptions::default())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        encode,
        encode_default,
        types::Delimiter,
    };

    #[test]
    fn test_reader_root_object_matches_in_memory() {
        let input = br#"{"name":"Alice","age":30,"tags":["a","b"]}"#;
        let options = EncodeOptions::default();

        let from_reader = encode_json_reader(&input[..], &options).unwrap();
        let in_memory =
            encode(&json!({"name":"Alice","age":30,"tags":["a","b"]}), &options).unwrap();

        assert_eq!(from_reader, in_memory);
    }

    #[test]
    fn test_reader_default_uses_default_options() {
        let input = br#"{"name":"Alice","age":30}"#;

        let from_reader = encode_json_reader_default(&input[..]).unwrap();
        let in_memory = encode_default(&json!({"name":"Alice","age":30})).unwrap();

        assert_eq!(from_reader, in_memory);
    }

    #[test]
    fn test_reader_root_primitive_array_matches_in_memory() {
        let input = br#"["reading","gaming","coding"]"#;
        let options = EncodeOptions::new().with_delimiter(Delimiter::Pipe);

        let from_reader = encode_json_reader(&input[..], &options).unwrap();
        let in_memory = encode(&json!(["reading", "gaming", "coding"]), &options).unwrap();

        assert_eq!(from_reader, in_memory);
    }

    #[test]
    fn test_writer_output_matches_in_memory() {
        let input = br#"[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"}]"#;
        let mut output = Vec::new();

        encode_json_stream_default(&input[..], &mut output).unwrap();

        let from_reader = String::from_utf8(output).unwrap();
        let in_memory = encode_default(&json!([
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"}
        ]))
        .unwrap();

        assert_eq!(from_reader, in_memory);
    }

    #[test]
    fn test_reader_rejects_invalid_json() {
        let input = br#"{"name":"#;
        assert!(encode_json_reader_default(&input[..]).is_err());
    }
}
