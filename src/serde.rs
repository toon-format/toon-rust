use std::io::{Read, Write};

use ::serde::{de::DeserializeOwned, Serialize};

use crate::types::ToonResult;
use crate::{decode, encode, DecodeOptions, EncodeOptions, ToonError};

/// Serialize a value to a TOON string using default options.
///
/// # Examples
/// ```
/// use toon_format::to_string;
/// let toon = to_string(&serde_json::json!({"a": 1}))?;
/// assert!(toon.contains("a: 1"));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn to_string<T: Serialize>(value: &T) -> ToonResult<String> {
    encode(value, &EncodeOptions::default())
}

/// Serialize a value to a TOON string using custom options.
///
/// # Examples
/// ```
/// use toon_format::{to_string_with_options, Delimiter, EncodeOptions};
/// let opts = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
/// let toon = to_string_with_options(&serde_json::json!({"items": ["a", "b"]}), &opts)?;
/// assert!(toon.contains('|'));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn to_string_with_options<T: Serialize>(value: &T, opts: &EncodeOptions) -> ToonResult<String> {
    encode(value, opts)
}

/// Serialize a value to a UTF-8 byte vector using default options.
///
/// # Examples
/// ```
/// use toon_format::to_vec;
/// let bytes = to_vec(&serde_json::json!({"a": 1}))?;
/// assert!(!bytes.is_empty());
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn to_vec<T: Serialize>(value: &T) -> ToonResult<Vec<u8>> {
    Ok(to_string(value)?.into_bytes())
}

/// Serialize a value to a writer using default options.
///
/// # Examples
/// ```
/// use toon_format::to_writer;
/// let mut buffer = Vec::new();
/// to_writer(&mut buffer, &serde_json::json!({"a": 1}))?;
/// assert!(!buffer.is_empty());
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn to_writer<T: Serialize, W: Write>(mut writer: W, value: &T) -> ToonResult<()> {
    let encoded = to_string(value)?;
    writer
        .write_all(encoded.as_bytes())
        .map_err(|err| ToonError::InvalidInput(format!("Failed to write output: {err}")))
}

/// Serialize a value to a writer using custom options.
///
/// # Examples
/// ```
/// use toon_format::{to_writer_with_options, Delimiter, EncodeOptions};
/// let opts = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
/// let mut buffer = Vec::new();
/// to_writer_with_options(&mut buffer, &serde_json::json!({"items": ["a", "b"]}), &opts)?;
/// assert!(std::str::from_utf8(&buffer).expect("valid UTF-8").contains('|'));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn to_writer_with_options<T: Serialize, W: Write>(
    mut writer: W,
    value: &T,
    opts: &EncodeOptions,
) -> ToonResult<()> {
    let encoded = to_string_with_options(value, opts)?;
    writer
        .write_all(encoded.as_bytes())
        .map_err(|err| ToonError::InvalidInput(format!("Failed to write output: {err}")))
}

/// Deserialize a value from a TOON string using default options.
///
/// # Examples
/// ```
/// use toon_format::from_str;
/// let value: serde_json::Value = from_str("a: 1")?;
/// assert_eq!(value, serde_json::json!({"a": 1}));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn from_str<T: DeserializeOwned>(input: &str) -> ToonResult<T> {
    decode(input, &DecodeOptions::default())
}

/// Deserialize a value from a TOON string using custom options.
///
/// # Examples
/// ```
/// use toon_format::{from_str_with_options, DecodeOptions};
/// let opts = DecodeOptions::new().with_strict(false);
/// let value: serde_json::Value = from_str_with_options("items[2]: a", &opts)?;
/// assert_eq!(value, serde_json::json!({"items": ["a"]}));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn from_str_with_options<T: DeserializeOwned>(
    input: &str,
    opts: &DecodeOptions,
) -> ToonResult<T> {
    decode(input, opts)
}

/// Deserialize a value from UTF-8 bytes using default options.
///
/// # Examples
/// ```
/// use toon_format::from_slice;
/// let value: serde_json::Value = from_slice(b"a: 1")?;
/// assert_eq!(value, serde_json::json!({"a": 1}));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn from_slice<T: DeserializeOwned>(input: &[u8]) -> ToonResult<T> {
    let s = std::str::from_utf8(input)
        .map_err(|err| ToonError::InvalidInput(format!("Input is not valid UTF-8: {err}")))?;
    from_str(s)
}

/// Deserialize a value from UTF-8 bytes using custom options.
///
/// # Examples
/// ```
/// use toon_format::{from_slice_with_options, DecodeOptions};
/// let opts = DecodeOptions::new().with_strict(false);
/// let value: serde_json::Value = from_slice_with_options(b"items[2]: a", &opts)?;
/// assert_eq!(value, serde_json::json!({"items": ["a"]}));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn from_slice_with_options<T: DeserializeOwned>(
    input: &[u8],
    opts: &DecodeOptions,
) -> ToonResult<T> {
    let s = std::str::from_utf8(input)
        .map_err(|err| ToonError::InvalidInput(format!("Input is not valid UTF-8: {err}")))?;
    from_str_with_options(s, opts)
}

/// Deserialize a value from a reader using default options.
///
/// # Examples
/// ```
/// use std::io::Cursor;
/// use toon_format::from_reader;
/// let mut reader = Cursor::new("a: 1");
/// let value: serde_json::Value = from_reader(&mut reader)?;
/// assert_eq!(value, serde_json::json!({"a": 1}));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn from_reader<T: DeserializeOwned, R: Read>(mut reader: R) -> ToonResult<T> {
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .map_err(|err| ToonError::InvalidInput(format!("Failed to read input: {err}")))?;
    from_slice(&buf)
}

/// Deserialize a value from a reader using custom options.
///
/// # Examples
/// ```
/// use std::io::Cursor;
/// use toon_format::{from_reader_with_options, DecodeOptions};
/// let opts = DecodeOptions::new().with_strict(false);
/// let mut reader = Cursor::new("items[2]: a");
/// let value: serde_json::Value = from_reader_with_options(&mut reader, &opts)?;
/// assert_eq!(value, serde_json::json!({"items": ["a"]}));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn from_reader_with_options<T: DeserializeOwned, R: Read>(
    mut reader: R,
    opts: &DecodeOptions,
) -> ToonResult<T> {
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .map_err(|err| ToonError::InvalidInput(format!("Failed to read input: {err}")))?;
    from_slice_with_options(&buf, opts)
}
