//! # TOON Format for Rust
//!
//! Token-Oriented Object Notation (TOON) is a compact, human-readable format
//! designed for passing structured data to Large Language Models with
//! significantly reduced token usage.
//!
//! This crate reserves the `toon-format` namespace for the official Rust
//! implementation. Full implementation coming soon!
//!
//! ## Resources
//!
//! - [TOON Specification](https://github.com/johannschopplich/toon/blob/main/SPEC.md)
//! - [Main Repository](https://github.com/johannschopplich/toon)
//! - [Other Implementations](https://github.com/johannschopplich/toon#other-implementations)
//!
//! ## Example Usage (Future)
//!
//! ```ignore
//! use toon_format::{encode, decode};
//!
//! let data = json!({"name": "Alice", "age": 30});
//! let toon_string = encode(&data)?;
//! let decoded = decode(&toon_string)?;
//! # Ok::<(), toon_format::ToonError>(())
//! ```
#![warn(rustdoc::missing_crate_level_docs)]

pub mod constants;
pub mod decode;
pub mod encode;
pub mod types;
pub mod utils;

pub use decode::{
    decode,
    decode_default,
    decode_no_coerce,
    decode_no_coerce_with_options,
    decode_strict,
    decode_strict_with_options,
};
pub use encode::{
    encode,
    encode_array,
    encode_default,
    encode_object,
};
pub use types::{
    DecodeOptions,
    Delimiter,
    EncodeOptions,
    Indent,
    ToonError,
};
pub use utils::{
    literal::{
        is_keyword,
        is_literal_like,
    },
    normalize,
    string::{
        escape_string,
        is_valid_unquoted_key,
        needs_quoting,
    },
};

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        constants::is_keyword,
        decode::{
            decode_default,
            decode_strict,
        },
        encode::{
            encode,
            encode_default,
        },
        types::{
            Delimiter,
            EncodeOptions,
        },
        utils::{
            escape_string,
            is_literal_like,
            needs_quoting,
            normalize,
        },
    };

    #[test]
    fn test_round_trip_simple() {
        let original = json!({"name": "Alice", "age": 30});
        let encoded = encode_default(&original).unwrap();
        let decoded = decode_default(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_round_trip_array() {
        let original = json!({"tags": ["reading", "gaming", "coding"]});
        let encoded = encode_default(&original).unwrap();
        let decoded = decode_default(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_round_trip_tabular() {
        let original = json!({
            "users": [
                {"id": 1, "name": "Alice", "role": "admin"},
                {"id": 2, "name": "Bob", "role": "user"}
            ]
        });
        let encoded = encode_default(&original).unwrap();
        let decoded = decode_default(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_custom_delimiter() {
        let original = json!({"tags": ["a", "b", "c"]});
        let opts = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
        let encoded = encode(&original, &opts).unwrap();
        assert!(encoded.contains("|"));

        let decoded = decode_default(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_decode_strict_helper() {
        let input = "items[2]: a,b";
        assert!(decode_strict(input).is_ok());

        let input = "items[3]: a,b";
        assert!(decode_strict(input).is_err());
    }

    #[test]
    fn test_normalize_exported() {
        let value = json!(f64::NAN);
        let normalized = normalize(value.into());
        assert_eq!(serde_json::Value::from(normalized), json!(null));
    }

    #[test]
    fn test_utilities_exported() {
        assert!(is_keyword("null"));
        assert!(is_literal_like("true"));
        assert_eq!(escape_string("hello\nworld"), "hello\\nworld");
        assert!(needs_quoting("true", Delimiter::Comma.as_char()));
    }
}
