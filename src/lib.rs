//! # TOON Format for Rust
//!
//! Token-Oriented Object Notation (TOON) is a compact, human-readable format
//! designed for passing structured data to Large Language Models with
//! significantly reduced token usage.
//!
//! This crate is the official Rust implementation of TOON, targeting
//! specification v4.1 (`toon-spec: 4.1`).
//!
//! Documented implementation-defined behavior:
//! - Numeric out-of-range policy (§4): integral tokens preserve full
//!   `i64`/`u64` precision; fractional and exponent forms parse as `f64`, with
//!   integer-valued results normalized to integers (`-1E+03` decodes as the
//!   integer `-1000`). A token whose value is not finite in `f64` decodes as a
//!   string.
//! - Host-type normalization (§3) follows `serde::Serialize`; Rust strings are
//!   always well-formed UTF-8, so unpaired surrogates cannot occur.
//! - Decoded objects preserve document key order (`serde_json` with
//!   `preserve_order`), and every key — including `__proto__`, `constructor`,
//!   and `prototype` — is an ordinary map entry (§15).
//! - Nesting depth limit (§15): encoding and decoding recurse over nesting, so
//!   both impose the documented limit of 256 levels — including nested field
//!   groups in headers — and report exceeding it as an error rather than
//!   exhausting the host stack.
//! - Non-strict tab indentation (§12): strict mode rejects tabs in indentation;
//!   non-strict mode accepts them, counting each leading tab as one depth level
//!   and each run of `indentSize` leading spaces as one.
//!
//! ## Resources
//!
//! - [TOON Specification](https://github.com/toon-format/spec/blob/main/SPEC.md)
//! - [Reference Implementation (JS/TS)](https://github.com/toon-format/toon)
//!
//! ## Example Usage
//!
//! ```
//! use serde_json::{json, Value};
//! use toon_format::{encode_default, decode_default};
//!
//! let data = json!({"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]});
//! let toon = encode_default(&data)?;
//! assert_eq!(toon, "users[2]{id,name}:\n  1,Alice\n  2,Bob");
//!
//! let decoded: Value = decode_default(&toon)?;
//! assert_eq!(decoded, data);
//! # Ok::<(), toon_format::ToonError>(())
//! ```
#![warn(rustdoc::missing_crate_level_docs)]

pub mod constants;
pub mod decode;
pub mod encode;
#[cfg(feature = "layout")]
pub mod layout;
#[cfg(feature = "cli")]
pub mod tui;
pub mod types;
pub mod utils;

#[cfg(feature = "layout")]
pub use decode::decode_with_layout;
pub use decode::{
    decode,
    decode_default,
    decode_strict,
    decode_strict_with_options,
};
#[cfg(feature = "json_stream")]
pub use encode::json_stream::{
    encode_json_reader,
    encode_json_reader_default,
    encode_json_stream,
    encode_json_stream_default,
};
pub use encode::{
    encode,
    encode_array,
    encode_default,
    encode_object,
};
#[cfg(feature = "layout")]
pub use layout::{
    FieldDescriptor,
    Layout,
    NodeLayout,
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
        unescape_string,
    },
};

#[cfg(test)]
mod tests {
    use serde_json::{
        json,
        Value,
    };

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
        let decoded: Value = decode_default(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_round_trip_array() {
        let original = json!({"tags": ["reading", "gaming", "coding"]});
        let encoded = encode_default(&original).unwrap();
        let decoded: Value = decode_default(&encoded).unwrap();
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
        let decoded: Value = decode_default(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_custom_delimiter() {
        let original = json!({"tags": ["a", "b", "c"]});
        let opts = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
        let encoded = encode(&original, &opts).unwrap();
        assert!(encoded.contains("|"));

        let decoded: Value = decode_default(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_decode_strict_helper() {
        let input = "items[2]: a,b";
        assert!(decode_strict::<Value>(input).is_ok());

        let input = "items[3]: a,b";
        assert!(decode_strict::<Value>(input).is_err());
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

    use serde::{
        Deserialize,
        Serialize,
    };

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestUser {
        name: String,
        age: u32,
        active: bool,
    }

    #[test]
    fn test_encode_decode_simple_struct() {
        use crate::{
            decode_default,
            encode_default,
        };

        let user = TestUser {
            name: "Alice".to_string(),
            age: 30,
            active: true,
        };

        let toon = encode_default(&user).unwrap();
        assert!(toon.contains("name: Alice"));
        assert!(toon.contains("age: 30"));
        assert!(toon.contains("active: true"));

        let decoded: TestUser = decode_default(&toon).unwrap();
        assert_eq!(user, decoded);
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestProduct {
        id: u64,
        name: String,
        tags: Vec<String>,
    }

    #[test]
    fn test_encode_decode_with_array() {
        use crate::{
            decode_default,
            encode_default,
        };

        let product = TestProduct {
            id: 42,
            name: "Widget".to_string(),
            tags: vec!["electronics".to_string(), "gadgets".to_string()],
        };

        let toon = encode_default(&product).unwrap();
        let decoded: TestProduct = decode_default(&toon).unwrap();
        assert_eq!(product, decoded);
    }

    #[test]
    fn test_encode_decode_vec_of_structs() {
        use crate::{
            decode_default,
            encode_default,
        };

        let users = vec![
            TestUser {
                name: "Alice".to_string(),
                age: 30,
                active: true,
            },
            TestUser {
                name: "Bob".to_string(),
                age: 25,
                active: false,
            },
        ];

        let toon = encode_default(&users).unwrap();
        let decoded: Vec<TestUser> = decode_default(&toon).unwrap();
        assert_eq!(users, decoded);
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct Nested {
        outer: OuterStruct,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct OuterStruct {
        inner: InnerStruct,
        value: i32,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct InnerStruct {
        data: String,
    }

    #[test]
    fn test_encode_decode_nested_structs() {
        use crate::{
            decode_default,
            encode_default,
        };

        let nested = Nested {
            outer: OuterStruct {
                inner: InnerStruct {
                    data: "test".to_string(),
                },
                value: 42,
            },
        };

        let toon = encode_default(&nested).unwrap();
        let decoded: Nested = decode_default(&toon).unwrap();
        assert_eq!(nested, decoded);
    }

    #[test]
    fn test_round_trip_list_item_tabular_v3() {
        use crate::{
            decode_default,
            encode_default,
        };

        let original = json!({
            "items": [
                {
                    "users": [
                        {"id": 1, "name": "Alice", "role": "admin"},
                        {"id": 2, "name": "Bob", "role": "user"}
                    ],
                    "status": "active",
                    "count": 2
                }
            ]
        });

        let encoded = encode_default(&original).unwrap();
        let decoded: Value = decode_default(&encoded).unwrap();

        assert_eq!(original, decoded);
    }

    #[test]
    fn test_round_trip_complex_list_item_tabular_v3() {
        use crate::{
            decode_default,
            encode_default,
        };

        let original = json!({
            "data": [
                {
                    "records": [
                        {"id": 1, "value": "x", "score": 100},
                        {"id": 2, "value": "y", "score": 200}
                    ],
                    "total": 2,
                    "status": "active"
                },
                {
                    "records": [
                        {"id": 3, "value": "z", "score": 300}
                    ],
                    "total": 1,
                    "status": "pending"
                }
            ]
        });

        let encoded = encode_default(&original).unwrap();
        let decoded: Value = decode_default(&encoded).unwrap();

        assert_eq!(original, decoded);
    }

    #[test]
    fn test_round_trip_mixed_list_items_v3() {
        use crate::{
            decode_default,
            encode_default,
        };

        let original = json!({
            "entries": [
                {
                    "type": "simple",
                    "value": 42
                },
                {
                    "people": [
                        {"name": "Alice", "age": 30},
                        {"name": "Bob", "age": 25}
                    ],
                    "type": "complex"
                },
                {
                    "tags": ["a", "b", "c"],
                    "type": "array"
                }
            ]
        });

        let encoded = encode_default(&original).unwrap();
        let decoded: Value = decode_default(&encoded).unwrap();

        assert_eq!(original, decoded);
    }
}
