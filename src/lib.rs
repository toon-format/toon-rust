/* rune-xero/src/lib.rs */
#![warn(rustdoc::missing_crate_level_docs)]
//!▫~•◦-------------------------------‣
//! # A high-performance Rust implementation of the RUNE data format.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! **RUNE (Token-Oriented Object Notation)** is a compact, human-readable data
//! format designed for efficient communication with Large Language Models (LLMs)
//! by significantly reducing token usage compared to formats like JSON.
//!
//! This crate provides a comprehensive toolkit for working with RUNE data, including:
//! - A robust, high-performance `serde` implementation for encoding and decoding.
//! - A feature-rich interactive Terminal UI for real-time conversion and analysis.
//! - A core parsing engine for the full RUNE language specification.
//! - Zero-copy and minimal-allocation utilities for maximum performance.
//!
//! ## Key Capabilities
//! - **Serialization/Deserialization**: `encode` and `decode` functions for any `serde`-compatible type.
//! - **Interactive TUI**: A full-featured terminal application, enabled via the `tui` feature.
//! - **RUNE Language Engine**: A parser and (optional) Hydron evaluator for the RUNE superset.
//! - **Performance-Obsessed**: Designed from the ground up to minimize allocations and CPU overhead.
//!
//! ### Example Usage
//! ```rust
//! use rune_xero::{encode_default, decode_default};
//! use serde_json::json;
//!
//! // Encode a serde_json::Value to a RUNE string
//! let data = json!({"name": "Alice", "age": 30});
//! let rune_string = encode_default(&data).unwrap();
//! println!("RUNE: {}", rune_string);
//!
//! // Decode a RUNE string back into a serde_json::Value
//! let decoded: serde_json::Value = decode_default(&rune_string).unwrap();
//! assert_eq!(decoded["name"], "Alice");
//! assert_eq!(decoded["age"], 30);
//! ```
//! RUNE Format is Copyright (c) 2025-PRESENT Shreyas S Bhat, Johann Schopplich
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod ast;
pub mod constants;
pub mod decoder;
pub mod encoder;
pub mod hydron;
pub mod operator;
pub mod renderer;
pub mod types;
pub mod utils;

#[cfg(feature = "tui")]
pub mod tui;

pub use decoder::{
    decode, decode_default, decode_no_coerce, decode_no_coerce_with_options, decode_strict,
    decode_strict_with_options,
};
pub use encoder::{encode, encode_ast, encode_default};
pub use hydron::{EvalContext, EvalError, Octonion, Value};
pub use types::{DecodeOptions, Delimiter, EncodeOptions, Indent, KeyFoldingMode, PathExpansionMode, RuneError};
pub use utils::{
    literal::{is_keyword, is_literal_like},
    normalize,
    string::{escape_string, is_valid_unquoted_key, needs_quoting},
};

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use serde_json::{json, Value};

    use crate::{
        constants::is_keyword,
        decoder::{decode_default, decode_strict},
        encoder::{encode, encode_default},
        types::{Delimiter, EncodeOptions},
        utils::{escape_string, is_literal_like, needs_quoting, normalize},
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
        assert!(encoded.contains('|'));

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
        // Test that the zero-copy escape_string now returns a Cow
        assert_eq!(escape_string("hello\nworld"), Cow::Owned::<String>("hello\\nworld".into()));
        assert_eq!(escape_string("hello world"), Cow::Borrowed("hello world"));
        assert!(needs_quoting("true", Delimiter::Comma.as_char()));
    }

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestUser {
        name: String,
        age: u32,
        active: bool,
    }

    #[test]
    fn test_encode_decode_simple_struct() {
        use crate::{decode_default, encode_default};

        let user = TestUser {
            name: "Alice".to_string(),
            age: 30,
            active: true,
        };

        let rune = encode_default(&user).unwrap();
        assert!(rune.contains("name: Alice"));
        assert!(rune.contains("age: 30"));
        assert!(rune.contains("active: true"));

        let decoded: TestUser = decode_default(&rune).unwrap();
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
        use crate::{decode_default, encode_default};

        let product = TestProduct {
            id: 42,
            name: "Widget".to_string(),
            tags: vec!["electronics".to_string(), "gadgets".to_string()],
        };

        let rune = encode_default(&product).unwrap();
        let decoded: TestProduct = decode_default(&rune).unwrap();
        assert_eq!(product, decoded);
    }

    #[test]
    fn test_encode_decode_vec_of_structs() {
        use crate::{decode_default, encode_default};

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

        let rune = encode_default(&users).unwrap();
        let decoded: Vec<TestUser> = decode_default(&rune).unwrap();
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
        use crate::{decode_default, encode_default};

        let nested = Nested {
            outer: OuterStruct {
                inner: InnerStruct {
                    data: "test".to_string(),
                },
                value: 42,
            },
        };

        let rune = encode_default(&nested).unwrap();
        let decoded: Nested = decode_default(&rune).unwrap();
        assert_eq!(nested, decoded);
    }

    #[test]
    fn test_round_trip_list_item_tabular_v3() {
        use crate::{decode_default, encode_default};

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
        use crate::{decode_default, encode_default};

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
        use crate::{decode_default, encode_default};

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
