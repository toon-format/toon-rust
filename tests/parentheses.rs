//! Regression test for parentheses in string values
//!
//! This test ensures that strings containing parentheses are properly quoted
//! during encoding to prevent parse errors during decoding.
//!
//! Bug: https://github.com/toon-format/toon-rust/issues/XX

use serde::{Deserialize, Serialize};
use toon_format::{decode_default, encode_default};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct DataWithParentheses {
    message: String,
    count: usize,
}

#[test]
fn test_parentheses_in_simple_string() {
    let data = DataWithParentheses {
        message: "Mostly Functions (3 of 3)".to_string(),
        count: 3,
    };

    let toon = encode_default(&data).expect("Should encode successfully");

    // Verify parentheses trigger quoting
    assert!(toon.contains("\"Mostly Functions (3 of 3)\""),
            "Parentheses should trigger quoting. Got: {}", toon);

    // Verify round-trip works
    let decoded: DataWithParentheses = decode_default(&toon)
        .expect("Should decode successfully");

    assert_eq!(data, decoded, "Round-trip failed");
}

#[test]
fn test_parentheses_with_multiple_fields() {
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct MultiField {
        field1: String,
        field2: String,
        field3: usize,
    }

    let data = MultiField {
        field1: "test".to_string(),
        field2: "Mostly Functions (3 of 3)".to_string(),
        field3: 42,
    };

    let toon = encode_default(&data).expect("Should encode successfully");
    let decoded: MultiField = decode_default(&toon)
        .expect("Should decode successfully - parentheses should be quoted");

    assert_eq!(data, decoded, "Round-trip failed");
}

#[test]
fn test_parentheses_in_array_of_objects() {
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct Item {
        name: String,
        description: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct Container {
        items: Vec<Item>,
    }

    let data = Container {
        items: vec![
            Item {
                name: "first".to_string(),
                description: "Test (with parentheses)".to_string(),
            },
            Item {
                name: "second".to_string(),
                description: "Another (test) case".to_string(),
            },
        ],
    };

    let toon = encode_default(&data).expect("Should encode successfully");
    let decoded: Container = decode_default(&toon)
        .expect("Should decode successfully");

    assert_eq!(data, decoded, "Round-trip with array failed");
}

#[test]
fn test_various_parentheses_patterns() {
    let test_cases = vec![
        "test()",
        "(hello)",
        "func(arg1, arg2)",
        "Mostly Functions (3 of 3)",
        "Multiple (parts) with (parens)",
        "Edge case: )",
        "Edge case: (",
        "Edge case: ()",
    ];

    for test_string in test_cases {
        let data = DataWithParentheses {
            message: test_string.to_string(),
            count: 1,
        };

        let toon = encode_default(&data)
            .unwrap_or_else(|e| panic!("Failed to encode '{}': {}", test_string, e));

        let decoded: DataWithParentheses = decode_default(&toon)
            .unwrap_or_else(|e| panic!("Failed to decode '{}': {}\nTOON: {}", test_string, e, toon));

        assert_eq!(data, decoded, "Round-trip failed for: {}", test_string);
    }
}
