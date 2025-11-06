use core::f64;

use serde_json::json;
use toon_format::{
    decode_default,
    encode_default,
};

#[test]
fn test_numeric_edge_cases() {
    let numbers = json!({
        "zero": 0,
        "negative": -42,
        "large": 9999999999i64,
        "small": -9999999999i64,
        "decimal": f64::consts::PI,
        "scientific": 1.23e10,
        "tiny": 0.0000001
    });

    let encoded = encode_default(&numbers).unwrap();
    let decoded = decode_default(&encoded).unwrap();

    assert_eq!(decoded["zero"], json!(0));
    assert_eq!(decoded["negative"], json!(-42));
}
