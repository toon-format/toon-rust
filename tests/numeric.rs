use core::f64;

use serde_json::{
    json,
    Value,
};
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
    let decoded: Value = decode_default(&encoded).unwrap();

    assert_eq!(decoded["zero"], json!(0));
    assert_eq!(decoded["negative"], json!(-42));
    assert_eq!(decoded["large"], json!(9999999999i64));
    assert_eq!(decoded["small"], json!(-9999999999i64));
    assert_eq!(decoded["decimal"], json!(f64::consts::PI));
    assert_eq!(decoded["tiny"], json!(0.0000001));

    // Integer-valued floats normalize to integers on the way back in.
    assert_eq!(decoded["scientific"], json!(12300000000i64));

    // Canonical form never uses exponent notation, in either direction.
    assert!(
        encoded.contains("scientific: 12300000000"),
        "large magnitudes must not use exponent form: {encoded}"
    );
    assert!(
        encoded.contains("tiny: 0.0000001"),
        "small magnitudes must not use exponent form: {encoded}"
    );
}

#[test]
fn test_integer_precision_round_trip() {
    // Integers outside the i64 domain must keep full precision instead of
    // degrading through f64.
    let cases: Vec<(Value, &str)> = vec![
        (json!(u64::MAX), "18446744073709551615"),
        (json!(9223372036854775808u64), "9223372036854775808"),
        (json!(i64::MAX), "9223372036854775807"),
        (json!(i64::MIN), "-9223372036854775808"),
    ];

    for (value, literal) in cases {
        let encoded = encode_default(&json!({ "v": value })).unwrap();
        assert_eq!(encoded, format!("v: {literal}"), "encoding {value}");

        let decoded: Value = decode_default(&encoded).unwrap();
        assert_eq!(decoded["v"], value, "round trip of {value}");
    }
}

#[test]
fn test_integral_f64_at_i64_boundary_keeps_its_value() {
    // 2^63 is one past i64::MAX. A saturating `as i64` conversion would encode
    // it as 9223372036854775807 — a different number, silently.
    let encoded = encode_default(&json!({ "v": 9223372036854775808.0f64 })).unwrap();
    assert_eq!(encoded, "v: 9223372036854775808");

    let decoded: Value = decode_default(&encoded).unwrap();
    assert_eq!(decoded["v"].as_f64(), Some(9223372036854775808.0));

    // -2^63 is exactly i64::MIN and must stay exact as well.
    let encoded = encode_default(&json!({ "v": -9223372036854775808.0f64 })).unwrap();
    assert_eq!(encoded, "v: -9223372036854775808");
}

#[test]
fn test_exponent_form_normalizes_within_the_integer_domain() {
    // An integer-valued exponent form decodes to the same JSON number as the
    // plain integer spelling, across the whole i64/u64 domain.
    let cases: Vec<(&str, Value)> = vec![
        ("1e19", json!(10000000000000000000u64)),
        ("-1E+03", json!(-1000)),
        ("9.223372036854775808e18", json!(9223372036854775808u64)),
        ("-9.223372036854775808e18", json!(i64::MIN)),
        // -0 decodes to 0.
        ("-0", json!(0)),
    ];

    for (token, expected) in cases {
        let decoded: Value = decode_default(&format!("k: {token}")).unwrap();
        assert_eq!(decoded["k"], expected, "token {token}");
    }

    // Outside that domain the value stays a float rather than saturating:
    // 1e20 is past 2^64 and -1e19 is past i64::MIN.
    for token in ["1e20", "-1e19"] {
        let decoded: Value = decode_default(&format!("k: {token}")).unwrap();
        assert!(
            decoded["k"].is_f64(),
            "token {token} became {}",
            decoded["k"]
        );
    }

    // Non-integral and non-finite tokens are untouched by the normalization.
    let decoded: Value = decode_default("k: 1.5").unwrap();
    assert_eq!(decoded["k"], json!(1.5));

    let decoded: Value = decode_default("k: 1e999").unwrap();
    assert_eq!(decoded["k"], json!("1e999"));
}
