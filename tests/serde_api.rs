use std::io::Cursor;

use serde::{Deserialize, Serialize};
use serde_json::json;
use toon_format::{
    from_reader, from_slice, from_str, from_str_with_options, to_string, to_string_with_options,
    to_vec, to_writer, DecodeOptions, Delimiter, EncodeOptions, ToonError,
};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct User {
    name: String,
    age: u32,
}

#[test]
fn test_round_trip_string_api() {
    let user = User {
        name: "Ada".to_string(),
        age: 37,
    };
    let encoded = to_string(&user).unwrap();
    let decoded: User = from_str(&encoded).unwrap();
    assert_eq!(decoded, user);
}

#[test]
fn test_writer_reader_round_trip() {
    let user = User {
        name: "Turing".to_string(),
        age: 41,
    };
    let mut buffer = Vec::new();
    to_writer(&mut buffer, &user).unwrap();

    let mut reader = Cursor::new(buffer);
    let decoded: User = from_reader(&mut reader).unwrap();
    assert_eq!(decoded, user);
}

#[test]
fn test_options_wiring() {
    let data = json!({"items": ["a", "b"]});
    let opts = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
    let encoded = to_string_with_options(&data, &opts).unwrap();
    assert!(encoded.contains('|'));

    let decode_opts = DecodeOptions::new().with_strict(false);
    let decoded: serde_json::Value = from_str_with_options("items[2]: a", &decode_opts).unwrap();
    assert_eq!(decoded, json!({"items": ["a"]}));
}

#[test]
fn test_vec_and_slice_api() {
    let user = User {
        name: "Grace".to_string(),
        age: 60,
    };
    let bytes = to_vec(&user).unwrap();
    let decoded: User = from_slice(&bytes).unwrap();
    assert_eq!(decoded, user);
}

#[test]
fn test_from_slice_invalid_utf8() {
    let err = from_slice::<serde_json::Value>(&[0xff]).unwrap_err();
    assert!(matches!(err, ToonError::InvalidInput(_)));
}
