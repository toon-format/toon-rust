use toon_format::{
    decode_default,
    encode_default,
};
use serde_json::json;

#[test]
fn test_unicode_strings() {
    let unicode = json!({
        "emoji": "😀🎉🦀",
        "chinese": "你好世界",
        "arabic": "مرحبا",
        "mixed": "Hello 世界 🌍"
    });

    let encoded = encode_default(&unicode).unwrap();
    let decoded = decode_default(&encoded).unwrap();
    assert_eq!(unicode, decoded);
}
