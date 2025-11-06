use serde_json::json;
use toon_format::{
    decode_default,
    encode_default,
};

pub fn round_trip() {
    let original = json!({
        "product": "Widget",
        "price": 29.99,
        "stock": 100,
        "categories": ["tools", "hardware"]
    });

    let encoded = encode_default(&original).unwrap();
    let decoded = decode_default(&encoded).unwrap();

    println!("Encoded:\n{encoded}",);
    println!("\nRound-trip equal: {}", original == decoded);
}
