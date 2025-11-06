use serde_json::json;
use toon_format::{
    encode,
    EncodeOptions,
};

pub fn length_marker() {
    let data = json!({
        "tags": ["reading", "gaming", "coding"],
        "items": [
            {"sku": "A1", "qty": 2, "price": 9.99},
            {"sku": "B2", "qty": 1, "price": 14.5}
        ]
    });

    let out = encode(&data, &EncodeOptions::new().with_length_marker('#')).unwrap();
    println!("{out}");
}
