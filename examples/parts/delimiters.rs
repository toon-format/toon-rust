use serde_json::json;
use toon_format::{encode, Delimiter, EncodeOptions};

pub fn delimiters() {
    let data = json!({
        "items": [
            {"sku": "A1", "name": "Widget", "qty": 2, "price": 9.99},
            {"sku": "B2", "name": "Gadget", "qty": 1, "price": 14.5}
        ]
    });

    // Tab delimiter (\t)
    let tab = encode(&data, &EncodeOptions::new().with_delimiter(Delimiter::Tab)).unwrap();
    println!("{tab}");

    // Pipe delimiter (|)
    let pipe = encode(&data, &EncodeOptions::new().with_delimiter(Delimiter::Pipe)).unwrap();
    println!("\n{pipe}");
}
