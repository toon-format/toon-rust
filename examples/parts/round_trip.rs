use serde::{
    Deserialize,
    Serialize,
};
use serde_json::{
    json,
    Value,
};
use toon_format::{
    decode_default,
    encode_default,
};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Product {
    product: String,
    price: f64,
    stock: i32,
    categories: Vec<String>,
}

pub fn round_trip() {
    // JSON example
    let original = json!({
        "product": "Widget",
        "price": 29.99,
        "stock": 100,
        "categories": ["tools", "hardware"]
    });

    let encoded = encode_default(&original).unwrap();
    let decoded: Value = decode_default(&encoded).unwrap();

    println!("Encoded:\n{encoded}",);
    println!("\nRound-trip equal: {}", original == decoded);

    // Struct example
    let original_product = Product {
        product: "Widget".to_string(),
        price: 29.99,
        stock: 100,
        categories: vec!["tools".to_string(), "hardware".to_string()],
    };

    let encoded = encode_default(&original_product).unwrap();
    let decoded: Product = decode_default(&encoded).unwrap();

    println!("\nStruct encoded:\n{encoded}");
    println!("\nStruct round-trip equal: {}", original_product == decoded);
}
