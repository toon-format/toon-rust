use serde::{
    Deserialize,
    Serialize,
};
use toon_format::{
    decode_default,
    encode_default,
};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct User {
    name: String,
    age: u32,
    email: String,
    active: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Product {
    id: u64,
    name: String,
    price: f64,
    tags: Vec<String>,
    metadata: Metadata,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Metadata {
    category: String,
    in_stock: bool,
}

pub fn serde_structs() {
    // Simple struct encode/decode
    let user = User {
        name: "Alice".to_string(),
        age: 30,
        email: "alice@example.com".to_string(),
        active: true,
    };
    let toon = encode_default(&user).unwrap();
    println!("{toon}");
    let decoded: User = decode_default(&toon).unwrap();
    assert_eq!(user, decoded);

    // Nested struct encode/decode
    let product = Product {
        id: 42,
        name: "Laptop".to_string(),
        price: 999.99,
        tags: vec!["electronics".to_string(), "computers".to_string()],
        metadata: Metadata {
            category: "Tech".to_string(),
            in_stock: true,
        },
    };
    let toon = encode_default(&product).unwrap();
    println!("\n{toon}");
    let decoded: Product = decode_default(&toon).unwrap();
    assert_eq!(product, decoded);
}
