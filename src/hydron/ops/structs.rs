#[cfg(feature = "hydron")]
use crate::rune::hydron::values::Value;
#[cfg(feature = "hydron")]
use crate::{decode_default, encode_default};
#[cfg(feature = "hydron")]
use std::collections::HashMap;

#[cfg(not(feature = "hydron"))]
pub fn serde_structs() {
    println!("Hydron feature required for dynamic structs");
}

#[cfg(feature = "hydron")]
pub fn serde_structs() {
    // Dynamic Struct (User)
    let mut user_map = HashMap::new();
    user_map.insert("name".to_string(), Value::String("Alice".to_string()));
    user_map.insert("age".to_string(), Value::Integer(30));
    user_map.insert(
        "email".to_string(),
        Value::String("alice@example.com".to_string()),
    );
    user_map.insert("active".to_string(), Value::Bool(true));
    let user = Value::Map(user_map);

    let toon = encode_default(&user).unwrap();
    println!("{toon}");

    let decoded: Value = decode_default(&toon).unwrap();
    assert_eq!(user, decoded);

    // Dynamic Nested Struct (Product)
    let mut metadata_map = HashMap::new();
    metadata_map.insert("category".to_string(), Value::String("Tech".to_string()));
    metadata_map.insert("in_stock".to_string(), Value::Bool(true));

    let tags = Value::Array(vec![
        Value::String("electronics".to_string()),
        Value::String("computers".to_string()),
    ]);

    let mut product_map = HashMap::new();
    product_map.insert("id".to_string(), Value::Integer(42));
    product_map.insert("name".to_string(), Value::String("Laptop".to_string()));
    product_map.insert("price".to_string(), Value::Float(999.99));
    product_map.insert("tags".to_string(), tags);
    product_map.insert("metadata".to_string(), Value::Map(metadata_map));
    let product = Value::Map(product_map);

    let toon = encode_default(&product).unwrap();
    println!("\n{toon}");

    let decoded: Value = decode_default(&toon).unwrap();
    assert_eq!(product, decoded);
}
