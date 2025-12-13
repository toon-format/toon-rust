#[cfg(feature = "hydron")]
use crate::rune::hydron::values::Value;
#[cfg(feature = "hydron")]
use crate::{decode_default, encode_default};
#[cfg(feature = "hydron")]
use std::collections::HashMap;

#[cfg(not(feature = "hydron"))]
pub fn round_trip() {
    println!("Hydron feature required for dynamic round trip");
}

#[cfg(feature = "hydron")]
pub fn round_trip() {
    // Dynamic Round Trip (Value)
    let mut product_map = HashMap::new();
    product_map.insert("product".to_string(), Value::String("Widget".to_string()));
    product_map.insert("price".to_string(), Value::Float(29.99));
    product_map.insert("stock".to_string(), Value::Integer(100));

    let categories = Value::Array(vec![
        Value::String("tools".to_string()),
        Value::String("hardware".to_string()),
    ]);
    product_map.insert("categories".to_string(), categories);

    let original = Value::Map(product_map);

    let encoded = encode_default(&original).unwrap();
    let decoded: Value = decode_default(&encoded).unwrap();

    println!("Encoded:\n{encoded}",);
    println!("\nRound-trip equal: {}", original == decoded);
}
