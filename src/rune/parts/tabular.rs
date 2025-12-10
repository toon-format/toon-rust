#[cfg(feature = "hydron")]
use crate::encode_default;
#[cfg(feature = "hydron")]
use crate::rune::hydron::values::Value;
#[cfg(feature = "hydron")]
use std::collections::HashMap;

#[cfg(not(feature = "hydron"))]
pub fn tabular() {
    println!("Hydron feature required for dynamic tabular data");
}

#[cfg(feature = "hydron")]
pub fn tabular() {
    // Dynamic Tabular Data
    let mut item1 = HashMap::new();
    item1.insert("sku".to_string(), Value::String("A1".to_string()));
    item1.insert("qty".to_string(), Value::Integer(2));
    item1.insert("price".to_string(), Value::Float(9.99));

    let mut item2 = HashMap::new();
    item2.insert("sku".to_string(), Value::String("B2".to_string()));
    item2.insert("qty".to_string(), Value::Integer(1));
    item2.insert("price".to_string(), Value::Float(14.5));

    let items_array = Value::Array(vec![Value::Map(item1), Value::Map(item2)]);

    let mut items_map = HashMap::new();
    items_map.insert("items".to_string(), items_array);
    let items = Value::Map(items_map);

    let out = encode_default(&items).unwrap();
    println!("{out}");

    // Dynamic Nested Tabular Data
    let mut user1 = HashMap::new();
    user1.insert("id".to_string(), Value::Integer(1));
    user1.insert("name".to_string(), Value::String("Ada".to_string()));

    let mut user2 = HashMap::new();
    user2.insert("id".to_string(), Value::Integer(2));
    user2.insert("name".to_string(), Value::String("Bob".to_string()));

    let users_array = Value::Array(vec![Value::Map(user1), Value::Map(user2)]);

    let mut container = HashMap::new();
    container.insert("users".to_string(), users_array);
    container.insert("status".to_string(), Value::String("active".to_string()));

    let items_array_nested = Value::Array(vec![Value::Map(container)]);

    let mut nested_map = HashMap::new();
    nested_map.insert("items".to_string(), items_array_nested);
    let nested = Value::Map(nested_map);

    let out_nested = encode_default(&nested).unwrap();
    println!("\n{out_nested}");
}
