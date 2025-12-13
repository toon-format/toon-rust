#[cfg(feature = "hydron")]
use crate::encode_default;
#[cfg(feature = "hydron")]
use crate::rune::hydron::values::Value;
#[cfg(feature = "hydron")]
use std::collections::HashMap;

#[cfg(not(feature = "hydron"))]
pub fn objects() {
    println!("Hydron feature required for dynamic objects");
}

#[cfg(feature = "hydron")]
pub fn objects() {
    // Dynamic Object: Simple
    let mut simple_map = HashMap::new();
    simple_map.insert("id".to_string(), Value::Integer(123));
    simple_map.insert("name".to_string(), Value::String("Ada".to_string()));
    simple_map.insert("active".to_string(), Value::Bool(true));
    let simple = Value::Map(simple_map);

    let out = encode_default(&simple).unwrap();
    println!("{out}");

    // Dynamic Object: Nested
    let mut user_info = HashMap::new();
    user_info.insert("id".to_string(), Value::Integer(123));
    user_info.insert("name".to_string(), Value::String("Ada".to_string()));

    let mut nested_map = HashMap::new();
    nested_map.insert("user".to_string(), Value::Map(user_info));
    let nested = Value::Map(nested_map);

    let out_nested = encode_default(&nested).unwrap();
    println!("\n{out_nested}");

    // Dynamic Array of Objects
    let mut user1 = HashMap::new();
    user1.insert("id".to_string(), Value::Integer(1));
    user1.insert("name".to_string(), Value::String("Alice".to_string()));
    user1.insert(
        "email".to_string(),
        Value::String("alice@example.com".to_string()),
    );
    user1.insert("active".to_string(), Value::Bool(true));

    let mut user2 = HashMap::new();
    user2.insert("id".to_string(), Value::Integer(2));
    user2.insert("name".to_string(), Value::String("Bob".to_string()));
    user2.insert(
        "email".to_string(),
        Value::String("bob@example.com".to_string()),
    );
    user2.insert("active".to_string(), Value::Bool(true));

    let users = Value::Array(vec![Value::Map(user1), Value::Map(user2)]);

    let out = encode_default(&users).unwrap();
    println!("\n{out}");
}
