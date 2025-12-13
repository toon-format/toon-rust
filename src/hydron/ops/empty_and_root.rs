#[cfg(feature = "hydron")]
use crate::encode_default;
#[cfg(feature = "hydron")]
use crate::rune::hydron::values::Value;
#[cfg(feature = "hydron")]
use std::collections::HashMap;

#[cfg(not(feature = "hydron"))]
pub fn empty_and_root() {
    println!("Hydron feature required for dynamic empty and root");
}

#[cfg(feature = "hydron")]
pub fn empty_and_root() {
    // Dynamic Empty Container
    let empty_array = Value::Array(vec![]);
    let mut empty_map = HashMap::new();
    empty_map.insert("items".to_string(), empty_array);
    let empty_items = Value::Map(empty_map);
    println!("{}", encode_default(&empty_items).unwrap());

    // Dynamic Root Array
    let root_array = Value::Array(vec![
        Value::String("x".to_string()),
        Value::String("y".to_string()),
    ]);
    println!("\n{}", encode_default(&root_array).unwrap());

    // Dynamic Empty Object
    let empty_obj = Value::Map(HashMap::new());
    let out = encode_default(&empty_obj).unwrap();
    if out.is_empty() {
        println!("\n(empty output)");
    } else {
        println!("\n{out}");
    }
}
