#[cfg(feature = "hydron")]
use crate::encode_default;
#[cfg(feature = "hydron")]
use crate::rune::hydron::values::Value;
#[cfg(feature = "hydron")]
use std::collections::HashMap;

#[cfg(not(feature = "hydron"))]
pub fn mixed_arrays() {
    println!("Hydron feature required for dynamic mixed arrays");
}

#[cfg(feature = "hydron")]
pub fn mixed_arrays() {
    // Dynamic Mixed Array
    let mut obj = HashMap::new();
    obj.insert("a".to_string(), Value::Integer(1));

    let mixed_array = Value::Array(vec![
        Value::Integer(1),
        Value::Map(obj),
        Value::String("text".to_string()),
    ]);

    let mut mixed_map = HashMap::new();
    mixed_map.insert("items".to_string(), mixed_array);
    let mixed = Value::Map(mixed_map);

    println!("{}", encode_default(&mixed).unwrap());

    // Dynamic List of Objects
    let mut item1 = HashMap::new();
    item1.insert("id".to_string(), Value::Integer(1));
    item1.insert("name".to_string(), Value::String("First".to_string()));

    let mut item2 = HashMap::new();
    item2.insert("id".to_string(), Value::Integer(2));
    item2.insert("name".to_string(), Value::String("Second".to_string()));
    item2.insert("extra".to_string(), Value::Bool(true));

    let list_array = Value::Array(vec![Value::Map(item1), Value::Map(item2)]);

    let mut list_map = HashMap::new();
    list_map.insert("items".to_string(), list_array);
    let list_objects = Value::Map(list_map);

    println!("\n{}", encode_default(&list_objects).unwrap());
}
