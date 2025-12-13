#[cfg(feature = "hydron")]
use crate::rune::hydron::values::Value;
#[cfg(feature = "hydron")]
use crate::{Delimiter, EncodeOptions, encode};
#[cfg(feature = "hydron")]
use std::collections::HashMap;

#[cfg(not(feature = "hydron"))]
pub fn delimiters() {
    println!("Hydron feature required for dynamic delimiters");
}

#[cfg(feature = "hydron")]
pub fn delimiters() {
    // Dynamic Data
    let mut item1 = HashMap::new();
    item1.insert("sku".to_string(), Value::String("A1".to_string()));
    item1.insert("name".to_string(), Value::String("Widget".to_string()));
    item1.insert("qty".to_string(), Value::Integer(2));
    item1.insert("price".to_string(), Value::Float(9.99));

    let mut item2 = HashMap::new();
    item2.insert("sku".to_string(), Value::String("B2".to_string()));
    item2.insert("name".to_string(), Value::String("Gadget".to_string()));
    item2.insert("qty".to_string(), Value::Integer(1));
    item2.insert("price".to_string(), Value::Float(14.5));

    let items_array = Value::Array(vec![Value::Map(item1), Value::Map(item2)]);

    let mut data_map = HashMap::new();
    data_map.insert("items".to_string(), items_array);
    let data = Value::Map(data_map);

    // Tab delimiter (\t)
    let tab = encode(&data, &EncodeOptions::new().with_delimiter(Delimiter::Tab)).unwrap();
    println!("{tab}");

    // Pipe delimiter (|)
    let pipe = encode(&data, &EncodeOptions::new().with_delimiter(Delimiter::Pipe)).unwrap();
    println!("\n{pipe}");
}
