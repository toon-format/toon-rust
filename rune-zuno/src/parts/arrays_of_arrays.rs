#[cfg(feature = "hydron")]
use crate::encode_default;
#[cfg(feature = "hydron")]
use crate::rune::hydron::values::Value;
#[cfg(feature = "hydron")]
use std::collections::HashMap;

#[cfg(not(feature = "hydron"))]
pub fn arrays_of_arrays() {
    println!("Hydron feature required for dynamic arrays of arrays");
}

#[cfg(feature = "hydron")]
pub fn arrays_of_arrays() {
    // Dynamic Arrays of Arrays (Integers)
    let pair1 = Value::Array(vec![Value::Integer(1), Value::Integer(2)]);
    let pair2 = Value::Array(vec![Value::Integer(3), Value::Integer(4)]);
    let pairs_array = Value::Array(vec![pair1, pair2]);

    let mut pairs_map = HashMap::new();
    pairs_map.insert("pairs".to_string(), pairs_array);
    let pairs = Value::Map(pairs_map);

    let out = encode_default(&pairs).unwrap();
    println!("{out}");

    // Dynamic Arrays of Arrays (Strings)
    let spair1 = Value::Array(vec![
        Value::String("a".to_string()),
        Value::String("b".to_string()),
    ]);
    let spair2 = Value::Array(vec![
        Value::String("c".to_string()),
        Value::String("d".to_string()),
    ]);
    let spairs_array = Value::Array(vec![spair1, spair2]);

    let mut spairs_map = HashMap::new();
    spairs_map.insert("pairs".to_string(), spairs_array);
    let string_pairs = Value::Map(spairs_map);

    let out = encode_default(&string_pairs).unwrap();
    println!("\n{out}");

    // Dynamic Matrix
    let row1 = Value::Array(vec![
        Value::Float(1.0),
        Value::Float(2.0),
        Value::Float(3.0),
    ]);
    let row2 = Value::Array(vec![
        Value::Float(4.0),
        Value::Float(5.0),
        Value::Float(6.0),
    ]);
    let row3 = Value::Array(vec![
        Value::Float(7.0),
        Value::Float(8.0),
        Value::Float(9.0),
    ]);
    let matrix_array = Value::Array(vec![row1, row2, row3]);

    let mut matrix_map = HashMap::new();
    matrix_map.insert("matrix".to_string(), matrix_array);
    let matrix = Value::Map(matrix_map);

    let out = encode_default(&matrix).unwrap();
    println!("\n{out}");
}
