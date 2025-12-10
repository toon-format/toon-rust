#[cfg(feature = "hydron")]
use crate::encode_default;
#[cfg(feature = "hydron")]
use crate::rune::hydron::values::Value;

#[cfg(not(feature = "hydron"))]
pub fn arrays() {
    println!("Hydron feature required for dynamic arrays");
}

#[cfg(feature = "hydron")]
pub fn arrays() {
    // Dynamic Array (Strings)
    let tags = Value::Array(vec![
        Value::String("admin".to_string()),
        Value::String("ops".to_string()),
        Value::String("dev".to_string()),
    ]);
    let out = encode_default(&tags).unwrap();
    println!("tags[3]: {out}");

    // Dynamic Array (Numbers)
    let nums = Value::Array(vec![
        Value::Integer(1),
        Value::Integer(2),
        Value::Integer(3),
        Value::Integer(4),
        Value::Integer(5),
    ]);
    let out = encode_default(&nums).unwrap();
    println!("\nnums[5]: {out}");

    // Dynamic Array (Mixed)
    let mixed = Value::Array(vec![
        Value::String("x".to_string()),
        Value::String("y".to_string()),
        Value::Bool(true),
        Value::Integer(10),
    ]);
    let out = encode_default(&mixed).unwrap();
    println!("\ndata[4]: {out}");
}
