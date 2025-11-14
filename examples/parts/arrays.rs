use serde::{
    Deserialize,
    Serialize,
};
use serde_json::json;
use toon_format::encode_default;

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    tags: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Data {
    nums: Vec<i32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Mixed {
    data: Vec<serde_json::Value>,
}

pub fn arrays() {
    // JSON example
    let data = json!({ "tags": ["admin", "ops", "dev"] });
    let out = encode_default(&data).unwrap();
    println!("{out}");

    // Struct with string array
    let config = Config {
        tags: vec!["admin".to_string(), "ops".to_string(), "dev".to_string()],
    };
    let out = encode_default(&config).unwrap();
    println!("\n{out}");

    // Struct with number array
    let data = Data {
        nums: vec![1, 2, 3, 4, 5],
    };
    let out = encode_default(&data).unwrap();
    println!("\n{out}");

    // Struct with mixed primitive array
    let mixed = Mixed {
        data: vec![
            serde_json::Value::String("x".to_string()),
            serde_json::Value::String("y".to_string()),
            serde_json::Value::Bool(true),
            serde_json::Value::Number(10.into()),
        ],
    };
    let out = encode_default(&mixed).unwrap();
    println!("\n{out}");
}
