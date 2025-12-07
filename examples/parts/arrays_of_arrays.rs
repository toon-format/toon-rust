use rune_format::encode_default;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize)]
struct Pairs {
    pairs: Vec<Vec<i32>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct StringPairs {
    pairs: Vec<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Matrix {
    matrix: Vec<Vec<f64>>,
}

pub fn arrays_of_arrays() {
    // JSON example: Arrays containing primitive inner arrays
    let pairs = json!({
        "pairs": [[1, 2], [3, 4]]
    });
    let out = encode_default(&pairs).unwrap();
    println!("{out}");

    // Struct with nested number arrays
    let pairs = Pairs {
        pairs: vec![vec![1, 2], vec![3, 4]],
    };
    let out = encode_default(&pairs).unwrap();
    println!("\n{out}");

    // Struct with nested string arrays
    let string_pairs = StringPairs {
        pairs: vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string(), "d".to_string()],
        ],
    };
    let out = encode_default(&string_pairs).unwrap();
    println!("\n{out}");

    // Struct with matrix (2D array)
    let matrix = Matrix {
        matrix: vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ],
    };
    let out = encode_default(&matrix).unwrap();
    println!("\n{out}");
}
