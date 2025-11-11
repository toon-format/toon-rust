use serde_json::json;
use toon_format::encode_default;

pub fn objects() {
    // Simple object
    let simple = json!({
        "id": 123,
        "name": "Ada",
        "active": true
    });
    let out = encode_default(&simple).unwrap();
    println!("{out}");

    // Nested object
    let nested = json!({
        "user": { "id": 123, "name": "Ada" }
    });
    let out_nested = encode_default(&nested).unwrap();
    println!("\n{out_nested}");

    let array_object = json!([
      {
        "id": 1,
        "name": "Alice",
        "email": "alice@example.com",
        "active": true
      },
      {
        "id": 2,
        "name": "Bob",
        "email": "bob@example.com",
        "active": true
      }
    ]);
    let out = encode_default(&array_object).unwrap();
    println!("{out}");
}
