use serde_json::json;
use toon_format::encode_default;

pub fn tabular() {
    // Arrays of objects (tabular)
    let items = json!({
        "items": [
            { "sku": "A1", "qty": 2, "price": 9.99 },
            { "sku": "B2", "qty": 1, "price": 14.5 }
        ]
    });
    let out = encode_default(&items).unwrap();
    println!("{out}");

    // Recursive tabular inside nested structures
    let nested = json!({
        "items": [
            {
                "users": [
                    { "id": 1, "name": "Ada" },
                    { "id": 2, "name": "Bob" }
                ],
                "status": "active"
            }
        ]
    });
    let out_nested = encode_default(&nested).unwrap();
    println!("\n{out_nested}");
}
