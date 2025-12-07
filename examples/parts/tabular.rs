use rune_format::encode_default;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize)]
struct Item {
    sku: String,
    qty: i32,
    price: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct Items {
    items: Vec<Item>,
}

#[derive(Debug, Serialize, Deserialize)]
struct User {
    id: i32,
    name: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Container {
    users: Vec<User>,
    status: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct NestedItems {
    items: Vec<Container>,
}

pub fn tabular() {
    // JSON example: Arrays of objects (tabular)
    let items = json!({
        "items": [
            { "sku": "A1", "qty": 2, "price": 9.99 },
            { "sku": "B2", "qty": 1, "price": 14.5 }
        ]
    });
    let out = encode_default(&items).unwrap();
    println!("{out}");

    // Struct with tabular array
    let items = Items {
        items: vec![
            Item {
                sku: "A1".to_string(),
                qty: 2,
                price: 9.99,
            },
            Item {
                sku: "B2".to_string(),
                qty: 1,
                price: 14.5,
            },
        ],
    };
    let out = encode_default(&items).unwrap();
    println!("\n{out}");

    // JSON example: Recursive tabular inside nested structures
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

    // Struct with nested tabular array
    let nested_items = NestedItems {
        items: vec![Container {
            users: vec![
                User {
                    id: 1,
                    name: "Ada".to_string(),
                },
                User {
                    id: 2,
                    name: "Bob".to_string(),
                },
            ],
            status: "active".to_string(),
        }],
    };
    let out = encode_default(&nested_items).unwrap();
    println!("\n{out}");
}
