use rune_format::{decode_default, encode_default};
use serde_json::{Value, json};

#[test]
fn test_real_world_github_data() {
    let github_repos = json!({
        "repositories": [
            {
                "id": 28457823,
                "name": "freeCodeCamp",
                "full_name": "freeCodeCamp/freeCodeCamp",
                "stars": 430886,
                "watchers": 8583,
                "forks": 42146,
                "language": "TypeScript",
                "has_issues": true,
                "has_wiki": true
            },
            {
                "id": 132750724,
                "name": "build-your-own-x",
                "full_name": "codecrafters-io/build-your-own-x",
                "stars": 430877,
                "watchers": 6332,
                "forks": 40453,
                "language": "Markdown",
                "has_issues": true,
                "has_wiki": false
            }
        ]
    });

    let encoded = encode_default(&github_repos).unwrap();
    let decoded: Value = decode_default(&encoded).unwrap();
    assert_eq!(github_repos, decoded);

    assert!(encoded.contains("repositories[2]{"));
    assert!(encoded.contains("}:"));
}

#[test]
fn test_real_world_e_commerce_data() {
    let order = json!({
        "order_id": "ORD-12345",
        "customer": {
            "id": 5678,
            "name": "John Doe",
            "email": "john@example.com"
        },
        "items": [
            {
                "sku": "WIDGET-001",
                "name": "Premium Widget",
                "quantity": 2,
                "price": 29.99,
                "discount": 0.1
            },
            {
                "sku": "GADGET-042",
                "name": "Super Gadget",
                "quantity": 1,
                "price": 149.99,
                "discount": 0.0
            }
        ],
        "shipping": {
            "method": "express",
            "cost": 15.50,
            "address": "123 Main St, City, State 12345"
        },
        "total": 224.46
    });

    let encoded = encode_default(&order).unwrap();
    let decoded: Value = decode_default(&encoded).unwrap();
    assert_eq!(decoded["order_id"], order["order_id"]);
    assert_eq!(decoded["customer"], order["customer"]);
    assert_eq!(decoded["shipping"], order["shipping"]);
    assert_eq!(decoded["total"], order["total"]);
    assert_eq!(decoded["items"].as_array().unwrap().len(), 2);
}
