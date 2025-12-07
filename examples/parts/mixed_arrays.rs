use rune_format::encode_default;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize)]
struct MixedItems {
    items: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Item {
    id: i32,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    extra: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ListItems {
    items: Vec<Item>,
}

pub fn mixed_arrays() {
    // JSON example: Mixed / non-uniform arrays (list format)
    let mixed = json!({
        "items": [1, {"a": 1}, "text"]
    });
    println!("{}", encode_default(&mixed).unwrap());

    // Struct with mixed array (using Value)
    let mixed_items = MixedItems {
        items: vec![
            serde_json::Value::Number(1.into()),
            serde_json::Value::Object({
                let mut map = serde_json::Map::new();
                map.insert("a".to_string(), serde_json::Value::Number(1.into()));
                map
            }),
            serde_json::Value::String("text".to_string()),
        ],
    };
    println!("\n{}", encode_default(&mixed_items).unwrap());

    // JSON example: Objects in list format: first field on hyphen line
    let list_objects = json!({
        "items": [
            {"id": 1, "name": "First"},
            {"id": 2, "name": "Second", "extra": true}
        ]
    });
    println!("\n{}", encode_default(&list_objects).unwrap());

    // Struct with non-uniform objects (different fields)
    let list_items = ListItems {
        items: vec![
            Item {
                id: 1,
                name: "First".to_string(),
                extra: None,
            },
            Item {
                id: 2,
                name: "Second".to_string(),
                extra: Some(true),
            },
        ],
    };
    println!("\n{}", encode_default(&list_items).unwrap());
}
