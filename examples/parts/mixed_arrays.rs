use serde_json::json;
use toon_format::encode_default;

pub fn mixed_arrays() {
    // Mixed / non-uniform arrays (list format)
    let mixed = json!({
        "items": [1, {"a": 1}, "text"]
    });
    println!("{}", encode_default(&mixed).unwrap());

    // Objects in list format: first field on hyphen line
    let list_objects = json!({
        "items": [
            {"id": 1, "name": "First"},
            {"id": 2, "name": "Second", "extra": true}
        ]
    });
    println!("\n{}", encode_default(&list_objects).unwrap());
}
