use rune_format::encode_default;
use serde_json::json;

pub fn empty_and_root() {
    // Empty containers
    let empty_items = json!({ "items": [] });
    println!("{}", encode_default(&empty_items).unwrap());

    // Root array
    let root_array = json!(["x", "y"]);
    println!("\n{}", encode_default(&root_array).unwrap());

    // Empty object at root encodes to empty output; print a marker
    let empty_obj = json!({});
    let out = encode_default(&empty_obj).unwrap();
    if out.is_empty() {
        println!("\n(empty output)");
    } else {
        println!("\n{out}");
    }
}
