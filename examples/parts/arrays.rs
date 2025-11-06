use serde_json::json;
use toon_format::encode_default;

pub fn arrays() {
    let data = json!({ "tags": ["admin", "ops", "dev"] });
    let out = encode_default(&data).unwrap();
    println!("{out}");
}
