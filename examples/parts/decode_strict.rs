use serde_json::Value;
use toon_format::{
    decode,
    DecodeOptions,
};

pub fn decode_strict() {
    // Malformed: header says 2 rows, but only 1 provided
    let malformed = "items[2]{id,name}:\n  1,Ada";

    let opts = DecodeOptions::new().with_strict(true);
    match decode::<Value>(malformed, &opts) {
        Ok(val) => println!("Unexpectedly decoded: {val}"),
        Err(err) => println!("Strict decode error: {err}"),
    }
}
