#![no_main]

use libfuzzer_sys::fuzz_target;
use serde_json::Value;
use toon_format::{encode, EncodeOptions};

fuzz_target!(|data: &[u8]| {
    if let Ok(json) = serde_json::from_slice::<Value>(data) {
        let _ = encode(&json, &EncodeOptions::default());
    }
});
