#![no_main]

use libfuzzer_sys::fuzz_target;
use toon_format::{decode, DecodeOptions};

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = decode::<serde_json::Value>(s, &DecodeOptions::default());

        let strict = DecodeOptions::new().with_strict(true);
        let _ = decode::<serde_json::Value>(s, &strict);
    }
});
