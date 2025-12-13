#[cfg(feature = "npy")]
#[test]
fn npy_round_trip_enabled() {
    use rune_cros::npy::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    // Write a small f64 slice into a .npy temp file and read bytes back
    let data = vec![1.0f64, 2.5, 3.0];
    let mut path = std::env::temp_dir();
    let suffix = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    path.push(format!("rune_cros_npy_test_{suffix}.npy"));
    npy::to_file(&path, data.clone()).expect("write npy");
    let buf = fs::read(&path).expect("read npy");

    // Now call our npy_to_json converter
    let json_str = npy_to_json(&buf).expect("npy -> json");
    assert!(json_str.contains("1.0") && json_str.contains("2.5") && json_str.contains("3.0"));

    // Also test direct Rune conversion
    let rune_str = npy_to_rune_string(&buf).expect("npy -> rune");
    assert!(rune_str.contains("[")); // rough sanity check; round-trip decode
    let val: serde_json::Value = rune_format::decode_default(&rune_str).expect("decode rune");
    let arr = val.as_array().cloned().expect("array");
    let floats: Vec<f64> = arr.iter().map(|v| v.as_f64().unwrap()).collect();
    assert_eq!(floats, data);

    // Cleanup best-effort
    let _ = fs::remove_file(path);
}
