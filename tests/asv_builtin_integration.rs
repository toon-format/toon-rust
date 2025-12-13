#[cfg(feature = "gsv")]
#[test]
fn asv_store_get_integration() {
    use rune_format::rune::hydron::values::{EvalContext, Value};
    let ctx = EvalContext::new();
    // Create a temporary per-test store so we don't write into user's HOME default vault
    let tmp = tempfile::tempdir().expect("create temp dir");
    let path = tmp.path().join("test_gsv.rune");
    let new_store =
        rune_gsv::store::QuantizedContinuum::new(path.to_string_lossy().to_string(), "test_gsv");
    {
        let store_lock = rune_gsv::store::default_store();
        let mut guard = store_lock.write().unwrap();
        *guard = new_store;
    }
    // intent string
    let intent = Value::String("test intent".into());
    // payload: map with intent and payload
    let mut payload_map = std::collections::HashMap::new();
    payload_map.insert("intent".into(), Value::String("test intent".into()));
    payload_map.insert(
        "payload".into(),
        Value::Map(std::collections::HashMap::new()),
    );
    let payload_val = Value::Map(payload_map);

    // Store
    let res = ctx.apply_builtin_by_name("ASV.Store", &[intent.clone(), payload_val.clone()]);
    assert!(res.is_ok());
    // Get
    let res2 = ctx.apply_builtin_by_name("ASV.Get", &[intent]);
    assert!(res2.is_ok());
    let retrieved = res2.unwrap();
    match retrieved {
        Value::Map(_) => {
            // ok
        }
        _ => panic!("ASV.Get did not return a map"),
    }
}
