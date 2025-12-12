use rune_zuno::parse;

#[test]
fn roundtrip_basic() {
    let src = r#"intent: "hello"
data: { a: 1, b: [2, 3] }
"#;
    let doc = parse(src).expect("parse");
    assert!(!doc.is_empty());
}
