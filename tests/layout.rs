#![cfg(feature = "layout")]

use toon_format::{
    decode_with_layout,
    DecodeOptions,
    Delimiter,
    FieldDescriptor,
    NodeLayout,
};

fn decode(input: &str) -> (serde_json::Value, toon_format::Layout) {
    decode_with_layout(input, &DecodeOptions::default()).expect("decode_with_layout failed")
}

#[test]
fn tabular_array_records_fields_and_delimiter() {
    let (_value, layout) = decode("users[2]{id,name}:\n  1,Alice\n  2,Bob");

    let node = layout.get("/users").expect("/users should be recorded");
    assert_eq!(
        node,
        &NodeLayout::Tabular {
            declared_len: 2,
            fields: vec![FieldDescriptor::leaf("id"), FieldDescriptor::leaf("name")],
            delimiter: Delimiter::Comma,
        }
    );
}

#[test]
fn list_form_array_records_declared_length() {
    let (_value, layout) = decode("items[2]:\n  - a\n  - b");
    assert_eq!(
        layout.get("/items"),
        Some(&NodeLayout::List { declared_len: 2 })
    );
}

#[test]
fn inline_array_records_delimiter() {
    let (_value, layout) = decode("tags[3]: reading,gaming,coding");
    assert_eq!(
        layout.get("/tags"),
        Some(&NodeLayout::InlineArray {
            declared_len: 3,
            delimiter: Delimiter::Comma,
        })
    );
}

#[test]
fn empty_array_records_as_list_with_zero_length() {
    let (_value, layout) = decode("items[0]:");
    assert_eq!(
        layout.get("/items"),
        Some(&NodeLayout::List { declared_len: 0 })
    );
}

#[test]
fn alternative_delimiter_is_recorded() {
    let (_value, layout) = decode("tags[3|]: a|b|c");
    assert_eq!(
        layout.get("/tags"),
        Some(&NodeLayout::InlineArray {
            declared_len: 3,
            delimiter: Delimiter::Pipe,
        })
    );
}

#[test]
fn root_level_inline_array_records_at_empty_pointer() {
    let (_value, layout) = decode("[2]: a,b");
    assert_eq!(
        layout.get(""),
        Some(&NodeLayout::InlineArray {
            declared_len: 2,
            delimiter: Delimiter::Comma,
        })
    );
}

#[test]
fn root_level_tabular_records_at_empty_pointer() {
    let (_value, layout) = decode("[2]{id,name}:\n  1,Alice\n  2,Bob");
    assert!(matches!(
        layout.get(""),
        Some(NodeLayout::Tabular {
            declared_len: 2,
            ..
        })
    ));
}

#[test]
fn nested_array_inside_list_item_object() {
    let input = "items[1]:\n  - users[2]{id,name}:\n      1,Alice\n      2,Bob";
    let (_value, layout) = decode(input);

    assert_eq!(
        layout.get("/items"),
        Some(&NodeLayout::List { declared_len: 1 })
    );
    assert!(matches!(
        layout.get("/items/0/users"),
        Some(NodeLayout::Tabular {
            declared_len: 2,
            ..
        })
    ));
}

#[test]
fn nested_inline_array_under_object_key() {
    let input = "outer:\n  inner[3]: a,b,c";
    let (_value, layout) = decode(input);

    assert_eq!(
        layout.get("/outer/inner"),
        Some(&NodeLayout::InlineArray {
            declared_len: 3,
            delimiter: Delimiter::Comma,
        })
    );
}

#[test]
fn rfc6901_special_chars_escaped_in_pointer() {
    let input = "\"a/b\":\n  inner[2]: x,y";
    let (_value, layout) = decode(input);
    assert!(layout.get("/a~1b/inner").is_some());
}

#[test]
fn multiple_arrays_at_root_object() {
    let input = "users[2]{id,name}:\n  1,Alice\n  2,Bob\ntags[2]: red,blue";
    let (_value, layout) = decode(input);

    assert!(matches!(
        layout.get("/users"),
        Some(NodeLayout::Tabular { .. })
    ));
    assert!(matches!(
        layout.get("/tags"),
        Some(NodeLayout::InlineArray { .. })
    ));
    assert_eq!(layout.len(), 2);
}

#[test]
fn no_layout_recorded_for_pure_objects() {
    let (_value, layout) = decode("name: Alice\nage: 30");
    assert!(layout.is_empty());
}

#[test]
fn iter_yields_pointer_and_node_pairs() {
    let (_value, layout) = decode("a[1]: x\nb[1]: y");
    let collected: Vec<&str> = layout.iter().map(|(p, _)| p).collect();
    assert_eq!(collected, vec!["/a", "/b"]);
}

#[test]
fn decode_without_layout_is_unaffected() {
    use toon_format::decode_default;

    let value: serde_json::Value =
        decode_default("users[2]{id,name}:\n  1,Alice\n  2,Bob").unwrap();
    assert_eq!(value["users"][0]["name"], "Alice");
}
