//! Encoder Implementation
//!
//! Emits TOON per spec v4.1: the form follows from the value's shape and
//! position (§1.4, §9), tabular and keyed tabular forms are mandatory
//! wherever detection succeeds and the position permits them, empty arrays
//! use the `key: []` / `[]` forms, and every header declares the document
//! delimiter (§11.1).
#[cfg(feature = "json_stream")]
pub mod json_stream;
pub(crate) mod tabular;

use indexmap::IndexMap;
use tabular::{
    collect_row_leaves,
    extract_keyed_tabular_fields,
    extract_tabular_fields,
    is_primitive,
};

use crate::{
    constants::MAX_DEPTH,
    types::{
        Delimiter,
        EncodeOptions,
        FieldNode,
        IntoJsonValue,
        JsonValue as Value,
        ToonError,
        ToonResult,
    },
    utils::{
        format_canonical_number,
        is_valid_unquoted_key,
        needs_quoting,
        normalize,
        string::quote_string,
        validation::validate_depth,
    },
};

/// Encode any serializable value to TOON format.
///
/// This function accepts any type implementing `serde::Serialize`, including:
/// - Custom structs with `#[derive(Serialize)]`
/// - `serde_json::Value`
/// - Built-in types (Vec, HashMap, etc.)
///
/// # Examples
///
/// **With custom structs:**
/// ```
/// use serde::Serialize;
/// use toon_format::{
///     encode,
///     EncodeOptions,
/// };
///
/// #[derive(Serialize)]
/// struct User {
///     name: String,
///     age: u32,
/// }
///
/// let user = User {
///     name: "Alice".to_string(),
///     age: 30,
/// };
/// let toon = encode(&user, &EncodeOptions::default())?;
/// assert!(toon.contains("name: Alice"));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
///
/// **With JSON values:**
/// ```
/// use toon_format::{encode, EncodeOptions, Delimiter};
/// use serde_json::json;
///
/// let data = json!({"tags": ["a", "b", "c"]});
/// let options = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
/// let toon = encode(&data, &options)?;
/// assert!(toon.contains("|"));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn encode<T: serde::Serialize>(value: &T, options: &EncodeOptions) -> ToonResult<String> {
    let json_value =
        serde_json::to_value(value).map_err(|e| ToonError::SerializationError(e.to_string()))?;
    let json_value: Value = json_value.into();
    encode_impl(&json_value, options)
}

pub(crate) fn encode_impl(value: &Value, options: &EncodeOptions) -> ToonResult<String> {
    // Zero spaces per level would emit nesting no decoder can recover; the
    // decoder rejects the same option at the same boundary.
    if options.indent.get_spaces() == 0 {
        return Err(ToonError::InvalidInput(
            "indentSize must be at least 1".to_string(),
        ));
    }

    let normalized: Value = normalize(value.clone());
    let mut lines = Vec::new();

    match &normalized {
        Value::Array(arr) => encode_array_lines(None, arr, 0, options, &mut lines)?,
        Value::Object(obj) => {
            // A keyed-eligible root object uses the keyless keyed header (§9.5).
            if let Some(fields) = extract_keyed_tabular_fields(obj) {
                encode_keyed_object_lines(None, obj, &fields, 0, options, &mut lines)?;
            } else {
                encode_object_lines(obj, 0, options, &mut lines)?;
            }
        }
        primitive => lines.push(encode_primitive(primitive, options.delimiter)),
    }

    Ok(lines.join("\n"))
}

/// Encode with default options (2-space indent, comma delimiter).
///
/// Works with any type implementing `serde::Serialize`.
///
/// # Examples
///
/// **With structs:**
/// ```
/// use serde::Serialize;
/// use toon_format::encode_default;
///
/// #[derive(Serialize)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// let person = Person {
///     name: "Alice".to_string(),
///     age: 30,
/// };
/// let toon = encode_default(&person)?;
/// assert!(toon.contains("name: Alice"));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
///
/// **With JSON values:**
/// ```
/// use toon_format::encode_default;
/// use serde_json::json;
///
/// let data = json!({"tags": ["reading", "gaming", "coding"]});
/// let toon = encode_default(&data)?;
/// assert_eq!(toon, "tags[3]: reading,gaming,coding");
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn encode_default<T: serde::Serialize>(value: &T) -> ToonResult<String> {
    encode(value, &EncodeOptions::default())
}

/// Encode a JSON object to TOON format (errors if not an object).
///
/// This function accepts either `JsonValue` or `serde_json::Value` and converts
/// automatically.
///
/// # Examples
///
/// ```
/// use toon_format::{encode_object, EncodeOptions};
/// use serde_json::json;
///
/// let data = json!({"name": "Alice", "age": 30});
/// let toon = encode_object(&data, &EncodeOptions::default())?;
/// assert!(toon.contains("name: Alice"));
///
/// // Will error if not an object
/// assert!(encode_object(json!(42), &EncodeOptions::default()).is_err());
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn encode_object<V: IntoJsonValue>(value: V, options: &EncodeOptions) -> ToonResult<String> {
    let json_value = value.into_json_value();
    if !json_value.is_object() {
        return Err(ToonError::TypeMismatch {
            expected: "object".to_string(),
            found: value_type_name(&json_value).to_string(),
        });
    }
    encode_impl(&json_value, options)
}

/// Encode a JSON array to TOON format (errors if not an array).
///
/// This function accepts either `JsonValue` or `serde_json::Value` and converts
/// automatically.
///
/// # Examples
///
/// ```
/// use toon_format::{encode_array, EncodeOptions};
/// use serde_json::json;
///
/// let data = json!(["a", "b", "c"]);
/// let toon = encode_array(&data, &EncodeOptions::default())?;
/// assert_eq!(toon, "[3]: a,b,c");
///
/// // Will error if not an array
/// assert!(encode_array(&json!({"key": "value"}), &EncodeOptions::default()).is_err());
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn encode_array<V: IntoJsonValue>(value: V, options: &EncodeOptions) -> ToonResult<String> {
    let json_value = value.into_json_value();
    if !json_value.is_array() {
        return Err(ToonError::TypeMismatch {
            expected: "array".to_string(),
            found: value_type_name(&json_value).to_string(),
        });
    }
    encode_impl(&json_value, options)
}

fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// #region Primitive, key, and header formatting (§2, §7)

fn encode_primitive(value: &Value, delimiter: Delimiter) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => format_canonical_number(n),
        Value::String(s) => encode_string_literal(s, delimiter),
        Value::Array(_) | Value::Object(_) => {
            unreachable!("encode_primitive is called with primitives only")
        }
    }
}

fn encode_string_literal(value: &str, delimiter: Delimiter) -> String {
    if needs_quoting(value, delimiter.as_char()) {
        quote_string(value)
    } else {
        value.to_string()
    }
}

fn encode_key(key: &str) -> String {
    if is_valid_unquoted_key(key) {
        key.to_string()
    } else {
        quote_string(key)
    }
}

fn encode_and_join_primitives(values: &[&Value], delimiter: Delimiter) -> String {
    values
        .iter()
        .map(|value| encode_primitive(value, delimiter))
        .collect::<Vec<_>>()
        .join(&delimiter.as_char().to_string())
}

/// Formats a header (§6): `key[N<delim?>]{fields}:`, with the keyed marker
/// `[N:<delim?>]` when requested. Every header declares the document
/// delimiter (§11.1); comma is declared by omission.
fn format_header(
    length: usize,
    key: Option<&str>,
    fields: Option<&[FieldNode]>,
    keyed: bool,
    delimiter: Delimiter,
) -> String {
    let mut header = String::new();

    if let Some(key) = key {
        header.push_str(&encode_key(key));
    }

    header.push('[');
    header.push_str(&length.to_string());
    if keyed {
        header.push(':');
    }
    if delimiter != Delimiter::Comma {
        header.push(delimiter.as_char());
    }
    header.push(']');

    if let Some(fields) = fields {
        header.push('{');
        format_field_segment(fields, delimiter, &mut header);
        header.push('}');
    }

    header.push(':');
    header
}

fn format_field_segment(fields: &[FieldNode], delimiter: Delimiter, out: &mut String) {
    for (i, field) in fields.iter().enumerate() {
        if i > 0 {
            out.push(delimiter.as_char());
        }
        out.push_str(&encode_key(&field.name));
        if let Some(children) = &field.children {
            out.push('{');
            format_field_segment(children, delimiter, out);
            out.push('}');
        }
    }
}

// #endregion

// #region Line emission helpers

fn push_line(lines: &mut Vec<String>, depth: usize, content: &str, options: &EncodeOptions) {
    let mut line = options.indent.get_string(depth);
    line.push_str(content);
    lines.push(line);
}

fn push_list_item(lines: &mut Vec<String>, depth: usize, content: &str, options: &EncodeOptions) {
    let mut line = options.indent.get_string(depth);
    line.push_str("- ");
    line.push_str(content);
    lines.push(line);
}

// #endregion

// #region Object encoding (§8)

fn encode_object_lines(
    obj: &IndexMap<String, Value>,
    depth: usize,
    options: &EncodeOptions,
    lines: &mut Vec<String>,
) -> ToonResult<()> {
    validate_depth(depth, MAX_DEPTH)?;
    for (key, value) in obj {
        encode_key_value_pair_lines(key, value, depth, options, lines)?;
    }
    Ok(())
}

fn encode_key_value_pair_lines(
    key: &str,
    value: &Value,
    depth: usize,
    options: &EncodeOptions,
    lines: &mut Vec<String>,
) -> ToonResult<()> {
    match value {
        Value::Array(arr) => encode_array_lines(Some(key), arr, depth, options, lines),
        Value::Object(obj) => {
            if let Some(fields) = extract_keyed_tabular_fields(obj) {
                return encode_keyed_object_lines(Some(key), obj, &fields, depth, options, lines);
            }

            push_line(lines, depth, &format!("{}:", encode_key(key)), options);
            if !obj.is_empty() {
                encode_object_lines(obj, depth + 1, options, lines)?;
            }
            Ok(())
        }
        primitive => {
            let content = format!(
                "{}: {}",
                encode_key(key),
                encode_primitive(primitive, options.delimiter)
            );
            push_line(lines, depth, &content, options);
            Ok(())
        }
    }
}

// #endregion

// #region Keyed tabular objects (§9.5)

fn encode_keyed_object_lines(
    key: Option<&str>,
    obj: &IndexMap<String, Value>,
    fields: &[FieldNode],
    depth: usize,
    options: &EncodeOptions,
    lines: &mut Vec<String>,
) -> ToonResult<()> {
    let header = format_header(obj.len(), key, Some(fields), true, options.delimiter);
    push_line(lines, depth, &header, options);
    encode_keyed_entry_rows(obj, fields, depth + 1, options, lines);
    Ok(())
}

fn encode_keyed_entry_rows(
    obj: &IndexMap<String, Value>,
    fields: &[FieldNode],
    depth: usize,
    options: &EncodeOptions,
    lines: &mut Vec<String>,
) {
    for (entry_key, entry_value) in obj {
        let Value::Object(entry_obj) = entry_value else {
            unreachable!("keyed tabular detection guarantees object entries");
        };
        let mut leaves = Vec::new();
        collect_row_leaves(entry_obj, fields, &mut leaves);
        let content = format!(
            "{}: {}",
            encode_key(entry_key),
            encode_and_join_primitives(&leaves, options.delimiter)
        );
        push_line(lines, depth, &content, options);
    }
}

// #endregion

// #region Array encoding (§9)

fn encode_array_lines(
    key: Option<&str>,
    arr: &[Value],
    depth: usize,
    options: &EncodeOptions,
    lines: &mut Vec<String>,
) -> ToonResult<()> {
    validate_depth(depth, MAX_DEPTH)?;

    // Empty arrays: `key: []` in field position, `[]` at the root (§9.1).
    if arr.is_empty() {
        let content = match key {
            Some(key) => format!("{}: []", encode_key(key)),
            None => "[]".to_string(),
        };
        push_line(lines, depth, &content, options);
        return Ok(());
    }

    if arr.iter().all(is_primitive) {
        let content = encode_inline_array_line(&arr.iter().collect::<Vec<_>>(), key, options);
        push_line(lines, depth, &content, options);
        return Ok(());
    }

    if arr.iter().all(|v| matches!(v, Value::Array(_))) {
        let all_primitive_arrays = arr.iter().all(|v| {
            let Value::Array(inner) = v else {
                unreachable!("checked above");
            };
            inner.iter().all(is_primitive)
        });
        if all_primitive_arrays {
            return encode_array_of_arrays_as_list_items(key, arr, depth, options, lines);
        }
    }

    if arr.iter().all(|v| matches!(v, Value::Object(_))) {
        if let Some(fields) = extract_tabular_fields(arr) {
            return encode_tabular_lines(key, arr, &fields, depth, options, lines);
        }
    }

    encode_mixed_array_as_list_items(key, arr, depth, options, lines)
}

fn encode_inline_array_line(
    values: &[&Value],
    key: Option<&str>,
    options: &EncodeOptions,
) -> String {
    let header = format_header(values.len(), key, None, false, options.delimiter);
    if values.is_empty() {
        return header;
    }
    format!(
        "{header} {}",
        encode_and_join_primitives(values, options.delimiter)
    )
}

// #endregion

// #region Arrays of primitive arrays – list form (§9.2)

fn encode_array_of_arrays_as_list_items(
    key: Option<&str>,
    values: &[Value],
    depth: usize,
    options: &EncodeOptions,
    lines: &mut Vec<String>,
) -> ToonResult<()> {
    let header = format_header(values.len(), key, None, false, options.delimiter);
    push_line(lines, depth, &header, options);

    for value in values {
        let Value::Array(inner) = value else {
            unreachable!("caller checked every element is an array");
        };
        let content = encode_inline_array_line(&inner.iter().collect::<Vec<_>>(), None, options);
        push_list_item(lines, depth + 1, &content, options);
    }
    Ok(())
}

// #endregion

// #region Arrays of objects – tabular form (§9.3)

fn encode_tabular_lines(
    key: Option<&str>,
    rows: &[Value],
    fields: &[FieldNode],
    depth: usize,
    options: &EncodeOptions,
    lines: &mut Vec<String>,
) -> ToonResult<()> {
    let header = format_header(rows.len(), key, Some(fields), false, options.delimiter);
    push_line(lines, depth, &header, options);
    write_tabular_rows(rows, fields, depth + 1, options, lines);
    Ok(())
}

fn write_tabular_rows(
    rows: &[Value],
    fields: &[FieldNode],
    depth: usize,
    options: &EncodeOptions,
    lines: &mut Vec<String>,
) {
    for row in rows {
        let Value::Object(row_obj) = row else {
            unreachable!("tabular detection guarantees object rows");
        };
        let mut leaves = Vec::new();
        collect_row_leaves(row_obj, fields, &mut leaves);
        push_line(
            lines,
            depth,
            &encode_and_join_primitives(&leaves, options.delimiter),
            options,
        );
    }
}

// #endregion

// #region Mixed and non-uniform arrays – list form (§9.4)

fn encode_mixed_array_as_list_items(
    key: Option<&str>,
    items: &[Value],
    depth: usize,
    options: &EncodeOptions,
    lines: &mut Vec<String>,
) -> ToonResult<()> {
    let header = format_header(items.len(), key, None, false, options.delimiter);
    push_line(lines, depth, &header, options);

    for item in items {
        encode_list_item_value(item, depth + 1, options, lines)?;
    }
    Ok(())
}

fn encode_list_item_value(
    value: &Value,
    depth: usize,
    options: &EncodeOptions,
    lines: &mut Vec<String>,
) -> ToonResult<()> {
    validate_depth(depth, MAX_DEPTH)?;
    match value {
        Value::Array(arr) => {
            if arr.iter().all(is_primitive) {
                let content =
                    encode_inline_array_line(&arr.iter().collect::<Vec<_>>(), None, options);
                push_list_item(lines, depth, &content, options);
            } else {
                // Tabular form is unavailable in this position: a keyless
                // fields-bearing header is valid only at the root (§9.4).
                let header = format_header(arr.len(), None, None, false, options.delimiter);
                push_list_item(lines, depth, &header, options);
                for item in arr {
                    encode_list_item_value(item, depth + 1, options, lines)?;
                }
            }
            Ok(())
        }
        Value::Object(obj) => encode_object_as_list_item(obj, depth, options, lines),
        primitive => {
            push_list_item(
                lines,
                depth,
                &encode_primitive(primitive, options.delimiter),
                options,
            );
            Ok(())
        }
    }
}

// #endregion

// #region Objects as list items (§10)

fn encode_object_as_list_item(
    obj: &IndexMap<String, Value>,
    depth: usize,
    options: &EncodeOptions,
    lines: &mut Vec<String>,
) -> ToonResult<()> {
    if obj.is_empty() {
        push_line(lines, depth, "-", options);
        return Ok(());
    }

    let (first_key, first_value) = obj.get_index(0).expect("object is non-empty");

    // A tabular first field sits on the hyphen line with rows at depth +2 (§10).
    if let Value::Array(arr) = first_value {
        if !arr.is_empty() && arr.iter().all(|v| matches!(v, Value::Object(_))) {
            if let Some(fields) = extract_tabular_fields(arr) {
                let header = format_header(
                    arr.len(),
                    Some(first_key),
                    Some(&fields),
                    false,
                    options.delimiter,
                );
                push_list_item(lines, depth, &header, options);
                write_tabular_rows(arr, &fields, depth + 2, options, lines);

                for (key, value) in obj.iter().skip(1) {
                    encode_key_value_pair_lines(key, value, depth + 1, options, lines)?;
                }
                return Ok(());
            }
        }
    }

    // A keyed tabular first field likewise: header on the hyphen line, entry
    // rows at depth +2, sibling fields at depth +1 (§10).
    if let Value::Object(first_obj) = first_value {
        if let Some(fields) = extract_keyed_tabular_fields(first_obj) {
            let header = format_header(
                first_obj.len(),
                Some(first_key),
                Some(&fields),
                true,
                options.delimiter,
            );
            push_list_item(lines, depth, &header, options);
            encode_keyed_entry_rows(first_obj, &fields, depth + 2, options, lines);

            for (key, value) in obj.iter().skip(1) {
                encode_key_value_pair_lines(key, value, depth + 1, options, lines)?;
            }
            return Ok(());
        }
    }

    let encoded_key = encode_key(first_key);

    match first_value {
        Value::Array(arr) => {
            if arr.is_empty() {
                push_list_item(lines, depth, &format!("{encoded_key}: []"), options);
            } else if arr.iter().all(is_primitive) {
                let content =
                    encode_inline_array_line(&arr.iter().collect::<Vec<_>>(), None, options);
                push_list_item(lines, depth, &format!("{encoded_key}{content}"), options);
            } else {
                // Non-inline array items sit at depth +2, below the hyphen line.
                let header = format_header(arr.len(), None, None, false, options.delimiter);
                push_list_item(lines, depth, &format!("{encoded_key}{header}"), options);
                for item in arr {
                    encode_list_item_value(item, depth + 2, options, lines)?;
                }
            }
        }
        Value::Object(first_obj) => {
            push_list_item(lines, depth, &format!("{encoded_key}:"), options);
            if !first_obj.is_empty() {
                encode_object_lines(first_obj, depth + 2, options, lines)?;
            }
        }
        primitive => {
            let content = format!(
                "{encoded_key}: {}",
                encode_primitive(primitive, options.delimiter)
            );
            push_list_item(lines, depth, &content, options);
        }
    }

    for (key, value) in obj.iter().skip(1) {
        encode_key_value_pair_lines(key, value, depth + 1, options, lines)?;
    }
    Ok(())
}

// #endregion

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::types::Indent;

    fn enc(value: serde_json::Value) -> String {
        encode(&value, &EncodeOptions::default()).unwrap()
    }

    #[test]
    fn test_encode_simple_object() {
        assert_eq!(
            enc(json!({"name": "Alice", "age": 30})),
            "name: Alice\nage: 30"
        );
    }

    #[test]
    fn test_encode_nested_object() {
        assert_eq!(
            enc(json!({"user": {"name": "Alice"}})),
            "user:\n  name: Alice"
        );
    }

    #[test]
    fn test_encode_inline_array() {
        assert_eq!(enc(json!({"tags": ["a", "b", "c"]})), "tags[3]: a,b,c");
    }

    #[test]
    fn test_encode_empty_arrays() {
        assert_eq!(enc(json!({"items": []})), "items: []");
        assert_eq!(enc(json!([])), "[]");
    }

    #[test]
    fn test_encode_tabular_array() {
        assert_eq!(
            enc(json!({"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]})),
            "users[2]{id,name}:\n  1,Alice\n  2,Bob"
        );
    }

    #[test]
    fn test_encode_nested_field_groups() {
        assert_eq!(
            enc(json!({"orders": [
                {"id": 1, "customer": {"name": "Ada", "country": "DK"}, "total": 99},
                {"id": 2, "customer": {"name": "Bob", "country": "UK"}, "total": 149}
            ]})),
            "orders[2]{id,customer{name,country},total}:\n  1,Ada,DK,99\n  2,Bob,UK,149"
        );
    }

    #[test]
    fn test_encode_keyed_tabular_object() {
        assert_eq!(
            enc(json!({"servers": {
                "alpha": {"host": "a.example.com", "port": 8080},
                "beta": {"host": "b.example.com", "port": 9090}
            }})),
            "servers[2:]{host,port}:\n  alpha: a.example.com,8080\n  beta: b.example.com,9090"
        );
    }

    #[test]
    fn test_encode_keyed_tabular_root() {
        assert_eq!(
            enc(json!({
                "alice": {"age": 30, "city": "Berlin"},
                "bob": {"age": 25, "city": "Oslo"}
            })),
            "[2:]{age,city}:\n  alice: 30,Berlin\n  bob: 25,Oslo"
        );
    }

    #[test]
    fn test_encode_list_form() {
        assert_eq!(
            enc(json!({"items": [1, [2, 3], {"a": 1}]})),
            "items[3]:\n  - 1\n  - [2]: 2,3\n  - a: 1"
        );
    }

    #[test]
    fn test_encode_array_of_empty_objects_uses_list_form() {
        assert_eq!(enc(json!({"items": [{}, {}]})), "items[2]:\n  -\n  -");
    }

    #[test]
    fn test_encode_root_primitive() {
        assert_eq!(enc(json!("hello")), "hello");
        assert_eq!(enc(json!(42)), "42");
        assert_eq!(enc(json!(true)), "true");
        assert_eq!(enc(json!(null)), "null");
    }

    #[test]
    fn test_encode_quoting() {
        assert_eq!(enc(json!("#hello")), "\"#hello\"");
        assert_eq!(enc(json!("+1")), "\"+1\"");
        assert_eq!(enc(json!({"café": 1})), "\"café\": 1");
    }

    #[test]
    fn test_encode_pipe_delimiter() {
        let options = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
        assert_eq!(
            encode(&json!({"tags": ["a", "b"]}), &options).unwrap(),
            "tags[2|]: a|b"
        );
    }

    #[test]
    fn test_encode_custom_indent() {
        let options = EncodeOptions::new().with_indent(Indent::Spaces(4));
        assert_eq!(
            encode(&json!({"user": {"name": "Alice"}}), &options).unwrap(),
            "user:\n    name: Alice"
        );
    }

    #[test]
    fn test_encode_object_and_array_type_guards() {
        assert!(encode_object(json!(42), &EncodeOptions::default()).is_err());
        assert!(encode_array(json!({"a": 1}), &EncodeOptions::default()).is_err());
    }

    #[test]
    fn test_encode_rejects_zero_indent_size() {
        let options = EncodeOptions::new().with_spaces(0);
        assert!(matches!(
            encode(&json!({"a": {"b": 1}}), &options),
            Err(ToonError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_encode_normalizes_non_finite_numbers() {
        assert_eq!(enc(json!({"a": f64::NAN})), "a: null");
    }
}
