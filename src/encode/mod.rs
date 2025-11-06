//! Encoder Implementation
pub mod primitives;
pub mod writer;
use indexmap::IndexMap;

use crate::{
    constants::MAX_DEPTH,
    types::{
        EncodeOptions,
        IntoJsonValue,
        JsonValue as Value,
        ToonError,
        ToonResult,
    },
    utils::{
        normalize,
        validation::validate_depth,
    },
};

/// Encode a JSON value to TOON format with custom options.
///
/// This function accepts either `JsonValue` or `serde_json::Value` and converts
/// automatically.
///
/// # Examples
///
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
pub fn encode<V: IntoJsonValue>(value: V, options: &EncodeOptions) -> ToonResult<String> {
    let json_value = value.into_json_value();
    encode_impl(&json_value, options)
}

fn encode_impl(value: &Value, options: &EncodeOptions) -> ToonResult<String> {
    let normalized: Value = normalize(value.clone());
    let mut writer = writer::Writer::new(options.clone());

    match &normalized {
        Value::Array(arr) => {
            write_array(&mut writer, None, arr, 0)?;
        }
        Value::Object(obj) => {
            write_object(&mut writer, obj, 0)?;
        }
        _ => {
            write_primitive_value(&mut writer, &normalized)?;
        }
    }

    Ok(writer.finish())
}

/// Encode a JSON value to TOON format with default options.
///
/// This function accepts either `JsonValue` or `serde_json::Value` and converts
/// automatically.
///
/// # Examples
///
/// ```
/// use toon_format::encode_default;
/// use serde_json::json;
///
/// // Simple object
/// let data = json!({"name": "Alice", "age": 30});
/// let toon = encode_default(&data)?;
/// assert!(toon.contains("name: Alice"));
/// assert!(toon.contains("age: 30"));
///
/// // Primitive array
/// let data = json!({"tags": ["reading", "gaming", "coding"]});
/// let toon = encode_default(&data)?;
/// assert_eq!(toon, "tags[3]: reading,gaming,coding");
///
/// // Tabular array
/// let data = json!({
///     "users": [
///         {"id": 1, "name": "Alice"},
///         {"id": 2, "name": "Bob"}
///     ]
/// });
/// let toon = encode_default(&data)?;
/// assert!(toon.contains("users[2]{id,name}:"));
/// # Ok::<(), toon_format::ToonError>(())
/// ```
pub fn encode_default<V: IntoJsonValue>(value: V) -> ToonResult<String> {
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
/// assert!(encode_object(&json!(42), &EncodeOptions::default()).is_err());
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

fn write_object(
    writer: &mut writer::Writer,
    obj: &IndexMap<String, Value>,
    depth: usize,
) -> ToonResult<()> {
    validate_depth(depth, MAX_DEPTH)?;

    let keys: Vec<&String> = obj.keys().collect();

    for (i, key) in keys.iter().enumerate() {
        if i > 0 {
            writer.write_newline()?;
        }

        if depth > 0 {
            writer.write_indent(depth)?;
        }

        let value = &obj[*key];

        match value {
            Value::Array(arr) => {
                write_array(writer, Some(key), arr, depth)?;
            }
            Value::Object(nested_obj) => {
                writer.write_key(key)?;
                writer.write_char(':')?;
                writer.write_newline()?;
                write_object(writer, nested_obj, depth + 1)?;
            }
            _ => {
                writer.write_key(key)?;
                writer.write_char(':')?;
                writer.write_char(' ')?;
                write_primitive_value(writer, value)?;
            }
        }
    }

    Ok(())
}

fn write_array(
    writer: &mut writer::Writer,
    key: Option<&str>,
    arr: &[Value],
    depth: usize,
) -> ToonResult<()> {
    validate_depth(depth, MAX_DEPTH)?;

    if arr.is_empty() {
        writer.write_empty_array_with_key(key)?;
        return Ok(());
    }

    // Choose encoding format: tabular > primitive inline > nested list
    if let Some(keys) = is_tabular_array(arr) {
        encode_tabular_array(writer, key, arr, &keys, depth)?;
    } else if is_primitive_array(arr) {
        encode_primitive_array(writer, key, arr, depth)?;
    } else {
        encode_nested_array(writer, key, arr, depth)?;
    }

    Ok(())
}

/// Check if an array can be encoded as tabular format (uniform objects with
/// primitive values).
fn is_tabular_array(arr: &[Value]) -> Option<Vec<String>> {
    if arr.is_empty() {
        return None;
    }

    let first = arr.first()?;
    if !first.is_object() {
        return None;
    }

    let first_obj = first.as_object()?;
    let keys: Vec<String> = first_obj.keys().cloned().collect();

    // All values must be primitives for tabular format
    for value in first_obj.values() {
        if !is_primitive(value) {
            return None;
        }
    }

    // All objects must have the same keys and primitive values
    for val in arr.iter().skip(1) {
        if let Some(obj) = val.as_object() {
            let obj_keys: Vec<String> = obj.keys().cloned().collect();
            if keys != obj_keys {
                return None;
            }
            for value in obj.values() {
                if !is_primitive(value) {
                    return None;
                }
            }
        } else {
            return None;
        }
    }

    Some(keys)
}

/// Check if a value is a primitive (not array or object).
fn is_primitive(value: &Value) -> bool {
    matches!(
        value,
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
    )
}

/// Check if all array elements are primitives.
fn is_primitive_array(arr: &[Value]) -> bool {
    arr.iter().all(is_primitive)
}

fn encode_primitive_array(
    writer: &mut writer::Writer,
    key: Option<&str>,
    arr: &[Value],
    depth: usize,
) -> ToonResult<()> {
    writer.write_array_header(key, arr.len(), None, depth)?;
    writer.write_char(' ')?;

    for (i, val) in arr.iter().enumerate() {
        if i > 0 {
            writer.write_delimiter()?;
        }
        write_primitive_value(writer, val)?;
    }

    Ok(())
}

fn write_primitive_value(writer: &mut writer::Writer, value: &Value) -> ToonResult<()> {
    match value {
        Value::Null => writer.write_str("null"),
        Value::Bool(b) => writer.write_str(&b.to_string()),
        Value::Number(n) => writer.write_str(&n.to_string()),
        Value::String(s) => {
            if writer.needs_quoting(s) {
                writer.write_quoted_string(s)
            } else {
                writer.write_str(s)
            }
        }
        _ => Err(ToonError::InvalidInput(
            "Expected primitive value".to_string(),
        )),
    }
}

fn encode_tabular_array(
    writer: &mut writer::Writer,
    key: Option<&str>,
    arr: &[Value],
    keys: &[String],
    depth: usize,
) -> ToonResult<()> {
    writer.write_array_header(key, arr.len(), Some(keys), depth)?;
    writer.write_newline()?;

    for (row_index, obj_val) in arr.iter().enumerate() {
        if let Some(obj) = obj_val.as_object() {
            writer.write_indent(depth + 1)?;

            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    writer.write_delimiter()?;
                }

                if let Some(val) = obj.get(key) {
                    write_primitive_value(writer, val)?;
                } else {
                    writer.write_str("null")?;
                }
            }

            if row_index < arr.len() - 1 {
                writer.write_newline()?;
            }
        }
    }

    Ok(())
}

fn encode_nested_array(
    writer: &mut writer::Writer,
    key: Option<&str>,
    arr: &[Value],
    depth: usize,
) -> ToonResult<()> {
    writer.write_array_header(key, arr.len(), None, depth)?;
    writer.write_newline()?;

    for (i, val) in arr.iter().enumerate() {
        writer.write_indent(depth + 1)?;
        writer.write_char('-')?;
        writer.write_char(' ')?;

        match val {
            Value::Array(inner_arr) => {
                write_array(writer, None, inner_arr, depth + 1)?;
            }
            Value::Object(obj) => {
                let keys: Vec<&String> = obj.keys().collect();
                if let Some(first_key) = keys.first() {
                    let first_val = &obj[*first_key];

                    writer.write_key(first_key)?;
                    writer.write_char(':')?;
                    writer.write_char(' ')?;
                    match first_val {
                        Value::Array(arr) => {
                            write_array(writer, None, arr, depth + 1)?;
                        }
                        Value::Object(nested_obj) => {
                            writer.write_newline()?;
                            write_object(writer, nested_obj, depth + 2)?;
                        }
                        _ => {
                            write_primitive_value(writer, first_val)?;
                        }
                    }

                    for key in keys.iter().skip(1) {
                        writer.write_newline()?;
                        writer.write_indent(depth + 2)?;
                        writer.write_key(key)?;
                        writer.write_char(':')?;
                        writer.write_char(' ')?;

                        let value = &obj[*key];
                        match value {
                            Value::Array(arr) => {
                                write_array(writer, None, arr, depth + 2)?;
                            }
                            Value::Object(nested_obj) => {
                                writer.write_newline()?;
                                write_object(writer, nested_obj, depth + 3)?;
                            }
                            _ => {
                                write_primitive_value(writer, value)?;
                            }
                        }
                    }
                }
            }
            _ => {
                write_primitive_value(writer, val)?;
            }
        }

        if i < arr.len() - 1 {
            writer.write_newline()?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use core::f64;

    use serde_json::json;

    use super::*;

    #[test]
    fn test_encode_null() {
        let value = json!(null);
        assert_eq!(encode_default(&value).unwrap(), "null");
    }

    #[test]
    fn test_encode_bool() {
        assert_eq!(encode_default(json!(true)).unwrap(), "true");
        assert_eq!(encode_default(json!(false)).unwrap(), "false");
    }

    #[test]
    fn test_encode_number() {
        assert_eq!(encode_default(json!(42)).unwrap(), "42");
        assert_eq!(
            encode_default(json!(f64::consts::PI)).unwrap(),
            "3.141592653589793"
        );
        assert_eq!(encode_default(json!(-5)).unwrap(), "-5");
    }

    #[test]
    fn test_encode_string() {
        assert_eq!(encode_default(json!("hello")).unwrap(), "hello");
        assert_eq!(
            encode_default(json!("hello world")).unwrap(),
            "\"hello world\""
        );
    }

    #[test]
    fn test_encode_simple_object() {
        let obj = json!({"name": "Alice", "age": 30});
        let result = encode_default(&obj).unwrap();
        assert!(result.contains("name: Alice"));
        assert!(result.contains("age: 30"));
    }

    #[test]
    fn test_encode_primitive_array() {
        let obj = json!({"tags": ["reading", "gaming", "coding"]});
        let result = encode_default(&obj).unwrap();
        assert_eq!(result, "tags[3]: reading,gaming,coding");
    }

    #[test]
    fn test_encode_tabular_array() {
        let obj = json!({
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ]
        });
        let result = encode_default(&obj).unwrap();
        assert!(result.contains("users[2]{id,name}:"));
        assert!(result.contains("1,Alice"));
        assert!(result.contains("2,Bob"));
    }

    #[test]
    fn test_encode_empty_array() {
        let obj = json!({"items": []});
        let result = encode_default(&obj).unwrap();
        assert_eq!(result, "items[0]:");
    }

    #[test]
    fn test_encode_nested_object() {
        let obj = json!({
            "user": {
                "name": "Alice",
                "age": 30
            }
        });
        let result = encode_default(&obj).unwrap();
        assert!(result.contains("user:"));
        assert!(result.contains("name: Alice"));
        assert!(result.contains("age: 30"));
    }
}
