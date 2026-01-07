use indexmap::IndexMap;

use crate::{
    constants::QUOTED_KEY_MARKER,
    types::{is_identifier_segment, JsonValue as Value, PathExpansionMode, ToonError, ToonResult},
};

pub fn should_expand_key(key: &str, mode: PathExpansionMode) -> Option<Vec<&str>> {
    match mode {
        PathExpansionMode::Off => None,
        PathExpansionMode::Safe => {
            // Quoted keys with dots shouldn't be expanded (they were explicitly quoted)
            if key.starts_with(QUOTED_KEY_MARKER) {
                return None;
            }

            if !key.contains('.') {
                return None;
            }

            let mut segment_count = 0;
            for segment in key.split('.') {
                if segment.is_empty() || !is_identifier_segment(segment) {
                    return None;
                }
                segment_count += 1;
            }

            if segment_count < 2 {
                return None;
            }

            let mut segments = Vec::with_capacity(segment_count);
            for segment in key.split('.') {
                segments.push(segment);
            }

            Some(segments)
        }
    }
}

pub fn deep_merge_value(
    target: &mut IndexMap<String, Value>,
    segments: &[&str],
    value: Value,
    strict: bool,
) -> ToonResult<()> {
    if segments.is_empty() {
        return Ok(());
    }

    if segments.len() == 1 {
        let key = segments[0];

        // Check for conflicts at leaf level
        if let Some(existing) = target.get(key) {
            if strict {
                return Err(ToonError::DeserializationError(format!(
                    "Path expansion conflict: key '{key}' already exists with value: {existing:?}",
                )));
            }
        }

        target.insert(key.to_string(), value);
        return Ok(());
    }

    let first_key = segments[0];
    let remaining_segments = &segments[1..];

    // Get or create nested object, handling type conflicts
    let nested_obj = if let Some(existing_value) = target.get_mut(first_key) {
        match existing_value {
            Value::Object(obj) => obj,
            _ => {
                if strict {
                    return Err(ToonError::DeserializationError(format!(
                        "Path expansion conflict: key '{first_key}' exists as non-object: \
                         {existing_value:?}",
                    )));
                }
                *existing_value = Value::Object(IndexMap::new());
                match existing_value {
                    Value::Object(obj) => obj,
                    _ => unreachable!(),
                }
            }
        }
    } else {
        target.insert(first_key.to_string(), Value::Object(IndexMap::new()));
        match target.get_mut(first_key).expect("key was just inserted") {
            Value::Object(obj) => obj,
            _ => unreachable!(),
        }
    };

    // Recurse into nested object
    deep_merge_value(nested_obj, remaining_segments, value, strict)
}

pub fn expand_paths_in_object(
    obj: IndexMap<String, Value>,
    mode: PathExpansionMode,
    strict: bool,
) -> ToonResult<IndexMap<String, Value>> {
    let mut result = IndexMap::with_capacity(obj.len());

    for (key, mut value) in obj {
        // Expand nested structures (arrays/objects) first (depth-first)
        value = expand_paths_recursive(value, mode, strict)?;

        match should_expand_key(&key, mode) {
            Some(segments) => {
                deep_merge_value(&mut result, &segments, value, strict)?;
            }
            None => {
                // Strip marker from quoted keys
                let clean_key = if key.starts_with(QUOTED_KEY_MARKER) {
                    let mut cleaned = key;
                    cleaned.remove(0);
                    cleaned
                } else {
                    key
                };

                // Check for conflicts with expanded keys
                if let Some(existing) = result.get(clean_key.as_str()) {
                    if strict {
                        return Err(ToonError::DeserializationError(format!(
                            "Key '{clean_key}' conflicts with existing value: {existing:?}",
                        )));
                    }
                }
                result.insert(clean_key, value);
            }
        }
    }

    Ok(result)
}

pub fn expand_paths_recursive(
    value: Value,
    mode: PathExpansionMode,
    strict: bool,
) -> ToonResult<Value> {
    match value {
        Value::Object(obj) => {
            let expanded = expand_paths_in_object(obj, mode, strict)?;
            Ok(Value::Object(expanded))
        }
        Value::Array(arr) => {
            let mut expanded = Vec::with_capacity(arr.len());
            for item in arr {
                expanded.push(expand_paths_recursive(item, mode, strict)?);
            }
            Ok(Value::Array(expanded))
        }
        _ => Ok(value),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_should_expand_key_off_mode() {
        assert!(should_expand_key("a.b.c", PathExpansionMode::Off).is_none());
    }

    #[test]
    fn test_should_expand_key_safe_mode() {
        // Valid expansions
        assert_eq!(
            should_expand_key("a.b", PathExpansionMode::Safe),
            Some(vec!["a", "b"])
        );
        assert_eq!(
            should_expand_key("a.b.c", PathExpansionMode::Safe),
            Some(vec!["a", "b", "c"])
        );

        // No dots
        assert!(should_expand_key("simple", PathExpansionMode::Safe).is_none());

        // Invalid segments (not IdentifierSegments)
        assert!(should_expand_key("a.bad-key", PathExpansionMode::Safe).is_none());
        assert!(should_expand_key("123.key", PathExpansionMode::Safe).is_none());
    }

    #[test]
    fn test_deep_merge_simple() {
        let mut target = IndexMap::new();
        deep_merge_value(&mut target, &["a", "b"], Value::from(json!(1)), true).unwrap();

        let expected = json!({"a": {"b": 1}});
        assert_eq!(Value::Object(target), Value::from(expected));
    }

    #[test]
    fn test_deep_merge_multiple_paths() {
        let mut target = IndexMap::new();

        deep_merge_value(&mut target, &["a", "b"], Value::from(json!(1)), true).unwrap();

        deep_merge_value(&mut target, &["a", "c"], Value::from(json!(2)), true).unwrap();

        let expected = json!({"a": {"b": 1, "c": 2}});
        assert_eq!(Value::Object(target), Value::from(expected));
    }

    #[test]
    fn test_deep_merge_conflict_strict() {
        let mut target = IndexMap::new();
        target.insert("a".to_string(), Value::from(json!({"b": 1})));

        let result = deep_merge_value(&mut target, &["a", "b"], Value::from(json!(2)), true);

        assert!(result.is_err());
    }

    #[test]
    fn test_deep_merge_conflict_non_strict() {
        let mut target = IndexMap::new();
        target.insert("a".to_string(), Value::from(json!({"b": 1})));

        deep_merge_value(&mut target, &["a", "b"], Value::from(json!(2)), false).unwrap();

        let expected = json!({"a": {"b": 2}});
        assert_eq!(Value::Object(target), Value::from(expected));
    }

    #[test]
    fn test_expand_paths_in_object() {
        let mut obj = IndexMap::new();
        obj.insert("a.b.c".to_string(), Value::from(json!(1)));
        obj.insert("simple".to_string(), Value::from(json!(2)));

        let result = expand_paths_in_object(obj, PathExpansionMode::Safe, true).unwrap();

        let expected = json!({"a": {"b": {"c": 1}}, "simple": 2});
        assert_eq!(Value::Object(result), Value::from(expected));
    }

    #[test]
    fn test_expand_paths_with_merge() {
        let mut obj = IndexMap::new();
        obj.insert("a.b".to_string(), Value::from(json!(1)));
        obj.insert("a.c".to_string(), Value::from(json!(2)));

        let result = expand_paths_in_object(obj, PathExpansionMode::Safe, true).unwrap();

        let expected = json!({"a": {"b": 1, "c": 2}});
        assert_eq!(Value::Object(result), Value::from(expected));
    }
}
