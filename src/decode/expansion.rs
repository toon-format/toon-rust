use indexmap::IndexMap;

use crate::{
    constants::QUOTED_KEY_MARKER,
    types::{is_identifier_segment, JsonValue as Value, PathExpansionMode, ToonError, ToonResult},
};

pub fn should_expand_key(key: &str, mode: PathExpansionMode) -> Option<Vec<String>> {
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

            let segments: Vec<String> = key.split('.').map(String::from).collect();

            if segments.len() < 2 {
                return None;
            }

            // Only expand if all segments are valid identifiers (safety requirement)
            if segments.iter().all(|s| is_identifier_segment(s)) {
                Some(segments)
            } else {
                None
            }
        }
    }
}

pub fn deep_merge_value(
    target: &mut IndexMap<String, Value>,
    segments: &[String],
    value: Value,
    strict: bool,
) -> ToonResult<()> {
    if segments.is_empty() {
        return Ok(());
    }

    if segments.len() == 1 {
        let key = &segments[0];

        // Check for conflicts at leaf level
        if let Some(existing) = target.get(key) {
            if strict {
                return Err(ToonError::DeserializationError(format!(
                    "Path expansion conflict: key '{key}' already exists with value: {existing:?}",
                )));
            }
        }

        target.insert(key.clone(), value);
        return Ok(());
    }

    let first_key = &segments[0];
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
                // Replace non-object with empty object in non-strict mode
                *existing_value = Value::Object(IndexMap::new());
                match existing_value {
                    Value::Object(obj) => obj,
                    _ => unreachable!(),
                }
            }
        }
    } else {
        target.insert(first_key.clone(), Value::Object(IndexMap::new()));
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
    let mut result = IndexMap::new();

    for (key, mut value) in obj {
        // Expand nested objects first (depth-first)
        if let Value::Object(nested_obj) = value {
            value = Value::Object(expand_paths_in_object(nested_obj, mode, strict)?);
        }

        // Strip marker from quoted keys
        let clean_key = if key.starts_with(QUOTED_KEY_MARKER) {
            key.strip_prefix(QUOTED_KEY_MARKER).unwrap().to_string()
        } else {
            key.clone()
        };

        if let Some(segments) = should_expand_key(&key, mode) {
            deep_merge_value(&mut result, &segments, value, strict)?;
        } else {
            // Check for conflicts with expanded keys
            if let Some(existing) = result.get(&clean_key) {
                if strict {
                    return Err(ToonError::DeserializationError(format!(
                        "Key '{clean_key}' conflicts with existing value: {existing:?}",
                    )));
                }
            }
            result.insert(clean_key, value);
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
            let expanded: Result<Vec<_>, _> = arr
                .into_iter()
                .map(|v| expand_paths_recursive(v, mode, strict))
                .collect();
            Ok(Value::Array(expanded?))
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
            Some(vec!["a".to_string(), "b".to_string()])
        );
        assert_eq!(
            should_expand_key("a.b.c", PathExpansionMode::Safe),
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()])
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
        deep_merge_value(
            &mut target,
            &["a".to_string(), "b".to_string()],
            Value::from(json!(1)),
            true,
        )
        .unwrap();

        let expected = json!({"a": {"b": 1}});
        assert_eq!(Value::Object(target), Value::from(expected));
    }

    #[test]
    fn test_deep_merge_multiple_paths() {
        let mut target = IndexMap::new();

        deep_merge_value(
            &mut target,
            &["a".to_string(), "b".to_string()],
            Value::from(json!(1)),
            true,
        )
        .unwrap();

        deep_merge_value(
            &mut target,
            &["a".to_string(), "c".to_string()],
            Value::from(json!(2)),
            true,
        )
        .unwrap();

        let expected = json!({"a": {"b": 1, "c": 2}});
        assert_eq!(Value::Object(target), Value::from(expected));
    }

    #[test]
    fn test_deep_merge_conflict_strict() {
        let mut target = IndexMap::new();
        target.insert("a".to_string(), Value::from(json!({"b": 1})));

        let result = deep_merge_value(
            &mut target,
            &["a".to_string(), "b".to_string()],
            Value::from(json!(2)),
            true,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_deep_merge_conflict_non_strict() {
        let mut target = IndexMap::new();
        target.insert("a".to_string(), Value::from(json!({"b": 1})));

        deep_merge_value(
            &mut target,
            &["a".to_string(), "b".to_string()],
            Value::from(json!(2)),
            false,
        )
        .unwrap();

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
