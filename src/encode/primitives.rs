pub fn is_primitive(value: &serde_json::Value) -> bool {
    matches!(
        value,
        serde_json::Value::Null
            | serde_json::Value::Bool(_)
            | serde_json::Value::Number(_)
            | serde_json::Value::String(_)
    )
}

pub fn all_primitives(values: &[serde_json::Value]) -> bool {
    values.iter().all(is_primitive)
}

pub fn normalize_value(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Null => serde_json::Value::Null,
        serde_json::Value::Bool(b) => serde_json::Value::Bool(b),
        serde_json::Value::Number(n) => serde_json::Value::Number(n),
        serde_json::Value::String(s) => serde_json::Value::String(s),
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(normalize_value).collect())
        }
        serde_json::Value::Object(obj) => serde_json::Value::Object(
            obj.into_iter()
                .map(|(k, v)| (k, normalize_value(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_is_primitive() {
        assert!(is_primitive(&json!(null)));
        assert!(is_primitive(&json!(true)));
        assert!(is_primitive(&json!(42)));
        assert!(is_primitive(&json!("hello")));
        assert!(!is_primitive(&json!([])));
        assert!(!is_primitive(&json!({})));
    }

    #[test]
    fn test_all_primitives() {
        assert!(all_primitives(&vec![json!(1), json!(2), json!(3)]));
        assert!(all_primitives(&vec![json!("a"), json!("b")]));
        assert!(all_primitives(&vec![json!(null), json!(true), json!(42)]));
        assert!(!all_primitives(&vec![json!(1), json!([]), json!(3)]));
        assert!(!all_primitives(&vec![json!({})]));
    }

    #[test]
    fn test_normalize_value() {
        assert_eq!(normalize_value(json!(null)), json!(null));
        assert_eq!(normalize_value(json!(42)), json!(42));
        assert_eq!(normalize_value(json!("hello")), json!("hello"));

        let normalized = normalize_value(json!({"a": 1, "b": [2, 3]}));
        assert_eq!(normalized, json!({"a": 1, "b": [2, 3]}));
    }
}
