pub mod literal;
pub mod number;
pub mod string;
pub mod validation;

use indexmap::IndexMap;
pub use literal::{
    is_keyword,
    is_literal_like,
    is_numeric_like,
};
pub use number::format_canonical_number;
pub use string::{
    escape_string,
    is_valid_unquoted_key,
    needs_quoting,
    quote_string,
    unescape_string,
};

use crate::{
    constants::MAX_DEPTH,
    types::{
        JsonValue as Value,
        Number,
    },
};

/// Normalize a JSON value (converts NaN/Infinity to null, -0 to 0).
pub fn normalize(value: Value) -> Value {
    normalize_at(value, 0)
}

/// Descent is bounded by [`MAX_DEPTH`]: this runs as a whole-tree pre-pass
/// before the encoder's own depth check, so without a bound a pathological
/// value would abort the process on a stack overflow instead of being
/// rejected. A subtree past the bound is returned untouched — the encoder
/// rejects the document at [`MAX_DEPTH`], so it never reaches output.
fn normalize_at(value: Value, depth: usize) -> Value {
    if depth > MAX_DEPTH {
        return value;
    }

    match value {
        Value::Number(n) => {
            // Handle NegInt(0) case - convert to PosInt(0)
            if let Number::NegInt(0) = n {
                Value::Number(Number::from(0u64))
            } else if let Some(f) = n.as_f64() {
                if f.is_nan() || f.is_infinite() {
                    Value::Null
                } else if f == 0.0 && f.is_sign_negative() {
                    Value::Number(Number::from(0u64))
                } else {
                    Value::Number(n)
                }
            } else {
                Value::Number(n)
            }
        }
        Value::Object(obj) => {
            let normalized: IndexMap<String, Value> = obj
                .into_iter()
                .map(|(k, v)| (k, normalize_at(v, depth + 1)))
                .collect();
            Value::Object(normalized)
        }
        Value::Array(arr) => {
            let normalized: Vec<Value> = arr
                .into_iter()
                .map(|v| normalize_at(v, depth + 1))
                .collect();
            Value::Array(normalized)
        }
        _ => value,
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use serde_json::json;

    use super::*;

    #[test]
    fn test_normalize_nan() {
        let value = Value::from(json!(f64::NAN));
        let normalized = normalize(value);
        assert_eq!(normalized, Value::from(json!(null)));
    }

    #[test]
    fn test_normalize_infinity() {
        let value = Value::from(json!(f64::INFINITY));
        let normalized = normalize(value);
        assert_eq!(normalized, Value::from(json!(null)));

        let value = Value::from(json!(f64::NEG_INFINITY));
        let normalized = normalize(value);
        assert_eq!(normalized, Value::from(json!(null)));
    }

    #[test]
    fn test_normalize_negative_zero() {
        let value = Value::from(json!(-0.0));
        let normalized = normalize(value);
        assert_eq!(normalized, Value::from(json!(0)));
    }

    #[test]
    fn test_normalize_nested() {
        let value = Value::from(json!({
            "a": f64::NAN,
            "b": {
                "c": f64::INFINITY
            },
            "d": [1, f64::NAN, 3]
        }));

        let normalized = normalize(value);
        assert_eq!(
            normalized,
            Value::from(json!({
                "a": null,
                "b": {
                    "c": null
                },
                "d": [1, null, 3]
            }))
        );
    }

    #[test]
    fn test_normalize_normal_values() {
        let value = Value::from(json!({
            "name": "Alice",
            "age": 30,
            "score": f64::consts::PI
        }));

        let normalized = normalize(value.clone());
        assert_eq!(normalized, value);
    }
}
