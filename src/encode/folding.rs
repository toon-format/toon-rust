use std::collections::HashSet;

use crate::types::{is_identifier_segment, JsonValue as Value, KeyFoldingMode};

/// Result of chain analysis for folding.
///
/// # Examples
/// ```
/// use std::collections::HashSet;
/// use serde_json::json;
/// use toon_format::encode::folding::analyze_foldable_chain;
/// use toon_format::types::JsonValue;
///
/// let value: JsonValue = json!({"b": {"c": 1}}).into();
/// let existing: HashSet<&str> = HashSet::new();
/// let chain = analyze_foldable_chain("a", &value, usize::MAX, &existing).unwrap();
/// assert_eq!(chain.folded_key, "a.b.c");
/// ```
pub struct FoldableChain<'a> {
    /// The folded key path (e.g., "a.b.c")
    pub folded_key: String,
    /// The leaf value at the end of the chain
    pub leaf_value: &'a Value,
    /// Number of segments that were folded
    pub depth_folded: usize,
}

/// Check if a value is a single-key object suitable for folding.
fn is_single_key_object(value: &Value) -> Option<(&str, &Value)> {
    if let Value::Object(obj) = value {
        if obj.len() == 1 {
            return obj.iter().next().map(|(key, val)| (key.as_str(), val));
        }
    }
    None
}

/// Analyze if a key-value pair can be folded into dotted notation.
///
/// # Examples
/// ```
/// use std::collections::HashSet;
/// use serde_json::json;
/// use toon_format::encode::folding::analyze_foldable_chain;
/// use toon_format::types::JsonValue;
///
/// let value: JsonValue = json!({"b": {"c": 1}}).into();
/// let existing: HashSet<&str> = HashSet::new();
/// let chain = analyze_foldable_chain("a", &value, usize::MAX, &existing).unwrap();
/// assert_eq!(chain.depth_folded, 3);
/// ```
pub fn analyze_foldable_chain<'a>(
    key: &'a str,
    value: &'a Value,
    flatten_depth: usize,
    existing_keys: &HashSet<&str>,
) -> Option<FoldableChain<'a>> {
    if !is_identifier_segment(key) {
        return None;
    }

    let mut segments = Vec::with_capacity(4);
    segments.push(key);
    let mut current_value = value;

    // Follow single-key object chain until we hit a multi-key object or leaf
    while let Some((next_key, next_value)) = is_single_key_object(current_value) {
        if flatten_depth != usize::MAX && segments.len() >= flatten_depth {
            break;
        }

        if !is_identifier_segment(next_key) {
            break;
        }

        segments.push(next_key);
        current_value = next_value;
    }

    // Must fold at least 2 segments to be worthwhile
    if segments.len() < 2 {
        return None;
    }

    let total_len =
        segments.iter().map(|segment| segment.len()).sum::<usize>() + segments.len() - 1;
    let mut folded_key = String::with_capacity(total_len);
    for (idx, segment) in segments.iter().enumerate() {
        if idx > 0 {
            folded_key.push('.');
        }
        folded_key.push_str(segment);
    }

    // Don't fold if it would collide with an existing key
    if existing_keys.contains(folded_key.as_str()) {
        return None;
    }

    Some(FoldableChain {
        folded_key,
        leaf_value: current_value,
        depth_folded: segments.len(),
    })
}

/// Return true when folding should be applied for the current mode.
///
/// # Examples
/// ```
/// use toon_format::encode::folding::should_fold;
/// use toon_format::types::KeyFoldingMode;
///
/// assert!(!should_fold(KeyFoldingMode::Off, &None));
/// ```
pub fn should_fold(mode: KeyFoldingMode, chain: &Option<FoldableChain>) -> bool {
    match mode {
        KeyFoldingMode::Off => false,
        KeyFoldingMode::Safe => chain.is_some(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_is_single_key_object() {
        let val = Value::from(json!({"a": 1}));
        assert!(is_single_key_object(&val).is_some());

        let val = Value::from(json!({"a": 1, "b": 2}));
        assert!(is_single_key_object(&val).is_none());

        let val = Value::from(json!(42));
        assert!(is_single_key_object(&val).is_none());
    }

    #[test]
    fn test_analyze_simple_chain() {
        let val = Value::from(json!({"b": {"c": 1}}));
        let existing: HashSet<&str> = HashSet::new();

        let result = analyze_foldable_chain("a", &val, usize::MAX, &existing);
        assert!(result.is_some());

        let chain = result.unwrap();
        assert_eq!(chain.folded_key, "a.b.c");
        assert_eq!(chain.depth_folded, 3);
        assert_eq!(chain.leaf_value, &Value::from(json!(1)));
    }

    #[test]
    fn test_analyze_with_flatten_depth() {
        let val = Value::from(json!({"b": {"c": {"d": 1}}}));
        let existing: HashSet<&str> = HashSet::new();

        let result = analyze_foldable_chain("a", &val, 2, &existing);
        assert!(result.is_some());

        let chain = result.unwrap();
        assert_eq!(chain.folded_key, "a.b");
        assert_eq!(chain.depth_folded, 2);
    }

    #[test]
    fn test_analyze_stops_at_multi_key() {
        let val = Value::from(json!({"b": {"c": 1, "d": 2}}));
        let existing: HashSet<&str> = HashSet::new();

        let result = analyze_foldable_chain("a", &val, usize::MAX, &existing);
        assert!(result.is_some());

        let chain = result.unwrap();
        assert_eq!(chain.folded_key, "a.b");
        assert_eq!(chain.depth_folded, 2);
    }

    #[test]
    fn test_analyze_rejects_non_identifier() {
        let val = Value::from(json!({"c": 1}));
        let existing: HashSet<&str> = HashSet::new();

        let result = analyze_foldable_chain("bad-key", &val, usize::MAX, &existing);
        assert!(result.is_none());
    }

    #[test]
    fn test_analyze_detects_collision() {
        let val = Value::from(json!({"b": 1}));
        let existing_key = String::from("a.b");
        let mut existing: HashSet<&str> = HashSet::new();
        existing.insert(existing_key.as_str());

        let result = analyze_foldable_chain("a", &val, usize::MAX, &existing);
        assert!(result.is_none());
    }

    #[test]
    fn test_analyze_too_short_chain() {
        let val = Value::from(json!(42));
        let existing: HashSet<&str> = HashSet::new();

        let result = analyze_foldable_chain("a", &val, usize::MAX, &existing);
        assert!(result.is_none());
    }
}
