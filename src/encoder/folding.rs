/* rune-xero/src/encoder/folding.rs */
//!▫~•◦-------------------------------‣
//! # RUNE-Xero – Zero-Copy Key Folding Module
//!▫~•◦----------------------------------------‣
//!
//! Analyzes object structures to detect foldable key chains (e.g., nesting
//! `{"a": {"b": 1}}` -> `a.b: 1`).
//! Optimized to minimize allocations during chain detection.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::decoder::parser::ast::Value;
use crate::types::{KeyFoldingMode, is_identifier_segment};
use std::borrow::Cow;

/// Result of chain analysis for folding.
/// Zero-copy variant: borrows the leaf value and holds joined key.
pub struct FoldableChain<'a> {
    /// The folded key path (e.g., "a.b.c"). Owned string required for output/collision check.
    pub folded_key: String,
    /// The leaf value at the end of the chain. Borrowed to avoid cloning tree.
    pub leaf_value: &'a Value<'a>,
    /// Number of segments that were folded
    pub depth_folded: usize,
}

/// Check if a value is a single-key object suitable for folding.
/// Returns borrowed key and value references.
fn is_single_key_object<'a>(value: &'a Value<'a>) -> Option<(&'a str, &'a Value<'a>)> {
    match value {
        Value::Object(obj) if obj.len() == 1 => {
            let (k, v) = &obj[0];
            Some((k, v))
        }
        _ => None,
    }
}

/// Analyze if a key-value pair can be folded into dotted notation.
pub fn analyze_foldable_chain<'a>(
    key: &'a str,
    value: &'a Value<'a>,
    flatten_depth: usize,
    existing_keys: &[String], // Check against owned strings
) -> Option<FoldableChain<'a>> {
    if !is_identifier_segment(key) {
        return None;
    }

    // Use a Vec of slices to avoid allocating intermediate Strings
    let mut segments: Vec<&str> = Vec::with_capacity(4);
    segments.push(key);
    
    let mut current_value = value;

    // Follow single-key object chain until we hit a multi-key object or leaf
    while let Some((next_key, next_value)) = is_single_key_object(current_value) {
        if segments.len() >= flatten_depth {
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

    // Construct the folded key once. This allocation is unavoidable if we need
    // to verify against a list of full keys or output a joined identifier.
    let folded_key = segments.join(".");

    // Don't fold if it would collide with an existing key
    if existing_keys.iter().any(|k| k == &folded_key) {
        return None;
    }

    Some(FoldableChain {
        folded_key,
        leaf_value: current_value, // Borrowed! No clone.
        depth_folded: segments.len(),
    })
}

pub fn should_fold(mode: KeyFoldingMode, chain: &Option<FoldableChain>) -> bool {
    match mode {
        KeyFoldingMode::Off => false,
        KeyFoldingMode::Safe => chain.is_some(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::parser::ast::Value;

    // Helper to construct AST objects easily for testing
    fn obj<'a>(key: &'a str, val: Value<'a>) -> Value<'a> {
        Value::Object(vec![(key, val)])
    }

    #[test]
    fn test_is_single_key_object() {
        let val = obj("a", Value::Float(1.0));
        assert!(is_single_key_object(&val).is_some());

        let val = Value::Object(vec![
            ("a", Value::Float(1.0)), 
            ("b", Value::Float(2.0))
        ]);
        assert!(is_single_key_object(&val).is_none());

        let val = Value::Float(42.0);
        assert!(is_single_key_object(&val).is_none());
    }

    #[test]
    fn test_analyze_simple_chain() {
        // {"b": {"c": 1}}
        let val = obj("b", obj("c", Value::Float(1.0)));
        let existing: Vec<String> = vec![];

        let result = analyze_foldable_chain("a", &val, usize::MAX, &existing);
        assert!(result.is_some());

        let chain = result.unwrap();
        assert_eq!(chain.folded_key, "a.b.c");
        assert_eq!(chain.depth_folded, 3);
        // Leaf should be 1.0
        match chain.leaf_value {
            Value::Float(f) => assert_eq!(*f, 1.0),
            _ => panic!("Wrong leaf value"),
        }
    }

    #[test]
    fn test_analyze_with_flatten_depth() {
        // {"b": {"c": {"d": 1}}}
        let val = obj("b", obj("c", obj("d", Value::Float(1.0))));
        let existing: Vec<String> = vec![];

        // Depth 2 means fold "a.b" but stop there
        let result = analyze_foldable_chain("a", &val, 2, &existing);
        assert!(result.is_some());

        let chain = result.unwrap();
        assert_eq!(chain.folded_key, "a.b");
        assert_eq!(chain.depth_folded, 2);
    }

    #[test]
    fn test_analyze_stops_at_multi_key() {
        // {"b": {"c": 1, "d": 2}}
        let val = obj("b", Value::Object(vec![
            ("c", Value::Float(1.0)),
            ("d", Value::Float(2.0))
        ]));
        let existing: Vec<String> = vec![];

        let result = analyze_foldable_chain("a", &val, usize::MAX, &existing);
        assert!(result.is_some());

        let chain = result.unwrap();
        assert_eq!(chain.folded_key, "a.b");
        assert_eq!(chain.depth_folded, 2);
    }

    #[test]
    fn test_analyze_detects_collision() {
        let val = obj("b", Value::Float(1.0));
        let existing: Vec<String> = vec![String::from("a.b")];

        let result = analyze_foldable_chain("a", &val, usize::MAX, &existing);
        assert!(result.is_none());
    }
}