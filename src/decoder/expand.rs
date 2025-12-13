/* src/decoder/expand.rs */
//!▫~•◦------------------------------‣
//! # Rune-Xero – Zero-Copy Path Expansion Logic for Rune Configuration.
//!▫~•◦-----------------------------------------------------------------‣
//!
//! This module handles the expansion of dot-notation keys (e.g., "a.b") into
//! nested objects. It uses zero-allocation iterators to split and traverse
//! paths, allocating memory only when inserting into the final structure.
//!
//! ## Key Capabilities
//! - **Iterator-Based Splitting:** Validates and merges paths without `Vec<String>`.
//! - **In-Place String Modification:** Strips quote markers by draining the existing buffer.
//! - **Recursion Safety:** Handles nested merges with strict conflict detection.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use indexmap::IndexMap;
use std::iter::Peekable;

use crate::{
    constants::QUOTED_KEY_MARKER,
    types::{PathExpansionMode, RuneError, RuneResult, is_identifier_segment},
};

// Local type alias using static lifetime for simplicity
type Value = crate::types::Value<'static>;

/// Checks if a key should be expanded based on the mode.
///
/// Returns `true` if the key contains valid segments for expansion.
/// This operation is zero-allocation; it validates slices in place.
pub fn should_expand_key(key: &str, mode: PathExpansionMode) -> bool {
    match mode {
        PathExpansionMode::Off => false,
        PathExpansionMode::Safe => {
            // Quoted keys with dots shouldn't be expanded (explicitly quoted)
            if key.starts_with(QUOTED_KEY_MARKER) {
                return false;
            }

            // Fast path: no dot means no expansion
            if !key.contains('.') {
                return false;
            }

            let mut segments = key.split('.');
            
            // Must have at least two segments (e.g., "a.b")
            let first = segments.next();
            let second = segments.next();

            if first.is_none() || second.is_none() {
                return false;
            }

            // Validate all segments are proper identifiers
            // We iterate the full split to ensure safety
            key.split('.').all(|s| !s.is_empty() && is_identifier_segment(s))
        }
    }
}

/// Merges a value into the target map at the path specified by the iterator.
///
/// Uses a `Peekable` iterator to detect leaf nodes without allocating a vector.
/// 
/// **Zero-Copy Note**: Uses `String` keys to match the `Object<'a>` type alias.
pub fn deep_merge_value<'a>(
    target: &mut IndexMap<String, Value>,
    mut segments: Peekable<impl Iterator<Item = &'a str> + 'a>,
    value: Value,
    strict: bool,
) -> RuneResult<()> {
    // Get the current segment
    let current_key = match segments.next() {
        Some(k) => k,
        None => return Ok(()), // Should allow empty paths? Currently no-op.
    };

    // Check if we are at the leaf node (no more segments)
    if segments.peek().is_none() {
        // Leaf Level: Insert value
        if strict && target.contains_key(current_key) {
            let existing = &target[current_key];
            return Err(RuneError::DeserializationError(format!(
                "Path expansion conflict: key '{current_key}' already exists with value: {existing:?}",
            )));
        }
        target.insert(current_key.to_string(), value);
    } else {
        // Intermediate Level: Navigate or create object
        
        // Check strictness before mutation
        if strict {
            if let Some(existing) = target.get(current_key) {
                if !matches!(existing, Value::Object(_)) {
                    return Err(RuneError::DeserializationError(format!(
                        "Path expansion conflict: key '{current_key}' exists as non-object: {existing:?}",
                    )));
                }
            }
        }

        // Get mutable reference to nested object, creating if necessary
        // We use the Entry API pattern logic manually here to satisfy borrow checker with recursive structs
        let key_string = current_key.to_string();
        let nested_obj = if let Some(val) = target.get_mut(&key_string) {
            match val {
                Value::Object(obj) => obj,
                _ => {
                    // Non-strict mode: overwrite non-object with new object
                    *val = Value::Object(IndexMap::new());
                    match val {
                        Value::Object(obj) => obj,
                        _ => unreachable!("Just set to object"),
                    }
                }
            }
        } else {
            target.insert(key_string.clone(), Value::Object(IndexMap::new()));
            match target.get_mut(&key_string).unwrap() {
                Value::Object(obj) => obj,
                _ => unreachable!("Just inserted object"),
            }
        };

        // Recurse
        deep_merge_value(nested_obj, segments, value, strict)?;
    }

    Ok(())
}

/// Expands dot-notation keys within an object.
///
/// Consumes the input object and returns a new one with expanded paths.
/// Reuses string allocations where possible.
/// 
/// **Zero-Copy Note**: Uses `String` keys to match the `Object<'a>` type alias.
pub fn expand_paths_in_object(
    obj: IndexMap<String, Value>,
    mode: PathExpansionMode,
    strict: bool,
) -> RuneResult<IndexMap<String, Value>> {
    let mut result = IndexMap::new();

    for (key, mut value) in obj {
        // Recursively expand nested objects first
        if let Value::Object(nested_obj) = value {
            value = Value::Object(expand_paths_in_object(nested_obj, mode, strict)?);
        }

        if should_expand_key(&key, mode) {
            // Convert key to owned String to ensure it lives long enough for segments
            let key_str = key.as_str();
            let segments = key_str.split('.').peekable();
            deep_merge_value(&mut result, segments, value, strict)?;
        } else {
            // Check for conflict
            let clean_key = if key.starts_with(QUOTED_KEY_MARKER) {
                // If quoted, strip the marker
                key[QUOTED_KEY_MARKER.len()..].to_string()
            } else {
                key
            };

            if strict {
                if let Some(existing) = result.get(&clean_key) {
                    return Err(RuneError::DeserializationError(format!(
                        "Key '{clean_key}' conflicts with existing value: {existing:?}",
                    )));
                }
            }
            result.insert(clean_key, value);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use super::*;

    #[test]
    fn test_should_expand_key_off_mode() {
        assert!(!should_expand_key("a.b.c", PathExpansionMode::Off));
    }

    #[test]
    fn test_should_expand_key_safe_mode() {
        // Valid expansions
        assert!(should_expand_key("a.b", PathExpansionMode::Safe));
        assert!(should_expand_key("a.b.c", PathExpansionMode::Safe));

        // No dots
        assert!(!should_expand_key("simple", PathExpansionMode::Safe));

        // Invalid segments
        assert!(!should_expand_key("a.bad-key", PathExpansionMode::Safe));
        assert!(!should_expand_key("123.key", PathExpansionMode::Safe));
        
        // Quoted
        assert!(!should_expand_key(&format!("{}a.b", QUOTED_KEY_MARKER), PathExpansionMode::Safe));
    }

    #[test]
    fn test_deep_merge_simple() {
        let mut target: IndexMap<String, Value> = IndexMap::new();
        deep_merge_value(
            &mut target,
            "a.b".split('.').peekable(),
            Value::from(json!(1)),
            true,
        )
        .unwrap();

        let expected = json!({"a": {"b": 1}});
        assert_eq!(Value::Object(target), Value::from(expected));
    }

    #[test]
    fn test_deep_merge_conflict_strict() {
        let mut target: IndexMap<String, Value> = IndexMap::new();
        target.insert("a".to_string(), Value::from(json!({"b": 1})));

        let result = deep_merge_value(
            &mut target,
            "a.b".split('.').peekable(),
            Value::from(json!(2)),
            true,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_expand_paths_with_drain_optimization() {
        let mut obj: IndexMap<String, Value> = IndexMap::new();
        // "___a.b" -> should become "a.b": 1 (no expansion, marker stripped)
        let quoted_key = format!("{}a.b", QUOTED_KEY_MARKER);
        obj.insert(quoted_key, Value::from(json!(1)));

        let result = expand_paths_in_object(obj, PathExpansionMode::Safe, true).unwrap();
        
        // Should contain key "a.b" literally
        assert!(result.contains_key("a.b"));
        assert_eq!(result["a.b"], Value::from(json!(1)));
    }
}
