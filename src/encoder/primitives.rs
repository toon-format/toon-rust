/* rune-xero/src/encoder/primitives.rs */
//!▫~•◦-----------------------------------‣
//! # RUNE-Xero – Zero-Copy Primitive Checks Module
//!▫~•◦---------------------------------------------‣
//!
//! Helper functions to inspect RUNE values without allocation.
//! Determines if values are primitives (scalar) or compound (arrays/objects)
//! to guide encoding strategies (e.g., inline vs block formatting).
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::decoder::parser::ast::Value;

/// Checks if a value is a primitive (scalar) type.
/// Returns true for Null, Bool, Float, Str, Raw.
/// Returns false for Array, Object.
#[inline]
pub fn is_primitive(value: &Value<'_>) -> bool {
    match value {
        Value::Null | Value::Bool(_) | Value::Float(_) | Value::Str(_) | Value::Raw(_) => true,
        Value::Array(_) | Value::Object(_) => false,
    }
}

/// Checks if a slice of values contains only primitives.
/// Used to determine if an array can be encoded as a compact/inline list.
pub fn all_primitives(values: &[Value<'_>]) -> bool {
    values.iter().all(is_primitive)
}

// Note: `normalize_value` removed as Zero-Copy architecture forbids 
// creating new value trees for normalization. Normalization (e.g. -0.0 -> 0.0)
// should occur Just-In-Time during the write phase in `writer.rs`.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::parser::ast::Value;

    #[test]
    fn test_is_primitive() {
        assert!(is_primitive(&Value::Null));
        assert!(is_primitive(&Value::Bool(true)));
        assert!(is_primitive(&Value::Float(42.0)));
        assert!(is_primitive(&Value::Str("hello")));
        assert!(!is_primitive(&Value::Array(vec![])));
        assert!(!is_primitive(&Value::Object(vec![])));
    }

    #[test]
    fn test_all_primitives() {
        let prims = vec![
            Value::Float(1.0),
            Value::Str("a"),
            Value::Bool(true)
        ];
        assert!(all_primitives(&prims));

        let mixed = vec![
            Value::Float(1.0),
            Value::Array(vec![]), // Compound
            Value::Str("b")
        ];
        assert!(!all_primitives(&mixed));
    }
}