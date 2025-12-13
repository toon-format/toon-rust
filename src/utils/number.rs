/* rune-xero/src/utils/number.rs */
//!▫~•◦-----------------------------‣
//! # RUNE-Xero – Number Formatting
//!▫~•◦-----------------------------‣
//!
//! Utilities for canonical number formatting.
//! Optimized to use `Cow` and avoid intermediate string allocations.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::types::Number;
use std::borrow::Cow;

/// Format a number in RUNE canonical form (no exponents, no trailing zeros).
/// Returns `Cow` to avoid allocation if the number is already stored as Raw string.
pub fn format_canonical_number<'a>(n: &'a Number<'a>) -> Cow<'a, str> {
    match n {
        Number::Raw(s) => s.clone(), // Cheap Cow clone
        Number::PosInt(u) => Cow::Owned(u.to_string()),
        Number::NegInt(i) => Cow::Owned(i.to_string()),
        Number::Float(f) => Cow::Owned(format_f64_canonical(*f)),
    }
}

fn format_f64_canonical(f: f64) -> String {
    if !f.is_finite() {
        return "0".to_string();
    }

    // Integer check
    if f.fract() == 0.0 && f.abs() <= i64::MAX as f64 {
        return (f as i64).to_string();
    }

    // Default format might use scientific notation
    let s = format!("{}", f);
    if s.contains('e') || s.contains('E') {
        // Fallback for large/small numbers to ensure decimal notation
        format_without_exponent(f)
    } else {
        // Standard formatting usually fine, just trim zeros
        trim_trailing_zeros_owned(s)
    }
}

fn format_without_exponent(f: f64) -> String {
    // Force decimal formatting
    let s = format!("{:.17}", f); // High precision
    trim_trailing_zeros_owned(s)
}

fn trim_trailing_zeros_owned(mut s: String) -> String {
    if !s.contains('.') { return s; }
    
    let trimmed_len = {
        let trimmed = s.trim_end_matches('0');
        let trimmed = trimmed.trim_end_matches('.');
        trimmed.len()
    };
    
    s.truncate(trimmed_len);
    s
}

#[cfg(test)]
mod tests {
    use std::f64;
    use super::*;

    #[test]
    fn test_format_canonical_integers() {
        let n = Number::from(42i64);
        assert_eq!(format_canonical_number(&n), "42");
    }

    #[test]
    fn test_format_canonical_floats() {
        let n = Number::from(1.5f64);
        assert_eq!(format_canonical_number(&n), "1.5");
        
        let n = Number::from(f64::consts::PI);
        let s = format_canonical_number(&n);
        assert!(s.starts_with("3.14"));
        assert!(!s.contains('e'));
    }
}