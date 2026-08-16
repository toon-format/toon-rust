use crate::types::Number;

/// Format a number in TOON canonical form (no exponents, no trailing zeros).
pub fn format_canonical_number(n: &Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }

    if let Some(u) = n.as_u64() {
        return u.to_string();
    }

    if let Some(f) = n.as_f64() {
        return format_f64_canonical(f);
    }

    n.to_string()
}

fn format_f64_canonical(f: f64) -> String {
    // Normalize integer-valued floats to integers. The range test is the
    // exact `i64` domain: `i64::MAX as f64` rounds up to 2^63, which is one
    // past the last representable `i64`, so `<=` would let `f as i64`
    // saturate and print 9223372036854775807 for the value 2^63. `i64::MIN
    // as f64` is exactly -2^63 and stays inclusive.
    if f.is_finite() && f.fract() == 0.0 && f >= i64::MIN as f64 && f < i64::MAX as f64 {
        return format!("{}", f as i64);
    }

    // Rust's f64 Display never uses exponent notation, so the shortest
    // representation is already in plain decimal form.
    remove_trailing_zeros(&format!("{f}"))
}

fn remove_trailing_zeros(s: &str) -> String {
    if !s.contains('.') {
        // No decimal point, return as-is
        return s.to_string();
    }

    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 2 {
        return s.to_string();
    }

    let int_part = parts[0];
    let mut frac_part = parts[1].to_string();

    frac_part = frac_part.trim_end_matches('0').to_string();

    if frac_part.is_empty() {
        // All zeros removed, return as integer
        int_part.to_string()
    } else {
        format!("{int_part}.{frac_part}")
    }
}

#[cfg(test)]
mod tests {
    use std::f64;

    use serde_json::json;

    use super::*;

    #[test]
    fn test_format_canonical_integers() {
        let n = Number::from(42i64);
        assert_eq!(format_canonical_number(&n), "42");

        let n = Number::from(-123i64);
        assert_eq!(format_canonical_number(&n), "-123");

        let n = Number::from(0i64);
        assert_eq!(format_canonical_number(&n), "0");
    }

    #[test]
    fn test_format_canonical_floats() {
        // Integer-valued floats
        let n = Number::from_f64(1.0).unwrap();
        assert_eq!(format_canonical_number(&n), "1");

        let n = Number::from_f64(42.0).unwrap();
        assert_eq!(format_canonical_number(&n), "42");

        // Non-integer floats
        let n = Number::from_f64(1.5).unwrap();
        assert_eq!(format_canonical_number(&n), "1.5");

        let n = Number::from_f64(f64::consts::PI).unwrap();
        let result = format_canonical_number(&n);
        assert!(result.starts_with("3.141592653589793"));
        assert!(!result.contains('e'));
        assert!(!result.contains('E'));
    }

    #[test]
    fn test_remove_trailing_zeros() {
        assert_eq!(remove_trailing_zeros("1.5000"), "1.5");
        assert_eq!(remove_trailing_zeros("1.0"), "1");
        assert_eq!(remove_trailing_zeros("1.500"), "1.5");
        assert_eq!(remove_trailing_zeros("42"), "42");
        assert_eq!(remove_trailing_zeros("0.0"), "0");
        assert_eq!(remove_trailing_zeros("1.23"), "1.23");
    }

    #[test]
    fn test_large_numbers_no_exponent() {
        // 1e6 should become 1000000
        let n = Number::from_f64(1_000_000.0).unwrap();
        let result = format_canonical_number(&n);
        assert_eq!(result, "1000000");
        assert!(!result.contains('e'));

        // 1e9
        let n = Number::from_f64(1_000_000_000.0).unwrap();
        let result = format_canonical_number(&n);
        assert_eq!(result, "1000000000");
        assert!(!result.contains('e'));
    }

    #[test]
    fn test_small_numbers_no_exponent() {
        // 1e-6 should become 0.000001
        let n = Number::from_f64(0.000001).unwrap();
        let result = format_canonical_number(&n);
        assert!(result.starts_with("0.000001"));
        assert!(!result.contains('e'));
        assert!(!result.contains('E'));

        // 1e-3
        let n = Number::from_f64(0.001).unwrap();
        let result = format_canonical_number(&n);
        assert_eq!(result, "0.001");
    }

    #[test]
    fn test_pi_formatting() {
        let n = Number::from_f64(std::f64::consts::PI).unwrap();
        let result = format_canonical_number(&n);

        // Should not have exponent
        assert!(!result.contains('e'));
        assert!(!result.contains('E'));

        // Should start with 3.14159...
        assert!(result.starts_with("3.14159"));
    }

    #[test]
    fn test_from_json_values() {
        // Test with actual JSON values
        let val = json!(1000000);
        if let Some(n) = val.as_i64() {
            let num = Number::from(n);
            assert_eq!(format_canonical_number(&num), "1000000");
        }

        let val = json!(1.5000);
        if let Some(f) = val.as_f64() {
            let num = Number::from_f64(f).unwrap();
            assert_eq!(format_canonical_number(&num), "1.5");
        }
    }

    #[test]
    fn test_integral_f64_outside_i64_domain_is_not_saturated() {
        // 2^63 is one past i64::MAX, so a saturating `as i64` would print
        // 9223372036854775807 — a different number than the input. The value
        // is inside the u64 domain, so it prints exactly.
        let n = Number::from_f64(9223372036854775808.0).unwrap();
        assert_eq!(format_canonical_number(&n), "9223372036854775808");

        // -2^63 is exactly i64::MIN and must stay exact.
        let n = Number::from_f64(-9223372036854775808.0).unwrap();
        assert_eq!(format_canonical_number(&n), "-9223372036854775808");

        // Far outside the i64 domain: no exponent, no saturation.
        let n = Number::from_f64(1e19).unwrap();
        assert_eq!(format_canonical_number(&n), "10000000000000000000");
    }

    #[test]
    fn test_u64_above_i64_max_keeps_full_precision() {
        let n = Number::from(u64::MAX);
        assert_eq!(format_canonical_number(&n), "18446744073709551615");

        let n = Number::from(9223372036854775808u64);
        assert_eq!(format_canonical_number(&n), "9223372036854775808");
    }

    #[test]
    fn test_negative_numbers() {
        let n = Number::from_f64(-1.5).unwrap();
        assert_eq!(format_canonical_number(&n), "-1.5");

        let n = Number::from(-42i64);
        assert_eq!(format_canonical_number(&n), "-42");

        let n = Number::from_f64(-1000000.0).unwrap();
        assert_eq!(format_canonical_number(&n), "-1000000");
    }
}
