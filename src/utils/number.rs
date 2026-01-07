use itoa::Buffer as ItoaBuffer;
use ryu::Buffer as RyuBuffer;

use crate::types::Number;

/// Format a number in TOON canonical form (no exponents, no trailing zeros).
pub fn format_canonical_number(n: &Number) -> String {
    let mut out = String::new();
    write_canonical_number_into(n, &mut out);
    out
}

pub fn write_canonical_number_into(n: &Number, out: &mut String) {
    match n {
        Number::PosInt(u) => write_u64(out, *u),
        Number::NegInt(i) => write_i64(out, *i),
        Number::Float(f) => write_f64_canonical_into(*f, out),
    }
}

fn write_u64(out: &mut String, value: u64) {
    let mut buf = ItoaBuffer::new();
    out.push_str(buf.format(value));
}

fn write_i64(out: &mut String, value: i64) {
    let mut buf = ItoaBuffer::new();
    out.push_str(buf.format(value));
}

fn write_f64_canonical_into(f: f64, out: &mut String) {
    // Normalize integer-valued floats to integers
    if f.is_finite() && f.fract() == 0.0 && f.abs() <= i64::MAX as f64 {
        write_i64(out, f as i64);
        return;
    }

    if !f.is_finite() {
        out.push('0');
        return;
    }

    let mut buf = RyuBuffer::new();
    let formatted = buf.format(f);

    // Handle cases where Rust would use exponential notation
    if formatted.contains('e') || formatted.contains('E') {
        write_without_exponent(f, out);
    } else {
        push_trimmed_decimal(formatted, out);
    }
}

fn write_without_exponent(f: f64, out: &mut String) {
    if !f.is_finite() {
        out.push('0');
        return;
    }

    if f.abs() >= 1.0 {
        let abs_f = f.abs();
        let int_part = abs_f.trunc();
        let frac_part = abs_f.fract();

        if frac_part == 0.0 {
            if abs_f <= i64::MAX as f64 {
                if f < 0.0 {
                    out.push('-');
                }
                write_i64(out, int_part as i64);
            } else {
                let result = format!("{f:.0}");
                push_trimmed_decimal(&result, out);
            }
        } else {
            // High precision to avoid exponent, then trim trailing zeros
            let result = format!("{f:.17}");
            push_trimmed_decimal(&result, out);
        }
    } else if f == 0.0 {
        out.push('0');
    } else {
        // Small numbers: use high precision to avoid exponent
        let result = format!("{f:.17}");
        push_trimmed_decimal(&result, out);
    }
}

#[cfg(test)]
fn remove_trailing_zeros(s: &str) -> String {
    if let Some((int_part, frac_part)) = s.split_once('.') {
        let trimmed = frac_part.trim_end_matches('0');
        if trimmed.is_empty() {
            int_part.to_string()
        } else {
            let mut out = String::with_capacity(int_part.len() + 1 + trimmed.len());
            out.push_str(int_part);
            out.push('.');
            out.push_str(trimmed);
            out
        }
    } else {
        // No decimal point, return as-is
        s.to_string()
    }
}

fn push_trimmed_decimal(s: &str, out: &mut String) {
    if let Some((int_part, frac_part)) = s.split_once('.') {
        let trimmed = frac_part.trim_end_matches('0');
        if trimmed.is_empty() {
            out.push_str(int_part);
        } else {
            out.push_str(int_part);
            out.push('.');
            out.push_str(trimmed);
        }
    } else {
        out.push_str(s);
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
    fn test_negative_numbers() {
        let n = Number::from_f64(-1.5).unwrap();
        assert_eq!(format_canonical_number(&n), "-1.5");

        let n = Number::from(-42i64);
        assert_eq!(format_canonical_number(&n), "-42");

        let n = Number::from_f64(-1000000.0).unwrap();
        assert_eq!(format_canonical_number(&n), "-1000000");
    }
}
