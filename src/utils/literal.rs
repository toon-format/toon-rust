use crate::constants;

/// Check if a string looks like a keyword or number (needs quoting, §7.2).
pub fn is_literal_like(s: &str) -> bool {
    is_keyword(s) || is_numeric_like(s)
}

#[inline]
pub fn is_keyword(s: &str) -> bool {
    constants::is_keyword(s)
}

#[inline]
pub fn is_structural_char(ch: char) -> bool {
    constants::is_structural_char(ch)
}

/// The §7.2 numeric-like quoting trigger:
/// `/^[+-]?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?$/i` (ASCII digits only).
///
/// Unlike the decoder number grammar (§4), this covers leading-plus forms
/// and leading zeros ("+1", "05"), so such strings are emitted quoted.
pub fn is_numeric_like(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0;

    if matches!(bytes.first(), Some(b'+' | b'-')) {
        i = 1;
    }

    let int_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == int_start {
        return false;
    }

    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == frac_start {
            return false;
        }
    }

    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        i += 1;
        if i < bytes.len() && matches!(bytes[i], b'+' | b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return false;
        }
    }

    i == bytes.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_literal_like() {
        assert!(is_literal_like("null"));
        assert!(is_literal_like("true"));
        assert!(is_literal_like("false"));
        assert!(is_literal_like("123"));
        assert!(is_literal_like("-456"));
        assert!(is_literal_like("3.14"));
        assert!(!is_literal_like("hello"));
        assert!(!is_literal_like(""));
    }

    #[test]
    fn test_is_keyword() {
        assert!(is_keyword("null"));
        assert!(is_keyword("true"));
        assert!(is_keyword("false"));
        assert!(!is_keyword("TRUE"));
        assert!(!is_keyword("hello"));
    }

    #[test]
    fn test_is_numeric_like() {
        assert!(is_numeric_like("123"));
        assert!(is_numeric_like("-456"));
        assert!(is_numeric_like("0"));
        assert!(is_numeric_like("3.14"));
        assert!(is_numeric_like("1e10"));
        assert!(is_numeric_like("1.5e-3"));
        assert!(is_numeric_like("1E+9"));

        // Leading-plus forms and leading zeros are numeric-like (§7.2),
        // so encoders quote them.
        assert!(is_numeric_like("+1"));
        assert!(is_numeric_like("05"));
        assert!(is_numeric_like("01"));
        assert!(is_numeric_like("00"));

        assert!(!is_numeric_like(""));
        assert!(!is_numeric_like("-"));
        assert!(!is_numeric_like("+"));
        assert!(!is_numeric_like("abc"));
        assert!(!is_numeric_like("1."));
        assert!(!is_numeric_like(".5"));
        assert!(!is_numeric_like("1e"));
        assert!(!is_numeric_like("1_000"));
    }
}
