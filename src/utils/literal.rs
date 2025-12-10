use crate::constants;

/// Check if a string looks like a keyword or number (needs quoting).
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

/// Check if a string looks like a number (starts with digit, no leading zeros).
pub fn is_numeric_like(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    if chars[i] == '-' {
        i += 1;
    }

    if i >= chars.len() {
        return false;
    }

    if !chars[i].is_ascii_digit() {
        return false;
    }

    if chars[i] == '0' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
        return false;
    }

    chars[i..].iter().all(|c| {
        c.is_ascii_digit() || *c == '.' || *c == 'e' || *c == 'E' || *c == '+' || *c == '-'
    })
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
    fn test_is_structural_char() {
        assert!(is_structural_char('['));
        assert!(is_structural_char('{'));
        assert!(is_structural_char(':'));
        assert!(!is_structural_char('a'));
    }

    #[test]
    fn test_is_numeric_like() {
        assert!(is_numeric_like("123"));
        assert!(is_numeric_like("-456"));
        assert!(is_numeric_like("0"));
        assert!(is_numeric_like("3.14"));
        assert!(is_numeric_like("1e10"));
        assert!(is_numeric_like("1.5e-3"));

        assert!(!is_numeric_like(""));
        assert!(!is_numeric_like("-"));
        assert!(!is_numeric_like("abc"));
        assert!(!is_numeric_like("01"));
        assert!(!is_numeric_like("00"));
    }
}
