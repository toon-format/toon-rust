use crate::constants;

/// Returns true when a string looks like a keyword or number (needs quoting).
///
/// # Examples
/// ```
/// use toon_format::utils::literal::is_literal_like;
///
/// assert!(is_literal_like("null"));
/// assert!(is_literal_like("123"));
/// assert!(!is_literal_like("hello"));
/// ```
pub fn is_literal_like(s: &str) -> bool {
    is_keyword(s) || is_numeric_like(s)
}

#[inline]
/// Returns true when the string matches a reserved TOON keyword.
///
/// # Examples
/// ```
/// use toon_format::utils::literal::is_keyword;
///
/// assert!(is_keyword("true"));
/// assert!(!is_keyword("TRUE"));
/// ```
pub fn is_keyword(s: &str) -> bool {
    constants::is_keyword(s)
}

#[inline]
/// Returns true when the character has structural meaning in TOON.
///
/// # Examples
/// ```
/// use toon_format::utils::literal::is_structural_char;
///
/// assert!(is_structural_char('['));
/// assert!(!is_structural_char('a'));
/// ```
pub fn is_structural_char(ch: char) -> bool {
    constants::is_structural_char(ch)
}

/// Returns true when the string looks like a number (starts with digit, no leading zeros).
///
/// # Examples
/// ```
/// use toon_format::utils::literal::is_numeric_like;
///
/// assert!(is_numeric_like("3.14"));
/// assert!(!is_numeric_like("01"));
/// ```
pub fn is_numeric_like(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return false;
    }

    let mut i = 0;
    if bytes[0] == b'-' {
        i = 1;
    }

    if i >= bytes.len() {
        return false;
    }

    let first = bytes[i];
    if !first.is_ascii_digit() {
        return false;
    }

    if first == b'0' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
        return false;
    }

    bytes[i..].iter().all(|b| {
        b.is_ascii_digit() || *b == b'.' || *b == b'e' || *b == b'E' || *b == b'+' || *b == b'-'
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
