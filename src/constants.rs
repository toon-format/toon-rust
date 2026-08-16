//! Constants
use crate::types::Delimiter;

/// TOON keywords that must be quoted when used as strings.
pub const KEYWORDS: &[&str] = &["null", "true", "false"];

/// Default indentation size (2 spaces).
pub const DEFAULT_INDENT: usize = 2;

/// Default delimiter (comma).
pub const DEFAULT_DELIMITER: Delimiter = Delimiter::Comma;

/// Maximum nesting depth to prevent stack overflow.
pub const MAX_DEPTH: usize = 256;

#[inline]
pub fn is_keyword(s: &str) -> bool {
    KEYWORDS.contains(&s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_keyword() {
        assert!(is_keyword("null"));
        assert!(is_keyword("true"));
        assert!(is_keyword("false"));
        assert!(!is_keyword("hello"));
        assert!(!is_keyword("TRUE"));
    }
}
