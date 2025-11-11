//! Constants
use crate::types::Delimiter;

/// Characters that have structural meaning in TOON format.
pub const STRUCTURAL_CHARS: &[char] = &['[', ']', '{', '}', ':', '-'];

/// TOON keywords that must be quoted when used as strings.
pub const KEYWORDS: &[&str] = &["null", "true", "false"];

/// Default indentation size (2 spaces).
pub const DEFAULT_INDENT: usize = 2;

/// Default delimiter (comma).
pub const DEFAULT_DELIMITER: Delimiter = Delimiter::Comma;

/// Maximum nesting depth to prevent stack overflow.
pub const MAX_DEPTH: usize = 256;

/// Internal marker prefix for quoted keys containing dots.
/// Used during path expansion to distinguish quoted keys (which should remain
/// literal) from unquoted keys (which may be expanded).
/// This marker is added during parsing and removed during expansion.
pub(crate) const QUOTED_KEY_MARKER: char = '\x00';

#[inline]
pub fn is_structural_char(ch: char) -> bool {
    STRUCTURAL_CHARS.contains(&ch)
}

#[inline]
pub fn is_keyword(s: &str) -> bool {
    KEYWORDS.contains(&s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_structural_char() {
        assert!(is_structural_char('['));
        assert!(is_structural_char(']'));
        assert!(is_structural_char('{'));
        assert!(is_structural_char('}'));
        assert!(is_structural_char(':'));
        assert!(is_structural_char('-'));
        assert!(!is_structural_char('a'));
        assert!(!is_structural_char(','));
    }

    #[test]
    fn test_is_keyword() {
        assert!(is_keyword("null"));
        assert!(is_keyword("true"));
        assert!(is_keyword("false"));
        assert!(!is_keyword("hello"));
        assert!(!is_keyword("TRUE"));
    }
}
