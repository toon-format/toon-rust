#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeyFoldingMode {
    /// No folding performed. All objects use standard nesting.
    #[default]
    Off,
    /// Fold eligible chains according to safety rules.
    Safe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathExpansionMode {
    /// Dotted keys are treated as literal keys. No expansion.
    #[default]
    Off,
    /// Expand eligible dotted keys according to safety rules.
    Safe,
}

/// Check if a key segment is a valid IdentifierSegment (stricter than unquoted keys).
pub fn is_identifier_segment(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let mut chars = s.chars();

    // First character must be letter or underscore
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };

    if !first.is_alphabetic() && first != '_' {
        return false;
    }

    // Remaining characters: letters, digits, or underscore (NO dots)
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_folding_mode_default() {
        assert_eq!(KeyFoldingMode::default(), KeyFoldingMode::Off);
    }

    #[test]
    fn test_path_expansion_mode_default() {
        assert_eq!(PathExpansionMode::default(), PathExpansionMode::Off);
    }

    #[test]
    fn test_is_identifier_segment() {
        // Valid segments
        assert!(is_identifier_segment("a"));
        assert!(is_identifier_segment("_private"));
        assert!(is_identifier_segment("userName"));
        assert!(is_identifier_segment("user_name"));
        assert!(is_identifier_segment("user123"));
        assert!(is_identifier_segment("_123"));

        // Invalid segments
        assert!(!is_identifier_segment(""));
        assert!(!is_identifier_segment("123"));
        assert!(!is_identifier_segment("user-name"));
        assert!(!is_identifier_segment("user.name")); // Contains dot
        assert!(!is_identifier_segment("user name")); // Contains space
        assert!(!is_identifier_segment("user:name")); // Contains colon
        assert!(!is_identifier_segment(".name")); // Starts with dot
    }

    #[test]
    fn test_identifier_segment_vs_general_key() {
        // These are valid unquoted keys but NOT IdentifierSegments
        assert!(!is_identifier_segment("a.b")); // Contains dot
        assert!(!is_identifier_segment("a.b.c")); // Contains dots

        // These are valid for both
        assert!(is_identifier_segment("abc"));
        assert!(is_identifier_segment("_private"));
        assert!(is_identifier_segment("key123"));
    }
}
