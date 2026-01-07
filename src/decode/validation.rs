use crate::types::{ToonError, ToonResult};
use std::collections::HashSet;

/// Validate that an array length matches the header value.
///
/// # Examples
/// ```
/// use toon_format::decode::validation::validate_array_length;
///
/// assert!(validate_array_length(2, 2).is_ok());
/// assert!(validate_array_length(2, 3).is_err());
/// ```
pub fn validate_array_length(expected: usize, actual: usize) -> ToonResult<()> {
    if expected != actual {
        return Err(ToonError::length_mismatch(expected, actual));
    }
    Ok(())
}

/// Validate that an array length is non-negative.
///
/// # Examples
/// ```
/// use toon_format::decode::validation::validate_array_length_non_negative;
///
/// assert!(validate_array_length_non_negative(0).is_ok());
/// assert!(validate_array_length_non_negative(-1).is_err());
/// ```
pub fn validate_array_length_non_negative(length: i64) -> ToonResult<()> {
    if length < 0 {
        return Err(ToonError::InvalidInput(
            "Array length must be non-negative".to_string(),
        ));
    }
    Ok(())
}

/// Validate that a tabular field list has unique entries.
///
/// # Examples
/// ```
/// use toon_format::decode::validation::validate_field_list;
///
/// let fields = vec!["a".to_string(), "b".to_string()];
/// assert!(validate_field_list(&fields).is_ok());
/// ```
pub fn validate_field_list(fields: &[String]) -> ToonResult<()> {
    if fields.is_empty() {
        return Err(ToonError::InvalidInput(
            "Field list cannot be empty for tabular arrays".to_string(),
        ));
    }

    let mut seen = HashSet::with_capacity(fields.len());
    for field in fields {
        if field.is_empty() {
            return Err(ToonError::InvalidInput(
                "Field name cannot be empty".to_string(),
            ));
        }
        if !seen.insert(field.as_str()) {
            return Err(ToonError::InvalidInput(format!(
                "Duplicate field name: '{field}'"
            )));
        }
    }

    Ok(())
}

/// Validate that a tabular row length matches the header.
///
/// # Examples
/// ```
/// use toon_format::decode::validation::validate_row_length;
///
/// assert!(validate_row_length(1, 2, 2).is_ok());
/// assert!(validate_row_length(1, 2, 1).is_err());
/// ```
pub fn validate_row_length(
    row_index: usize,
    expected_fields: usize,
    actual_values: usize,
) -> ToonResult<()> {
    if expected_fields != actual_values {
        return Err(ToonError::InvalidStructure(format!(
            "Row {row_index} has {actual_values} values but expected {expected_fields} fields"
        )));
    }
    Ok(())
}

/// Validate that the row delimiter matches the header.
///
/// # Examples
/// ```
/// use toon_format::decode::validation::validate_delimiter_consistency;
/// use toon_format::Delimiter;
///
/// assert!(
///     validate_delimiter_consistency(Some(Delimiter::Comma), Some(Delimiter::Comma)).is_ok()
/// );
/// ```
pub fn validate_delimiter_consistency(
    detected: Option<crate::types::Delimiter>,
    expected: Option<crate::types::Delimiter>,
) -> ToonResult<()> {
    if let (Some(detected), Some(expected)) = (detected, expected) {
        if detected != expected {
            return Err(ToonError::InvalidDelimiter(format!(
                "Detected delimiter {detected} but expected {expected}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_array_length() {
        assert!(validate_array_length(5, 3).is_err());
        assert!(validate_array_length(3, 5).is_err());
        assert!(validate_array_length(5, 5).is_ok());
    }

    #[test]
    fn test_validate_array_length_non_negative() {
        assert!(validate_array_length_non_negative(0).is_ok());
        assert!(validate_array_length_non_negative(5).is_ok());
        assert!(validate_array_length_non_negative(-1).is_err());
    }

    #[test]
    fn test_validate_field_list() {
        assert!(validate_field_list(&["id".to_string(), "name".to_string()]).is_ok());
        assert!(validate_field_list(&["field1".to_string()]).is_ok());

        assert!(validate_field_list(&[]).is_err());

        assert!(
            validate_field_list(&["id".to_string(), "name".to_string(), "id".to_string()]).is_err()
        );

        assert!(
            validate_field_list(&["id".to_string(), "".to_string(), "name".to_string()]).is_err()
        );
    }

    #[test]
    fn test_validate_row_length() {
        assert!(validate_row_length(0, 3, 3).is_ok());
        assert!(validate_row_length(1, 5, 5).is_ok());

        assert!(validate_row_length(0, 3, 2).is_err());
        assert!(validate_row_length(1, 3, 4).is_err());
    }

    #[test]
    fn test_validate_delimiter_consistency() {
        use crate::types::Delimiter;

        assert!(
            validate_delimiter_consistency(Some(Delimiter::Comma), Some(Delimiter::Comma)).is_ok()
        );

        assert!(
            validate_delimiter_consistency(Some(Delimiter::Comma), Some(Delimiter::Pipe)).is_err()
        );

        assert!(validate_delimiter_consistency(None, Some(Delimiter::Comma)).is_ok());
        assert!(validate_delimiter_consistency(Some(Delimiter::Comma), None).is_ok());
        assert!(validate_delimiter_consistency(None, None).is_ok());
    }
}
