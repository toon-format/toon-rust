use crate::types::{
    ToonError,
    ToonResult,
};

/// Validate that nesting depth doesn't exceed the maximum.
pub fn validate_depth(depth: usize, max_depth: usize) -> ToonResult<()> {
    if depth > max_depth {
        return Err(ToonError::InvalidStructure(format!(
            "Maximum nesting depth of {max_depth} exceeded"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_depth() {
        assert!(validate_depth(0, 10).is_ok());
        assert!(validate_depth(5, 10).is_ok());
        assert!(validate_depth(10, 10).is_ok());
        assert!(validate_depth(11, 10).is_err());
    }
}
