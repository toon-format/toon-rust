/* rune-xero/src/utils/validation.rs */
//!▫~•◦-------------------------------‣
//! # HPC validation for RUNE using zero-copy, SIMD-accelerated parsing.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module provides validation functions that operate on a borrow-based abstract
//! syntax tree (AST), parsed by a SIMD-accelerated engine (`simd-json`). This
//! approach avoids all heap allocations during parsing and validation, making it
//! suitable for extremely high-throughput data pipelines.
//!
//! ## Key Capabilities
//! - **Depth Validation:** Checks for excessive nesting in data structures.
//! - **Field Name Validation:** Ensures that all field names are non-empty.
//! - **Zero-Copy Value Validation:** Traverses a `simd_json::BorrowedValue` tree,
//!   which is a collection of pointers into the original input buffer, ensuring
//!   validation occurs with zero allocations and maximum performance.
//!
//! ### Architectural Notes
//! The use of `simd_json::BorrowedValue` imposes a lifetime constraint: the validated
//! object cannot outlive the input buffer from which it was parsed. This is a core
//! principle of zero-copy design and must be respected by the calling architecture.
//!
//! #### Example
//! ```rust
//! use crate::utils::validation::validate_borrowed_value;
//!
//! // The input buffer MUST be mutable for simd-json's parsing.
//! let mut json_bytes = br#"{ "user": { "id": 1, "name": "test" } }"#.to_vec();
//! let valid_value = simd_json::to_borrowed_value(&mut json_bytes).unwrap();
//! assert!(validate_borrowed_value(&valid_value).is_ok());
//!
//! let mut invalid_json_bytes = br#"{ "user": { "": 1 } }"#.to_vec();
//! let invalid_value = simd_json::to_borrowed_value(&mut invalid_json_bytes).unwrap();
//! assert!(validate_borrowed_value(&invalid_value).is_err());
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::types::{RuneError, RuneResult};
use simd_json::BorrowedValue;

/// Validate that nesting depth doesn't exceed the maximum.
pub fn validate_depth(depth: usize, max_depth: usize) -> RuneResult<()> {
    if depth > max_depth {
        // Allocation is necessary for this dynamically formatted error message.
        // This is an error path, so the performance impact is acceptable.
        return Err(RuneError::InvalidStructure(format!(
            "Maximum nesting depth of {max_depth} exceeded"
        )));
    }
    Ok(())
}

/// Validate that a field name is not empty. This function remains zero-copy.
pub fn validate_field_name(name: &str) -> RuneResult<()> {
    if name.is_empty() {
        // Allocation is dictated by the `RuneError` enum definition. This is a cold path.
        return Err(RuneError::InvalidInput(
            "Field name cannot be empty".to_string(),
        ));
    }
    Ok(())
}

/// Recursively validate a `simd_json::BorrowedValue` and all nested fields.
/// This function is truly zero-copy, as it operates on a borrowed view of the
/// original input buffer.
pub fn validate_borrowed_value(value: &BorrowedValue) -> RuneResult<()> {
    match value {
        BorrowedValue::Object(obj) => {
            for (key, val) in obj.iter() {
                validate_field_name(key)?;
                validate_borrowed_value(val)?;
            }
        }
        BorrowedValue::Array(arr) => {
            for val in arr.iter() {
                validate_borrowed_value(val)?;
            }
        }
        _ => {} // Primitives (String, Number, etc.) are views and inherently valid.
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper macro for creating a mutable Vec<u8> and parsing it for tests
    macro_rules! parse_json_borrowed {
        ($json_literal:expr) => {{
            let mut bytes = $json_literal.as_bytes().to_vec();
            simd_json::to_borrowed_value(&mut bytes)
        }};
    }

    #[test]
    fn test_validate_depth() {
        assert!(validate_depth(0, 10).is_ok());
        assert!(validate_depth(10, 10).is_ok());
        assert!(validate_depth(11, 10).is_err());
    }

    #[test]
    fn test_validate_field_name() {
        assert!(validate_field_name("name").is_ok());
        assert!(validate_field_name("").is_err());
    }

    #[test]
    fn test_validate_borrowed_value() {
        let null_val = parse_json_borrowed!("null").unwrap();
        assert!(validate_borrowed_value(&null_val).is_ok());

        let num_val = parse_json_borrowed!("123").unwrap();
        assert!(validate_borrowed_value(&num_val).is_ok());

        let str_val = parse_json_borrowed!("\"hello\"").unwrap();
        assert!(validate_borrowed_value(&str_val).is_ok());

        let good_obj = parse_json_borrowed!(r#"{ "name": "Alice" }"#).unwrap();
        assert!(validate_borrowed_value(&good_obj).is_ok());

        let good_arr = parse_json_borrowed!("[1, 2, 3]").unwrap();
        assert!(validate_borrowed_value(&good_arr).is_ok());

        let bad_obj = parse_json_borrowed!(r#"{ "": "value" }"#).unwrap();
        assert!(validate_borrowed_value(&bad_obj).is_err());

        // Test nested bad value
        let nested_bad_obj = parse_json_borrowed!(r#"{ "user": { "": "bad_key" } }"#).unwrap();
        assert!(validate_borrowed_value(&nested_bad_obj).is_err());
    }
}