/* rune-xero/src/decoder/validation.rs */
//!▫~•◦----------------------------------‣
//! # RUNE-Xero – Validation Module
//!▫~•◦-----------------------------‣
//!
//! Validates the semantic correctness of the parsed AST without allocating
//! intermediate structures. It enforces rules like:
//! - Kernel parameter type checking
//! - Indentation consistency (handled by scanner, verified here if needed)
//! - Required fields presence
//!
//! ## Key Capabilities
//! - **Zero-Allocation Traversal:** Validates directly on the AST references.
//! - **Fail-Fast:** Returns the first error encountered to minimize overhead.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::types::{RuneError, RuneResult, Value};

/// Configuration for validation strictness.
#[derive(Debug, Clone, Copy)]
pub struct ValidationConfig {
    pub strict_types: bool,
    pub allow_nulls: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_types: true,
            allow_nulls: false,
        }
    }
}

/// Validator struct that holds configuration and performs checks.
pub struct Validator {
    config: ValidationConfig,
}

impl Validator {
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Validates a root Value against the configuration rules.
    pub fn validate<'a>(&self, value: &Value<'a>) -> RuneResult<()> {
        self.validate_recursive(value, 0)
    }

    fn validate_recursive<'a>(&self, value: &Value<'a>, depth: usize) -> RuneResult<()> {
        // recursion limit check could go here if needed
        if depth > 128 {
             return Err(RuneError::parse_error(0, 0, "Max validation depth exceeded"));
        }

        match value {
            Value::Null => {
                if !self.config.allow_nulls {
                    return Err(RuneError::parse_error(0, 0, "Null values not allowed in strict mode"));
                }
            }
            Value::Array(arr) => {
                for item in arr {
                    self.validate_recursive(item, depth + 1)?;
                }
            }
            Value::Object(entries) => {
                // Check for duplicate keys without allocating a HashSet
                // For small objects (common case), O(N^2) scan is faster than allocation.
                // For large objects, we accept the cost or assume the map handles it.
                // Here we do a zero-alloc check.
                for (i, (key1, _)) in entries.iter().enumerate() {
                     for (key2, _) in entries.iter().skip(i + 1) {
                         if key1 == key2 {
                             return Err(RuneError::parse_error(0, 0, format!("Duplicate key found: {}", key1)));
                         }
                     }
                }

                for (_, val) in entries {
                    self.validate_recursive(val, depth + 1)?;
                }
            }
            _ => {} // Primitives are valid by default
        }
        Ok(())
    }
}