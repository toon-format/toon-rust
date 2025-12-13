/* rune-xero/src/utils/san.rs */
//!▫~•◦--------------------------‣
//! # UTF-8 Sanitizer Utility for handling invalid byte sequences.
//!▫~•◦-----------------------------------------------------------‣
//!
//! This module provides robust UTF-8 validation and sanitization capabilities,
//! ensuring text files contain valid UTF-8 encodings by replacing invalid
//! sequences with the Unicode replacement character (U+FFFD).
//!
//! ### Key Capabilities
//! - **Lossless Valid Content Preservation:** All valid UTF-8 sequences remain unchanged.
//! - **Invalid Sequence Replacement:** Non-compliant bytes are replaced with � (U+FFFD).
//! - **Detailed Diagnostics:** Reports the number of invalid sequences encountered.
//!
//! ### Architectural Notes
//! This module operates as a standalone utility and CLI tool. It can be integrated
//! into larger pipelines requiring UTF-8 compliance enforcement. The sanitization
//! leverages Rust's standard library `String::from_utf8_lossy` for deterministic,
//! safe conversion.
//!
//! ### Example
//! ```rust
//! use utf8_sanitizer::sanitize_utf8;
//!
//! let raw_bytes = b"Hello\xFFWorld";
//! let (sanitized, invalid_count) = sanitize_utf8(raw_bytes);
//!
//! assert_eq!(invalid_count, 1);
//! assert!(sanitized.contains('\u{FFFD}'));
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::fs;
use std::io;

/// Sanitizes UTF-8 content by replacing invalid bytes with the Unicode replacement character (�).
///
/// This function processes raw byte sequences and produces a valid UTF-8 string. Invalid
/// sequences are replaced with U+FFFD (�), and the function reports the total count of
/// such replacements.
///
/// # Arguments
/// * `input` - A byte slice potentially containing invalid UTF-8 sequences.
///
/// # Returns
/// A tuple containing:
/// - The sanitized UTF-8 string
/// - The count of invalid sequences replaced
///
/// # Examples
/// ```rust
/// use utf8_sanitizer::sanitize_utf8;
///
/// let valid = b"Hello, world!";
/// let (result, count) = sanitize_utf8(valid);
/// assert_eq!(count, 0);
/// assert_eq!(result, "Hello, world!");
///
/// let invalid = b"Hello\xFFWorld";
/// let (result, count) = sanitize_utf8(invalid);
/// assert_eq!(count, 1);
/// assert!(result.contains('\u{FFFD}'));
/// ```
pub fn sanitize_utf8(input: &[u8]) -> (String, usize) {
    let s = String::from_utf8_lossy(input);

    let invalid_count = match s {
        std::borrow::Cow::Borrowed(_) => 0,
        std::borrow::Cow::Owned(ref content) => {
            content.chars().filter(|&c| c == '\u{FFFD}').count()
        }
    };

    (s.into_owned(), invalid_count)
}

/// Main entry point for the UTF-8 sanitization CLI tool.
///
/// Reads the target file as raw bytes, sanitizes invalid UTF-8 sequences,
/// and writes the sanitized content back to the original file. Provides
/// diagnostic output indicating the number of invalid sequences replaced.
///
/// # Errors
/// Returns `io::Error` if file read or write operations fail.
fn main() -> io::Result<()> {
    let file_path = "src/rune/.runeFiles/rune.txt";

    let bytes = fs::read(file_path)?;

    println!("Read {} bytes from {}", bytes.len(), file_path);

    let (sanitized, invalid_count) = sanitize_utf8(&bytes);

    println!("UTF-8 sanitization completed:");
    println!("- Invalid sequences replaced: {}", invalid_count);
    println!(
        "- Total characters in output: {}",
        sanitized.chars().count()
    );

    fs::write(file_path, sanitized)?;

    if invalid_count > 0 {
        println!(
            "Warning: {} invalid UTF-8 sequences were replaced with �",
            invalid_count
        );
    } else {
        println!("✓ File was already valid UTF-8 - no changes made");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_utf8() {
        let input = "Hello, 世界!".as_bytes();
        let (result, count) = sanitize_utf8(input);
        assert_eq!(count, 0);
        assert_eq!(result, "Hello, 世界!");
    }

    #[test]
    fn test_invalid_utf8() {
        let invalid_bytes = b"Hello\xFFWorld";
        let (result, count) = sanitize_utf8(invalid_bytes);
        assert_eq!(count, 1);
        assert!(result.contains('\u{FFFD}'));
    }

    #[test]
    fn test_multiple_invalid_sequences() {
        let invalid_bytes = b"Test\xFF\xFE\xFDData";
        let (result, count) = sanitize_utf8(invalid_bytes);
        assert_eq!(count, 3);
        assert_eq!(result.chars().filter(|&c| c == '\u{FFFD}').count(), 3);
    }

    #[test]
    fn test_empty_input() {
        let (result, count) = sanitize_utf8(&[]);
        assert_eq!(count, 0);
        assert_eq!(result, "");
    }

    #[test]
    fn test_all_invalid() {
        let invalid_bytes = b"\xFF\xFE\xFD";
        let (result, count) = sanitize_utf8(invalid_bytes);
        assert_eq!(count, 3);
        assert_eq!(result.len(), '\u{FFFD}'.len_utf8() * 3);
    }

    #[test]
    fn test_mixed_valid_invalid() {
        let mixed = b"Valid\xFFText\xFE\xFDHere";
        let (result, count) = sanitize_utf8(mixed);
        assert_eq!(count, 3);
        assert!(result.starts_with("Valid"));
        assert!(result.contains("Text"));
        assert!(result.ends_with("Here"));
    }
}
