/* xuid/src/error.rs */
//! XUID Error Types
//!
/// Comprehensive error taxonomy replacing generic `anyerror`.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣
use yoshi_derive::AnyError;

/// XUID operation errors
#[derive(AnyError, Debug, Clone, PartialEq, Eq)]
pub enum XuidError {
    /// Invalid XUID string format
    #[anyerror("Invalid XUID format: {0}")]
    InvalidFormat(String),

    /// Parse error in XUID string
    #[anyerror("Parse error at position {pos}: {msg}")]
    ParseError { pos: usize, msg: String },

    /// Invalid XUID type discriminant
    #[anyerror("Invalid XUID type: {0}")]
    InvalidType(u8),

    /// Hex decoding error
    #[anyerror("Invalid hex encoding: {0}")]
    InvalidHex(String),

    /// Compression error
    #[anyerror("Compression failed: {0}")]
    CompressionError(String),

    /// Decompression error
    #[anyerror("Decompression failed: {0}")]
    DecompressionError(String),

    /// Binary format error
    #[anyerror("Invalid binary format: {0}")]
    BinaryFormatError(String),

    /// Unsupported version
    #[anyerror("Unsupported XUID version: {0}")]
    UnsupportedVersion(u8),

    /// E8 quantization error
    #[anyerror("E8 quantization failed: {0}")]
    E8Error(String),

    /// Cryptographic operation error
    #[anyerror("Cryptographic operation failed: {0}")]
    CryptoError(String),

    /// Registry error
    #[anyerror("Registry operation failed: {0}")]
    RegistryError(String),

    /// I/O error
    #[anyerror("I/O error: {0}")]
    IoError(String),

    /// Data too large
    #[anyerror("Data exceeds maximum size: {size} > {max}")]
    DataTooLarge { size: usize, max: usize },

    /// Invalid semantic path
    #[anyerror("Invalid semantic path: {0}")]
    InvalidSemanticPath(String),
}

impl From<std::io::Error> for XuidError {
    fn from(e: std::io::Error) -> Self {
        XuidError::IoError(e.to_string())
    }
}

impl From<hex::FromHexError> for XuidError {
    fn from(e: hex::FromHexError) -> Self {
        XuidError::InvalidHex(e.to_string())
    }
}

/// Result type for XUID operations
pub type XuidResult<T> = Result<T, XuidError>;
