/* src/addressing/addressing.rs */
//!▫~•◦-------------------------------‣
//! # Weyl Semantic Addressing System for deterministic intent-based addressing.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-gsv to achieve efficient graph-based data storage and querying system.
//!
//! ### Key Capabilities
//! - **Deterministic Address Generation:** Creates stable 8-head addresses from textual intents using SHA-256 hashing.
//! - **Coordinate-based Addressing:** Supports creation of addresses from numeric coordinates with base-240 mapping.
//! - **Similarity Matching:** Provides head-matching functionality for semantic similarity analysis.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `slot` and `store`.
//! Address structures adhere to the semantic addressing pipeline and are compatible
//! with the system's serialization and persistence layers.
//!
//! ### Example
//! ```rust
//! use crate::rune_gsv::{WeylSemanticAddress};
//!
//! let address = WeylSemanticAddress::from_text_intent("find similar documents").unwrap();
//! let key = address.to_key();
//! let other_address = WeylSemanticAddress::from_text_intent("related docs").unwrap();
//! let matches = address.matches(&other_address);
//!
//! // The 'address' can now be used for storage and retrieval operations.
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use yoshi::error::Result;

/// WeylSemanticAddress: deterministic 8-head address derived from SHA-256 of intent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeylSemanticAddress {
    pub heads: [u8; 8],
    pub digest: [u8; 32],
    pub context: Option<String>,
}

impl WeylSemanticAddress {
    /// Create a Weyl address deterministically from a textual intent.
    pub fn from_text_intent(intent: &str) -> Result<Self> {
        let normalized = intent.trim().to_lowercase();
        let mut hasher = Sha256::new();
        hasher.update(normalized.as_bytes());
        let digest = hasher.finalize();
        let mut d32 = [0u8; 32];
        d32.copy_from_slice(&digest[..32]);
        let mut heads = [0u8; 8];
        for i in 0..8 {
            heads[i] = (d32[i] % 240) as u8; // base-240 mapping
        }
        Ok(Self {
            heads,
            digest: d32,
            context: None,
        })
    }

    /// Create address from numeric coordinates (E8 vector or similar), optional rounding.
    pub fn from_coords(coords: &[f32; 8]) -> Result<Self> {
        // simple deterministic mapping: scale and quantize
        let mut d32 = [0u8; 32];
        // fill digest by copying little-endian f32 bytes
        for (i, f) in coords.iter().enumerate().take(8) {
            let bytes = f.to_le_bytes();
            d32[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }
        // fallback to sha of the bytes for stability
        let mut hasher = Sha256::new();
        hasher.update(&d32);
        let h = hasher.finalize();
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&h[..32]);
        let mut heads = [0u8; 8];
        for i in 0..8 {
            heads[i] = (digest[i] % 240) as u8;
        }
        Ok(Self {
            heads,
            digest,
            context: None,
        })
    }

    /// Stable string key for storage
    pub fn to_key(&self) -> String {
        self.heads
            .iter()
            .map(|h| h.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Count matching heads with another address
    pub fn matches(&self, other: &Self) -> usize {
        self.heads
            .iter()
            .zip(other.heads.iter())
            .filter(|(a, b)| a == b)
            .count()
    }
}
