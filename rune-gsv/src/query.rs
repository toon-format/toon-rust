/* src/query/query.rs */
//!▫~•◦-------------------------------‣
//! # Semantic Query Utilities for head-based similarity matching.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-gsv to achieve efficient graph-based data storage and querying system.
//!
//! ### Key Capabilities
//! - **Head Matching:** Provides utility functions for counting matching heads between addresses and slots.
//! - **Similarity Analysis:** Enables semantic similarity comparison using Weyl address heads.
//! - **Query Support:** Offers foundational functions for similarity-based retrieval operations.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `addressing` and `slot`.
//! Query functions adhere to the semantic addressing pattern and are compatible
//! with the system's similarity search pipeline.
//!
//! ### Example
//! ```rust
//! use crate::rune_gsv::{count_head_matches, WeylSemanticAddress, SGLRuneSlot};
//!
//! let address = WeylSemanticAddress::from_text_intent("query").unwrap();
//! let slot = SGLRuneSlot::default();
//! let matches = count_head_matches(&address, &slot);
//!
//! // The match count can be used for similarity ranking.
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::addressing::WeylSemanticAddress;
use crate::slot::SGLRuneSlot;

pub fn count_head_matches(a: &WeylSemanticAddress, b: &SGLRuneSlot) -> usize {
    let bh: [u8; 8] = b.address.clone().try_into().unwrap_or([0u8; 8]);
    let wb = WeylSemanticAddress {
        heads: bh,
        digest: [0u8; 32],
        context: None,
    };
    a.matches(&wb)
}
