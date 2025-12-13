//! XuidLens: Zero-Cost Inspection & Semantic Dictionary
//!
//! # Overview
//!
//! `XuidLens` provides "on-the-fly" inspection of `XuidConstruct` strings and binary data
//! without requiring full allocation or parsing. It is designed for "blazing fast ops"
//! where you need to check a timestamp, type, or semantic tag in a hot loop.
//!
//! Additionally, it defines a `LensDictionary` interface for precomputed lookups, allowing
//! opaque hashes (like `S=...` or `P=...`) to be resolved to their full semantic meaning
//! instantly if they are registered.
//!
//! # Usage
//!
//! ```rust
//! use xuid::lens::XuidView;
//!
//! let xu_str = "XU:E8Q:1234abcd:5678efgh:S=1234:ID";
//! let view = XuidView::new(xu_str).unwrap();
//!
//! assert_eq!(view.xuid_type(), "E8Q");
//! assert_eq!(view.delta(), "1234abcd");
//! // Zero-allocation access
//! ```

use crate::error::XuidError;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// A zero-copy view into a canonical XUID string.
///
/// It holds a reference to the string slice and computes offsets lazily or eagerly
/// (implementation detail) to allow O(1) access to components.
#[derive(Debug, Clone, Copy)]
pub struct XuidView<'a> {
    raw: &'a str,
    // Store slice ranges or just split iterator references?
    // For "blazing fast", we might want to store indices if we parse once,
    // or just provide methods that slice on demand if the format is strict.
    // Given XU format is colon-delimited, let's store the splits.
    parts: [&'a str; 12], // Increased to 12 to safely hold XU, TYPE, DELTA, SIG, S, P, B, H, ID + slack
    len: usize,
}

impl<'a> XuidView<'a> {
    /// Create a new view from a string slice.
    ///
    /// This performs a shallow pass to identify segment boundaries.
    pub fn new(s: &'a str) -> Result<Self, XuidError> {
        if !s.starts_with("XU:") {
            return Err(XuidError::InvalidFormat("Missing XU prefix".into()));
        }

        let mut parts = [""; 12];
        let mut count = 0;
        for (i, part) in s.split(':').enumerate() {
            if i >= parts.len() {
                // If we exceed our static buffer, we might miss tail segments.
                break;
            }
            parts[i] = part;
            count = i + 1;
        }

        if count < 4 {
            return Err(XuidError::InvalidFormat("Too few segments".into()));
        }

        Ok(Self {
            raw: s,
            parts,
            len: count,
        })
    }

    /// Get the raw underlying string.
    pub fn as_str(&self) -> &'a str {
        self.raw
    }

    /// Get the XUID type string (e.g. "E8Q").
    pub fn xuid_type(&self) -> &'a str {
        self.parts[1]
    }

    /// Get the Delta Hex string.
    pub fn delta(&self) -> &'a str {
        self.parts[2]
    }

    /// Get the Signature Hex string.
    pub fn signature(&self) -> &'a str {
        self.parts[3]
    }

    /// Extract the timestamp from the Delta Hex (if standard packing used).
    /// Returns None if parsing fails.
    pub fn timestamp_ms(&self) -> Option<u64> {
        let delta_hex = self.delta();
        if delta_hex.len() != 32 { return None; }
        // First 12 hex chars = 6 bytes = 48 bits
        let ts_hex = &delta_hex[0..12];
        u64::from_str_radix(ts_hex, 16).ok()
    }

    /// Get semantic segment (S=...) if present.
    pub fn semantic_segment(&self) -> Option<&'a str> {
        self.find_segment("S=")
    }

    /// Get provenance segment (P=...) if present.
    pub fn provenance_segment(&self) -> Option<&'a str> {
        self.find_segment("P=")
    }

    /// Get bug reference (B=...) if present.
    pub fn bug_ref(&self) -> Option<&'a str> {
        self.find_segment("B=")
    }

    /// Get healing hint (H=...) if present.
    pub fn healing_hint(&self) -> Option<&'a str> {
        self.find_segment("H=")
    }

    /// Get the tail ID.
    pub fn tail_id(&self) -> Option<&'a str> {
        // ID is the last segment that isn't a known tag key
        if self.len > 0 {
            let last = self.parts[self.len - 1];
            if !last.contains('=') {
                return Some(last);
            }
        }
        None
    }

    fn find_segment(&self, prefix: &str) -> Option<&'a str> {
        // Skip XU, TYPE, DELTA, SIG (indices 0..3)
        for i in 4..self.len {
            if self.parts[i].starts_with(prefix) {
                return Some(&self.parts[i][prefix.len()..]);
            }
        }
        None
    }
}

// ============================================================================
// Dictionary / Registry
// ============================================================================

/// A thread-safe dictionary for resolving semantic hashes to content.
///
/// This enables the "Lens" functionality: looking up opaque identifiers (like
/// hashed semantic paths) to see their full meaning instantly.
pub trait LensDictionary: Send + Sync {
    /// Resolve a hash (hex string) to its semantic content.
    fn resolve(&self, hash_hex: &str) -> Option<String>;

    /// Register a mapping.
    fn register(&self, hash_hex: String, content: String);
}

/// A standard in-memory dictionary using RwLock.
#[derive(Debug, Default)]
pub struct InMemoryDictionary {
    map: RwLock<HashMap<String, String>>,
}

impl InMemoryDictionary {
    pub fn new() -> Self {
        Self::default()
    }
}

impl LensDictionary for InMemoryDictionary {
    fn resolve(&self, hash_hex: &str) -> Option<String> {
        let r = self.map.read().ok()?;
        r.get(hash_hex).cloned()
    }

    fn register(&self, hash_hex: String, content: String) {
        if let Ok(mut w) = self.map.write() {
            w.insert(hash_hex, content);
        }
    }
}

/// Global singleton for the default lens dictionary.
///
/// Usage: `xuid::lens::register(...)` or `xuid::lens::resolve(...)`.
static GLOBAL_LENS: std::sync::OnceLock<Arc<dyn LensDictionary>> = std::sync::OnceLock::new();

fn get_global_lens() -> &'static Arc<dyn LensDictionary> {
    GLOBAL_LENS.get_or_init(|| Arc::new(InMemoryDictionary::new()))
}

/// Resolve a hash using the global dictionary.
pub fn resolve(hash_hex: &str) -> Option<String> {
    get_global_lens().resolve(hash_hex)
}

/// Register a mapping in the global dictionary.
pub fn register(hash_hex: impl Into<String>, content: impl Into<String>) {
    get_global_lens().register(hash_hex.into(), content.into());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xuid_view() {
        let raw = "XU:E8Q:0000019515abcdeffedcba9876543210:abba...:S=cafe:ID123";
        let view = XuidView::new(raw).expect("Valid format");

        assert_eq!(view.xuid_type(), "E8Q");
        assert_eq!(view.delta(), "0000019515abcdeffedcba9876543210");
        assert_eq!(view.signature(), "abba...");
        assert_eq!(view.semantic_segment(), Some("cafe"));
        assert_eq!(view.tail_id(), Some("ID123"));
        assert_eq!(view.bug_ref(), None);
    }

    #[test]
    fn test_timestamp_extraction() {
        // Timestamp: 1700000000000 ms
        // python3 -c "print(hex(1700000000000))" => 0x18bcfe56800
        // We used to test manual padding, but now we should align with what
        // Xuid::pack_delta would produce if it were hex encoded.
        // pack_delta layout is [u32 time | u16 node | u8 hash...]
        // We haven't updated pack_delta yet, so this test is testing the *old* logic
        // which assumed 48-bit timestamp.
        // Let's just fix the test to match the current (old) logic until we update it.
        // Old logic: first 12 hex chars = 48 bits.
        // 0x18bcfe56800 is 44 bits.
        // Padded to 48 bits (12 hex chars): 018bcfe56800
        // (The leading zero was in the wrong place in previous attempt)

        let delta = "018bcfe56800ffffffffffffffffffff";
        let raw = format!("XU:E8Q:{}:sig", delta);
        let view = XuidView::new(&raw).unwrap();

        assert_eq!(view.timestamp_ms(), Some(1700000000000));
    }

    #[test]
    fn test_dictionary() {
        let hash = "deadbeef";
        let content = "The quick brown fox";

        // Ensure not found initially
        assert_eq!(resolve(hash), None);

        // Register
        register(hash, content);

        // Resolve
        assert_eq!(resolve(hash), Some(content.to_string()));
    }
}
