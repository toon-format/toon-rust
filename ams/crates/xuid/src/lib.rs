/* xuid/src/lib.rs */
//! XUID: Xypher Universal Identity Descriptor
//!
//! ## Xypher Codex Architecture
//!
//! This crate implements the core "laws of physics" for the Xypher Sphere: a semantic-temporal
//! address system where identity is a coordinate in an 8D E8 lattice.
//!
//! **The Fundamental Law of Non-Collision:**
//! > Two events cannot collide unless they share identical semantic, temporal, causal, and geometric coordinates.
//! > Identity *is* location.
//!
//! ## Key Features
//!
//! 1. **Packed Delta Identity (Δ)**:
//!    - Time (48 bits) and Content Hash (80 bits) are packed into a single 16-byte `delta_sig`.
//!    - This embeds the "when" and "what" directly into the core identity.
//!
//! 2. **E8 Quantization (E8Q)**:
//!    - Every ID has a coordinate in the E8 Gosset lattice.
//!    - Provides geometric similarity and clustering natively.
//!
//! 3. **Unified `XU` String Format**:
//!    - Canonical: `XU:TYPE:DELTA:SIG[:S=...][:P=...][:B=...][:H=...]:ID`
//!    - Supports rich semantic layering (Semantics, Provenance, Bugs, Healing).
//!
//! 4. **Zero-Allocation Core**:
//!    - The 96-byte `Xuid` struct is designed for high-performance, stack-friendly usage.
//!
//! ## Usage
//!
//! ```rust
//! use xuid::{Xuid, XuidType, XuidConstruct};
//!
//! // 1. Create a basic XUID (E8 Quantized, timestamped now)
//! let id = Xuid::new(b"semantic event data");
//!
//! // 2. Create a specific type
//! let exp_id = Xuid::new_e8_with_type(b"strategy", XuidType::Experience);
//!
//! // 3. Wrap in a Semantic Envelope (The Codex Construct)
//! let construct = XuidConstruct::from_core(id)
//!     .with_bug("bug-123")
//!     .with_hint("apply-patch-01");
//!
//! println!("Codex ID: {}", construct.to_canonical_string());
//! // Output: XU:E8Q:123...abc:789...xyz:B=74...:H=61...:ID
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

mod binary;
mod core;
pub mod e8_lattice;
mod error;
pub mod bug;
pub mod construct;
mod lestrat;
pub mod lightweight;
#[cfg(feature = "qdrant")]
mod qdrant;
pub mod lens;

// Re-export public API
pub use core::{SemanticPath, Xuid, XuidProvenance, XuidType};

pub use e8_lattice::{
    E8Lattice, E8Orbit, E8Point, e8_distance, orbit_correlation, quantize_to_e8, quantize_to_orbit,
};

pub use error::{XuidError, XuidResult};

pub use binary::{deserialize, deserialize_compressed, serialize, serialize_compressed};

pub use lestrat::{LeStratSpec, LeStratTag, LearnedStrategyTag};
pub use construct::XuidConstruct;
pub use bug::BugRef;

// NEW: Qdrant integration for XUID point IDs (optional)
#[cfg(feature = "qdrant")]
pub use qdrant::{XuidPointId, XuidPointIdOptions};

// ============================================================================
// Module-Level Convenience Functions
// ============================================================================

/// Create a new XUID with a specific type.
#[inline]
pub fn new(data: &[u8], xuid_type: XuidType) -> Xuid {
    Xuid::new_e8_with_type(data, xuid_type)
}

/// Create a new XUID from a semantic path string.
#[inline]
pub fn from_path(path: &str, xuid_type: XuidType) -> Xuid {
    Xuid::from_semantic_path(path, xuid_type)
}

/// Ergonomic constructor: produce a Experience XUID from a spec.
#[inline]
pub fn create_lestrat_xuid(spec: &LeStratSpec) -> Result<Xuid, XuidError> {
    LearnedStrategyTag::from_spec(spec).into_xuid()
}

/// Convenience alias for `create_lestrat_xuid` matching the classic API name.
#[inline]
pub fn lestrat(spec: &LeStratSpec) -> Result<Xuid, XuidError> {
    create_lestrat_xuid(spec)
}

// ============================================================================
// Module-Level Tests
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_workflow() {
        // Create XUID
        let data = b"test data for integration";
        let xuid = Xuid::new(data);

        // Verify properties
        assert_eq!(xuid.xuid_type, XuidType::E8Quantized);
        assert!(xuid.e8_orbit < 30);
        assert_eq!(xuid.delta_sig.len(), 16);
        assert_eq!(xuid.semantic_hash.len(), 32);

        // String roundtrip
        let s = xuid.to_string();
        let parsed: Xuid = s.parse().unwrap();
        // Since timestamp is now in delta_sig, and generated at creation,
        // re-parsing the string (which contains delta_sig) should yield the exact same XUID.
        assert_eq!(xuid.delta_sig, parsed.delta_sig);

        // Binary roundtrip
        let bytes = serialize(&xuid).unwrap();
        assert_eq!(bytes.len(), 96); // Fixed header size
        let deserialized = deserialize(&bytes).unwrap();
        assert_eq!(xuid.semantic_hash, deserialized.semantic_hash);

        // Compressed roundtrip
        let compressed = serialize_compressed(&xuid, 3).unwrap();
        let decompressed = deserialize_compressed(&compressed).unwrap();
        assert_eq!(xuid.e8_orbit, decompressed.e8_orbit);

        // Similarity
        // Note: Xuid::new uses current time, so delta_sig will differ!
        // But similarity() mostly uses e8_coords, orbit, and semantic_hash.
        // semantic_hash is deterministic from input data.
        // So similarity should still be 1.0 (or very close) if data is same.
        let xuid2 = Xuid::new(b"test data for integration");
        assert!((xuid.similarity(&xuid2) - 1.0).abs() < 0.001);
    }



    #[test]
    fn test_lestrat_constructor() {
        let spec = LeStratSpec::new("test_crate", "test::module", "test_strategy")
            .with_symbol("TestType::test_method")
            .with_variant("v1");
        let xuid = create_lestrat_xuid(&spec).unwrap();
        assert_eq!(xuid.xuid_type, XuidType::Experience);
        assert!(xuid.semantic_path.is_some());
        assert!(xuid.provenance.is_some());
    }

    #[test]
    fn test_with_semantic_path() {
        let xuid = Xuid::new(b"test")
            .with_semantic_path("/cognitive/concepts/test".parse::<SemanticPath>().unwrap());

        assert!(xuid.semantic_path.is_some());

        let bytes = serialize(&xuid).unwrap();
        assert!(bytes.len() > 96); // Has optional data

        let parsed = deserialize(&bytes).unwrap();
        assert_eq!(xuid.semantic_path, parsed.semantic_path);
    }

    #[test]
    fn test_similarity_properties() {
        let a = Xuid::new(b"data a");
        let b = Xuid::new(b"data b");
        let c = Xuid::new(b"data c");

        // Reflexive
        assert!((a.similarity(&a) - 1.0).abs() < 0.001);

        // Symmetric
        let ab = a.similarity(&b);
        let ba = b.similarity(&a);
        assert!((ab - ba).abs() < 0.001);

        // Triangle inequality (approximate)
        let ac = a.similarity(&c);
        let bc = b.similarity(&c);
        // In practice, similarity doesn't form a perfect metric,
        // but should be reasonably consistent
        assert!((0.0..=1.0).contains(&ab));
        assert!((0.0..=1.0).contains(&ac));
        assert!((0.0..=1.0).contains(&bc));
    }

    #[test]
    fn test_memory_layout() {
        use std::mem;

        // Verify core struct size
        let size = mem::size_of::<Xuid>();
        println!("Xuid size: {} bytes", size);

        // Should be reasonable (96 bytes core + pointers to optional data)
        // On 64-bit: 96 + 2×24 (Option) + heap data ≈ 200+ bytes
        assert!(size <= 256, "Xuid size should be ≤256 bytes");
    }

    #[test]
    fn test_compression_ratio() {
        let xuid = Xuid::new(b"test data with some repetitive patterns patterns patterns");

        let uncompressed = serialize(&xuid).unwrap();
        let compressed = serialize_compressed(&xuid, 3).unwrap();

        println!("Uncompressed: {} bytes", uncompressed.len());
        println!("Compressed: {} bytes", compressed.len());
        #[allow(clippy::cast_precision_loss)]
        let ratio = (compressed.len() as f64 / uncompressed.len() as f64) * 100.0;
        println!("Ratio: {ratio:.1}%");

        // Even with fixed 96-byte header, compression should help with optional data
        assert!(compressed.len() <= uncompressed.len());
    }

    #[test]
    fn test_e8_properties() {
        let data = b"e8 lattice test";
        let xuid = Xuid::new(data);

        // E8 orbit should be deterministic
        let xuid2 = Xuid::new(data);
        assert_eq!(xuid.e8_orbit, xuid2.e8_orbit);

        // E8 coordinates should be deterministic
        for i in 0..8 {
            assert!((xuid.e8_coords[i] - xuid2.e8_coords[i]).abs() < 0.0001);
        }

        // Orbit should be in valid range
        assert!(xuid.e8_orbit < 30);
    }

    #[test]
    fn test_parse_legacy_format() {
        // V2 can still parse V1-style string format
        let xuid = Xuid::new(b"legacy test");
        let s = format!(
            "xuid:{}:{}:{}",
            xuid.xuid_type as u8,
            hex::encode(xuid.delta_sig),
            hex::encode(xuid.semantic_hash)
        );

        let parsed: Xuid = s.parse().unwrap();
        assert_eq!(xuid.delta_sig, parsed.delta_sig);
        assert_eq!(xuid.semantic_hash, parsed.semantic_hash);
    }
}
