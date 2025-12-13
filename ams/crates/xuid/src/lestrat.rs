/* src/lestrat.rs */
//! LeStrat (= Experience) XUID helpers: canonicalization & constructors.
//!
//! # XUID LeStrat – Experience Module
//!▫~•◦------------------------------------------------‣
//!
//! This module standardizes how strategy-oriented identities are created.
//! Given a semantic location (e.g., `crate::module::Type::method`) and a strategy
//! description/fingerprint, it produces a stable `Xuid` with type
//! `XuidType::Experience`, a canonical semantic path, and rich provenance.
//!
//! ### Canonical Path
//! `/strategy/local/{source_crate}/{module_path}/{strategy_name}[/{symbol}][/{variant}]`
//!
//! ### Examples
//! ```rust
//! use xuid::{LeStratSpec, LeStratTag, XuidType};
//! let spec = LeStratSpec::new("my_crate", "agnite::planner", "beam_search")
//!     .with_symbol("Planner::decide")
//!     .with_variant("v2")
//!     .with_version("1.4.0");
//! let tag = LeStratTag::from_spec(&spec);
//! let xuid = tag.into_xuid().unwrap();
//! assert_eq!(xuid.xuid_type, XuidType::Experience);
//! ```
//!
//! ```rust
//! use xuid::{LeStratSpec, Xuid};
//! let spec = LeStratSpec::new("crate", "module::sub", "strategy");
//! let xuid = Xuid::new_lestrat(&spec);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::core::{SemanticPath, Xuid, XuidProvenance, XuidType};
use crate::e8_lattice::quantize_to_orbit;
use crate::error::XuidError;
use blake3::Hasher;
use std::collections::BTreeMap;

/// Specification used to derive a canonical Experience (LeStrat) identity.
#[derive(Debug, Clone)]
pub struct LeStratSpec {
    pub source_crate: String,
    pub module_path: String,      // e.g., "agnite::planner"
    pub strategy_name: String,    // e.g., "beam_search"
    pub symbol: Option<String>,   // e.g., "Planner::decide"
    pub variant: Option<String>,  // e.g., "v2" or "cuda"
    pub version: Option<String>,  // semantic version or commit-ish
    pub salt_opt: Option<String>, // optional: to disambiguate experiments
}

/// Short aliases (preferred modern names).
/// LeStratSpec is the main struct name (no alias needed)
impl LeStratSpec {
    pub fn new(
        source_crate: impl Into<String>,
        module_path: impl Into<String>,
        strategy_name: impl Into<String>,
    ) -> Self {
        Self {
            source_crate: source_crate.into(),
            module_path: module_path.into(),
            strategy_name: strategy_name.into(),
            symbol: None,
            variant: None,
            version: None,
            salt_opt: None,
        }
    }
    pub fn with_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = Some(symbol.into());
        self
    }
    pub fn with_variant(mut self, variant: impl Into<String>) -> Self {
        self.variant = Some(variant.into());
        self
    }
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }
    pub fn with_salt(mut self, salt: impl Into<String>) -> Self {
        self.salt_opt = Some(salt.into());
        self
    }
}

/// Canonicalized path + finalized data to mint a strategy XUID.
#[derive(Debug, Clone)]
pub struct LeStratTag {
    pub canonical_path: String,
    pub semantic_bytes: [u8; 32], // BLAKE3 of canonical tuple (stable)
}

/// Short alias (preferred modern name).
pub type LearnedStrategyTag = LeStratTag;

impl LeStratTag {
    /// Create from a spec by canonicalizing and hashing the identity tuple.
    pub fn from_spec(spec: &LeStratSpec) -> Self {
        // Canonical path: /strategy/local/{crate}/{module}/{strategy}[/{symbol}][/{variant}]
        let mut path = format!(
            "/strategy/local/{}/{}/{}",
            sanitize(&spec.source_crate),
            sanitize(&spec.module_path),
            sanitize(&spec.strategy_name),
        );
        if let Some(sym) = &spec.symbol {
            path.push('/');
            path.push_str(&sanitize(sym));
        }
        if let Some(var) = &spec.variant {
            path.push('/');
            path.push_str(&sanitize(var));
        }

        // Canonical tuple (location + strategy semantics + optional metadata)
        let mut hasher = Hasher::new();
        hasher.update(path.as_bytes());
        if let Some(ver) = &spec.version {
            hasher.update(b"|ver:");
            hasher.update(ver.as_bytes());
        }
        if let Some(salt) = &spec.salt_opt {
            hasher.update(b"|salt:");
            hasher.update(salt.as_bytes());
        }
        let semantic = hasher.finalize();

        let mut semantic_bytes = [0u8; 32];
        semantic_bytes.copy_from_slice(semantic.as_bytes());
        Self {
            canonical_path: path,
            semantic_bytes,
        }
    }

    /// Convert this tag to a fully formed XUID (Experience type).
    pub fn into_xuid(self) -> Result<Xuid, XuidError> {
        // Reuse E8 quantization on the semantic bytes for geometric placement.
        let (e8_orbit, _e8_coords) = quantize_to_orbit(&self.semantic_bytes)?;
        
        let semantic_path = SemanticPath::new(
            self.canonical_path
                .split('/')
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect(),
        );

        let tick_cs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64 / 10; // Convert milliseconds to centiseconds

        // NodeID defaults to 0, Lane defaults to 0, and Epoch defaults to 0 for Experience XUIDs
        let node_id = 0;
        let lane = 0;
        let epoch = 0;

        let mut xuid = Xuid::create(
            &self.semantic_bytes,
            XuidType::Experience,
            tick_cs,
            node_id,
            lane,
            epoch,
        );

        // Explicitly set semantic path and provenance
        xuid.semantic_path = Some(semantic_path);
        
        // Panic-free timestamp acquisition (epoch fallback = 0s)
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let mut metadata = BTreeMap::new();
        metadata.insert("canonical_path".into(), self.canonical_path.clone());
        metadata.insert("subtype".into(), "learned_strategy".into());
        metadata.insert("hash_algo".into(), "blake3".into());

        xuid.provenance = Some(XuidProvenance {
            source: "xuid::lestrat".into(),
            timestamp: Some(ts),
            metadata,
        });

        xuid.e8_orbit = e8_orbit; // Ensure E8 orbit from quantization is used
        xuid.e8_coords = _e8_coords; // Ensure E8 coords from quantization is used

        Ok(xuid)
    }
}

/// Ergonomic constructor on `Xuid`.
impl Xuid {
    /// Build a Experience / LeStrat XUID directly from a spec.
    #[inline]
    pub fn new_lestrat(spec: &LeStratSpec) -> Result<Self, XuidError> {
        LeStratTag::from_spec(spec).into_xuid()
    }
}

fn sanitize(s: &str) -> String {
    // Keep it filesystem/URL friendly and deterministic
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '_' | '-' | '.' | ':' => out.push(ch),
            ';' | ' ' => out.push('_'),
            '/' | '\\' => out.push_str("::"),
            _ => out.push('_'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_identity_for_same_tuple() {
        let spec = LeStratSpec::new("my_crate", "agnite::planner", "beam_search")
            .with_symbol("Planner::decide")
            .with_variant("v2")
            .with_version("1.0.0");
        let t1 = LeStratTag::from_spec(&spec);
        let t2 = LeStratTag::from_spec(&spec);
        assert_eq!(t1.semantic_bytes, t2.semantic_bytes);
        assert_eq!(t1.canonical_path, t2.canonical_path);
        let x1 = t1.into_xuid().unwrap();
        let x2 = t2.into_xuid().unwrap();
        assert_eq!(x1.delta_sig, x2.delta_sig);
        assert_eq!(x1.semantic_hash, x2.semantic_hash);
        assert_eq!(x1.xuid_type, XuidType::Experience);
    }

    #[test]
    fn canonical_path_formatting() {
        let spec = LeStratSpec::new("test_crate", "module::sub", "strategy")
            .with_symbol("Type::method")
            .with_variant("cuda");
        let tag = LeStratTag::from_spec(&spec);
        assert!(
            tag.canonical_path
                .starts_with("/strategy/local/test_crate/module::sub/strategy/Type::method/cuda")
        );
    }

    #[test]
    fn xuid_type_is_experience() {
        let spec = LeStratSpec::new("crate", "mod", "strat");
        let xuid = LeStratTag::from_spec(&spec).into_xuid().unwrap();
        assert_eq!(xuid.xuid_type, XuidType::Experience);
    }

    #[test]
    fn aliases_behave_identically() {
        let a = LeStratSpec::new("c", "m::p", "s").with_variant("v");
        let b = LeStratSpec::new("c", "m::p", "s").with_variant("v");
        let ta = LeStratTag::from_spec(&a);
        let tb = LeStratTag::from_spec(&b);
        assert_eq!(ta.canonical_path, tb.canonical_path);
        assert_eq!(ta.semantic_bytes, tb.semantic_bytes);
    }

    #[test]
    fn ctor_matches_tag_into_xuid() {
        let spec = LeStratSpec::new("cr", "mod::path", "name")
            .with_symbol("T::m")
            .with_version("2.0.0");
        let via_tag = LeStratTag::from_spec(&spec).into_xuid().unwrap();
        let via_ctor = Xuid::new_lestrat(&spec).unwrap();
        assert_eq!(via_tag.xuid_type, via_ctor.xuid_type);
        assert_eq!(via_tag.delta_sig, via_ctor.delta_sig);
        assert_eq!(via_tag.semantic_hash, via_ctor.semantic_hash);
    }
}
