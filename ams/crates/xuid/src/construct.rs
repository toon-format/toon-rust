/* xuid/src/construct.rs */
//!▫~•◦---------------------------‣
//! XuidConstruct: The Semantic Envelope of the Xypher Codex
//!▫~•◦------------------------------------------------------‣
//!
//! This module defines the `XuidConstruct`, which wraps the core 96-byte `Xuid` identity
//! with the rich semantic envelope required by the Xypher Sphere.
//!
//! # The Unified XU Format
//!
//! `XU:TYPE:DELTA:SIG[:S=...][:P=...][:B=...][:H=...]:ID`
//!
//! - **TYPE**: The `XuidType` (e.g., `E8Q`, `EXP`).
//! - **DELTA**: Packed 16-byte delta signature (Time + Hash).
//! - **SIG**: 32-byte semantic hash.
//! - **S**: Semantic Context (e.g., Semantic Path).
//! - **P**: Provenance (Source/Origin).
//! - **B**: Bug Reference (if any).
//! - **H**: Healing/Hint Strategy (if any).
//! - **ID**: Literal suffix.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::{SemanticPath, Xuid, XuidError, XuidProvenance, XuidType};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fmt;
use std::str::FromStr;

/// High-level construct wrapper binding the core XUID to semantic envelope data.
///
/// Zero-copy intent:
/// - This type does **not** eagerly allocate semantic/provenance strings from `core`.
/// - Optional overrides can be borrowed (`Cow::Borrowed`) at construction time.
/// - Rendering avoids intermediate `Vec<String>`/`hex::encode(String)` allocations by
///   writing directly into a single output `String`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XuidConstruct<'a> {
	/// Canonical 96-byte identity (source of truth).
	pub core: Xuid,

	/// Semantic Context segment override (S). If `None`, render falls back to `core.semantic_path`.
	#[serde(borrow)]
	pub semantics: Option<Cow<'a, str>>,

	/// Provenance segment override (P). If `None`, render falls back to `core.provenance.source`.
	#[serde(borrow)]
	pub provenance: Option<Cow<'a, str>>,

	/// Bug reference segment (B).
	#[serde(borrow)]
	pub bug_ref: Option<Cow<'a, str>>,

	/// Healing/Hint segment (H).
	#[serde(borrow)]
	pub healing_hint: Option<Cow<'a, str>>,
}

impl<'a> XuidConstruct<'a> {
	/// Create a construct from a core XUID.
	///
	/// Zero-copy: does not allocate semantic/provenance strings; those are rendered
	/// from `core` unless explicitly overridden.
	pub fn from_core(core: Xuid) -> Self {
		Self {
			core,
			semantics: None,
			provenance: None,
			bug_ref: None,
			healing_hint: None,
		}
	}

	/// Add a bug reference (B) without allocating.
	pub fn with_bug(mut self, bug: &'a str) -> Self {
		self.bug_ref = Some(Cow::Borrowed(bug));
		self
	}

	/// Add a healing hint (H) without allocating.
	pub fn with_hint(mut self, hint: &'a str) -> Self {
		self.healing_hint = Some(Cow::Borrowed(hint));
		self
	}

	/// Add explicit semantic context override (S) without allocating.
	pub fn with_semantics(mut self, s: &'a str) -> Self {
		self.semantics = Some(Cow::Borrowed(s));
		self
	}

	/// Add explicit provenance context override (P) without allocating.
	pub fn with_provenance_str(mut self, p: &'a str) -> Self {
		self.provenance = Some(Cow::Borrowed(p));
		self
	}

	/// Render the canonical string format.
	///
	/// Note: producing the final `String` necessarily allocates once for the output buffer.
	/// Everything else is written directly into that buffer (no intermediate `Vec<String>`,
	/// no intermediate `hex::encode(String)` allocations).
	pub fn to_canonical_string(&self) -> String {
		let mut out = String::with_capacity(self.estimate_canonical_len());
		self.write_canonical_into(&mut out);
		out
	}

	#[inline]
	fn estimate_canonical_len(&self) -> usize {
		// Baseline:
		// "XU:" + TYPE + ":" + delta(32) + ":" + sig(64) + ":ID"
		let mut n = 3 + 3 + 1 + 32 + 1 + 64 + 3;

		// Optional segments add:
		// ":" + "X=" + hex(payload)
		// where hex(payload) = 2 * payload_len
		if let Some(s) = self.semantics.as_deref() {
			n += 3 + (2 * s.len()) + 1; // ":S=" + hex + (conservative +1)
		} else if let Some(path) = &self.core.semantic_path {
			let s = path.to_string();
			n += 3 + (2 * s.len()) + 1;
		}

		if let Some(p) = self.provenance.as_deref() {
			n += 3 + (2 * p.len()) + 1;
		} else if let Some(prov) = &self.core.provenance {
			n += 3 + (2 * prov.source.len()) + 1;
		}

		if let Some(b) = self.bug_ref.as_deref() {
			n += 3 + (2 * b.len()) + 1;
		}
		if let Some(h) = self.healing_hint.as_deref() {
			n += 3 + (2 * h.len()) + 1;
		}

		n
	}

	fn write_canonical_into(&self, out: &mut String) {
		// Prefix
		out.push_str("XU:");
		out.push_str(type_to_str(self.core.xuid_type));
		out.push(':');

		// delta_sig (16 bytes -> 32 hex chars)
		append_hex_bytes(out, &self.core.delta_sig);
		out.push(':');

		// semantic_hash (32 bytes -> 64 hex chars)
		append_hex_bytes(out, &self.core.semantic_hash);

		// Optional segments (S, P, B, H) encoded as hex of UTF-8 bytes.
		if let Some(s) = self.semantics.as_deref() {
			out.push_str(":S=");
			append_hex_str(out, s);
		} else if let Some(path) = &self.core.semantic_path {
			out.push_str(":S=");
			// Avoid allocating an intermediate hex string; we do have to materialize the path
			// as UTF-8 bytes via `to_string()`, because SemanticPath is not stored as &str.
			let tmp = path.to_string();
			append_hex_str(out, &tmp);
		}

		if let Some(p) = self.provenance.as_deref() {
			out.push_str(":P=");
			append_hex_str(out, p);
		} else if let Some(prov) = &self.core.provenance {
			out.push_str(":P=");
			append_hex_str(out, &prov.source);
		}

		if let Some(b) = self.bug_ref.as_deref() {
			out.push_str(":B=");
			append_hex_str(out, b);
		}

		if let Some(h) = self.healing_hint.as_deref() {
			out.push_str(":H=");
			append_hex_str(out, h);
		}

		// Literal ID suffix
		out.push_str(":ID");
	}
}

impl fmt::Display for XuidConstruct<'_> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(&self.to_canonical_string())
	}
}

impl FromStr for XuidConstruct<'static> {
	type Err = XuidError;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		let mut iter = s.split(':');

		let prefix = iter
			.next()
			.ok_or_else(|| XuidError::InvalidFormat("empty".into()))?;
		if prefix != "XU" {
			return Err(XuidError::InvalidFormat("missing XU prefix".into()));
		}

		let type_str = iter
			.next()
			.ok_or_else(|| XuidError::InvalidFormat("missing type".into()))?;
		let delta_hex = iter
			.next()
			.ok_or_else(|| XuidError::InvalidFormat("missing delta".into()))?;
		let sig_hex = iter
			.next()
			.ok_or_else(|| XuidError::InvalidFormat("missing signature".into()))?;

		let mut semantics: Option<Cow<'static, str>> = None;
		let mut provenance: Option<Cow<'static, str>> = None;
		let mut bug_ref: Option<Cow<'static, str>> = None;
		let mut healing_hint: Option<Cow<'static, str>> = None;

		for segment in iter {
			if segment == "ID" {
				break;
			}

			if let Some(val) = segment.strip_prefix("S=") {
				let decoded = hex::decode(val)
					.map_err(|_| XuidError::InvalidFormat("invalid hex in S".into()))?;
				let s = String::from_utf8(decoded)
					.map_err(|_| XuidError::InvalidFormat("invalid utf8 in S".into()))?;
				semantics = Some(Cow::Owned(s));
			} else if let Some(val) = segment.strip_prefix("P=") {
				let decoded = hex::decode(val)
					.map_err(|_| XuidError::InvalidFormat("invalid hex in P".into()))?;
				let s = String::from_utf8(decoded)
					.map_err(|_| XuidError::InvalidFormat("invalid utf8 in P".into()))?;
				provenance = Some(Cow::Owned(s));
			} else if let Some(val) = segment.strip_prefix("B=") {
				let decoded = hex::decode(val)
					.map_err(|_| XuidError::InvalidFormat("invalid hex in B".into()))?;
				let s = String::from_utf8(decoded)
					.map_err(|_| XuidError::InvalidFormat("invalid utf8 in B".into()))?;
				bug_ref = Some(Cow::Owned(s));
			} else if let Some(val) = segment.strip_prefix("H=") {
				let decoded = hex::decode(val)
					.map_err(|_| XuidError::InvalidFormat("invalid hex in H".into()))?;
				let s = String::from_utf8(decoded)
					.map_err(|_| XuidError::InvalidFormat("invalid utf8 in H".into()))?;
				healing_hint = Some(Cow::Owned(s));
			}
		}

		// Reconstruct core
		let xuid_type = type_from_str(type_str)?;
		let mut core = Xuid::new_e8_with_type(&[0u8; 0], xuid_type);

		let delta_bytes = hex::decode(delta_hex)
			.map_err(|_| XuidError::InvalidFormat("delta hex".into()))?;
		if delta_bytes.len() == 16 {
			core.delta_sig.copy_from_slice(&delta_bytes);
		} else {
			return Err(XuidError::InvalidFormat("delta length".into()));
		}

		let sig_bytes = hex::decode(sig_hex)
			.map_err(|_| XuidError::InvalidFormat("sig hex".into()))?;
		if sig_bytes.len() == 32 {
			core.semantic_hash.copy_from_slice(&sig_bytes);
		} else {
			return Err(XuidError::InvalidFormat("sig length".into()));
		}

		// Best-effort: if S parses as a SemanticPath, store it on core.
		if let Some(s) = semantics.as_deref() {
			if let Ok(path) = SemanticPath::from_str(s) {
				core.semantic_path = Some(path);
			}
		}

		// Best-effort: attach provenance to core without extra clones where possible.
		if let Some(p) = provenance.as_deref() {
			core.provenance = Some(XuidProvenance {
				source: p.to_owned(),
				timestamp: None,
				metadata: std::collections::BTreeMap::new(),
			});
		}

		Ok(Self {
			core,
			semantics,
			provenance,
			bug_ref,
			healing_hint,
		})
	}
}

// ----------------------------------------------------------------------------
// Hex helpers (no intermediate allocations)
// ----------------------------------------------------------------------------

#[inline(always)]
fn append_hex_bytes(out: &mut String, bytes: &[u8]) {
	for &b in bytes {
		push_hex_byte(out, b);
	}
}

#[inline(always)]
fn append_hex_str(out: &mut String, s: &str) {
	append_hex_bytes(out, s.as_bytes());
}

#[inline(always)]
fn push_hex_byte(out: &mut String, b: u8) {
	const LUT: &[u8; 16] = b"0123456789abcdef";
	out.push(LUT[(b >> 4) as usize] as char);
	out.push(LUT[(b & 0x0F) as usize] as char);
}

// Duplicated helpers to avoid cyclic deps or exposure issues (internal to crate logic)
fn type_to_str(t: XuidType) -> &'static str {
	match t {
		XuidType::E8Quantized => "E8Q",
		XuidType::Experience => "EXP",
		XuidType::Anomaly => "ERR",
		XuidType::Healing => "HEL",
		XuidType::Codex => "CDX",
	}
}

fn type_from_str(s: &str) -> Result<XuidType, XuidError> {
	match s {
		"E8Q" | "E8Quantized" => Ok(XuidType::E8Quantized),
		"EXP" | "Experience" => Ok(XuidType::Experience),
		"ERR" | "Anomaly" => Ok(XuidType::Anomaly),
		"HEL" | "Healing" => Ok(XuidType::Healing),
		"CDX" | "Codex" => Ok(XuidType::Codex),
		_ => Err(XuidError::InvalidFormat(format!("unknown type {s}"))),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_xuid_construct_canonical() {
		let xuid = Xuid::new(b"codex-event");
		let construct = XuidConstruct::from_core(xuid.clone())
			.with_bug("bug-null-deref")
			.with_hint("apply-patch-01");

		let s = construct.to_canonical_string();
		println!("Canonical: {}", s);

		assert!(s.starts_with("XU:E8Q:"));
		assert!(s.contains("B="));
		assert!(s.contains("H="));
		assert!(s.ends_with(":ID"));

		let parsed: XuidConstruct = s.parse().unwrap();
		assert_eq!(parsed.core.delta_sig, xuid.delta_sig);
		assert_eq!(parsed.bug_ref.unwrap(), "bug-null-deref");
		assert_eq!(parsed.healing_hint.unwrap(), "apply-patch-01");
	}
}
