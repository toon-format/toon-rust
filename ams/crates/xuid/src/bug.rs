/* xuid/src/bugref.rs */
//!▫~•◦------------------------‣
//! # XUID Bug Reference (Zero-Copy)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Bug representation and linking for XUID constructs.
//!
//! This module is intentionally minimal: it defines a typed reference that can
//! be embedded in `XuidConstruct` bug segments without changing the 96-byte core.
//!
//! ## Zero-Copy Guarantees
//! - Construction from `&'a str` is **allocation-free** and **copy-free**.
//! - Deserialization is **zero-copy** when the backing input supports borrowing;
//!   otherwise serde may fall back to owned storage (format-dependent).
//!
//! ### Architectural Notes
//! This type is designed to be embedded in higher-level XUID structures while
//! keeping the core binary representation stable. It prefers reference semantics
//! (`Cow<'a, str>`) to avoid heap traffic on hot paths.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use core::borrow::Borrow;
use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// Lightweight bug/link descriptor (zero-copy).
///
/// `id` and `classifier` are stored as `Cow<'a, str>`:
/// - `Borrowed(&'a str)` for zero-copy references
/// - `Owned(String)` only when required by the input source (e.g., some deserializers)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BugRef<'a> {
    /// Human-meaningful bug identifier (opaque to the core).
    #[serde(borrow)]
    pub id: Cow<'a, str>,

    /// Optional classifier or source subsystem.
    #[serde(borrow)]
    pub classifier: Option<Cow<'a, str>>,
}

impl<'a> BugRef<'a> {
    /// Zero-copy constructor (borrows `id`).
    #[inline(always)]
    pub fn new(id: &'a str) -> Self {
        Self {
            id: Cow::Borrowed(id),
            classifier: None,
        }
    }

    /// Zero-copy classifier setter (borrows `classifier`).
    #[inline(always)]
    pub fn with_classifier(mut self, classifier: &'a str) -> Self {
        self.classifier = Some(Cow::Borrowed(classifier));
        self
    }

    /// Explicit constructor that accepts owned or borrowed parts without copying.
    #[inline(always)]
    pub fn from_parts(id: Cow<'a, str>, classifier: Option<Cow<'a, str>>) -> Self {
        Self { id, classifier }
    }

    /// Accessor: always returns a borrowed `&str` view.
    #[inline(always)]
    pub fn id(&self) -> &str {
        self.id.borrow()
    }

    /// Accessor: returns an optional borrowed `&str` view.
    #[inline(always)]
    pub fn classifier(&self) -> Option<&str> {
        self.classifier.as_deref()
    }
}
