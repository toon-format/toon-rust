//! Decoder layout metadata.
//!
//! [`Layout`] captures structural information about how a TOON document was
//! encoded on the wire — information that is otherwise lost once decoded into
//! a [`serde_json::Value`].
//!
//! Obtain a `Layout` via [`crate::decode::decode_with_layout`].
//!
//! Available only when the `layout` cargo feature is enabled.
//!
//! **Experimental.** This module supports independent exploration of schema
//! and tooling use cases (validators, formatters, linters). It is not part
//! of the TOON specification and its API may evolve independently of the
//! core decoder.
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "layout")]
//! # fn main() -> Result<(), toon_format::ToonError> {
//! use toon_format::{
//!     decode_with_layout,
//!     DecodeOptions,
//! };
//!
//! let toon = "users[2]{id,name}:\n  1,Alice\n  2,Bob";
//! let (_value, layout) = decode_with_layout(toon, &DecodeOptions::default())?;
//!
//! assert!(matches!(
//!     layout.get("/users"),
//!     Some(toon_format::NodeLayout::Tabular { .. })
//! ));
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "layout"))]
//! # fn main() {}
//! ```

use std::collections::BTreeMap;

use crate::types::Delimiter;

/// Layout metadata for a decoded TOON document, keyed by JSON Pointer
/// (RFC 6901). The empty pointer `""` refers to the document root. Only
/// nodes with layout-relevant information are present; primitive scalars
/// are not recorded.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Layout {
    nodes: BTreeMap<String, NodeLayout>,
}

impl Layout {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up the layout at a JSON Pointer. Use `""` for the document root.
    pub fn get(&self, json_pointer: &str) -> Option<&NodeLayout> {
        self.nodes.get(json_pointer)
    }

    /// Iterate `(json_pointer, node_layout)` pairs sorted by pointer.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &NodeLayout)> {
        self.nodes.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub(crate) fn insert(&mut self, json_pointer: String, node: NodeLayout) {
        self.nodes.insert(json_pointer, node);
    }
}

/// Per-node layout describing how a value was written in the source TOON.
///
/// Only array-shaped nodes are recorded in this release. Object key order
/// and key-folding metadata are planned for a follow-up.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum NodeLayout {
    /// Tabular array: `key[N]{f1,f2,...}:` with rows on subsequent lines.
    Tabular {
        declared_len: usize,
        fields: Vec<FieldDescriptor>,
        delimiter: Delimiter,
    },

    /// List-form array: `key[N]:` followed by `- item` lines.
    List { declared_len: usize },

    /// Inline primitive array: `key[N]: a,b,c` on a single line.
    InlineArray {
        declared_len: usize,
        delimiter: Delimiter,
    },
}

/// A field declared in a tabular array header.
///
/// `nested` is reserved for forward compatibility with proposals that allow
/// tabular fields to contain nested object schemas (e.g. spec RFC #46). For
/// TOON v3.0 it is always `None`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDescriptor {
    pub name: String,
    pub nested: Option<Box<NodeLayout>>,
}

impl FieldDescriptor {
    pub fn leaf(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nested: None,
        }
    }
}
