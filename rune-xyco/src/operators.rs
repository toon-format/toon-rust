/* src/operators.rs */
//!▫~•◦-------------------------------‣
//! # Semantic Operators (Tokenless)
//!▫~•◦-------------------------------------------------------------------‣
//! Implements Spec B: Grammar-as-Meaning Operators.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// The 7 Semantic Operator Families (OpKind).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum OpKind {
    ENT = 0, // Entity / Referent
    PRD = 1, // Predicate / Event / State
    MOD = 2, // Modifier
    REL = 3, // Relation / Attachment
    CMP = 4, // Composition / Glue
    AFF = 5, // Affect / Discourse Signal
    GRD = 6, // Grounding / Quantification
}

/// A tokenless semantic operator instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operator {
    pub id: u64,
    pub kind: OpKind,
    pub root_idx: Option<u16>, // Resolved Codex Root Index
    pub params: HashMap<String, String>, // Deterministic parameters
    pub binds: Vec<u64>, // Adjacency (other operator IDs)
}

impl Operator {
    pub fn new(id: u64, kind: OpKind) -> Self {
        Self {
            id,
            kind,
            root_idx: None,
            params: HashMap::new(),
            binds: Vec::new(),
        }
    }

    pub fn with_param(mut self, key: &str, value: &str) -> Self {
        self.params.insert(key.to_string(), value.to_string());
        self
    }

    pub fn bind(mut self, target_id: u64) -> Self {
        self.binds.push(target_id);
        self
    }
}
