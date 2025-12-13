/* src/codex.rs */
//!▫~•◦-------------------------------‣
//! # Codex Core Structures
//!▫~•◦-------------------------------------------------------------------‣
//! Defines the fundamental types for the Xypher Codex hierarchy.

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Tier {
    Taproot = 0,
    Lateral = 1,
    Tertiary = 2,
    Cross = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Domain(pub u8);

impl Domain {
    pub const COUNT: usize = 8;
    pub const SIZE: usize = 30;

    pub fn from_root_id(root_id: u16) -> Self {
        Self((root_id as usize / Self::SIZE) as u8)
    }

    pub fn base_index(&self) -> u16 {
        self.0 as u16 * Self::SIZE as u16
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexRoot {
    pub id: u16,
    pub vector: [f32; 8],
    pub tier: Tier,
    pub domain: Domain,
    // Bitmask for allowed operator families (Spec C Requirement)
    pub family_mask: u32,
}

impl CodexRoot {
    pub fn new(id: u16, vector: [f32; 8], tier: Tier, family_mask: u32) -> Self {
        Self {
            id,
            vector,
            tier,
            domain: Domain::from_root_id(id),
            family_mask,
        }
    }
}
