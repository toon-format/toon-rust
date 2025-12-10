use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
        Ok(Self { heads, digest: d32, context: None })
    }

    /// Create address from numeric coordinates (E8 vector or similar), optional rounding.
    pub fn from_coords(coords: &[f32; 8]) -> Result<Self> {
        // simple deterministic mapping: scale and quantize
        let mut d32 = [0u8; 32];
        // fill digest by copying little-endian f32 bytes
        for (i, f) in coords.iter().enumerate().take(8) {
            let bytes = f.to_le_bytes();
            d32[i*4..i*4+4].copy_from_slice(&bytes);
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
        Ok(Self { heads, digest, context: None })
    }

    /// Stable string key for storage
    pub fn to_key(&self) -> String {
        self.heads.iter().map(|h| h.to_string()).collect::<Vec<_>>().join(",")
    }

    /// Count matching heads with another address
    pub fn matches(&self, other: &Self) -> usize {
        self.heads.iter().zip(other.heads.iter()).filter(|(a,b)| a==b).count()
    }
}
