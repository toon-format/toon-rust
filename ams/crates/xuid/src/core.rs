/* xuid/src/core.rs */
//! Core XUID Implementation
//!
//! XUID: Xypher Universal Identifier
//!
//! # The Semantic Spacetime Coordinate System
//!
//! An XUID is not just an ID; it is a coordinate in an 8D semantic-temporal space (the Xypher Sphere).
//! It is composed of:
//! - **E8Q**: E8 Quantization (Orbit + Coordinates) representing the "where" in semantic space.
//! - **Δ (Delta)**: Packed 48-bit Timestamp + 80-bit Hash representing the "when" and "what".
//! - **S (Semantics)**: Semantic Hash + Path representing the "meaning".
//! - **P (Provenance)**: Origin/Source representing the "causality".
//! - **B (Bug)**: Bug tracking (optional).
//! - **H (Healing)**: Recovery strategy (optional).
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣
use super::e8_lattice::{E8Orbit, E8Point, e8_distance, orbit_correlation, quantize_to_orbit};
use super::error::{XuidError, XuidResult};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Core Types
// ============================================================================

/// XUID type discriminant
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum XuidType {
    /// E8 lattice quantized (default)
    E8Quantized = 0,
    /// Learned strategy identity
    Experience = 1,
    /// Anomaly/Bug detected
    Anomaly = 2,
    /// Healing/Recovery strategy
    Healing = 3,
    /// Xypher Codex defined XUID
    Codex = 6,
}

impl XuidType {
    pub fn from_u8(v: u8) -> XuidResult<Self> {
        match v {
            0 => Ok(Self::E8Quantized),
            1 => Ok(Self::Experience),
            2 => Ok(Self::Anomaly),
            3 => Ok(Self::Healing),
            6 => Ok(Self::Codex),
            _ => Err(XuidError::InvalidType(v)),
        }
    }
}

/// Semantic path component
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SemanticPath {
    pub components: Vec<String>,
}

impl SemanticPath {
    pub fn new(components: Vec<String>) -> Self {
        Self { components }
    }
}

impl FromStr for SemanticPath {
    type Err = XuidError;

    fn from_str(path: &str) -> Result<Self, Self::Err> {
        Ok(Self {
            components: path
                .split('/')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect(),
        })
    }
}

impl fmt::Display for SemanticPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "/{}", self.components.join("/"))
    }
}

/// XUID provenance metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XuidProvenance {
    pub source: String,
    pub timestamp: Option<i64>,
    pub metadata: std::collections::BTreeMap<String, String>,
}

// ============================================================================
// Main XUID Structure (96 bytes, zero allocations for core)
// ============================================================================

/// Xypher Universal Identifier (V2)
///
/// **Memory Layout** (80 bytes total):
/// ```text
/// [0-0]   XuidType (1 byte)
/// [1-1]   E8 Orbit (1 byte)
/// [2-6]   Delta Signature - Tick (40 bits, centiseconds since epoch)
/// [7-8]   Delta Signature - NodeID (16 bits, Codex root_id)
/// [9-9]   Delta Signature - Lane (8 bits)
/// [10-10] Delta Signature - Epoch (8 bits)
/// [11-42] Semantic Hash (32 bytes)
/// [43-74] E8 Coordinates (8×f32 = 32 bytes)
/// [75-79] Reserved/Padding (5 bytes)
/// ```
#[repr(C, align(16))]
#[derive(Debug, Clone)]
pub struct Xuid {
    /// XUID type discriminant
    pub xuid_type: XuidType,

    /// E8 orbit classification (0-29)
    pub e8_orbit: E8Orbit,

    /// Delta signature: Packed 40-bit Tick + 16-bit NodeID + 8-bit Lane + 8-bit Epoch.
    /// Stored as 16 bytes for binary compatibility (7 padding bytes).
    pub delta_sig: [u8; 16],

    /// Semantic hash (32-byte BLAKE3)
    pub semantic_hash: [u8; 32],

    /// E8 lattice coordinates
    pub e8_coords: E8Point,

    /// Optional semantic path (heap-allocated only when set)
    pub semantic_path: Option<SemanticPath>,

    /// Optional provenance (heap-allocated only when set)
    pub provenance: Option<XuidProvenance>,
}

// ============================================================================
// PartialEq and Eq Implementation
// ============================================================================

impl PartialEq for Xuid {
    fn eq(&self, other: &Self) -> bool {
        self.delta_sig == other.delta_sig
            && self.semantic_hash == other.semantic_hash
            && self.xuid_type == other.xuid_type
            && self.e8_orbit == other.e8_orbit
            && self.e8_coords.iter().zip(other.e8_coords.iter()).all(|(a, b)| a.to_bits() == b.to_bits())
            && self.semantic_path == other.semantic_path
            && self.provenance == other.provenance
    }
}
impl Eq for Xuid {}

impl std::hash::Hash for Xuid {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.xuid_type.hash(state);
        self.e8_orbit.hash(state);
        self.delta_sig.hash(state);
        self.semantic_hash.hash(state);
    }
}

// ============================================================================
// Packed Delta Implementation
// ============================================================================

impl Xuid {
    /// Pack Tick (centiseconds), NodeID, Lane, and Epoch into the delta signature.
    ///
    /// Layout:
    /// - Bytes 0..4 (40 bits): Tick (centiseconds since Unix epoch, Little Endian)
    /// - Bytes 5..6 (16 bits): NodeID (Codex root_id, Little Endian)
    /// - Byte 7 (8 bits): Lane
    /// - Byte 8 (8 bits): Epoch
    /// - Bytes 9..15 (56 bits): Reserved (zero)
    pub fn pack_delta(tick_cs: u64, node_id: u16, lane: u8, epoch: u8) -> [u8; 16] {
        let mut delta = [0u8; 16];

        // Bytes 0-4: Tick (40 bits, use 5 bytes of u64, Little Endian)
        let tick_bytes = tick_cs.to_le_bytes();
        delta[0..5].copy_from_slice(&tick_bytes[0..5]);

        // Bytes 5-6: NodeID (16 bits, Little Endian)
        delta[5..7].copy_from_slice(&node_id.to_le_bytes());

        // Byte 7: Lane (8 bits)
        delta[7] = lane;

        // Byte 8: Epoch (8 bits)
        delta[8] = epoch;

        // Bytes 9-15: Reserved (set to zero)
        // Already initialized to zero

        delta
    }

    /// Extract Tick (centiseconds) from delta signature.
    pub fn tick_cs(&self) -> u64 {
        let mut tick_bytes = [0u8; 8];
        tick_bytes[0..5].copy_from_slice(&self.delta_sig[0..5]);
        u64::from_le_bytes(tick_bytes)
    }

    /// Extract NodeID (Codex root_id) from delta signature.
    pub fn node_id(&self) -> u16 {
        u16::from_le_bytes(self.delta_sig[5..7].try_into().unwrap())
    }

    /// Extract Lane from delta signature.
    pub fn lane(&self) -> u8 {
        self.delta_sig[7]
    }

    /// Extract Epoch from delta signature.
    pub fn epoch(&self) -> u8 {
        self.delta_sig[8]
    }
}

// ============================================================================
// Construction
// ============================================================================

impl Default for Xuid {
    fn default() -> Self {
        Self::new(b"")
    }
}

impl Xuid {
    /// Create new XUID from data (E8 quantized).
    /// Uses current system time for the Tick component of Delta.
    /// Node ID defaults to 0, Lane defaults to 0, and Epoch defaults to 0.
    pub fn new(data: &[u8]) -> Self {
        let tick_cs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64 / 10; // Convert milliseconds to centiseconds
        Self::create(data, XuidType::E8Quantized, tick_cs, 0, 0, 0)
    }

    /// Create new XUID with explicit type (uses current time).
    /// Node ID defaults to 0, Lane defaults to 0, and Epoch defaults to 0.
    pub fn new_e8_with_type(data: &[u8], xuid_type: XuidType) -> Self {
        let tick_cs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64 / 10; // Convert milliseconds to centiseconds
        Self::create(data, xuid_type, tick_cs, 0, 0, 0)
    }

    /// Create new XUID with explicit type, timestamp (seconds), node ID, and optional causal hash.
    /// This is the "God Mode" constructor for the Codex.
    /// A root event (no parents) should provide `None` for `causal_hash_80_bits`.
    #[allow(clippy::too_many_arguments)] // This is the "God Mode" constructor.
    pub fn create(
        data: &[u8],
        xuid_type: XuidType,
        tick_cs: u64,
        node_id: u16,
        lane: u8,
        epoch: u8,
    ) -> Self {
        let semantic_hash = *blake3::hash(data).as_bytes();
        let delta_sig = Self::pack_delta(tick_cs, node_id, lane, epoch);
        let (e8_orbit, e8_coords) = quantize_to_orbit(data).unwrap_or((0, [0.0; 8]));

        Self {
            xuid_type,
            e8_orbit,
            delta_sig,
            semantic_hash,
            e8_coords,
            semantic_path: None,
            provenance: None,
        }
    }

    /// Create from semantic path string.
    /// Uses current system time for the Tick, Node ID defaults to 0, Lane defaults to 0,
    /// and Epoch defaults to 0.
    pub fn from_semantic_path(path: &str, xuid_type: XuidType) -> Self {
        let tick_cs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64 / 10; // Convert milliseconds to centiseconds
        let mut xuid = Self::create(path.as_bytes(), xuid_type, tick_cs, 0, 0, 0);
        xuid.semantic_path = Some(SemanticPath::from_str(path).expect("Invalid semantic path"));
        xuid
    }

    // Builder methods
    pub fn with_semantic_path(mut self, path: SemanticPath) -> Self {
        self.semantic_path = Some(path);
        self
    }

    pub fn with_provenance(mut self, provenance: XuidProvenance) -> Self {
        self.provenance = Some(provenance);
        self
    }

    /// Create v4-style random XUID (Semantic Traceable).
    /// Generates random timestamp (seconds), Node ID, and causal hash components.
    pub fn new_v4() -> Self {
        use rand::{Rng, RngCore, SeedableRng, rngs::SmallRng};
        use std::cell::RefCell;

        struct V4Generator {
            rng: SmallRng,
        }

        impl V4Generator {
            fn new() -> Self {
                let mut seed = [0u8; 32];
                rand::rng().fill_bytes(&mut seed);
                Self {
                    rng: SmallRng::from_seed(seed),
                }
            }

            fn next_v4(&mut self) -> Xuid {
                let mut random_bytes = [0u8; 16];
                self.rng.fill_bytes(&mut random_bytes);
                
                let tick_cs = self.rng.random::<u64>() % (1 << 40); // Generate 40-bit random tick
                let node_id = self.rng.random::<u16>();
                let lane = self.rng.random::<u8>();
                let epoch = self.rng.random::<u8>();

                let mut xuid = Xuid::create(
                    &random_bytes,
                    XuidType::E8Quantized,
                    tick_cs,
                    node_id,
                    lane,
                    epoch,
                );

                xuid.semantic_path = Some(SemanticPath::new(vec![
                    "random".to_string(),
                    "v4".to_string(),
                ]));

                let mut metadata = std::collections::BTreeMap::new();
                metadata.insert("random_bytes".to_string(), hex::encode(random_bytes));
                metadata.insert("source".to_string(), "Xuid::new_v4".to_string());

                xuid.provenance = Some(XuidProvenance {
                    source: "xuid::new_v4".to_string(),
                    timestamp: Some(tick_cs as i64), // Use new tick_cs
                    metadata,
                });

                xuid
            }
        }

        thread_local! {
            static G: RefCell<V4Generator> = RefCell::new(V4Generator::new());
        }

        G.with(|g| g.borrow_mut().next_v4())
    }
}

// ============================================================================
// Similarity Computation
// ============================================================================

impl Xuid {
    pub fn similarity(&self, other: &Self) -> f32 {
        let e8_dist = e8_distance(&self.e8_coords, &other.e8_coords);
        let e8_sim = (-e8_dist / 10.0).exp();
        let orbit_sim = orbit_correlation(self.e8_orbit, other.e8_orbit);
        let hash_sim = hamming_similarity(&self.semantic_hash[..16], &other.semantic_hash[..16]);
        0.5 * e8_sim + 0.3 * orbit_sim + 0.2 * hash_sim
    }
}

#[inline]
fn hamming_similarity(a: &[u8], b: &[u8]) -> f32 {
    let mut diff_bits = 0u32;
    for i in 0..a.len().min(b.len()) {
        diff_bits += (a[i] ^ b[i]).count_ones();
    }
    let total_bits = (a.len().min(b.len()) * 8) as f32;
    1.0 - (diff_bits as f32 / total_bits)
}

// ============================================================================
// Display & Parsing (Canonical Format)
// ============================================================================

impl fmt::Display for Xuid {
    /// Canonical Display Format: `XU:TYPE:DELTA:SIG[:S=...][:P=...]:ID`
    ///
    /// This format embeds the 96-byte core + optional heap metadata into the strict
    /// Xypher Codex format.
    ///
    /// - `S`: Hex-encoded semantic path string.
    /// - `P`: Hex-encoded provenance source string.
    /// - `:ID`: Literal suffix.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "XU:{}:{}:{}",
            type_to_str(self.xuid_type),
            hex::encode(self.delta_sig),
            hex::encode(self.semantic_hash)
        )?;

        if let Some(path) = &self.semantic_path {
            write!(f, ":S={}", hex::encode(path.to_string()))?;
        }

        if let Some(prov) = &self.provenance {
            write!(f, ":P={}", hex::encode(&prov.source))?;
        }

        // The ID tail is the literal suffix requirement.
        write!(f, ":ID")
    }
}

impl std::str::FromStr for Xuid {
    type Err = XuidError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Fallback to legacy parsing if it starts with "xuid:" (lowercase)
        if s.starts_with("xuid:") {
            return parse_legacy_xuid(s);
        }

        // Canonical parsing
        let mut iter = s.split(':');

        // Prefix XU
        match iter.next() {
            Some("XU") => {},
            _ => return Err(XuidError::InvalidFormat("Missing XU prefix".into())),
        }

        let type_str = iter.next().ok_or(XuidError::InvalidFormat("Missing type".into()))?;
        let delta_hex = iter.next().ok_or(XuidError::InvalidFormat("Missing delta".into()))?;
        let sig_hex = iter.next().ok_or(XuidError::InvalidFormat("Missing signature".into()))?;

        // Process optionals until we hit "ID" suffix or run out
        let mut semantic_path = None;
        let mut provenance = None;

        for part in iter {
            if part == "ID" {
                break; // End of canonical format
            }
            if let Some(val) = part.strip_prefix("S=") {
                let decoded = hex::decode(val).map_err(|_| XuidError::InvalidFormat("Invalid hex in S".into()))?;
                let path_str = String::from_utf8(decoded).map_err(|_| XuidError::InvalidFormat("Invalid UTF8 in S".into()))?;
                semantic_path = Some(SemanticPath::from_str(&path_str)?);
            } else if let Some(val) = part.strip_prefix("P=") {
                let decoded = hex::decode(val).map_err(|_| XuidError::InvalidFormat("Invalid hex in P".into()))?;
                let source_str = String::from_utf8(decoded).map_err(|_| XuidError::InvalidFormat("Invalid UTF8 in P".into()))?;
                provenance = Some(XuidProvenance {
                    source: source_str,
                    timestamp: None,
                    metadata: std::collections::BTreeMap::new(),
                });
            }
        }

        let xuid_type = type_from_str(type_str)?;

        let delta_vec = hex::decode(delta_hex).map_err(|_| XuidError::InvalidFormat("Invalid delta hex".into()))?;
        let sig_vec = hex::decode(sig_hex).map_err(|_| XuidError::InvalidFormat("Invalid sig hex".into()))?;

                if delta_vec.len() != 16 || sig_vec.len() != 32 {
                    return Err(XuidError::InvalidFormat("Invalid binary lengths".into()));
                }
        
                let mut delta_sig = [0u8; 16]; 
                delta_sig.copy_from_slice(&delta_vec);        let mut semantic_hash = [0u8; 32];
        semantic_hash.copy_from_slice(&sig_vec);

        let (e8_orbit, e8_coords) = quantize_to_orbit(&semantic_hash)?;

        Ok(Self {
            xuid_type,
            e8_orbit,
            delta_sig,
            semantic_hash,
            e8_coords,
            semantic_path,
            provenance,
        })
    }
}

// Helpers for Display/FromStr

fn type_to_str(t: XuidType) -> &'static str {
    match t {
        XuidType::E8Quantized => "E8Q",
        XuidType::Experience => "EXP",
        XuidType::Anomaly => "ERR",
        XuidType::Healing => "HEL",
        XuidType::Codex => "CDX", // New string representation for Codex type
    }
}

fn type_from_str(s: &str) -> Result<XuidType, XuidError> {
    match s {
        "E8Q" | "E8Quantized" => Ok(XuidType::E8Quantized),
        "EXP" | "Experience" => Ok(XuidType::Experience),
        "ERR" | "Anomaly" => Ok(XuidType::Anomaly),
        "HEL" | "Healing" => Ok(XuidType::Healing),
        "CDX" | "Codex" => Ok(XuidType::Codex), // New string representation for Codex type
        // Fallback for number-based legacy
        n if n.chars().all(char::is_numeric) => {
             let u: u8 = n.parse().map_err(|_| XuidError::InvalidFormat("bad type num".into()))?;
             XuidType::from_u8(u)
        }
        _ => Err(XuidError::InvalidFormat(format!("Unknown type: {}", s))),
    }
}

fn parse_legacy_xuid(s: &str) -> Result<Xuid, XuidError> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() < 4 { return Err(XuidError::InvalidFormat("Legacy format error".into())); }

    let type_u8: u8 = parts[1].parse().map_err(|_| XuidError::InvalidFormat("Invalid type number".into()))?;
    let xuid_type = XuidType::from_u8(type_u8)?;
    let delta_vec = hex::decode(parts[2]).map_err(|_| XuidError::InvalidFormat("Legacy delta hex error".into()))?;
    let hash_vec = hex::decode(parts[3]).map_err(|_| XuidError::InvalidFormat("Legacy hash hex error".into()))?;

    if delta_vec.len() != 9 || hash_vec.len() != 32 {
         return Err(XuidError::InvalidFormat("Legacy length error".into()));
    }

    let mut delta = [0u8; 16];
    // Only copy first 9 bytes from legacy format, rest remain 0
    delta[0..9].copy_from_slice(&delta_vec);
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&hash_vec);

    let (e8_orbit, e8_coords) = quantize_to_orbit(&hash)?;

    Ok(Xuid {
        xuid_type,
        e8_orbit,
        delta_sig: delta,
        semantic_hash: hash,
        e8_coords,
        semantic_path: None,
        provenance: None,
    })
}

// ============================================================================
// Serde
// ============================================================================

impl Serialize for Xuid {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Xuid", 7)?;
        state.serialize_field("xuid_type", &(self.xuid_type as u8))?;
        state.serialize_field("e8_orbit", &self.e8_orbit)?;
        state.serialize_field("delta_sig", &hex::encode(self.delta_sig))?;
        state.serialize_field("semantic_hash", &hex::encode(self.semantic_hash))?;
        state.serialize_field("e8_coords", &self.e8_coords)?;
        state.serialize_field("semantic_path", &self.semantic_path)?;
        state.serialize_field("provenance", &self.provenance)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Xuid {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        #[derive(Deserialize)]
        struct XuidHelper {
            xuid_type: u8,
            e8_orbit: u8,
            delta_sig: String,
            semantic_hash: String,
            e8_coords: E8Point,
            semantic_path: Option<SemanticPath>,
            provenance: Option<XuidProvenance>,
        }
        let h = XuidHelper::deserialize(deserializer)?;
        let xuid_type = XuidType::from_u8(h.xuid_type).map_err(serde::de::Error::custom)?;
        let d_vec = hex::decode(&h.delta_sig).map_err(serde::de::Error::custom)?;
        let s_vec = hex::decode(&h.semantic_hash).map_err(serde::de::Error::custom)?;

        let mut d = [0u8; 16];
        let mut s = [0u8; 32];
        // Handle variable-length delta_sig (for backwards compatibility)
        if d_vec.len() >= 9 {
            d[0..9].copy_from_slice(&d_vec[0..9]);
            if d_vec.len() > 9 {
                let copy_len = d_vec.len().min(16);
                d[0..copy_len].copy_from_slice(&d_vec[0..copy_len]);
            }
        }
        if s_vec.len() == 32 { s.copy_from_slice(&s_vec); }

        Ok(Self {
            xuid_type,
            e8_orbit: h.e8_orbit,
            delta_sig: d,
            semantic_hash: s,
            e8_coords: h.e8_coords,
            semantic_path: h.semantic_path,
            provenance: h.provenance,
        })
    }
}

impl Xuid {
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts((self as *const Self) as *const u8, std::mem::size_of::<Self>()) }
    }
}
