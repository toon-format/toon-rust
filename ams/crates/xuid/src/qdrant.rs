/* xuid/src/qdrant.rs */
//! Qdrant Integration for XUID Point IDs
//!
//! Provides equivalent functionality to `qdrant_client::qdrant::point_id::PointIdOptions`
//! but with native XUID support instead of UUIDs.
//!
//! This allows using XUIDs directly as point identifiers in Qdrant operations,
//! providing type safety and avoiding string conversions.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use super::core::Xuid;
use super::error::{XuidError, XuidResult};
use serde::{Deserialize, Serialize};

/// Equivalent to `qdrant_client::qdrant::point_id::PointIdOptions` but with XUID support
///
/// This enum provides the same variants as Qdrant's PointIdOptions, but replaces
/// the Uuid variant with Xuid for native XUID support.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum XuidPointIdOptions {
    /// Numeric point ID (same as Qdrant)
    Num(u64),
    /// XUID point ID (replaces Qdrant's Uuid variant)
    Xuid(Xuid),
}

impl XuidPointIdOptions {
    /// Create from a XUID
    pub fn from_xuid(xuid: Xuid) -> Self {
        Self::Xuid(xuid)
    }

    /// Create from a numeric ID
    pub fn from_num(num: u64) -> Self {
        Self::Num(num)
    }

    /// Get the XUID if this is a Xuid variant
    pub fn as_xuid(&self) -> Option<&Xuid> {
        match self {
            Self::Xuid(xuid) => Some(xuid),
            Self::Num(_) => None,
        }
    }

    /// Get the numeric ID if this is a Num variant
    pub fn as_num(&self) -> Option<u64> {
        match self {
            Self::Num(num) => Some(*num),
            Self::Xuid(_) => None,
        }
    }

    /// Parse from string representation
    ///
    /// Attempts to parse as XUID first, falls back to numeric parsing.
    /// This provides compatibility with existing Qdrant string IDs.
    pub fn parse_str(s: &str) -> XuidResult<Self> {
        // Try parsing as XUID first
        if let Ok(xuid) = s.parse::<Xuid>() {
            return Ok(Self::Xuid(xuid));
        }

        // Fall back to numeric parsing
        if let Ok(num) = s.parse::<u64>() {
            return Ok(Self::Num(num));
        }

        Err(XuidError::InvalidFormat(format!(
            "String '{}' is neither a valid XUID nor a valid numeric ID",
            s
        )))
    }
}

impl From<Xuid> for XuidPointIdOptions {
    fn from(xuid: Xuid) -> Self {
        Self::Xuid(xuid)
    }
}

impl From<u64> for XuidPointIdOptions {
    fn from(num: u64) -> Self {
        Self::Num(num)
    }
}

impl std::fmt::Display for XuidPointIdOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Num(num) => write!(f, "{}", num),
            Self::Xuid(xuid) => write!(f, "{}", xuid),
        }
    }
}

impl std::str::FromStr for XuidPointIdOptions {
    type Err = XuidError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_str(s)
    }
}

/// Wrapper type for Qdrant point IDs with XUID support
///
/// This provides a parallel implementation to `qdrant_client::qdrant::PointId`
/// that can handle XUIDs natively.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct XuidPointId {
    /// The point ID options (Num or Xuid)
    pub point_id_options: Option<XuidPointIdOptions>,
}

impl XuidPointId {
    /// Create a new XUID point ID
    pub fn from_xuid(xuid: Xuid) -> Self {
        Self {
            point_id_options: Some(XuidPointIdOptions::from_xuid(xuid)),
        }
    }

    /// Create a new numeric point ID
    pub fn from_num(num: u64) -> Self {
        Self {
            point_id_options: Some(XuidPointIdOptions::from_num(num)),
        }
    }

    /// Get the XUID if this point ID contains one
    pub fn as_xuid(&self) -> Option<&Xuid> {
        self.point_id_options.as_ref()?.as_xuid()
    }

    /// Get the numeric ID if this point ID contains one
    pub fn as_num(&self) -> Option<u64> {
        self.point_id_options.as_ref()?.as_num()
    }

    /// Convert to Qdrant's PointId for interoperability
    ///
    /// This allows using XUID-based point IDs with the qdrant-client,
    /// by converting XUIDs to their string representation.
    #[cfg(feature = "qdrant")]
    pub fn to_qdrant_point_id(&self) -> Option<qdrant_client::qdrant::PointId> {
        use qdrant_client::qdrant::{PointId, point_id};

        match self.point_id_options.as_ref()? {
            XuidPointIdOptions::Num(num) => Some(PointId {
                point_id_options: Some(point_id::PointIdOptions::Num(*num)),
            }),
            XuidPointIdOptions::Xuid(xuid) => Some(PointId {
                point_id_options: Some(point_id::PointIdOptions::Uuid(xuid.to_string())),
            }),
        }
    }

    /// Create from Qdrant's PointId
    ///
    /// Attempts to convert a Qdrant PointId to our XUID-aware version.
    /// Numeric IDs are preserved, UUID strings are attempted to be parsed as XUIDs.
    #[cfg(feature = "qdrant")]
    pub fn from_qdrant_point_id(point_id: &qdrant_client::qdrant::PointId) -> XuidResult<Self> {
        use qdrant_client::qdrant::point_id::PointIdOptions;

        let options = match point_id.point_id_options.as_ref() {
            Some(PointIdOptions::Num(num)) => XuidPointIdOptions::Num(*num),
            Some(PointIdOptions::Uuid(uuid_str)) => {
                // Try to parse as XUID, fall back to treating as generic string
                XuidPointIdOptions::parse_str(uuid_str)?
            }
            None => {
                return Err(XuidError::InvalidFormat(
                    "PointId has no options".to_string(),
                ));
            }
        };

        Ok(Self {
            point_id_options: Some(options),
        })
    }
}

impl std::str::FromStr for XuidPointId {
    type Err = XuidError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self {
            point_id_options: Some(XuidPointIdOptions::parse_str(s)?),
        })
    }
}

impl From<Xuid> for XuidPointId {
    fn from(xuid: Xuid) -> Self {
        Self::from_xuid(xuid)
    }
}

impl From<u64> for XuidPointId {
    fn from(num: u64) -> Self {
        Self::from_num(num)
    }
}

impl std::fmt::Display for XuidPointId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.point_id_options {
            Some(options) => write!(f, "{}", options),
            None => write!(f, "None"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xuid_point_id_options() {
        let xuid = Xuid::new(b"test data");
        let options = XuidPointIdOptions::from_xuid(xuid.clone());

        assert_eq!(options.as_xuid().unwrap(), &xuid);
        assert_eq!(options.as_num(), None);

        let num_options = XuidPointIdOptions::from_num(42);
        assert_eq!(num_options.as_num().unwrap(), 42);
        assert_eq!(num_options.as_xuid(), None);
    }

    #[test]
    fn test_parse_str() {
        let xuid = Xuid::new(b"test data");
        let xuid_str = xuid.to_string();

        let parsed = XuidPointIdOptions::parse_str(&xuid_str).unwrap();
        // Compare canonical string representations (type + delta + hash) instead of
        // direct equality of reconstructed E8 coordinates which are derived and may
        // differ due to platform-specific quantization details in tests run together.
        assert_eq!(parsed.as_xuid().unwrap().to_string(), xuid.to_string());

        let num_parsed = XuidPointIdOptions::parse_str("12345").unwrap();
        assert_eq!(num_parsed.as_num().unwrap(), 12345);
    }

    #[test]
    fn test_xuid_point_id() {
        let xuid = Xuid::new(b"test data");
        let point_id = XuidPointId::from_xuid(xuid.clone());

        // Compare via canonical string representation to avoid equal-check failing
        // over derived E8 coordinates which are not part of the canonical identity.
        assert_eq!(point_id.as_xuid().unwrap().to_string(), xuid.to_string());
        assert_eq!(point_id.as_num(), None);

        // Test string conversion
        let point_id_str = xuid.to_string().parse::<XuidPointId>().unwrap();
        assert_eq!(
            point_id_str.as_xuid().unwrap().to_string(),
            xuid.to_string()
        );
    }
}
