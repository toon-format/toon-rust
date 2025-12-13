/* src/lib.rs */
//! Hydron - E8 Geometric Mathematics Engine
//!
//! Pure mathematical implementations of E8 lattice geometry with multi-geometric layers:
//! - Fisher information geometry (statistical manifolds)
//! - Symplectic T*E8 geometry (Hamiltonian dynamics)
//! - Hyperbolic H8 geometry (Poincaré ball model)
//! - Topological analysis (persistent homology)
//! - Lorentzian geometry (spacetime metrics)
//! - Quaternion algebra (rotations, SLERP)
//! - Spherical S7 geometry (unit sphere)
//!
//! All modules provide pure geometric operations. Application-specific extensions
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod fisher;
// pub mod gf8; // Replaced by crate gf8
pub mod hyperbolic;
pub mod intrinsics;
pub mod lorentzian;
pub mod quaternion;
pub mod spherical;
pub mod symplectic;
pub mod topological;

// Re-export commonly used types from submodules for convenience
pub use fisher::FisherLayer;
pub use gf8::{Gf8, Gf8Tensor}; // Re-export from the gf8 crate
pub use hyperbolic::HyperbolicLayer;
pub use intrinsics::intrinsics_for_f32_width;
pub use lorentzian::{LorentzianCausalLayer, LorentzianLayer, SpacetimePoint};
pub use quaternion::QuaternionOps;
pub use spherical::SphericalLayer;
pub use symplectic::SymplecticLayer;
pub use topological::{PersistencePair, TopologicalLayer};

// Helper for root generation using gf8 crate
pub fn get_e8_roots() -> Vec<Gf8> {
    // Use the canonical codebook from the gf8 crate
    gf8::get_e8_codebook().roots.to_vec()
}
