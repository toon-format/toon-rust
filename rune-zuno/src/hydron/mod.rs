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
//! (e.g., causal DAGs, event systems) are clearly separated.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod eval;
pub mod perception;
pub mod topology;
pub mod values;

// Re-export the hydron-core crate's math modules when the feature is enabled.
// This keeps the Rune crate's public API stable while the actual math is
// implemented in the `hydron-core` crate.
#[cfg(feature = "hydron")]
pub use hydron_core::{
    FisherLayer, Gf8, Gf8Tensor, HyperbolicLayer, LorentzianCausalLayer, LorentzianLayer,
    PersistencePair, QuaternionOps, SpacetimePoint, SphericalLayer, SymplecticLayer,
    TopologicalLayer, intrinsics_for_f32_width,
};

// When hydron feature is disabled, keep local types for the evaluator and values as-is.
// Re-export the runtime value types so code can always import `crate::rune::hydron::Value`.
pub use crate::rune::hydron::values::{EvalContext, EvalError, Octonion, Value};
