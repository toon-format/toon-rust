# Hydron-Core: E8 Geometric Mathematics Engine

A pure, high-performance Rust library for advanced E8 lattice geometry and differential geometry computations. Provides nine interconnected geometric layers for representing and manipulating 8-dimensional geometric data.

## Features

**Nine Geometric Computation Modules:**

- **Gf8 (GeoFloat8):** 8-dimensional normalized geometric floats on the unit hypersphere S⁷
- **Spherical Geometry (S⁷):** Geodesic distances, spherical linear interpolation (SLERP), antipodal points, and great circle arcs
- **Hyperbolic Geometry (H⁸):** Poincaré ball model with Möbius group operations and hyperbolic metrics
- **Fisher Information Geometry:** Statistical manifolds, Fisher information matrices, KL divergence, and information-theoretic metrics
- **Symplectic Geometry (T*E⁸):** Hamiltonian mechanics on the cotangent bundle, phase space dynamics, and canonical transformations
- **Lorentzian Geometry:** Spacetime metrics, causal structure, light cone analysis, and relativistic invariants
- **Quaternion Algebra:** 4D rotation algebra, quaternion composition, SLERP, and axis-angle conversions
- **Topological Analysis:** Persistent homology computation, Betti numbers, and cohomology groups
- **SIMD Intrinsics:** Runtime feature detection and scalar fallbacks for portable performance

## Quick Start

### Adding to Your Project

```toml
[dependencies]
hydron-core = { path = "hydron-core" }
```

### Basic Usage

```rust
use hydron_core::{Gf8, Gf8Tensor, SphericalLayer};

// Create normalized 8D geometric floats
let a = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
let b = Gf8::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

// Compute spherical geodesic distance
let distance = SphericalLayer::distance(a.as_slice(), b.as_slice());
println!("Geodesic distance on S⁷: {}", distance);

// Spherical interpolation (SLERP)
let interpolated = SphericalLayer::slerp(a.as_slice(), b.as_slice(), 0.5);
println!("Midpoint: {:?}", interpolated);

// Dot product (cosine similarity)
let similarity = a.dot(b.coords());
println!("Cosine similarity: {}", similarity);
```

### All Nine Modules

```rust
use hydron_core::{
    Gf8, SphericalLayer, HyperbolicLayer, FisherLayer,
    SymplecticLayer, LorentzianLayer, QuaternionOps,
    TopologicalLayer, PersistencePair
};

// Hyperbolic distance (Poincaré ball)
let h_dist = HyperbolicLayer::distance(&a_coords, &b_coords);

// Fisher information metric
let fisher_dist = FisherLayer::fisher_distance(&a_coords, &b_coords);

// Quaternion rotation
let rotation = QuaternionOps::slerp(&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0], 0.5);

// Topological analysis
let betti = TopologicalLayer::compute_betti_numbers(&point_cloud);
println!("Betti numbers (b₀, b₁, b₂): {:?}", betti);
```

## Module Overview

### Gf8 (GeoFloat8)

The foundational type representing an 8-dimensional unit vector on the hypersphere S⁷. All coordinates are automatically normalized to unit length, providing:

- Inherent numerical stability
- Perfect alignment with 256-bit SIMD registers (AVX, AVX2)
- Efficient representation for E8 lattice geometry

```rust
let v = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
let coords: [f32; 8] = v.coords();  // Access as array
let slice: &[f32] = v.as_slice();   // Access as slice
```

### Spherical Geometry (S⁷)

Operations on the 7-sphere embedded in 8-dimensional Euclidean space:

- `distance()` — Geodesic distance via arccos of dot product
- `slerp()` — Spherical linear interpolation maintaining geodesic paths
- `antipodal()` — Antipodal point (negation on sphere)
- `mean()` — Riemannian mean of point clouds
- `project()` — Orthogonal projection onto sphere

### Hyperbolic Geometry (H⁸)

Poincaré ball model of hyperbolic space in 8 dimensions:

- `distance()` — Hyperbolic distance metric
- `mobius_add()` — Möbius group addition operation
- Suitable for hierarchical data and tree-like structures

### Fisher Information Geometry

Statistical manifold of probability distributions:

- `fisher_matrix()` — Compute Fisher information matrix
- `kl_divergence()` — Kullback-Leibler divergence between distributions
- `information_metric()` — Riemannian metric on manifold
- `entropy()` — Differential entropy calculation
- `fisher_distance()` — Distance via Fisher metric

### Symplectic Geometry (T*E⁸)

Hamiltonian mechanics on the cotangent bundle of E⁸:

- `hamiltonian()` — Total energy (kinetic + potential)
- Preserves symplectic structure and canonical transformations
- Foundation for phase space dynamics

### Lorentzian Geometry

Spacetime geometry with relativistic invariants:

- `in_past_light_cone()` — Causality predicate
- `in_future_light_cone()` — Forward causality
- `spacelike_separation()` — Spacelike interval check
- Minkowski metric with signature (−,+,+,+,+,+,+,+)

### Quaternion Algebra

4D rotation algebra for rigid body kinematics:

- `slerp()` — Spherical linear interpolation of rotations
- `compose()` — Quaternion multiplication (rotation composition)
- `conjugate()` — Inverse rotation
- Smooth interpolation without gimbal lock

### Topological Analysis

Persistent homology and topological invariants:

- `compute_betti_numbers()` — Count topological features
- `PersistencePair` — Birth-death pairs of topological features
- Robust to noise via persistence filtering

### SIMD Intrinsics

Runtime CPU feature detection and portable SIMD:

- `intrinsics_for_f32_width()` — Query available SIMD widths
- Automatic fallback to scalar implementations
- Support for AVX, NEON, and other SIMD targets

## Architecture

All modules operate on **pure mathematics** without dependencies on:

- Application-specific evaluation logic
- Serialization formats
- Runtime value systems

This enables:

1. **Reusability** across different contexts (RUNE, WASM, C/C++, etc.)
2. **Portability** to embedded systems and specialized hardware
3. **Testability** with pure mathematical specifications
4. **Performance** through aggressive optimization without framework overhead

┌────────────────────────────────────────┐
│  High-Level Applications (RUNE, etc.)  │
└────────────────┬───────────────────────┘
                 │ Optional dependency
┌────────────────▼───────────────────────┐
│         Hydron-Core (this crate)       │
│  Pure E8 Geometry + Linear Algebra     │
│  (9 interconnected modules)            │
└────────────────────────────────────────┘

## Building & Testing

### Build

```powershell
cargo build
```

### Run Tests

```powershell
cargo test
```

### Generate Documentation

```powershell
cargo doc --open
```

## Performance Characteristics

- **Vectorized:** Targets 256-bit SIMD (AVX/AVX2 on x86_64, NEON on ARM)
- **Normalized:** Unit-sphere representation provides numerical stability
- **Zero-allocation:** Pure stack-based computation in most operations
- **Portable:** Scalar fallbacks ensure compatibility across all platforms

## Dependencies

**No external dependencies** — pure Rust with optional standard library features.

## Crate Versions

- **Edition:** 2024
- **MSRV:** 1.80+ (tested on Rust 1.89.0)

## Integration Points

Hydron-core is used by:

- **hydron-ffi:** FFI wrapper for C/C++ and WebAssembly
- **rune-format:** RUNE evaluation engine and expression evaluator
- Standalone projects requiring pure geometric computation

## Mathematical References

The geometric foundations draw from:

- **E8 Lattice:** Coxeter's regular polytope theory
- **Differential Geometry:** Riemannian manifolds and geodesics
- **Information Geometry:** Amari & Nagaoka's statistical manifold framework
- **Symplectic Geometry:** Arnold's classical mechanics formulation
- **Hyperbolic Geometry:** Poincaré models and Möbius transformations
- **Persistent Homology:** Edelsbrunner & Harer's topological data analysis

## License

MIT OR Apache-2.0

## Contributing

Hydron-core is part of the ArcMoon Studios E8 ecosystem. Contributions should maintain:

- Pure mathematical semantics
- Zero external dependencies
- Portable, standards-compliant Rust
- Comprehensive documentation and doctests

---

**For FFI bindings and application integration, see:** [hydron-ffi](../hydron-ffi/README.md)

**For RUNE language integration, see:** [rune-format](../README.md)
