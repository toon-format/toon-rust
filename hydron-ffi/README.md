# Hydron-FFI: E8 Geometry Engine for WebAssembly & Native Platforms

A high-performance foreign function interface (FFI) for the **Hydron** E8 geometric mathematics engine. Provides seamless access to advanced geometric computing across multiple platforms:

- **WebAssembly (WASM):** Browser and Node.js via `wasm-bindgen`
- **C/C++:** Native code via stable C-ABI exports
- **Rust:** Direct crate dependency with full type safety

## Features

**Hydron-FFI exposes the complete Hydron-core geometric toolkit:**

- **Gf8 (GeoFloat8):** 8-dimensional normalized geometric floats on the unit hypersphere
- **Spherical Geometry (S⁷):** Geodesic distances, spherical interpolation (SLERP), and antipodal operations
- **Hyperbolic Geometry (H⁸):** Poincaré ball model with Möbius arithmetic
- **Fisher Information Geometry:** Statistical manifolds, KL divergence, and information metrics
- **Symplectic Geometry (T*E⁸):** Hamiltonian phase space dynamics and canonical transformations
- **Lorentzian Geometry:** Spacetime metrics, causal structure, and light cone analysis
- **Quaternion Algebra:** 4D rotations, composition, and SLERP
- **Topological Analysis:** Persistent homology, Betti numbers, and cohomology

## Quick Start

### Native Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
hydron-ffi = { path = "hydron-ffi" }
```

Use in your code:

```rust
use hydron_ffi::{Gf8, SphericalLayer, Gf8Tensor};

// Create normalized 8D points
let a = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
let b = Gf8::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

// Compute spherical geodesic distance
let distance = SphericalLayer::distance(a.as_slice(), b.as_slice());
println!("S⁷ distance: {}", distance);
```

### Testing

```powershell
cd hydron-ffi
cargo test
```

### WebAssembly

Build the WASM module with `wasm-pack`:

```powershell
# Requires wasm-pack: https://rustwasm.github.io/wasm-pack/installer/

cd hydron-ffi
wasm-pack build --release --target web
```

This generates `pkg/` containing:

- `hydron_ffi.js` — JavaScript bindings
- `hydron_ffi_bg.wasm` — Compiled WebAssembly module

Use in a web project:

```javascript
import * as hydron from './pkg/hydron_ffi.js';

// Access all geometry operations
const distance = hydron.s7_distance_rust([1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0]);
console.log("S⁷ distance:", distance);
```

### C/C++ FFI

All public Rust functions are C-ABI compatible. Example C code:

```c
#include <stdint.h>

// C declaration
extern float s7_distance(const float *a, const float *b);

int main() {
    float a[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float b[8] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    float distance = s7_distance(a, b);
    printf("S⁷ distance: %f\n", distance);
    return 0;
}
```

## Architecture

**Hydron-FFI** serves as the public API surface for the **Hydron-core** geometry engine:

```
┌─────────────────────────────────────┐
│   Consumer Applications             │
│  (Browser, Node.js, C/C++, Rust)    │
└────────────┬────────────────────────┘
             │ FFI / wasm-bindgen
┌────────────▼────────────────────────┐
│        Hydron-FFI (this crate)      │
│      C-ABI & WASM Bindings          │
└────────────┬────────────────────────┘
             │ Re-exports all types
┌────────────▼────────────────────────┐
│        Hydron-Core                  │
│   Pure E8 Geometry Mathematics      │
│  (9 geometric computation modules)  │
└─────────────────────────────────────┘
```

## Dependencies

- **hydron-core:** E8 geometry engine (path dependency)
- **wasm-bindgen:** (optional) JavaScript integration for WebAssembly

## Build & Runtime Characteristics

All geometric operations are:

- **Vectorized:** Designed for 256-bit SIMD (AVX, NEON) with scalar fallbacks
- **Zero-copy:** Direct pointer access in C mode, minimal allocation in WASM
- **Normalized:** Unit-sphere representations provide numerical stability

## Learn More

For detailed geometric theory and API documentation, see:

- [Hydron-core documentation](../hydron-core/README.md)
- Inline code documentation: `cargo doc --open`

## License

MIT OR Apache-2.0
