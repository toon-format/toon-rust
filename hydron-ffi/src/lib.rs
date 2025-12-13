#![allow(clippy::missing_safety_doc)]
//! hydron-ffi: Complete FFI and wasm-bindgen wrapper for Hydron geometry
//!
//! This crate exposes the full Hydron geometry engine via:
//! - C-ABI foreign function interface (for native code)
//! - wasm-bindgen JavaScript bindings (for WebAssembly)
//!
//! The objective is to provide full access to the E8 geometry compute engine
//! in a single, well-defined entry point that can be compiled to WebAssembly
//! and used in the browser/Node.js or any other host that supports Wasm.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

// Re-export all hydron-core types and operations
pub use hydron_core::{
    FisherLayer, Gf8, Gf8Tensor, HyperbolicLayer, LorentzianCausalLayer, LorentzianLayer,
    PersistencePair, QuaternionOps, SpacetimePoint, SphericalLayer, SymplecticLayer,
    TopologicalLayer, intrinsics_for_f32_width,
};

/// C-ABI: compute spherical geodesic distance between two 8-element float arrays
/// `a_ptr` and `b_ptr` point to 8 contiguous `f32` values.
pub unsafe extern "C" fn s7_distance(a_ptr: *const f32, b_ptr: *const f32) -> f32 {
    // Basic null pointer checks
    if a_ptr.is_null() || b_ptr.is_null() {
        return f32::NAN;
    }

    // Unsafe block to read raw pointers
    let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, 8) };
    let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, 8) };

    // Copy into fixed arrays expected by SphericalLayer::distance
    let mut a = [0.0f32; 8];
    let mut b = [0.0f32; 8];
    a.copy_from_slice(a_slice);
    b.copy_from_slice(b_slice);

    SphericalLayer::distance(&a, &b)
}

/// Safe Rust wrapper for unit tests and consumers that don't want to use raw
/// pointers.
pub fn s7_distance_rust(a: [f32; 8], b: [f32; 8]) -> f32 {
    SphericalLayer::distance(&a, &b)
}

// Optional wasm-bindgen exported function for JS consumers. This uses a
// Float32Array-compatible signature: two slices of `f32`.
#[cfg(feature = "wasm-bindgen")]
mod wasm_bindings {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(js_name = s7_distance)]
    pub fn s7_distance_js(a: &[f32], b: &[f32]) -> f32 {
        // Minimal validation: require length 8
        if a.len() != 8 || b.len() != 8 {
            return f32::NAN;
        }
        let mut aa = [0.0f32; 8];
        let mut bb = [0.0f32; 8];
        aa.copy_from_slice(a);
        bb.copy_from_slice(b);
        crate::s7_distance_rust(aa, bb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s7_distance_rust() {
        let a = [
            1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32,
        ];
        let b = [
            0.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32,
        ];
        let d = s7_distance_rust(a, b);
        // With orthogonal unit vectors, distance is π/2
        let expected = std::f32::consts::FRAC_PI_2;
        assert!((d - expected).abs() < 1e-6);
    }

    #[test]
    fn test_s7_distance_null_ptr() {
        // Passing a null pointer returns NaN
        let d = unsafe { s7_distance(std::ptr::null(), std::ptr::null()) };
        assert!(d.is_nan());
    }
}
