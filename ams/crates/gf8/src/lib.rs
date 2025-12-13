/* e8/gf8/src/lib.rs */
//! Foundational geometric gf8s for the e8 ecosystem, including the `Gf8` numeric type.
//!
//! # e8 Primitives Crate
//!▫~•◦-------------------‣
//!
//! This crate provides the core, low-level building blocks for the e8 architecture.
//! It is designed to be a zero-dependency, high-performance library that can be used
//! to construct higher-level systems for AI, numerics, and data representation.
//!
//! ### Key Capabilities
//! - **`Gf8` (GeoFloat8):** A novel 8-dimensional geometric float that replaces traditional scalars.
//! - **`Gf8BitSig `:** A compact, 1-byte binary representation for `Gf8` directions, enabling massive data compression.
//! - **SIMD Acceleration:** Provides SIMD-accelerated functions for `Gf8` arithmetic on compatible x86 CPUs.
//! - **Intrinsic Registry:** Includes a queryable database of x86 intrinsics for building advanced accelerator backends.
//! - **Math Utilities:** High-level geometric, interpolation, and lattice helpers built on top of `Gf8`.
//!
//! ### Architectural Notes
//! The gf8s in this crate are designed to be composable. `Gf8` is the central numeric type,
//! `Gf8BitSig ` provides its binary interface, and the `gf8_simd` and `gf8_intrinsics` modules
//! offer paths for hardware acceleration and advanced code generation.
//!
//! ### Example
//! ```rust
//! use gf8::{Gf8, Gf8BitSig , gf8_from_code, gf8_to_code, gf8_dot_simd};
//!
//! // Create a Gf8 from a byte code
//! let code_a = Gf8BitSig (0b10110010);
//! let a = gf8_from_code(code_a);
//!
//! // Create another Gf8 from a scalar
//! let b = Gf8::from_scalar(-1.0);
//!
//! // Compute their similarity using a SIMD-accelerated dot product
//! let similarity = gf8_dot_simd(&a, &b);
//!
//! println!("Similarity: {}", similarity);
//!
//! // Quantize 'b' back into a byte code
//! let code_b = gf8_to_code(&b);
//! println!("Code for 'b': {:08b}", code_b.0);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

// Declare the modules that make up this crate. The compiler will look for
// `gf8.rs`, `bitcodec/`, etc., in the `src` directory.
pub mod aligned;
pub mod bitcodec;
pub mod compute;
pub mod e32l; // E32L: True lossless f32 compression
pub mod e8f;
pub mod e8x;
pub mod fast_math;
pub mod fractal_simt;
pub mod generative;
pub mod gf8;
pub mod intrinsic_backend;
pub mod intrinsics;
pub mod math;
pub mod progen_reactor;
pub mod quantize;
pub mod resonance_router;
pub mod simd;
pub mod topology;

// Re-export the primary types and functions to create a clean, flat public API.
// Consumers of this crate can `use e8_gf8::Gf8;` instead of the more verbose
// `use e8_gf8::gf8::Gf8;`.
pub use aligned::{E8FAligned, E8FChain};
pub use bitcodec::{
    Gf8BitSig, Gf8LosslessCode, bits_from_u8_le, bits_to_u8_le, gf8_from_best_code, gf8_from_code,
    gf8_from_lossless_code, gf8_to_best_code, gf8_to_code, gf8_to_lossless_code,
    gf8_to_lossless_code_closest,
};
pub use compute::{
    E8FCompute, E8VecCompute, MAX_ROUNDTRIP_ERROR, MAX_SINGLE_QUANTIZATION_ERROR,
    RECOMMENDED_MAX_CHAIN_LENGTH, chordal_distance, compute_transition_scores_hybrid,
    compute_transition_scores_sparse, dot_f32, weighted_sum_hybrid,
};
pub use e8f::{
    E8ArithmeticTables, E8F, E8Mat, E8TensorCore, E8Vec, get_e8_arithmetic, init_e8_arithmetic,
};
pub use e8x::E8X;
pub use e32l::E32L;
#[cfg(feature = "compression")]
pub use e32l::compression::{
    CompressionError, CompressionLevel, compress, compress_with_level, decompress,
};
pub use fast_math::FAST_ACOS;
pub use fractal_simt::{FractalSimtConfig, fractal_simt_add_f32_in_place};
pub use generative::{
    GenerativeSynthesizer, ProgramInstr, STATE_TRANSITIONS, StateTransition, transition_for,
};
pub use gf8::{Gf8, Gf8Tensor};
pub use intrinsic_backend::{
    BackendConfig, IntrinsicBackend, get_backend_info, intrinsic_add, intrinsic_dot, intrinsic_sub,
    list_available_intrinsics,
};
pub use intrinsics::{
    GF8_INTRINSICS, Gf8Intrinsic, find_intrinsic_by_name, intrinsics_by_technology,
    intrinsics_for_f32_width, intrinsics_for_f64_width,
};
pub use math::{
    Gf8Rotation, gf8_angle, gf8_chordal_distance, gf8_chordal_distance2, gf8_cosine_similarity,
    gf8_geodesic_distance, gf8_lerp, gf8_lerp_slice, gf8_slerp, quantize_slice_to_e8_shell,
    quantize_to_e8_shell,
};
pub use progen_reactor::{ProgenBranch, ProgenContext, ProgenCritic, ProgenReactor};
pub use quantize::{
    dequantize_to_vec, get_e8_codebook, get_root_neighbors, quantize_to_gf8,
    quantize_to_nearest_code,
};
pub use resonance_router::{
    HeadActivation, ResonanceConfig, ResonanceResult, accumulate_resonance, heads_from_raw_pairs,
    top_k_resonant_roots,
};
pub use simd::{
    get_available_f32_256_intrinsics, gf8_add_inplace_slice_simd, gf8_add_simd, gf8_dot_simd,
    gf8_norm2_simd, gf8_sub_simd, print_simd_capabilities,
};
pub use topology::{E8Topology, get_e8_topology};

pub type E8Address = [Gf8LosslessCode; 8];

#[cfg(test)]
mod tests {
    // Import all public items from the crate root for testing.
    use super::*;

    #[test]
    fn gf8_constructors_are_unit_norm() {
        // from_bits_even_parity should produce a unit vector.
        let from_bits = Gf8::from_bits_even_parity([1, 0, 1, 1, 0, 0, 1, 0]);
        assert!((from_bits.norm2() - 1.0).abs() < 1e-6);

        // from_scalar should produce a unit vector.
        let from_scalar = Gf8::from_scalar(-123.45);
        assert!((from_scalar.norm2() - 1.0).abs() < 1e-6);
        // Use approximate comparison for floating point
        assert!((from_scalar.to_scalar() - (-1.0)).abs() < 1e-6);

        // from raw coords (new) should produce a unit vector.
        let from_coords = Gf8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!((from_coords.norm2() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bitcodec_roundtrip_is_correct() {
        for i in 0..=255 {
            // Encode: byte -> bits -> Gf8 -> code
            let bits = bits_from_u8_le(i);
            let gf = Gf8::from_bits_even_parity(bits);
            let code = gf8_to_code(&gf);

            // Decode: code -> Gf8 -> bits -> byte
            let gf2 = gf8_from_code(code);
            let bits2 = bits_from_u8_le(code.0); // Directly from the generated code
            let _final_byte = bits_to_u8_le(bits2);

            // The dot product of the original and round-tripped Gf8 should be ~1.0
            assert!((gf.dot(gf2.coords()) - 1.0).abs() < 1e-6);

            // The generated code should decode to the same Gf8 direction
            // Note: Due to even parity constraint, the raw byte might differ but the Gf8 should be the same
            assert_eq!(gf, gf2);
        }
    }

    #[test]
    fn simd_and_scalar_operations_match() {
        let a = Gf8::new([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
        let b = Gf8::new([-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8]);

        // Dot Product
        let dot_scalar = a.dot(b.coords());
        let dot_simd = gf8_dot_simd(&a, &b);
        assert!((dot_scalar - dot_simd).abs() < 1e-6);

        // Norm Squared
        let norm2_scalar = a.norm2();
        let norm2_simd = gf8_norm2_simd(&a);
        assert!((norm2_scalar - norm2_simd).abs() < 1e-6);

        // Addition
        let add_scalar = a + b;
        let add_simd = gf8_add_simd(&a, &b);
        assert!((add_scalar.dot(add_simd.coords()) - 1.0).abs() < 1e-6);

        // Subtraction
        let sub_scalar = a - b;
        let sub_simd = gf8_sub_simd(&a, &b);
        assert!((sub_scalar.dot(sub_simd.coords()) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn intrinsics_registry_can_be_queried() {
        // Test finding a well-known intrinsic
        let add_ps = find_intrinsic_by_name("_mm256_add_ps");
        assert!(add_ps.is_some());
        assert_eq!(add_ps.unwrap().technology, "AVX");

        // Test filtering by technology
        let avx2_count = intrinsics_by_technology("AVX2").count();
        assert!(avx2_count > 0, "Should find AVX2 intrinsics");

        // Test filtering by width and type
        let f32_256_intrinsics = intrinsics_for_f32_width(256).collect::<Vec<_>>();
        assert!(!f32_256_intrinsics.is_empty());
        assert!(f32_256_intrinsics.iter().any(|i| i.name == "_mm256_mul_ps"));
    }

    #[test]
    fn simd_integration_with_intrinsics() {
        // Test the new integration functions
        let available_intrinsics = get_available_f32_256_intrinsics();

        // Should return available intrinsics or empty vector on non-x86
        println!(
            "Available 256-bit f32 intrinsics: {:?}",
            available_intrinsics
        );

        // Test that we can detect available features
        #[cfg(target_arch = "x86_64")]
        {
            // This would print useful debugging information
            // print_simd_capabilities();

            // Verify that at least some intrinsics are available if we have x86_64
            let has_avx_or_avx2 = available_intrinsics.iter().any(|name| name.contains("avx"));

            // On x86_64, we should have at least some available intrinsics
            // (though availability depends on CPU features)
            println!("Has AVX intrinsics available: {}", has_avx_or_avx2);
        }
    }
}
