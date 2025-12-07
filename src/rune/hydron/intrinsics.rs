/* src/rune/hydron/intrinsics.rs */
//! A queryable registry of x86 SIMD intrinsics for backend code generation.
//!
//! # e8 Primitives – Gf8 Intrinsics Module
//!▫~•◦-----------------------------------------‣
//!
//! This module contains a comprehensive, static list of x86 intrinsics, auto-generated
//! from external documentation. It is designed to be used by the `simd` backend
//! and future procedural code generators to reason about available hardware instructions.
//!
//! ### Key Capabilities
//! - **Static Registry:** Provides `GF8_INTRINSICS`, a constant slice of `Gf8Intrinsic` structs.
//! - **Queryable API:** Offers helper functions to filter and find intrinsics by name, technology, or SIMD width.
//! - **Metadata Rich:** Each entry includes the intrinsic's name, required technology (e.g., AVX2), header, and C prototype.
//!
//! ### Architectural Notes
//! This module acts as a "database" for the compiler backend. Instead of hard-coding
//! intrinsic names, higher-level modules can query this registry to make dynamic
//! decisions about which instructions to use, enabling more flexible and future-proof
//! code generation.
//!
//! ### Example
//! ```rust
//! // This example assumes this module is part of the e8_gf8 crate.
//! // use e8_gf8::intrinsics::{find_intrinsic_by_name, intrinsics_for_f32_width};
//!
//! // fn main() {
//!     // Find a specific intrinsic by name
//!     // if let Some(intrinsic) = find_intrinsic_by_name("_mm256_add_ps") {
//!     //     println!("Found AVX add for f32: {}", intrinsic.prototype);
//!     // }
//!
//!     // Find all 256-bit f32 intrinsics
//!     // let avx_f32_intrinsics = intrinsics_for_f32_width(256).count();
//!     // println!("There are {} relevant 256-bit f32 intrinsics.", avx_f32_intrinsics);
//! // }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// Represents the metadata for a single x86 hardware intrinsic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Gf8Intrinsic {
    /// The C/C++ name of the intrinsic function (e.g., `_mm256_add_ps`).
    pub name: &'static str,
    /// The required CPU feature flag or technology (e.g., "AVX2", "SSE4.1").
    pub technology: &'static str,
    /// The C header file where the intrinsic is typically defined (e.g., "immintrin.h").
    pub header: &'static str,
    /// The C function prototype for the intrinsic.
    pub prototype: &'static str,
}

impl Gf8Intrinsic {
    /// Returns `true` if this intrinsic's prototype suggests it operates on `f32` vectors.
    pub fn is_f32_vector(&self) -> bool {
        self.prototype.contains("__m128")
            || self.prototype.contains("__m256")
            || self.prototype.contains("__m512")
            || self.prototype.contains("ps") // Packed Single
    }

    /// Returns `true` if this intrinsic's prototype suggests it operates on `f64` vectors.
    pub fn is_f64_vector(&self) -> bool {
        self.prototype.contains("__m128d")
            || self.prototype.contains("__m256d")
            || self.prototype.contains("__m512d")
            || self.prototype.contains("pd") // Packed Double
    }

    /// Returns the SIMD vector width in bits, if it can be inferred from the prototype.
    pub fn simd_width_bits(&self) -> Option<u32> {
        if self.prototype.contains("__m512") {
            Some(512)
        } else if self.prototype.contains("__m256") {
            Some(256)
        } else if self.prototype.contains("__m128") {
            Some(128)
        } else if self.prototype.contains("__m64") {
            Some(64)
        } else {
            None
        }
    }
}

/// A static, compile-time registry of all known x86 intrinsics from the source file.
pub const GF8_INTRINSICS: &[Gf8Intrinsic] = &[
    Gf8Intrinsic {
        name: "_m_from_float",
        technology: "3DNOW",
        header: "intrin.h",
        prototype: "__m64 _m_from_float(float);",
    },
    Gf8Intrinsic {
        name: "_m_from_int",
        technology: "MMX",
        header: "intrin.h",
        prototype: "__m64 _m_from_int(int);",
    },
    Gf8Intrinsic {
        name: "_m_maskmovq",
        technology: "SSE",
        header: "intrin.h",
        prototype: "void _m_maskmovq(__m64, __m64, char*);",
    },
    Gf8Intrinsic {
        name: "_mm_abs_epi16",
        technology: "SSSE3",
        header: "intrin.h",
        prototype: "__m128i _mm_abs_epi16(__m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_abs_epi32",
        technology: "SSSE3",
        header: "intrin.h",
        prototype: "__m128i _mm_abs_epi32(__m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_abs_epi8",
        technology: "SSSE3",
        header: "intrin.h",
        prototype: "__m128i _mm_abs_epi8(__m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_add_epi16",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128i _mm_add_epi16(__m128i, __m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_add_epi32",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128i _mm_add_epi32(__m128i, __m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_add_epi64",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128i _mm_add_epi64(__m128i, __m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_add_epi8",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128i _mm_add_epi8(__m128i, __m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_add_pd",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128d _mm_add_pd(__m128d, __m128d);",
    },
    Gf8Intrinsic {
        name: "_mm_add_ps",
        technology: "SSE",
        header: "intrin.h",
        prototype: "__m128 _mm_add_ps(__m128, __m128);",
    },
    Gf8Intrinsic {
        name: "_mm_add_sd",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128d _mm_add_sd(__m128d, __m128d);",
    },
    Gf8Intrinsic {
        name: "_mm_add_ss",
        technology: "SSE",
        header: "intrin.h",
        prototype: "__m128 _mm_add_ss(__m128, __m128);",
    },
    Gf8Intrinsic {
        name: "_mm_adds_epi16",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128i _mm_adds_epi16(__m128i, __m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_adds_epi8",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128i _mm_adds_epi8(__m128i, __m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_addsub_pd",
        technology: "SSE3",
        header: "intrin.h",
        prototype: "__m128d _mm_addsub_pd(__m128d, __m128d);",
    },
    Gf8Intrinsic {
        name: "_mm_addsub_ps",
        technology: "SSE3",
        header: "intrin.h",
        prototype: "__m128 _mm_addsub_ps(__m128, __m128);",
    },
    Gf8Intrinsic {
        name: "_mm_aesdec_si128",
        technology: "AESNI",
        header: "immintrin.h",
        prototype: "__m128i _mm_aesdec_si128(__m128i, __m128i);",
    },
    Gf8Intrinsic {
        name: "_mm256_add_pd",
        technology: "AVX",
        header: "immintrin.h",
        prototype: "__m256d _mm256_add_pd(__m256d, __m256d);",
    },
    Gf8Intrinsic {
        name: "_mm256_add_ps",
        technology: "AVX",
        header: "immintrin.h",
        prototype: "__m256 _mm256_add_ps(__m256, __m256);",
    },
    Gf8Intrinsic {
        name: "_mm256_sub_ps",
        technology: "AVX",
        header: "immintrin.h",
        prototype: "__m256 _mm256_sub_ps(__m256, __m256);",
    },
    Gf8Intrinsic {
        name: "_mm256_dp_ps",
        technology: "AVX",
        header: "immintrin.h",
        prototype: "__m256 _mm256_dp_ps(__m256, __m256, const int);",
    },
    // Add only essential AVX intrinsics for gf8
];

/// Look up an intrinsic by exact name (e.g. "_mm256_add_ps").
pub fn find_intrinsic_by_name(name: &str) -> Option<&'static Gf8Intrinsic> {
    GF8_INTRINSICS.iter().find(|i| i.name == name)
}

/// All intrinsics for a given technology (e.g. "AVX2", "AVX-512F").
pub fn intrinsics_by_technology(tech: &str) -> impl Iterator<Item = &'static Gf8Intrinsic> {
    GF8_INTRINSICS.iter().filter(move |i| i.technology == tech)
}

/// All intrinsics that look like f32 SIMD of a particular width (128/256/512).
pub fn intrinsics_for_f32_width(width_bits: u32) -> impl Iterator<Item = &'static Gf8Intrinsic> {
    GF8_INTRINSICS
        .iter()
        .filter(move |i| i.is_f32_vector() && i.simd_width_bits() == Some(width_bits))
}

/// All intrinsics that look like f64 SIMD of a particular width (128/256/512).
pub fn intrinsics_for_f64_width(width_bits: u32) -> impl Iterator<Item = &'static Gf8Intrinsic> {
    GF8_INTRINSICS
        .iter()
        .filter(move |i| i.is_f64_vector() && i.simd_width_bits() == Some(width_bits))
}
