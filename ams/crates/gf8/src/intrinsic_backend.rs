/* e8/gf8/src/intrinsic_backend.rs */
//! Intrinsic-driven backend for GF8 operations with runtime dispatch and optimization.
//!
//! # e8 Primitives – Gf8 Intrinsic Backend
//!▫~•◦-----------------------------------------------‣
//!
//! This module provides a sophisticated backend that uses the intrinsic registry
//! to dynamically select optimal CPU instructions based on:
//! - Available hardware features (AVX, AVX2, FMA, etc.)
//! - Precision requirements (f32 vs f64)
//! - Operation characteristics (vector width, latency, throughput)
//!
//! ### Key Capabilities
//! - **Runtime Intrinsic Selection:** Choose optimal intrinsics from the registry
//! - **Performance-Aware Dispatch:** Prioritize low-latency, high-throughput instructions
//! - **Fallback Chains:** Graceful degradation from advanced to basic instructions
//! - **Architecture Portability:** x86_64, ARM64, and generic scalar fallbacks
//!
//! ### Architectural Notes
//! This is the "brain" of the GF8 SIMD system. It queries the intrinsic registry
//! to make intelligent decisions about which CPU instructions to use, rather than
//! hard-coding specific intrinsics. This makes the system more maintainable and
//! future-proof as new instruction sets emerge.
//!
//! The backend implements a priority system:
//! 1. **Optimal:** Latest instruction set with best performance characteristics
//! 2. **Compatible:** Older but widely supported instructions
//! 3. **Scalar:** Generic fallback for any platform
//!
//! ### Example
//! ```rust
//! use gf8::{Gf8, intrinsic_add, intrinsic_dot};
//!
//! let a = Gf8::from_scalar(1.0);
//! let b = Gf8::from_scalar(-0.5);
//!
//! // Automatically selects best available intrinsic
//! let sum = intrinsic_add(&a, &b);
//! let dot = intrinsic_dot(&a, &b);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::{Gf8, Gf8Intrinsic, intrinsics_for_f32_width};
use std::collections::HashMap;

/// Performance characteristics for instruction selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntrinsicMetrics {
    /// Relative latency (lower is better)
    pub latency: f32,
    /// Relative throughput (higher is better)
    pub throughput: f32,
    /// Whether the instruction supports FMA (fused multiply-add)
    pub supports_fma: bool,
    /// Whether the instruction is vectorizable
    pub is_vectorizable: bool,
}

/// Pre-computed performance metrics for common intrinsics
const INTRINSIC_PERFORMANCE: &[(&str, IntrinsicMetrics)] = &[
    // AVX2+FMA optimized instructions (best performance)
    (
        "_mm256_fmadd_ps",
        IntrinsicMetrics {
            latency: 4.0,
            throughput: 2.0,
            supports_fma: true,
            is_vectorizable: true,
        },
    ),
    (
        "_mm256_fmsub_ps",
        IntrinsicMetrics {
            latency: 4.0,
            throughput: 2.0,
            supports_fma: true,
            is_vectorizable: true,
        },
    ),
    // AVX2 instructions (very good performance)
    (
        "_mm256_add_ps",
        IntrinsicMetrics {
            latency: 3.0,
            throughput: 2.0,
            supports_fma: false,
            is_vectorizable: true,
        },
    ),
    (
        "_mm256_sub_ps",
        IntrinsicMetrics {
            latency: 3.0,
            throughput: 2.0,
            supports_fma: false,
            is_vectorizable: true,
        },
    ),
    (
        "_mm256_mul_ps",
        IntrinsicMetrics {
            latency: 4.0,
            throughput: 2.0,
            supports_fma: false,
            is_vectorizable: true,
        },
    ),
    // AVX instructions (good performance)
    (
        "_mm256_dp_ps",
        IntrinsicMetrics {
            latency: 10.0,
            throughput: 1.0,
            supports_fma: false,
            is_vectorizable: true,
        },
    ),
    (
        "_mm256_hadd_ps",
        IntrinsicMetrics {
            latency: 5.0,
            throughput: 1.0,
            supports_fma: false,
            is_vectorizable: true,
        },
    ),
    // SSE instructions (acceptable performance)
    (
        "_mm_add_ps",
        IntrinsicMetrics {
            latency: 3.0,
            throughput: 1.0,
            supports_fma: false,
            is_vectorizable: true,
        },
    ),
    (
        "_mm_mul_ps",
        IntrinsicMetrics {
            latency: 4.0,
            throughput: 1.0,
            supports_fma: false,
            is_vectorizable: true,
        },
    ),
];

/// Backend configuration for instruction selection
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Preferred instruction sets (in priority order)
    pub preferred_isas: Vec<String>,
    /// Whether to prefer FMA instructions when available
    pub prefer_fma: bool,
    /// Performance threshold for acceptable instructions
    pub min_throughput: f32,
    /// Maximum acceptable latency
    pub max_latency: f32,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            preferred_isas: vec![
                "AVX2".to_string(),
                "AVX".to_string(),
                "SSE4.1".to_string(),
                "SSE2".to_string(),
            ],
            prefer_fma: true,
            min_throughput: 1.0,
            max_latency: 10.0,
        }
    }
}

/// Intrinsic backend with runtime dispatch
pub struct IntrinsicBackend {
    config: BackendConfig,
    /// Cache of selected intrinsics for each operation
    operation_cache: HashMap<String, Option<&'static Gf8Intrinsic>>,
}

impl IntrinsicBackend {
    pub fn new(config: BackendConfig) -> Self {
        Self {
            config,
            operation_cache: HashMap::new(),
        }
    }

    /// Select the best intrinsic for a given operation
    pub fn select_intrinsic(
        &mut self,
        operation: &str,
        width_bits: u32,
    ) -> Option<&'static Gf8Intrinsic> {
        // Check cache first
        let cache_key = format!("{}:{}", operation, width_bits);
        if let Some(cached) = self.operation_cache.get(&cache_key) {
            return *cached;
        }

        let candidates = intrinsics_for_f32_width(width_bits)
            .filter(|intrinsic| {
                // Check if this instruction supports the requested operation
                let matches_op = match operation {
                    "add" => intrinsic.name.contains("_add_"),
                    "sub" => intrinsic.name.contains("_sub_"),
                    "mul" => intrinsic.name.contains("_mul_"),
                    "dot" => intrinsic.name.contains("_dp_") || intrinsic.name.contains("_hadd"),
                    "fma" => {
                        intrinsic.name.contains("_fmadd_") || intrinsic.name.contains("_fmsub_")
                    }
                    _ => true,
                };

                if !matches_op {
                    return false;
                }

                // Check if instruction set is preferred
                let is_preferred = self
                    .config
                    .preferred_isas
                    .iter()
                    .any(|isa| intrinsic.technology == *isa);

                // Check performance thresholds
                if let Some(metrics) = self.get_metrics(intrinsic.name) {
                    metrics.throughput >= self.config.min_throughput
                        && metrics.latency <= self.config.max_latency
                } else {
                    // Unknown instruction, accept if from preferred ISA
                    is_preferred
                }
            })
            .collect::<Vec<_>>();

        let selected = self.rank_intrinsics(candidates);

        self.operation_cache.insert(cache_key, selected);
        selected
    }

    /// Rank intrinsics by performance and preference
    fn rank_intrinsics(
        &self,
        mut candidates: Vec<&'static Gf8Intrinsic>,
    ) -> Option<&'static Gf8Intrinsic> {
        if candidates.is_empty() {
            return None;
        }

        // Sort by preference and performance
        candidates.sort_by(|a, b| {
            use std::cmp::Ordering;

            // Primary: ISA preference
            let a_pref = self
                .config
                .preferred_isas
                .iter()
                .position(|isa| a.technology == *isa)
                .unwrap_or(usize::MAX);
            let b_pref = self
                .config
                .preferred_isas
                .iter()
                .position(|isa| b.technology == *isa)
                .unwrap_or(usize::MAX);

            let isa_ordering = a_pref.cmp(&b_pref);
            if isa_ordering != Ordering::Equal {
                return isa_ordering;
            }

            // Secondary: Performance metrics
            if let (Some(a_metrics), Some(b_metrics)) =
                (self.get_metrics(a.name), self.get_metrics(b.name))
            {
                // Prefer FMA if configured
                if self.config.prefer_fma {
                    let a_fma = a_metrics.supports_fma.cmp(&b_metrics.supports_fma);
                    if a_fma != Ordering::Equal {
                        return a_fma.reverse(); // Prefer FMA (true > false)
                    }
                }

                // Prefer higher throughput
                b_metrics
                    .throughput
                    .partial_cmp(&a_metrics.throughput)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| {
                        // Break ties with lower latency
                        a_metrics
                            .latency
                            .partial_cmp(&b_metrics.latency)
                            .unwrap_or(Ordering::Equal)
                    })
            } else {
                Ordering::Equal
            }
        });

        candidates.first().copied()
    }

    /// Get performance metrics for an intrinsic name
    fn get_metrics(&self, name: &str) -> Option<IntrinsicMetrics> {
        INTRINSIC_PERFORMANCE
            .iter()
            .find(|(intrinsic_name, _)| *intrinsic_name == name)
            .map(|(_, metrics)| *metrics)
    }

    /// Check if a specific instruction is available on this CPU
    pub fn is_intrinsic_available(&self, intrinsic: &Gf8Intrinsic) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            match intrinsic.technology {
                "AVX2" => is_x86_feature_detected!("avx2"),
                "AVX" => is_x86_feature_detected!("avx"),
                "FMA" => is_x86_feature_detected!("fma"),
                "SSE4.1" => is_x86_feature_detected!("sse4.1"),
                "SSE2" => is_x86_feature_detected!("sse2"),
                _ => false,
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Get the best available intrinsic for addition
    pub fn get_add_intrinsic(&mut self, width_bits: u32) -> Option<&'static Gf8Intrinsic> {
        self.select_intrinsic("add", width_bits)
    }

    /// Get the best available intrinsic for subtraction
    pub fn get_sub_intrinsic(&mut self, width_bits: u32) -> Option<&'static Gf8Intrinsic> {
        self.select_intrinsic("sub", width_bits)
    }

    /// Get the best available intrinsic for multiplication
    pub fn get_mul_intrinsic(&mut self, width_bits: u32) -> Option<&'static Gf8Intrinsic> {
        self.select_intrinsic("mul", width_bits)
    }

    /// Get the best available intrinsic for dot product
    pub fn get_dot_intrinsic(&mut self, width_bits: u32) -> Option<&'static Gf8Intrinsic> {
        // Prefer FMA-based dot product if available
        if self.config.prefer_fma
            && let Some(fma_intrinsic) = self.select_intrinsic("fma", width_bits)
            && self.is_intrinsic_available(fma_intrinsic)
        {
            return Some(fma_intrinsic);
        }
        self.select_intrinsic("dot", width_bits)
    }
}

// Global backend instance (lazy-initialized)
lazy_static::lazy_static! {
    static ref GLOBAL_BACKEND: std::sync::Mutex<IntrinsicBackend> =
        std::sync::Mutex::new(IntrinsicBackend::new(BackendConfig::default()));
}

/// Get the global backend instance
fn get_backend() -> std::sync::MutexGuard<'static, IntrinsicBackend> {
    GLOBAL_BACKEND.lock().unwrap()
}

/// High-level functions that use the intrinsic backend/// Add two Gf8 values using the best available intrinsic
pub fn intrinsic_add(a: &Gf8, b: &Gf8) -> Gf8 {
    let mut backend = get_backend();
    let intrinsic = backend.get_add_intrinsic(256);

    if let Some(intrin) = intrinsic
        && backend.is_intrinsic_available(intrin)
    {
        return unsafe { gf8_add_with_intrinsic(a, b, intrin) };
    }

    // Fallback to scalar
    *a + *b
}

/// Subtract two Gf8 values using the best available intrinsic
pub fn intrinsic_sub(a: &Gf8, b: &Gf8) -> Gf8 {
    let mut backend = get_backend();
    let intrinsic = backend.get_sub_intrinsic(256);

    if let Some(intrin) = intrinsic
        && backend.is_intrinsic_available(intrin)
    {
        return unsafe { gf8_sub_with_intrinsic(a, b, intrin) };
    }

    // Fallback to scalar
    *a - *b
}

/// Compute dot product using the best available intrinsic
pub fn intrinsic_dot(a: &Gf8, b: &Gf8) -> f32 {
    let mut backend = get_backend();
    let intrinsic = backend.get_dot_intrinsic(256);

    if let Some(intrin) = intrinsic
        && backend.is_intrinsic_available(intrin)
    {
        return unsafe { gf8_dot_with_intrinsic(a, b, intrin) };
    }

    // Fallback to scalar
    a.dot(b.coords())
}

/// Unsafe implementation of addition using a specific intrinsic
#[cfg(target_arch = "x86_64")]
unsafe fn gf8_add_with_intrinsic(a: &Gf8, b: &Gf8, intrinsic: &Gf8Intrinsic) -> Gf8 {
    use std::arch::x86_64::*;

    unsafe {
        let va = _mm256_loadu_ps(a.coords().as_ptr());
        let vb = _mm256_loadu_ps(b.coords().as_ptr());

        let result = if intrinsic.name == "_mm256_add_ps" {
            if is_x86_feature_detected!("avx") {
                _mm256_add_ps(va, vb)
            } else {
                return *a + *b; // Fallback
            }
        } else {
            // Unknown addition intrinsic, fallback
            return *a + *b;
        };

        let mut result_coords = [0.0f32; 8];
        _mm256_storeu_ps(result_coords.as_mut_ptr(), result);
        Gf8::new(result_coords)
    }
}

/// Unsafe implementation of subtraction using a specific intrinsic
#[cfg(target_arch = "x86_64")]
unsafe fn gf8_sub_with_intrinsic(a: &Gf8, b: &Gf8, intrinsic: &Gf8Intrinsic) -> Gf8 {
    use std::arch::x86_64::*;

    unsafe {
        let va = _mm256_loadu_ps(a.coords().as_ptr());
        let vb = _mm256_loadu_ps(b.coords().as_ptr());

        let result = if intrinsic.name == "_mm256_sub_ps" {
            if is_x86_feature_detected!("avx") {
                _mm256_sub_ps(va, vb)
            } else {
                return *a - *b; // Fallback
            }
        } else {
            // Unknown subtraction intrinsic, fallback
            return *a - *b;
        };

        let mut result_coords = [0.0f32; 8];
        _mm256_storeu_ps(result_coords.as_mut_ptr(), result);
        Gf8::new(result_coords)
    }
}

/// Unsafe implementation of dot product using a specific intrinsic
#[cfg(target_arch = "x86_64")]
unsafe fn gf8_dot_with_intrinsic(a: &Gf8, b: &Gf8, intrinsic: &Gf8Intrinsic) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let va = _mm256_loadu_ps(a.coords().as_ptr());
        let vb = _mm256_loadu_ps(b.coords().as_ptr());

        let result = if intrinsic.name == "_mm256_fmadd_ps" {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // FMA-based dot product
                let zero = _mm256_setzero_ps();
                let prod = _mm256_fmadd_ps(va, vb, zero);

                // Horizontal sum
                let hi_128 = _mm256_extractf128_ps(prod, 1);
                let lo_128 = _mm256_castps256_ps128(prod);
                let sum_128 = _mm_add_ps(hi_128, lo_128);
                let hsum = _mm_hadd_ps(sum_128, sum_128);
                return _mm_cvtss_f32(_mm_hadd_ps(hsum, hsum));
            } else {
                // Fallback to regular multiply
                _mm256_mul_ps(va, vb)
            }
        } else if intrinsic.name == "_mm256_dp_ps" {
            if is_x86_feature_detected!("avx") {
                // Direct dot product instruction (if available)
                _mm256_dp_ps(va, vb, 0xFF)
            } else {
                // Fallback
                _mm256_mul_ps(va, vb)
            }
        } else {
            // Generic multiply fallback
            _mm256_mul_ps(va, vb)
        };

        // Horizontal sum for non-FMA paths
        let h = _mm256_hadd_ps(result, result);
        let h = _mm256_hadd_ps(h, h);
        let h = _mm256_castps256_ps128(h);
        _mm_cvtss_f32(h)
    }
}

/// Utility function to get backend information for debugging
pub fn get_backend_info() -> String {
    let backend = get_backend();
    format!(
        "GF8 Intrinsic Backend\n\
        Preferred ISAs: {:?}\n\
        Prefer FMA: {}\n\
        Cache size: {} entries",
        backend.config.preferred_isas,
        backend.config.prefer_fma,
        backend.operation_cache.len()
    )
}

/// Utility function to list available intrinsics for debugging
pub fn list_available_intrinsics() -> Vec<String> {
    let mut backend = get_backend();
    let mut results = Vec::new();

    for &op in &["add", "sub", "mul", "dot"] {
        if let Some(intrin) = backend.select_intrinsic(op, 256) {
            let available = backend.is_intrinsic_available(intrin);
            results.push(format!(
                "{}: {} ({} - {})",
                op,
                intrin.name,
                intrin.technology,
                if available {
                    "AVAILABLE"
                } else {
                    "UNAVAILABLE"
                }
            ));
        } else {
            results.push(format!("{}: NO SELECTION", op));
        }
    }

    results
}
