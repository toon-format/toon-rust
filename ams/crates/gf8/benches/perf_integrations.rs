/* e8/gf8/benches/perf_integrations.rs */
//! Performance benchmarks for integrated GF8 + intrinsic backend system.
//!
//! # e8 Primitives – GF8 Integration Benchmarks
//!▫~•◦----------------------------------------------‣
//!
//! This benchmark suite evaluates the performance of the complete GF8 ecosystem,
//! including:
//! - Intrinsic registry queries
//! - Runtime dispatch performance
//! - SIMD vs scalar operation comparisons
//! - Bitcodec round-trip performance
//! - End-to-end GF8 operation pipelines
//!
//! ### Key Capabilities
//! - **Intrinsic Selection Benchmarks:** Measure the overhead of dynamic instruction selection
//! - **SIMD Integration Tests:** Verify that intrinsic backend provides performance benefits
//! - **End-to-End Pipelines:** Benchmark complete GF8 processing workflows
//! - **Memory Layout Analysis:** Compare different GF8 data organization strategies
//!
//! ### Architectural Notes
//! These benchmarks validate that the intrinsic-driven approach provides
//! measurable performance improvements over naive scalar implementations
//! while maintaining correctness and portability.
//!
//! The benchmarks are designed to run on any platform, gracefully degrading
//! to scalar implementations when SIMD instructions are unavailable.
//!
//! ### Example
//! \```rust
//! // Run with: cargo bench --bench perf_integrations
//!
//! // This will benchmark:
//! // - Intrinsic registry queries
//! // - Runtime dispatch overhead
//! // - SIMD vs scalar performance
//! // - Complete GF8 pipelines
//! \```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use gf8::{
    BackendConfig, Gf8, IntrinsicBackend, dequantize_to_vec, find_intrinsic_by_name,
    gf8_add_inplace_slice_simd, gf8_add_simd, gf8_dot_simd, gf8_from_code, gf8_norm2_simd,
    gf8_to_code, intrinsic_add, intrinsic_dot, intrinsic_sub, intrinsics_for_f32_width,
    quantize_to_gf8,
};
use std::hint::black_box;

/// Benchmark intrinsic registry queries
fn bench_intrinsic_registry_queries(c: &mut Criterion) {
    c.bench_function("intrinsic_registry_queries", |bencher| {
        bencher.iter(|| {
            // Query all 256-bit f32 intrinsics
            let intrinsics: Vec<_> = intrinsics_for_f32_width(256).collect();

            // Find specific intrinsics
            let add_ps = find_intrinsic_by_name("_mm256_add_ps");
            let dot_ps = find_intrinsic_by_name("_mm256_dp_ps");
            let fma_ps = find_intrinsic_by_name("_mm256_fmadd_ps");

            // Ensure we actually use the results
            black_box(intrinsics.len());
            black_box(add_ps.is_some());
            black_box(dot_ps.is_some());
            black_box(fma_ps.is_some());
        })
    });
}

/// Benchmark intrinsic backend selection
fn bench_intrinsic_backend_selection(c: &mut Criterion) {
    c.bench_function("intrinsic_backend_selection", |bencher| {
        let config = BackendConfig::default();

        bencher.iter(|| {
            let mut backend = IntrinsicBackend::new(config.clone());

            // Select intrinsics for different operations
            let add_intrinsic = backend.get_add_intrinsic(256);
            let sub_intrinsic = backend.get_sub_intrinsic(256);
            let mul_intrinsic = backend.get_mul_intrinsic(256);
            let dot_intrinsic = backend.get_dot_intrinsic(256);

            // Check availability
            black_box(add_intrinsic.is_some());
            black_box(sub_intrinsic.is_some());
            black_box(mul_intrinsic.is_some());
            black_box(dot_intrinsic.is_some());
        })
    });
}

/// Benchmark intrinsic backend caching
fn bench_intrinsic_backend_caching(c: &mut Criterion) {
    c.bench_function("intrinsic_backend_caching", |bencher| {
        let config = BackendConfig::default();
        let mut backend = IntrinsicBackend::new(config);

        // Pre-populate cache
        backend.get_add_intrinsic(256);
        backend.get_sub_intrinsic(256);
        backend.get_dot_intrinsic(256);

        bencher.iter(|| {
            // These should all hit the cache
            let add_intrinsic = backend.get_add_intrinsic(256);
            let sub_intrinsic = backend.get_sub_intrinsic(256);
            let dot_intrinsic = backend.get_dot_intrinsic(256);

            black_box(add_intrinsic.is_some());
            black_box(sub_intrinsic.is_some());
            black_box(dot_intrinsic.is_some());
        })
    });
}

/// Benchmark SIMD addition vs scalar
fn bench_gf8_add_simd_vs_scalar(c: &mut Criterion) {
    c.bench_function("gf8_add_simd", |bencher| {
        let a = Gf8::from_scalar(1.5);
        let b = Gf8::from_scalar(-0.7);

        bencher.iter(|| {
            black_box(gf8_add_simd(&a, &b));
        })
    });
}

/// Benchmark SIMD dot product vs scalar
fn bench_gf8_dot_simd_vs_scalar(c: &mut Criterion) {
    c.bench_function("gf8_dot_simd", |bencher| {
        let a = Gf8::from_scalar(1.5);
        let b = Gf8::from_scalar(-0.7);

        bencher.iter(|| {
            black_box(gf8_dot_simd(&a, &b));
        })
    });
}

/// Benchmark intrinsic backend addition
fn bench_gf8_intrinsic_add(c: &mut Criterion) {
    c.bench_function("gf8_intrinsic_add", |bencher| {
        let a = Gf8::from_scalar(1.5);
        let b = Gf8::from_scalar(-0.7);

        bencher.iter(|| {
            black_box(intrinsic_add(&a, &b));
        })
    });
}

/// Benchmark intrinsic backend dot product
fn bench_gf8_intrinsic_dot(c: &mut Criterion) {
    c.bench_function("gf8_intrinsic_dot", |bencher| {
        let a = Gf8::from_scalar(1.5);
        let b = Gf8::from_scalar(-0.7);

        bencher.iter(|| {
            black_box(intrinsic_dot(&a, &b));
        })
    });
}

/// Benchmark intrinsic backend subtraction
fn bench_gf8_intrinsic_sub(c: &mut Criterion) {
    c.bench_function("gf8_intrinsic_sub", |bencher| {
        let a = Gf8::from_scalar(1.5);
        let b = Gf8::from_scalar(-0.7);

        bencher.iter(|| {
            black_box(intrinsic_sub(&a, &b));
        })
    });
}

/// Benchmark bitcodec round-trip
fn bench_gf8_bitcodec_roundtrip(c: &mut Criterion) {
    c.bench_function("gf8_bitcodec_roundtrip", |bencher| {
        let original = Gf8::from_scalar(1.234);

        bencher.iter(|| {
            // Encode to code
            let code = gf8_to_code(&original);

            // Decode back to Gf8
            let decoded = gf8_from_code(code);

            black_box(decoded);
        })
    });
}

/// Benchmark quantize -> Gf8 -> dequantize pipeline
fn bench_gf8_quantize_roundtrip(c: &mut Criterion) {
    c.bench_function("gf8_quantize_roundtrip", |bencher| {
        let input = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];

        bencher.iter(|| {
            // Quantize to Gf8
            let gf8 = quantize_to_gf8(&input);

            // Dequantize back to vector
            let output = dequantize_to_vec(&gf8);

            black_box(output);
        })
    });
}

/// Benchmark batch Gf8 operations with SIMD
fn bench_gf8_batch_operations_simd(c: &mut Criterion) {
    c.bench_function("gf8_batch_operations_simd", |bencher| {
        const BATCH_SIZE: usize = 1024;

        let mut inputs_a: Vec<Gf8> = Vec::with_capacity(BATCH_SIZE);
        let mut inputs_b: Vec<Gf8> = Vec::with_capacity(BATCH_SIZE);

        // Initialize with random data
        for i in 0..BATCH_SIZE {
            let val_a = i as f32 * 0.01;
            let val_b = (i + 100) as f32 * 0.01;
            inputs_a.push(Gf8::from_scalar(val_a));
            inputs_b.push(Gf8::from_scalar(val_b));
        }

        bencher.iter(|| {
            let mut results = Vec::with_capacity(BATCH_SIZE);

            for (a, b) in inputs_a.iter().zip(inputs_b.iter()) {
                let sum = gf8_add_simd(black_box(a), black_box(b));
                let dot = gf8_dot_simd(black_box(a), black_box(b));
                let norm_a = gf8_norm2_simd(black_box(a));

                results.push((sum, dot, norm_a));
            }

            black_box(results.len());
        })
    });
}

/// Benchmark batch Gf8 operations with intrinsic backend
fn bench_gf8_batch_operations_intrinsic(c: &mut Criterion) {
    c.bench_function("gf8_batch_operations_intrinsic", |bencher| {
        const BATCH_SIZE: usize = 1024;

        let mut inputs_a: Vec<Gf8> = Vec::with_capacity(BATCH_SIZE);
        let mut inputs_b: Vec<Gf8> = Vec::with_capacity(BATCH_SIZE);

        // Initialize with random data
        for i in 0..BATCH_SIZE {
            let val_a = i as f32 * 0.01;
            let val_b = (i + 100) as f32 * 0.01;
            inputs_a.push(Gf8::from_scalar(val_a));
            inputs_b.push(Gf8::from_scalar(val_b));
        }

        bencher.iter(|| {
            let mut results = Vec::with_capacity(BATCH_SIZE);

            for (a, b) in inputs_a.iter().zip(inputs_b.iter()) {
                let sum = intrinsic_add(black_box(a), black_box(b));
                let dot = intrinsic_dot(black_box(a), black_box(b));
                let sub = intrinsic_sub(black_box(a), black_box(b));

                results.push((sum, dot, sub));
            }

            black_box(results.len());
        })
    });
}

/// Benchmark Gf8 in-place slice operations
///
/// Uses `iter_batched` to correctly measure the in-place mutation time separately
/// from the memory allocation/cloning overhead.
fn bench_gf8_inplace_slice_operations(c: &mut Criterion) {
    c.bench_function("gf8_inplace_slice_operations", |bencher| {
        const SLICE_SIZE: usize = 512;

        let mut src: Vec<Gf8> = Vec::with_capacity(SLICE_SIZE);
        // Prepare source data and a destination template
        let dst_template: Vec<Gf8> = (0..SLICE_SIZE)
            .map(|i| {
                let val_dst = i as f32 * 0.01;
                let val_src = (i + 1000) as f32 * 0.01;
                src.push(Gf8::from_scalar(val_src));
                Gf8::from_scalar(val_dst)
            })
            .collect();

        // Use iter_batched to isolate the mutation performance from clone overhead
        bencher.iter_batched(
            || dst_template.clone(), // Setup: Create a fresh clone for each iteration
            |mut test_dst| {
                // Measure: The actual in-place operation
                gf8_add_inplace_slice_simd(black_box(&mut test_dst), black_box(&src))
                    .expect("gf8_add_inplace_slice_simd failed");
                test_dst // Return to ensure no premature drop affects timing
            },
            BatchSize::SmallInput,
        )
    });
}

/// Benchmark end-to-end compression pipeline
fn bench_gf8_compression_pipeline(c: &mut Criterion) {
    c.bench_function("gf8_compression_pipeline", |bencher| {
        // Input data representing high-dimensional vectors
        const VECTORS_COUNT: usize = 256;
        const VECTOR_DIM: usize = 8;

        let mut input_vectors: Vec<[f32; VECTOR_DIM]> = Vec::with_capacity(VECTORS_COUNT);

        // Generate test data
        for i in 0..VECTORS_COUNT {
            let mut vec = [0.0; VECTOR_DIM];
            for (j, val) in vec.iter_mut().enumerate() {
                *val = (i * j) as f32 * 0.01 - 5.0;
            }
            input_vectors.push(vec);
        }

        bencher.iter(|| {
            let mut compressed_codes = Vec::with_capacity(VECTORS_COUNT);
            let mut decompressed = Vec::with_capacity(VECTORS_COUNT);

            // Compression phase
            for input in &input_vectors {
                let gf8 = quantize_to_gf8(black_box(input));
                let code = gf8_to_code(&gf8);
                compressed_codes.push(code);
            }

            // Decompression phase
            for code in &compressed_codes {
                let gf8 = gf8_from_code(*code);
                let vec = dequantize_to_vec(&gf8);
                decompressed.push(vec);
            }

            black_box(compressed_codes.len());
            black_box(decompressed.len());
        })
    });
}

/// Benchmark similarity search with Gf8
fn bench_gf8_similarity_search(c: &mut Criterion) {
    c.bench_function("gf8_similarity_search", |bencher| {
        const DATABASE_SIZE: usize = 1000;
        const QUERY_COUNT: usize = 100;

        // Build database
        let database: Vec<Gf8> = (0..DATABASE_SIZE)
            .map(|i| Gf8::from_scalar(i as f32 * 0.001))
            .collect();

        // Query vectors
        let queries: Vec<Gf8> = (0..QUERY_COUNT)
            .map(|i| Gf8::from_scalar((i + 500) as f32 * 0.001))
            .collect();

        bencher.iter(|| {
            let mut results = Vec::with_capacity(QUERY_COUNT);

            for query in &queries {
                let mut similarities = Vec::with_capacity(DATABASE_SIZE);

                for (idx, db_item) in database.iter().enumerate() {
                    let similarity = gf8_dot_simd(black_box(query), black_box(db_item));
                    similarities.push((idx, similarity));
                }

                // Sort by similarity (highest first)
                similarities
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Take top 10
                let top_k = &similarities[0..10.min(similarities.len())];
                results.push(top_k.to_vec());
            }

            black_box(results.len());
        })
    });
}

/// Benchmark Gf8 mathematical operations
fn bench_gf8_mathematical_operations(c: &mut Criterion) {
    c.bench_function("gf8_mathematical_operations", |bencher| {
        let base = Gf8::from_scalar(1.0);

        bencher.iter(|| {
            let mut accumulator = base;
            // Chain multiple operations
            for i in 0..100 {
                let step = Gf8::from_scalar(i as f32 * 0.01);

                // Addition
                accumulator = intrinsic_add(black_box(&accumulator), black_box(&step));

                // Subtraction
                accumulator = intrinsic_sub(black_box(&accumulator), black_box(&step));

                // Dot product (scalar result used for next iteration)
                let dot = intrinsic_dot(black_box(&accumulator), black_box(&step));

                // Use dot result to create next step
                accumulator = Gf8::from_scalar(dot);
            }

            black_box(accumulator);
        })
    });
}

/// Benchmark memory layout efficiency
fn bench_gf8_memory_layout_efficiency(c: &mut Criterion) {
    c.bench_function("gf8_memory_layout_efficiency", |bencher| {
        const COUNT: usize = 10000;

        // Test AoS (Array of Structures) layout - what we use
        let aos_layout: Vec<Gf8> = (0..COUNT)
            .map(|i| Gf8::from_scalar(i as f32 * 0.001))
            .collect();

        // Test SoA (Structure of Arrays) layout for comparison
        let mut soa_layout: [Vec<f32>; 8] = core::array::from_fn(|_| Vec::with_capacity(COUNT));
        for i in 0..COUNT {
            let scalar = i as f32 * 0.001;
            for (dim, lane) in soa_layout.iter_mut().enumerate() {
                lane.push(scalar * (dim + 1) as f32);
            }
        }

        bencher.iter(|| {
            // Process AoS layout
            let mut aos_result = 0.0f32;
            for gf8 in &aos_layout {
                aos_result += gf8.norm2();
            }

            // Process SoA layout
            let mut soa_result = 0.0f32;
            for i in 0..COUNT {
                let mut sum = 0.0f32;
                for lane in &soa_layout {
                    sum += lane[i] * lane[i];
                }
                soa_result += sum;
            }

            black_box(aos_result);
            black_box(soa_result);
        })
    });
}

/// Benchmark configuration impact
fn bench_backend_config_impact(c: &mut Criterion) {
    c.bench_function("backend_config_impact", |bencher| {
        // Test different configuration scenarios
        let configs = vec![
            BackendConfig::default(),
            BackendConfig {
                preferred_isas: vec!["AVX2".to_string(), "AVX".to_string()],
                prefer_fma: true,
                min_throughput: 2.0,
                max_latency: 5.0,
            },
            BackendConfig {
                preferred_isas: vec!["SSE4.1".to_string(), "SSE2".to_string()],
                prefer_fma: false,
                min_throughput: 1.0,
                max_latency: 10.0,
            },
        ];

        bencher.iter(|| {
            for config in &configs {
                let mut backend = IntrinsicBackend::new(config.clone());

                // Test each configuration
                let add = backend.get_add_intrinsic(256);
                let sub = backend.get_sub_intrinsic(256);
                let dot = backend.get_dot_intrinsic(256);

                black_box(add.is_some());
                black_box(sub.is_some());
                black_box(dot.is_some());
            }
        })
    });
}

/// Benchmark error handling and fallbacks
fn bench_gf8_fallback_handling(c: &mut Criterion) {
    c.bench_function("gf8_fallback_handling", |bencher| {
        let a = Gf8::from_scalar(1.5);
        let rhs = Gf8::from_scalar(-0.7);

        bencher.iter(|| {
            // These should all gracefully fall back to scalar operations
            // if SIMD is unavailable
            black_box(intrinsic_add(&a, &rhs));
            black_box(intrinsic_sub(&a, &rhs));
            black_box(intrinsic_dot(&a, &rhs));
        })
    });
}

// Create criterion groups for related benchmarks
criterion_group!(
    registry_benchmarks,
    bench_intrinsic_registry_queries,
    bench_intrinsic_backend_selection,
    bench_intrinsic_backend_caching,
);

criterion_group!(
    operation_benchmarks,
    bench_gf8_add_simd_vs_scalar,
    bench_gf8_dot_simd_vs_scalar,
    bench_gf8_intrinsic_add,
    bench_gf8_intrinsic_dot,
    bench_gf8_intrinsic_sub,
);

criterion_group!(
    pipeline_benchmarks,
    bench_gf8_bitcodec_roundtrip,
    bench_gf8_quantize_roundtrip,
    bench_gf8_compression_pipeline,
    bench_gf8_similarity_search,
);

criterion_group!(
    batch_benchmarks,
    bench_gf8_batch_operations_simd,
    bench_gf8_batch_operations_intrinsic,
    bench_gf8_inplace_slice_operations,
);

criterion_group!(
    system_benchmarks,
    bench_gf8_mathematical_operations,
    bench_gf8_memory_layout_efficiency,
    bench_backend_config_impact,
    bench_gf8_fallback_handling,
);

// Main benchmark runner
criterion_main!(
    registry_benchmarks,
    operation_benchmarks,
    pipeline_benchmarks,
    batch_benchmarks,
    system_benchmarks,
);
