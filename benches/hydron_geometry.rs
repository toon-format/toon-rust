/* benches/hydron_geometry.rs */
//! Hydron geometric layer performance benchmarks.
//!
//! # TOON-RUNE – Hydron Geometry Benchmarks
//!▫~•◦---------------------------------------‣
//!
//! This benchmark suite measures performance of E8 geometric operations across
//! different mathematical geometries (spherical, hyperbolic, symplectic, etc.).
//!
//! ### Key Capabilities
//! - **Fisher Information**: Statistical geometry and information metrics.
//! - **Spherical S7 Operations**: Projection, distance, slerp interpolation.
//! - **Quaternion Algebra**: Multiplication, normalization, SLERP operations.
//! - **Hyperbolic Geometry**: Möbius transformations and geodesic operations.
//! - **Symplectic Mechanics**: Hamiltonian evolution and phase space operations.
//! - **Topological Analysis**: Betti numbers, persistence diagrams, signatures.
//! - **Lorentzian Geometry**: Minkowski metrics and causal structure analysis.
//!
//! ### Architectural Notes
//! Benchmarks run conditionally with `#[cfg(feature = "hydron")]` and use Criterion
//! for statistical rigor. Performance results guide optimization of geometric
//! primitives for E8 ecosystem simulations.
//!
//! ### Example
//! ```bash
//! cargo bench --features hydron --bench hydron_geometry
//! ```
//!
//! Results help identify bottlenecks in geometric computations for real-time E8 applications.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#[cfg(feature = "hydron")]
use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "hydron")]
mod hydron_benches {
    use super::*;
    use rune_format::rune::hydron::{
        FisherLayer, Gf8, HyperbolicLayer, LorentzianCausalLayer, LorentzianLayer, QuaternionOps,
        SpacetimePoint, SphericalLayer, SymplecticLayer, TopologicalLayer,
    };

    pub fn bench_spherical_ops(c: &mut Criterion) {
        let mut group = c.benchmark_group("spherical_s7");

        let v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Benchmark with current configuration (SIMD if enabled)
        group.bench_function("distance", |b| {
            b.iter(|| {
                SphericalLayer::distance(std::hint::black_box(&v1), std::hint::black_box(&v2))
            });
        });

        group.bench_function("slerp", |b| {
            b.iter(|| {
                SphericalLayer::slerp(
                    std::hint::black_box(&v1),
                    std::hint::black_box(&v2),
                    std::hint::black_box(0.5),
                )
            });
        });

        // Scalar fallback for comparison
        group.bench_function("distance_scalar_fallback", |b| {
            b.iter(|| {
                let dot: f32 = v1.iter().zip(v2.iter()).map(|(xi, yi)| xi * yi).sum();
                let dot_clamped = dot.clamp(-1.0, 1.0);
                dot_clamped.acos()
            });
        });

        group.bench_function("project", |b| {
            b.iter(|| SphericalLayer::project(std::hint::black_box(&v1)));
        });

        group.bench_function("normalized_entropy", |b| {
            b.iter(|| SphericalLayer::normalized_entropy(std::hint::black_box(&v1)));
        });

        group.bench_function("mean_multiple_points", |b| {
            let points = vec![v1, v2, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
            b.iter(|| SphericalLayer::mean(std::hint::black_box(&points)));
        });

        group.finish();
    }

    pub fn bench_quaternion_ops(c: &mut Criterion) {
        let mut group = c.benchmark_group("quaternion");

        let q1 = [1.0, 0.0, 0.0, 0.0];
        let q2 = [0.707, 0.707, 0.0, 0.0];

        group.bench_function("multiply", |b| {
            b.iter(|| {
                QuaternionOps::multiply(std::hint::black_box(&q1), std::hint::black_box(&q2))
            });
        });

        group.bench_function("conjugate", |b| {
            b.iter(|| QuaternionOps::conjugate(std::hint::black_box(&q1)));
        });

        group.bench_function("normalize", |b| {
            b.iter(|| QuaternionOps::normalize(std::hint::black_box(&q1)));
        });

        group.bench_function("slerp", |b| {
            b.iter(|| {
                QuaternionOps::slerp(
                    std::hint::black_box(&q1),
                    std::hint::black_box(&q2),
                    std::hint::black_box(0.5),
                )
            });
        });

        group.bench_function("from_axis_angle", |b| {
            let axis = [1.0, 0.0, 0.0];
            b.iter(|| {
                QuaternionOps::from_axis_angle(
                    std::hint::black_box(&axis),
                    std::hint::black_box(1.57),
                )
            });
        });

        group.finish();
    }

    pub fn bench_hyperbolic_ops(c: &mut Criterion) {
        let mut group = c.benchmark_group("hyperbolic");

        let v1 = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Current implementation (SIMD if enabled)
        group.bench_function("project", |b| {
            b.iter(|| HyperbolicLayer::project(std::hint::black_box(&v1)));
        });

        group.bench_function("distance", |b| {
            b.iter(|| {
                HyperbolicLayer::distance(std::hint::black_box(&v1), std::hint::black_box(&v2))
            });
        });

        group.bench_function("mobius_add", |b| {
            b.iter(|| {
                HyperbolicLayer::mobius_add(std::hint::black_box(&v1), std::hint::black_box(&v2))
            });
        });

        // Scalar fallback for comparison
        group.bench_function("mobius_add_scalar_fallback", |b| {
            b.iter(|| {
                let a_norm_sq: f32 = v1.iter().map(|x| x * x).sum();
                let b_norm_sq: f32 = v2.iter().map(|x| x * x).sum();
                let dot_ab: f32 = v1.iter().zip(v2.iter()).map(|(ai, bi)| ai * bi).sum();
                let numerator_a_coeff = 1.0 + 2.0 * dot_ab + b_norm_sq;
                let numerator_b_coeff = 1.0 - a_norm_sq;
                let denominator = 1.0 + 2.0 * dot_ab + a_norm_sq * b_norm_sq;
                if denominator.abs() < 1e-8 {
                    [0.0; 8]
                } else {
                    let mut result = [0.0f32; 8];
                    for i in 0..8 {
                        result[i] =
                            (numerator_a_coeff * v1[i] + numerator_b_coeff * v2[i]) / denominator;
                    }
                    result
                }
            });
        });

        group.bench_function("norm", |b| {
            b.iter(|| HyperbolicLayer::norm(std::hint::black_box(&v1)));
        });

        group.bench_function("interpolate", |b| {
            b.iter(|| {
                HyperbolicLayer::interpolate(
                    std::hint::black_box(&v1),
                    std::hint::black_box(&v2),
                    std::hint::black_box(0.5),
                )
            });
        });

        group.finish();
    }

    pub fn bench_symplectic_ops(c: &mut Criterion) {
        let mut group = c.benchmark_group("symplectic");

        let layer = SymplecticLayer::new();
        let q = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let p = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Current implementation (SIMD if enabled)
        group.bench_function("hamiltonian", |b| {
            b.iter(|| layer.hamiltonian(std::hint::black_box(&q), std::hint::black_box(&p)));
        });

        group.bench_function("evolve_single_step", |b| {
            b.iter(|| {
                let mut q_copy = q;
                let mut p_copy = p;
                layer.evolve(
                    std::hint::black_box(&mut q_copy),
                    std::hint::black_box(&mut p_copy),
                    std::hint::black_box(0.01),
                );
            });
        });

        group.bench_function("evolve_100_steps", |b| {
            b.iter(|| {
                let mut q_copy = q;
                let mut p_copy = p;
                for _ in 0..100 {
                    layer.evolve(
                        std::hint::black_box(&mut q_copy),
                        std::hint::black_box(&mut p_copy),
                        std::hint::black_box(0.01),
                    );
                }
            });
        });

        // Scalar fallback for hamiltonian comparison
        group.bench_function("hamiltonian_scalar_fallback", |b| {
            b.iter(|| {
                let kinetic: f32 = p.iter().map(|&pi| pi * pi).sum::<f32>() * 0.5;
                let k = 0.1;
                let potential: f32 = q.iter().map(|&qi| qi * qi).sum::<f32>() * 0.5 * k;
                kinetic + potential
            });
        });

        group.finish();
    }

    pub fn bench_fisher_ops(c: &mut Criterion) {
        let mut group = c.benchmark_group("fisher_information");

        let dist1 = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0];
        let dist2 = [0.5, 0.3, 0.15, 0.05, 0.0, 0.0, 0.0, 0.0];
        let fisher_matrix = [[1.0; 8]; 8];

        group.bench_function("information_metric", |b| {
            b.iter(|| FisherLayer::information_metric(std::hint::black_box(&fisher_matrix)));
        });

        group.bench_function("fisher_distance", |b| {
            b.iter(|| {
                FisherLayer::fisher_distance(
                    std::hint::black_box(&dist1),
                    std::hint::black_box(&dist2),
                    std::hint::black_box(&fisher_matrix),
                )
            });
        });

        group.bench_function("kl_divergence", |b| {
            b.iter(|| {
                FisherLayer::kl_divergence(
                    std::hint::black_box(&dist1),
                    std::hint::black_box(&dist2),
                )
            });
        });

        group.bench_function("entropy", |b| {
            b.iter(|| FisherLayer::entropy(std::hint::black_box(&dist1)));
        });

        group.bench_function("uncertainty", |b| {
            b.iter(|| FisherLayer::uncertainty(std::hint::black_box(&fisher_matrix)));
        });

        group.finish();
    }

    pub fn bench_topological_ops(c: &mut Criterion) {
        let mut group = c.benchmark_group("topological");

        // Small point cloud (10 points)
        for i in 0..10 {
            let _p = [
                i as f32 * 0.1,
                (i as f32 * 0.1).sin(),
                (i as f32 * 0.1).cos(),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ];
            group.bench_function(&format!("betti_10_points"), |b| {
                b.iter(|| {
                    let mut layer = TopologicalLayer::new();
                    for j in 0..10 {
                        layer.add_point([
                            j as f32 * 0.1,
                            (j as f32 * 0.1).sin(),
                            (j as f32 * 0.1).cos(),
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]);
                    }
                    layer
                        .compute_betti_numbers(std::hint::black_box(2.0), std::hint::black_box(10));
                });
            });
            break; // Only benchmark once
        }

        // Medium point cloud (50 points)
        group.bench_function("betti_50_points", |b| {
            b.iter(|| {
                let mut layer = TopologicalLayer::new();
                for i in 0..50 {
                    layer.add_point([
                        i as f32 * 0.02,
                        (i as f32 * 0.1).sin(),
                        (i as f32 * 0.1).cos(),
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]);
                }
                layer.compute_betti_numbers(std::hint::black_box(2.0), std::hint::black_box(10));
            });
        });

        // Persistence diagram
        group.bench_function("persistence_diagram_50_points", |b| {
            b.iter(|| {
                let mut layer = TopologicalLayer::new();
                for i in 0..50 {
                    layer.add_point([
                        i as f32 * 0.02,
                        (i as f32 * 0.1).sin(),
                        (i as f32 * 0.1).cos(),
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]);
                }
                layer.compute_persistence_diagram_dim0(
                    std::hint::black_box(2.0),
                    std::hint::black_box(10),
                );
            });
        });

        // Signature
        group.bench_function("signature", |b| {
            b.iter(|| {
                let mut layer = TopologicalLayer::new();
                for i in 0..10 {
                    layer.add_point([
                        i as f32 * 0.1,
                        (i as f32 * 0.1).sin(),
                        (i as f32 * 0.1).cos(),
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]);
                }
                layer.signature()
            });
        });

        group.finish();
    }

    pub fn bench_lorentzian_ops(c: &mut Criterion) {
        let mut group = c.benchmark_group("lorentzian");

        let p1 = SpacetimePoint::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p2 = SpacetimePoint::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let velocity = 0.5;

        let layer = LorentzianLayer::new();

        group.bench_function("minkowski_interval", |b| {
            b.iter(|| p1.minkowski_interval(std::hint::black_box(&p2)));
        });

        group.bench_function("causal_relation", |b| {
            b.iter(|| p1.causal_relation(std::hint::black_box(&p2)));
        });

        group.bench_function("lorentz_boost", |b| {
            b.iter(|| {
                layer.lorentz_boost(std::hint::black_box(&p1), std::hint::black_box(velocity))
            });
        });

        // Causal DAG operations
        let mut dag = LorentzianCausalLayer::new();
        for i in 0..10 {
            let root_idx = i;
            dag.add_event(
                root_idx,
                EventType::Emergence {
                    concept_id: i as u64,
                },
                &[],
            );
        }

        group.bench_function("causal_dag_10_events", |b| {
            b.iter(|| {
                let mut new_dag = LorentzianCausalLayer::new();
                for i in 0..10 {
                    new_dag.add_event(
                        i,
                        EventType::Emergence {
                            concept_id: i as u64,
                        },
                        &[],
                    );
                }
                std::hint::black_box(new_dag);
            });
        });

        group.bench_function("topological_order_10_events", |b| {
            b.iter(|| dag.dag.topological_order());
        });

        group.finish();
    }

    pub fn bench_simd_gf8_ops(c: &mut Criterion) {
        let mut group = c.benchmark_group("simd_gf8");

        // Create test data - normalized 8D vectors
        let a = Gf8::new([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0]);
        let b = Gf8::new([0.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0]);

        // Benchmark Gf8 addition
        #[cfg(feature = "simd")]
        group.bench_function("add_simd", |bencher| {
            bencher.iter(|| {
                use hydron_core::gf8::gf8_add_simd;
                gf8_add_simd(std::hint::black_box(&a), std::hint::black_box(&b))
            });
        });

        group.bench_function("add_scalar", |bencher| {
            bencher.iter(|| *std::hint::black_box(&a) + *std::hint::black_box(&b));
        });

        // Benchmark Gf8 subtraction
        #[cfg(feature = "simd")]
        group.bench_function("sub_simd", |bencher| {
            bencher.iter(|| {
                use hydron_core::gf8::gf8_sub_simd;
                gf8_sub_simd(std::hint::black_box(&a), std::hint::black_box(&b))
            });
        });

        group.bench_function("sub_scalar", |bencher| {
            bencher.iter(|| *std::hint::black_box(&a) - *std::hint::black_box(&b));
        });

        // Benchmark dot product
        #[cfg(feature = "simd")]
        group.bench_function("dot_simd", |bencher| {
            bencher.iter(|| {
                use hydron_core::gf8::gf8_dot_simd;
                gf8_dot_simd(std::hint::black_box(&a), std::hint::black_box(&b))
            });
        });

        group.bench_function("dot_scalar", |bencher| {
            bencher.iter(|| a.dot(std::hint::black_box(b.coords())));
        });

        // Benchmark norm squared (which uses dot product internally)
        #[cfg(feature = "simd")]
        group.bench_function("norm2_simd", |bencher| {
            bencher.iter(|| {
                use hydron_core::gf8::gf8_norm2_simd;
                gf8_norm2_simd(std::hint::black_box(&a))
            });
        });

        group.bench_function("norm2_scalar", |bencher| {
            bencher.iter(|| a.norm2());
        });

        // Benchmark in-place addition
        #[cfg(feature = "simd")]
        group.bench_function("add_inplace_simd", |bencher| {
            bencher.iter(|| {
                use hydron_core::gf8::gf8_add_inplace_slice_simd;
                let mut dst = *std::hint::black_box(a.coords());
                gf8_add_inplace_slice_simd(&mut dst, std::hint::black_box(b.coords()));
                dst
            });
        });

        group.bench_function("add_inplace_scalar", |bencher| {
            bencher.iter(|| {
                let mut dst = *std::hint::black_box(a.coords());
                for i in 0..8 {
                    dst[i] += b.coords()[i];
                }
                dst
            });
        });

        // Benchmark matrix-vector multiplication
        #[cfg(feature = "simd")]
        group.bench_function("matvec_simd", |bencher| {
            let matrix = [[1.0; 8]; 8];
            bencher.iter(|| {
                use hydron_core::gf8::gf8_matvec_simd;
                gf8_matvec_simd(std::hint::black_box(&matrix), std::hint::black_box(&a))
            });
        });

        group.bench_function("matvec_scalar", |bencher| {
            let matrix = [[1.0; 8]; 8];
            bencher.iter(|| {
                let mut result_coords = [0.0f32; 8];
                for (i, row) in matrix.iter().enumerate() {
                    result_coords[i] = a.dot(row);
                }
                Gf8::new(result_coords)
            });
        });

        group.finish();
    }
}

#[cfg(feature = "hydron")]
criterion_group!(
    benches,
    hydron_benches::bench_spherical_ops,
    hydron_benches::bench_quaternion_ops,
    hydron_benches::bench_hyperbolic_ops,
    hydron_benches::bench_symplectic_ops,
    hydron_benches::bench_fisher_ops,
    hydron_benches::bench_topological_ops,
    hydron_benches::bench_lorentzian_ops,
    hydron_benches::bench_simd_gf8_ops
);

#[cfg(not(feature = "hydron"))]
fn dummy_bench(_c: &mut Criterion) {}

#[cfg(not(feature = "hydron"))]
criterion_group!(benches, dummy_bench);

criterion_main!(benches);
