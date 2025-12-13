/* benches/rune_operators.rs */
//! RUNE operator and value evaluation performance benchmarks.
//!
//! # TOON-RUNE – RUNE Operator Benchmarks
//!▫~•◦--------------------------------------‣
//!
//! This benchmark suite measures performance of RUNE operator evaluation,
//! value arithmetic, and built-in function execution for E8 ecosystem operations.
//!
//! ### Key Capabilities
//! - **Value Arithmetic**: Performance of scalar, vector, and finite field operations.
//! - **Built-in Functions**: Geometric operations (S7, quaternions, symplectic, etc.).
//! - **Octonion Algebra**: Non-associative multiplication and normalization.
//! - **Operator Evaluation**: Expression parsing and semantic execution.
//!
//! ### Architectural Notes
//! Benchmarks use the Hydron computation layer with conditional compilation
//! (`#[cfg(feature = "hydron")]`). Performance metrics help optimize the
//! E8 runtime execution pipeline.
//!
//! ### Example
//! ```bash
//! cargo bench --features hydron --bench rune_operators
//! ```
//!
//! Benchmarks identify computational bottlenecks in E8 geometric calculations.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "hydron")]
mod operator_benches {
    use super::*;
    use rune_format::rune::hydron::Gf8;
    use rune_format::rune::hydron::values::{EvalContext, RuneBuiltin, Value};

    pub fn bench_value_arithmetic(c: &mut Criterion) {
        let mut group = c.benchmark_group("value_arithmetic");

        let a = Value::Scalar(42.0);
        let val_b = Value::Scalar(7.0);

        group.bench_function("scalar_add", |b| {
            b.iter(|| a.add(std::hint::black_box(&val_b)).unwrap());
        });

        group.bench_function("scalar_mul", |b| {
            b.iter(|| a.mul(std::hint::black_box(&val_b)).unwrap());
        });

        group.bench_function("scalar_sub", |b| {
            b.iter(|| a.sub(std::hint::black_box(&val_b)).unwrap());
        });

        // Vec8 operations
        let v1 = Value::Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let v2 = Value::Vec8([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

        group.bench_function("vec8_add", |b| {
            b.iter(|| v1.add(std::hint::black_box(&v2)).unwrap());
        });

        group.bench_function("vec8_mul_scalar", |b| {
            let scalar = Value::Scalar(2.0);
            b.iter(|| v1.mul(std::hint::black_box(&scalar)).unwrap());
        });

        // GF(8) operations
        let gf1 = Value::Gf8(Gf8::from_scalar(3.0));
        let gf2 = Value::Gf8(Gf8::from_scalar(5.0));

        group.bench_function("gf8_add", |b| {
            b.iter(|| gf1.add(std::hint::black_box(&gf2)).unwrap());
        });

        let scalar = Value::Scalar(2.0);
        group.bench_function("gf8_scale", |b| {
            b.iter(|| gf1.mul(std::hint::black_box(&scalar)).unwrap());
        });

        group.finish();
    }

    pub fn bench_builtin_ops(c: &mut Criterion) {
        let mut group = c.benchmark_group("builtin_ops");
        let ctx = EvalContext::new();

        // S7 operations
        let v1 = Value::Vec8([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v2 = Value::Vec8([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let t = Value::Scalar(0.5);

        group.bench_function("s7_project", |b| {
            b.iter(|| {
                ctx.apply_builtin(RuneBuiltin::S7Project, std::hint::black_box(&[v1.clone()]))
                    .unwrap()
            });
        });

        group.bench_function("s7_distance", |b| {
            b.iter(|| {
                ctx.apply_builtin(
                    RuneBuiltin::S7Distance,
                    std::hint::black_box(&[v1.clone(), v2.clone()]),
                )
                .unwrap()
            });
        });

        group.bench_function("s7_slerp", |b| {
            b.iter(|| {
                ctx.apply_builtin(
                    RuneBuiltin::S7Slerp,
                    std::hint::black_box(&[v1.clone(), v2.clone(), t.clone()]),
                )
                .unwrap()
            });
        });

        // Quaternion operations
        let q1 = Value::Quaternion([1.0, 0.0, 0.0, 0.0]);
        let q2 = Value::Quaternion([0.707, 0.707, 0.0, 0.0]);

        group.bench_function("quat_slerp", |b| {
            b.iter(|| {
                ctx.apply_builtin(
                    RuneBuiltin::QuatSlerp,
                    std::hint::black_box(&[q1.clone(), q2.clone(), t.clone()]),
                )
                .unwrap()
            });
        });

        // Symplectic operations
        let state = Value::Vec16([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // position
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // momentum
        ]);
        let dt = Value::Scalar(0.01);

        group.bench_function("symplectic_hamiltonian", |b| {
            b.iter(|| {
                ctx.apply_builtin(
                    RuneBuiltin::SymHamiltonian,
                    std::hint::black_box(&[state.clone()]),
                )
                .unwrap()
            });
        });

        group.bench_function("symplectic_evolve", |b| {
            b.iter(|| {
                ctx.apply_builtin(
                    RuneBuiltin::SymEvolveStep,
                    std::hint::black_box(&[state.clone(), dt.clone()]),
                )
                .unwrap()
            });
        });

        // Topological operations
        let points = Value::Vec16([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // point 1
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // point 2
        ]);

        group.bench_function("topo_betti", |b| {
            b.iter(|| {
                ctx.apply_builtin(
                    RuneBuiltin::TopoBetti,
                    std::hint::black_box(&[points.clone()]),
                )
                .unwrap()
            });
        });

        group.bench_function("topo_signature", |b| {
            b.iter(|| {
                ctx.apply_builtin(
                    RuneBuiltin::TopoSignature,
                    std::hint::black_box(&[points.clone()]),
                )
                .unwrap()
            });
        });

        group.finish();
    }

    pub fn bench_octonion_ops(c: &mut Criterion) {
        let mut group = c.benchmark_group("octonion");

        use rune_format::rune::hydron::values::Octonion;

        let o1 = Octonion::new(1.0, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let o2 = Octonion::new(0.707, [0.707, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        group.bench_function("multiply", |b| {
            b.iter(|| o1.mul(std::hint::black_box(&o2)));
        });

        group.bench_function("conjugate", |b| {
            b.iter(|| o1.conjugate());
        });

        group.bench_function("norm", |b| {
            b.iter(|| o1.norm());
        });

        group.finish();
    }
}

#[cfg(feature = "hydron")]
criterion_group!(
    benches,
    operator_benches::bench_value_arithmetic,
    operator_benches::bench_builtin_ops,
    operator_benches::bench_octonion_ops
);

#[cfg(not(feature = "hydron"))]
fn dummy_bench(_c: &mut Criterion) {}

#[cfg(not(feature = "hydron"))]
criterion_group!(benches, dummy_bench);

criterion_main!(benches);
