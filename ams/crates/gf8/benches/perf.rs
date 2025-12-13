// benches/perf.rs
//! Throughput benchmarks for the `e8_gf8` crate.
//!
//! Run with:
//! ```bash
//! cargo bench --bench perf
//! ```
//!
//! Make sure `Cargo.toml` has:
//! ```toml
//! [dev-dependencies]
//! criterion = "0.5"
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use gf8::{
    Gf8, Gf8BitSig, gf8_add_inplace_slice_simd, gf8_add_simd, gf8_chordal_distance,
    gf8_chordal_distance2, gf8_cosine_similarity, gf8_dot_simd, gf8_from_code, gf8_lerp,
    gf8_norm2_simd, gf8_slerp, gf8_sub_simd, gf8_to_code, quantize_to_e8_shell,
};
use std::hint::black_box;

fn bench_gf8_construction(c: &mut Criterion) {
    let bits = [1u8, 0, 1, 1, 0, 0, 1, 0];

    c.bench_function("gf8_from_bits_even_parity", |b| {
        b.iter(|| {
            let v = Gf8::from_bits_even_parity(black_box(bits));
            black_box(v)
        })
    });

    c.bench_function("gf8_from_scalar", |b| {
        b.iter(|| {
            let v = Gf8::from_scalar(black_box(-123.45));
            black_box(v)
        })
    });

    c.bench_function("gf8_from_coords/new", |b| {
        let coords = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        b.iter(|| {
            let v = Gf8::from_coords(black_box(coords));
            black_box(v)
        })
    });
}

fn bench_bitcodec(c: &mut Criterion) {
    c.bench_function("bitcodec_u8_to_gf8_and_back", |b| {
        b.iter(|| {
            let code_u8 = black_box(0b1011_0010u8);
            let code = Gf8BitSig(code_u8);
            let gf = gf8_from_code(code);
            let code2 = gf8_to_code(&gf);
            black_box((gf, code2))
        })
    });

    c.bench_function("bitcodec_roundtrip_scan_0_255", |b| {
        b.iter(|| {
            let mut acc = 0f32;
            for i in 0u8..=255 {
                let bits = gf8::bits_from_u8_le(i);
                let gf = Gf8::from_bits_even_parity(bits);
                let code = gf8_to_code(&gf);
                let gf2 = gf8_from_code(code);
                acc += gf.dot(gf2.coords());
            }
            black_box(acc)
        })
    });
}

fn scalar_dot(a: &Gf8, b: &Gf8) -> f32 {
    // Use a scalar dot (existing API) as the baseline
    a.dot(b.coords())
}

fn scalar_norm2(v: &Gf8) -> f32 {
    v.coords().iter().map(|x| x * x).sum()
}

fn bench_simd_vs_scalar(c: &mut Criterion) {
    let a = Gf8::new([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
    let b = Gf8::new([-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8]);

    c.bench_function("dot_scalar", |bch| {
        bch.iter(|| {
            let v = scalar_dot(black_box(&a), black_box(&b));
            black_box(v)
        })
    });

    c.bench_function("dot_simd", |bch| {
        bch.iter(|| {
            let v = gf8_dot_simd(black_box(&a), black_box(&b));
            black_box(v)
        })
    });

    c.bench_function("norm2_scalar", |bch| {
        bch.iter(|| {
            let v = scalar_norm2(black_box(&a));
            black_box(v)
        })
    });

    c.bench_function("norm2_simd", |bch| {
        bch.iter(|| {
            let v = gf8_norm2_simd(black_box(&a));
            black_box(v)
        })
    });

    c.bench_function("add_scalar_vs_sub_scalar", |bch| {
        bch.iter(|| {
            let add = a + b;
            let sub = a - b;
            black_box((add, sub))
        })
    });

    c.bench_function("add_simd_vs_sub_simd", |bch| {
        bch.iter(|| {
            let add = gf8_add_simd(black_box(&a), black_box(&b));
            let sub = gf8_sub_simd(black_box(&a), black_box(&b));
            black_box((add, sub))
        })
    });
}

fn bench_simd_slice_ops(c: &mut Criterion) {
    let n = 1024usize;
    let src_a: Vec<Gf8> = (0..n)
        .map(|i| {
            let f = (i as f32).sin();
            Gf8::new([
                f,
                -f,
                f * 0.5,
                -f * 0.5,
                f * 1.5,
                -f * 1.5,
                f * 2.0,
                -f * 2.0,
            ])
        })
        .collect();

    let src_b: Vec<Gf8> = (0..n)
        .map(|i| {
            let f = (i as f32).cos();
            Gf8::new([
                f * 0.3,
                -f * 0.3,
                f * 0.7,
                -f * 0.7,
                f * 1.1,
                -f * 1.1,
                f * 1.9,
                -f * 1.9,
            ])
        })
        .collect();

    c.bench_function("gf8_add_inplace_slice_simd_1024", |bch| {
        bch.iter_batched(
            || src_a.clone(),
            |mut dst| {
                let _ = gf8_add_inplace_slice_simd(black_box(&mut dst), black_box(&src_b));
                black_box(dst)
            },
            BatchSize::LargeInput,
        )
    });
}

fn bench_math_ops(c: &mut Criterion) {
    let a = Gf8::new([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
    let b = Gf8::new([-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8]);

    c.bench_function("gf8_cosine_similarity", |bch| {
        bch.iter(|| {
            let v = gf8_cosine_similarity(black_box(&a), black_box(&b));
            black_box(v)
        })
    });

    c.bench_function("gf8_angle_and_geodesic", |bch| {
        bch.iter(|| {
            let d_chord2 = gf8_chordal_distance2(black_box(&a), black_box(&b));
            let d_chord = gf8_chordal_distance(black_box(&a), black_box(&b));
            black_box((d_chord2, d_chord))
        })
    });

    c.bench_function("gf8_lerp", |bch| {
        bch.iter(|| {
            let v = gf8_lerp(black_box(&a), black_box(&b), black_box(0.33));
            black_box(v)
        })
    });

    c.bench_function("gf8_slerp", |bch| {
        bch.iter(|| {
            let v = gf8_slerp(black_box(&a), black_box(&b), black_box(0.33));
            black_box(v)
        })
    });

    let coords = [0.1f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
    c.bench_function("quantize_to_e8_shell", |bch| {
        bch.iter(|| {
            let v = quantize_to_e8_shell(black_box(&coords));
            black_box(v)
        })
    });
}

fn bench_end_to_end_pipeline(c: &mut Criterion) {
    // Simulate a tiny "pipeline": random-ish coords → Gf8 → code → decode → dot + math.
    let coords_a = [0.1f32, -0.2, 0.35, -0.4, 0.55, -0.6, 0.7, -0.8];
    let coords_b = [-0.3f32, 0.25, -0.15, 0.45, -0.65, 0.85, -0.95, 0.15];

    c.bench_function("pipeline_coords_to_code_to_math", |bch| {
        bch.iter(|| {
            let ga = Gf8::from_coords(black_box(coords_a));
            let gb = Gf8::from_coords(black_box(coords_b));

            let code_a = gf8_to_code(&ga);
            let code_b = gf8_to_code(&gb);

            let ga2 = gf8_from_code(code_a);
            let gb2 = gf8_from_code(code_b);

            let dot = gf8_dot_simd(&ga2, &gb2);
            let cos = gf8_cosine_similarity(&ga2, &gb2);
            let slerp_mid = gf8_slerp(&ga2, &gb2, 0.5);

            black_box((dot, cos, slerp_mid))
        })
    });
}

fn criterion_benches(c: &mut Criterion) {
    bench_gf8_construction(c);
    bench_bitcodec(c);
    bench_simd_vs_scalar(c);
    bench_simd_slice_ops(c);
    bench_math_ops(c);
    bench_end_to_end_pipeline(c);
}

criterion_group!(benches, criterion_benches);
criterion_main!(benches);
