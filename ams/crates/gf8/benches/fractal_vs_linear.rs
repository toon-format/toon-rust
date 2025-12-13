use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use gf8::{FractalSimtConfig, fractal_simt_add_f32_in_place};
use std::hint::black_box;

fn linear_add_in_place(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

fn bench_fractal_vs_linear(c: &mut Criterion) {
    let n = 1 << 20; // 1M elements
    let base_a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
    let cfg = FractalSimtConfig::default();

    c.bench_function("linear_add_in_place", |bch| {
        bch.iter_batched(
            || base_a.clone(),
            |mut a| {
                linear_add_in_place(black_box(&mut a), black_box(&b));
                black_box(a)
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("fractal_simt_add_f32_in_place", |bch| {
        bch.iter_batched(
            || base_a.clone(),
            |mut a| {
                fractal_simt_add_f32_in_place(black_box(&mut a), black_box(&b), black_box(&cfg));
                black_box(a)
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench_fractal_vs_linear);
criterion_main!(benches);
