use criterion::{Criterion, criterion_group, criterion_main};
use hydron_core::Gf8;
use std::hint::black_box;

fn bench_serde_hydron(c: &mut Criterion) {
    let gf8 = Gf8::from_scalar(0.5);

    let mut group = c.benchmark_group("serde_hydron");

    group.bench_function("serialize_gf8", |b| {
        b.iter(|| serde_json::to_vec(black_box(&gf8)).unwrap())
    });

    let bytes_gf8 = serde_json::to_vec(&gf8).unwrap();
    group.bench_function("deserialize_gf8", |b| {
        b.iter(|| serde_json::from_slice::<Gf8>(black_box(&bytes_gf8)).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_serde_hydron);
criterion_main!(benches);
