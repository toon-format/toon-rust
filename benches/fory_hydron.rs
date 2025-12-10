use criterion::{Criterion, criterion_group, criterion_main};
use fory::Fory;
use hydron_core::Gf8;
use std::hint::black_box;

fn bench_fory_hydron(c: &mut Criterion) {
    let mut fory = Fory::default();
    // Register types
    fory.register::<Gf8>(10).unwrap();

    let gf8 = Gf8::from_scalar(0.5);

    let mut group = c.benchmark_group("fory_hydron");

    group.bench_function("serialize_gf8", |b| {
        b.iter(|| fory.serialize(black_box(&gf8)).unwrap())
    });

    let bytes_gf8 = fory.serialize(&gf8).unwrap();
    group.bench_function("deserialize_gf8", |b| {
        b.iter(|| fory.deserialize::<Gf8>(black_box(&bytes_gf8)).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_fory_hydron);
criterion_main!(benches);
