use criterion::{Criterion, criterion_group, criterion_main};
use fory::Fory;
use rune_format::rune::{MathOp, OpCategory, RuneOp};
use std::hint::black_box;

fn bench_fory_ops(c: &mut Criterion) {
    let mut fory = Fory::default();
    // Register types
    fory.register::<OpCategory>(1).unwrap();
    fory.register::<RuneOp>(2).unwrap();
    fory.register::<MathOp>(3).unwrap();

    let op_category = OpCategory::Relation;
    let rune_op = RuneOp::FlowRight;
    let math_op = MathOp::Add;

    let mut group = c.benchmark_group("fory_ops");

    group.bench_function("serialize_op_category", |b| {
        b.iter(|| fory.serialize(black_box(&op_category)).unwrap())
    });

    let bytes_cat = fory.serialize(&op_category).unwrap();
    group.bench_function("deserialize_op_category", |b| {
        b.iter(|| {
            fory.deserialize::<OpCategory>(black_box(&bytes_cat))
                .unwrap()
        })
    });

    group.bench_function("serialize_rune_op", |b| {
        b.iter(|| fory.serialize(black_box(&rune_op)).unwrap())
    });

    let bytes_rune = fory.serialize(&rune_op).unwrap();
    group.bench_function("deserialize_rune_op", |b| {
        b.iter(|| fory.deserialize::<RuneOp>(black_box(&bytes_rune)).unwrap())
    });

    group.bench_function("serialize_math_op", |b| {
        b.iter(|| fory.serialize(black_box(&math_op)).unwrap())
    });

    let bytes_math = fory.serialize(&math_op).unwrap();
    group.bench_function("deserialize_math_op", |b| {
        b.iter(|| fory.deserialize::<MathOp>(black_box(&bytes_math)).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_fory_ops);
criterion_main!(benches);
