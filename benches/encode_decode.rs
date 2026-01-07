use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use serde_json::{json, Value};
use toon_format::{decode_default, encode_default};

fn make_tabular(rows: usize) -> Value {
    let mut items = Vec::with_capacity(rows);
    for i in 0..rows {
        items.push(json!({
            "id": i,
            "name": format!("User_{i}"),
            "score": i * 2,
            "active": i % 2 == 0,
            "tag": format!("tag{i}"),
        }));
    }
    Value::Array(items)
}

fn make_deep_object(depth: usize) -> Value {
    let mut value = json!({
        "leaf": "value",
        "count": 1,
    });

    for i in 0..depth {
        value = json!({
            format!("level_{i}"): value,
        });
    }

    value
}

fn make_long_unquoted(words: usize) -> String {
    let mut parts = Vec::with_capacity(words);
    for i in 0..words {
        parts.push(format!("word{i}"));
    }
    parts.join(" ")
}

fn bench_tabular(c: &mut Criterion) {
    let mut group = c.benchmark_group("tabular");
    for rows in [128_usize, 1024] {
        let value = make_tabular(rows);
        let toon = encode_default(&value).expect("encode tabular");

        group.bench_with_input(BenchmarkId::new("encode", rows), &value, |b, val| {
            b.iter(|| encode_default(black_box(val)).expect("encode tabular"));
        });

        group.bench_with_input(BenchmarkId::new("decode", rows), &toon, |b, input| {
            b.iter(|| decode_default::<Value>(black_box(input)).expect("decode tabular"));
        });
    }
    group.finish();
}

fn bench_deep_object(c: &mut Criterion) {
    let mut group = c.benchmark_group("deep_object");
    for depth in [32_usize, 128] {
        let value = make_deep_object(depth);
        let toon = encode_default(&value).expect("encode deep object");

        group.bench_with_input(BenchmarkId::new("encode", depth), &value, |b, val| {
            b.iter(|| encode_default(black_box(val)).expect("encode deep object"));
        });

        group.bench_with_input(BenchmarkId::new("decode", depth), &toon, |b, input| {
            b.iter(|| decode_default::<Value>(black_box(input)).expect("decode deep object"));
        });
    }
    group.finish();
}

fn bench_long_unquoted(c: &mut Criterion) {
    let words = 512;
    let long_value = make_long_unquoted(words);
    let toon = format!("value: {long_value}");

    c.bench_function("decode_long_unquoted", |b| {
        b.iter(|| decode_default::<Value>(black_box(&toon)).expect("decode long unquoted"));
    });
}

criterion_group!(
    benches,
    bench_tabular,
    bench_deep_object,
    bench_long_unquoted
);
criterion_main!(benches);
