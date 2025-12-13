use criterion::{Criterion, criterion_group, criterion_main};
use rune_format::rune;

// Sample data representing typical RUNE usage with semantic prefixes and arrays
const RUNE_SAMPLE: &str = r#"
root: e8_continuum

# Semantic prefix usage (domain-specific identifiers)
T:Gf8: 2.5
V:velocity: [1.0, 2.0, 3.0]
M:transform: [[1,0,0],[0,1,0],[0,0,1]]

# Array operations
coords: [10, 20, 30]
weights: [0.5, 0.3, 0.2]

# Traditional TOON data block
layers ~RUNE:
  config[3]{id, type, active}:
    1,Lattice,true
    2,Projection,true
    3,Transform,false

# Semantic flow
T:Gf8 * V:velocity -> R:result
"#;

const TOON_EQUIVALENT: &str = r#"
root: e8_continuum

tensor_Gf8: 2.5
vector_velocity: [1.0, 2.0, 3.0]
matrix_transform: [[1,0,0],[0,1,0],[0,0,1]]

coords: [10, 20, 30]
weights: [0.5, 0.3, 0.2]

layers:
  config[3]{id, type, active}:
    1,Lattice,true
    2,Projection,true
    3,Transform,false
"#;

// Complex nested structure
const RUNE_COMPLEX: &str = r#"
root: tensor_network

# Multi-dimensional tensor with semantic prefix
T:Gf8_primary: [
  [1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0],
  [7.0, 8.0, 9.0]
]

# Vector array with domain notation
V:velocities: [
  [0.5, 0.5, 0.5],
  [1.0, 1.0, 1.0],
  [1.5, 1.5, 1.5]
]

# Semantic operations
T:result: T:Gf8_primary * V:velocities

# Metadata block
metadata ~RUNE:
  dimensions: [3, 3]
  timestamp: 1638835200
  layers[5]{id, name, type}:
    1,input,vector
    2,hidden1,tensor
    3,hidden2,tensor
    4,hidden3,tensor
    5,output,scalar
"#;

fn benchmark_rune_parse(c: &mut Criterion) {
    c.bench_function("parse_rune_simple", |b| {
        b.iter(|| rune::parse(std::hint::black_box(RUNE_SAMPLE)))
    });

    c.bench_function("parse_rune_complex", |b| {
        b.iter(|| rune::parse(std::hint::black_box(RUNE_COMPLEX)))
    });
}

fn benchmark_toon_equivalent(c: &mut Criterion) {
    c.bench_function("parse_toon_equivalent", |b| {
        b.iter(|| rune::parse(std::hint::black_box(TOON_EQUIVALENT)))
    });
}

fn benchmark_semantic_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("semantic_features");

    // Semantic prefix parsing
    group.bench_function("semantic_prefix", |b| {
        b.iter(|| rune::parse(std::hint::black_box("T:Gf8 * V:velocity -> R:result")))
    });

    // Array literal parsing
    group.bench_function("array_literal", |b| {
        b.iter(|| rune::parse(std::hint::black_box("[1, 2, 3, 4, 5]")))
    });

    // Nested array math
    group.bench_function("nested_array_math", |b| {
        b.iter(|| rune::parse(std::hint::black_box("[[3,3,3]*[3,3,3]]")))
    });

    // Combined semantic + arrays
    group.bench_function("semantic_with_arrays", |b| {
        b.iter(|| {
            rune::parse(std::hint::black_box(
                "T:tensor: [1, 2, 3]\nV:vector: [4, 5, 6]",
            ))
        })
    });

    group.finish();
}

fn benchmark_token_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_comparison");

    // Measure relative token counts (simulated via string length as proxy)
    group.bench_function("rune_format_size", |b| {
        b.iter(|| std::hint::black_box(RUNE_SAMPLE.len()))
    });

    group.bench_function("toon_format_size", |b| {
        b.iter(|| std::hint::black_box(TOON_EQUIVALENT.len()))
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_rune_parse,
    benchmark_toon_equivalent,
    benchmark_semantic_features,
    benchmark_token_comparison
);
criterion_main!(benches);
