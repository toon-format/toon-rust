use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::Rng;
use std::time::Duration;

use xuid::e8_lattice::quantize_to_e8;

/// Generate a random 8D point for E8 operations
fn random_e8_point<R: Rng>(rng: &mut R) -> [f64; 8] {
    let mut p = [0.0f64; 8];
    for i in 0..8 {
        p[i] = rng.random_range(-4.0..4.0);
    }
    p
}

fn bench_e8_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("E8 Quantize");
    group.measurement_time(Duration::from_secs(5));
    let mut rng = rand::rng();

    for n in [128usize, 512, 2048].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &size| {
            let points: Vec<[f64; 8]> = (0..size).map(|_| random_e8_point(&mut rng)).collect();
            b.iter(|| {
                for p in &points {
                    // measure E8 quantization
                    let _ = quantize_to_e8(p);
                }
            });
        });
    }
    group.finish();
}

// Simple naive matrix multiplication implementation
fn matmul(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

fn bench_matrix_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Mul");
    group.measurement_time(Duration::from_secs(5));
    let mut rng = rand::rng();

    for &n in &[64usize, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &size| {
            let a: Vec<f32> = (0..size * size).map(|_| rng.random()).collect();
            let b_mat: Vec<f32> = (0..size * size).map(|_| rng.random()).collect();
            let c_mat: Vec<f32> = vec![0f32; size * size];
            b.iter(|| {
                matmul(&a, &b_mat, &mut c_mat.clone(), size);
            });
        });
    }
    group.finish();
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cosine Similarity");
    group.measurement_time(Duration::from_secs(5));
    let mut rng = rand::rng();

    for &len in &[256usize, 1024, 4096] {
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, &size| {
            let a: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
            let b_vec: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
            b.iter(|| {
                // compute dot and norms
                let dot: f32 = a.iter().zip(&b_vec).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                let _cosine = dot / (norm_a * norm_b + 1e-12);
            });
        });
    }

    group.finish();
}

fn bench_diffuser_heat(c: &mut Criterion) {
    use ndarray::Array2;
    use geoshi::diffuser::{Diffuser, DiffusionConfig, DiffusionMethod};

    let mut group = c.benchmark_group("Diffusion::Heat");
    group.measurement_time(Duration::from_secs(5));
    let mut rng = rand::rng();

    for &size in &[32usize, 64, 128] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &sz| {
            let values: Vec<f64> = (0..sz * sz).map(|_| rng.random()).collect();
            let grid = Array2::from_shape_vec((sz, sz), values).unwrap();

            let config = DiffusionConfig {
                method: DiffusionMethod::Heat,
                iterations: 50,
                time_step: 0.1,
                ..Default::default()
            };
            let diffuser = Diffuser::new(config);

            b.iter(|| {
                let _ = diffuser.diffuse_grid(grid.view()).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_anisotropic_diffuser(c: &mut Criterion) {
    use ndarray::Array2;
    use geoshi::diffuser::{AnisotropicDiffuser, DiffusionConfig};

    let mut group = c.benchmark_group("Diffusion::Anisotropic");
    group.measurement_time(Duration::from_secs(5));
    let mut rng = rand::rng();

    for &size in &[32usize, 64, 128] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &sz| {
            let values: Vec<f64> = (0..sz * sz).map(|_| rng.random()).collect();
            let grid = Array2::from_shape_vec((sz, sz), values).unwrap();

            let config = DiffusionConfig { iterations: 30, time_step: 0.08, ..Default::default() };
            let diff = AnisotropicDiffuser::new(config);

            b.iter(|| {
                let _ = diff.diffuse(grid.view()).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_graph_diffuser(c: &mut Criterion) {
    use geoshi::diffuser::GraphDiffuser;

    let mut group = c.benchmark_group("Diffusion::Graph");
    group.measurement_time(Duration::from_secs(5));
    let mut rng = rand::rng();

    for &n in &[128usize, 512, 2048] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &nodes| {
            // Create random adjacency (each node connects to up to 4 neighbors)
            let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); nodes];
            let mut weights: Vec<Vec<f64>> = (0..nodes).map(|_| vec![0.0f64; nodes]).collect();

            for i in 0..nodes {
                for _ in 0..4 {
                    let j = rng.random_range(0..nodes);
                    if !adjacency[i].contains(&j) && i != j {
                        adjacency[i].push(j);
                        weights[i][j] = rng.random_range(0.0..1.0);
                    }
                }
            }

            let initial_values: Vec<f64> = (0..nodes).map(|_| rng.random_range(0.0..1.0)).collect();
            let diffuser = GraphDiffuser::new(adjacency, weights);

            b.iter(|| {
                let _ = diffuser.diffuse(&initial_values, 10).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_e8_quantize, bench_matrix_mul, bench_cosine_similarity, bench_diffuser_heat, bench_anisotropic_diffuser, bench_graph_diffuser);
criterion_main!(benches);
