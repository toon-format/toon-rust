/*
.\scripts\build_cuda_kernels.ps1          # build + CUDA demo (includes throughput stats)
.\scripts\build_cuda_kernels.ps1 -Scanner # build + CUDA demo + scanner
 */

#![cfg(feature = "cuda")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use cust::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

fn load_ptx() -> String {
    if let Ok(path) = std::env::var("MGSK_PTX") {
        return fs::read_to_string(path).expect("Failed to read MGSK_PTX");
    }

    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../../target/cuda-build/mgsk.ptx");
    fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("MGSK_PTX not set and {} missing", path.display()))
}

fn bench_distance_matrix(c: &mut Criterion) {
    let _context = cust::quick_init().expect("CUDA init failed");
    let module = Module::from_ptx(load_ptx(), &[]).expect("PTX load failed");
    let func = module
        .get_function("distance_matrix_kernel")
        .expect("distance_matrix_kernel not found");
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let mut group = c.benchmark_group("cuda_distance_matrix");
    group.measurement_time(Duration::from_secs(5));

    for &n in &[256usize, 512, 1024] {
        let n_features = 8usize;
        let total = n * n_features;
        let host_a: Vec<f64> = (0..total).map(|i| (i % 13) as f64 * 0.01).collect();
        let host_b = host_a.clone();
        let mut host_out = vec![0.0f64; n * n];

        let d_a = DeviceBuffer::from_slice(&host_a).unwrap();
        let d_b = DeviceBuffer::from_slice(&host_b).unwrap();
        let d_out = DeviceBuffer::from_slice(&host_out).unwrap();

        let block = (16u32, 16u32, 1u32);
        let grid = (
            ((n as u32) + block.0 - 1) / block.0,
            ((n as u32) + block.1 - 1) / block.1,
            1u32,
        );

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                unsafe {
                    launch!(func<<<grid, block, 0, stream>>>(
                        d_a.as_device_ptr(),
                        d_b.as_device_ptr(),
                        d_out.as_device_ptr(),
                        n as i32,
                        n as i32,
                        n_features as i32,
                        0i32,       // Euclidean
                        2.0f64      // p (unused for Euclidean)
                    ))
                }
                .unwrap();
                stream.synchronize().unwrap();
            });
        });

        d_out.copy_to(&mut host_out).unwrap();
        assert!(host_out.iter().all(|v| v.is_finite()));
    }

    group.finish();
}

fn bench_conv2d(c: &mut Criterion) {
    let _context = cust::quick_init().expect("CUDA init failed");
    let module = Module::from_ptx(load_ptx(), &[]).expect("PTX load failed");
    let func = module
        .get_function("conv2d_kernel")
        .expect("conv2d_kernel not found");
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let mut group = c.benchmark_group("cuda_conv2d");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[64usize, 128, 256] {
        let kernel_size = 3usize;
        let stride = 1usize;
        let out_dim = (size - kernel_size) / stride + 1;

        let input: Vec<f64> = (0..size * size)
            .map(|i| ((i % 7) as f64) * 0.25)
            .collect();
        let kernel: Vec<f64> = vec![0.111_111_f64; kernel_size * kernel_size];
        let mut output = vec![0.0f64; out_dim * out_dim];

        let d_input = DeviceBuffer::from_slice(&input).unwrap();
        let d_kernel = DeviceBuffer::from_slice(&kernel).unwrap();
        let d_out = DeviceBuffer::from_slice(&output).unwrap();

        let block = (16u32, 16u32, 1u32);
        let grid = (
            ((out_dim as u32) + block.0 - 1) / block.0,
            ((out_dim as u32) + block.1 - 1) / block.1,
            1u32,
        );

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                unsafe {
                    launch!(func<<<grid, block, 0, stream>>>(
                        d_input.as_device_ptr(),
                        d_kernel.as_device_ptr(),
                        d_out.as_device_ptr(),
                        size as i32,
                        size as i32,
                        kernel_size as i32,
                        stride as i32
                    ))
                }
                .unwrap();
                stream.synchronize().unwrap();
            });
        });

        d_out.copy_to(&mut output).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    group.finish();
}

criterion_group!(benches, bench_distance_matrix, bench_conv2d);
criterion_main!(benches);
