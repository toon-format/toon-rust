//! Example to demonstrate Geoshi CUDA kernel functionality
//!
//! Run with: `cargo run --example test_cuda --features cuda`
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#[cfg(feature = "cuda")]
use cust::prelude::*;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "cuda"))]
    {
        println!("⚠️  This example requires the 'cuda' feature.");
        println!("   Please run with: cargo run --example test_cuda --features cuda");
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    run_cuda_demo()
}

#[cfg(feature = "cuda")]
fn run_cuda_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Geoshi CUDA Kernel Demonstration");
    println!("===================================");
    println!();

    // Initialize CUDA context
    println!("Initializing CUDA context...");
    let _ctx = cust::quick_init().map_err(|e| format!("Failed to init CUDA: {:?}", e))?;
    let device = cust::device::Device::get_device(0)
        .map_err(|e| format!("Failed to get CUDA device: {:?}", e))?;

    // Get device info
    let device_name = device.name()
        .map_err(|e| format!("Failed to get device name: {:?}", e))?;
    println!("  ✅ Using CUDA device: {}", device_name);

    // Load the PTX module
    let ptx_path = std::env::var("MGSK_PTX")
        .unwrap_or_else(|_| "target/cuda-build/mgsk.ptx".to_string());

    if !std::path::Path::new(&ptx_path).exists() {
        println!("  ❌ PTX file not found at: {}", ptx_path);
        println!("     Please run the build_cuda_kernels.ps1 script first.");
        return Ok(());
    }

    println!("Loading PTX module from: {}", ptx_path);
    let ptx_data = std::fs::read_to_string(&ptx_path)
        .map_err(|e| format!("Failed to read PTX file: {:?}", e))?;

    let module = Module::from_ptx(ptx_data, &[])
        .map_err(|e| format!("Failed to load PTX module: {:?}", e))?;
    println!("  ✅ PTX module loaded successfully");

    // Get kernel function
    let function = module.get_function("distance_matrix_kernel")
        .map_err(|e| format!("Failed to get kernel function: {:?}", e))?;
    println!("  ✅ Kernel function 'distance_matrix_kernel' found");

    // Create test data
    println!();
    println!("2️⃣ Testing CUDA Kernel Execution:");

    let n_points = 4; // Number of points
    let n_features = 2; // 2D points
    let mut host_a = vec![0.0f64; n_points * n_features];
    let mut host_b = vec![0.0f64; n_points * n_features];
    let mut host_output = vec![0.0f64; n_points * n_points];

    // Initialize with some 2D points
    host_a[0] = 0.0; host_a[1] = 0.0; // (0,0)
    host_a[2] = 1.0; host_a[3] = 0.0; // (1,0)
    host_a[4] = 0.0; host_a[5] = 1.0; // (0,1)
    host_a[6] = 1.0; host_a[7] = 1.0; // (1,1)

    // Copy A to B for distance matrix computation
    host_b.copy_from_slice(&host_a);

    println!("  📊 Test points:");
    for i in 0..n_points {
        println!("    Point {}: ({:.1}, {:.1})",
                i, host_a[i*2], host_a[i*2+1]);
    }

    // Allocate device memory
    let dev_a = DeviceBuffer::from_slice(&host_a)?;
    let dev_b = DeviceBuffer::from_slice(&host_b)?;
    let dev_output = DeviceBuffer::from_slice(&host_output)?;

    // Launch kernel
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    println!("  🚀 Launching CUDA distance matrix kernel (sanity check)...");

    // 1 block is enough for 4 points; scale if you adjust n_points.
    let block = (16u32, 16u32, 1u32);
    let grid = (1u32, 1u32, 1u32);

    unsafe {
        launch!(
            function<<<grid, block, 0, stream>>>(
                dev_a.as_device_ptr(),
                dev_b.as_device_ptr(),
                dev_output.as_device_ptr(),
                n_points as i32,
                n_points as i32,
                n_features as i32,
                0, // Euclidean distance
                2.0f64 // p parameter (unused for Euclidean)
            )
        )?;
    }

    stream.synchronize()?;

    // Copy results back
    dev_output.copy_to(&mut host_output)?;

    println!("  ✅ Kernel execution completed");
    println!("  📊 Distance matrix:");
    for i in 0..n_points {
        print!("    ");
        for j in 0..n_points {
            print!("{:4.2} ", host_output[i * n_points + j]);
        }
        println!();
    }

    println!();
    println!("🎉 Geoshi CUDA Kernel Demonstration Complete!");
    println!();
    println!("💡 This proves Geoshi can now:");
    println!("   • Load and execute CUDA kernels from PTX");
    println!("   • Perform GPU-accelerated geometric computations");
    println!("   • Integrate CUDA with Rust applications");
    println!("   • Enable high-performance spatial intelligence");

    // ---------------------------------------------------------------------
    // Throughput benchmark (GFLOPS / GB/s)
    // ---------------------------------------------------------------------
    println!();
    println!("⚡ Throughput Benchmark (distance_matrix_kernel)");

    let bench_points: usize = 4096;
    let bench_features: usize = 8;
    let iterations: u32 = 10;

    let bench_len = bench_points * bench_features;
    let bench_out_len = bench_points * bench_points;

    let bench_host_a: Vec<f64> = (0..bench_len).map(|i| (i % 13) as f64 * 0.01).collect();
    let bench_host_b = bench_host_a.clone();

    let bench_dev_a = DeviceBuffer::from_slice(&bench_host_a)?;
    let bench_dev_b = DeviceBuffer::from_slice(&bench_host_b)?;
    let bench_dev_out = DeviceBuffer::<f64>::zeroed(bench_out_len)?;

    let block = (16u32, 16u32, 1u32);
    let grid = (
        ((bench_points as u32) + block.0 - 1) / block.0,
        ((bench_points as u32) + block.1 - 1) / block.1,
        1u32,
    );

    // Warm-up
    unsafe {
        launch!(
            function<<<grid, block, 0, stream>>>(
                bench_dev_a.as_device_ptr(),
                bench_dev_b.as_device_ptr(),
                bench_dev_out.as_device_ptr(),
                bench_points as i32,
                bench_points as i32,
                bench_features as i32,
                0,    // Euclidean
                2.0f64
            )
        )?;
    }
    stream.synchronize()?;

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        unsafe {
            launch!(
                function<<<grid, block, 0, stream>>>(
                    bench_dev_a.as_device_ptr(),
                    bench_dev_b.as_device_ptr(),
                    bench_dev_out.as_device_ptr(),
                    bench_points as i32,
                    bench_points as i32,
                    bench_features as i32,
                    0,
                    2.0f64
                )
            )?;
        }
    }
    stream.synchronize()?;

    let total_ms = start.elapsed().as_secs_f64() * 1e3;
    let avg_ms = total_ms / iterations as f64;
    let total_pairs = bench_points as f64 * bench_points as f64;
    let flops_per_pair = (bench_features as f64 * 2.0) + 1.0; // diff+mul per feature + sqrt
    let total_flops = total_pairs * flops_per_pair;
    let gflops = (total_flops / (avg_ms / 1_000.0)) / 1e9;

    let bytes_per_pair = (2.0 * bench_features as f64 * 8.0) + 8.0; // read A/B + write out
    let total_bytes = total_pairs * bytes_per_pair;
    let gbps = (total_bytes / (avg_ms / 1_000.0)) / 1e9;

    println!("  Points: {}", bench_points);
    println!("  Features: {}", bench_features);
    println!("  Average time: {:.3} ms ({} iters)", avg_ms, iterations);
    println!("  Throughput: {:.2} GFLOPS", gflops);
    println!("  Bandwidth (approx): {:.2} GB/s", gbps);

    Ok(())
}
