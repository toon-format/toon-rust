/* cuda/kernels.cu */
//! GPU computation kernels and accelerated geometric operations
//!
//! # KERNELS MODULE
//!▫~•◦------------------------------------------------‣
//!
//! GPU-accelerated computation kernels for high-performance geometric
//! transformations, cognition acceleration, and parallel processing operations.
//! Implements CUDA kernels for lattice computations and geometric synthesis.
//!
//! ### Key Capabilities
//! - **CUDA Acceleration:** GPU-accelerated geometric computations.
//! - **Kernel Operations:** Distance, RBF, lattice, convolution, and mesh kernels.
//! - **High-Performance Lattices:** Accelerated E8 lattice operations.
//! - **Parallel Cognition:** Parallel processing of geometric intelligence tasks.
//! - **Memory Optimization:** GPU memory management for large-scale computations.
//!
//! ### Technical Features
//! - **SIMT Processing:** Single-Instruction, Multiple-Thread execution models.
//! - **Memory Hierarchy:** Efficient use of shared and global memory.
//! - **Kernel Fusion:** Combined operations to minimize memory transfers.
//! - **Precision Control:** Configurable floating-point precision for tradeoffs.
//! - **Multi-GPU Ready:** Can be launched on any CUDA device; multi-GPU is handled by the caller.
//!
//! ### Usage Patterns (Rust FFI Example)
//! - Launch `distance_matrix_kernel` or `conv2d_kernel` via `cust::launch!`
//! - Use the provided `ams_launch_*` helpers if calling from C FFI
//! 
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef AMS_KERNELS_FLOAT
// Default numeric type: double. Define AMS_KERNELS_FLOAT as float to switch.
typedef double ams_float;
#else
typedef AMS_KERNELS_FLOAT ams_float;
#endif

// -------------------------------------------------------------------------------------------------
// Utility: CUDA error check helper (host-side, optional for direct C++ usage)
// -------------------------------------------------------------------------------------------------
extern "C"
cudaError_t ams_kernels_last_error() {
    return cudaGetLastError();
}

// -------------------------------------------------------------------------------------------------
// DEVICE HELPERS
// -------------------------------------------------------------------------------------------------

__device__ inline ams_float ams_square(ams_float x) {
    return x * x;
}

// Euclidean distance between two feature vectors (row-major, contiguous)
__device__ inline ams_float ams_euclidean_distance(
    const ams_float* __restrict__ a,
    const ams_float* __restrict__ b,
    int n_features
) {
    ams_float acc = 0.0;
    for (int d = 0; d < n_features; ++d) {
        ams_float diff = a[d] - b[d];
        acc += diff * diff;
    }
    return sqrt(acc);
}

// -------------------------------------------------------------------------------------------------
// 1. RBF (RADIAL BASIS FUNCTION) KERNEL
// -------------------------------------------------------------------------------------------------
//
// Kernel name: rbf_kernel
//
// Computes:
//   result[target_idx, feature_idx] = sum_j exp(-||target_j - input_j||^2 / (2*sigma^2))
//                                      * input[j, feature_idx]
//
// - d_input:  [n_inputs  * n_features]
// - d_target: [n_targets * n_features]
// - d_output: [n_targets * n_features]
// - n_inputs, n_targets, n_features as usual
// - sigma: RBF width parameter
//
extern "C" __global__ void rbf_kernel(
    const ams_float* __restrict__ d_input,
    const ams_float* __restrict__ d_target,
    ams_float* __restrict__ d_output,
    int n_inputs,
    int n_targets,
    int n_features,
    ams_float sigma
) {
    int target_idx  = blockIdx.x * blockDim.x + threadIdx.x; // row in target
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y; // feature

    if (target_idx >= n_targets || feature_idx >= n_features) {
        return;
    }

    const ams_float* target_row = d_target + target_idx * n_features;
    ams_float sum = 0.0;
    ams_float denom = 2.0 * sigma * sigma;

    for (int j = 0; j < n_inputs; ++j) {
        const ams_float* input_row = d_input + j * n_features;

        // Compute squared distance between target_row and input_row
        ams_float dist_sq = 0.0;
        for (int d = 0; d < n_features; ++d) {
            ams_float diff = target_row[d] - input_row[d];
            dist_sq += diff * diff;
        }

        ams_float weight = exp(-dist_sq / denom);
        sum += weight * input_row[feature_idx];
    }

    d_output[target_idx * n_features + feature_idx] = sum;
}

// -------------------------------------------------------------------------------------------------
// 2. DISTANCE MATRIX KERNEL
// -------------------------------------------------------------------------------------------------

enum AmsDistanceMetric : int {
    AMS_DIST_EUCLIDEAN = 0,
    AMS_DIST_MANHATTAN = 1,
    AMS_DIST_CHEBYSHEV = 2,
    AMS_DIST_MINKOWSKI = 3
};

// Compute distance matrix between A (n_a x n_features) and B (n_b x n_features).
// d_out is [n_a * n_b] row-major: out[i, j] = distance(A[i], B[j])
extern "C" __global__ void distance_matrix_kernel(
    const ams_float* __restrict__ d_a,
    const ams_float* __restrict__ d_b,
    ams_float* __restrict__ d_out,
    int n_a,
    int n_b,
    int n_features,
    int metric,
    ams_float p // used only for Minkowski
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // index in A
    int j = blockIdx.y * blockDim.y + threadIdx.y; // index in B

    if (i >= n_a || j >= n_b) {
        return;
    }

    const ams_float* a_row = d_a + i * n_features;
    const ams_float* b_row = d_b + j * n_features;

    ams_float result = 0.0;

    if (metric == AMS_DIST_EUCLIDEAN) {
        // sqrt(sum (|a-b|^2))
        ams_float acc = 0.0;
        for (int d = 0; d < n_features; ++d) {
            ams_float diff = a_row[d] - b_row[d];
            acc += diff * diff;
        }
        result = sqrt(acc);
    } else if (metric == AMS_DIST_MANHATTAN) {
        // sum (|a-b|)
        ams_float acc = 0.0;
        for (int d = 0; d < n_features; ++d) {
            ams_float diff = fabs(a_row[d] - b_row[d]);
            acc += diff;
        }
        result = acc;
    } else if (metric == AMS_DIST_CHEBYSHEV) {
        // max (|a-b|)
        ams_float maxv = 0.0;
        for (int d = 0; d < n_features; ++d) {
            ams_float diff = fabs(a_row[d] - b_row[d]);
            if (diff > maxv) {
                maxv = diff;
            }
        }
        result = maxv;
    } else if (metric == AMS_DIST_MINKOWSKI) {
        // (sum (|a-b|^p))^(1/p)
        ams_float acc = 0.0;
        for (int d = 0; d < n_features; ++d) {
            ams_float diff = fabs(a_row[d] - b_row[d]);
            acc += pow(diff, p);
        }
        result = pow(acc, 1.0 / p);
    }

    d_out[i * n_b + j] = result;
}

// -------------------------------------------------------------------------------------------------
// 3. LATTICE KERNELS (INNER PRODUCTS + NEAREST POINTS)
// -------------------------------------------------------------------------------------------------

enum AmsLatticeType : int {
    AMS_LATTICE_E8 = 0,
    AMS_LATTICE_D4 = 1,
    AMS_LATTICE_A3 = 2
};

// Inner products: given N vectors (N x D), compute N x N inner product matrix.
extern "C" __global__ void lattice_inner_products_kernel(
    const ams_float* __restrict__ d_vectors,
    ams_float* __restrict__ d_products,
    int n,
    int dim,
    int lattice_type
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    int j = blockIdx.y * blockDim.y + threadIdx.y; // col

    if (i >= n || j >= n) return;

    const ams_float* vi = d_vectors + i * dim;
    const ams_float* vj = d_vectors + j * dim;

    ams_float dot = 0.0;
    for (int d = 0; d < dim; ++d) {
        dot += vi[d] * vj[d];
    }

    // Lattice normalization approximations
    if (lattice_type == AMS_LATTICE_E8) {
        // E8: round to nearest multiple of 0.5
        ams_float val = dot * 2.0;
        val = rint(val); // nearest integer
        dot = val / 2.0;
    } else if (lattice_type == AMS_LATTICE_D4) {
        // D4: integer rounding
        dot = rint(dot);
    } else if (lattice_type == AMS_LATTICE_A3) {
        // A3 (FCC): nearest multiple of 1/6
        ams_float val = dot * 6.0;
        val = rint(val);
        dot = val / 6.0;
    }

    d_products[i * n + j] = dot;
}

// Nearest lattice points by simple rounding rules.
extern "C" __global__ void nearest_lattice_points_kernel(
    const ams_float* __restrict__ d_vectors,
    ams_float* __restrict__ d_out,
    int n,
    int dim,
    int lattice_type
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const ams_float* in_row  = d_vectors + i * dim;
    ams_float*       out_row = d_out     + i * dim;

    for (int d = 0; d < dim; ++d) {
        ams_float v = in_row[d];
        if (lattice_type == AMS_LATTICE_E8) {
            // Simple integer rounding as placeholder
            out_row[d] = rint(v);
        } else if (lattice_type == AMS_LATTICE_D4) {
            // nearest multiple of 0.5
            ams_float val = v * 2.0;
            val = rint(val);
            out_row[d] = val / 2.0;
        } else if (lattice_type == AMS_LATTICE_A3) {
            // nearest multiple of 1/3
            ams_float val = v * 3.0;
            val = rint(val);
            out_row[d] = val / 3.0;
        }
    }
}

// -------------------------------------------------------------------------------------------------
// 4. 2D CONVOLUTION (GEOMETRIC CONVOLUTION KERNEL)
// -------------------------------------------------------------------------------------------------
//
// d_input:  [H * W]
// d_kernel: [K * K] (kernel_size x kernel_size)
// d_out:    [outH * outW] where outH = (H - K) / stride + 1, outW = (W - K) / stride + 1
//
extern "C" __global__ void conv2d_kernel(
    const ams_float* __restrict__ d_input,
    const ams_float* __restrict__ d_kernel,
    ams_float* __restrict__ d_out,
    int input_h,
    int input_w,
    int kernel_size,
    int stride
) {
    int out_h = (input_h - kernel_size) / stride + 1;
    int out_w = (input_w - kernel_size) / stride + 1;

    int oy = blockIdx.y * blockDim.y + threadIdx.y; // output row
    int ox = blockIdx.x * blockDim.x + threadIdx.x; // output col

    if (oy >= out_h || ox >= out_w) return;

    ams_float sum = 0.0;
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            int iy = oy * stride + ky;
            int ix = ox * stride + kx;
            if (iy < input_h && ix < input_w) {
                ams_float v = d_input[iy * input_w + ix];
                ams_float w = d_kernel[ky * kernel_size + kx];
                sum += v * w;
            }
        }
    }

    d_out[oy * out_w + ox] = sum;
}

// -------------------------------------------------------------------------------------------------
// 5. MESH VERTEX NORMALS
// -------------------------------------------------------------------------------------------------
//
// vertices: [n_vertices * 3]
// faces:    [n_faces * 3] of uint32_t indices
// normals:  [n_vertices * 3], accumulated and normalized
//
// NOTE: This is a straightforward GPU version of the CPU logic in the Rust module.
//
__device__ inline void ams_cross_product(
    const ams_float* __restrict__ a,
    const ams_float* __restrict__ b,
    ams_float* __restrict__ out
) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

extern "C" __global__ void mesh_vertex_normals_kernel(
    const ams_float* __restrict__ d_vertices,
    const uint32_t* __restrict__ d_faces,
    ams_float* __restrict__ d_normals,
    int n_vertices,
    int n_faces
) {
    // Each thread handles one face and accumulates into normals using atomic adds.
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_faces) return;

    uint32_t i0 = d_faces[f * 3 + 0];
    uint32_t i1 = d_faces[f * 3 + 1];
    uint32_t i2 = d_faces[f * 3 + 2];

    if (i0 >= (uint32_t)n_vertices || i1 >= (uint32_t)n_vertices || i2 >= (uint32_t)n_vertices) {
        return;
    }

    const ams_float* v0 = d_vertices + i0 * 3;
    const ams_float* v1 = d_vertices + i1 * 3;
    const ams_float* v2 = d_vertices + i2 * 3;

    ams_float e1[3];
    ams_float e2[3];

    e1[0] = v1[0] - v0[0];
    e1[1] = v1[1] - v0[1];
    e1[2] = v1[2] - v0[2];

    e2[0] = v2[0] - v0[0];
    e2[1] = v2[1] - v0[1];
    e2[2] = v2[2] - v0[2];

    ams_float face_normal[3];
    ams_cross_product(e1, e2, face_normal);

    ams_float len_sq = face_normal[0] * face_normal[0]
                     + face_normal[1] * face_normal[1]
                     + face_normal[2] * face_normal[2];

    if (len_sq > 0.0) {
        ams_float inv_len = rsqrt(len_sq);
        face_normal[0] *= inv_len;
        face_normal[1] *= inv_len;
        face_normal[2] *= inv_len;
    } else {
        face_normal[0] = 0.0;
        face_normal[1] = 0.0;
        face_normal[2] = 0.0;
    }

    // Atomically accumulate into vertex normals
    atomicAdd(&d_normals[i0 * 3 + 0], face_normal[0]);
    atomicAdd(&d_normals[i0 * 3 + 1], face_normal[1]);
    atomicAdd(&d_normals[i0 * 3 + 2], face_normal[2]);

    atomicAdd(&d_normals[i1 * 3 + 0], face_normal[0]);
    atomicAdd(&d_normals[i1 * 3 + 1], face_normal[1]);
    atomicAdd(&d_normals[i1 * 3 + 2], face_normal[2]);

    atomicAdd(&d_normals[i2 * 3 + 0], face_normal[0]);
    atomicAdd(&d_normals[i2 * 3 + 1], face_normal[1]);
    atomicAdd(&d_normals[i2 * 3 + 2], face_normal[2]);
}

// Normalize vertex normals (invoke after mesh_vertex_normals_kernel).
__global__ void normalize_vertex_normals_kernel(
    ams_float* __restrict__ d_normals,
    int n_vertices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vertices) return;

    ams_float* vn = d_normals + i * 3;
    ams_float len_sq = vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2];
    if (len_sq > 0.0) {
        ams_float inv_len = rsqrt(len_sq);
        vn[0] *= inv_len;
        vn[1] *= inv_len;
        vn[2] *= inv_len;
    }
}

// -------------------------------------------------------------------------------------------------
// 6. GAUSSIAN KERNEL GENERATION (HOST-SIDE HELPER)
// -------------------------------------------------------------------------------------------------
//
// Pure CPU helper that mirrors the Rust `gaussian_kernel` function.
//
// size = 2*radius + 1, output is row-major [size * size].
//
extern "C"
void ams_gaussian_kernel(
    int radius,
    ams_float sigma,
    ams_float* out_kernel // size * size
) {
    int size = radius * 2 + 1;
    ams_float center = (ams_float)radius;

    ams_float sum = 0.0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            ams_float x = (ams_float)i - center;
            ams_float y = (ams_float)j - center;
            ams_float dist_sq = x * x + y * y;
            ams_float val = exp(-dist_sq / (2.0 * sigma * sigma));
            out_kernel[i * size + j] = val;
            sum += val;
        }
    }

    if (sum > 0.0) {
        ams_float inv_sum = 1.0 / sum;
        for (int idx = 0; idx < size * size; ++idx) {
            out_kernel[idx] *= inv_sum;
        }
    }
}

// -------------------------------------------------------------------------------------------------
// 7. OPTIONAL: MINIMAL HOST-LAUNCH HELPERS (C INTERFACE)
// -------------------------------------------------------------------------------------------------
//
// These are optional conveniences if you want to call from C/Rust via FFI without managing grid
// sizes manually. You can ignore them and just launch kernels directly if you prefer.
// -------------------------------------------------------------------------------------------------

extern "C"
void ams_launch_rbf_kernel(
    const ams_float* d_input,
    const ams_float* d_target,
    ams_float* d_output,
    int n_inputs,
    int n_targets,
    int n_features,
    ams_float sigma,
    cudaStream_t stream
) {
    dim3 block(16, 16, 1);
    dim3 grid(
        (n_targets  + block.x - 1) / block.x,
        (n_features + block.y - 1) / block.y,
        1
    );
    rbf_kernel<<<grid, block, 0, stream>>>(
        d_input, d_target, d_output,
        n_inputs, n_targets, n_features, sigma
    );
}

extern "C"
void ams_launch_distance_matrix(
    const ams_float* d_a,
    const ams_float* d_b,
    ams_float* d_out,
    int n_a,
    int n_b,
    int n_features,
    int metric,
    ams_float p,
    cudaStream_t stream
) {
    dim3 block(16, 16, 1);
    dim3 grid(
        (n_a + block.x - 1) / block.x,
        (n_b + block.y - 1) / block.y,
        1
    );
    distance_matrix_kernel<<<grid, block, 0, stream>>>(
        d_a, d_b, d_out, n_a, n_b, n_features, metric, p
    );
}

extern "C"
void ams_launch_lattice_inner_products(
    const ams_float* d_vectors,
    ams_float* d_products,
    int n,
    int dim,
    int lattice_type,
    cudaStream_t stream
) {
    dim3 block(16, 16, 1);
    dim3 grid(
        (n + block.x - 1) / block.x,
        (n + block.y - 1) / block.y,
        1
    );
    lattice_inner_products_kernel<<<grid, block, 0, stream>>>(
        d_vectors, d_products, n, dim, lattice_type
    );
}

extern "C"
void ams_launch_nearest_lattice_points(
    const ams_float* d_vectors,
    ams_float* d_out,
    int n,
    int dim,
    int lattice_type,
    cudaStream_t stream
) {
    dim3 block(256, 1, 1);
    dim3 grid((n + block.x - 1) / block.x, 1, 1);
    nearest_lattice_points_kernel<<<grid, block, 0, stream>>>(
        d_vectors, d_out, n, dim, lattice_type
    );
}

extern "C"
void ams_launch_conv2d(
    const ams_float* d_input,
    const ams_float* d_kernel,
    ams_float* d_out,
    int input_h,
    int input_w,
    int kernel_size,
    int stride,
    cudaStream_t stream
) {
    int out_h = (input_h - kernel_size) / stride + 1;
    int out_w = (input_w - kernel_size) / stride + 1;

    dim3 block(16, 16, 1);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        1
    );
    conv2d_kernel<<<grid, block, 0, stream>>>(
        d_input, d_kernel, d_out, input_h, input_w, kernel_size, stride
    );
}

extern "C"
void ams_launch_mesh_vertex_normals(
    const ams_float* d_vertices,
    const uint32_t* d_faces,
    ams_float* d_normals,
    int n_vertices,
    int n_faces,
    cudaStream_t stream
) {
    // Clear normals (caller can also memset if preferred).
    cudaMemsetAsync(d_normals, 0, sizeof(ams_float) * n_vertices * 3, stream);

    dim3 block_faces(256, 1, 1);
    dim3 grid_faces((n_faces + block_faces.x - 1) / block_faces.x, 1, 1);
    mesh_vertex_normals_kernel<<<grid_faces, block_faces, 0, stream>>>(
        d_vertices, d_faces, d_normals, n_vertices, n_faces
    );

    dim3 block_vertices(256, 1, 1);
    dim3 grid_vertices((n_vertices + block_vertices.x - 1) / block_vertices.x, 1, 1);
    normalize_vertex_normals_kernel<<<grid_vertices, block_vertices, 0, stream>>>(
        d_normals, n_vertices
    );
}
