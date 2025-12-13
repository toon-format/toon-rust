/* src/e8.rs */
//!▫~•◦-------------------------------‣
//! # E8 Geometric Generator
//!▫~•◦-------------------------------------------------------------------‣
//! Deterministic generation of the 240 E8 roots.

/// Generates the 240 E8 roots in the canonical deterministic order defined in Spec C.
pub fn generate_canonical_roots() -> Vec<[f32; 8]> {
    let mut roots = Vec::with_capacity(240);

    // 1. Type I Roots (112): ±e_i ± e_j (i < j)
    // Sorted lexicographically by (i, j, s_i, s_j)
    for i in 0..8 {
        for j in (i + 1)..8 {
            for &si in &[-1.0, 1.0] {
                for &sj in &[-1.0, 1.0] {
                    let mut v = [0.0; 8];
                    v[i] = si;
                    v[j] = sj;
                    roots.push(v);
                }
            }
        }
    }

    // 2. Type II Roots (128): 1/2(±1, ..., ±1) with even number of minus signs
    // Sorted lexicographically by sign bitmask (0..255)
    for mask in 0..256u16 {
        let mut v = [0.0; 8];
        let mut minus_count = 0;
        for k in 0..8 {
            if (mask >> k) & 1 == 1 {
                v[k] = -0.5;
                minus_count += 1;
            } else {
                v[k] = 0.5;
            }
        }

        if minus_count % 2 == 0 {
            roots.push(v);
        }
    }

    assert_eq!(roots.len(), 240);
    roots
}

/// Returns the 8 canonical simple roots (Taproots)
pub fn simple_roots() -> Vec<[f32; 8]> {
    vec![
        [ 1.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], // α1
        [ 0.0,  1.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0], // α2
        [ 0.0,  0.0,  1.0, -1.0,  0.0,  0.0,  0.0,  0.0], // α3
        [ 0.0,  0.0,  0.0,  1.0, -1.0,  0.0,  0.0,  0.0], // α4
        [ 0.0,  0.0,  0.0,  0.0,  1.0, -1.0,  0.0,  0.0], // α5
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -1.0,  0.0], // α6
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -1.0], // α7
        [ 0.5,  0.5,  0.5,  0.5, -0.5, -0.5, -0.5, -0.5], // α8
    ]
}

pub fn dot(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn add(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    let mut r = [0.0; 8];
    for i in 0..8 {
        r[i] = a[i] + b[i];
    }
    r
}

pub fn sub(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    let mut r = [0.0; 8];
    for i in 0..8 {
        r[i] = a[i] - b[i];
    }
    r
}

pub fn neg(a: &[f32; 8]) -> [f32; 8] {
    let mut r = [0.0; 8];
    for i in 0..8 {
        r[i] = -a[i];
    }
    r
}

pub fn distance_sq(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

// Snaps a vector to the nearest root in the provided list
pub fn snap_to_root(target: &[f32; 8], candidates: &[[f32; 8]]) -> [f32; 8] {
    let mut best_dist = f32::MAX;
    let mut best_root = candidates[0];

    for root in candidates {
        let d = distance_sq(target, root);
        if d < best_dist {
            best_dist = d;
            best_root = *root;
        }
    }
    best_root
}
