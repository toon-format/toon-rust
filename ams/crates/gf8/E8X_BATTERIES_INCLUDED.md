# E8X - The Batteries-Included E8 Type

## TL;DR

**Use `E8X` for applications. Use `E8F` for low-level operations.**

```rust
use gf8::E8X;  // Single import, everything works!
```

## What is E8X?

E8X (E8 Cross) is the "batteries included" type that combines:
- **E8F core operations** (zero-FLOP arithmetic via lookup tables)
- **Automatic error management** (re-alignment after N operations)
- **Hybrid computation** (seamless E8F ↔ f32 conversion)
- **Drift tracking** (built-in metrics for error monitoring)

## The Problem E8X Solves

### Before E8X (Manual Wiring)

```rust
use gf8::E8F;                    // Core type
use gf8::aligned::E8FAligned;    // Error management
use gf8::compute::E8FCompute;    // Hybrid compute

// User has to wire these together manually!
let mut aligned = E8FAligned::new(E8F::new(42));
aligned.op(|e| e + E8F::new(10));
aligned.op(|e| e * E8F::new(5));

// Need to manually track drift
let coords = aligned.value().to_f32_coords();
// ... more manual work
```

### After E8X (Batteries Included)

```rust
use gf8::E8X;  // Single import!

// Everything works out of the box
let mut x = E8X::new_from_index(42);
x += E8X::new_from_index(10);  // Automatic re-alignment
x *= E8X::new_from_index(5);   // Drift tracking built-in

// Hybrid compute built-in
let coords = x.to_f32_coords();
println!("Max drift: {}", x.max_drift());
```

## Features

### 1. Zero-FLOP Operations

All E8F operations work via precomputed lookup tables:

```rust
let a = E8X::new_from_index(10);
let b = E8X::new_from_index(20);

let c = a + b;  // Lookup table addition
let d = a * b;  // Lookup table multiplication
let dot = a.dot(b);  // Lookup table dot product
```

### 2. Automatic Error Management

Re-alignment happens automatically after N operations (default: 10):

```rust
let mut x = E8X::new_from_index(0);

// Chain 20 operations - automatic re-alignment at 10 and 20
for i in 0..20 {
    x += E8X::new_from_index(i);
}

// Error is bounded
assert!(x.max_drift() < 0.3);
```

### 3. Hybrid Computation

Seamless conversion between E8F and f32:

```rust
let x = E8X::new_from_index(42);

// Convert to f32 for computation
let coords = x.to_f32_coords();

// Do f32 math
let scaled: [f32; 8] = coords.map(|c| c * 2.0);

// Convert back to E8X
let (y, error) = E8X::from_f32_coords(&scaled);
```

### 4. Drift Tracking

Built-in metrics for error monitoring:

```rust
let mut x = E8X::new_from_index(0);

for i in 0..10 {
    x += E8X::new_from_index(i * 10);
}

println!("Max drift: {:.4}", x.max_drift());
println!("Mean drift: {:.4}", x.mean_drift());
println!("Current drift: {:.4}", x.current_drift());
```

### 5. Batch Operations

Convenient batch processing:

```rust
let batch = vec![
    E8X::new_from_index(0),
    E8X::new_from_index(1),
    E8X::new_from_index(2),
];

// Convert to bytes for storage (3 bytes total)
let bytes = E8X::batch_to_bytes(&batch);

// Convert back
let recovered = E8X::batch_from_bytes(&bytes);
```

## When to Use E8X vs E8F

### Use E8X for:
- ✅ Applications and high-level code
- ✅ Media compression
- ✅ Neural networks
- ✅ Any code requiring robust E8 operations
- ✅ When you want automatic error management
- ✅ When you need drift tracking

### Use E8F for:
- ✅ Low-level operations
- ✅ Building your own wrappers
- ✅ Fine-grained control over alignment
- ✅ Performance-critical inner loops (no overhead)
- ✅ When you know operations won't accumulate error

## API Overview

### Constructors

```rust
// From E8F
let x = E8X::new(E8F::new(42));

// From index
let x = E8X::new_from_index(42);

// From f32 coordinates
let (x, error) = E8X::from_f32_coords(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

// From Gf8
let x = E8X::from_gf8(&gf8);

// With custom alignment threshold
let x = E8X::with_max_ops(E8F::new(42), 5);
```

### Arithmetic Operations

```rust
let a = E8X::new_from_index(10);
let b = E8X::new_from_index(20);

// Operator overloads
let c = a + b;
let d = a - b;
let e = a * b;

// Mutable operations
let mut x = a;
x += b;
x -= b;
x *= b;

// Method calls
x.add_e8x(b);
x.sub_e8x(b);
x.mul_e8x(b);

// Dot product
let dot = a.dot(b);  // Lookup table
let dot_f32 = a.dot_f32(&b);  // Exact f32
```

### Hybrid Computation

```rust
let x = E8X::new_from_index(42);

// E8X → f32
let coords = x.to_f32_coords();
let gf8 = x.to_gf8();
let u32_val = x.to_u32();

// f32 → E8X
let (y, error) = E8X::from_f32_coords(&coords);
```

### Error Management

```rust
let mut x = E8X::new_from_index(0);

// Check if alignment is needed
if x.needs_alignment() {
    x.align();  // Force re-alignment
}

// Get alignment settings
let max_ops = x.max_ops_before_align();
x.set_max_ops(5);  // Change threshold
```

### Drift Tracking

```rust
let x = E8X::new_from_index(0);

// Get drift metrics
let max = x.max_drift();
let mean = x.mean_drift();
let current = x.current_drift();

// Get operation count
let ops = x.ops_since_alignment();
```

### Batch Operations

```rust
// Batch conversion
let batch = vec![E8X::new_from_index(0), E8X::new_from_index(1)];
let bytes = E8X::batch_to_bytes(&batch);
let recovered = E8X::batch_from_bytes(&bytes);

// Weighted sum
let weights = vec![E8X::new_from_index(100), E8X::new_from_index(100)];
let values = vec![E8X::new_from_index(0), E8X::new_from_index(1)];
let (result, error) = E8X::weighted_sum(&weights, &values);
```

## Storage Size

E8X is a zero-cost wrapper over E8F:

```rust
assert_eq!(std::mem::size_of::<E8F>(), 1);   // 1 byte
assert_eq!(std::mem::size_of::<E8X>(), 24);  // 24 bytes (includes tracking)
```

For storage, convert to bytes:

```rust
let x = E8X::new_from_index(42);
let byte = x.index();  // 1 byte for storage
```

## Error Bounds

E8X inherits E8F's error bounds:

- **Single quantization**: ≤ 0.087 radians (chordal distance)
- **Roundtrip (E8X → f32 → E8X)**: ≤ 0.15 radians
- **Chain of 10 operations**: ≤ 0.3 radians (with automatic re-alignment)

## Example: Media Compression

```rust
use gf8::E8X;

// Compress image patch (256 x 8D vectors → 256 bytes)
let features: Vec<[f32; 8]> = extract_features(image_patch);

let compressed: Vec<E8X> = features.iter()
    .map(|coords| E8X::from_f32_coords(coords).0)
    .collect();

// Store as bytes (256 bytes total)
let bytes = E8X::batch_to_bytes(&compressed);

// Can perform operations directly on compressed data
let energy: f32 = compressed.iter()
    .map(|x| x.dot(*x))
    .sum();

// Decompress when needed
let reconstructed: Vec<[f32; 8]> = compressed.iter()
    .map(|x| x.to_f32_coords())
    .collect();
```

## Example: Neural Network

```rust
use gf8::E8X;

// Weights stored as E8X (1 byte each)
let mut weights: Vec<E8X> = load_compressed_weights();

// Forward pass with automatic re-alignment
for epoch in 0..100 {
    for (input, weight) in inputs.iter().zip(weights.iter_mut()) {
        *weight *= *input;  // Automatic re-alignment after 10 ops
    }

    // Check drift
    let max_drift = weights.iter()
        .map(|w| w.max_drift())
        .fold(0.0f32, f32::max);

    println!("Epoch {}: max drift = {:.4}", epoch, max_drift);
}
```

## Comparison with E8F

| Feature | E8F | E8X |
|---------|-----|-----|
| **Storage size** | 1 byte | 24 bytes (runtime) |
| **Arithmetic operations** | ✅ Zero-FLOP | ✅ Zero-FLOP |
| **Automatic re-alignment** | ❌ Manual | ✅ Automatic |
| **Drift tracking** | ❌ Manual | ✅ Built-in |
| **Hybrid compute** | ❌ Manual trait | ✅ Built-in methods |
| **Operator overloads** | ✅ Yes | ✅ Yes |
| **Batch operations** | ❌ Manual | ✅ Built-in |
| **Use case** | Low-level | Applications |

## Migration from E8F

E8X is a drop-in enhancement for E8F:

```rust
// Before (E8F)
use gf8::E8F;

let mut x = E8F::new(42);
x = x + E8F::new(10);
x = x * E8F::new(5);

// After (E8X)
use gf8::E8X;

let mut x = E8X::new_from_index(42);
x += E8X::new_from_index(10);  // Automatic re-alignment
x *= E8X::new_from_index(5);   // Drift tracking

// Check drift
println!("Max drift: {}", x.max_drift());
```

## Conclusion

E8X is the recommended type for most E8 applications. It provides:

1. ✅ **Single import** - `use gf8::E8X;` and you're done
2. ✅ **Automatic error management** - Re-alignment just works
3. ✅ **Hybrid computation** - Seamless E8F ↔ f32 conversion
4. ✅ **Drift tracking** - Built-in metrics
5. ✅ **Batch operations** - Convenient helpers
6. ✅ **Same compression** - 1 byte per value for storage

Use E8F when you need fine-grained control. Use E8X for everything else.
