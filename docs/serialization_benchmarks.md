# Serialization Performance Comparison: Fory vs Serde-JSON

**Date:** December 8, 2025  
**System:** Windows 11, Release Profile (optimized)  
**Test Subject:** `Gf8` (8-dimensional Galois Field scalar from Hydron E8 geometry engine)  
**Features:** Both benchmarks run with `hydron`, `fory`, and `simd` features enabled

## Executive Summary

This benchmark compares two serialization approaches for the `Gf8` type from the Hydron geometric mathematics engine, with SIMD optimizations enabled for fair comparison:

- **Fory**: A binary serialization format optimized for speed and compact representation
- **Serde-JSON**: Standard JSON text serialization via `serde_json`

## Performance Results

### Serialization (Encoding)

| Format      | Mean Time  | Performance      |
|-------------|------------|------------------|
| **Fory**    | 43.06 ns   | **1.61× faster** |
| Serde-JSON  | 69.35 ns   | baseline         |

### Deserialization (Decoding)

| Format      | Mean Time  | Performance      |
|-------------|------------|------------------|
| **Fory**    | 16.83 ns   | **4.71× faster** |
| Serde-JSON  | 79.29 ns   | baseline         |

### Combined Round-Trip Performance

| Format      | Total Time  | Performance      |
|-------------|-------------|------------------|
| **Fory**    | 59.89 ns    | **2.48× faster** |
| Serde-JSON  | 148.64 ns   | baseline         |

## Analysis

### Key Findings

1. **Deserialization Dominance**: Fory demonstrates exceptional deserialization performance, achieving **4.71× speedup** over JSON. This is particularly significant for read-heavy workloads common in geometric computation pipelines.

2. **Consistent Advantage**: Fory outperforms JSON in both encoding and decoding operations, with the deserialization gap being substantially larger than the serialization gap.

3. **Absolute Performance**: Both serializers operate at nanosecond scale, but Fory's sub-17ns deserialization is particularly impressive for a complex 8-dimensional geometric type.

4. **SIMD Impact**: With SIMD enabled for both benchmarks, the performance advantage remains consistent, demonstrating that Fory's gains come from format efficiency rather than just vectorization.

### Statistical Reliability

Both benchmarks showed excellent stability:

- **Fory**: 2-3% outliers (high mild), indicating consistent performance
- **Serde-JSON**: 3-8% outliers (high mild/severe), slightly more variance but still reliable

All measurements collected over 100 samples with proper warm-up cycles using Criterion's statistical methodology.

## Recommendations

### Use Fory When

- **Performance is critical**: Real-time geometric computations, hot-path operations
- **Binary data is acceptable**: Internal protocols, file formats, network transmission
- **Size matters**: Fory's binary format is more compact than JSON text
- **High deserialize frequency**: Read-heavy workloads benefit most from Fory's 4.71× advantage

### Use Serde-JSON When

- **Human readability required**: Configuration files, debugging output, APIs
- **Interoperability is key**: Cross-language communication, web APIs, standard protocols
- **Text-based workflows**: Version control, diffs, manual inspection
- **Performance is adequate**: 79ns deserialization is still fast for most applications

## Technical Context

### Gf8 Structure

The `Gf8` type represents an 8-dimensional vector in the E8 lattice, a fundamental geometric primitive in the Hydron mathematics engine. It's used throughout RUNE for:

- Geometric transformations and projections
- Fisher information geometry calculations
- Symplectic and hyperbolic space operations
- Topological analysis and persistent homology

### Implementation Notes

**Fory Benchmark:**

```rust
// Type registration with schema ID
fory.register::<Gf8>(10).unwrap();
let bytes = fory.serialize(&gf8).unwrap();
let decoded: Gf8 = fory.deserialize(&bytes).unwrap();
```

**Serde-JSON Benchmark:**

```rust
// Direct serialization to bytes
let bytes = serde_json::to_vec(&gf8).unwrap();
let decoded: Gf8 = serde_json::from_slice(&bytes).unwrap();
```

## Conclusion

For the Hydron geometric engine's `Gf8` type, **Fory provides substantial performance advantages** with 2.48× faster round-trip time. The deserialization speedup of 4.71× makes it particularly valuable for computational pipelines that repeatedly decode geometric data.

However, both serializers are performant enough for most use cases. The choice should primarily be driven by:

1. **Application requirements** (human-readable vs. binary)
2. **Workload characteristics** (read-heavy favors Fory)
3. **Integration constraints** (ecosystem compatibility)

For internal geometric computations and high-frequency operations, Fory is the clear winner. For external APIs and human-readable formats, Serde-JSON remains the pragmatic choice despite the performance gap.

---

*Benchmarks conducted with Criterion 0.8.1 using Plotters backend. All results represent mean values from 100 samples with outlier detection enabled.*
