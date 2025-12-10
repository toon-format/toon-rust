//! RUNE Evaluation Engine - Runtime Value System
//!
//! Provides runtime evaluation for RUNE expressions, including:
//! - E8 geometric types (vectors, octonions)
//! - GF(8) Galois field arithmetic
//! - Context-aware evaluation based on root declarations
//! - Built-in operations that bridge RUNE into Hydron geometry layers
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, hash_map::Entry};
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};
use thiserror::Error;

// Hydron geometry layers from hydron-core
use hydron_core::{
    FisherLayer, Gf8, HyperbolicLayer, LorentzianCausalLayer, QuaternionOps, SpacetimePoint,
    SphericalLayer, SymplecticLayer, TopologicalLayer,
};

// Local SIMD implementations (fallback when feature is disabled)
pub fn gf8_add_simd(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    // Simple scalar implementation
    let mut result = [0.0f32; 8];
    for i in 0..8 {
        result[i] = a[i] + b[i];
    }
    result
}

pub fn gf8_sub_simd(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    // Simple scalar implementation
    let mut result = [0.0f32; 8];
    for i in 0..8 {
        result[i] = a[i] - b[i];
    }
    result
}

pub fn gf8_matvec_simd(matrix: &[[f32; 8]; 8], vec: &[f32; 8]) -> [f32; 8] {
    // Matrix-vector multiplication
    let mut result = [0.0f32; 8];
    for i in 0..8 {
        for j in 0..8 {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    result
}

pub fn gf8_norm2_simd(vec: &[f32; 8]) -> f32 {
    // Squared norm
    vec.iter().map(|x| x * x).sum()
}

pub fn gf8_dot_simd(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    // Dot product
    let mut sum = 0.0f32;
    for i in 0..8 {
        sum += a[i] * b[i];
    }
    sum
}

pub fn get_available_f32_256_intrinsics() -> Vec<&'static str> {
    // Return empty vec when SIMD not available
    vec![]
}

pub fn print_simd_capabilities() {
    // No-op when SIMD not available
}

use rune_hex::hex as hex_model;

/// Runtime value types in the E8 ecosystem
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    /// Boolean value
    Bool(bool),

    /// Floating-point number (for RUNE expressions)
    Float(f64),

    /// String value
    String(String),

    /// Array of values
    Array(Vec<Value>),

    /// Scalar numeric value (f32)
    Scalar(f32),

    /// 8-dimensional geometric float (canonical Gf8)
    Gf8(hydron_core::Gf8),

    /// Geo-Semantic lattice frame (allowed E8 root indices)
    Frame(Vec<u8>),

    /// Spatially indexed associative memory (E8 root -> data)
    Atlas(HashMap<u8, Vec<Value>>),

    /// Spacetime point (Lorentzian coords)
    Spacetime(SpacetimePoint),

    /// DomR result (dominant E8 roots)
    DomR(hex_model::DomR),

    /// 8-dimensional vector in E8 lattice
    Vec8([f32; 8]),

    /// 16-dimensional phase space vector (position + momentum)
    Vec16([f32; 16]),

    /// Octonion (8-dimensional non-associative algebra)
    Octonion(Octonion),

    /// Quaternion (4D rotation)
    Quaternion([f32; 4]),

    /// Symbolic reference (unevaluated)
    Symbol(String),

    /// 8x8 matrix (Fisher information, etc.)
    Matrix8x8([[f32; 8]; 8]),

    /// Betti numbers (topological invariants)
    Betti([u32; 3]),

    /// Collection of Vec8 points (for point clouds)
    PointCloud(Vec<[f32; 8]>),

    // --- Extended Types ---
    Integer(i128),
    Byte(u8),
    Char(char),
    Map(HashMap<String, Value>),
    Bytes(Vec<u8>),
    Null,
    Complex([f64; 2]),

    // Advanced Types
    BigInt(Vec<u64>),   // Arbitrary precision integer parts
    Decimal(i128, u32), // Mantissa, Scale (Decimal = m * 10^-s)

    // Structural Types
    Object(RuneObject),
    Enum(String, String, Option<Box<Value>>), // EnumName, Variant, Payload
    Union(Box<Value>),                        // Type-erased union value
    Struct(String, Vec<Value>),               // StructName, Tuple-like fields
    Tuple(Vec<Value>),
    Set(Vec<Value>), // Using Vec for set to allow non-hashable values (linear scan)

    // Functional & Async
    Function(RuneFunction),
    Lambda(RuneLambda),
    #[serde(skip)]
    Future(RuneFuture),
    #[serde(skip)]
    Stream(RuneStream),
    Promise(RunePromise),
    Coroutine(RuneCoroutine),

    // System
    Pointer(usize),
    Interface(String),            // Interface name/ID
    Class(String),                // Class name/ID
    Generic(String, Vec<String>), // Name, TypeParams

    /// Error value
    Error(String),
}

// --- Advanced Type Implementations ---

/// Glyph-capable structural algebra over runtime values.
pub trait RuneGeometric {
    /// Split-join (midpoint / meet) glyph `/\`.
    fn meet(&self, other: &Self) -> Result<Value, EvalError>
    where
        Self: Sized;

    /// Join-split (antipodal midpoint) glyph `\/`.
    fn join(&self, other: &Self) -> Result<Value, EvalError>
    where
        Self: Sized;

    /// Projection glyph `|\`.
    fn project(&self, target: &Self) -> Result<Value, EvalError>
    where
        Self: Sized;

    /// Rejection glyph `\|` (component orthogonal to target).
    fn reject(&self, target: &Self) -> Result<Value, EvalError>
    where
        Self: Sized;

    /// Universal distance glyph `|/` returning a scalar distance.
    fn distance(&self, other: &Self) -> Result<f32, EvalError>
    where
        Self: Sized;

    /// Structural match check for filtering.
    fn matches_pattern(&self, pattern: &Self) -> bool;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuneObject {
    pub class: String,
    pub fields: HashMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuneFunction {
    pub name: String,
    pub args: Vec<String>,
    pub body: String, // AST or Bytecode reference
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuneLambda {
    pub captures: HashMap<String, Value>,
    pub args: Vec<String>,
    pub body: String,
}

#[derive(Debug, Clone)]
pub struct RuneFuture {
    pub id: String,
    pub state: Arc<Mutex<FutureState>>,
}

impl PartialEq for RuneFuture {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

#[derive(Debug, Clone)]
pub enum FutureState {
    Pending,
    Resolved(Value),
    Rejected(String),
}

#[derive(Debug, Clone)]
pub struct RuneStream {
    pub id: String,
    pub buffer: Arc<Mutex<Vec<Value>>>,
}

impl PartialEq for RuneStream {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunePromise {
    pub id: String,
    // Promise is the write-side of a Future
}

impl PartialEq for RunePromise {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuneCoroutine {
    pub id: String,
    pub pc: usize, // Program counter
}

impl PartialEq for RuneCoroutine {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

/// Octonion representation: (scalar, 7 imaginary units)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Octonion {
    pub scalar: f32,
    pub i: [f32; 7], // e1, e2, e3, e4, e5, e6, e7
}

impl Octonion {
    /// Create a new octonion
    pub fn new(scalar: f32, i: [f32; 7]) -> Self {
        Self { scalar, i }
    }

    /// Create a real octonion (pure scalar)
    pub fn real(scalar: f32) -> Self {
        Self {
            scalar,
            i: [0.0; 7],
        }
    }

    /// Octonion multiplication (non-associative!)
    ///
    /// Implements full Fano-plane based multiplication:
    /// (a0 + a·e) * (b0 + b·e) =
    ///   (a0*b0 - a·b) + (a0*b + b0*a + a × b),
    /// where a × b is the G₂-invariant 7D cross product induced by the Fano plane.
    pub fn mul(&self, other: &Octonion) -> Octonion {
        let a0 = self.scalar;
        let b0 = other.scalar;
        let a = &self.i;
        let b = &other.i;

        // Scalar part: a0*b0 - a·b
        let mut scalar = a0 * b0;
        for k in 0..7 {
            scalar -= a[k] * b[k];
        }

        // Imaginary part: a0*b + b0*a + a × b
        let mut imag = [0.0f32; 7];

        // Linear terms a0*b + b0*a
        for k in 0..7 {
            imag[k] += a0 * b[k] + b0 * a[k];
        }

        // Cross product term a × b via Fano plane structure constants
        //
        // We encode the oriented Fano triples for the imaginary units e1..e7.
        // Indices 0..6 correspond to e1..e7.
        //
        // The triples below define:
        //   e_i * e_j =  e_k  if (i,j,k) in oriented triple
        //   e_j * e_i = -e_k  (anti-commutativity)
        //
        // The chosen convention is one standard G₂ / octonion orientation:
        //   (1,2,3), (1,4,5), (1,6,7),
        //   (2,4,6), (2,5,7), (3,4,7), (3,5,6)
        const FANO_TRIPLES: &[(usize, usize, usize)] = &[
            (0, 1, 2),
            (0, 3, 4),
            (0, 5, 6),
            (1, 3, 5),
            (1, 4, 6),
            (2, 3, 6),
            (2, 4, 5),
        ];

        // Helper: product of basis elements e_(i+1) * e_(j+1)
        // Returns (scalar_part, imag_basis) where imag_basis[k] is the coefficient of e_(k+1).
        fn basis_mul(i: usize, j: usize) -> (f32, [f32; 7]) {
            debug_assert!(i < 7 && j < 7);
            if i == j {
                // e_i * e_i = -1
                return (-1.0, [0.0; 7]);
            }

            for &(a, b, c) in FANO_TRIPLES.iter() {
                // e_a * e_b =  e_c, e_b * e_a = -e_c
                if i == a && j == b {
                    let mut v = [0.0f32; 7];
                    v[c] = 1.0;
                    return (0.0, v);
                }
                if i == b && j == a {
                    let mut v = [0.0f32; 7];
                    v[c] = -1.0;
                    return (0.0, v);
                }

                // e_b * e_c =  e_a, e_c * e_b = -e_a
                if i == b && j == c {
                    let mut v = [0.0f32; 7];
                    v[a] = 1.0;
                    return (0.0, v);
                }
                if i == c && j == b {
                    let mut v = [0.0f32; 7];
                    v[a] = -1.0;
                    return (0.0, v);
                }

                // e_c * e_a =  e_b, e_a * e_c = -e_b
                if i == c && j == a {
                    let mut v = [0.0f32; 7];
                    v[b] = 1.0;
                    return (0.0, v);
                }
                if i == a && j == c {
                    let mut v = [0.0f32; 7];
                    v[b] = -1.0;
                    return (0.0, v);
                }
            }

            // This should never be reached if FANO_TRIPLES covers all oriented pairs.
            (0.0, [0.0; 7])
        }

        // Accumulate a × b via bilinearity:
        // (∑ a_i e_i) * (∑ b_j e_j) = ∑_{i,j} a_i b_j (e_i * e_j)
        // We already handled the i == j scalar contribution above,
        // so here we only need i != j and only add imaginary parts.
        for i in 0..7 {
            if a[i] == 0.0 {
                continue;
            }
            for j in 0..7 {
                if b[j] == 0.0 || i == j {
                    continue;
                }
                let (_s_part, basis_vec) = basis_mul(i, j);
                let coeff = a[i] * b[j];

                // Only imaginary contributions are expected here (_s_part is 0.0 for i != j).
                for k in 0..7 {
                    imag[k] += coeff * basis_vec[k];
                }
            }
        }

        Octonion { scalar, i: imag }
    }

    /// Conjugate of octonion
    pub fn conjugate(&self) -> Octonion {
        let mut neg_i = self.i;
        for x in &mut neg_i {
            *x = -*x;
        }
        Octonion {
            scalar: self.scalar,
            i: neg_i,
        }
    }

    /// Norm (magnitude) of octonion
    pub fn norm(&self) -> f32 {
        let mut sum = self.scalar * self.scalar;
        for &x in &self.i {
            sum += x * x;
        }
        sum.sqrt()
    }
}

impl fmt::Display for Octonion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.scalar)?;
        for (i, &val) in self.i.iter().enumerate() {
            if val != 0.0 {
                write!(f, " + {}e{}", val, i + 1)?;
            }
        }
        Ok(())
    }
}

// Shared causal layer for Rune runtime (Value payloads)
static CAUSAL_LAYER: OnceLock<Mutex<LorentzianCausalLayer<Value>>> = OnceLock::new();

fn causal_layer() -> &'static Mutex<LorentzianCausalLayer<Value>> {
    CAUSAL_LAYER.get_or_init(|| Mutex::new(LorentzianCausalLayer::new()))
}

// Gf8 is imported from hydron-core via the module re-exports
// (see src/rune/hydron/mod.rs)

impl Value {
    /// Insert data into an Atlas at the location defined by a vector.
    pub fn atlas_insert(&mut self, key_vector: &Value, data: Value) -> Result<(), EvalError> {
        match self {
            Value::Atlas(map) => {
                let gf8 = match key_vector {
                    Value::Gf8(g) => *g,
                    Value::Vec8(v) => hydron_core::Gf8::new(*v),
                    _ => return Err(EvalError::TypeMismatch("Atlas key must be a vector".into())),
                };

                let (idx, _root) = gf8.quantize();

                match map.entry(idx) {
                    Entry::Occupied(mut e) => {
                        e.get_mut().push(data);
                    }
                    Entry::Vacant(e) => {
                        e.insert(vec![data]);
                    }
                }
                Ok(())
            }
            _ => Err(EvalError::TypeMismatch("Target is not an Atlas".into())),
        }
    }

    /// Recall data from an Atlas near the location defined by a vector.
    pub fn atlas_recall(&self, query_vector: &Value) -> Result<Value, EvalError> {
        match self {
            Value::Atlas(map) => {
                let gf8 = match query_vector {
                    Value::Gf8(g) => *g,
                    Value::Vec8(v) => hydron_core::Gf8::new(*v),
                    _ => {
                        return Err(EvalError::TypeMismatch(
                            "Atlas query must be a vector".into(),
                        ));
                    }
                };

                let (idx, _root) = gf8.quantize();

                if let Some(items) = map.get(&idx) {
                    Ok(Value::Array(items.clone()))
                } else {
                    Ok(Value::Array(vec![]))
                }
            }
            _ => Err(EvalError::TypeMismatch("Target is not an Atlas".into())),
        }
    }
    /// Add two values
    pub fn add(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a + b)),

            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(format!(
                        "Cannot add arrays of different lengths: {} and {}",
                        a.len(),
                        b.len()
                    )));
                }
                let mut result = Vec::new();
                for (va, vb) in a.iter().zip(b.iter()) {
                    result.push(va.add(vb)?);
                }
                Ok(Value::Array(result))
            }

            (Value::Vec8(a), Value::Vec8(b)) => {
                let mut result = [0.0; 8];
                for i in 0..8 {
                    result[i] = a[i] + b[i];
                }
                Ok(Value::Vec8(result))
            }

            (Value::Gf8(a), Value::Gf8(b)) => {
                #[cfg(feature = "simd")]
                {
                    let result_coords = gf8_add_simd(a.coords(), b.coords());
                    Ok(Value::Gf8(Gf8::new(result_coords)))
                }
                #[cfg(not(feature = "simd"))]
                {
                    Ok(Value::Gf8(*a + *b))
                }
            }

            (Value::Octonion(a), Value::Octonion(b)) => {
                let mut result_i = [0.0f32; 7];
                for (i, result) in result_i.iter_mut().enumerate() {
                    *result = a.i[i] + b.i[i];
                }
                Ok(Value::Octonion(Octonion {
                    scalar: a.scalar + b.scalar,
                    i: result_i,
                }))
            }

            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot add {:?} and {:?}",
                self, other
            ))),
        }
    }

    /// Multiply two values
    pub fn mul(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a * b)),

            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(format!(
                        "Cannot multiply arrays of different lengths: {} and {}",
                        a.len(),
                        b.len()
                    )));
                }
                let mut result = Vec::new();
                for (va, vb) in a.iter().zip(b.iter()) {
                    result.push(va.mul(vb)?);
                }
                Ok(Value::Array(result))
            }

            (Value::Scalar(s), Value::Vec8(v)) | (Value::Vec8(v), Value::Scalar(s)) => {
                let mut result = [0.0; 8];
                for i in 0..8 {
                    result[i] = v[i] * s;
                }
                Ok(Value::Vec8(result))
            }

            (Value::Gf8(a), Value::Scalar(s)) | (Value::Scalar(s), Value::Gf8(a)) => {
                Ok(Value::Gf8(*a * *s))
            }

            (Value::Octonion(a), Value::Octonion(b)) => Ok(Value::Octonion(a.mul(b))),

            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot multiply {:?} and {:?}",
                self, other
            ))),
        }
    }

    /// Subtract two values
    pub fn sub(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a - b)),

            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(format!(
                        "Cannot subtract arrays of different lengths: {} and {}",
                        a.len(),
                        b.len()
                    )));
                }
                let mut result = Vec::new();
                for (va, vb) in a.iter().zip(b.iter()) {
                    result.push(va.sub(vb)?);
                }
                Ok(Value::Array(result))
            }

            (Value::Vec8(a), Value::Vec8(b)) => {
                let mut result = [0.0; 8];
                for i in 0..8 {
                    result[i] = a[i] - b[i];
                }
                Ok(Value::Vec8(result))
            }

            (Value::Gf8(a), Value::Gf8(b)) => Ok(Value::Gf8(*a - *b)),

            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot subtract {:?} and {:?}",
                self, other
            ))),
        }
    }

    /// Divide two values
    pub fn div(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => {
                if *b == 0.0 {
                    return Err(EvalError::DivisionByZero);
                }
                Ok(Value::Float(a / b))
            }

            (Value::Scalar(a), Value::Scalar(b)) => {
                if *b == 0.0 {
                    return Err(EvalError::DivisionByZero);
                }
                Ok(Value::Scalar(a / b))
            }

            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(format!(
                        "Cannot divide arrays of different lengths: {} and {}",
                        a.len(),
                        b.len()
                    )));
                }
                let mut result = Vec::new();
                for (va, vb) in a.iter().zip(b.iter()) {
                    result.push(va.div(vb)?);
                }
                Ok(Value::Array(result))
            }

            (Value::Vec8(v), Value::Scalar(s)) => {
                if *s == 0.0 {
                    return Err(EvalError::DivisionByZero);
                }
                let mut result = [0.0; 8];
                for i in 0..8 {
                    result[i] = v[i] / s;
                }
                Ok(Value::Vec8(result))
            }

            (Value::Gf8(_a), Value::Gf8(_b)) => {
                // Division for geometric Gf8 not directly supported
                Err(EvalError::TypeMismatch(
                    "Division not supported for Gf8 geometric types".to_string(),
                ))
            }

            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot divide {:?} by {:?}",
                self, other
            ))),
        }
    }

    /// Power operation
    pub fn pow(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.powf(*b))),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a.powf(*b))),

            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot raise {:?} to power {:?}",
                self, other
            ))),
        }
    }

    /// Modulo operation
    pub fn modulo(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => {
                if *b == 0.0 {
                    return Err(EvalError::DivisionByZero);
                }
                Ok(Value::Float(a % b))
            }

            (Value::Scalar(a), Value::Scalar(b)) => {
                if *b == 0.0 {
                    return Err(EvalError::DivisionByZero);
                }
                Ok(Value::Scalar(a % b))
            }

            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot compute {:?} mod {:?}",
                self, other
            ))),
        }
    }

    /// Geometric midpoint (type-preserving where possible).
    /// Used by the `/\` and `/|` glyphs.
    pub fn geometric_midpoint(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            // Preserve Gf8 on the spherical manifold via SLERP at t = 0.5
            (Value::Gf8(a), Value::Gf8(b)) => Ok(Value::Gf8(a.spherical_slerp(b, 0.5))),

            // Quaternion midpoint via SLERP
            (Value::Quaternion(a), Value::Quaternion(b)) => {
                Ok(Value::Quaternion(QuaternionOps::slerp(a, b, 0.5)))
            }

            // Octonion linear average (vector space)
            (Value::Octonion(a), Value::Octonion(b)) => {
                let scalar = (a.scalar + b.scalar) * 0.5;
                let mut i = [0.0; 7];
                for k in 0..7 {
                    i[k] = (a.i[k] + b.i[k]) * 0.5;
                }
                Ok(Value::Octonion(Octonion { scalar, i }))
            }

            // Vec8 midpoint
            (Value::Vec8(a), Value::Vec8(b)) => {
                let mut res = [0.0; 8];
                for k in 0..8 {
                    res[k] = (a[k] + b[k]) * 0.5;
                }
                Ok(Value::Vec8(res))
            }

            // Vec16 midpoint
            (Value::Vec16(a), Value::Vec16(b)) => {
                let mut res = [0.0; 16];
                for k in 0..16 {
                    res[k] = (a[k] + b[k]) * 0.5;
                }
                Ok(Value::Vec16(res))
            }

            // Scalar/float midpoint returned as Scalar (geometry is f32-based)
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar((a + b) * 0.5)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Scalar((*a as f32 + *b as f32) * 0.5)),
            (Value::Scalar(a), Value::Float(b)) | (Value::Float(b), Value::Scalar(a)) => {
                Ok(Value::Scalar((*a + *b as f32) * 0.5))
            }

            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot compute midpoint of {:?} and {:?}",
                self, other
            ))),
        }
    }

    /// Antipodal midpoint (mean then negate). Used by `\/`.
    pub fn geometric_antipode_midpoint(&self, other: &Value) -> Result<Value, EvalError> {
        let mid = self.geometric_midpoint(other)?;
        mid.negate()
    }

    /// Project `self` onto `target` (|\ glyph).
    pub fn geometric_project(&self, target: &Value) -> Result<Value, EvalError> {
        match (self, target) {
            (Value::Gf8(v), Value::Gf8(u)) => {
                let dot = v.dot(u.coords());
                let base = *u.coords();
                Ok(Value::Vec8(base.map(|x| x * dot)))
            }
            (Value::Vec8(v), Value::Vec8(u)) => {
                let dot: f32 = v.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
                let norm_sq: f32 = u.iter().map(|a| a * a).sum();
                if norm_sq < 1e-9 {
                    return Ok(Value::Vec8([0.0; 8]));
                }
                let scale = dot / norm_sq;
                Ok(Value::Vec8((*u).map(|x| x * scale)))
            }

            (Value::Vec16(v), Value::Vec16(u)) => {
                let dot: f32 = v.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
                let norm_sq: f32 = u.iter().map(|a| a * a).sum();
                if norm_sq < 1e-9 {
                    return Ok(Value::Vec16([0.0; 16]));
                }
                let scale = dot / norm_sq;
                Ok(Value::Vec16(u.map(|x| x * scale)))
            }
            _ => Err(EvalError::TypeMismatch(format!(
                "Projection requires compatible vector types: {:?} -> {:?}",
                self, target
            ))),
        }
    }

    /// Reject `self` from `target` (component orthogonal to target). Used by `\|`.
    pub fn geometric_reject(&self, target: &Value) -> Result<Value, EvalError> {
        match (self, target) {
            (Value::Gf8(v), Value::Gf8(u)) => {
                let v_coords = *v.coords();
                let u_coords = *u.coords();
                let dot = v.dot(u.coords());
                let norm_sq: f32 = u_coords.iter().map(|x| x * x).sum();
                if norm_sq < 1e-9 {
                    return Ok(Value::Vec8(v_coords));
                }
                let scale = dot / norm_sq;
                let mut rej = [0.0f32; 8];
                for i in 0..8 {
                    rej[i] = v_coords[i] - u_coords[i] * scale;
                }
                Ok(Value::Vec8(rej))
            }
            (Value::Vec8(v), Value::Vec8(u)) => {
                let dot: f32 = v.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
                let norm_sq: f32 = u.iter().map(|a| a * a).sum();
                if norm_sq < 1e-9 {
                    return Ok(Value::Vec8(*v));
                }
                let scale = dot / norm_sq;
                let mut rej = [0.0f32; 8];
                for i in 0..8 {
                    rej[i] = v[i] - u[i] * scale;
                }
                Ok(Value::Vec8(rej))
            }

            (Value::Vec16(v), Value::Vec16(u)) => {
                let dot: f32 = v.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
                let norm_sq: f32 = u.iter().map(|a| a * a).sum();
                if norm_sq < 1e-9 {
                    return Ok(Value::Vec16(*v));
                }
                let scale = dot / norm_sq;
                let mut rej = [0.0f32; 16];
                for i in 0..16 {
                    rej[i] = v[i] - u[i] * scale;
                }
                Ok(Value::Vec16(rej))
            }
            _ => Err(EvalError::TypeMismatch(format!(
                "Rejection requires compatible vector types: {:?} ⟂ {:?}",
                self, target
            ))),
        }
    }

    /// Geometric distance (context-aware where possible). Used by `|/`.
    pub fn geometric_distance(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            // Spherical distance for Gf8
            (Value::Gf8(a), Value::Gf8(b)) => Ok(Value::Scalar(a.spherical_distance_to(b))),

            // Euclidean distance for Vec8
            (Value::Vec8(a), Value::Vec8(b)) => {
                let mut sum = 0.0f32;
                for i in 0..8 {
                    let d = a[i] - b[i];
                    sum += d * d;
                }
                Ok(Value::Scalar(sum.sqrt()))
            }

            // Euclidean distance for Vec16
            (Value::Vec16(a), Value::Vec16(b)) => {
                let mut sum = 0.0f32;
                for i in 0..16 {
                    let d = a[i] - b[i];
                    sum += d * d;
                }
                Ok(Value::Scalar(sum.sqrt()))
            }

            // Scalar distance (absolute difference)
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar((a - b).abs())),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Scalar((*a as f32 - *b as f32).abs())),
            (Value::Scalar(a), Value::Float(b)) | (Value::Float(b), Value::Scalar(a)) => {
                Ok(Value::Scalar((*a - *b as f32).abs()))
            }

            _ => Err(EvalError::TypeMismatch(format!(
                "Distance requires compatible geometric types: {:?} and {:?}",
                self, other
            ))),
        }
    }

    /// Negate a value (unary minus)
    pub fn negate(&self) -> Result<Value, EvalError> {
        match self {
            Value::Float(a) => Ok(Value::Float(-a)),
            Value::Scalar(a) => Ok(Value::Scalar(-a)),

            Value::Array(a) => {
                let mut result = Vec::new();
                for val in a.iter() {
                    result.push(val.negate()?);
                }
                Ok(Value::Array(result))
            }

            Value::Vec8(v) => {
                let mut result = [0.0; 8];
                for i in 0..8 {
                    result[i] = -v[i];
                }
                Ok(Value::Vec8(result))
            }

            Value::Vec16(v) => {
                let mut result = [0.0; 16];
                for i in 0..16 {
                    result[i] = -v[i];
                }
                Ok(Value::Vec16(result))
            }

            Value::Quaternion(q) => Ok(Value::Quaternion([-q[0], -q[1], -q[2], -q[3]])),

            Value::Gf8(g) => Ok(Value::Gf8(-*g)),

            Value::Octonion(o) => Ok(Value::Octonion(Octonion {
                scalar: -o.scalar,
                i: o.i.map(|x| -x),
            })),

            Value::Map(m) => {
                let mut out = HashMap::with_capacity(m.len());
                for (k, v) in m {
                    out.insert(k.clone(), v.negate()?);
                }
                Ok(Value::Map(out))
            }

            Value::Object(obj) => {
                let mut fields = HashMap::with_capacity(obj.fields.len());
                for (k, v) in &obj.fields {
                    fields.insert(k.clone(), v.negate()?);
                }
                Ok(Value::Object(RuneObject {
                    class: obj.class.clone(),
                    fields,
                }))
            }

            Value::Tuple(vals) => {
                let mut out = Vec::with_capacity(vals.len());
                for v in vals {
                    out.push(v.negate()?);
                }
                Ok(Value::Tuple(out))
            }

            Value::Struct(name, vals) => {
                let mut out = Vec::with_capacity(vals.len());
                for v in vals {
                    out.push(v.negate()?);
                }
                Ok(Value::Struct(name.clone(), out))
            }

            _ => Err(EvalError::TypeMismatch(format!("Cannot negate {:?}", self))),
        }
    }

    /// Less than comparison
    pub fn lt(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Bool(a < b)),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot compare {:?} < {:?}",
                self, other
            ))),
        }
    }

    /// Less than or equal comparison
    pub fn le(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Bool(a <= b)),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot compare {:?} <= {:?}",
                self, other
            ))),
        }
    }

    /// Greater than comparison
    pub fn gt(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Bool(a > b)),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot compare {:?} > {:?}",
                self, other
            ))),
        }
    }

    /// Greater than or equal comparison
    pub fn ge(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Bool(a >= b)),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot compare {:?} >= {:?}",
                self, other
            ))),
        }
    }

    /// Logical AND
    pub fn and(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a && *b)),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot apply AND to {:?} and {:?}",
                self, other
            ))),
        }
    }

    /// Logical OR
    pub fn or(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a || *b)),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot apply OR to {:?} and {:?}",
                self, other
            ))),
        }
    }
}

impl RuneGeometric for Value {
    /// Structural containment check: returns true if `self` matches the pattern structurally.
    /// Arrays: all elements must match any element in pattern array? Here we require same length and per-index match.
    /// Maps/Objects: pattern keys must exist in self with matching substructure.
    fn matches_pattern(&self, pattern: &Value) -> bool {
        match (self, pattern) {
            (Value::Map(m), Value::Map(p)) => {
                for (k, pv) in p {
                    if let Some(v) = m.get(k) {
                        if !v.matches_pattern(pv) {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                true
            }
            (Value::Object(o), Value::Object(p)) => {
                if o.class != p.class {
                    return false;
                }
                for (k, pv) in &p.fields {
                    if let Some(v) = o.fields.get(k) {
                        if !v.matches_pattern(pv) {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                true
            }
            (Value::Struct(name_a, a), Value::Struct(name_b, b)) => {
                if name_a != name_b || a.len() != b.len() {
                    return false;
                }
                a.iter()
                    .zip(b.iter())
                    .all(|(va, vb)| va.matches_pattern(vb))
            }
            (Value::Tuple(a), Value::Tuple(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                a.iter()
                    .zip(b.iter())
                    .all(|(va, vb)| va.matches_pattern(vb))
            }
            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                a.iter()
                    .zip(b.iter())
                    .all(|(va, vb)| va.matches_pattern(vb))
            }
            // Primitive equality fallback
            _ => self == pattern,
        }
    }
    fn meet(&self, other: &Self) -> Result<Value, EvalError> {
        match (self, other) {
            // Structural recursion
            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(
                        "Array length mismatch in structural glyph".into(),
                    ));
                }
                let mut out = Vec::with_capacity(a.len());
                for (va, vb) in a.iter().zip(b.iter()) {
                    out.push(va.meet(vb)?);
                }
                Ok(Value::Array(out))
            }

            (Value::Tuple(a), Value::Tuple(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(
                        "Tuple length mismatch in structural glyph".into(),
                    ));
                }
                let mut out = Vec::with_capacity(a.len());
                for (va, vb) in a.iter().zip(b.iter()) {
                    out.push(va.meet(vb)?);
                }
                Ok(Value::Tuple(out))
            }

            (Value::Struct(name_a, a), Value::Struct(name_b, b)) => {
                if name_a != name_b {
                    return Err(EvalError::TypeMismatch(format!(
                        "Struct name mismatch: {} vs {}",
                        name_a, name_b
                    )));
                }
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(
                        "Struct field arity mismatch in structural glyph".into(),
                    ));
                }
                let mut out = Vec::with_capacity(a.len());
                for (va, vb) in a.iter().zip(b.iter()) {
                    out.push(va.meet(vb)?);
                }
                Ok(Value::Struct(name_a.clone(), out))
            }

            (Value::Map(a), Value::Map(b)) => {
                let mut out = HashMap::new();
                for (k, va) in a {
                    if let Some(vb) = b.get(k) {
                        out.insert(k.clone(), va.meet(vb)?);
                    }
                }
                Ok(Value::Map(out))
            }

            (Value::Object(a), Value::Object(b)) => {
                if a.class != b.class {
                    return Err(EvalError::TypeMismatch(format!(
                        "Cannot merge different classes: {} and {}",
                        a.class, b.class
                    )));
                }
                let mut fields = HashMap::new();
                for (k, va) in &a.fields {
                    if let Some(vb) = b.fields.get(k) {
                        fields.insert(k.clone(), va.meet(vb)?);
                    }
                }
                Ok(Value::Object(RuneObject {
                    class: a.class.clone(),
                    fields,
                }))
            }

            // Leaf path
            _ => self.geometric_midpoint(other),
        }
    }

    fn join(&self, other: &Self) -> Result<Value, EvalError> {
        let mid = self.meet(other)?;
        mid.negate()
    }

    fn project(&self, target: &Self) -> Result<Value, EvalError> {
        match (self, target) {
            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(
                        "Array length mismatch in projection".into(),
                    ));
                }
                let mut out = Vec::with_capacity(a.len());
                for (va, vb) in a.iter().zip(b.iter()) {
                    out.push(va.project(vb)?);
                }
                Ok(Value::Array(out))
            }

            (Value::Tuple(a), Value::Tuple(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(
                        "Tuple length mismatch in projection".into(),
                    ));
                }
                let mut out = Vec::with_capacity(a.len());
                for (va, vb) in a.iter().zip(b.iter()) {
                    out.push(va.project(vb)?);
                }
                Ok(Value::Tuple(out))
            }

            (Value::Struct(name_a, a), Value::Struct(name_b, b)) => {
                if name_a != name_b {
                    return Err(EvalError::TypeMismatch(format!(
                        "Struct name mismatch: {} vs {}",
                        name_a, name_b
                    )));
                }
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(
                        "Struct field arity mismatch in projection".into(),
                    ));
                }
                let mut out = Vec::with_capacity(a.len());
                for (va, vb) in a.iter().zip(b.iter()) {
                    out.push(va.project(vb)?);
                }
                Ok(Value::Struct(name_a.clone(), out))
            }

            (Value::Map(a), Value::Map(b)) => {
                let mut out = HashMap::new();
                for (k, va) in a {
                    if let Some(vb) = b.get(k) {
                        out.insert(k.clone(), va.project(vb)?);
                    }
                }
                Ok(Value::Map(out))
            }

            (Value::Object(a), Value::Object(b)) => {
                if a.class != b.class {
                    return Err(EvalError::TypeMismatch(format!(
                        "Cannot project different classes: {} and {}",
                        a.class, b.class
                    )));
                }
                let mut fields = HashMap::new();
                for (k, va) in &a.fields {
                    if let Some(vb) = b.fields.get(k) {
                        fields.insert(k.clone(), va.project(vb)?);
                    }
                }
                Ok(Value::Object(RuneObject {
                    class: a.class.clone(),
                    fields,
                }))
            }

            _ => self.geometric_project(target),
        }
    }

    fn reject(&self, target: &Self) -> Result<Value, EvalError> {
        match (self, target) {
            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(
                        "Array length mismatch in rejection".into(),
                    ));
                }
                let mut out = Vec::with_capacity(a.len());
                for (va, vb) in a.iter().zip(b.iter()) {
                    out.push(va.reject(vb)?);
                }
                Ok(Value::Array(out))
            }

            (Value::Tuple(a), Value::Tuple(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(
                        "Tuple length mismatch in rejection".into(),
                    ));
                }
                let mut out = Vec::with_capacity(a.len());
                for (va, vb) in a.iter().zip(b.iter()) {
                    out.push(va.reject(vb)?);
                }
                Ok(Value::Tuple(out))
            }

            (Value::Struct(name_a, a), Value::Struct(name_b, b)) => {
                if name_a != name_b {
                    return Err(EvalError::TypeMismatch(format!(
                        "Struct name mismatch: {} vs {}",
                        name_a, name_b
                    )));
                }
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(
                        "Struct field arity mismatch in rejection".into(),
                    ));
                }
                let mut out = Vec::with_capacity(a.len());
                for (va, vb) in a.iter().zip(b.iter()) {
                    out.push(va.reject(vb)?);
                }
                Ok(Value::Struct(name_a.clone(), out))
            }

            (Value::Map(a), Value::Map(b)) => {
                let mut out = HashMap::new();
                for (k, va) in a {
                    if let Some(vb) = b.get(k) {
                        out.insert(k.clone(), va.reject(vb)?);
                    }
                }
                Ok(Value::Map(out))
            }

            (Value::Object(a), Value::Object(b)) => {
                if a.class != b.class {
                    return Err(EvalError::TypeMismatch(format!(
                        "Cannot reject across different classes: {} and {}",
                        a.class, b.class
                    )));
                }
                let mut fields = HashMap::new();
                for (k, va) in &a.fields {
                    if let Some(vb) = b.fields.get(k) {
                        fields.insert(k.clone(), va.reject(vb)?);
                    }
                }
                Ok(Value::Object(RuneObject {
                    class: a.class.clone(),
                    fields,
                }))
            }

            _ => self.geometric_reject(target),
        }
    }

    fn distance(&self, other: &Self) -> Result<f32, EvalError> {
        match (self, other) {
            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(
                        "Array length mismatch in distance".into(),
                    ));
                }
                let mut accum = 0.0f32;
                for (va, vb) in a.iter().zip(b.iter()) {
                    accum += va.distance(vb)?;
                }
                let n = a.len() as f32;
                Ok(if n > 0.0 { accum / n } else { 0.0 })
            }

            (Value::Tuple(a), Value::Tuple(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(
                        "Tuple length mismatch in distance".into(),
                    ));
                }
                let mut accum = 0.0f32;
                for (va, vb) in a.iter().zip(b.iter()) {
                    accum += va.distance(vb)?;
                }
                let n = a.len() as f32;
                Ok(if n > 0.0 { accum / n } else { 0.0 })
            }

            (Value::Struct(name_a, a), Value::Struct(name_b, b)) => {
                if name_a != name_b {
                    return Err(EvalError::TypeMismatch(format!(
                        "Struct name mismatch: {} vs {}",
                        name_a, name_b
                    )));
                }
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(
                        "Struct field arity mismatch in distance".into(),
                    ));
                }
                let mut accum = 0.0f32;
                for (va, vb) in a.iter().zip(b.iter()) {
                    accum += va.distance(vb)?;
                }
                let n = a.len() as f32;
                Ok(if n > 0.0 { accum / n } else { 0.0 })
            }

            (Value::Map(a), Value::Map(b)) => {
                let mut accum = 0.0f32;
                let mut count = 0usize;
                for (k, va) in a {
                    if let Some(vb) = b.get(k) {
                        accum += va.distance(vb)?;
                        count += 1;
                    }
                }
                Ok(if count > 0 { accum / count as f32 } else { 0.0 })
            }

            (Value::Object(a), Value::Object(b)) => {
                if a.class != b.class {
                    return Err(EvalError::TypeMismatch(format!(
                        "Cannot measure distance across different classes: {} and {}",
                        a.class, b.class
                    )));
                }
                let mut accum = 0.0f32;
                let mut count = 0usize;
                for (k, va) in &a.fields {
                    if let Some(vb) = b.fields.get(k) {
                        accum += va.distance(vb)?;
                        count += 1;
                    }
                }
                Ok(if count > 0 { accum / count as f32 } else { 0.0 })
            }

            _ => match self.geometric_distance(other)? {
                Value::Scalar(s) => Ok(s),
                Value::Float(f) => Ok(f as f32),
                other => Err(EvalError::TypeMismatch(format!(
                    "Distance expected scalar, got {:?}",
                    other
                ))),
            },
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(b) => write!(f, "{}", b),
            Value::Float(v) => write!(f, "{}", v),
            Value::String(s) => write!(f, "{}", s),
            Value::Array(arr) => {
                write!(f, "[")?;
                for (i, val) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", val)?;
                }
                write!(f, "]")
            }
            Value::Scalar(v) => write!(f, "{}", v),
            Value::Gf8(g) => write!(f, "Gf8({})", g.to_scalar()),
            Value::Vec8(v) => write!(
                f,
                "Vec8[{}, {}, {}, {}, {}, {}, {}, {}]",
                v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]
            ),
            Value::Vec16(v) => write!(
                f,
                "Vec16[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]",
                v[0],
                v[1],
                v[2],
                v[3],
                v[4],
                v[5],
                v[6],
                v[7],
                v[8],
                v[9],
                v[10],
                v[11],
                v[12],
                v[13],
                v[14],
                v[15]
            ),
            Value::Octonion(o) => write!(f, "{}", o),
            Value::Quaternion(q) => write!(f, "Quat[{}, {}, {}, {}]", q[0], q[1], q[2], q[3]),
            Value::Spacetime(p) => write!(
                f,
                "Spacetime[t={}, x1={}, x2={}, x3={}, x4={}, x5={}, x6={}, x7={}]",
                p.coords[0],
                p.coords[1],
                p.coords[2],
                p.coords[3],
                p.coords[4],
                p.coords[5],
                p.coords[6],
                p.coords[7]
            ),
            Value::DomR(d) => write!(
                f,
                "DomR(roots={}, scores={})",
                d.roots.len(),
                d.scores.len()
            ),
            Value::Frame(indices) => write!(f, "Frame({} indices)", indices.len()),
            Value::Atlas(map) => write!(f, "Atlas({} roots)", map.len()),
            Value::Symbol(s) => write!(f, "{}", s),
            Value::Matrix8x8(_) => write!(f, "Matrix8x8[...]"),
            Value::Betti(b) => write!(f, "Betti[{}, {}, {}]", b[0], b[1], b[2]),
            Value::PointCloud(points) => write!(f, "PointCloud[{} points]", points.len()),

            // Extended Types Display
            Value::Integer(i) => write!(f, "{}", i),
            Value::Byte(b) => write!(f, "0x{:02X}", b),
            Value::Char(c) => write!(f, "'{}'", c),
            Value::Map(m) => {
                write!(f, "{{")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Bytes(b) => write!(f, "Bytes[{}]", b.len()),
            Value::Null => write!(f, "null"),
            Value::Complex(c) => write!(f, "{} + {}i", c[0], c[1]),

            // Advanced Types
            Value::BigInt(parts) => write!(f, "BigInt({:?})", parts),
            Value::Decimal(m, s) => write!(f, "Decimal({}e-{})", m, s),

            // Structural
            Value::Object(obj) => write!(f, "Object({})", obj.class),
            Value::Enum(name, variant, _) => write!(f, "{}::{}", name, variant),
            Value::Union(val) => write!(f, "Union({})", val),
            Value::Struct(name, _) => write!(f, "Struct({})", name),
            Value::Tuple(vals) => write!(f, "Tuple({})", vals.len()),
            Value::Set(vals) => write!(f, "Set({})", vals.len()),

            // Functional & Async
            Value::Function(func) => write!(f, "Fn({})", func.name),
            Value::Lambda(_) => write!(f, "Lambda"),
            Value::Future(fut) => write!(f, "Future({})", fut.id),
            Value::Stream(s) => write!(f, "Stream({})", s.id),
            Value::Promise(p) => write!(f, "Promise({})", p.id),
            Value::Coroutine(c) => write!(f, "Coroutine({})", c.id),

            // System
            Value::Pointer(p) => write!(f, "Ptr(0x{:x})", p),
            Value::Interface(i) => write!(f, "Interface({})", i),
            Value::Class(c) => write!(f, "Class({})", c),
            Value::Generic(n, _) => write!(f, "Generic({})", n),

            Value::Error(e) => write!(f, "Error: {}", e),
        }
    }
}

/// Evaluation context with variable bindings and root context
#[derive(Debug, Clone)]
pub struct EvalContext {
    /// Variable bindings
    pub variables: HashMap<String, Value>,

    /// Semantic variable bindings (prefix:name -> value)
    pub semantic_vars: HashMap<String, Value>,

    /// Current root context (affects interpretation)
    root: Option<String>,
}

impl EvalContext {
    /// Create a new evaluation context
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            semantic_vars: HashMap::new(),
            root: None,
        }
    }

    /// Set the root context
    pub fn set_root(&mut self, root: String) {
        self.root = Some(root);
    }

    /// Get the current root context
    pub fn root(&self) -> Option<&str> {
        self.root.as_deref()
    }

    /// Bind a variable to a value
    pub fn bind(&mut self, name: String, value: Value) {
        self.variables.insert(name, value);
    }

    /// Look up a variable
    pub fn lookup(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }
}

impl Default for EvalContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluation errors
#[derive(Debug, Error)]
pub enum EvalError {
    #[error("Type mismatch: {0}")]
    TypeMismatch(String),

    #[error("Undefined variable: {0}")]
    UndefinedVariable(String),

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

// ===================================
// From trait implementations - automatic Value wrapping
// ===================================

impl From<[f32; 8]> for Value {
    fn from(arr: [f32; 8]) -> Self {
        Value::Vec8(arr)
    }
}

impl From<[f32; 16]> for Value {
    fn from(arr: [f32; 16]) -> Self {
        Value::Vec16(arr)
    }
}

impl From<[f32; 4]> for Value {
    fn from(arr: [f32; 4]) -> Self {
        Value::Quaternion(arr)
    }
}

impl From<[u32; 3]> for Value {
    fn from(arr: [u32; 3]) -> Self {
        Value::Betti(arr)
    }
}

// ===================================
// RuneBuiltin - Geometric Operation Dispatch
// ===================================

/// Built-in geometric operations that bridge RUNE into Hydron
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuneBuiltin {
    // Gf8 core operations
    Gf8Norm,      // [f32;8] → f32
    Gf8Normalize, // [f32;8] → [f32;8]
    Gf8Dot,       // [f32;8], [f32;8] → f32

    // Spherical (S7) operations
    S7Project,   // [f32;8] → [f32;8]
    S7Distance,  // [f32;8], [f32;8] → f32
    S7Slerp,     // [f32;8], [f32;8], f32 → [f32;8]
    S7Antipodal, // [f32;8] → [f32;8]
    S7Mean,      // [[f32;8]] → [f32;8]

    // Hyperbolic operations
    H8Distance,  // [f32;8], [f32;8] → f32
    H8MobiusAdd, // [f32;8], [f32;8] → [f32;8]

    // Fisher information geometry
    FisherDistance, // [f32;8], [f32;8] → f32
    FisherMatrix,   // [f32;8] → [[f32;8];8]
    KLDivergence,   // [f32;8], [f32;8] → f32
    FisherFilter,   // Array, threshold -> Array (novelty filter)

    // Quaternion operations
    QuatSlerp,     // [f32;4], [f32;4], f32 → [f32;4]
    QuatCompose,   // [f32;4], [f32;4] → [f32;4]
    QuatConjugate, // [f32;4] → [f32;4]

    // Symplectic operations
    SymHamiltonian, // [f32;16] → f32
    SymEvolveStep,  // [f32;16], f32 → [f32;16]

    // Lorentzian spacetime operations
    LorentzianCausal,   // [f32;8], [f32;8] → bool
    LorentzianDistance, // [f32;8], [f32;8] → f32
    CausalNow,          // () -> [f32;8]
    CausalEmit,         // any, optional root, optional causes -> event_id
    CausalLink,         // cause_id, effect_id -> ()
    CausalConePast,     // id -> [ids]
    CausalConeFuture,   // id -> [ids]
    CausalVerify,       // () -> bool
    Fold,               // [values], op -> value
    Filter,             // [values], pattern -> [values]
    AtlasNew,           // -> Atlas
    AtlasInsert,        // [Atlas, KeyVec, Data] -> Atlas
    AtlasRecall,        // [Atlas, QueryVec] -> Array
    Neighbors,          // Integer -> Array<Integer>
    Reflect,            // [Vec8, Vec8] -> Vec8 (Weyl reflection)
    Diffuse,            // [Array<f32;240], Scalar] -> Array<f32;240>

    // Topological operations
    TopoBetti,     // [[f32;8]] → [u32;3]
    TopoSignature, // [[f32;8]] → symbol

    // CUDA orchestration
    CudaVecDot,        // GPU row-wise dot
    CudaTopK,          // GPU top-k
    CudaDomR,          // GPU DomR (E8-native dominant roots)
    CudaArchetypeDomR, // Archetype dispatch for DomR

    // E8 graph/ontology helpers
    E8TypeI,      // Axes[] -> Vec<Vec8> (Type-I roots)
    E8TypeII,     // Axes[] -> Vec<Vec8> (Type-II spinors)
    E8EdgesWhere, // Vertices[] -> Edges[] (inner product rule)
    HexGraph,     // (vertices, edges, axes) -> Map graph

    // Perception operations
    Perceive, // String → Vec8 (Signal /\ Structure)
    // ASV / GeoSynthetic Vault builtins (rune-gsv)
    AsvStore,
    AsvGet,
    AsvQuery,
}

impl RuneBuiltin {
    /// Create a RuneBuiltin from a name string (case-insensitive).
    pub fn from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {

            // Perception operations
            "perceive" => Some(RuneBuiltin::Perceive),

            // ASV builtins
            "asv.store" => Some(RuneBuiltin::AsvStore),
            "asv.get" => Some(RuneBuiltin::AsvGet),
            "asv.query" => Some(RuneBuiltin::AsvQuery),

            // E8 graph helpers
            "e8typei" => Some(RuneBuiltin::E8TypeI),
            "e8typeii" => Some(RuneBuiltin::E8TypeII),
            "e8edgeswhere" => Some(RuneBuiltin::E8EdgesWhere),
            "e8edges" => Some(RuneBuiltin::E8EdgesWhere),
            "HexGraph" => Some(RuneBuiltin::HexGraph),
            "t:HexGraph" => Some(RuneBuiltin::HexGraph),

            // Gf8 operations
            "gf8norm" => Some(RuneBuiltin::Gf8Norm),
            "gf8normalize" => Some(RuneBuiltin::Gf8Normalize),
            "gf8dot" => Some(RuneBuiltin::Gf8Dot),

            // Spherical operations
            "s7project" => Some(RuneBuiltin::S7Project),
            "s7distance" => Some(RuneBuiltin::S7Distance),
            "s7slerp" => Some(RuneBuiltin::S7Slerp),
            "s7antipodal" => Some(RuneBuiltin::S7Antipodal),
            "s7mean" => Some(RuneBuiltin::S7Mean),

            // Hyperbolic operations
            "h8distance" => Some(RuneBuiltin::H8Distance),
            "h8mobiusadd" => Some(RuneBuiltin::H8MobiusAdd),

            // Fisher geometry
            "fisherdistance" => Some(RuneBuiltin::FisherDistance),
            "fishermatrix" => Some(RuneBuiltin::FisherMatrix),
            "kldivergence" => Some(RuneBuiltin::KLDivergence),
            "fisherfilter" => Some(RuneBuiltin::FisherFilter),

            // Quaternion operations
            "quatslerp" => Some(RuneBuiltin::QuatSlerp),
            "quatcompose" => Some(RuneBuiltin::QuatCompose),
            "quatconjugate" => Some(RuneBuiltin::QuatConjugate),

            // Symplectic operations
            "symhamiltonian" => Some(RuneBuiltin::SymHamiltonian),
            "symevolvestep" => Some(RuneBuiltin::SymEvolveStep),

            // Lorentzian operations
            "lorentziancausal" => Some(RuneBuiltin::LorentzianCausal),
            "lorentziandistance" => Some(RuneBuiltin::LorentzianDistance),
            "causalnow" => Some(RuneBuiltin::CausalNow),
            "causalemit" => Some(RuneBuiltin::CausalEmit),
            "causallink" => Some(RuneBuiltin::CausalLink),
            "causalconepast" => Some(RuneBuiltin::CausalConePast),
            "causalconefuture" => Some(RuneBuiltin::CausalConeFuture),
            "causalverify" => Some(RuneBuiltin::CausalVerify),
            "fold" => Some(RuneBuiltin::Fold),
            "filter" => Some(RuneBuiltin::Filter),
            "atlasnew" => Some(RuneBuiltin::AtlasNew),
            "atlasinsert" => Some(RuneBuiltin::AtlasInsert),
            "atlasrecall" => Some(RuneBuiltin::AtlasRecall),
            "neighbors" => Some(RuneBuiltin::Neighbors),
            "reflect" => Some(RuneBuiltin::Reflect),
            "diffuse" => Some(RuneBuiltin::Diffuse),

            // Topological operations
            "topobetti" => Some(RuneBuiltin::TopoBetti),
            "toposignature" => Some(RuneBuiltin::TopoSignature),

            // CUDA builtins (feature-gated at execution)
            "cuda:vecdot" => Some(RuneBuiltin::CudaVecDot),
            "cuda:topk" => Some(RuneBuiltin::CudaTopK),
            "cuda:domr" => Some(RuneBuiltin::CudaDomR),
            "cuda:archetype:domr" => Some(RuneBuiltin::CudaArchetypeDomR),
            
            _ => None,
        }
    }
}

impl EvalContext {
    /// Apply a built-in geometric operation
    ///
    /// This is the bridge layer that makes RUNE expressions actually drive Hydron geometry.
    pub fn apply_builtin(&self, op: RuneBuiltin, args: &[Value]) -> Result<Value, EvalError> {
        match op {
            // Spherical S7 operations
            RuneBuiltin::S7Project => {
                let v = expect_vec8(args.first())?;
                let projected = SphericalLayer::project(&v);
                Ok(Value::Vec8(projected))
            }

            RuneBuiltin::S7Distance => {
                let a = expect_vec8(args.first())?;
                let b = expect_vec8(args.get(1))?;
                let dist = SphericalLayer::distance(&a, &b);
                Ok(Value::Scalar(dist))
            }

            RuneBuiltin::S7Slerp => {
                let a = expect_vec8(args.first())?;
                let b = expect_vec8(args.get(1))?;
                let t = expect_scalar(args.get(2))?;
                let result = SphericalLayer::slerp(&a, &b, t);
                Ok(Value::Vec8(result))
            }

            // Quaternion operations
            RuneBuiltin::QuatSlerp => {
                let a = expect_quat(args.first())?;
                let b = expect_quat(args.get(1))?;
                let t = expect_scalar(args.get(2))?;
                let result = QuaternionOps::slerp(&a, &b, t);
                Ok(Value::Quaternion(result))
            }

            // Symplectic operations
            RuneBuiltin::SymHamiltonian => {
                let state = expect_vec16(args.first())?;
                let (q, p) = split_phase_space(&state);
                let layer = SymplecticLayer::new();
                let h = layer.hamiltonian(&q, &p);
                Ok(Value::Scalar(h))
            }

            RuneBuiltin::SymEvolveStep => {
                let state = expect_vec16(args.first())?;
                let dt = expect_scalar(args.get(1))?;
                let (mut q, mut p) = split_phase_space(&state);
                let layer = SymplecticLayer::new();
                layer.evolve(&mut q, &mut p, dt);
                let evolved = merge_phase_space(&q, &p);
                Ok(Value::Vec16(evolved))
            }

            // Topological operations
            RuneBuiltin::TopoBetti => {
                let points = extract_point_cloud(args)?;
                let mut layer = TopologicalLayer::new();
                for point in points {
                    layer.add_point(point);
                }
                layer.compute_betti_numbers(2.0, 10); // max_radius=2.0, steps=10
                Ok(Value::Betti(layer.betti))
            }

            RuneBuiltin::TopoSignature => {
                let points = extract_point_cloud(args)?;
                let mut layer = TopologicalLayer::new();
                for point in points {
                    layer.add_point(point);
                }
                layer.compute_betti_numbers(2.0, 10);
                let sig = format!("β={:?}", layer.betti);
                Ok(Value::Symbol(sig))
            }

            // Gf8 core operations
            RuneBuiltin::Gf8Norm => {
                let v = expect_vec8(args.first())?;
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                Ok(Value::Scalar(norm))
            }

            RuneBuiltin::Gf8Normalize => {
                let v = expect_vec8(args.first())?;
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    let normalized = v.map(|x| x / norm);
                    Ok(Value::Vec8(normalized))
                } else {
                    Ok(Value::Vec8(v))
                }
            }

            RuneBuiltin::Gf8Dot => {
                let a = expect_vec8(args.first())?;
                let b = expect_vec8(args.get(1))?;
                let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
                Ok(Value::Scalar(dot))
            }

            // Spherical operations
            RuneBuiltin::S7Antipodal => {
                let v = expect_vec8(args.first())?;
                let antipodal = v.map(|x| -x);
                Ok(Value::Vec8(antipodal))
            }

            RuneBuiltin::S7Mean => {
                let points = extract_point_cloud(args)?;
                if points.is_empty() {
                    return Err(EvalError::InvalidOperation(
                        "Cannot compute mean of empty point cloud".to_string(),
                    ));
                }
                let result = SphericalLayer::mean(&points);
                Ok(Value::Vec8(result))
            }

            // Hyperbolic operations
            RuneBuiltin::H8Distance => {
                let a = expect_vec8(args.first())?;
                let b = expect_vec8(args.get(1))?;
                let dist = HyperbolicLayer::distance(&a, &b);
                Ok(Value::Scalar(dist))
            }

            RuneBuiltin::H8MobiusAdd => {
                let a = expect_vec8(args.first())?;
                let b = expect_vec8(args.get(1))?;
                let result = HyperbolicLayer::mobius_add(&a, &b);
                Ok(Value::Vec8(result))
            }

            // Fisher information geometry
            RuneBuiltin::FisherDistance => {
                let a = expect_vec8(args.first())?;
                let b = expect_vec8(args.get(1))?;
                let dist = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt();
                Ok(Value::Scalar(dist))
            }

            RuneBuiltin::FisherMatrix => {
                let flat: Vec<Value> = (0..64)
                    .map(|i| {
                        let diag = if i / 8 == i % 8 { 1.0 } else { 0.0 };
                        Value::Scalar(diag)
                    })
                    .collect();
                Ok(Value::Array(flat))
            }

            RuneBuiltin::KLDivergence => {
                let p = expect_vec8(args.first())?;
                let q = expect_vec8(args.get(1))?;
                let kl = FisherLayer::kl_divergence(&p, &q);
                Ok(Value::Scalar(kl))
            }

            // Quaternion operations
            RuneBuiltin::QuatCompose => {
                let a = expect_quat(args.first())?;
                let b = expect_quat(args.get(1))?;
                // Quaternion multiplication: (a0,a)(b0,b) = (a0*b0 - a·b, a0*b + b0*a + a×b)
                let result = [
                    a[0] * b[0] - (a[1] * b[1] + a[2] * b[2] + a[3] * b[3]),
                    a[0] * b[1] + b[0] * a[1] + (a[2] * b[3] - a[3] * b[2]),
                    a[0] * b[2] + b[0] * a[2] + (a[3] * b[1] - a[1] * b[3]),
                    a[0] * b[3] + b[0] * a[3] + (a[1] * b[2] - a[2] * b[1]),
                ];
                Ok(Value::Quaternion(result))
            }

            RuneBuiltin::QuatConjugate => {
                let q = expect_quat(args.first())?;
                let conj = [q[0], -q[1], -q[2], -q[3]];
                Ok(Value::Quaternion(conj))
            }

            // Lorentzian operations
            RuneBuiltin::LorentzianCausal => {
                let a = expect_vec8(args.first())?;
                let b = expect_vec8(args.get(1))?;
                let interval = a[0].powi(2)
                    - a[1..]
                        .iter()
                        .zip(&b[1..])
                        .map(|(x, y)| (x - y).powi(2))
                        .sum::<f32>();
                Ok(Value::Bool(interval > 0.0))
            }

            RuneBuiltin::LorentzianDistance => {
                let a = expect_vec8(args.first())?;
                let b = expect_vec8(args.get(1))?;
                let dist = (a[0] - b[0]).powi(2)
                    - a[1..]
                        .iter()
                        .zip(&b[1..])
                        .map(|(x, y)| (x - y).powi(2))
                        .sum::<f32>();
                Ok(Value::Scalar(dist.abs().sqrt()))
            }

            // E8 graph helpers -------------------------------------------------
            RuneBuiltin::E8TypeI => {
                // Expect axes array of maps with at least "index" (0..7) and optional "weight"
                let axes_val = args.get(0).ok_or_else(|| {
                    EvalError::InvalidOperation("E8TypeI expects axes array".into())
                })?;
                let axes = parse_axes(axes_val)?;
                if axes.len() < 2 {
                    return Err(EvalError::InvalidOperation(
                        "E8TypeI requires at least two axes".into(),
                    ));
                }
                let mut verts = Vec::new();
                for a in 0..axes.len() {
                    for b in (a + 1)..axes.len() {
                        let (idx_a, w_a) = axes[a];
                        let (idx_b, w_b) = axes[b];
                        // four sign combinations
                        let signs = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)];
                        for (sa, sb) in signs {
                            let mut v = [0.0f32; 8];
                            v[idx_a] = w_a * sa;
                            v[idx_b] = w_b * sb;
                            normalize_vec8(&mut v);
                            verts.push(Value::Vec8(v));
                        }
                    }
                }
                Ok(Value::Array(verts))
            }

            RuneBuiltin::E8TypeII => {
                let axes_val = args.get(0).ok_or_else(|| {
                    EvalError::InvalidOperation("E8TypeII expects axes array".into())
                })?;
                let axes = parse_axes(axes_val)?;
                if axes.len() != 8 {
                    return Err(EvalError::InvalidOperation(
                        "E8TypeII expects exactly 8 axes".into(),
                    ));
                }
                let mut verts = Vec::new();
                // 2^8 sign patterns, keep even number of negatives (128 spinors)
                for mask in 0u16..256 {
                    let negs = mask.count_ones();
                    if negs % 2 != 0 {
                        continue;
                    }
                    let mut v = [0.0f32; 8];
                    for (i, (idx, w)) in axes.iter().enumerate() {
                        let sign = if (mask & (1 << i)) != 0 { -0.5 } else { 0.5 };
                        v[*idx] = sign * *w;
                    }
                    normalize_vec8(&mut v);
                    verts.push(Value::Vec8(v));
                }
                Ok(Value::Array(verts))
            }

            RuneBuiltin::E8EdgesWhere => {
                // Args: vertices array (Vec8/Gf8), optional threshold (default 0.5), optional tolerance
                let verts_val = args.get(0).ok_or_else(|| {
                    EvalError::InvalidOperation("E8EdgesWhere expects vertices array".into())
                })?;
                let verts = parse_vec8_list(verts_val)?;
                let threshold = match args.get(1) {
                    Some(Value::Scalar(s)) => *s,
                    Some(Value::Float(f)) => *f as f32,
                    _ => 0.5f32,
                };
                let tol = match args.get(2) {
                    Some(Value::Scalar(s)) => s.abs(),
                    Some(Value::Float(f)) => (*f as f32).abs(),
                    _ => 1e-4f32,
                };
                let mut edges = Vec::new();
                for i in 0..verts.len() {
                    for j in (i + 1)..verts.len() {
                        let dot: f32 = verts[i]
                            .iter()
                            .zip(verts[j].iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        if (dot - threshold).abs() <= tol {
                            let mut map = HashMap::new();
                            map.insert("u".to_string(), Value::Integer(i as i128));
                            map.insert("v".to_string(), Value::Integer(j as i128));
                            map.insert("strength".to_string(), Value::Scalar(dot));
                            map.insert("relationship".to_string(), Value::Symbol("Similar".into()));
                            edges.push(Value::Map(map));
                        }
                    }
                }
                Ok(Value::Array(edges))
            }

            RuneBuiltin::HexGraph => {
                // Accept tuple or array of three: (vertices, edges, axes)
                let pack = args.get(0).ok_or_else(|| {
                    EvalError::InvalidOperation("HexGraph expects (vertices, edges, axes)".into())
                })?;
                let (verts, edges, axes) = match pack {
                    Value::Tuple(v) if v.len() == 3 => (&v[0], &v[1], &v[2]),
                    Value::Array(v) if v.len() == 3 => (&v[0], &v[1], &v[2]),
                    _ => {
                        return Err(EvalError::InvalidOperation(
                            "HexGraph expects tuple/array of (vertices, edges, axes)".into(),
                        ));
                    }
                };
                let mut graph = HashMap::new();
                graph.insert("vertices".into(), verts.clone());
                graph.insert("edges".into(), edges.clone());
                graph.insert("axes".into(), axes.clone());
                graph.insert("domain_stats".into(), Value::Map(HashMap::new()));
                Ok(Value::Map(graph))
            }

            // Perception operations
            RuneBuiltin::Perceive => {
                let input = match args.first() {
                    Some(Value::String(s)) => s,
                    Some(Value::Symbol(s)) => s,
                    _ => return Err(EvalError::TypeMismatch("Perceive requires a string".into())),
                };

                let sig = crate::rune::hydron::perception::signal_encode(input.as_bytes());
                let morph = crate::rune::hydron::perception::morph_analyze(input);

                let v_sig = Value::Vec8(sig);
                let v_morph = Value::Vec8(morph);

                // Synthesis: Signal /\ Structure (Geometric Midpoint)
                v_sig.geometric_midpoint(&v_morph)
            }

            RuneBuiltin::CausalNow => {
                let layer = causal_layer();
                let guard = layer.lock().unwrap();
                let mut coords = [0.0f64; 8];
                coords[0] = guard.proper_time;
                Ok(Value::Spacetime(SpacetimePoint::new(coords)))
            }

            RuneBuiltin::CausalEmit => {
                if args.is_empty() {
                    return Err(EvalError::InvalidOperation(
                        "CausalEmit requires at least a payload".into(),
                    ));
                }
                let payload = args[0].clone();
                let mut idx = 1;
                let mut location: Option<SpacetimePoint> = None;
                let mut root: usize = 0;

                if let Some(arg1) = args.get(idx) {
                    if let Value::Spacetime(p) = arg1 {
                        location = Some(p.clone());
                        idx += 1;
                    } else if let Ok(r) = expect_scalar(Some(arg1)) {
                        root = r as usize;
                        idx += 1;
                    }
                }

                let causes: Vec<u64> = if let Some(cause_val) = args.get(idx) {
                    match cause_val {
                        Value::Array(arr) => {
                            let mut out = Vec::new();
                            for v in arr {
                                out.push(expect_id(v)?);
                            }
                            out
                        }
                        other => vec![expect_id(other)?],
                    }
                } else {
                    Vec::new()
                };

                let layer = causal_layer();
                let mut guard = layer.lock().unwrap();
                let id = guard.add_event(root, payload, &causes, location);
                Ok(Value::Integer(id as i128))
            }

            RuneBuiltin::CausalLink => {
                if args.len() < 2 {
                    return Err(EvalError::InvalidOperation(
                        "CausalLink requires cause and effect ids".into(),
                    ));
                }
                let cause = expect_id(&args[0])?;
                let effect = expect_id(&args[1])?;
                let layer = causal_layer();
                let mut guard = layer.lock().unwrap();
                guard
                    .add_link(cause, effect)
                    .map_err(|e| EvalError::InvalidOperation(e.to_string()))?;
                Ok(Value::Null)
            }

            RuneBuiltin::CausalConePast => {
                if args.is_empty() {
                    return Err(EvalError::InvalidOperation(
                        "CausalConePast requires an event id".into(),
                    ));
                }
                let id = expect_id(&args[0])?;
                let layer = causal_layer();
                let guard = layer.lock().unwrap();
                let cone = guard.past_light_cone(id);
                Ok(Value::Array(
                    cone.into_iter()
                        .map(|i| Value::Integer(i as i128))
                        .collect(),
                ))
            }

            RuneBuiltin::CausalConeFuture => {
                if args.is_empty() {
                    return Err(EvalError::InvalidOperation(
                        "CausalConeFuture requires an event id".into(),
                    ));
                }
                let id = expect_id(&args[0])?;
                let layer = causal_layer();
                let guard = layer.lock().unwrap();
                let cone = guard.future_light_cone(id);
                Ok(Value::Array(
                    cone.into_iter()
                        .map(|i| Value::Integer(i as i128))
                        .collect(),
                ))
            }

            RuneBuiltin::CausalVerify => {
                let layer = causal_layer();
                let guard = layer.lock().unwrap();
                Ok(Value::Bool(guard.verify_consistency()))
            }

            RuneBuiltin::FisherFilter => {
                if args.is_empty() {
                    return Err(EvalError::InvalidOperation(
                        "FisherFilter expects [values], optional threshold".into(),
                    ));
                }
                let list = match &args[0] {
                    Value::Array(arr) => arr,
                    _ => {
                        return Err(EvalError::TypeMismatch(
                            "FisherFilter first argument must be Array".into(),
                        ));
                    }
                };
                let threshold = args
                    .get(1)
                    .map(|v| expect_scalar(Some(v)))
                    .transpose()?
                    .unwrap_or(0.1);

                if list.is_empty() {
                    return Ok(Value::Array(Vec::new()));
                }

                let mut filtered = Vec::new();
                let mut prev_dist: Option<Vec<f32>> = None;
                for item in list.iter() {
                    let dist = value_to_distribution(item)?;
                    let keep = if let Some(prev) = &prev_dist {
                        let len = dist.len().min(prev.len());
                        let kl = FisherLayer::kl_divergence(&dist[..len], &prev[..len]);
                        kl > threshold
                    } else {
                        true
                    };
                    if keep {
                        filtered.push(item.clone());
                        prev_dist = Some(dist);
                    }
                }

                Ok(Value::Array(filtered))
            }

            RuneBuiltin::Fold => {
                if args.len() < 2 {
                    return Err(EvalError::InvalidOperation(
                        "Fold expects [values], operator".into(),
                    ));
                }
                let list = match &args[0] {
                    Value::Array(arr) => arr.clone(),
                    _ => {
                        return Err(EvalError::TypeMismatch(
                            "Fold first argument must be Array".into(),
                        ));
                    }
                };
                let op_name = match &args[1] {
                    Value::Symbol(s) | Value::String(s) => s.clone(),
                    _ => {
                        return Err(EvalError::TypeMismatch(
                            "Fold operator must be Symbol/String".into(),
                        ));
                    }
                };

                let mut iter = list.into_iter();
                let mut acc = match iter.next() {
                    Some(first) => first,
                    None => return Ok(args.get(2).cloned().unwrap_or(Value::Null)),
                };

                for item in iter {
                    acc = match op_name.as_str() {
                        "/\\" => acc.meet(&item)?,
                        "\\/" => acc.join(&item)?,
                        "\\|" => acc.reject(&item)?,
                        "|\\" => acc.project(&item)?,
                        _ => {
                            return Err(EvalError::InvalidOperation(format!(
                                "Unknown fold operator {}",
                                op_name
                            )));
                        }
                    };
                }
                Ok(acc)
            }

            RuneBuiltin::Filter => {
                if args.len() < 2 {
                    return Err(EvalError::InvalidOperation(
                        "Filter expects [values], pattern".into(),
                    ));
                }
                let list = match &args[0] {
                    Value::Array(arr) => arr,
                    _ => {
                        return Err(EvalError::TypeMismatch(
                            "Filter first argument must be Array".into(),
                        ));
                    }
                };
                let pattern = &args[1];
                let mut out = Vec::new();
                for item in list {
                    if item.matches_pattern(pattern) {
                        out.push(item.clone());
                    }
                }
                Ok(Value::Array(out))
            }

            RuneBuiltin::AtlasNew => Ok(Value::Atlas(HashMap::new())),

            RuneBuiltin::AtlasInsert => {
                if args.len() < 3 {
                    return Err(EvalError::InvalidOperation(
                        "AtlasInsert expects [Atlas, Vector, Data]".into(),
                    ));
                }
                let mut atlas = args[0].clone();
                let key_vec = &args[1];
                let data = args[2].clone();
                atlas.atlas_insert(key_vec, data)?;
                Ok(atlas)
            }

            RuneBuiltin::AtlasRecall => {
                if args.len() < 2 {
                    return Err(EvalError::InvalidOperation(
                        "AtlasRecall expects [Atlas, Vector]".into(),
                    ));
                }
                let atlas = &args[0];
                let query = &args[1];
                atlas.atlas_recall(query)
            }

            RuneBuiltin::Neighbors => {
                let idx = match args.get(0) {
                    Some(Value::Integer(i)) => *i as usize,
                    Some(Value::Scalar(s)) => *s as usize,
                    Some(Value::Float(f)) => *f as usize,
                    _ => {
                        return Err(EvalError::TypeMismatch(
                            "Neighbors expects a root index".into(),
                        ));
                    }
                };
                let neighbors = crate::rune::hydron::topology::get_neighbors(idx);
                let vals = neighbors
                    .into_iter()
                    .map(|i| Value::Integer(i as i128))
                    .collect();
                Ok(Value::Array(vals))
            }

            RuneBuiltin::Reflect => {
                let vec = expect_vec8(args.get(0))?;
                let mirror = expect_vec8(args.get(1))?;
                let reflected = crate::rune::hydron::topology::weyl_reflect(&vec, &mirror);
                Ok(Value::Vec8(reflected))
            }

            RuneBuiltin::Diffuse => {
                if args.len() < 2 {
                    return Err(EvalError::InvalidOperation(
                        "Diffuse expects [Array(240), rate]".into(),
                    ));
                }
                let energy = match &args[0] {
                    Value::Array(arr) => {
                        if arr.len() != 240 {
                            return Err(EvalError::TypeMismatch(
                                "Diffuse energy array must have length 240".into(),
                            ));
                        }
                        let mut out = [0.0f32; 240];
                        for (i, v) in arr.iter().enumerate() {
                            out[i] = match v {
                                Value::Scalar(s) => *s,
                                Value::Float(f) => *f as f32,
                                _ => {
                                    return Err(EvalError::TypeMismatch(
                                        "Diffuse energy entries must be numeric".into(),
                                    ));
                                }
                            };
                        }
                        out
                    }
                    _ => {
                        return Err(EvalError::TypeMismatch(
                            "Diffuse first argument must be Array".into(),
                        ));
                    }
                };

                let rate = expect_scalar(args.get(1).map(|v| v))?;
                let diffused = crate::rune::hydron::topology::diffuse_energy(&energy, rate);
                let vals = diffused
                    .iter()
                    .map(|f| Value::Scalar(*f))
                    .collect::<Vec<_>>();
                Ok(Value::Array(vals))
            }

            // CUDA builtins - use CUDA accelerator if available
            RuneBuiltin::CudaDomR | RuneBuiltin::CudaArchetypeDomR => {
                #[cfg(feature = "cuda")]
                {
                    match crate::rune::hydron::cuda::get_cuda_accelerator().execute_domr("CudaDomR", args) {
                        Ok(result) => return Ok(result),
                        Err(_) => {
                            // Fall through to CPU implementation below
                        }
                    }
                }
                
                // CPU implementation fallback
                let energy_vec = expect_energy(args.get(0))?;
                let n_dr = match args.get(1) {
                    Some(Value::Integer(i)) => *i as usize,
                    Some(Value::Scalar(s)) => *s as usize,
                    Some(Value::Float(f)) => *f as usize,
                    _ => 8usize, // default
                };

                let graph = hex_model::default_graph();
                let domr = hex_model::domr_cpu(graph, &energy_vec, n_dr)
                    .map_err(|e| EvalError::InvalidOperation(e.to_string()))?;
                Ok(Value::DomR(domr))
            }

            RuneBuiltin::CudaVecDot => Err(EvalError::UnsupportedOperation("CudaVecDot not implemented".into())),

            // ASV builtins
            RuneBuiltin::AsvStore => {
                #[cfg(feature = "gsv")]
                {
                    use serde_json::Value as JsonValue;
                    use rune_gsv::builtins as gsb;
                    // Expect: [intent, payload]
                    let intent = match args.get(0) {
                        Some(Value::String(s)) => s.clone(),
                        Some(Value::Symbol(s)) => s.clone(),
                        _ => return Err(EvalError::TypeMismatch("ASV.Store expects intent string as first arg".into())),
                    };
                    let payload_val = args.get(1).cloned().ok_or_else(|| EvalError::InvalidOperation("ASV.Store expects payload as second arg".into()))?;
                    let json_payload = hydron_value_to_serde_json(&payload_val)?;
                    let store_lock = rune_gsv::store::default_store();
                    let mut store = store_lock.write().unwrap();
                    match gsb::asv_store(&mut *store, &intent, json_payload) {
                        Ok(k) => Ok(Value::String(k)),
                        Err(e) => Err(EvalError::InvalidOperation(format!("ASV.Store: {}", e))),
                    }
                }
                #[cfg(not(feature = "gsv"))]
                {
                    Err(EvalError::UnsupportedOperation("ASV builtins require 'gsv' feature".to_string()))
                }
            }

            RuneBuiltin::AsvGet => {
                #[cfg(feature = "gsv")]
                {
                    use rune_gsv::builtins as gsb;
                    let intent = match args.get(0) {
                        Some(Value::String(s)) => s.clone(),
                        Some(Value::Symbol(s)) => s.clone(),
                        _ => return Err(EvalError::TypeMismatch("ASV.Get expects intent string as first arg".into())),
                    };
                    let store_lock = rune_gsv::store::default_store();
                    let store = store_lock.read().unwrap();
                    match gsb::asv_get(&*store, &intent) {
                        Ok(Some(v)) => Ok(serde_json_to_hydron_value(v).map_err(|e| EvalError::InvalidOperation(format!("ASV.Get conversion: {}", e)))?),
                        Ok(None) => Ok(Value::Null),
                        Err(e) => Err(EvalError::InvalidOperation(format!("ASV.Get: {}", e))),
                    }
                }
                #[cfg(not(feature = "gsv"))]
                {
                    Err(EvalError::UnsupportedOperation("ASV builtins require 'gsv' feature".to_string()))
                }
            }

            RuneBuiltin::AsvQuery => {
                #[cfg(feature = "gsv")]
                {
                    use rune_gsv::builtins as gsb;
                    let intent = match args.get(0) {
                        Some(Value::String(s)) => s.clone(),
                        Some(Value::Symbol(s)) => s.clone(),
                        _ => return Err(EvalError::TypeMismatch("ASV.Query expects intent string as first arg".into())),
                    };
                    let k = match args.get(1) {
                        Some(Value::Integer(i)) => *i as usize,
                        Some(Value::Scalar(s)) => *s as usize,
                        Some(Value::Float(f)) => *f as usize,
                        _ => 10usize,
                    };
                    let store_lock = rune_gsv::store::default_store();
                    let store = store_lock.read().unwrap();
                    match gsb::asv_query(&*store, &intent, k) {
                        Ok(v) => Ok(serde_json_to_hydron_value(v).map_err(|e| EvalError::InvalidOperation(format!("ASV.Query conversion: {}", e)))?),
                        Err(e) => Err(EvalError::InvalidOperation(format!("ASV.Query: {}", e))),
                    }
                }
                #[cfg(not(feature = "gsv"))]
                {
                    Err(EvalError::UnsupportedOperation("ASV builtins require 'gsv' feature".to_string()))
                }
            }

            RuneBuiltin::CudaTopK => Err(EvalError::UnsupportedOperation("CudaTopK not implemented".into())),
        }
    }

    /// Apply a builtin by string name; returns Err if unknown name.
    pub fn apply_builtin_by_name(&self, name: &str, args: &[Value]) -> Result<Value, EvalError> {
        if let Some(b) = RuneBuiltin::from_str(name) {
            self.apply_builtin(b, args)
        } else {
            Err(EvalError::InvalidOperation(format!(
                "Unknown builtin: {}",
                name
            )))
        }
    }
}

// ===================================
// Helper functions for type extraction
// ===================================

fn expect_vec8(val: Option<&Value>) -> Result<[f32; 8], EvalError> {
    match val {
        Some(Value::Vec8(v)) => Ok(*v),
        Some(Value::Array(arr)) => {
            // Ensure array has length 8 and extract floats
            if arr.len() != 8 {
                return Err(EvalError::TypeMismatch(format!(
                    "Expected Vec8 (array of 8 floats), got array length {}",
                    arr.len()
                )));
            }
            let mut v = [0.0f32; 8];
            for (i, elem) in arr.iter().enumerate() {
                match elem {
                    Value::Float(f) => v[i] = *f as f32,
                    Value::Scalar(s) => v[i] = *s,
                    _ => {
                        return Err(EvalError::TypeMismatch(format!(
                            "Expected numeric values for Vec8, found {:?}",
                            elem
                        )));
                    }
                }
            }
            Ok(v)
        }
        Some(other) => Err(EvalError::TypeMismatch(format!(
            "Expected Vec8, got {}",
            match other {
                Value::Scalar(_) => "Scalar",
                Value::Vec16(_) => "Vec16",
                Value::Quaternion(_) => "Quaternion",
                Value::Gf8(_) => "Gf8",
                Value::Octonion(_) => "Octonion",
                Value::Symbol(_) => "Symbol",
                Value::Matrix8x8(_) => "Matrix8x8",
                Value::Betti(_) => "Betti",
                _ => "unknown",
            }
        ))),
        None => Err(EvalError::InvalidOperation("Missing argument".to_string())),
    }
}

fn expect_vec16(val: Option<&Value>) -> Result<[f32; 16], EvalError> {
    match val {
        Some(Value::Vec16(v)) => Ok(*v),
        Some(_) => Err(EvalError::TypeMismatch("Expected Vec16".to_string())),
        None => Err(EvalError::InvalidOperation("Missing argument".to_string())),
    }
}

fn expect_quat(val: Option<&Value>) -> Result<[f32; 4], EvalError> {
    match val {
        Some(Value::Quaternion(q)) => Ok(*q),
        Some(_) => Err(EvalError::TypeMismatch("Expected Quaternion".to_string())),
        None => Err(EvalError::InvalidOperation("Missing argument".to_string())),
    }
}

fn expect_scalar(val: Option<&Value>) -> Result<f32, EvalError> {
    match val {
        Some(Value::Scalar(s)) => Ok(*s),
        Some(Value::Float(f)) => Ok(*f as f32),
        Some(_) => Err(EvalError::TypeMismatch(
            "Expected Scalar or Float".to_string(),
        )),
        None => Err(EvalError::InvalidOperation("Missing argument".to_string())),
    }
}

fn expect_id(val: &Value) -> Result<u64, EvalError> {
    match val {
        Value::Integer(i) => Ok(*i as u64),
        Value::Scalar(s) => Ok(*s as u64),
        Value::Float(f) => Ok(*f as u64),
        _ => Err(EvalError::TypeMismatch("Expected event id".to_string())),
    }
}

fn expect_energy(val: Option<&Value>) -> Result<Vec<f32>, EvalError> {
    let arr = match val {
        Some(Value::Array(arr)) => arr,
        Some(_) => {
            return Err(EvalError::TypeMismatch(
                "Energy must be an Array of 240 numeric values".into(),
            ));
        }
        None => {
            return Err(EvalError::InvalidOperation(
                "Missing energy argument".into(),
            ));
        }
    };
    if arr.len() != 240 {
        return Err(EvalError::TypeMismatch(format!(
            "Energy array must have length 240, got {}",
            arr.len()
        )));
    }
    let mut out = Vec::with_capacity(240);
    for v in arr {
        match v {
            Value::Scalar(s) => out.push(*s),
            Value::Float(f) => out.push(*f as f32),
            Value::Integer(i) => out.push(*i as f32),
            _ => {
                return Err(EvalError::TypeMismatch(
                    "Energy entries must be numeric".into(),
                ));
            }
        }
    }
    Ok(out)
}

// Convert a hydron Value into serde_json::Value
fn hydron_value_to_serde_json(val: &Value) -> Result<serde_json::Value, EvalError> {
    match val {
        Value::Null => Ok(serde_json::Value::Null),
        Value::Float(f) => serde_json::Number::from_f64(*f as f64)
            .map(serde_json::Value::Number)
            .ok_or_else(|| EvalError::InvalidOperation("Float cannot be represented in JSON".to_string())),
        Value::Scalar(s) => serde_json::Number::from_f64(*s as f64)
            .map(serde_json::Value::Number)
            .ok_or_else(|| EvalError::InvalidOperation("Scalar cannot be represented in JSON".to_string())),
        Value::Integer(i) => Ok(serde_json::Value::Number((*i as i64).into())),
        Value::String(s) | Value::Symbol(s) => Ok(serde_json::Value::String(s.clone())),
        Value::Bool(b) => Ok(serde_json::Value::Bool(*b)),
        Value::Array(arr) | Value::Tuple(arr) => {
            let mut vec = Vec::with_capacity(arr.len());
            for v in arr.iter() { vec.push(hydron_value_to_serde_json(v)?); }
            Ok(serde_json::Value::Array(vec))
        }
        Value::Map(map) => {
            let mut obj = serde_json::Map::new();
            for (k, v) in map.iter() {
                obj.insert(k.clone(), hydron_value_to_serde_json(v)?);
            }
            Ok(serde_json::Value::Object(obj))
        }
        Value::Vec8(v) => {
            let arr = v.iter().map(|x| serde_json::Value::Number(serde_json::Number::from_f64(*x as f64).unwrap_or(serde_json::Number::from(0)))).collect();
            Ok(serde_json::Value::Array(arr))
        }
        Value::Vec16(v) => {
            let arr = v.iter().map(|x| serde_json::Value::Number(serde_json::Number::from_f64(*x as f64).unwrap_or(serde_json::Number::from(0)))).collect();
            Ok(serde_json::Value::Array(arr))
        }
        Value::Quaternion(q) => {
            let arr = q.iter().map(|x| serde_json::Value::Number(serde_json::Number::from_f64(*x as f64).unwrap_or(serde_json::Number::from(0)))).collect();
            Ok(serde_json::Value::Array(arr))
        }
        // For other types, fallback to string representation
        other => Ok(serde_json::Value::String(format!("{:?}", other))),
    }
}

// Convert serde_json::Value into hydron Value
fn serde_json_to_hydron_value(val: serde_json::Value) -> Result<Value, String> {
    match val {
        serde_json::Value::Null => Ok(Value::Null),
        serde_json::Value::Bool(b) => Ok(Value::Bool(b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() { Ok(Value::Integer(i as i128)) }
            else if let Some(f) = n.as_f64() { Ok(Value::Float(f)) }
            else { Err("JSON number neither i64 nor f64".into()) }
        }
        serde_json::Value::String(s) => Ok(Value::String(s)),
        serde_json::Value::Array(arr) => {
            let mut vals = Vec::new();
            for v in arr { vals.push(serde_json_to_hydron_value(v)?); }
            Ok(Value::Array(vals))
        }
        serde_json::Value::Object(obj) => {
            let mut map = HashMap::new();
            for (k, v) in obj { map.insert(k, serde_json_to_hydron_value(v)?); }
            Ok(Value::Map(map))
        }
    }
}



fn value_to_distribution(val: &Value) -> Result<Vec<f32>, EvalError> {
    let mut dist = match val {
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                match v {
                    Value::Scalar(s) => out.push(*s),
                    Value::Float(f) => out.push(*f as f32),
                    _ => {
                        return Err(EvalError::TypeMismatch(
                            "Array must contain numeric values for FisherFilter".into(),
                        ));
                    }
                }
            }
            out
        }
        Value::Vec8(v) => v.to_vec(),
        Value::Vec16(v) => v.to_vec(),
        Value::Gf8(g) => g.coords().to_vec(),
        Value::Quaternion(q) => q.to_vec(),
        Value::Scalar(s) => vec![*s],
        Value::Float(f) => vec![*f as f32],
        _ => {
            return Err(EvalError::TypeMismatch(
                "Unsupported value for FisherFilter distribution".into(),
            ));
        }
    };

    let sum: f32 = dist.iter().map(|x| x.abs()).sum();
    if sum > 1e-8 {
        for x in dist.iter_mut() {
            *x /= sum;
        }
    }
    Ok(dist)
}

fn extract_point_cloud(args: &[Value]) -> Result<Vec<[f32; 8]>, EvalError> {
    // Handle multiple argument formats:
    // 1. Single PointCloud value
    // 2. Single Vec16 (two packed points)
    // 3. Multiple Vec8 arguments

    if args.is_empty() {
        return Err(EvalError::InvalidOperation(
            "No points provided".to_string(),
        ));
    }

    // Case 1: PointCloud value
    if args.len() == 1 {
        if let Value::PointCloud(points) = &args[0] {
            return Ok(points.clone());
        }

        // Case 2: Vec16 (two packed points)
        if let Value::Vec16(v16) = &args[0] {
            let p1 = [
                v16[0], v16[1], v16[2], v16[3], v16[4], v16[5], v16[6], v16[7],
            ];
            let p2 = [
                v16[8], v16[9], v16[10], v16[11], v16[12], v16[13], v16[14], v16[15],
            ];
            return Ok(vec![p1, p2]);
        }
    }

    // Case 3: Multiple Vec8 arguments
    let mut points = Vec::new();
    for arg in args {
        match arg {
            Value::Vec8(v) => points.push(*v),
            Value::PointCloud(pc) => points.extend_from_slice(pc),
            _ => {
                return Err(EvalError::TypeMismatch(
                    "Expected Vec8, Vec16, or PointCloud for point cloud".to_string(),
                ));
            }
        }
    }
    Ok(points)
}

/// Normalize a mutable Vec8 in-place; no-op for near-zero norm.
fn normalize_vec8(v: &mut [f32; 8]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if norm_sq > 1e-9 {
        let inv = 1.0 / norm_sq.sqrt();
        for x in v {
            *x *= inv;
        }
    }
}

/// Extract axes as (index, weight) pairs from a Value::Array of maps.
fn parse_axes(val: &Value) -> Result<Vec<(usize, f32)>, EvalError> {
    let mut axes = Vec::new();
    let arr = match val {
        Value::Array(a) => a,
        _ => return Err(EvalError::TypeMismatch("Axes must be an array".into())),
    };
    for item in arr {
        match item {
            Value::Map(m) => {
                let idx_val = m
                    .get("index")
                    .ok_or_else(|| EvalError::TypeMismatch("Axis missing 'index' field".into()))?;
                let idx = expect_id(idx_val)? as usize;
                let w = match m.get("weight") {
                    Some(Value::Scalar(s)) => *s,
                    Some(Value::Float(f)) => *f as f32,
                    _ => 1.0,
                };
                axes.push((idx, w));
            }
            _ => return Err(EvalError::TypeMismatch("Axis must be a map".into())),
        }
    }
    // Deterministic ordering
    axes.sort_by_key(|(idx, _)| *idx);
    Ok(axes)
}

/// Parse a list of 8D vectors from various value representations.
fn parse_vec8_list(val: &Value) -> Result<Vec<[f32; 8]>, EvalError> {
    match val {
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                out.push(expect_vec8(Some(v))?);
            }
            Ok(out)
        }
        Value::PointCloud(points) => Ok(points.clone()),
        _ => Err(EvalError::TypeMismatch(
            "Expected Array of Vec8 for vertices".into(),
        )),
    }
}

/// Split Vec16 phase space into position and momentum
fn split_phase_space(state: &[f32; 16]) -> ([f32; 8], [f32; 8]) {
    let mut q = [0.0f32; 8];
    let mut p = [0.0f32; 8];
    q.copy_from_slice(&state[..8]);
    p.copy_from_slice(&state[8..]);
    (q, p)
}

/// Merge position and momentum into Vec16 phase space
fn merge_phase_space(q: &[f32; 8], p: &[f32; 8]) -> [f32; 16] {
    let mut state = [0.0f32; 16];
    state[..8].copy_from_slice(q);
    state[8..].copy_from_slice(p);
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_arithmetic() {
        let a = Value::Scalar(5.0);
        let b = Value::Scalar(3.0);

        assert_eq!(a.add(&b).unwrap(), Value::Scalar(8.0));
        assert_eq!(a.mul(&b).unwrap(), Value::Scalar(15.0));
        assert_eq!(a.sub(&b).unwrap(), Value::Scalar(2.0));
    }

    #[test]
    fn test_gf8_arithmetic() {
        use crate::rune::hydron::Gf8;

        // Test Gf8 addition (geometric addition on unit sphere)
        let gf_a = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let gf_b = Gf8::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let a = Value::Gf8(gf_a);
        let b = Value::Gf8(gf_b);

        // Geometric Gf8 addition
        let result = a.add(&b).unwrap();
        assert!(matches!(result, Value::Gf8(_)));
    }

    #[test]
    fn test_octonion_multiplication() {
        let a = Octonion::real(2.0);
        let b = Octonion::real(3.0);
        let c = a.mul(&b);

        assert_eq!(c.scalar, 6.0);
    }

    #[test]
    fn test_vec8_operations() {
        let a = Value::Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = Value::Vec8([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let result = a.add(&b).unwrap();
        if let Value::Vec8(v) = result {
            assert_eq!(v[0], 2.0);
            assert_eq!(v[7], 9.0);
        }
    }

    // ===================================
    // Integration Tests: RUNE → Hydron Geometry
    // ===================================

    #[test]
    fn test_rune_drives_spherical_geometry() {
        let ctx = EvalContext::new();

        // Test S7 projection
        let v = Value::Vec8([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let result = ctx.apply_builtin(RuneBuiltin::S7Project, &[v]).unwrap();

        if let Value::Vec8(projected) = result {
            // Should be normalized to unit sphere
            let norm: f32 = projected.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "S7 projection should normalize");
        } else {
            panic!("Expected Vec8 result from S7Project");
        }

        // Test S7 distance
        let a = Value::Vec8([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = Value::Vec8([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let dist = ctx
            .apply_builtin(RuneBuiltin::S7Distance, &[a.clone(), b.clone()])
            .unwrap();

        if let Value::Scalar(d) = dist {
            assert!(
                d > 0.0,
                "Distance between distinct points should be positive"
            );
        }

        // Test S7 slerp
        let t = Value::Scalar(0.5);
        let interp = ctx.apply_builtin(RuneBuiltin::S7Slerp, &[a, b, t]).unwrap();

        assert!(matches!(interp, Value::Vec8(_)), "Slerp should return Vec8");
    }

    #[test]
    fn test_rune_drives_symplectic_geometry() {
        let ctx = EvalContext::new();

        // Create a symplectic state (position + momentum)
        let state = Value::Vec16([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // position
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // momentum
        ]);

        // Test Hamiltonian computation
        let h = ctx
            .apply_builtin(RuneBuiltin::SymHamiltonian, &[state.clone()])
            .unwrap();

        if let Value::Scalar(energy) = h {
            assert!(energy >= 0.0, "Hamiltonian should be non-negative");
        } else {
            panic!("Expected Scalar from SymHamiltonian");
        }

        // Test symplectic evolution
        let dt = Value::Scalar(0.1);
        let evolved = ctx
            .apply_builtin(RuneBuiltin::SymEvolveStep, &[state, dt])
            .unwrap();

        assert!(
            matches!(evolved, Value::Vec16(_)),
            "Symplectic evolution should return Vec16"
        );
    }

    #[test]
    fn test_rune_drives_topological_analysis() {
        let ctx = EvalContext::new();

        // Create a point cloud (2 points packed into Vec16)
        let points = Value::Vec16([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // point 1
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // point 2
        ]);

        // Test Betti number computation
        let betti = ctx
            .apply_builtin(RuneBuiltin::TopoBetti, &[points.clone()])
            .unwrap();

        if let Value::Betti([b0, b1, b2]) = betti {
            assert!(b0 > 0, "Should have at least one connected component");
            // b1, b2 depend on point cloud structure
            let _ = (b1, b2);
        } else {
            panic!("Expected Betti from TopoBetti");
        }

        // Test topological signature
        let sig = ctx
            .apply_builtin(RuneBuiltin::TopoSignature, &[points])
            .unwrap();

        assert!(
            matches!(sig, Value::Symbol(_)),
            "Topological signature should return Symbol"
        );
    }

    #[test]
    fn test_from_trait_conversions() {
        // Test automatic Value wrapping
        let v8: Value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].into();
        assert!(matches!(v8, Value::Vec8(_)));

        let v16: Value = [0.0; 16].into();
        assert!(matches!(v16, Value::Vec16(_)));

        let quat: Value = [1.0, 0.0, 0.0, 0.0].into();
        assert!(matches!(quat, Value::Quaternion(_)));

        let betti: Value = [1, 0, 0].into();
        assert!(matches!(betti, Value::Betti(_)));
    }
}
