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

use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

// Hydron geometry layers
use super::quaternion::QuaternionOps;
use super::spherical::SphericalLayer;
use super::symplectic::SymplecticLayer;
use super::topological::TopologicalLayer;

/// Runtime value types in the E8 ecosystem
#[derive(Debug, Clone, PartialEq)]
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
    Gf8(super::gf8::Gf8),

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

    /// Error value
    Error(String),
}

/// Octonion representation: (scalar, 7 imaginary units)
#[derive(Debug, Clone, Copy, PartialEq)]
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

// Re-export canonical Gf8 from gf8 module
pub use super::gf8::Gf8;

// Re-export SIMD functions when feature is enabled
#[cfg(feature = "simd")]
pub use super::gf8::{
    get_available_f32_256_intrinsics, gf8_add_inplace_slice_simd, gf8_add_simd, gf8_dot_simd,
    gf8_matvec_simd, gf8_norm2_simd, gf8_sub_simd, print_simd_capabilities,
};

impl Value {
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
                #[cfg(feature = "simd")]
                {
                    use super::gf8::gf8_add_inplace_slice_simd;
                    let mut result = *a;
                    gf8_add_inplace_slice_simd(&mut result, b);
                    Ok(Value::Vec8(result))
                }
                #[cfg(not(feature = "simd"))]
                {
                    let mut result = [0.0; 8];
                    for i in 0..8 {
                        result[i] = a[i] + b[i];
                    }
                    Ok(Value::Vec8(result))
                }
            }

            (Value::Gf8(a), Value::Gf8(b)) => {
                #[cfg(feature = "simd")]
                {
                    let result_coords = super::gf8::gf8_add_simd(a.coords(), b.coords());
                    Ok(Value::Gf8(super::gf8::Gf8::new(result_coords)))
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

            (Value::Gf8(a), Value::Gf8(b)) => {
                #[cfg(feature = "simd")]
                {
                    let result_coords = super::gf8::gf8_sub_simd(a.coords(), b.coords());
                    Ok(Value::Gf8(super::gf8::Gf8::new(result_coords)))
                }
                #[cfg(not(feature = "simd"))]
                {
                    Ok(Value::Gf8(*a - *b))
                }
            }

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

            Value::Gf8(g) => Ok(Value::Gf8(-*g)),

            Value::Octonion(o) => Ok(Value::Octonion(Octonion {
                scalar: -o.scalar,
                i: o.i.map(|x| -x),
            })),

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

    /// Compute dot product (SIMD-accelerated when available)
    #[cfg(feature = "simd")]
    pub fn dot_simd(&self, other: &Value) -> Result<f32, EvalError> {
        match (self, other) {
            (Value::Gf8(a), Value::Gf8(b)) => Ok(super::gf8::gf8_dot_simd(a.coords(), b.coords())),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot compute dot product of {:?} and {:?}",
                self, other
            ))),
        }
    }

    /// Compute squared norm (SIMD-accelerated when available)
    #[cfg(feature = "simd")]
    pub fn norm2_simd(&self) -> Result<f32, EvalError> {
        match self {
            Value::Gf8(a) => Ok(super::gf8::gf8_norm2_simd(a.coords())),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot compute norm of {:?}",
                self
            ))),
        }
    }

    /// Apply matrix transformation (SIMD-accelerated when available)
    #[cfg(feature = "simd")]
    pub fn matrix_transform(&self, matrix: &[[f32; 8]; 8]) -> Result<Value, EvalError> {
        match self {
            Value::Gf8(a) => {
                let result_coords = super::gf8::gf8_matvec_simd(matrix, a.coords());
                Ok(Value::Gf8(super::gf8::Gf8::new(result_coords)))
            }
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot apply matrix transform to {:?}",
                self
            ))),
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
            Value::Symbol(s) => write!(f, "{}", s),
            Value::Matrix8x8(_) => write!(f, "Matrix8x8[...]"),
            Value::Betti(b) => write!(f, "Betti[{}, {}, {}]", b[0], b[1], b[2]),
            Value::PointCloud(points) => write!(f, "PointCloud[{} points]", points.len()),
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
    // Spherical (S7) operations
    S7Project,  // [f32;8] → [f32;8]
    S7Distance, // [f32;8], [f32;8] → f32
    S7Slerp,    // [f32;8], [f32;8], f32 → [f32;8]

    // Quaternion operations
    QuatSlerp, // [f32;4], [f32;4], f32 → [f32;4]

    // Symplectic operations
    SymHamiltonian, // [f32;16] → f32
    SymEvolveStep,  // [f32;16], f32 → [f32;16]

    // Topological operations
    TopoBetti,     // [[f32;8]] → [u32;3]
    TopoSignature, // [[f32;8]] → symbol
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
        }
    }
}

// ===================================
// Helper functions for type extraction
// ===================================

fn expect_vec8(val: Option<&Value>) -> Result<[f32; 8], EvalError> {
    match val {
        Some(Value::Vec8(v)) => Ok(*v),
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
        Some(_) => Err(EvalError::TypeMismatch("Expected Scalar".to_string())),
        None => Err(EvalError::InvalidOperation("Missing argument".to_string())),
    }
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

    if points.is_empty() {
        return Err(EvalError::InvalidOperation("Empty point cloud".to_string()));
    }

    Ok(points)
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
        use crate::rune::hydron::gf8::Gf8;

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
