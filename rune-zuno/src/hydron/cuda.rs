//! CUDA-specific evaluation bridge for Hydron operations.
//!
//! Provides GPU-accelerated implementations for DomR computations using the rune-curs
//! CUDA acceleration layer.

#[cfg(feature = "cuda")]
use rune_curs as curs;
use rune_hex::hex;

/// CUDA accelerator for Hydron operations
#[derive(Debug, Clone)]
pub struct CudaAccelerator {
    /// Whether CUDA is available on this system
    cuda_available: bool,
}

impl CudaAccelerator {
    /// Create a new CUDA accelerator and check availability
    pub fn new() -> Self {
        #[cfg(feature = "cuda")]
        let cuda_available = true; // Assume available if compiled with feature

        #[cfg(not(feature = "cuda"))]
        let cuda_available = false;

        Self { cuda_available }
    }

    /// Check if CUDA acceleration is available
    pub fn is_available(&self) -> bool {
        self.cuda_available
    }

    /// Execute CUDA-accelerated DomR computation if available
    pub fn execute_domr(
        &self,
        operation: &str,
        energy_args: &[crate::rune::hydron::values::Value],
    ) -> Result<crate::rune::hydron::values::Value, crate::rune::hydron::values::EvalError> {
        use crate::rune::hydron::values::{EvalError, Value};

        if !self.cuda_available {
            return Err(EvalError::UnsupportedOperation(
                "CUDA not available on this system".to_string(),
            ));
        }

        match operation {
            "CudaDomR" | "CudaArchetypeDomR" => {
                // Extract energy array from first argument
                let energy_vec = extract_energy_array(&energy_args[0])?;
                let n_dr = energy_args
                    .get(1)
                    .map(|v| extract_usize(v))
                    .transpose()?
                    .unwrap_or(8);

                #[cfg(feature = "cuda")]
                {
                    // Get the default hex graph
                    let graph = hex::default_graph();

                    // Compute scores using CUDA
                    let scores =
                        curs::domr_scores_gpu(&energy_vec, graph.coords()).map_err(|e| {
                            EvalError::InvalidOperation(format!("CUDA DomR failed: {}", e))
                        })?;

                    if scores.len() != graph.coords().len() {
                        return Err(EvalError::InvalidOperation(
                            "CUDA returned mismatched score length".to_string(),
                        ));
                    }

                    // Sort by score descending and take top n_dr
                    let mut pairs: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
                    pairs
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    let take = n_dr.min(pairs.len());
                    let mut roots = Vec::with_capacity(take);
                    let mut out_scores = Vec::with_capacity(take);

                    for (idx, score) in pairs.into_iter().take(take) {
                        roots.push(idx as u8);
                        out_scores.push(score);
                    }

                    Ok(Value::DomR(hex::DomR {
                        roots,
                        scores: out_scores,
                    }))
                }

                #[cfg(not(feature = "cuda"))]
                {
                    Err(EvalError::UnsupportedOperation(
                        "CUDA feature not enabled".to_string(),
                    ))
                }
            }

            _ => Err(EvalError::UnsupportedOperation(format!(
                "Unknown CUDA operation: {} (only DomR operations supported)",
                operation
            ))),
        }
    }
}

impl Default for CudaAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Global CUDA accelerator instance
static CUDA_ACCELERATOR: std::sync::OnceLock<CudaAccelerator> = std::sync::OnceLock::new();

/// Get or initialize the global CUDA accelerator
pub fn get_cuda_accelerator() -> &'static CudaAccelerator {
    CUDA_ACCELERATOR.get_or_init(|| CudaAccelerator::new())
}

// Helper functions for extracting values from Value enum

fn extract_energy_array(
    val: &crate::rune::hydron::values::Value,
) -> Result<Vec<f32>, crate::rune::hydron::values::EvalError> {
    use crate::rune::hydron::values::{EvalError, Value};

    match val {
        Value::Array(arr) => {
            if arr.len() != 240 {
                return Err(EvalError::TypeMismatch(format!(
                    "Energy array must have length 240, got {}",
                    arr.len()
                )));
            }
            arr.iter().map(|v| extract_f32(v)).collect()
        }
        _ => Err(EvalError::TypeMismatch(
            "Energy argument must be an Array of 240 floats".to_string(),
        )),
    }
}

fn extract_f32(
    val: &crate::rune::hydron::values::Value,
) -> Result<f32, crate::rune::hydron::values::EvalError> {
    use crate::rune::hydron::values::{EvalError, Value};

    match val {
        Value::Scalar(f) => Ok(*f),
        Value::Float(f) => Ok(*f as f32),
        Value::Integer(i) => Ok(*i as f32),
        _ => Err(EvalError::TypeMismatch(
            "Value must be numeric (Scalar, Float, or Integer)".to_string(),
        )),
    }
}

fn extract_usize(
    val: &crate::rune::hydron::values::Value,
) -> Result<usize, crate::rune::hydron::values::EvalError> {
    use crate::rune::hydron::values::{EvalError, Value};

    match val {
        Value::Scalar(f) => Ok(*f as usize),
        Value::Float(f) => Ok(*f as usize),
        Value::Integer(i) => Ok(*i as usize),
        _ => Err(EvalError::TypeMismatch("Index must be numeric".to_string())),
    }
}
