//! Generative Archetype Engine for CUDA kernels.
//!
//! This module provides the `ArchetypeEngine` that compiles CUDA kernel templates
//! into PTX modules on demand, with caching for performance.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use cust::module::Module;
use sha2::{Digest, Sha256};

/// Engine for generating and compiling CUDA kernel archetypes from templates.
pub struct ArchetypeEngine {
    cache_dir: PathBuf,
}

impl Default for ArchetypeEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ArchetypeEngine {
    /// Creates a new `ArchetypeEngine` with default cache directory.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache_dir: Path::new("target").join("rune").join("cache"),
        }
    }

    /// Compiles an archetype kernel from the template by replacing parameters and caching the PTX.
    ///
    /// # Arguments
    /// * `func_name` - The function name to use in the kernel
    /// * `d_dim` - The dimension parameter for the kernel
    ///
    /// # Returns
    /// A compiled CUDA module ready for kernel execution
    ///
    /// # Errors
    /// Returns an error if template loading, code generation, file I/O, or CUDA compilation fails.
    pub fn compile_archetype(
        &self,
        func_name: &str,
        d_dim: usize,
    ) -> Result<Module, Box<dyn std::error::Error>> {
        // Load the template
        let template = include_str!("archetypes/row_dot.cu.template");

        // Instantiate the template by replacing placeholders
        let code = template
            .replace("$FUNC_NAME", func_name)
            .replace("$D_DIM", &d_dim.to_string());

        // Compute SHA-256 hash of the instantiated code for caching
        let mut hasher = Sha256::new();
        hasher.update(&code);
        let hash = format!("{:x}", hasher.finalize());

        // Define cache paths
        let ptx_path = self.cache_dir.join(format!("{hash}.ptx"));
        let cu_path = self.cache_dir.join(format!("{hash}.cu"));

        // Compile if PTX doesn't exist
        if !ptx_path.exists() {
            // Ensure cache directory exists
            fs::create_dir_all(&self.cache_dir)?;

            // Write the instantiated CUDA code to a temporary file
            fs::write(&cu_path, &code)?;

            // Compile using nvcc
            let output = Command::new("nvcc")
                .args([
                    "-ptx",
                    "-o",
                    &ptx_path.to_string_lossy(),
                    &cu_path.to_string_lossy(),
                ])
                .output()?;

            if !output.status.success() {
                return Err(format!(
                    "nvcc compilation failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                )
                .into());
            }
        }

        // Load the PTX module
        let module = Module::from_file(&ptx_path)?;

        Ok(module)
    }
}
