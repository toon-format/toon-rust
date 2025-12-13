use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only attempt to compile CUDA kernels if the `cuda` feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:warning=Building CUDA kernels (nvcc required)");

        // Compile CUDA kernel bundle (mgsk = Geoshi ops)
        let kernels = [("kernels/mgsk.cu", "mgsk.ptx")];

        let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");

        // Try to find nvcc in PATH
        if which::which("nvcc").is_ok() {
            for (src_file, dest_name) in &kernels {
                let src = PathBuf::from(src_file);
                let dest = PathBuf::from(&out_dir).join(dest_name);

                let status = Command::new("nvcc")
                    .args([
                        "-ptx",
                        "-arch", "sm_89",
                        "-O3",
                        "-use_fast_math",
                        "--generate-line-info",
                        "-Xptxas", "-v",
                        src.to_str().unwrap(),
                        "-o",
                        dest.to_str().unwrap(),
                    ])
                    .status();

                match status {
                    Ok(status) if status.success() => {
                        println!("cargo:rustc-env=MGSK_PTX={}", dest.to_str().unwrap());
                        println!("✅ Successfully compiled {}", src_file);
                    }
                    Ok(status) => {
                        println!("cargo:warning=nvcc returned {} while building {}", status, src_file);
                    }
                    Err(e) => {
                        println!("cargo:warning=Failed to invoke nvcc for {}: {}", src_file, e);
                    }
                }
            }
        } else {
            println!("cargo:warning=nvcc not found in PATH; skipping CUDA PTX build");
        }
    }
}
