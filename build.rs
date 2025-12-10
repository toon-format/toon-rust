/* build.rs */
//! Build script to compile CUDA kernels (feature-gated).
//! When the `cuda` feature is enabled, this script is intended to invoke
//! `ptx-builder` to compile the kernel sources into PTX for runtime loading.

#[cfg(feature = "cuda")]
fn main() {
    // TODO: wire up ptx-builder once CUDA kernels are implemented.
    // Left as a no-op placeholder so default builds are unaffected.
    // Avoid emitting cargo::warning to keep the build free of warnings.
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    // No-op when CUDA is not enabled.
}
