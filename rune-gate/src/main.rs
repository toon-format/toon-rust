/* src/main.rs */
//!▫~•◦-------------------------------‣
//! # Main entry point for the rune-gate application with feature-based configuration.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-gate to provide the executable
//! entry point with conditional compilation based on feature flags.
//!
//! ### Key Capabilities
//! - **Feature Detection:** Conditionally compiles based on the `viewer` feature flag.
//! - **Backend Integration:** Launches the viewer with the Hydron backend when enabled.
//! - **Graceful Degradation:** Provides informative error messages when features are missing.
//!
//! ### Architectural Notes
//! This module serves as the main entry point that delegates to the app module.
//! It works closely with the `app` module to provide the complete application experience.
//!
//! ### Example
//! ```rust
//! // When built with --features viewer, launches the full 3D viewer
//! cargo run --features viewer
//!
//! // Without the viewer feature, shows an informative message
//! cargo run
//! ```

#[cfg(feature = "viewer")]
fn main() {
    rune_gate::app::run_viewer_with_backend(rune_gate::HydronBackend::new());
}

#[cfg(not(feature = "viewer"))]
fn main() {
    eprintln!("rune-gate built without `viewer` feature; enable with --features viewer");
}
