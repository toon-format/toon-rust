/* src/lib.rs */
//!▫~•◦-------------------------------‣
//! # Core library module for rune-gate providing Bevy-based 3D visualization of E8-Life geometry.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-gate to achieve comprehensive 3D visualization
//! and interaction with E8-Life geometric structures.
//!
//! ### Key Capabilities
//! - **Bevy ECS Integration:** Provides a complete Bevy plugin system for 3D rendering and UI.
//! - **E8 Geometry Visualization:** Renders complex 8-dimensional E8 root systems in 3D space.
//! - **Modular Architecture:** Organized into app, bridge, scene, and UI components for clean separation.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `app`, `bridge`, `scene`, and `ui`.
//! Result structures adhere to the Bevy ECS patterns and are compatible
//! with the system's rendering pipeline.
//!
//! ### Example
//! ```rust
//! use crate::rune_gate::{RuneGate, run_viewer_with_backend};
//!
//! // Launch the viewer with default backend
//! run_viewer_with_backend(HydronBackend::new());
//!
//! // Or use the plugin directly in your Bevy app
//! app.add_plugins(RuneGate);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#[cfg(feature = "viewer")]
pub mod app;
pub mod bridge;
#[cfg(feature = "viewer")]
pub mod scene;
#[cfg(feature = "viewer")]
pub mod ui;

#[cfg(feature = "viewer")]
pub use app::{RuneGate, run_viewer_with_backend};
pub use bridge::{
    BackendHandle, DomainSummary, E8Backend, E8Query, HydronBackend, PathResult, QueryResult,
    SelectedVertex, VertexDetail,
};
#[cfg(feature = "viewer")]
pub use ui::{PanelSide, ViewerLayout};
