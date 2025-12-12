/* src/app.rs */
//!▫~•◦-------------------------------‣
//! # Application entry point and plugin assembly for the rune-gate 3D viewer.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-gate to provide the main application
//! structure and plugin coordination for E8-Life visualization.
//!
//! ### Key Capabilities
//! - **Bevy Application Setup:** Configures the main Bevy app with default plugins and window settings.
//! - **Plugin Assembly:** Coordinates the integration of scene, UI, and backend plugins.
//! - **Viewer Launching:** Provides convenience functions for launching the viewer with different backends.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `scene`, `ui`, and `bridge`.
//! It serves as the main entry point that assembles all the components into a cohesive application.
//!
//! ### Example
//! ```rust
//! use crate::rune_gate::{run_viewer, RuneGate};
//!
//! // Launch the viewer with default settings
//! run_viewer();
//!
//! // Or use the plugin in a custom Bevy app
//! app.add_plugins(RuneGate);
//! ```

use bevy::prelude::*;
use bevy::window::WindowResolution;

use crate::bridge::{BackendHandle, E8Backend, HydronBackend, SelectedVertex};
use crate::scene::{E8Projection, HoveredVertex, ScenePlugin};
use crate::ui::UiPlugin;

/// Marker plugin to assemble the viewer.
pub struct RuneGate;

impl Plugin for RuneGate {
    fn build(&self, app: &mut App) {
        if app.world_mut().get_resource::<BackendHandle>().is_none() {
            app.insert_resource(BackendHandle::new(HydronBackend::new()));
        }
        app.init_resource::<E8Projection>()
            .init_resource::<SelectedVertex>()
            .init_resource::<HoveredVertex>()
            .add_plugins(ScenePlugin)
            .add_plugins(UiPlugin);
    }
}

/// Launch the Bevy app with the default viewer stack.
pub fn run_viewer_with_backend<B: E8Backend>(backend: B) {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "rune-gate • E8-Life Viewer".into(),
                resolution: WindowResolution::new(1400, 900),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(BackendHandle::new(backend))
        .add_plugins(RuneGate)
        .run();
}

/// Convenience launcher using the Hydron-backed implementation.
pub fn run_viewer() {
    run_viewer_with_backend(HydronBackend::new());
}
