/* src/lib.rs */
//!▫~•◦-------------------------------‣
//! # Rune-Gate: Visual & Runtime Harness
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Entry point for the Bevy-based visualization and interaction layer.

#[cfg(feature = "viewer")]
pub mod ui;
pub mod app;
pub mod bridge;
pub mod scene;

#[cfg(feature = "viewer")]
use bevy::prelude::*;

pub use bridge::HydronBackend;

#[cfg(feature = "viewer")]
pub struct RuneGatePlugin;

#[cfg(feature = "viewer")]
impl Plugin for RuneGatePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ui::ChatState::default())
           .add_systems(Startup, ui::setup_chat_ui)
           .add_systems(Update, ui::chat_input_system);
    }
}
