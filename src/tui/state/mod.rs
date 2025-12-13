/* src/tui/state/mod.rs */
//!▫~•◦-------------------------------‣
//! # Centralized state management for the TUI.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module aggregates and exports all state-related structs for the entire
//! terminal user interface. It acts as the public API for the `state` module,
//! providing a single point of import for parent modules like `app` and `ui`.
//!
//! ## Key Capabilities
//! - **Module Aggregation**: Declares the `app_state`, `editor_state`, `file_state`,
//!   and `repl_state` sub-modules.
//! - **Type Re-exporting**: Publicly exports the primary state structs and enums from
//!   its children for convenient access, such as `AppState`, `EditorState`, etc.
//!
//! ### Architectural Notes
//! This `mod.rs` file follows the standard Rust pattern for organizing a complex
//! module. By re-exporting key types, it creates a clean and ergonomic interface
//! for other parts of the application to use, abstracting away the internal file
//! structure of the `state` module.
//!
//! #### Example
//! ```rust
//! // Instead of:
//! // use crate::tui::state::app_state::AppState;
//! // use crate::tui::state::editor_state::EditorState;
//!
//! // Other modules can simply use:
//! use crate::tui::state::{AppState, EditorState};
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod app_state;
pub mod editor_state;
pub mod file_state;
pub mod repl_state;

pub use app_state::{AppState, ConfirmationAction, ConversionStats, Mode};
pub use editor_state::{EditorMode, EditorState};
pub use file_state::{ConversionHistory, FileEntry, FileState};
pub use repl_state::{ReplLine, ReplLineKind, ReplState};