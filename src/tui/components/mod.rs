/* src/tui/components/mod.rs */
//!▫~•◦-------------------------‣
//! # UI components for the RUNE TUI.
//!▫~•◦------------------------------‣
//!
//! This module aggregates and exports all stateless UI component structs for the
//! terminal user interface. It serves as the public API for the `components`
//! module, providing a clean, single point of import for the main `ui` rendering module.
//!
//! ## Key Capabilities
//! - **Module Aggregation**: Declares all component sub-modules, such as `editor`,
//!   `file_browser`, `status_bar`, etc.
//! - **Type Re-exporting**: Publicly exports the primary component structs (e.g.,
//!   `EditorComponent`, `FileBrowser`, `StatusBar`) for convenient access.
//!
//! ### Architectural Notes
//! Following the standard Rust pattern for module organization, this `mod.rs` file
//! creates an ergonomic interface for the rest of the application. The main UI
//! renderer can import all necessary components with a single `use` statement.
//!
//! #### Example
//! ```rust
//! // Instead of multiple imports:
//! // use crate::tui::components::editor::EditorComponent;
//! // use crate::tui::components::status_bar::StatusBar;
//!
//! // The main ui.rs file can simply use:
//! use crate::tui::components::{EditorComponent, StatusBar};
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod confirmation_dialog;
pub mod diff_viewer;
pub mod editor;
pub mod file_browser;
pub mod help_screen;
pub mod history_panel;
pub mod repl_panel;
pub mod settings_panel;
pub mod stats_bar;
pub mod status_bar;

pub use confirmation_dialog::ConfirmationDialog;
pub use diff_viewer::DiffViewer;
pub use editor::EditorComponent;
pub use file_browser::FileBrowser;
pub use help_screen::HelpScreen;
pub use history_panel::HistoryPanel;
pub use repl_panel::ReplPanel;
pub use settings_panel::SettingsPanel;
pub use stats_bar::StatsBar;
pub use status_bar::StatusBar;