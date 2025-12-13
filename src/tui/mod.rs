/* src/tui/mod.rs */
//!▫~•◦-------------------------------‣
//! # Terminal User Interface for RUNE format conversion.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is the root of the entire TUI application. It declares all
//! sub-modules (components, state, events, etc.) and provides the main entry
//! point, `run()`, to launch the interactive interface.
//!
//! ## Key Capabilities
//! - **Application Bootstrap**: The `run()` function handles all terminal setup
//!   (entering raw mode, alternate screen) and teardown, ensuring the user's
//!   terminal is restored to its original state on exit.
//! - **Module Organization**: Declares the hierarchy for all TUI-related code,
//!   including UI components, state management, event handling, and theming.
//!
//! ### Architectural Notes
//! The `run()` function encapsulates the entire lifecycle of the TUI application.
//! It creates the `TuiApp` instance and starts its main event loop. The use of `?`
//! and the final `res` return ensures that terminal restoration code is always
//! executed, even if the application exits with an error. This is a critical
//! pattern for robust TUI applications.
//!
//! #### Example
//! ```rust,no_run
//! // In main.rs
//! fn main() -> yoshi::error::Result<()> {
//!     rune_xero::tui::run()?;
//!     Ok(())
//! }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::io;

use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use yoshi::error::Result;

pub use self::app::TuiApp;

pub mod app;
pub mod components;
pub mod events;
pub mod keybindings;
pub mod message;
pub mod repl_command;
pub mod state;
pub mod theme;
pub mod ui;

/// Initializes the terminal, runs the TUI application, and restores the terminal on exit.
///
/// This function is the main entry point for the entire interactive TUI.
pub fn run() -> Result<()> {
    // --- Terminal Setup ---
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // --- Application Lifecycle ---
    let mut app = TuiApp::new();
    let res = app.run(&mut terminal); // This runs the main event loop

    // --- Terminal Teardown ---
    // This block is crucial: it ensures the terminal is restored to a usable
    // state even if the application encounters an error during its run.
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    // Return the result from the application's run loop.
    res
}