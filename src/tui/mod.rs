//! Terminal User Interface for TOON format conversion.
//!
//! Provides an interactive TUI with real-time conversion, REPL, and settings
//! panels.

pub mod app;
pub mod components;
pub mod events;
pub mod keybindings;
pub mod repl_command;
pub mod state;
pub mod theme;
pub mod ui;

use std::io;

use anyhow::Result;
pub use app::TuiApp;
use crossterm::{
    execute,
    terminal::{
        disable_raw_mode,
        enable_raw_mode,
        EnterAlternateScreen,
        LeaveAlternateScreen,
    },
};
use ratatui::{
    backend::CrosstermBackend,
    Terminal,
};

/// Initialize and run the TUI application.
///
/// Sets up terminal in raw mode, runs the app, then restores terminal state.
pub fn run() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = TuiApp::new();
    let res = app.run(&mut terminal);

    // Always restore terminal, even on error
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    res
}
