/* src/tui/keybindings.rs */
//!▫~•◦-------------------------------‣
//! # Keyboard shortcuts and action mapping for the TUI.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module defines the `Action` enum, which represents all possible user
//! commands, and the `KeyBindings` struct, which is responsible for mapping raw
//! keyboard events (`KeyEvent`) to these actions.
//!
//! ## Key Capabilities
//! - **Action Enum**: Provides a comprehensive, type-safe list of all application commands.
//! - **Key Mapping**: The `handle` function serves as a central, stateless lookup
//!   table for all global keyboard shortcuts.
//! - **Help Screen Data**: The `shortcuts` method provides a static, pre-formatted
//!   list of keybindings for display on the help screen, optimized for zero-copy rendering.
//!
//! ### Architectural Notes
//! The `handle` function is designed to be a pure, high-performance lookup. It has no
//! side effects and performs no allocations, making it suitable for being called on
//! every key event in the main application loop. The `shortcuts` list uses pre-padded
//! static strings to ensure the help screen can render them without any runtime
//! formatting overhead.
//!
//! #### Example
//! ```rust
//! use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
//! use rune_xero::tui::keybindings::{KeyBindings, Action};
//!
//! let event = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
//! let action = KeyBindings::handle(event);
//! assert_eq!(action, Action::Quit);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// Actions that can be triggered by keyboard shortcuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Quit,
    ToggleMode,
    SwitchPanel,
    OpenFile,
    SaveFile,
    Refresh,
    ToggleSettings,
    ToggleHelp,
    ToggleFileBrowser,
    ToggleHistory,
    ToggleDiff,
    ToggleTheme,
    CopyOutput,
    CopySelection,
    PasteInput,
    ClearInput,
    NewFile,
    RoundTrip,
    OpenRepl,
    None,
}

/// A stateless utility for handling keybindings.
pub struct KeyBindings;

impl KeyBindings {
    /// Maps a crossterm `KeyEvent` to a specific `Action`.
    pub fn handle(key: KeyEvent) -> Action {
        match (key.code, key.modifiers) {
            // Quit
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => Action::Quit,
            (KeyCode::Char('q'), KeyModifiers::CONTROL) => Action::Quit,

            // Mode & Panel
            (KeyCode::Char('e'), KeyModifiers::CONTROL) => Action::ToggleMode,
            (KeyCode::Char('m'), KeyModifiers::CONTROL) => Action::ToggleMode,
            (KeyCode::Tab, KeyModifiers::NONE) => Action::SwitchPanel,

            // File Operations
            (KeyCode::Char('o'), KeyModifiers::CONTROL) => Action::OpenFile,
            (KeyCode::Char('s'), KeyModifiers::CONTROL) => Action::SaveFile,
            (KeyCode::Char('n'), KeyModifiers::CONTROL) => Action::NewFile,
            (KeyCode::Char('l'), KeyModifiers::CONTROL) => Action::ClearInput,

            // Modals & Views
            (KeyCode::Char('p'), KeyModifiers::CONTROL) => Action::ToggleSettings,
            (KeyCode::F(1), KeyModifiers::NONE) => Action::ToggleHelp,
            (KeyCode::Char('?'), KeyModifiers::NONE) => Action::ToggleHelp,
            (KeyCode::Char('f'), KeyModifiers::CONTROL) => Action::ToggleFileBrowser,
            (KeyCode::Char('h'), KeyModifiers::CONTROL) => Action::ToggleHistory,
            (KeyCode::Char('d'), KeyModifiers::CONTROL) => Action::ToggleDiff,
            (KeyCode::Char('r'), KeyModifiers::CONTROL) => Action::OpenRepl,

            // App-level Actions
            (KeyCode::F(5), KeyModifiers::NONE) => Action::Refresh,
            (KeyCode::Char('t'), KeyModifiers::CONTROL) => Action::ToggleTheme,
            (KeyCode::Char('b'), KeyModifiers::CONTROL) => Action::RoundTrip,

            // Clipboard
            (KeyCode::Char('y'), KeyModifiers::CONTROL) => Action::CopyOutput,
            (KeyCode::Char('k'), KeyModifiers::CONTROL) => Action::CopySelection,
            (KeyCode::Char('v'), KeyModifiers::CONTROL) => Action::PasteInput,

            _ => Action::None,
        }
    }

    /// Gets a list of shortcuts for the help screen display.
    /// The keys are pre-padded to a width of 18 characters for zero-copy rendering.
    pub fn shortcuts() -> Vec<(&'static str, &'static str)> {
        vec![
            ("Ctrl+C / Ctrl+Q", "Quit Application"),
            ("Ctrl+E / Ctrl+M", "Toggle Mode (Encode/Decode/Rune)"),
            ("Tab", "Switch Active Panel (Input/Output)"),
            ("Ctrl+R", "Open/Close REPL"),
            ("", ""), // Spacer
            ("F5", "Refresh Conversion"),
            ("Ctrl+B", "Perform Round-Trip Test"),
            ("", ""), // Spacer
            ("Ctrl+O", "Open File..."),
            ("Ctrl+S", "Save Output File"),
            ("Ctrl+N", "New File"),
            ("Ctrl+L", "Clear Input and Output"),
            ("", ""), // Spacer
            ("Ctrl+Y", "Copy All Output to Clipboard"),
            ("Ctrl+K", "Copy Selection to Clipboard"),
            ("Ctrl+V", "Paste from Clipboard into Input"),
            ("", ""), // Spacer
            ("Ctrl+P", "Toggle Settings Panel"),
            ("F1 / ?", "Toggle This Help Screen"),
            ("Ctrl+F", "Toggle File Browser"),
            ("Ctrl+H", "Toggle Conversion History"),
            ("Ctrl+D", "Toggle Side-by-Side Diff View"),
            ("Ctrl+T", "Toggle Color Theme"),
        ]
    }
}