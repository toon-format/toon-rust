//! Application messages (Elm-style) for TUI
//!
// This module defines the central `Msg` enum which represents all
// user actions and events that can modify application state.

use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub enum Msg {
    Quit,
    ToggleMode,
    ToggleSettings,
    ToggleHelp,
    ToggleFileBrowser,
    ToggleHistory,
    ToggleDiff,
    ToggleTheme,
    OpenFile(PathBuf),
    SaveFile,
    NewFile,
    Refresh,
    ExecuteRepl(String),
    CopyOutput,
    CopySelection,
    PasteInput,
    RoundTrip,
    ClearInput,
    SetError(String),
    SetStatus(String),
    ClearError,
    ClearStatus,
    // Add more messages as needed for components
}
