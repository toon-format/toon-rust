//! Keyboard shortcuts and action mapping.

use crossterm::event::{
    KeyCode,
    KeyEvent,
    KeyModifiers,
};

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

pub struct KeyBindings;

impl KeyBindings {
    /// Map key event to action.
    pub fn handle(key: KeyEvent) -> Action {
        match (key.code, key.modifiers) {
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => Action::Quit,
            (KeyCode::Char('q'), KeyModifiers::CONTROL) => Action::Quit,
            (KeyCode::Char('e'), KeyModifiers::CONTROL) => Action::ToggleMode,
            (KeyCode::Char('m'), KeyModifiers::CONTROL) => Action::ToggleMode,
            (KeyCode::Tab, KeyModifiers::NONE) => Action::SwitchPanel,
            (KeyCode::Char('o'), KeyModifiers::CONTROL) => Action::OpenFile,
            (KeyCode::Char('s'), KeyModifiers::CONTROL) => Action::SaveFile,
            (KeyCode::Char('n'), KeyModifiers::CONTROL) => Action::NewFile,
            (KeyCode::Char('p'), KeyModifiers::CONTROL) => Action::ToggleSettings,
            (KeyCode::F(1), KeyModifiers::NONE) => Action::ToggleHelp,
            (KeyCode::Char('f'), KeyModifiers::CONTROL) => Action::ToggleFileBrowser,
            (KeyCode::Char('h'), KeyModifiers::CONTROL) => Action::ToggleHistory,
            (KeyCode::Char('d'), KeyModifiers::CONTROL) => Action::ToggleDiff,
            (KeyCode::Char('t'), KeyModifiers::CONTROL) => Action::ToggleTheme,
            (KeyCode::Char('y'), KeyModifiers::CONTROL) => Action::CopyOutput,
            (KeyCode::Char('k'), KeyModifiers::CONTROL) => Action::CopySelection,
            (KeyCode::Char('v'), KeyModifiers::CONTROL) => Action::PasteInput,
            (KeyCode::Char('b'), KeyModifiers::CONTROL) => Action::RoundTrip,
            (KeyCode::Char('r'), KeyModifiers::CONTROL) => Action::OpenRepl,
            (KeyCode::Char('l'), KeyModifiers::CONTROL) => Action::ClearInput,

            _ => Action::None,
        }
    }

    /// Get list of shortcuts for help display.
    pub fn shortcuts() -> Vec<(&'static str, &'static str)> {
        vec![
            ("Ctrl+C/Q", "Quit"),
            ("Ctrl+E/M", "Toggle Mode"),
            ("Tab", "Switch Panel"),
            ("Ctrl+R", "Open REPL"),
            ("Ctrl+O", "Open File"),
            ("Ctrl+S", "Save File"),
            ("Ctrl+N", "New File"),
            ("Ctrl+P", "Settings"),
            ("F1", "Help"),
            ("Ctrl+F", "File Browser"),
            ("Ctrl+H", "History"),
            ("Ctrl+D", "Diff View"),
            ("Ctrl+T", "Toggle Theme"),
            ("Ctrl+Y", "Copy All Output"),
            ("Ctrl+K", "Copy Selection"),
            ("Ctrl+V", "Paste Input"),
            ("Ctrl+B", "Round Trip Test"),
            ("Ctrl+L", "Clear Input"),
        ]
    }
}
