//! UI components for the TUI.

pub mod editor;
pub mod file_browser;
pub mod help_screen;
pub mod history_panel;
pub mod repl_panel;
pub mod settings_panel;
pub mod stats_bar;
pub mod status_bar;
pub mod diff_viewer;

pub use editor::EditorComponent;
pub use file_browser::FileBrowser;
pub use help_screen::HelpScreen;
pub use history_panel::HistoryPanel;
pub use repl_panel::ReplPanel;
pub use settings_panel::SettingsPanel;
pub use stats_bar::StatsBar;
pub use status_bar::StatusBar;
pub use diff_viewer::DiffViewer;

