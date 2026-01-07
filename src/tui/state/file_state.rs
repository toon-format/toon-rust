//! File management and conversion history.

use std::path::PathBuf;

#[cfg(feature = "tui-time")]
use chrono::{DateTime, Local};

/// Timestamp type for UI history entries.
///
/// # Examples
/// ```
/// use toon_format::tui::state::now_timestamp;
///
/// let _ = now_timestamp();
/// ```
#[cfg(feature = "tui-time")]
pub type Timestamp = DateTime<Local>;

/// Timestamp type for UI history entries.
///
/// # Examples
/// ```
/// use toon_format::tui::state::now_timestamp;
///
/// let _ = now_timestamp();
/// ```
#[cfg(not(feature = "tui-time"))]
pub type Timestamp = ();

/// Return the current timestamp when supported.
///
/// # Examples
/// ```
/// use toon_format::tui::state::now_timestamp;
///
/// let _ = now_timestamp();
/// ```
pub fn now_timestamp() -> Option<Timestamp> {
    #[cfg(feature = "tui-time")]
    {
        Some(Local::now())
    }

    #[cfg(not(feature = "tui-time"))]
    {
        None
    }
}

/// Format a timestamp for display.
///
/// # Examples
/// ```
/// use toon_format::tui::state::format_timestamp;
///
/// let out = format_timestamp(&None);
/// assert!(!out.is_empty());
/// ```
pub fn format_timestamp(timestamp: &Option<Timestamp>) -> String {
    #[cfg(feature = "tui-time")]
    {
        timestamp
            .as_ref()
            .map(|ts| ts.format("%H:%M:%S").to_string())
            .unwrap_or_else(|| "--:--:--".to_string())
    }

    #[cfg(not(feature = "tui-time"))]
    {
        let _ = timestamp;
        "--:--:--".to_string()
    }
}

/// A file or directory entry.
///
/// # Examples
/// ```
/// use std::path::PathBuf;
/// use toon_format::tui::state::file_state::FileEntry;
///
/// let entry = FileEntry {
///     path: PathBuf::from("data.json"),
///     is_dir: false,
///     size: 0,
///     modified: None,
/// };
/// let _ = entry;
/// ```
#[derive(Debug, Clone)]
pub struct FileEntry {
    pub path: PathBuf,
    pub is_dir: bool,
    pub size: u64,
    pub modified: Option<Timestamp>,
}

impl FileEntry {
    /// Return the file name for display.
    ///
    /// # Examples
    /// ```
    /// use std::path::PathBuf;
    /// use toon_format::tui::state::file_state::FileEntry;
    ///
    /// let entry = FileEntry {
    ///     path: PathBuf::from("data.json"),
    ///     is_dir: false,
    ///     size: 0,
    ///     modified: None,
    /// };
    /// assert_eq!(entry.name(), "data.json");
    /// ```
    pub fn name(&self) -> String {
        self.path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string()
    }

    /// Return true if the entry is a JSON file.
    ///
    /// # Examples
    /// ```
    /// use std::path::PathBuf;
    /// use toon_format::tui::state::file_state::FileEntry;
    ///
    /// let entry = FileEntry {
    ///     path: PathBuf::from("data.json"),
    ///     is_dir: false,
    ///     size: 0,
    ///     modified: None,
    /// };
    /// assert!(entry.is_json());
    /// ```
    pub fn is_json(&self) -> bool {
        !self.is_dir && self.path.extension().and_then(|e| e.to_str()) == Some("json")
    }

    /// Return true if the entry is a TOON file.
    ///
    /// # Examples
    /// ```
    /// use std::path::PathBuf;
    /// use toon_format::tui::state::file_state::FileEntry;
    ///
    /// let entry = FileEntry {
    ///     path: PathBuf::from("data.toon"),
    ///     is_dir: false,
    ///     size: 0,
    ///     modified: None,
    /// };
    /// assert!(entry.is_toon());
    /// ```
    pub fn is_toon(&self) -> bool {
        !self.is_dir && self.path.extension().and_then(|e| e.to_str()) == Some("toon")
    }
}

/// Record of a conversion operation.
///
/// # Examples
/// ```
/// use toon_format::tui::state::ConversionHistory;
///
/// let history = ConversionHistory {
///     timestamp: None,
///     mode: "Encode".to_string(),
///     input_file: None,
///     output_file: None,
///     token_savings: None,
///     byte_savings: None,
/// };
/// let _ = history;
/// ```
#[derive(Debug, Clone)]
pub struct ConversionHistory {
    pub timestamp: Option<Timestamp>,
    pub mode: String,
    pub input_file: Option<PathBuf>,
    pub output_file: Option<PathBuf>,
    pub token_savings: Option<f64>,
    pub byte_savings: Option<f64>,
}

/// File browser and conversion history state.
///
/// # Examples
/// ```
/// use toon_format::tui::state::FileState;
///
/// let state = FileState::new();
/// let _ = state;
/// ```
pub struct FileState {
    pub current_file: Option<PathBuf>,
    pub current_dir: PathBuf,
    pub selected_files: Vec<PathBuf>,
    pub history: Vec<ConversionHistory>,
    pub is_modified: bool,
}

impl FileState {
    /// Create a new file state.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::FileState;
    ///
    /// let state = FileState::new();
    /// let _ = state;
    /// ```
    pub fn new() -> Self {
        Self {
            current_file: None,
            current_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            selected_files: Vec::new(),
            history: Vec::new(),
            is_modified: false,
        }
    }

    /// Set the current file and update the working directory.
    ///
    /// # Examples
    /// ```
    /// use std::path::PathBuf;
    /// use toon_format::tui::state::FileState;
    ///
    /// let mut state = FileState::new();
    /// state.set_current_file(PathBuf::from("data.json"));
    /// ```
    pub fn set_current_file(&mut self, path: PathBuf) {
        self.current_file = Some(path.clone());
        self.current_dir = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        self.is_modified = false;
    }

    /// Clear the current file selection.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::FileState;
    ///
    /// let mut state = FileState::new();
    /// state.clear_current_file();
    /// ```
    pub fn clear_current_file(&mut self) {
        self.current_file = None;
        self.is_modified = false;
    }

    /// Mark the current file as modified.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::FileState;
    ///
    /// let mut state = FileState::new();
    /// state.mark_modified();
    /// ```
    pub fn mark_modified(&mut self) {
        self.is_modified = true;
    }

    /// Add a conversion entry to history.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::{ConversionHistory, FileState};
    ///
    /// let mut state = FileState::new();
    /// state.add_to_history(ConversionHistory {
    ///     timestamp: None,
    ///     mode: "Encode".to_string(),
    ///     input_file: None,
    ///     output_file: None,
    ///     token_savings: None,
    ///     byte_savings: None,
    /// });
    /// ```
    pub fn add_to_history(&mut self, entry: ConversionHistory) {
        self.history.push(entry);
        if self.history.len() > 50 {
            self.history.remove(0);
        }
    }

    /// Toggle a file's selection in the browser.
    ///
    /// # Examples
    /// ```
    /// use std::path::PathBuf;
    /// use toon_format::tui::state::FileState;
    ///
    /// let mut state = FileState::new();
    /// state.toggle_file_selection(PathBuf::from("data.json"));
    /// ```
    pub fn toggle_file_selection(&mut self, path: PathBuf) {
        if let Some(pos) = self.selected_files.iter().position(|p| p == &path) {
            self.selected_files.remove(pos);
        } else {
            self.selected_files.push(path);
        }
    }

    /// Clear all selected files.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::FileState;
    ///
    /// let mut state = FileState::new();
    /// state.clear_selection();
    /// ```
    pub fn clear_selection(&mut self) {
        self.selected_files.clear();
    }

    /// Return true if the path is selected.
    ///
    /// # Examples
    /// ```
    /// use std::path::PathBuf;
    /// use toon_format::tui::state::FileState;
    ///
    /// let state = FileState::new();
    /// let _ = state.is_selected(&PathBuf::from("data.json"));
    /// ```
    pub fn is_selected(&self, path: &PathBuf) -> bool {
        self.selected_files.contains(path)
    }
}

impl Default for FileState {
    fn default() -> Self {
        Self::new()
    }
}
