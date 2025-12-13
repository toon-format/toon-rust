/* src/tui/state/file_state.rs */
//!▫~•◦-------------------------------‣
//! # State management for files, directories, and conversion history.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module defines the `FileState` struct and related data structures, which
//! are responsible for tracking the current file, directory, user selections, and
//! a log of past conversion operations.
//!
//! ## Key Capabilities
//! - **File and Directory Tracking**: Manages the current working directory and open file.
//! - **Conversion History**: Maintains a capped-size log of `ConversionHistory` entries.
//! - **Selection Management**: Tracks a list of user-selected files.
//! - **Performance-Optimized**: Data structures and methods are designed to minimize
//!   or eliminate unnecessary heap allocations, especially for frequently-accessed
//!   data like file names.
//!
//! ### Architectural Notes
//! `FileEntry::name()` returns a `Cow<str>` by using `to_string_lossy`, which provides
//! a zero-allocation path for valid UTF-8 file names. `ConversionHistory` uses the
//! `Mode` enum instead of a `String` to represent the conversion type, making it
//! more type-safe and efficient.
//!
//! #### Example
//! ```rust
//! use std::path::PathBuf;
//! use rune_xero::tui::state::file_state::{FileState, FileEntry};
//!
//! let mut file_state = FileState::new();
//! file_state.set_current_file(PathBuf::from("/path/to/file.json"));
//! assert!(file_state.current_file.is_some());
//!
//! let entry = FileEntry { path: PathBuf::from("file.json"), is_dir: false, size: 0, modified: None };
//! // name() is an efficient, often non-allocating, operation.
//! let file_name = entry.name();
//! assert_eq!(file_name, "file.json");
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Local};

use super::app_state::Mode;

/// A file or directory entry.
#[derive(Debug, Clone)]
pub struct FileEntry {
    pub path: PathBuf,
    pub is_dir: bool,
    pub size: u64,
    pub modified: Option<DateTime<Local>>,
}

impl FileEntry {
    /// Returns the file name for display, avoiding allocation for valid UTF-8.
    ///
    /// Uses `to_string_lossy` which returns a `Cow<str>`. This is a `&str` slice
    /// if the name is valid UTF-8, and an owned `String` only if replacement
    /// characters were needed.
    pub fn name(&self) -> Cow<str> {
        self.path
            .file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or(Cow::Borrowed(""))
    }

    /// Checks if the file has a `.json` extension. Zero-copy.
    pub fn is_json(&self) -> bool {
        !self.is_dir && self.path.extension().and_then(|e| e.to_str()) == Some("json")
    }

    /// Checks if the file has a `.rune` extension. Zero-copy.
    pub fn is_rune(&self) -> bool {
        !self.is_dir && self.path.extension().and_then(|e| e.to_str()) == Some("rune")
    }
}

/// Record of a conversion operation.
#[derive(Debug, Clone)]
pub struct ConversionHistory {
    pub timestamp: DateTime<Local>,
    pub mode: Mode, // Use the efficient Mode enum instead of String
    pub input_file: Option<PathBuf>,
    pub output_file: Option<PathBuf>,
    pub token_savings: f64,
    pub byte_savings: f64,
}

/// File browser and conversion history state.
pub struct FileState {
    pub current_file: Option<PathBuf>,
    pub current_dir: PathBuf,
    pub selected_files: Vec<PathBuf>,
    pub history: Vec<ConversionHistory>,
    pub is_modified: bool,
}

impl FileState {
    pub fn new() -> Self {
        Self {
            current_file: None,
            current_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            selected_files: Vec::new(),
            history: Vec::new(),
            is_modified: false,
        }
    }

    /// Sets the current file and updates the current directory.
    /// This takes ownership of the PathBuf to avoid an internal clone.
    pub fn set_current_file(&mut self, path: PathBuf) {
        if let Some(parent) = path.parent() {
            self.current_dir = parent.to_path_buf();
        }
        self.current_file = Some(path);
        self.is_modified = false;
    }

    pub fn clear_current_file(&mut self) {
        self.current_file = None;
        self.is_modified = false;
    }

    pub fn mark_modified(&mut self) {
        self.is_modified = true;
    }

    pub fn add_to_history(&mut self, entry: ConversionHistory) {
        self.history.push(entry);
        // Capped at 50 entries for performance.
        if self.history.len() > 50 {
            self.history.remove(0);
        }
    }

    pub fn toggle_file_selection(&mut self, path: &Path) {
        if let Some(pos) = self.selected_files.iter().position(|p| p == path) {
            self.selected_files.remove(pos);
        } else {
            self.selected_files.push(path.to_path_buf());
        }
    }

    pub fn clear_selection(&mut self) {
        self.selected_files.clear();
    }

    pub fn is_selected(&self, path: &Path) -> bool {
        self.selected_files.iter().any(|p| p == path)
    }
}

impl Default for FileState {
    fn default() -> Self {
        Self::new()
    }
}