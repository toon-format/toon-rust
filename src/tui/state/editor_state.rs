/* src/tui/state/editor_state.rs */
//!▫~•◦-------------------------------‣
//! # Editor state for the TUI's input and output text areas.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module defines `EditorState`, which manages the state for the two main
//! text panels in the application, powered by the `tui-textarea` crate.
//!
//! ## Key Capabilities
//! - **Text Management**: Holds the content for both the input and output editors.
//! - **Focus Handling**: Tracks which editor panel (`Input` or `Output`) is currently active.
//! - **Performance-Optimized**: Provides zero-copy or minimal-copy methods for setting
//!   and getting editor content, avoiding unnecessary `String` allocations and data
//!   shuffling, which is critical for handling large files.
//!
//! ### Architectural Notes
//! The `set_input` and `set_output` methods are generic and accept any type that can
//! be converted into a `Cow<str>`, allowing for efficient updates from both owned
//! `String`s and borrowed `&str` slices. The primary `get_` methods return a slice
//! of lines (`&[String]`) to provide a zero-copy view into the editor's buffer.
//!
//! #### Example
//! ```rust
//! use rune_xero::tui::state::editor_state::EditorState;
//!
//! let mut editor_state = EditorState::new();
//! let text_to_set = "line one\nline two";
//!
//! // Setting text is efficient, avoiding per-line allocations.
//! editor_state.set_input(text_to_set);
//!
//! // Getting a view of the lines is a zero-copy operation.
//! let lines: &[String] = editor_state.get_input();
//! assert_eq!(lines, &["line one", "line two"]);
//!
//! // Getting the content as a single String is an explicit, allocating operation.
//! let content_string = editor_state.get_input_as_string();
//! assert_eq!(content_string, "line one\nline two");
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::borrow::Cow;
use tui_textarea::TextArea;

/// Which editor panel is currently active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorMode {
    Input,
    Output,
}

/// State for input and output text areas.
pub struct EditorState<'a> {
    pub input: TextArea<'a>,
    pub output: TextArea<'a>,
    pub active: EditorMode,
}

impl<'a> EditorState<'a> {
    pub fn new() -> Self {
        let mut input = TextArea::default();
        input.set_placeholder_text("Enter JSON here or open a file (Ctrl+O)");

        let mut output = TextArea::default();
        output.set_placeholder_text("RUNE output will appear here");

        Self {
            input,
            output,
            active: EditorMode::Input,
        }
    }

    /// Sets the input text efficiently, avoiding per-line allocations.
    /// Accepts any type that can be turned into a `Cow<str>` (e.g., `String`, `&str`).
    pub fn set_input(&mut self, text: impl Into<Cow<'a, str>>) {
        let text_cow = text.into();
        self.input = TextArea::new(text_cow.lines().collect::<Vec<_>>());
        self.input
            .set_placeholder_text("Enter JSON here or open a file (Ctrl+O)");
    }

    /// Sets the output text efficiently, avoiding per-line allocations.
    /// Accepts any type that can be turned into a `Cow<str>` (e.g., `String`, `&str`).
    pub fn set_output(&mut self, text: impl Into<Cow<'a, str>>) {
        let text_cow = text.into();
        self.output = TextArea::new(text_cow.lines().collect::<Vec<_>>());
        self.output
            .set_placeholder_text("RUNE output will appear here");
    }

    /// Returns a zero-copy view of the input editor's lines.
    pub fn get_input(&self) -> &[String] {
        self.input.lines()
    }

    /// Returns a zero-copy view of the output editor's lines.
    pub fn get_output(&self) -> &[String] {
        self.output.lines()
    }

    /// Returns the input editor's content as a single, heap-allocated String.
    /// This is an explicit allocation and should be used only when necessary.
    pub fn get_input_as_string(&self) -> String {
        self.input.lines().join("\n")
    }

    /// Returns the output editor's content as a single, heap-allocated String.
    /// This is an explicit allocation and should be used only when necessary.
    pub fn get_output_as_string(&self) -> String {
        self.output.lines().join("\n")
    }

    pub fn clear_input(&mut self) {
        self.input = TextArea::default();
        self.input
            .set_placeholder_text("Enter JSON here or open a file (Ctrl+O)");
    }

    pub fn clear_output(&mut self) {
        self.output = TextArea::default();
        self.output
            .set_placeholder_text("RUNE output will appear here");
    }

    pub fn toggle_active(&mut self) {
        self.active = match self.active {
            EditorMode::Input => EditorMode::Output,
            EditorMode::Output => EditorMode::Input,
        };
    }

    pub fn is_input_active(&self) -> bool {
        self.active == EditorMode::Input
    }

    pub fn is_output_active(&self) -> bool {
        self.active == EditorMode::Output
    }
}

impl<'a> Default for EditorState<'a> {
    fn default() -> Self {
        Self::new()
    }
}