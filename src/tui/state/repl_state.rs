/* src/tui/state/repl_state.rs */
//!▫~•◦-------------------------------‣
//! # State management for the Read-Eval-Print-Loop (REPL).
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module defines `ReplState`, which encapsulates all data and logic for the
//! interactive REPL session, including command history, session variables, and
//! output logs.
//!
//! ## Key Capabilities
//! - **Session Management**: Tracks active state, input buffer, and scroll position.
//! - **Command History**: Maintains a capped-size list of executed commands for navigation.
//! - **Variable Storage**: A `HashMap` for storing session variables.
//! - **Performance-Optimized**: Methods for adding output are generic over `impl Into<Cow>`,
//!   eliminating unnecessary `String` allocations from call sites and internal logic.
//!
//! ### Architectural Notes
//! The design prioritizes low-latency interaction. By accepting `Cow` for message
//! content, the REPL can process static help text, simple error messages, and dynamic
//! multi-line output with maximum efficiency, only allocating when necessary.
//!
//! #### Example
//! ```rust
//! use rune_xero::tui::state::repl_state::ReplState;
//!
//! let mut repl_state = ReplState::new();
//!
//! // Adding static text does not require a String allocation at the call site.
//! repl_state.add_info("Processing command...");
//! repl_state.add_error("Command not found");
//!
//! // Multi-line success messages are handled efficiently.
//! let multi_line_output = "Result:\n  - Item 1\n  - Item 2";
//! repl_state.add_success(multi_line_output);
//!
//! assert!(repl_state.output.len() > 3);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::borrow::Cow;
use std::collections::HashMap;

/// REPL session state
#[derive(Debug, Clone)]
pub struct ReplState {
    /// Whether REPL is active
    pub active: bool,
    /// Current input line
    pub input: String,
    /// Cursor position within the input line
    pub cursor_position: usize,
    /// Session history (output lines)
    pub output: Vec<ReplLine>,
    /// Variables stored in session
    pub variables: HashMap<String, String>,
    /// Command history
    pub history: Vec<String>,
    /// History index for navigation
    pub history_index: Option<usize>,
    /// Last result (for _ variable)
    pub last_result: Option<String>,
    /// Scroll offset for output
    pub scroll_offset: usize,
}

/// A line in the REPL output
#[derive(Debug, Clone)]
pub struct ReplLine {
    pub kind: ReplLineKind,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplLineKind {
    Prompt,
    Success,
    Error,
    Info,
}

impl ReplState {
    pub fn new() -> Self {
        Self {
            active: false,
            input: String::new(),
            cursor_position: 0,
            output: vec![ReplLine {
                kind: ReplLineKind::Info,
                content: "RUNE REPL - Type 'help' for commands, 'exit' to close".to_string(),
            }],
            variables: HashMap::new(),
            history: Vec::new(),
            history_index: None,
            last_result: None,
            scroll_offset: 0,
        }
    }

    pub fn activate(&mut self) {
        self.active = true;
        self.input.clear();
        self.cursor_position = 0;
        self.history_index = None;
    }

    pub fn deactivate(&mut self) {
        self.active = false;
        self.input.clear();
        self.cursor_position = 0;
        self.history_index = None;
    }

    pub fn add_prompt(&mut self, cmd: &str) {
        let mut content = String::with_capacity(2 + cmd.len());
        content.push_str("> ");
        content.push_str(cmd);
        self.output.push(ReplLine {
            kind: ReplLineKind::Prompt,
            content,
        });
    }

    pub fn add_success(&mut self, msg: impl Into<Cow<'static, str>>) {
        for line in msg.into().lines() {
            self.output.push(ReplLine {
                kind: ReplLineKind::Success,
                content: line.to_string(),
            });
        }
    }

    pub fn add_error(&mut self, msg: impl Into<Cow<'static, str>>) {
        let cow = msg.into();
        let mut content = String::with_capacity(2 + cow.len());
        content.push_str("✗ ");
        content.push_str(&cow);
        self.output.push(ReplLine {
            kind: ReplLineKind::Error,
            content,
        });
    }

    pub fn add_info(&mut self, msg: impl Into<Cow<'static, str>>) {
        let cow = msg.into();
        let content =
            if cow.is_empty() || cow.starts_with("  ") || cow.starts_with('📖') {
                cow.into_owned()
            } else {
                let mut content = String::with_capacity(2 + cow.len());
                content.push_str("✓ ");
                content.push_str(&cow);
                content
            };

        self.output.push(ReplLine {
            kind: ReplLineKind::Info,
            content,
        });
    }

    pub fn add_to_history(&mut self, cmd: impl Into<String>) {
        let cmd_str = cmd.into();
        if cmd_str.trim().is_empty() {
            return;
        }
        if self.history.last() == Some(&cmd_str) {
            return;
        }
        self.history.push(cmd_str);
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }

    pub fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let new_index = self.history_index.map_or(self.history.len() - 1, |i| i.saturating_sub(1));

        self.input = self.history[new_index].clone();
        self.cursor_position = self.input.len();
        self.history_index = Some(new_index);
    }

    pub fn history_down(&mut self) {
        if let Some(i) = self.history_index {
            if i >= self.history.len() - 1 {
                self.input.clear();
                self.cursor_position = 0;
                self.history_index = None;
            } else {
                let new_idx = i + 1;
                self.input = self.history[new_idx].clone();
                self.cursor_position = self.input.len();
                self.history_index = Some(new_idx);
            }
        }
    }

    pub fn scroll_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(1);
    }

    pub fn scroll_down(&mut self, view_height: usize) {
        let max_scroll = self.output.len().saturating_sub(view_height);
        if self.scroll_offset < max_scroll {
            self.scroll_offset += 1;
        }
    }

    pub fn scroll_to_bottom(&mut self, view_height: usize) {
        self.scroll_offset = self.output.len().saturating_sub(view_height);
    }
}

impl Default for ReplState {
    fn default() -> Self {
        Self::new()
    }
}