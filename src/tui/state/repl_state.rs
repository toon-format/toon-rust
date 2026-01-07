//! REPL state - separate from command mode

use std::collections::HashMap;

/// REPL session state.
///
/// # Examples
/// ```
/// use toon_format::tui::state::ReplState;
///
/// let state = ReplState::new();
/// let _ = state;
/// ```
#[derive(Debug, Clone)]
pub struct ReplState {
    /// Whether REPL is active
    pub active: bool,
    /// Current input line
    pub input: String,
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

/// A line in the REPL output.
///
/// # Examples
/// ```
/// use toon_format::tui::state::{ReplLine, ReplLineKind};
///
/// let line = ReplLine {
///     kind: ReplLineKind::Info,
///     content: "hello".to_string(),
/// };
/// let _ = line;
/// ```
#[derive(Debug, Clone)]
pub struct ReplLine {
    pub kind: ReplLineKind,
    pub content: String,
}

/// Classification of REPL output lines.
///
/// # Examples
/// ```
/// use toon_format::tui::state::ReplLineKind;
///
/// let kind = ReplLineKind::Success;
/// let _ = kind;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum ReplLineKind {
    Prompt,
    Success,
    Error,
    Info,
}

impl ReplState {
    /// Create a new REPL state with defaults.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::ReplState;
    ///
    /// let state = ReplState::new();
    /// let _ = state;
    /// ```
    pub fn new() -> Self {
        Self {
            active: false,
            input: String::new(),
            output: vec![ReplLine {
                kind: ReplLineKind::Info,
                content: "TOON REPL - Type 'help' for commands, 'exit' to close".to_string(),
            }],
            variables: HashMap::new(),
            history: Vec::new(),
            history_index: None,
            last_result: None,
            scroll_offset: 0,
        }
    }

    /// Activate REPL mode and reset input.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::ReplState;
    ///
    /// let mut state = ReplState::new();
    /// state.activate();
    /// ```
    pub fn activate(&mut self) {
        self.active = true;
        self.input.clear();
        self.history_index = None;
    }

    /// Deactivate REPL mode and reset input.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::ReplState;
    ///
    /// let mut state = ReplState::new();
    /// state.deactivate();
    /// ```
    pub fn deactivate(&mut self) {
        self.active = false;
        self.input.clear();
        self.history_index = None;
    }

    /// Add a prompt line to the REPL output.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::ReplState;
    ///
    /// let mut state = ReplState::new();
    /// state.add_prompt("help");
    /// ```
    pub fn add_prompt(&mut self, cmd: &str) {
        self.output.push(ReplLine {
            kind: ReplLineKind::Prompt,
            content: format!("> {cmd}"),
        });
    }

    /// Add a success message to the REPL output.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::ReplState;
    ///
    /// let mut state = ReplState::new();
    /// state.add_success("ok".to_string());
    /// ```
    pub fn add_success(&mut self, msg: String) {
        for line in msg.lines() {
            self.output.push(ReplLine {
                kind: ReplLineKind::Success,
                content: line.to_string(),
            });
        }
    }

    /// Add an error message to the REPL output.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::ReplState;
    ///
    /// let mut state = ReplState::new();
    /// state.add_error("oops".to_string());
    /// ```
    pub fn add_error(&mut self, msg: String) {
        self.output.push(ReplLine {
            kind: ReplLineKind::Error,
            content: format!("✗ {msg}"),
        });
    }

    /// Add an informational message to the REPL output.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::ReplState;
    ///
    /// let mut state = ReplState::new();
    /// state.add_info("info".to_string());
    /// ```
    pub fn add_info(&mut self, msg: String) {
        let content = if msg.is_empty() || msg.starts_with("  ") || msg.starts_with("📖") {
            msg
        } else {
            format!("✓ {msg}")
        };

        self.output.push(ReplLine {
            kind: ReplLineKind::Info,
            content,
        });
    }

    /// Add a command to the history buffer.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::ReplState;
    ///
    /// let mut state = ReplState::new();
    /// state.add_to_history("encode".to_string());
    /// ```
    pub fn add_to_history(&mut self, cmd: String) {
        if cmd.trim().is_empty() {
            return;
        }
        if self.history.last() == Some(&cmd) {
            return;
        }
        self.history.push(cmd);
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }

    /// Move up in command history.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::ReplState;
    ///
    /// let mut state = ReplState::new();
    /// state.add_to_history("encode".to_string());
    /// state.history_up();
    /// ```
    pub fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let new_index = match self.history_index {
            None => Some(self.history.len() - 1),
            Some(0) => Some(0),
            Some(i) => Some(i - 1),
        };
        if let Some(idx) = new_index {
            self.input = self.history[idx].clone();
            self.history_index = new_index;
        }
    }

    /// Move down in command history.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::ReplState;
    ///
    /// let mut state = ReplState::new();
    /// state.add_to_history("encode".to_string());
    /// state.history_up();
    /// state.history_down();
    /// ```
    pub fn history_down(&mut self) {
        match self.history_index {
            None => (),
            Some(i) if i >= self.history.len() - 1 => {
                self.input.clear();
                self.history_index = None;
            }
            Some(i) => {
                let new_idx = i + 1;
                self.input = self.history[new_idx].clone();
                self.history_index = Some(new_idx);
            }
        }
    }

    /// Scroll the REPL output up.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::ReplState;
    ///
    /// let mut state = ReplState::new();
    /// state.scroll_up();
    /// ```
    pub fn scroll_up(&mut self) {
        if self.scroll_offset > 0 {
            self.scroll_offset -= 1;
        }
    }

    /// Scroll the REPL output down.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::ReplState;
    ///
    /// let mut state = ReplState::new();
    /// state.scroll_down(10);
    /// ```
    pub fn scroll_down(&mut self, visible_lines: usize) {
        let max_scroll = self.output.len().saturating_sub(visible_lines);
        if self.scroll_offset < max_scroll {
            self.scroll_offset += 1;
        }
    }

    /// Scroll to the bottom of the REPL output.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::ReplState;
    ///
    /// let mut state = ReplState::new();
    /// state.scroll_to_bottom();
    /// ```
    pub fn scroll_to_bottom(&mut self) {
        if self.output.len() <= 30 {
            self.scroll_offset = 0;
        } else {
            self.scroll_offset = self.output.len().saturating_sub(30);
        }
    }
}

impl Default for ReplState {
    fn default() -> Self {
        Self::new()
    }
}
