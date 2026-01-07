//! Editor state for input/output text areas.

use tui_textarea::TextArea;

/// Which panel is currently active.
///
/// # Examples
/// ```
/// use toon_format::tui::state::editor_state::EditorMode;
///
/// let mode = EditorMode::Input;
/// let _ = mode;
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EditorMode {
    Input,
    Output,
}

/// State for input and output text areas.
///
/// # Examples
/// ```
/// use toon_format::tui::state::EditorState;
///
/// let state = EditorState::new();
/// let _ = state;
/// ```
pub struct EditorState<'a> {
    pub input: TextArea<'a>,
    pub output: TextArea<'a>,
    pub active: EditorMode,
}

impl<'a> EditorState<'a> {
    /// Create a new editor state with placeholder text.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::EditorState;
    ///
    /// let state = EditorState::new();
    /// let _ = state;
    /// ```
    pub fn new() -> Self {
        let mut input = TextArea::default();
        input.set_placeholder_text("Enter JSON here or open a file (Ctrl+O)");

        let mut output = TextArea::default();
        output.set_placeholder_text("TOON output will appear here");

        Self {
            input,
            output,
            active: EditorMode::Input,
        }
    }

    /// Replace the input text.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::EditorState;
    ///
    /// let mut state = EditorState::new();
    /// state.set_input("{}".to_string());
    /// ```
    pub fn set_input(&mut self, text: String) {
        let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        self.input = TextArea::from(lines);
    }

    /// Replace the output text.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::EditorState;
    ///
    /// let mut state = EditorState::new();
    /// state.set_output("result".to_string());
    /// ```
    pub fn set_output(&mut self, text: String) {
        let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        self.output = TextArea::from(lines);
    }

    /// Read the current input text.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::EditorState;
    ///
    /// let state = EditorState::new();
    /// let _ = state.get_input();
    /// ```
    pub fn get_input(&self) -> String {
        self.input.lines().join("\n")
    }

    /// Read the current output text.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::EditorState;
    ///
    /// let state = EditorState::new();
    /// let _ = state.get_output();
    /// ```
    pub fn get_output(&self) -> String {
        self.output.lines().join("\n")
    }

    /// Clear the input editor and restore the placeholder.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::EditorState;
    ///
    /// let mut state = EditorState::new();
    /// state.clear_input();
    /// ```
    pub fn clear_input(&mut self) {
        self.input = TextArea::default();
        self.input
            .set_placeholder_text("Enter JSON here or open a file (Ctrl+O)");
    }

    /// Clear the output editor and restore the placeholder.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::EditorState;
    ///
    /// let mut state = EditorState::new();
    /// state.clear_output();
    /// ```
    pub fn clear_output(&mut self) {
        self.output = TextArea::default();
        self.output
            .set_placeholder_text("TOON output will appear here");
    }

    /// Toggle which panel is active.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::EditorState;
    ///
    /// let mut state = EditorState::new();
    /// state.toggle_active();
    /// ```
    pub fn toggle_active(&mut self) {
        self.active = match self.active {
            EditorMode::Input => EditorMode::Output,
            EditorMode::Output => EditorMode::Input,
        };
    }

    /// Return true when the input panel is active.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::EditorState;
    ///
    /// let state = EditorState::new();
    /// assert!(state.is_input_active());
    /// ```
    pub fn is_input_active(&self) -> bool {
        self.active == EditorMode::Input
    }

    /// Return true when the output panel is active.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::EditorState;
    ///
    /// let mut state = EditorState::new();
    /// state.toggle_active();
    /// assert!(state.is_output_active());
    /// ```
    pub fn is_output_active(&self) -> bool {
        self.active == EditorMode::Output
    }
}

impl<'a> Default for EditorState<'a> {
    fn default() -> Self {
        Self::new()
    }
}
