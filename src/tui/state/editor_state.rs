//! Editor state for input/output text areas.

use tui_textarea::TextArea;

/// Which panel is currently active.
#[derive(Debug, Clone, Copy, PartialEq)]
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
        output.set_placeholder_text("TOON output will appear here");

        Self {
            input,
            output,
            active: EditorMode::Input,
        }
    }

    pub fn set_input(&mut self, text: String) {
        let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        self.input = TextArea::from(lines);
    }

    pub fn set_output(&mut self, text: String) {
        let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        self.output = TextArea::from(lines);
    }

    pub fn get_input(&self) -> String {
        self.input.lines().join("\n")
    }

    pub fn get_output(&self) -> String {
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
            .set_placeholder_text("TOON output will appear here");
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
