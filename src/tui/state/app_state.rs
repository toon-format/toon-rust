//! Main application state.

use super::{EditorState, FileState, ReplState};
use crate::{
    tui::theme::Theme,
    types::{DecodeOptions, Delimiter, EncodeOptions, Indent, KeyFoldingMode, PathExpansionMode},
};

/// Conversion mode (encode/decode).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Encode,
    Decode,
}

impl Mode {
    pub fn toggle(&self) -> Self {
        match self {
            Mode::Encode => Mode::Decode,
            Mode::Decode => Mode::Encode,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Mode::Encode => "Encode (JSON → TOON)",
            Mode::Decode => "Decode (TOON → JSON)",
        }
    }

    pub fn short_name(&self) -> &'static str {
        match self {
            Mode::Encode => "Encode",
            Mode::Decode => "Decode",
        }
    }
}

/// Statistics from the last conversion.
#[derive(Debug, Clone)]
pub struct ConversionStats {
    pub json_tokens: usize,
    pub toon_tokens: usize,
    pub json_bytes: usize,
    pub toon_bytes: usize,
    pub token_savings: f64,
    pub byte_savings: f64,
}

/// Central application state containing all UI and conversion state.
pub struct AppState<'a> {
    pub mode: Mode,
    pub editor: EditorState<'a>,
    pub file_state: FileState,
    pub repl: ReplState,
    pub theme: Theme,
    pub encode_options: EncodeOptions,
    pub decode_options: DecodeOptions,
    pub show_settings: bool,
    pub show_help: bool,
    pub show_file_browser: bool,
    pub show_history: bool,
    pub show_diff: bool,
    pub error_message: Option<String>,
    pub status_message: Option<String>,
    pub stats: Option<ConversionStats>,
    pub should_quit: bool,
}

impl<'a> AppState<'a> {
    pub fn new() -> Self {
        Self {
            mode: Mode::Encode,
            editor: EditorState::new(),
            file_state: FileState::new(),
            repl: ReplState::new(),
            theme: Theme::default(),

            encode_options: EncodeOptions::default(),
            decode_options: DecodeOptions::default(),

            show_settings: false,
            show_help: false,
            show_file_browser: false,
            show_history: false,
            show_diff: false,

            error_message: None,
            status_message: None,
            stats: None,

            should_quit: false,
        }
    }

    pub fn toggle_mode(&mut self) {
        self.mode = self.mode.toggle();
        self.clear_error();
        self.clear_status();
    }

    pub fn toggle_theme(&mut self) {
        self.theme = self.theme.toggle();
        self.set_status("Theme toggled".to_string());
    }

    pub fn set_error(&mut self, msg: String) {
        self.error_message = Some(msg);
        self.status_message = None;
    }

    pub fn set_status(&mut self, msg: String) {
        self.status_message = Some(msg);
        self.error_message = None;
    }

    pub fn clear_error(&mut self) {
        self.error_message = None;
    }

    pub fn clear_status(&mut self) {
        self.status_message = None;
    }

    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    pub fn toggle_settings(&mut self) {
        self.show_settings = !self.show_settings;
        if self.show_settings {
            self.show_help = false;
            self.show_file_browser = false;
            self.show_history = false;
            self.show_diff = false;
        }
    }

    pub fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
        if self.show_help {
            self.show_settings = false;
            self.show_file_browser = false;
            self.show_history = false;
            self.show_diff = false;
        }
    }

    pub fn toggle_file_browser(&mut self) {
        self.show_file_browser = !self.show_file_browser;
        if self.show_file_browser {
            self.show_settings = false;
            self.show_help = false;
            self.show_history = false;
            self.show_diff = false;
        }
    }

    pub fn toggle_history(&mut self) {
        self.show_history = !self.show_history;
        if self.show_history {
            self.show_settings = false;
            self.show_help = false;
            self.show_file_browser = false;
            self.show_diff = false;
        }
    }

    pub fn toggle_diff(&mut self) {
        self.show_diff = !self.show_diff;
        if self.show_diff {
            self.show_settings = false;
            self.show_help = false;
            self.show_file_browser = false;
            self.show_history = false;
        }
    }

    pub fn cycle_delimiter(&mut self) {
        self.encode_options.delimiter = match self.encode_options.delimiter {
            Delimiter::Comma => Delimiter::Tab,
            Delimiter::Tab => Delimiter::Pipe,
            Delimiter::Pipe => Delimiter::Comma,
        };
    }

    pub fn increase_indent(&mut self) {
        let Indent::Spaces(current) = self.encode_options.indent;
        if current < 8 {
            self.encode_options.indent = Indent::Spaces(current + 1);
        }
    }

    pub fn decrease_indent(&mut self) {
        let Indent::Spaces(current) = self.encode_options.indent;
        if current > 1 {
            self.encode_options.indent = Indent::Spaces(current - 1);
        }
    }

    pub fn toggle_fold_keys(&mut self) {
        self.encode_options.key_folding = match self.encode_options.key_folding {
            KeyFoldingMode::Off => KeyFoldingMode::Safe,
            KeyFoldingMode::Safe => KeyFoldingMode::Off,
        };
    }

    pub fn increase_flatten_depth(&mut self) {
        if self.encode_options.flatten_depth == usize::MAX {
            self.encode_options.flatten_depth = 2;
        } else if self.encode_options.flatten_depth < 10 {
            self.encode_options.flatten_depth += 1;
        }
    }

    pub fn decrease_flatten_depth(&mut self) {
        if self.encode_options.flatten_depth == 2 {
            self.encode_options.flatten_depth = usize::MAX;
        } else if self.encode_options.flatten_depth > 2
            && self.encode_options.flatten_depth != usize::MAX
        {
            self.encode_options.flatten_depth -= 1;
        }
    }

    pub fn toggle_flatten_depth(&mut self) {
        if self.encode_options.flatten_depth == usize::MAX {
            self.encode_options.flatten_depth = 2;
        } else {
            self.encode_options.flatten_depth = usize::MAX;
        }
    }

    pub fn toggle_expand_paths(&mut self) {
        self.decode_options.expand_paths = match self.decode_options.expand_paths {
            PathExpansionMode::Off => PathExpansionMode::Safe,
            PathExpansionMode::Safe => PathExpansionMode::Off,
        };
    }

    pub fn toggle_strict(&mut self) {
        self.decode_options.strict = !self.decode_options.strict;
    }

    pub fn toggle_coerce_types(&mut self) {
        self.decode_options.coerce_types = !self.decode_options.coerce_types;
    }
}

impl<'a> Default for AppState<'a> {
    fn default() -> Self {
        Self::new()
    }
}
