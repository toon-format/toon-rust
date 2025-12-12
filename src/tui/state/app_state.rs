//! Main application state.

use super::{EditorState, FileState, ReplState};
#[cfg(feature = "hydron")]
use crate::rune::hydron::eval::Evaluator;
use crate::tui::message::Msg;
use crate::{
    tui::theme::Theme,
    types::{DecodeOptions, Delimiter, EncodeOptions, Indent, KeyFoldingMode, PathExpansionMode},
};

/// Conversion mode (encode/decode/parse).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Encode, // JSON → TOON
    Decode, // TOON → JSON
    Rune,   // RUNE → Parsed AST + TOON blocks
}

impl Mode {
    pub fn toggle(&self) -> Self {
        match self {
            Mode::Encode => Mode::Decode,
            Mode::Decode => Mode::Rune,
            Mode::Rune => Mode::Encode,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Mode::Encode => "Encode (JSON → TOON)",
            Mode::Decode => "Decode (TOON → JSON)",
            Mode::Rune => "Parse (RUNE → Results)",
        }
    }

    pub fn short_name(&self) -> &'static str {
        match self {
            Mode::Encode => "Encode",
            Mode::Decode => "Decode",
            Mode::Rune => "RUNE",
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
    #[cfg(feature = "hydron")]
    pub rune_eval: Evaluator,
    pub theme: Theme,
    pub encode_options: EncodeOptions,
    pub decode_options: DecodeOptions,
    pub show_settings: bool,
    pub show_help: bool,
    pub show_file_browser: bool,
    pub show_history: bool,
    pub show_diff: bool,
    pub show_confirmation: bool,
    pub confirmation_action: ConfirmationAction,
    pub error_message: Option<String>,
    pub status_message: Option<String>,
    pub stats: Option<ConversionStats>,
    pub should_quit: bool,
}

/// Actions that require user confirmation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConfirmationAction {
    None,
    NewFile,
    Quit,
    DeleteFile,
}

impl<'a> AppState<'a> {
    pub fn new() -> Self {
        Self {
            mode: Mode::Encode,
            editor: EditorState::new(),
            file_state: FileState::new(),
            repl: ReplState::new(),
            #[cfg(feature = "hydron")]
            rune_eval: Evaluator::new(),
            theme: Theme::default(),

            encode_options: EncodeOptions::default(),
            decode_options: DecodeOptions::default(),

            show_settings: false,
            show_help: false,
            show_file_browser: false,
            show_history: false,
            show_diff: false,
            show_confirmation: false,
            confirmation_action: ConfirmationAction::None,

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
        self.encode_options =
            self.encode_options
                .clone()
                .with_delimiter(match self.encode_options.delimiter {
                    Delimiter::Comma => Delimiter::Tab,
                    Delimiter::Tab => Delimiter::Pipe,
                    Delimiter::Pipe => Delimiter::Comma,
                });
    }

    pub fn increase_indent(&mut self) {
        let Indent::Spaces(current) = self.encode_options.indent;
        if current < 8 {
            self.encode_options = self
                .encode_options
                .clone()
                .with_indent(Indent::Spaces(current + 1));
        }
    }

    pub fn decrease_indent(&mut self) {
        let Indent::Spaces(current) = self.encode_options.indent;
        if current > 1 {
            self.encode_options = self
                .encode_options
                .clone()
                .with_indent(Indent::Spaces(current - 1));
        }
    }

    pub fn toggle_fold_keys(&mut self) {
        self.encode_options =
            self.encode_options
                .clone()
                .with_key_folding(match self.encode_options.key_folding {
                    KeyFoldingMode::Off => KeyFoldingMode::Safe,
                    KeyFoldingMode::Safe => KeyFoldingMode::Off,
                });
    }

    pub fn increase_flatten_depth(&mut self) {
        if self.encode_options.flatten_depth == usize::MAX {
            self.encode_options = self.encode_options.clone().with_flatten_depth(2);
        } else if self.encode_options.flatten_depth < 10 {
            self.encode_options = self
                .encode_options
                .clone()
                .with_flatten_depth(self.encode_options.flatten_depth + 1);
        }
    }

    pub fn decrease_flatten_depth(&mut self) {
        if self.encode_options.flatten_depth == 2 {
            self.encode_options = self.encode_options.clone().with_flatten_depth(usize::MAX);
        } else if self.encode_options.flatten_depth > 2
            && self.encode_options.flatten_depth != usize::MAX
        {
            self.encode_options = self
                .encode_options
                .clone()
                .with_flatten_depth(self.encode_options.flatten_depth - 1);
        }
    }

    pub fn toggle_flatten_depth(&mut self) {
        if self.encode_options.flatten_depth == usize::MAX {
            self.encode_options = self.encode_options.clone().with_flatten_depth(2);
        } else {
            self.encode_options = self.encode_options.clone().with_flatten_depth(usize::MAX);
        }
    }

    pub fn toggle_expand_paths(&mut self) {
        self.decode_options =
            self.decode_options
                .clone()
                .with_expand_paths(match self.decode_options.expand_paths {
                    PathExpansionMode::Off => PathExpansionMode::Safe,
                    PathExpansionMode::Safe => PathExpansionMode::Off,
                });
    }

    pub fn toggle_strict(&mut self) {
        let strict = !self.decode_options.strict;
        self.decode_options = self.decode_options.clone().with_strict(strict);
    }

    pub fn toggle_coerce_types(&mut self) {
        let coerce = !self.decode_options.coerce_types;
        self.decode_options = self.decode_options.clone().with_coerce_types(coerce);
    }

    /// Central message handler for TUI actions (Elm-style update).
    pub fn update(&mut self, msg: Msg) -> Option<Msg> {
        match msg {
            Msg::Quit => {
                self.quit();
                None
            }
            Msg::ToggleMode => {
                self.toggle_mode();
                None
            }
            Msg::ToggleSettings => {
                self.toggle_settings();
                None
            }
            Msg::ToggleHelp => {
                self.toggle_help();
                None
            }
            Msg::ToggleFileBrowser => {
                self.toggle_file_browser();
                None
            }
            Msg::ToggleHistory => {
                self.toggle_history();
                None
            }
            Msg::ToggleDiff => {
                self.toggle_diff();
                None
            }
            Msg::ToggleTheme => {
                self.toggle_theme();
                None
            }
            Msg::SetError(e) => {
                self.set_error(e);
                None
            }
            Msg::SetStatus(s) => {
                self.set_status(s);
                None
            }
            Msg::ClearError => {
                self.clear_error();
                None
            }
            Msg::ClearStatus => {
                self.clear_status();
                None
            }
            _ => None, // Not yet handled messages
        }
    }
}

impl<'a> Default for AppState<'a> {
    fn default() -> Self {
        Self::new()
    }
}
