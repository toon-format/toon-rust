//! Main application state.

use super::{EditorState, FileState, ReplState};
use crate::{
    tui::theme::Theme,
    types::{DecodeOptions, Delimiter, EncodeOptions, Indent, KeyFoldingMode, PathExpansionMode},
};

/// Conversion mode (encode/decode).
///
/// # Examples
/// ```
/// use toon_format::tui::state::app_state::Mode;
///
/// let mode = Mode::Encode;
/// assert_eq!(mode.short_name(), "Encode");
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Encode,
    Decode,
}

impl Mode {
    /// Toggle between encode and decode.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::app_state::Mode;
    ///
    /// let mode = Mode::Encode.toggle();
    /// assert_eq!(mode, Mode::Decode);
    /// ```
    pub fn toggle(&self) -> Self {
        match self {
            Mode::Encode => Mode::Decode,
            Mode::Decode => Mode::Encode,
        }
    }

    /// Return the full display name for the mode.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::app_state::Mode;
    ///
    /// assert!(Mode::Encode.as_str().contains("Encode"));
    /// ```
    pub fn as_str(&self) -> &'static str {
        match self {
            Mode::Encode => "Encode (JSON → TOON)",
            Mode::Decode => "Decode (TOON → JSON)",
        }
    }

    /// Return a short display name for the mode.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::app_state::Mode;
    ///
    /// assert_eq!(Mode::Decode.short_name(), "Decode");
    /// ```
    pub fn short_name(&self) -> &'static str {
        match self {
            Mode::Encode => "Encode",
            Mode::Decode => "Decode",
        }
    }
}

/// Statistics from the last conversion.
///
/// # Examples
/// ```
/// use toon_format::tui::state::app_state::ConversionStats;
///
/// let stats = ConversionStats {
///     json_tokens: 1,
///     toon_tokens: 1,
///     json_bytes: 1,
///     toon_bytes: 1,
///     token_savings: 0.0,
///     byte_savings: 0.0,
/// };
/// let _ = stats;
/// ```
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
///
/// # Examples
/// ```
/// use toon_format::tui::state::AppState;
///
/// let state = AppState::new();
/// let _ = state;
/// ```
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
    /// Create a new application state with default values.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let state = AppState::new();
    /// let _ = state;
    /// ```
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

    /// Toggle between encode and decode modes.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.toggle_mode();
    /// ```
    pub fn toggle_mode(&mut self) {
        self.mode = self.mode.toggle();
        self.clear_error();
        self.clear_status();
    }

    /// Toggle the active theme.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.toggle_theme();
    /// ```
    pub fn toggle_theme(&mut self) {
        self.theme = self.theme.toggle();
        self.set_status("Theme toggled".to_string());
    }

    /// Set an error message and clear the status.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.set_error("Oops".to_string());
    /// ```
    pub fn set_error(&mut self, msg: String) {
        self.error_message = Some(msg);
        self.status_message = None;
    }

    /// Set a status message and clear the error.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.set_status("OK".to_string());
    /// ```
    pub fn set_status(&mut self, msg: String) {
        self.status_message = Some(msg);
        self.error_message = None;
    }

    /// Clear the current error message.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.clear_error();
    /// ```
    pub fn clear_error(&mut self) {
        self.error_message = None;
    }

    /// Clear the current status message.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.clear_status();
    /// ```
    pub fn clear_status(&mut self) {
        self.status_message = None;
    }

    /// Mark the application to quit.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.quit();
    /// ```
    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    /// Toggle the settings panel.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.toggle_settings();
    /// ```
    pub fn toggle_settings(&mut self) {
        self.show_settings = !self.show_settings;
        if self.show_settings {
            self.show_help = false;
            self.show_file_browser = false;
            self.show_history = false;
            self.show_diff = false;
        }
    }

    /// Toggle the help overlay.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.toggle_help();
    /// ```
    pub fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
        if self.show_help {
            self.show_settings = false;
            self.show_file_browser = false;
            self.show_history = false;
            self.show_diff = false;
        }
    }

    /// Toggle the file browser panel.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.toggle_file_browser();
    /// ```
    pub fn toggle_file_browser(&mut self) {
        self.show_file_browser = !self.show_file_browser;
        if self.show_file_browser {
            self.show_settings = false;
            self.show_help = false;
            self.show_history = false;
            self.show_diff = false;
        }
    }

    /// Toggle the conversion history panel.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.toggle_history();
    /// ```
    pub fn toggle_history(&mut self) {
        self.show_history = !self.show_history;
        if self.show_history {
            self.show_settings = false;
            self.show_help = false;
            self.show_file_browser = false;
            self.show_diff = false;
        }
    }

    /// Toggle the diff panel.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.toggle_diff();
    /// ```
    pub fn toggle_diff(&mut self) {
        self.show_diff = !self.show_diff;
        if self.show_diff {
            self.show_settings = false;
            self.show_help = false;
            self.show_file_browser = false;
            self.show_history = false;
        }
    }

    /// Cycle the active encode delimiter.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.cycle_delimiter();
    /// ```
    pub fn cycle_delimiter(&mut self) {
        self.encode_options.delimiter = match self.encode_options.delimiter {
            Delimiter::Comma => Delimiter::Tab,
            Delimiter::Tab => Delimiter::Pipe,
            Delimiter::Pipe => Delimiter::Comma,
        };
    }

    /// Increase indentation (up to the maximum).
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.increase_indent();
    /// ```
    pub fn increase_indent(&mut self) {
        let Indent::Spaces(current) = self.encode_options.indent;
        if current < 8 {
            self.encode_options.indent = Indent::Spaces(current + 1);
        }
    }

    /// Decrease indentation (down to the minimum).
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.decrease_indent();
    /// ```
    pub fn decrease_indent(&mut self) {
        let Indent::Spaces(current) = self.encode_options.indent;
        if current > 1 {
            self.encode_options.indent = Indent::Spaces(current - 1);
        }
    }

    /// Toggle key folding on/off.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.toggle_fold_keys();
    /// ```
    pub fn toggle_fold_keys(&mut self) {
        self.encode_options.key_folding = match self.encode_options.key_folding {
            KeyFoldingMode::Off => KeyFoldingMode::Safe,
            KeyFoldingMode::Safe => KeyFoldingMode::Off,
        };
    }

    /// Increase the flatten depth used for key folding.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.increase_flatten_depth();
    /// ```
    pub fn increase_flatten_depth(&mut self) {
        if self.encode_options.flatten_depth == usize::MAX {
            self.encode_options.flatten_depth = 2;
        } else if self.encode_options.flatten_depth < 10 {
            self.encode_options.flatten_depth += 1;
        }
    }

    /// Decrease the flatten depth used for key folding.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.decrease_flatten_depth();
    /// ```
    pub fn decrease_flatten_depth(&mut self) {
        if self.encode_options.flatten_depth == 2 {
            self.encode_options.flatten_depth = usize::MAX;
        } else if self.encode_options.flatten_depth > 2
            && self.encode_options.flatten_depth != usize::MAX
        {
            self.encode_options.flatten_depth -= 1;
        }
    }

    /// Toggle key folding depth between default and minimum.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.toggle_flatten_depth();
    /// ```
    pub fn toggle_flatten_depth(&mut self) {
        if self.encode_options.flatten_depth == usize::MAX {
            self.encode_options.flatten_depth = 2;
        } else {
            self.encode_options.flatten_depth = usize::MAX;
        }
    }

    /// Toggle path expansion on/off.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.toggle_expand_paths();
    /// ```
    pub fn toggle_expand_paths(&mut self) {
        self.decode_options.expand_paths = match self.decode_options.expand_paths {
            PathExpansionMode::Off => PathExpansionMode::Safe,
            PathExpansionMode::Safe => PathExpansionMode::Off,
        };
    }

    /// Toggle strict mode for decoding.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.toggle_strict();
    /// ```
    pub fn toggle_strict(&mut self) {
        self.decode_options.strict = !self.decode_options.strict;
    }

    /// Toggle type coercion for decoding.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::state::AppState;
    ///
    /// let mut state = AppState::new();
    /// state.toggle_coerce_types();
    /// ```
    pub fn toggle_coerce_types(&mut self) {
        self.decode_options.coerce_types = !self.decode_options.coerce_types;
    }
}

impl<'a> Default for AppState<'a> {
    fn default() -> Self {
        Self::new()
    }
}
