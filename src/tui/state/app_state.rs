/* src/tui/state/app_state.rs */
//!▫~•◦-------------------------------‣
//! # Main application state for the RUNE TUI.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module defines the central `AppState` struct, which holds all UI and
//! conversion state for the entire application. It acts as the single source of
//! truth that all components render from and all events operate on.
//!
//! ## Key Capabilities
//! - **Centralized State**: Contains editor state, file state, REPL state, options,
//!   and UI visibility flags.
//! - **Mode Management**: Handles the current application mode (Encode, Decode, Rune).
//! - **Message Handling**: Provides an Elm-style `update` function for centralized
//!   event and message processing.
//! - **Performance-Optimized**: Uses `Cow<'static, str>` for status/error messages to
//!   avoid unnecessary allocations and updates configuration options in-place to

//!   avoid struct cloning.
//!
//! ### Architectural Notes
//! The `AppState` is the core of the TUI. Its design prioritizes clear ownership
//! and efficient updates. By using `Cow` for messages, the system can pass static
//! string literals from most parts of the code without heap allocation, while still
//! supporting dynamic, formatted error messages when required.
//!
//! #### Example
//! ```rust
//! // In the main application loop:
//! let mut app_state = AppState::new();
//! let msg = Msg::ToggleTheme; // An example message
//! app_state.update(msg);
//! // The theme is toggled and a status message is set without a String allocation.
//! assert_eq!(app_state.status_message.unwrap(), "Theme toggled");
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::borrow::Cow;
use std::time::{Duration, Instant};

use super::{EditorState, FileState, ReplState};
#[cfg(feature = "hydron")]
use crate::rune::hydron::eval::Evaluator;
use crate::tui::message::Msg;
use crate::{
    tui::theme::Theme,
    types::{DecodeOptions, Delimiter, EncodeOptions, Indent, KeyFoldingMode, PathExpansionMode},
};

/// Conversion mode (encode/decode/parse).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Encode, // JSON → RUNE
    Decode, // RUNE → JSON
    Rune,   // RUNE → Parsed AST + RUNE blocks
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
            Mode::Encode => "Encode (JSON → RUNE)",
            Mode::Decode => "Decode (RUNE → JSON)",
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

/// A message with an expiry time.
pub struct TimedMessage {
    pub content: Cow<'static, str>,
    pub expiry: Instant,
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
    pub error_message: Option<TimedMessage>,
    pub status_message: Option<TimedMessage>,
    pub stats: Option<ConversionStats>,
    pub should_quit: bool,
}

/// Actions that require user confirmation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    /// Checks for and removes expired messages.
    pub fn clear_expired_messages(&mut self) {
        let now = Instant::now();
        if self.status_message.as_ref().is_some_and(|m| now > m.expiry) {
            self.status_message = None;
        }
        if self.error_message.as_ref().is_some_and(|m| now > m.expiry) {
            self.error_message = None;
        }
    }

    pub fn toggle_mode(&mut self) {
        self.mode = self.mode.toggle();
        self.clear_messages();
    }

    pub fn toggle_theme(&mut self) {
        self.theme = self.theme.toggle();
        self.set_status("Theme toggled");
    }

    pub fn set_error(&mut self, msg: impl Into<Cow<'static, str>>) {
        self.error_message = Some(TimedMessage {
            content: msg.into(),
            expiry: Instant::now() + Duration::from_secs(5),
        });
        self.status_message = None;
    }

    pub fn set_status(&mut self, msg: impl Into<Cow<'static, str>>) {
        self.status_message = Some(TimedMessage {
            content: msg.into(),
            expiry: Instant::now() + Duration::from_secs(5),
        });
        self.error_message = None;
    }

    pub fn clear_messages(&mut self) {
        self.error_message = None;
        self.status_message = None;
    }

    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    pub fn cycle_delimiter(&mut self) {
        self.encode_options.delimiter = match self.encode_options.delimiter {
            Delimiter::Comma => Delimiter::Tab,
            Delimiter::Tab => Delimiter::Pipe,
            Delimiter::Pipe => Delimiter::Comma,
        };
    }

    pub fn increase_indent(&mut self) {
        if let Indent::Spaces(current) = &mut self.encode_options.indent {
            if *current < 8 {
                *current += 1;
            }
        }
    }

    pub fn decrease_indent(&mut self) {
        if let Indent::Spaces(current) = &mut self.encode_options.indent {
            if *current > 1 {
                *current -= 1;
            }
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

    pub fn set_flatten_depth_unlimited(&mut self) {
        self.encode_options.flatten_depth = usize::MAX;
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