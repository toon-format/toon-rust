/* src/tui/app.rs */
//!▫~•◦-------------------------------‣
//! # Main TUI application managing state, events, and rendering.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module contains the `TuiApp` struct, the core of the interactive TUI.
//! It owns the application state, manages the main event loop, handles all user
//! input, and orchestrates rendering.
//!
//! ## Key Capabilities
//! - **Event Loop Management**: Runs the main loop, polling for keyboard and terminal events.
//! - **State Ownership**: Owns the single `AppState` struct, the source of truth for the UI.
//! - **Event Handling**: Dispatches all keyboard events to the appropriate handlers based
//!   on the current application context (e.g., editor, REPL, file browser).
//! - **Business Logic**: Contains the core logic for file operations, text conversion
//!   (encode/decode), and REPL command execution.
//!
//! ### Architectural Notes
//! The `TuiApp` is the highest-level component in the TUI. Its design focuses on a
//! clear, centralized event handling pipeline. All operations are designed to be
//! highly performant, leveraging the zero-copy APIs of the underlying state modules
//! to ensure a responsive user experience with minimal allocations.
//!
//! #### Example
//! ```rust
//! // In main.rs, after setting up the terminal:
//! // let mut app = TuiApp::new();
//! // app.run(&mut terminal).unwrap();
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::{fmt::Write, fs, path::Path, time::Duration};

use chrono::Local;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tiktoken_rs::cl100k_base;
use yoshi::{yoshi, Context, Hatch};

#[cfg(feature = "hydron")]
use crate::rune::hydron::values::Value;

use crate::{
    decode, encode,
    tui::{
        events::{Event, EventHandler},
        keybindings::{Action, KeyBindings},
        repl_command::ReplCommand,
        state::{
            app_state::{ConfirmationAction, ConversionStats},
            AppState, ConversionHistory, Mode,
        },
        ui,
    },
};

/// Main TUI application managing state, events, and rendering.
pub struct TuiApp<'a> {
    pub app_state: AppState<'a>,
}

impl<'a> TuiApp<'a> {
    pub fn new() -> Self {
        Self {
            app_state: AppState::new(),
        }
    }

    /// Runs the main application event loop.
    pub fn run<B: ratatui::backend::Backend>(
        &mut self,
        terminal: &mut ratatui::Terminal<B>,
    ) -> Hatch<()> {
        let mut event_handler = EventHandler::new();
        while !self.app_state.should_quit {
            self.app_state.clear_expired_messages();
            terminal.draw(|f| ui::render(f, &mut self.app_state))?;

            if let Some(event) = event_handler.poll(Duration::from_millis(100))? {
                self.handle_event(event)?;
            }
        }
        Ok(())
    }

    /// The main event dispatcher.
    fn handle_event(&mut self, event: Event) -> Hatch<()> {
        if let Event::Key(key) = event {
            self.handle_key_event(key)?;
        }
        Ok(())
    }

    /// Dispatches key events to the correct handler based on application context.
    fn handle_key_event(&mut self, key: KeyEvent) -> Hatch<()> {
        if self.app_state.show_confirmation {
            return self.handle_confirmation_key(key);
        }
        if self.app_state.repl.active {
            return self.handle_repl_key(key);
        }
        if self.app_state.show_help
            || self.app_state.show_file_browser
            || self.app_state.show_history
            || self.app_state.show_diff
            || self.app_state.show_settings
        {
            return self.handle_modal_key(key);
        }
        self.handle_editor_key(key)
    }
}

/// TuiApp impl block for event handlers.
impl<'a> TuiApp<'a> {
    fn handle_modal_key(&mut self, key: KeyEvent) -> Hatch<()> {
        let mut close_all_modals = false;
        match key.code {
            KeyCode::Esc => close_all_modals = true,
            KeyCode::F(1) if self.app_state.show_help => self.app_state.show_help = false,
            _ => {
                if self.app_state.show_file_browser {
                    self.handle_file_browser_key(key)?;
                } else if self.app_state.show_settings {
                    self.handle_settings_key(key)?;
                }
            }
        }

        if close_all_modals {
            self.app_state.show_help = false;
            self.app_state.show_file_browser = false;
            self.app_state.show_history = false;
            self.app_state.show_diff = false;
            self.app_state.show_settings = false;
        }

        Ok(())
    }

    fn handle_editor_key(&mut self, key: KeyEvent) -> Hatch<()> {
        let action = KeyBindings::handle(key);
        match action {
            Action::Quit => {
                if self.app_state.file_state.is_modified {
                    self.app_state.show_confirmation = true;
                    self.app_state.confirmation_action = ConfirmationAction::Quit;
                } else {
                    self.app_state.quit();
                }
            }
            Action::ToggleMode => {
                self.app_state.toggle_mode();
                self.perform_conversion();
            }
            Action::SwitchPanel => self.app_state.editor.toggle_active(),
            Action::OpenFile => self.app_state.show_file_browser = true,
            Action::SaveFile => self.save_output()?,
            Action::NewFile => self.new_file(),
            Action::Refresh => self.perform_conversion(),
            Action::ToggleSettings => self.app_state.show_settings = !self.app_state.show_settings,
            Action::ToggleHelp => self.app_state.show_help = !self.app_state.show_help,
            Action::ToggleFileBrowser => {
                self.app_state.show_file_browser = !self.app_state.show_file_browser
            }
            Action::ToggleHistory => self.app_state.show_history = !self.app_state.show_history,
            Action::ToggleDiff => self.app_state.show_diff = !self.app_state.show_diff,
            Action::ToggleTheme => self.app_state.toggle_theme(),
            Action::CopyOutput => self.copy_to_clipboard()?,
            Action::OpenRepl => self.app_state.repl.activate(),
            Action::CopySelection => self.copy_selection_to_clipboard()?,
            Action::PasteInput => self.paste_from_clipboard()?,
            Action::RoundTrip => self.perform_round_trip()?,
            Action::ClearInput => {
                self.app_state.editor.clear_input();
                self.app_state.editor.clear_output();
                self.app_state.stats = None;
            }
            Action::None => {
                let mut needs_reconversion = false;
                if self.app_state.editor.is_input_active() {
                    let before_text = self.app_state.editor.get_input_as_string();
                    self.app_state.editor.input.input(key);
                    let after_text = self.app_state.editor.get_input_as_string();
                    if before_text != after_text {
                        self.app_state.file_state.mark_modified();
                        needs_reconversion = true;
                    }
                } else {
                    self.app_state.editor.output.input(key);
                }
                if needs_reconversion {
                    self.perform_conversion();
                }
            }
        }
        Ok(())
    }

    fn handle_file_browser_key(&mut self, key: KeyEvent) -> Hatch<()> {
        match key.code {
            KeyCode::Up => self.app_state.file_browser.move_up(),
            KeyCode::Down => self.app_state.file_browser.move_down(&mut self.app_state),
            KeyCode::Enter => self.handle_file_selection()?,
            KeyCode::Char(' ') => self.handle_file_toggle_selection()?,
            _ => {}
        }
        Ok(())
    }

    fn handle_settings_key(&mut self, key: KeyEvent) -> Hatch<()> {
        let mut needs_reconversion = true;
        match key.code {
            KeyCode::Char('d') => self.app_state.cycle_delimiter(),
            KeyCode::Char('+') | KeyCode::Char('=') => self.app_state.increase_indent(),
            KeyCode::Char('-') | KeyCode::Char('_') => self.app_state.decrease_indent(),
            KeyCode::Char('f') => self.app_state.toggle_fold_keys(),
            KeyCode::Char('p') => self.app_state.toggle_expand_paths(),
            KeyCode::Char('s') => self.app_state.toggle_strict(),
            KeyCode::Char('c') => self.app_state.toggle_coerce_types(),
            KeyCode::Char('[') | KeyCode::Char('{') => self.app_state.decrease_flatten_depth(),
            KeyCode::Char(']') | KeyCode::Char('}') => self.app_state.increase_flatten_depth(),
            KeyCode::Char('u') => self.app_state.set_flatten_depth_unlimited(),
            _ => needs_reconversion = false,
        }
        if needs_reconversion {
            self.perform_conversion();
        }
        Ok(())
    }

    fn handle_confirmation_key(&mut self, key: KeyEvent) -> Hatch<()> {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                match self.app_state.confirmation_action {
                    ConfirmationAction::NewFile => {
                        self.app_state.editor.clear_input();
                        self.app_state.editor.clear_output();
                        self.app_state.file_state.clear_current_file();
                        self.app_state.set_status("New file created");
                    }
                    ConfirmationAction::Quit => self.app_state.quit(),
                    ConfirmationAction::DeleteFile => {
                        if let Some(file) = self.app_state.file_state.current_file.clone() {
                            if let Err(e) = fs::remove_file(&file) {
                                self.app_state.set_error(format!("Delete failed: {e}"));
                            } else {
                                self.app_state.set_status("File deleted");
                                self.app_state.file_state.clear_current_file();
                                self.app_state.editor.clear_input();
                                self.app_state.editor.clear_output();
                            }
                        }
                    }
                    ConfirmationAction::None => {}
                }
                self.app_state.show_confirmation = false;
                self.app_state.confirmation_action = ConfirmationAction::None;
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                self.app_state.show_confirmation = false;
                self.app_state.confirmation_action = ConfirmationAction::None;
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_repl_key(&mut self, key: KeyEvent) -> Hatch<()> {
        match key.code {
            KeyCode::Esc => self.app_state.repl.deactivate(),
            KeyCode::Char('r') if key.modifiers == KeyModifiers::CONTROL => {
                self.app_state.repl.deactivate()
            }
            KeyCode::Enter => {
                let cmd_input = self.app_state.repl.input.clone();
                if !cmd_input.trim().is_empty() {
                    self.app_state.repl.add_prompt(&cmd_input);
                    self.app_state.repl.add_to_history(cmd_input.clone());
                    if let Err(e) = self.execute_repl_command(&cmd_input) {
                        self.app_state.repl.add_error(e.to_string());
                    }
                    self.app_state.repl.input.clear();
                    self.app_state.repl.cursor_position = 0;
                    let view_height = 30; // A reasonable estimate
                    self.app_state.repl.scroll_to_bottom(view_height);
                }
            }
            KeyCode::Up => self.app_state.repl.history_up(),
            KeyCode::Down => self.app_state.repl.history_down(),
            KeyCode::PageUp => self.app_state.repl.scroll_up(),
            KeyCode::PageDown => self.app_state.repl.scroll_down(20), // Estimate
            KeyCode::Char(c) => {
                self.app_state.repl.input.insert(self.app_state.repl.cursor_position, c);
                self.app_state.repl.cursor_position += 1;
            }
            KeyCode::Backspace => {
                if self.app_state.repl.cursor_position > 0 {
                    self.app_state.repl.cursor_position -= 1;
                    self.app_state.repl.input.remove(self.app_state.repl.cursor_position);
                }
            }
            KeyCode::Left => {
                self.app_state.repl.cursor_position = self.app_state.repl.cursor_position.saturating_sub(1);
            }
            KeyCode::Right => {
                if self.app_state.repl.cursor_position < self.app_state.repl.input.len() {
                    self.app_state.repl.cursor_position += 1;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// TuiApp impl block for application logic.
impl<'a> TuiApp<'a> {
    fn perform_conversion(&mut self) {
        // Use the zero-copy method for the check, but get the owned string for processing.
        if self.app_state.editor.get_input().iter().all(|line| line.trim().is_empty()) {
            self.app_state.editor.clear_output();
            self.app_state.stats = None;
            self.app_state.clear_messages();
            return;
        }
        let input = self.app_state.editor.get_input_as_string();

        self.app_state.clear_messages();
        match self.app_state.mode {
            Mode::Encode => self.encode_input(&input),
            Mode::Decode => self.decode_input(&input),
            Mode::Rune => self.parse_rune_input(&input),
        }
    }

    fn encode_input(&mut self, input: &str) {
        match serde_json::from_str::<serde_json::Value>(input) {
            Ok(json_value) => match encode(&json_value, &self.app_state.encode_options) {
                Ok(rune_str) => {
                    self.app_state.editor.set_output(rune_str.clone()); // set_output is efficient
                    if let Ok(bpe) = cl100k_base() {
                        let json_tokens = bpe.encode_with_special_tokens(input).len();
                        let rune_tokens = bpe.encode_with_special_tokens(&rune_str).len();
                        let stats = Self::calculate_stats(json_tokens, rune_tokens, input.len(), rune_str.len());
                        self.add_conversion_to_history(stats.clone());
                        self.app_state.stats = Some(stats);
                    }
                }
                Err(e) => self.app_state.set_error(format!("Encode error: {e:?}")),
            },
            Err(e) => self.app_state.set_error(format!("Invalid JSON: {e}")),
        }
    }

    fn decode_input(&mut self, input: &str) {
        match decode::<serde_json::Value>(input, &self.app_state.decode_options) {
            Ok(json_value) => match serde_json::to_string_pretty(&json_value) {
                Ok(json_str) => {
                    self.app_state.editor.set_output(json_str.clone()); // set_output is efficient
                    if let Ok(bpe) = cl100k_base() {
                        let rune_tokens = bpe.encode_with_special_tokens(input).len();
                        let json_tokens = bpe.encode_with_special_tokens(&json_str).len();
                        let stats = Self::calculate_stats(json_tokens, rune_tokens, json_str.len(), input.len());
                        self.add_conversion_to_history(stats.clone());
                        self.app_state.stats = Some(stats);
                    }
                }
                Err(e) => self.app_state.set_error(format!("JSON serialization error: {e}")),
            },
            Err(e) => self.app_state.set_error(format!("Decode error: {e:?}")),
        }
    }

    fn parse_rune_input(&mut self, input: &str) {
        match crate::rune::parse_rune(input) {
            Ok(statements) => {
                let mut output = String::with_capacity(input.len());
                for stmt in &statements {
                    let _ = writeln!(output, "{stmt}");
                }
                self.app_state.editor.set_output(output);
            }
            Err(e) => self.app_state.set_error(format!("RUNE parse error: {e:?}")),
        }
    }
    
    fn calculate_stats(json_tokens: usize, rune_tokens: usize, json_bytes: usize, rune_bytes: usize) -> ConversionStats {
        let token_savings = 100.0 * (1.0 - (rune_tokens as f64 / json_tokens.max(1) as f64));
        let byte_savings = 100.0 * (1.0 - (rune_bytes as f64 / json_bytes.max(1) as f64));
        ConversionStats { json_tokens, toon_tokens: rune_tokens, json_bytes, toon_bytes: rune_bytes, token_savings, byte_savings }
    }

    fn add_conversion_to_history(&mut self, stats: ConversionStats) {
        self.app_state.file_state.add_to_history(ConversionHistory {
            timestamp: Local::now(),
            mode: self.app_state.mode,
            input_file: self.app_state.file_state.current_file.clone(),
            output_file: None,
            token_savings: stats.token_savings,
            byte_savings: stats.byte_savings,
        });
    }

    fn save_output(&mut self) -> Hatch<()> {
        let output = self.app_state.editor.get_output_as_string();
        if output.trim().is_empty() {
            self.app_state.set_error("Nothing to save");
            return Ok(());
        }
        let extension = match self.app_state.mode {
            Mode::Encode => "rune",
            Mode::Decode => "json",
            Mode::Rune => "txt",
        };
        let path = self.app_state.file_state.current_file.as_ref().map_or_else(
            || Path::new("output").with_extension(extension),
            |p| p.with_extension(extension),
        );
        fs::write(&path, output).context("Failed to save file")?;
        self.app_state.set_status(format!("Saved to {}", path.display()));
        self.app_state.file_state.is_modified = false;
        Ok(())
    }

    fn new_file(&mut self) {
        if self.app_state.file_state.is_modified {
            self.app_state.show_confirmation = true;
            self.app_state.confirmation_action = ConfirmationAction::NewFile;
            return;
        }
        self.app_state.editor.clear_input();
        self.app_state.editor.clear_output();
        self.app_state.file_state.clear_current_file();
        self.app_state.stats = None;
        self.app_state.set_status("New file created");
    }

    fn copy_to_clipboard(&mut self) -> Hatch<()> {
        let output = self.app_state.editor.get_output_as_string();
        if output.trim().is_empty() {
            self.app_state.set_error("Nothing to copy");
            return Ok(());
        }
        #[cfg(not(target_os = "unknown"))] {
            use arboard::Clipboard;
            let mut clipboard = Clipboard::new().map_err(|e| yoshi!("Clipboard error: {e}"))?;
            clipboard.set_text(output).map_err(|e| yoshi!("Clipboard error: {e}"))?;
            self.app_state.set_status("Copied to clipboard");
        }
        #[cfg(target_os = "unknown")] {
            self.app_state.set_error("Clipboard not supported on this platform");
        }
        Ok(())
    }

    fn paste_from_clipboard(&mut self) -> Hatch<()> {
        #[cfg(not(target_os = "unknown"))] {
            use arboard::Clipboard;
            let mut clipboard = Clipboard::new().map_err(|e| yoshi!("Clipboard error: {e}"))?;
            let text = clipboard.get_text().map_err(|e| yoshi!("Clipboard error: {e}"))?;
            self.app_state.editor.set_input(text);
            self.app_state.file_state.mark_modified();
            self.perform_conversion();
            self.app_state.set_status("Pasted from clipboard");
        }
        #[cfg(target_os = "unknown")] {
            self.app_state.set_error("Clipboard not supported on this platform");
        }
        Ok(())
    }
    
    fn handle_file_selection(&mut self) -> Hatch<()> {
        if let Some(selected_path) = self.app_state.file_browser.get_selected_entry(&mut self.app_state) {
            if selected_path.is_dir() {
                self.app_state.file_state.current_dir = selected_path;
            } else if selected_path.is_file() {
                match fs::read_to_string(&selected_path) {
                    Ok(content) => {
                        if let Some(ext) = selected_path.extension().and_then(|e| e.to_str()) {
                            self.app_state.mode = match ext {
                                "json" => Mode::Encode, "rune" => Mode::Rune, _ => Mode::Decode
                            };
                        }
                        self.app_state.editor.set_input(content);
                        self.app_state.file_state.set_current_file(selected_path.clone());
                        self.perform_conversion();
                        self.app_state.show_file_browser = false;
                        self.app_state.set_status(format!("Opened {}", selected_path.display()));
                    }
                    Err(e) => self.app_state.set_error(format!("Failed to read file: {e}")),
                }
            }
        }
        Ok(())
    }
    
    fn handle_file_toggle_selection(&mut self) -> Hatch<()> {
        if let Some(selected_path) = self.app_state.file_browser.get_selected_entry(&mut self.app_state) {
            if selected_path.is_file() {
                self.app_state.file_state.toggle_file_selection(&selected_path);
                let action = if self.app_state.file_state.is_selected(&selected_path) { "Selected" } else { "Deselected" };
                self.app_state.set_status(format!("{action} {}", selected_path.display()));
            }
        }
        Ok(())
    }
    
    fn copy_selection_to_clipboard(&mut self) -> Hatch<()> {
        let text = if self.app_state.editor.is_input_active() {
            self.app_state.editor.input.yank_text()
        } else {
            self.app_state.editor.output.yank_text()
        };
        if text.is_empty() {
            self.app_state.set_error("Nothing to copy");
            return Ok(());
        }
        #[cfg(not(target_os = "unknown"))] {
            use arboard::Clipboard;
            let mut clipboard = Clipboard::new().map_err(|e| yoshi!("Clipboard error: {e}"))?;
            clipboard.set_text(text).map_err(|e| yoshi!("Clipboard error: {e}"))?;
            self.app_state.set_status("Copied selection to clipboard");
        }
        #[cfg(target_os = "unknown")] {
            self.app_state.set_error("Clipboard not supported on this platform");
        }
        Ok(())
    }
    
    fn perform_round_trip(&mut self) -> Hatch<()> {
        let output = self.app_state.editor.get_output_as_string();
        if output.trim().is_empty() {
            self.app_state.set_error("No output to round-trip test. Convert something first!");
            return Ok(());
        }
        let original_input = self.app_state.editor.get_input_as_string();
        self.app_state.editor.set_input(output);
        self.app_state.toggle_mode();
        self.perform_conversion();
        let roundtrip_output = self.app_state.editor.get_output_as_string();
        if roundtrip_output.trim().is_empty() {
            self.app_state.set_error("Round-trip failed! Conversion produced no output. Check for errors.");
            return Ok(());
        }
        if self.compare_data(&original_input, &roundtrip_output) {
            self.app_state.set_status("✓ Round-trip successful! Output matches original.");
        } else {
            self.app_state.set_error(format!(
                "⚠ Round-trip mismatch! Original had {} chars, round-trip has {} chars.",
                original_input.len(), roundtrip_output.len()
            ));
        }
        Ok(())
    }
    
    fn compare_data(&self, original: &str, roundtrip: &str) -> bool {
        if let (Ok(orig_json), Ok(rt_json)) = (
            serde_json::from_str::<serde_json::Value>(original),
            serde_json::from_str::<serde_json::Value>(roundtrip),
        ) {
            return orig_json == rt_json;
        }
        // Fallback to whitespace-normalized string comparison
        original.split_whitespace().eq(roundtrip.split_whitespace())
    }

    fn execute_repl_command(&mut self, input: &str) -> Hatch<()> {
        let cmd = ReplCommand::parse(input)?;
        let data = cmd.inline_data.as_deref().unwrap_or("");
        let mut substituted_data = self.substitute_variables(data);

        match cmd.name.as_str() {
            "encode" | "e" => {
                if substituted_data.is_empty() {
                    self.app_state.repl.add_error("Usage: encode {\"data\": true} or encode $var");
                    return Ok(());
                }
                match serde_json::from_str::<serde_json::Value>(&substituted_data) {
                    Ok(json) => match encode(&json, &self.app_state.encode_options) {
                        Ok(rune) => { self.app_state.repl.last_result = Some(rune.clone()); self.app_state.repl.add_success(rune); }
                        Err(e) => self.app_state.repl.add_error(format!("Encode error: {e:?}")),
                    },
                    Err(e) => self.app_state.repl.add_error(format!("Invalid JSON: {e}")),
                }
            }
            "decode" | "d" => {
                 if substituted_data.is_empty() {
                    self.app_state.repl.add_error("Usage: decode name: Alice or decode $var");
                    return Ok(());
                }
                match decode::<serde_json::Value>(&substituted_data, &self.app_state.decode_options) {
                    Ok(json) => match serde_json::to_string_pretty(&json) {
                        Ok(json_str) => { self.app_state.repl.last_result = Some(json_str.clone()); self.app_state.repl.add_success(json_str); }
                        Err(e) => self.app_state.repl.add_error(format!("JSON error: {e}")),
                    },
                    Err(e) => self.app_state.repl.add_error(format!("Decode error: {e:?}")),
                }
            }
            "let" => {
                 let parts: Vec<_> = input.splitn(2, '=').collect();
                 if parts.len() == 2 {
                     let var_part = parts[0].trim().strip_prefix("let").unwrap_or("").trim();
                     let data_part = parts[1].trim();
                     if !var_part.is_empty() && !data_part.is_empty() {
                         let var_name = var_part.strip_prefix('$').unwrap_or(var_part);
                         self.app_state.repl.variables.insert(var_name.to_string(), data_part.to_string());
                         self.app_state.repl.add_info(format!("Stored in ${var_name}"));
                         self.app_state.repl.last_result = Some(data_part.to_string());
                     } else { self.app_state.repl.add_error("Usage: let $var = data"); }
                 } else { self.app_state.repl.add_error("Usage: let $var = data"); }
            }
            "vars" => {
                if self.app_state.repl.variables.is_empty() {
                    self.app_state.repl.add_info("No variables defined");
                } else {
                    let mut vars: Vec<_> = self.app_state.repl.variables.keys().collect();
                    vars.sort();
                    for k in vars {
                        self.app_state.repl.add_info(format!("${k}"));
                    }
                }
            }
            "clear" => {
                self.app_state.repl.output.clear();
                self.app_state.repl.add_info("Cleared");
            }
            "help" | "h" => {
                 self.app_state.repl.add_info("📖 REPL Commands:");
                 self.app_state.repl.add_info("");
                 self.app_state.repl.add_info("  encode|e <JSON>      - Encode JSON to RUNE");
                 self.app_state.repl.add_info("  decode|d <RUNE>      - Decode RUNE to JSON");
                 self.app_state.repl.add_info("  rune|r <RUNE_SRC>    - Parse and evaluate RUNE");
                 self.app_state.repl.add_info("  let $var = <data>    - Store data in a variable");
                 self.app_state.repl.add_info("  vars                 - List all variables");
                 self.app_state.repl.add_info("  vars                 - List all variables");
                 self.app_state.repl.add_info("  clear                - Clear session output");
                 self.app_state.repl.add_info("  help|h               - Show this help message");
                 self.app_state.repl.add_info("  exit|quit|q          - Close the REPL session");
                 self.app_state.repl.add_info("");
                 self.app_state.repl.add_info("  Use $_ for the last result, e.g., 'let $v = $_'");
                 self.app_state.repl.add_info("  Press ↑/↓ for history, Esc or Ctrl+R to close");
            }
            "exit" | "quit" | "q" => {
                self.app_state.repl.add_info("Closing REPL...");
                self.app_state.repl.deactivate();
            }
            _ => {
                self.app_state.repl.add_error(format!("Unknown command: '{}'. Type 'help'.", cmd.name));
            }
        }
        Ok(())
    }

    /// Replace $var and $_ with their stored values.
    fn substitute_variables(&self, text: &str) -> String {
        // This is a simple substitution. A more complex regex-based approach might be
        // more robust but would be slower and require another dependency. Given the
        // context of a simple REPL, this is a reasonable trade-off.
        if !text.contains('$') {
            return text.to_string();
        }

        let mut result = text.to_string();

        // $_ is a special variable for the last result.
        if let Some(last) = &self.app_state.repl.last_result {
            result = result.replace("$_", last);
        }

        for (var_name, var_value) in &self.app_state.repl.variables {
            let pattern = format!("${var_name}");
            // Simple replace; doesn't handle word boundaries but is sufficient for now.
            result = result.replace(&pattern, var_value);
        }

        result
    }

    #[cfg(feature = "hydron")]
    fn format_rune_value(&self, value: &Value) -> String {
        let mut s = String::with_capacity(128);
        self.write_rune_value(&mut s, value)
            .unwrap_or_else(|_| "Formatting error".to_string());
        s
    }

    #[cfg(feature = "hydron")]
    fn write_rune_value(&self, mut f: &mut (dyn Write), value: &Value) -> std::fmt::Result {
        match value {
            Value::Bool(b) => write!(f, "{b}"),
            Value::Scalar(s) => write!(f, "{s}"),
            Value::Float(f) => write!(f, "{f}"),
            Value::String(s) => write!(f, "{s}"),
            Value::Vec8(v) => {
                write!(f, "Vec8(")?;
                for (i, x) in v.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{:.4}", x)?;
                }
                write!(f, ")")
            }
            Value::Vec16(v) => {
                write!(f, "Vec16(")?;
                for (i, x) in v.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{:.4}", x)?;
                }
                write!(f, ")")
            }
            Value::Array(items) | Value::Tuple(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    self.write_rune_value(&mut f, item)?;
                }
                write!(f, "]")
            }
            Value::Struct(name, items) => {
                write!(f, "{name}(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    self.write_rune_value(&mut f, item)?;
                }
                write!(f, ")")
            }
            Value::Map(map) => {
                write!(f, "{{")?;
                let mut first = true;
                for (k, v) in map.iter() {
                    if !first { write!(f, ", ")?; }
                    write!(f, "{k}: ")?;
                    self.write_rune_value(&mut f, v)?;
                    first = false;
                }
                write!(f, "}}")
            }
            _ => write!(f, "{value:?}"), // Fallback for other types
        }
    }
}

impl<'a> Default for TuiApp<'a> {
    fn default() -> Self {
        Self::new()
    }
}