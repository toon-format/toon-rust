use std::{
    fs,
    path::PathBuf,
    time::Duration,
};

use anyhow::{
    Context,
    Result,
};
use chrono::Local;
use crossterm::event::{
    KeyCode,
    KeyEvent,
};
use tiktoken_rs::cl100k_base;

use crate::{
    decode,
    encode,
    tui::{
        components::FileBrowser,
        events::{
            Event,
            EventHandler,
        },
        keybindings::{
            Action,
            KeyBindings,
        },
        repl_command::ReplCommand,
        state::{
            app_state::ConversionStats,
            AppState,
            ConversionHistory,
        },
        ui,
    },
};

/// Main TUI application managing state, events, and rendering.
pub struct TuiApp<'a> {
    pub app_state: AppState<'a>,
    pub file_browser: FileBrowser,
}

impl<'a> TuiApp<'a> {
    pub fn new() -> Self {
        Self {
            app_state: AppState::new(),
            file_browser: FileBrowser::new(),
        }
    }

    pub fn run<B: ratatui::backend::Backend>(
        &mut self,
        terminal: &mut ratatui::Terminal<B>,
    ) -> Result<()> {
        loop {
            terminal.draw(|f| ui::render(f, &mut self.app_state, &mut self.file_browser))?;

            if let Some(event) = EventHandler::poll(Duration::from_millis(100))? {
                self.handle_event(event)?;
            }

            if self.app_state.should_quit {
                break;
            }
        }
        Ok(())
    }

    fn handle_event(&mut self, event: Event) -> Result<()> {
        match event {
            Event::Key(key) => self.handle_key_event(key)?,
            Event::Resize => {}
            Event::Tick => {}
        }
        Ok(())
    }

    fn handle_key_event(&mut self, key: KeyEvent) -> Result<()> {
        // REPL takes priority when active
        if self.app_state.repl.active {
            return self.handle_repl_key(key);
        }

        // Handle overlay panels (help, file browser, settings, etc.)
        if self.app_state.show_help
            || self.app_state.show_file_browser
            || self.app_state.show_history
            || self.app_state.show_diff
            || self.app_state.show_settings
        {
            match key.code {
                KeyCode::Esc => {
                    self.app_state.show_help = false;
                    self.app_state.show_file_browser = false;
                    self.app_state.show_history = false;
                    self.app_state.show_diff = false;
                    self.app_state.show_settings = false;
                    return Ok(());
                }
                KeyCode::F(1) if self.app_state.show_help => {
                    self.app_state.show_help = false;
                    return Ok(());
                }
                _ => {}
            }

            if self.app_state.show_file_browser {
                match key.code {
                    KeyCode::Up => {
                        self.file_browser.move_up();
                        return Ok(());
                    }
                    KeyCode::Down => {
                        let count = self
                            .file_browser
                            .get_entry_count(&self.app_state.file_state.current_dir);
                        self.file_browser.move_down(count);
                        return Ok(());
                    }
                    KeyCode::Enter => {
                        self.handle_file_selection()?;
                        return Ok(());
                    }
                    KeyCode::Char(' ') => {
                        self.handle_file_toggle_selection()?;
                        return Ok(());
                    }
                    _ => {}
                }
            }

            if self.app_state.show_settings {
                match key.code {
                    KeyCode::Esc => {
                        self.app_state.show_settings = false;
                        return Ok(());
                    }
                    KeyCode::Char('d') => {
                        self.app_state.cycle_delimiter();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        self.app_state.increase_indent();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('-') | KeyCode::Char('_') => {
                        self.app_state.decrease_indent();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('f') => {
                        self.app_state.toggle_fold_keys();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('p') => {
                        self.app_state.toggle_expand_paths();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('s') => {
                        self.app_state.toggle_strict();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('c') => {
                        self.app_state.toggle_coerce_types();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('[') | KeyCode::Char('{') => {
                        self.app_state.decrease_flatten_depth();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char(']') | KeyCode::Char('}') => {
                        self.app_state.increase_flatten_depth();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('u') => {
                        self.app_state.toggle_flatten_depth();
                        self.perform_conversion();
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }

        let action = KeyBindings::handle(key);
        match action {
            Action::Quit => self.app_state.quit(),
            Action::ToggleMode => {
                self.app_state.toggle_mode();
                self.perform_conversion();
            }
            Action::SwitchPanel => {
                self.app_state.editor.toggle_active();
            }
            Action::OpenFile => {
                self.open_file_dialog()?;
            }
            Action::SaveFile => {
                self.save_output()?;
            }
            Action::NewFile => {
                self.new_file();
            }
            Action::Refresh => {
                self.perform_conversion();
            }
            Action::ToggleSettings => {
                self.app_state.toggle_settings();
            }
            Action::ToggleHelp => {
                self.app_state.toggle_help();
            }
            Action::ToggleFileBrowser => {
                self.app_state.toggle_file_browser();
            }
            Action::ToggleHistory => {
                self.app_state.toggle_history();
            }
            Action::ToggleDiff => {
                self.app_state.toggle_diff();
            }
            Action::ToggleTheme => {
                self.app_state.toggle_theme();
            }
            Action::CopyOutput => {
                self.copy_to_clipboard()?;
            }
            Action::OpenRepl => {
                self.app_state.repl.activate();
            }
            Action::CopySelection => {
                self.copy_selection_to_clipboard()?;
            }
            Action::PasteInput => {
                self.paste_from_clipboard()?;
            }
            Action::RoundTrip => {
                self.perform_round_trip()?;
            }
            Action::ClearInput => {
                self.app_state.editor.clear_input();
                self.app_state.editor.clear_output();
                self.app_state.stats = None;
            }
            Action::None => {
                if self.app_state.editor.is_input_active() {
                    self.app_state.editor.input.input(key);
                    self.app_state.file_state.mark_modified();
                    self.perform_conversion();
                } else if self.app_state.editor.is_output_active() {
                    // Output is read-only, only allow navigation
                    match key.code {
                        KeyCode::Up
                        | KeyCode::Down
                        | KeyCode::Left
                        | KeyCode::Right
                        | KeyCode::PageUp
                        | KeyCode::PageDown
                        | KeyCode::Home
                        | KeyCode::End => {
                            self.app_state.editor.output.input(key);
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }

    /// Convert input based on current mode (encode/decode).
    fn perform_conversion(&mut self) {
        let input = self.app_state.editor.get_input();
        if input.trim().is_empty() {
            self.app_state.editor.clear_output();
            self.app_state.stats = None;
            self.app_state.clear_error();
            return;
        }

        self.app_state.clear_error();

        match self.app_state.mode {
            crate::tui::state::app_state::Mode::Encode => {
                self.encode_input(&input);
            }
            crate::tui::state::app_state::Mode::Decode => {
                self.decode_input(&input);
            }
        }
    }

    fn encode_input(&mut self, input: &str) {
        self.app_state.editor.clear_output();

        match serde_json::from_str::<serde_json::Value>(input) {
            Ok(json_value) => match encode(&json_value, &self.app_state.encode_options) {
                Ok(toon_str) => {
                    self.app_state.editor.set_output(toon_str.clone());
                    self.app_state.clear_error();

                    if let Ok(bpe) = cl100k_base() {
                        let json_tokens = bpe.encode_with_special_tokens(input).len();
                        let toon_tokens = bpe.encode_with_special_tokens(&toon_str).len();
                        let json_bytes = input.len();
                        let toon_bytes = toon_str.len();

                        let token_savings =
                            100.0 * (1.0 - (toon_tokens as f64 / json_tokens as f64));
                        let byte_savings = 100.0 * (1.0 - (toon_bytes as f64 / json_bytes as f64));

                        self.app_state.stats = Some(ConversionStats {
                            json_tokens,
                            toon_tokens,
                            json_bytes,
                            toon_bytes,
                            token_savings,
                            byte_savings,
                        });

                        self.app_state.file_state.add_to_history(ConversionHistory {
                            timestamp: Local::now(),
                            mode: "Encode".to_string(),
                            input_file: self.app_state.file_state.current_file.clone(),
                            output_file: None,
                            token_savings,
                            byte_savings,
                        });
                    }
                }
                Err(e) => {
                    self.app_state.set_error(format!("Encode error: {e}"));
                }
            },
            Err(e) => {
                self.app_state.set_error(format!("Invalid JSON: {e}"));
            }
        }
    }

    fn decode_input(&mut self, input: &str) {
        self.app_state.editor.clear_output();

        match decode::<serde_json::Value>(input, &self.app_state.decode_options) {
            Ok(json_value) => match serde_json::to_string_pretty(&json_value) {
                Ok(json_str) => {
                    self.app_state.editor.set_output(json_str.clone());
                    self.app_state.clear_error();

                    if let Ok(bpe) = cl100k_base() {
                        let toon_tokens = bpe.encode_with_special_tokens(input).len();
                        let json_tokens = bpe.encode_with_special_tokens(&json_str).len();
                        let toon_bytes = input.len();
                        let json_bytes = json_str.len();

                        let token_savings =
                            100.0 * (1.0 - (toon_tokens as f64 / json_tokens as f64));
                        let byte_savings = 100.0 * (1.0 - (toon_bytes as f64 / json_bytes as f64));

                        self.app_state.stats = Some(ConversionStats {
                            json_tokens,
                            toon_tokens,
                            json_bytes,
                            toon_bytes,
                            token_savings,
                            byte_savings,
                        });

                        self.app_state.file_state.add_to_history(ConversionHistory {
                            timestamp: Local::now(),
                            mode: "Decode".to_string(),
                            input_file: self.app_state.file_state.current_file.clone(),
                            output_file: None,
                            token_savings,
                            byte_savings,
                        });
                    }
                }
                Err(e) => {
                    self.app_state
                        .set_error(format!("JSON serialization error: {e}"));
                }
            },
            Err(e) => {
                self.app_state.set_error(format!("Decode error: {e}"));
            }
        }
    }

    fn open_file_dialog(&mut self) -> Result<()> {
        self.app_state.toggle_file_browser();
        Ok(())
    }

    fn save_output(&mut self) -> Result<()> {
        let output = self.app_state.editor.get_output();
        if output.trim().is_empty() {
            self.app_state.set_error("Nothing to save".to_string());
            return Ok(());
        }

        let extension = match self.app_state.mode {
            crate::tui::state::app_state::Mode::Encode => "toon",
            crate::tui::state::app_state::Mode::Decode => "json",
        };

        let path = if let Some(current) = &self.app_state.file_state.current_file {
            current.with_extension(extension)
        } else {
            PathBuf::from(format!("output.{extension}"))
        };

        fs::write(&path, output).context("Failed to save file")?;
        self.app_state
            .set_status(format!("Saved to {}", path.display()));
        self.app_state.file_state.is_modified = false;

        Ok(())
    }

    fn new_file(&mut self) {
        if self.app_state.file_state.is_modified {
            // TODO: confirmation dialog
        }
        self.app_state.editor.clear_input();
        self.app_state.editor.clear_output();
        self.app_state.file_state.clear_current_file();
        self.app_state.stats = None;
        self.app_state.set_status("New file created".to_string());
    }

    fn copy_to_clipboard(&mut self) -> Result<()> {
        let output = self.app_state.editor.get_output();
        if output.trim().is_empty() {
            self.app_state.set_error("Nothing to copy".to_string());
            return Ok(());
        }

        #[cfg(not(target_os = "unknown"))]
        {
            use arboard::Clipboard;
            let mut clipboard = Clipboard::new()?;
            clipboard.set_text(output)?;
            self.app_state.set_status("Copied to clipboard".to_string());
        }

        #[cfg(target_os = "unknown")]
        {
            self.app_state
                .set_error("Clipboard not supported on this platform".to_string());
        }

        Ok(())
    }

    fn paste_from_clipboard(&mut self) -> Result<()> {
        #[cfg(not(target_os = "unknown"))]
        {
            use arboard::Clipboard;
            let mut clipboard = Clipboard::new()?;
            let text = clipboard.get_text()?;
            self.app_state.editor.set_input(text);
            self.app_state.file_state.mark_modified();
            self.perform_conversion();
            self.app_state
                .set_status("Pasted from clipboard".to_string());
        }

        #[cfg(target_os = "unknown")]
        {
            self.app_state
                .set_error("Clipboard not supported on this platform".to_string());
        }

        Ok(())
    }

    fn handle_file_selection(&mut self) -> Result<()> {
        let current_dir = self.app_state.file_state.current_dir.clone();
        if let Some(selected_path) = self.file_browser.get_selected_entry(&current_dir) {
            if selected_path.is_dir() {
                // Navigate into directory
                self.app_state.file_state.current_dir = selected_path;
                self.file_browser.selected_index = 0;
                self.app_state.set_status(format!(
                    "Navigated to {}",
                    self.app_state.file_state.current_dir.display()
                ));
            } else if selected_path.is_file() {
                // Open file
                match fs::read_to_string(&selected_path) {
                    Ok(content) => {
                        self.app_state.editor.set_input(content);
                        self.app_state
                            .file_state
                            .set_current_file(selected_path.clone());

                        // Auto-detect mode based on extension
                        if let Some(ext) = selected_path.extension().and_then(|e| e.to_str()) {
                            match ext {
                                "json" => {
                                    self.app_state.mode =
                                        crate::tui::state::app_state::Mode::Encode;
                                }
                                "toon" => {
                                    self.app_state.mode =
                                        crate::tui::state::app_state::Mode::Decode;
                                }
                                _ => {}
                            }
                        }

                        self.perform_conversion();
                        self.app_state.show_file_browser = false;
                        self.app_state
                            .set_status(format!("Opened {}", selected_path.display()));
                    }
                    Err(e) => {
                        self.app_state
                            .set_error(format!("Failed to read file: {e}"));
                    }
                }
            }
        }
        Ok(())
    }

    fn handle_file_toggle_selection(&mut self) -> Result<()> {
        let current_dir = self.app_state.file_state.current_dir.clone();
        if let Some(selected_path) = self.file_browser.get_selected_entry(&current_dir) {
            if selected_path.is_file() {
                self.app_state
                    .file_state
                    .toggle_file_selection(selected_path.clone());
                let is_selected = self.app_state.file_state.is_selected(&selected_path);
                let action = if is_selected {
                    "Selected"
                } else {
                    "Deselected"
                };
                self.app_state
                    .set_status(format!("{} {}", action, selected_path.display()));
            }
        }
        Ok(())
    }

    fn copy_selection_to_clipboard(&mut self) -> Result<()> {
        let text = if self.app_state.editor.is_input_active() {
            self.app_state.editor.input.yank_text()
        } else {
            self.app_state.editor.output.yank_text()
        };

        if text.is_empty() {
            self.app_state.set_error("Nothing to copy".to_string());
            return Ok(());
        }

        #[cfg(not(target_os = "unknown"))]
        {
            use arboard::Clipboard;
            let mut clipboard = Clipboard::new()?;
            clipboard.set_text(text)?;
            self.app_state
                .set_status("Copied selection to clipboard".to_string());
        }

        #[cfg(target_os = "unknown")]
        {
            self.app_state
                .set_error("Clipboard not supported on this platform".to_string());
        }

        Ok(())
    }

    /// Round-trip test: convert output back to input and verify.
    fn perform_round_trip(&mut self) -> Result<()> {
        let output = self.app_state.editor.get_output();
        if output.trim().is_empty() {
            self.app_state
                .set_error("No output to round-trip test. Convert something first!".to_string());
            return Ok(());
        }

        let original_input = self.app_state.editor.get_input();
        self.app_state.editor.set_input(output.clone());
        self.app_state.toggle_mode();
        self.perform_conversion();

        let roundtrip_output = self.app_state.editor.get_output();

        if roundtrip_output.trim().is_empty() {
            self.app_state.set_error(
                "Round-trip failed! Conversion produced no output. Check for errors.".to_string(),
            );
            return Ok(());
        }

        let matches = self.compare_data(&original_input, &roundtrip_output);

        if matches {
            self.app_state
                .set_status("✓ Round-trip successful! Output matches original.".to_string());
        } else {
            self.app_state.set_error(format!(
                "⚠ Round-trip mismatch! Original had {} chars, round-trip has {} chars.",
                original_input.len(),
                roundtrip_output.len()
            ));
        }

        Ok(())
    }

    /// Compare data semantically, trying JSON parse first.
    fn compare_data(&self, original: &str, roundtrip: &str) -> bool {
        // Try JSON comparison for accuracy
        if let (Ok(orig_json), Ok(rt_json)) = (
            serde_json::from_str::<serde_json::Value>(original),
            serde_json::from_str::<serde_json::Value>(roundtrip),
        ) {
            return orig_json == rt_json;
        }

        let original_normalized: String = original.split_whitespace().collect();
        let roundtrip_normalized: String = roundtrip.split_whitespace().collect();
        original_normalized == roundtrip_normalized
    }

    /// Handle keyboard input when REPL is active.
    fn handle_repl_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Esc => {
                self.app_state.repl.deactivate();
            }
            KeyCode::Char('r')
                if key
                    .modifiers
                    .contains(crossterm::event::KeyModifiers::CONTROL) =>
            {
                self.app_state.repl.deactivate();
            }
            KeyCode::Enter => {
                let cmd_input = self.app_state.repl.input.clone();
                if !cmd_input.trim().is_empty() {
                    self.app_state.repl.add_prompt(&cmd_input);
                    self.app_state.repl.add_to_history(cmd_input.clone());

                    if let Err(e) = self.execute_repl_command(&cmd_input) {
                        self.app_state.repl.add_error(format!("{e}"));
                    }

                    self.app_state.repl.input.clear();
                    self.app_state.repl.scroll_to_bottom();
                }
            }
            KeyCode::Up => {
                self.app_state.repl.history_up();
            }
            KeyCode::Down => {
                self.app_state.repl.history_down();
            }
            KeyCode::PageUp => {
                self.app_state.repl.scroll_up();
            }
            KeyCode::PageDown => {
                self.app_state.repl.scroll_down(20);
            }
            KeyCode::Char(c) => {
                self.app_state.repl.input.push(c);
            }
            KeyCode::Backspace => {
                self.app_state.repl.input.pop();
            }
            _ => {}
        }
        Ok(())
    }

    /// Execute parsed REPL command and update state.
    fn execute_repl_command(&mut self, input: &str) -> Result<()> {
        let cmd = ReplCommand::parse(input)?;

        match cmd.name.as_str() {
            "encode" | "e" => {
                let mut data = cmd
                    .inline_data
                    .as_ref()
                    .map(|s| s.to_string())
                    .unwrap_or_else(String::new);

                data = self.substitute_variables(&data);

                if data.is_empty() {
                    self.app_state
                        .repl
                        .add_error("Usage: encode {\"data\": true} or encode $var".to_string());
                    return Ok(());
                }

                match serde_json::from_str::<serde_json::Value>(&data) {
                    Ok(json_value) => match encode(&json_value, &self.app_state.encode_options) {
                        Ok(toon_str) => {
                            self.app_state.repl.add_success(toon_str.clone());
                            self.app_state.repl.last_result = Some(toon_str);
                        }
                        Err(e) => {
                            self.app_state.repl.add_error(format!("Encode error: {e}"));
                        }
                    },
                    Err(e) => {
                        self.app_state.repl.add_error(format!("Invalid JSON: {e}"));
                    }
                }
            }
            "decode" | "d" => {
                let mut data = cmd
                    .inline_data
                    .as_ref()
                    .map(|s| s.to_string())
                    .unwrap_or_else(String::new);

                data = self.substitute_variables(&data);

                if data.is_empty() {
                    self.app_state
                        .repl
                        .add_error("Usage: decode name: Alice or decode $var".to_string());
                    return Ok(());
                }

                match decode::<serde_json::Value>(&data, &self.app_state.decode_options) {
                    Ok(json_value) => match serde_json::to_string_pretty(&json_value) {
                        Ok(json_str) => {
                            self.app_state.repl.add_success(json_str.clone());
                            self.app_state.repl.last_result = Some(json_str);
                        }
                        Err(e) => {
                            self.app_state.repl.add_error(format!("JSON error: {e}"));
                        }
                    },
                    Err(e) => {
                        self.app_state.repl.add_error(format!("Decode error: {e}"));
                    }
                }
            }
            "let" => {
                let parts: Vec<&str> = input.splitn(2, '=').collect();
                if parts.len() == 2 {
                    let var_part = parts[0].trim().trim_start_matches("let").trim();
                    let data_part = parts[1].trim();

                    if !var_part.is_empty() && !data_part.is_empty() {
                        let var_name = var_part.trim_start_matches('$');
                        self.app_state
                            .repl
                            .variables
                            .insert(var_name.to_string(), data_part.to_string());
                        self.app_state
                            .repl
                            .add_info(format!("Stored in ${var_name}"));
                        self.app_state.repl.last_result = Some(data_part.to_string());
                    } else {
                        self.app_state
                            .repl
                            .add_error("Usage: let $var = {\"data\": true}".to_string());
                    }
                } else {
                    self.app_state
                        .repl
                        .add_error("Usage: let $var = {\"data\": true}".to_string());
                }
            }
            "vars" => {
                if self.app_state.repl.variables.is_empty() {
                    self.app_state
                        .repl
                        .add_info("No variables defined".to_string());
                } else {
                    let vars: Vec<String> = self
                        .app_state
                        .repl
                        .variables
                        .keys()
                        .map(|k| format!("${k}"))
                        .collect();
                    for var in vars {
                        self.app_state.repl.add_info(var);
                    }
                }
            }
            "clear" => {
                self.app_state.repl.output.clear();
                self.app_state
                    .repl
                    .output
                    .push(crate::tui::state::ReplLine {
                        kind: crate::tui::state::ReplLineKind::Info,
                        content: "Cleared".to_string(),
                    });
            }
            "help" | "h" => {
                self.app_state
                    .repl
                    .add_info("📖 REPL Commands:".to_string());
                self.app_state.repl.add_info("".to_string());
                self.app_state
                    .repl
                    .add_info("  encode {\"data\": true}  - Encode JSON to TOON".to_string());
                self.app_state
                    .repl
                    .add_info("  decode name: Alice      - Decode TOON to JSON".to_string());
                self.app_state
                    .repl
                    .add_info("  let $var = {...}        - Store data in variable".to_string());
                self.app_state
                    .repl
                    .add_info("  vars                    - List all variables".to_string());
                self.app_state
                    .repl
                    .add_info("  clear                   - Clear session".to_string());
                self.app_state
                    .repl
                    .add_info("  help                    - Show this help".to_string());
                self.app_state
                    .repl
                    .add_info("  exit                    - Close REPL".to_string());
                self.app_state.repl.add_info("".to_string());
                self.app_state
                    .repl
                    .add_info("Press ↑/↓ for history, Esc to close".to_string());
            }
            "exit" | "quit" | "q" => {
                self.app_state.repl.add_info("Closing REPL...".to_string());
                self.app_state.repl.deactivate();
            }
            _ => {
                self.app_state
                    .repl
                    .add_error(format!("Unknown command: {}. Type 'help'", cmd.name));
            }
        }

        Ok(())
    }

    /// Replace $var and $_ with their stored values.
    fn substitute_variables(&self, text: &str) -> String {
        let mut result = text.to_string();

        // $_ is the last result
        if let Some(last) = &self.app_state.repl.last_result {
            result = result.replace("$_", last);
        }

        // Variables are stored without $, add it for matching
        for (var_name, var_value) in &self.app_state.repl.variables {
            let pattern = format!("${var_name}");
            result = result.replace(&pattern, var_value);
        }

        result
    }
}

impl<'a> Default for TuiApp<'a> {
    fn default() -> Self {
        Self::new()
    }
}
