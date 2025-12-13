/* src/tui/ui.rs */
//!▫~•◦-------------------------------‣
//! # Main UI rendering orchestrator for the RUNE TUI.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module contains the primary `render` function that draws the entire
//! terminal user interface. It acts as a dispatcher, deciding which components
//! to render based on the current `AppState`.
//!
//! ## Key Capabilities
//! - **Component Orchestration**: Conditionally renders different UI components like the
//!   editor, REPL, help screen, file browser, etc., based on modal state.
//! - **Layout Management**: Defines the main application layout using `ratatui`.
//! - **Performance-Optimized**: The entire rendering pipeline, including all sub-components,
//!   is designed to be zero-copy or minimal-copy, ensuring a highly responsive UI
//!   with no allocations in the hot render path.
//!
//! ### Architectural Notes
//! The `render` function takes a mutable reference to `AppState` as the single
//! source of truth. It passes immutable references down to stateless rendering
//! components. All dynamic text in the header is generated on the stack using the
//! `itoa` crate and `Span` composition to avoid `format!`.
//!
//! #### Example
//! ```rust
//! // In the main application loop (app.rs):
//! // terminal.draw(|f| ui::render(f, &mut self.app_state))?;
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use super::{
    components::{
        ConfirmationDialog, DiffViewer, EditorComponent, FileBrowser, HelpScreen, HistoryPanel,
        ReplPanel, SettingsPanel, StatsBar, StatusBar,
    },
    state::AppState,
};
use crate::types::{Delimiter, Indent, KeyFoldingMode, PathExpansionMode};

/// The main render function, which orchestrates all UI components.
pub fn render(f: &mut Frame, app: &mut AppState) {
    let theme = &app.theme;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3), // StatsBar
            Constraint::Length(1), // StatusBar
        ])
        .split(f.size());

    render_header(f, chunks[0], app);

    // Render the main content area based on the current modal view.
    // The order determines the rendering priority.
    if app.repl.active {
        // REPL takes the full main area.
        ReplPanel::render(f, chunks[1], app);
    } else if app.show_help {
        HelpScreen::render(f, chunks[1], theme);
    } else if app.show_file_browser {
        app.file_browser.render(f, chunks[1], app, theme);
    } else if app.show_history {
        HistoryPanel::render(f, chunks[1], app, theme);
    } else if app.show_diff {
        DiffViewer::render(f, chunks[1], app, theme);
    } else if app.show_settings {
        SettingsPanel::render(f, chunks[1], app, theme);
    } else {
        // Default view: Editor panels.
        let editor_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(48),
                Constraint::Percentage(4),
                Constraint::Percentage(48),
            ])
            .split(chunks[1]);

        EditorComponent::render(f, editor_chunks[0], editor_chunks[2], app, theme);
        render_arrow(f, editor_chunks[1], app);
    }

    StatsBar::render(f, chunks[2], app, theme);
    StatusBar::render(f, chunks[3], app, theme);

    // Render confirmation dialog on top of everything if active.
    if app.show_confirmation {
        ConfirmationDialog::render(f, f.size(), app.confirmation_action);
    }
}

/// Renders the conversion arrow and round-trip button between editor panels.
fn render_arrow(f: &mut Frame, area: Rect, app: &AppState) {
    let theme = &app.theme;
    let arrow_symbol = match app.mode {
        crate::tui::state::app_state::Mode::Encode => "→",
        crate::tui::state::app_state::Mode::Decode => "←",
        crate::tui::state::app_state::Mode::Rune => "🪄",
    };

    let arrow_text = vec![
        Line::from(""), // Vertical spacing
        Line::from(Span::styled(arrow_symbol, theme.info_style())),
        Line::from(""), // Vertical spacing
        Line::from(Span::styled("Ctrl+B", theme.line_number_style())),
        Line::from(Span::styled("Round", theme.line_number_style())),
        Line::from(Span::styled("Trip", theme.line_number_style())),
    ];

    let arrow_para = Paragraph::new(arrow_text).alignment(Alignment::Center);

    f.render_widget(arrow_para, area);
}

/// Renders the header with title, mode, and current settings. Zero-copy.
fn render_header(f: &mut Frame, area: Rect, app: &AppState) {
    let theme = &app.theme;

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(34),
            Constraint::Percentage(33),
        ])
        .split(area);

    let title = Paragraph::new(Line::from(vec![
        Span::styled("📋 ", theme.normal_style()),
        Span::styled("RUNE", theme.title_style()),
        Span::styled(" Format", theme.info_style()),
    ]))
    .block(Block::default().borders(Borders::ALL).border_style(theme.border_style(false)));
    f.render_widget(title, chunks[0]);

    let mode_text = Paragraph::new(Line::from(Span::styled(
        app.mode.as_str(),
        theme.highlight_style(),
    )))
    .alignment(Alignment::Center)
    .block(Block::default().borders(Borders::ALL).border_style(theme.border_style(false)));
    f.render_widget(mode_text, chunks[1]);

    // Stack buffers for allocation-free number formatting.
    let mut indent_buf = itoa::Buffer::new();
    let mut depth_buf = itoa::Buffer::new();

    // Show relevant settings based on current mode using static spans.
    let settings_spans = match app.mode {
        crate::tui::state::app_state::Mode::Encode => {
            let delimiter_str = match app.encode_options.delimiter {
                Delimiter::Comma => "comma",
                Delimiter::Tab => "tab",
                Delimiter::Pipe => "pipe",
            };
            let indent_str = match app.encode_options.indent {
                Indent::Spaces(n) => indent_buf.format(n),
            };

            let mut spans = vec![
                Span::styled("Delim:", theme.line_number_style()),
                Span::styled(" ", theme.info_style()),
                Span::styled(delimiter_str, theme.info_style()),
                Span::styled(" | Indent:", theme.line_number_style()),
                Span::styled(" ", theme.info_style()),
                Span::styled(indent_str, theme.info_style()),
                Span::styled("sp", theme.info_style()),
            ];

            if let KeyFoldingMode::Safe = app.encode_options.key_folding {
                let depth_str = if app.encode_options.flatten_depth == usize::MAX {
                    "∞"
                } else {
                    depth_buf.format(app.encode_options.flatten_depth)
                };
                spans.extend(vec![
                    Span::styled(" | fold:", theme.line_number_style()),
                    Span::styled(" on", theme.info_style()),
                    Span::styled(" (", theme.line_number_style()),
                    Span::styled(depth_str, theme.info_style()),
                    Span::styled(")", theme.line_number_style()),
                ]);
            }
            spans
        }
        crate::tui::state::app_state::Mode::Decode => {
            let mut spans = vec![
                Span::styled("Strict:", theme.line_number_style()),
                Span::styled(if app.decode_options.strict { " on" } else { " off" }, theme.info_style()),
                Span::styled(" | Coerce:", theme.line_number_style()),
                Span::styled(if app.decode_options.coerce_types { " on" } else { " off" }, theme.info_style()),
            ];
            if let PathExpansionMode::Safe = app.decode_options.expand_paths {
                spans.extend(vec![
                    Span::styled(" | expand:", theme.line_number_style()),
                    Span::styled(" on", theme.info_style()),
                ]);
            }
            spans
        }
        crate::tui::state::app_state::Mode::Rune => {
            vec![
                Span::styled("Engine:", theme.line_number_style()),
                Span::styled(" Hydron", theme.info_style()),
            ]
        }
    };

    let settings = Paragraph::new(Line::from(settings_spans))
        .alignment(Alignment::Right)
        .block(Block::default().borders(Borders::ALL).border_style(theme.border_style(false)));
    f.render_widget(settings, chunks[2]);
}