/* src/tui/components/repl_panel.rs */
//!▫~•◦-------------------------------‣
//! # Read-Eval-Print-Loop (REPL) panel component for the TUI.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module provides the `ReplPanel` component, which renders an interactive REPL
//! session within the TUI. It handles the display of command history, output, and
//! the active input line.
//!
//! ## Key Capabilities
//! - **Input/Output Display**: Renders a scrollable output area and a separate input line.
//! - **Syntax Highlighting**: Applies different styles for prompts, successes, errors, and info.
//! - **Scrollbar**: Shows a vertical scrollbar when the output exceeds the viewport height.
//! - **Performance-Optimized**: The render logic is zero-copy, operating on borrowed
//!   string data from the application state to ensure fluid interaction.
//!
//! ### Architectural Notes
//! The component is stateless and renders directly from `AppState`. It correctly
//! uses borrowed slices (`&str`) for all text content, ensuring that no `String`
//! allocations occur within the hot render loop. This maintains high performance
//! even with a large amount of REPL history.
//!
//! #### Example
//! ```rust
//! // This is a conceptual example, as a real implementation requires a full TUI loop.
//! use rune_xero::tui::{state::AppState, components::repl_panel::ReplPanel};
//! use ratatui::{Frame, layout::Rect};
//!
//! fn render_repl(frame: &mut Frame, area: Rect, app: &mut AppState) {
//!     // In your TUI rendering loop, you would call:
//!     ReplPanel::render(frame, area, app);
//! }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use ratatui::{
    layout::{Constraint, Direction, Layout, Margin, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap},
    Frame,
};

use crate::tui::state::{AppState, ReplLineKind};

/// A stateless component for rendering the REPL panel.
pub struct ReplPanel;

impl ReplPanel {
    /// Renders the entire REPL panel, including output and input areas.
    pub fn render(f: &mut Frame, area: Rect, app: &mut AppState) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(3)])
            .split(area);

        Self::render_output(f, chunks[0], app);
        Self::render_input(f, chunks[1], app);
    }

    /// Renders the scrollable output section of the REPL.
    fn render_output(f: &mut Frame, area: Rect, app: &mut AppState) {
        // This is a necessary allocation to gather the lines for the Paragraph widget.
        // Importantly, the content of each line is a borrowed `&str`, not a new String.
        let lines: Vec<Line> = app
            .repl
            .output
            .iter()
            .skip(app.repl.scroll_offset)
            .map(|line| {
                let style = match line.kind {
                    ReplLineKind::Prompt => Style::default().fg(Color::Cyan),
                    ReplLineKind::Success => Style::default().fg(Color::Green),
                    ReplLineKind::Error => Style::default().fg(Color::Red),
                    ReplLineKind::Info => Style::default().fg(Color::Yellow),
                };
                // `line.content` is borrowed, making this a zero-copy operation for the text itself.
                Line::from(Span::styled(&line.content, style))
            })
            .collect();

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(" REPL Session (Ctrl+R to toggle, Esc to close) ");

        let paragraph = Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false });

        f.render_widget(paragraph, area);

        let content_height = app.repl.output.len();
        let view_height = area.height.saturating_sub(2) as usize;

        // Render scrollbar only if content overflows the viewable area.
        if content_height > view_height {
            let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("↑"))
                .end_symbol(Some("↓"));

            let mut scrollbar_state =
                ScrollbarState::new(content_height).position(app.repl.scroll_offset);

            f.render_stateful_widget(
                scrollbar,
                area.inner(Margin {
                    vertical: 1,
                    horizontal: 0,
                }),
                &mut scrollbar_state,
            );
        }
    }

    /// Renders the user input line of the REPL.
    fn render_input(f: &mut Frame, area: Rect, app: &mut AppState) {
        let prompt = Span::styled(
            "> ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );

        // The input text is borrowed directly from the app state. Zero-copy.
        let input_text = Span::raw(&app.repl.input);

        // The cursor position is calculated, and only shown when the REPL is active.
        let line = if app.repl.is_active() {
            let (before_cursor, after_cursor) = app.repl.input.split_at(app.repl.cursor_position);
            let before_span = Span::raw(before_cursor);
            let cursor_span = Span::styled("█", Style::default().fg(Color::White));
            let after_span = Span::raw(after_cursor);
            Line::from(vec![prompt, before_span, cursor_span, after_span])
        } else {
            Line::from(vec![prompt, input_text])
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));

        let paragraph = Paragraph::new(line).block(block);

        f.render_widget(paragraph, area);
    }
}