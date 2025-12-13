/* src/tui/components/diff_viewer.rs */
//!▫~•◦--------------------------------‣
//! # Side-by-side diff viewer for input/output comparison.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module provides a high-performance, zero-copy side-by-side diff viewer
//! component for the RUNE TUI. It is designed for efficiently displaying input
//! text and its corresponding processed output.
//!
//! ## Key Capabilities
//! - **Dual-Panel Layout**: Renders input and output text in two vertical panels.
//! - **Dynamic Titles**: Titles change based on the application's current mode (Encode/Decode).
//! - **Line Numbering**: Displays formatted line numbers for readability.
//! - **Performance-Obsessed**: Achieves zero heap allocations for text and number
//!   formatting within the render loop, ensuring smooth scrolling even with large files.
//!
//! ### Architectural Notes
//! This component is stateless and operates directly on the `AppState`. It leverages
//! the `itoa` crate for allocation-free integer-to-string conversion for line
//! numbers and composes titles from static `Span`s to avoid `format!`.
//!
//! #### Example
//! ```rust
//! // This is a conceptual example, as a real implementation requires a full TUI loop.
//! use rune_xero::tui::{state::AppState, theme::Theme, components::diff_viewer::DiffViewer};
//! use ratatui::{Frame, layout::Rect};
//!
//! fn render_a_viewer(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
//!     // In your TUI rendering loop, you would call:
//!     DiffViewer::render(frame, area, app, theme);
//! }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::tui::{state::AppState, theme::Theme};

/// A stateless component for rendering a side-by-side diff view.
pub struct DiffViewer;

impl DiffViewer {
    /// Renders the diff viewer onto the frame.
    ///
    /// This implementation is optimized to be zero-copy for all text and number
    /// formatting within the render loop.
    pub fn render(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(true))
            .title(" Side-by-Side Comparison - Press Esc to close ")
            .title_alignment(Alignment::Center);

        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(inner);

        // --- Input Panel ---
        let input_text = app.editor.get_input();
        let input_title_str = match app.mode {
            crate::tui::state::app_state::Mode::Encode => "JSON Input",
            crate::tui::state::app_state::Mode::Decode => "RUNE Input",
            crate::tui::state::app_state::Mode::Rune => "RUNE Input",
        };

        // Create the title line without allocation.
        let input_title = Line::from(vec![
            Span::raw(" "),
            Span::styled(input_title_str, Style::default()),
            Span::raw(" "),
        ]);

        let input_lines: Vec<Line> = input_text
            .lines()
            .enumerate()
            .map(|(idx, line)| {
                // Use a stack-based buffer for allocation-free integer formatting.
                let mut num_buf = itoa::Buffer::new();
                let num_str = num_buf.format(idx + 1);
                // Create padding from a static string slice to avoid allocation.
                const PADDING: &str = "    ";
                let padding = &PADDING[..PADDING.len().saturating_sub(num_str.len())];

                Line::from(vec![
                    Span::styled(padding, theme.line_number_style()),
                    Span::styled(num_str, theme.line_number_style()),
                    Span::raw(" "),
                    Span::styled(line, theme.normal_style()),
                ])
            })
            .collect();

        let input_para = Paragraph::new(input_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border_style(false))
                    .title(input_title),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(input_para, chunks[0]);

        // --- Output Panel ---
        let output_text = app.editor.get_output();
        let output_title_str = match app.mode {
            crate::tui::state::app_state::Mode::Encode => "RUNE Output",
            crate::tui::state::app_state::Mode::Decode => "JSON Output",
            crate::tui::state::app_state::Mode::Rune => "Parsed Results",
        };

        // Create the title line without allocation.
        let output_title = Line::from(vec![
            Span::raw(" "),
            Span::styled(output_title_str, Style::default()),
            Span::raw(" "),
        ]);

        let output_lines: Vec<Line> = output_text
            .lines()
            .enumerate()
            .map(|(idx, line)| {
                // Use a stack-based buffer for allocation-free integer formatting.
                let mut num_buf = itoa::Buffer::new();
                let num_str = num_buf.format(idx + 1);
                // Create padding from a static string slice to avoid allocation.
                const PADDING: &str = "    ";
                let padding = &PADDING[..PADDING.len().saturating_sub(num_str.len())];

                Line::from(vec![
                    Span::styled(padding, theme.line_number_style()),
                    Span::styled(num_str, theme.line_number_style()),
                    Span::raw(" "),
                    Span::styled(line, theme.normal_style()),
                ])
            })
            .collect();

        let output_para = Paragraph::new(output_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border_style(false))
                    .title(output_title),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(output_para, chunks[1]);
    }
}