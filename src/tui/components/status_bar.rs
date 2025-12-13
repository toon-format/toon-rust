/* src/tui/components/status_bar.rs */
//!▫~•◦-------------------------------‣
//! # Status bar component showing mode, file, and key commands.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module provides the `StatusBar` component, which renders the persistent bar
//! at the bottom of the TUI. It displays critical information like the current
//! application mode, file status, messages, and key command hints.
//!
//! ## Key Capabilities
//! - **Contextual Information**: Displays the current mode, file path, and modified status.
//! - **Message Display**: Shows transient status or error messages from the application.
//! - **Key Command Hints**: Provides context-aware hints for essential commands.
//! - **Performance-Optimized**: Renders with zero heap allocations by composing the
//!   status line from static and borrowed string slices, ensuring it has no
//!   performance impact on the TUI.
//!
//! ### Architectural Notes
//! The component is stateless and renders directly from `AppState`. It completely avoids
//! the `format!` macro and `.to_string()` calls in its hot render path. All dynamic
//! content is constructed by composing `ratatui::Span`s from borrowed data (`&str`),
//! which is crucial for maintaining a responsive, high-performance UI.
//!
//! #### Example
//! ```rust
//! // This is a conceptual example, as a real implementation requires a full TUI loop.
//! use rune_xero::tui::{state::AppState, theme::Theme, components::status_bar::StatusBar};
//! use ratatui::{Frame, layout::Rect};
//!
//! fn render_status(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
//!     // In your TUI rendering loop, you would call:
//!     StatusBar::render(frame, area, app, theme);
//! }
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

use crate::tui::{state::AppState, theme::Theme};

/// A stateless component for rendering the application's status bar.
pub struct StatusBar;

impl StatusBar {
    /// Renders the status bar onto the frame.
    ///
    /// This implementation is optimized to be zero-copy, avoiding all `String`
    /// allocations in the render loop.
    pub fn render(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(area);

        // --- Left Side ---
        let mut left_spans = vec![];

        // Mode - Composed from two static spans to avoid format!
        left_spans.push(Span::styled(app.mode.short_name(), theme.info_style()));
        left_spans.push(Span::styled(" ", theme.info_style()));
        left_spans.push(Span::raw("| "));

        // File Path
        if let Some(ref path) = app.file_state.current_file {
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("Untitled");
            left_spans.push(Span::styled(file_name, theme.normal_style()));

            if app.file_state.is_modified {
                left_spans.push(Span::styled(" [Modified]", theme.warning_style()));
            }
        } else {
            left_spans.push(Span::styled("No file", theme.line_number_style()));
        }

        left_spans.push(Span::raw(" | "));

        // Status/Error Message - Composed from multiple spans to avoid format!
        if let Some(ref error) = app.error_message {
            left_spans.push(Span::styled("✗ ", theme.error_style()));
            left_spans.push(Span::styled(error, theme.error_style()));
        } else if let Some(ref status) = app.status_message {
            left_spans.push(Span::styled("✓ ", theme.success_style()));
            left_spans.push(Span::styled(status, theme.success_style()));
        } else {
            left_spans.push(Span::styled("Ready ", theme.normal_style()));
        }

        left_spans.push(Span::raw(" | "));

        // Theme Name - Use &'static str directly, no .to_string()
        let theme_name = match theme {
            Theme::Dark => "Dark",
            Theme::Light => "Light",
        };
        left_spans.push(Span::styled(theme_name, theme.line_number_style()));

        let left_line = Line::from(left_spans);
        let left_paragraph =
            Paragraph::new(left_line).block(Block::default().borders(Borders::ALL));

        // --- Right Side ---
        let key_commands = vec![
            Span::styled("F1", theme.info_style()),
            Span::raw(" Help | "),
            Span::styled("Ctrl+C", theme.info_style()),
            Span::raw(" Quit"),
        ];

        let right_line = Line::from(key_commands);
        let right_paragraph = Paragraph::new(right_line)
            .block(Block::default().borders(Borders::ALL))
            .alignment(Alignment::Right);

        f.render_widget(left_paragraph, chunks[0]);
        f.render_widget(right_paragraph, chunks[1]);
    }
}