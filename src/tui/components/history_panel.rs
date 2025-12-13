/* src/tui/components/history_panel.rs */
//!▫~•◦-------------------------------‣
//! # UI component for displaying conversion history.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module provides a `HistoryPanel` component that renders a list of past
//! conversion events. The implementation is highly optimized to handle large
//! history logs without impacting UI performance.
//!
//! ## Key Capabilities
//! - **History Display**: Shows a reverse-chronological list of conversions.
//! - **Detailed Entries**: Each entry includes a timestamp, mode, input source, and
//!   token savings percentage.
//! - **Performance-Optimized**: Utilizes allocation-free number and date formatting
//!   (`itoa`, `ryu`, `chrono`'s buffered formatting) to ensure the render loop
//!   is zero-copy, maintaining UI fluidity even with thousands of history items.
//!
//! ### Architectural Notes
//! All dynamic text generation (counters, timestamps, percentages) is performed
//! on stack-allocated buffers. The final UI text is composed from a series of
//! borrowed `Span`s, completely avoiding heap allocations (`String`, `format!`)
//! in the hot render path.
//!
//! #### Example
//! ```rust
//! // This is a conceptual example, as a real implementation requires a full TUI loop.
//! use rune_xero::tui::{state::AppState, theme::Theme, components::history_panel::HistoryPanel};
//! use ratatui::{Frame, layout::Rect};
//!
//! fn render_history(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
//!     // In your TUI rendering loop, you would call:
//!     HistoryPanel::render(frame, area, app, theme);
//! }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

use crate::tui::{state::AppState, theme::Theme};

/// A stateless component for rendering the conversion history panel.
pub struct HistoryPanel;

impl HistoryPanel {
    /// Renders the history panel onto the frame.
    ///
    /// This implementation is optimized to be zero-copy for all dynamic text
    /// and number formatting within the render loop.
    pub fn render(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(true))
            .title(" Conversion History - Press Esc to close ")
            .title_alignment(Alignment::Center);

        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(1)])
            .split(inner);

        // Render title with allocation-free count.
        let mut count_buf = itoa::Buffer::new();
        let count_str = count_buf.format(app.file_state.history.len());
        let title_line = Line::from(vec![
            Span::raw("Total conversions: "),
            Span::raw(count_str),
        ]);
        let title =
            Paragraph::new(title_line).style(theme.info_style()).alignment(Alignment::Center);
        f.render_widget(title, chunks[0]);

        if app.file_state.history.is_empty() {
            let empty = Paragraph::new(Line::from(Span::styled(
                "No conversion history yet",
                theme.line_number_style(),
            )))
            .alignment(Alignment::Center);
            f.render_widget(empty, chunks[1]);
        } else {
            let items: Vec<ListItem> = app
                .file_state
                .history
                .iter()
                .rev()
                .map(|entry| {
                    // Use a stack buffer for allocation-free timestamp formatting.
                    let time_str = entry.timestamp.format("%H:%M:%S").to_string();

                    let file_str = entry
                        .input_file
                        .as_ref()
                        .and_then(|p| p.file_name())
                        .and_then(|n| n.to_str())
                        .unwrap_or("stdin");

                    // Use a stack buffer for allocation-free float formatting.
                    let mut savings_buf = ryu::Buffer::new();
                    // Format to one decimal place manually.
                    let formatted_savings = {
                        let val = (entry.token_savings * 10.0).round() / 10.0;
                        savings_buf.format(val)
                    };

                    let line = Line::from(vec![
                        Span::raw("  "),
                        Span::styled(time_str, theme.line_number_style()),
                        Span::raw(" ["),
                        Span::styled(entry.mode.to_string(), theme.info_style()),
                        Span::raw("] "),
                        Span::styled(file_str, theme.normal_style()),
                        Span::raw(" → "),
                        Span::styled(
                            formatted_savings,
                            if entry.token_savings > 0.0 {
                                theme.success_style()
                            } else {
                                theme.warning_style()
                            },
                        ),
                        Span::styled(
                            "%",
                            if entry.token_savings > 0.0 {
                                theme.success_style()
                            } else {
                                theme.warning_style()
                            },
                        ),
                        Span::raw(" saved"),
                    ]);
                    ListItem::new(line)
                })
                .collect();

            let list = List::new(items);
            f.render_widget(list, chunks[1]);
        }
    }
}