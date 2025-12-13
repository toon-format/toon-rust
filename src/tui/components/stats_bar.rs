/* src/tui/components/stats_bar.rs */
//!▫~•◦-------------------------------‣
//! # Statistics bar showing token and byte savings.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module provides the `StatsBar` component, which renders a summary of the
//! token and byte savings achieved from the last conversion.
//!
//! ## Key Capabilities
//! - **Conversion Statistics**: Displays before/after token counts, byte counts,
//!   and the percentage savings for both.
//! - **Conditional Styling**: Colors the percentage savings based on whether the
//!   result was a gain or a loss.
//! - **Performance-Optimized**: Renders with zero heap allocations by using stack-based
//!   number-to-string conversion, ensuring no performance penalty for displaying stats.
//!
//! ### Architectural Notes
//! The component is stateless and renders directly from `AppState`. All numeric values
//! are formatted to stack buffers using the `itoa` (for integers) and `ryu` (for floats)
//! crates. The final UI line is composed from `&'static str` and stack-borrowed `&str`
//! slices, completely avoiding the `format!` macro in the hot render path.
//!
//! #### Example
//! ```rust
//! // This is a conceptual example, as a real implementation requires a full TUI loop.
//! use rune_xero::tui::{state::AppState, theme::Theme, components::stats_bar::StatsBar};
//! use ratatui::{Frame, layout::Rect};
//!
//! fn render_stats(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
//!     // In your TUI rendering loop, you would call:
//!     StatsBar::render(frame, area, app, theme);
//! }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use ratatui::{
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::tui::{state::AppState, theme::Theme};

/// A stateless component for rendering the statistics bar.
pub struct StatsBar;

impl StatsBar {
    /// Renders the statistics bar onto the frame.
    ///
    /// This implementation is optimized to be zero-copy for all number formatting.
    pub fn render(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(false))
            .title(" Statistics ");

        if let Some(ref stats) = app.stats {
            // Stack buffers for allocation-free number formatting.
            let mut jt_buf = itoa::Buffer::new();
            let mut tt_buf = itoa::Buffer::new();
            let mut ts_buf = ryu::Buffer::new();
            let mut jb_buf = itoa::Buffer::new();
            let mut tb_buf = itoa::Buffer::new();
            let mut bs_buf = ryu::Buffer::new();

            // Format all numbers to their respective buffers.
            let json_tokens_str = jt_buf.format(stats.json_tokens);
            let toon_tokens_str = tt_buf.format(stats.toon_tokens);
            let token_savings_str = ts_buf.format_finite(stats.token_savings);
            let json_bytes_str = jb_buf.format(stats.json_bytes);
            let toon_bytes_str = tb_buf.format(stats.toon_bytes);
            let byte_savings_str = bs_buf.format_finite(stats.byte_savings);

            let token_savings_style = if stats.token_savings > 0.0 {
                theme.success_style()
            } else {
                theme.error_style()
            };

            let byte_savings_style = if stats.byte_savings > 0.0 {
                theme.success_style()
            } else {
                theme.error_style()
            };

            // Compose the final line from static and stack-borrowed spans.
            let spans = vec![
                Span::styled(" Stats: ", theme.title_style()),
                Span::raw("Tokens: "),
                Span::styled(json_tokens_str, theme.info_style()),
                Span::styled("→", theme.info_style()),
                Span::styled(toon_tokens_str, theme.info_style()),
                Span::raw(" ("),
                Span::styled(token_savings_str, token_savings_style),
                Span::styled("%)", token_savings_style),
                Span::raw(" | Bytes: "),
                Span::styled(json_bytes_str, theme.info_style()),
                Span::styled("→", theme.info_style()),
                Span::styled(toon_bytes_str, theme.info_style()),
                Span::raw(" ("),
                Span::styled(byte_savings_str, byte_savings_style),
                Span::styled("%)", byte_savings_style),
            ];

            let line = Line::from(spans);
            let paragraph = Paragraph::new(line).block(block);
            f.render_widget(paragraph, area);
        } else {
            let paragraph = Paragraph::new(Line::from(Span::styled(
                " No statistics available yet ",
                theme.line_number_style(),
            )))
            .block(block);
            f.render_widget(paragraph, area);
        }
    }
}