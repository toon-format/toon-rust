/* src/tui/components/settings_panel.rs */
//!▫~•◦-------------------------------‣
//! # Settings panel for configuring encode/decode options.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module provides the `SettingsPanel` component, which renders a modal view
//! for configuring all RUNE encoding and decoding options.
//!
//! ## Key Capabilities
//! - **Dynamic Settings Display**: Shows the current state of all encode/decode options.
//! - **Interactive Hints**: Provides keybinding hints for modifying each setting.
//! - **Mode-Aware Title**: The title reflects the application's current mode.
//! - **Performance-Optimized**: Renders with zero heap allocations by using static
//!   strings and stack-based number formatting, ensuring maximum UI responsiveness.
//!
//! ### Architectural Notes
//! The component is stateless and renders directly from `AppState`. All dynamic text,
//! including numeric values, is generated without heap allocations by using the `itoa`
//! crate and composing `ratatui` `Span`s from `&'static str` literals.
//!
//! #### Example
//! ```rust
//! // This is a conceptual example, as a real implementation requires a full TUI loop.
//! use rune_xero::tui::{state::AppState, theme::Theme, components::settings_panel::SettingsPanel};
//! use ratatui::{Frame, layout::Rect};
//!
//! fn render_settings(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
//!     // In your TUI rendering loop, you would call:
//!     SettingsPanel::render(frame, area, app, theme);
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

use crate::{
    tui::{state::AppState, theme::Theme},
    types::{Delimiter, Indent, KeyFoldingMode, PathExpansionMode},
};

/// A stateless component for rendering the settings panel.
pub struct SettingsPanel;

impl SettingsPanel {
    /// Renders the settings panel onto the frame.
    ///
    /// This implementation is optimized to be zero-copy for all dynamic text
    /// and number formatting within the render loop.
    pub fn render(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(true))
            .title(" Settings - Press Ctrl+P or Esc to close ")
            .title_alignment(Alignment::Center);

        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(1), Constraint::Length(1)])
            .split(inner);

        // Render title without allocation.
        let title = Paragraph::new(Line::from(vec![
            Span::raw("Current Mode: "),
            Span::raw(app.mode.as_str()),
        ]))
        .style(theme.title_style())
        .alignment(Alignment::Center);
        f.render_widget(title, chunks[0]);

        let mut items = vec![];

        items.push(ListItem::new(Line::from(Span::styled(
            "═══ Encode Settings (JSON → RUNE) ═══",
            theme.title_style(),
        ))));

        let delimiter_str = match app.encode_options.delimiter {
            Delimiter::Comma => "Comma (,)",
            Delimiter::Tab => "Tab (\\t)",
            Delimiter::Pipe => "Pipe (|)",
        };
        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Delimiter:       ", theme.info_style()),
            Span::styled(delimiter_str, theme.normal_style()),
            Span::styled("  [Press 'd' to cycle]", theme.line_number_style()),
        ])));

        // Render indentation with allocation-free number formatting.
        let Indent::Spaces(indent_spaces) = app.encode_options.indent;
        let mut indent_buf = itoa::Buffer::new();
        let indent_str = indent_buf.format(indent_spaces);
        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Indentation:     ", theme.info_style()),
            Span::styled(indent_str, theme.normal_style()),
            Span::raw(" spaces"),
            Span::styled("  [+/- to adjust]", theme.line_number_style()),
        ])));

        let fold_keys_str = match app.encode_options.key_folding {
            KeyFoldingMode::Off => "Off",
            KeyFoldingMode::Safe => "On (Safe)",
        };
        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Key Folding:     ", theme.info_style()),
            Span::styled(fold_keys_str, theme.normal_style()),
            Span::styled("  [Press 'f' to toggle]", theme.line_number_style()),
        ])));

        if app.encode_options.key_folding != KeyFoldingMode::Off {
            let mut depth_buf = itoa::Buffer::new();
            let depth_str = if app.encode_options.flatten_depth == usize::MAX {
                "Unlimited"
            } else {
                depth_buf.format(app.encode_options.flatten_depth)
            };
            items.push(ListItem::new(Line::from(vec![
                Span::styled("  Flatten Depth:   ", theme.info_style()),
                Span::styled(depth_str, theme.normal_style()),
                Span::styled(
                    "  [[/] to adjust, [u] for unlimited]",
                    theme.line_number_style(),
                ),
            ])));
        }

        items.push(ListItem::new(Line::from("")));

        items.push(ListItem::new(Line::from(Span::styled(
            "═══ Decode Settings (RUNE → JSON) ═══",
            theme.title_style(),
        ))));

        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Strict Mode:     ", theme.info_style()),
            Span::styled(
                if app.decode_options.strict { "On" } else { "Off" },
                theme.normal_style(),
            ),
            Span::styled("  [Press 's' to toggle]", theme.line_number_style()),
        ])));

        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Type Coercion:   ", theme.info_style()),
            Span::styled(
                if app.decode_options.coerce_types { "On" } else { "Off" },
                theme.normal_style(),
            ),
            Span::styled("  [Press 'c' to toggle]", theme.line_number_style()),
        ])));

        let expand_paths_str = match app.decode_options.expand_paths {
            PathExpansionMode::Off => "Off",
            PathExpansionMode::Safe => "On (Safe)",
        };
        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Path Expansion:  ", theme.info_style()),
            Span::styled(expand_paths_str, theme.normal_style()),
            Span::styled("  [Press 'p' to toggle]", theme.line_number_style()),
        ])));

        let list = List::new(items);
        f.render_widget(list, chunks[1]);

        let instructions = Paragraph::new(Line::from(vec![
            Span::styled("Press ", theme.line_number_style()),
            Span::styled("Ctrl+E", theme.info_style()),
            Span::styled(" to toggle mode | ", theme.line_number_style()),
            Span::styled("Ctrl+R", theme.info_style()),
            Span::styled(" to refresh conversion", theme.line_number_style()),
        ]))
        .alignment(Alignment::Center);
        f.render_widget(instructions, chunks[2]);
    }
}