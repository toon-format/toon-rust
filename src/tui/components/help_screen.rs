/* src/tui/components/help_screen.rs */
//!▫~•◦-------------------------------‣
//! # Help screen component showing keyboard shortcuts.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module provides a stateless `HelpScreen` component that displays a modal
//! view with a list of all available keyboard shortcuts and application information.
//!
//! ## Key Capabilities
//! - **Static Content Display**: Renders a pre-defined list of keybindings and descriptions.
//! - **Styled Layout**: Uses `ratatui` to present the information in a clear, bordered layout.
//! - **Performance-Optimized**: Renders with zero heap allocations by using pre-formatted
//!   static strings for all content, ensuring it has no performance impact on the TUI.
//!
//! ### Architectural Notes
//! All text displayed by this component is sourced from `&'static str` literals.
//! The keybindings are retrieved from the `KeyBindings` utility, which provides
//! pre-padded strings to avoid any runtime `format!` calls in the render loop.
//!
//! #### Example
//! ```rust
//! // This is a conceptual example, as a real implementation requires a full TUI loop.
//! use rune_xero::tui::{theme::Theme, components::help_screen::HelpScreen};
//! use ratatui::{Frame, layout::Rect};
//!
//! fn render_help(frame: &mut Frame, area: Rect, theme: &Theme) {
//!     // In your TUI rendering loop, you would call:
//!     HelpScreen::render(frame, area, theme);
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

use crate::tui::{keybindings::KeyBindings, theme::Theme};

/// A stateless component for rendering the help screen.
pub struct HelpScreen;

impl HelpScreen {
    /// Renders the help screen onto the frame.
    ///
    /// This implementation is fully zero-copy, using only static string slices
    /// for all text content to avoid allocations in the render loop.
    pub fn render(f: &mut Frame, area: Rect, theme: &Theme) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(true))
            .title(" Help - Press F1 or Esc to close ")
            .title_alignment(Alignment::Center);

        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(3),
            ])
            .split(inner);

        let title = Paragraph::new(vec![
            Line::from(Span::styled(
                "RUNE Format - Interactive TUI",
                theme.title_style(),
            )),
            Line::from(Span::styled(
                "Token-Oriented Object Notation",
                theme.info_style(),
            )),
        ])
        .alignment(Alignment::Center);
        f.render_widget(title, chunks[0]);

        // Keybindings are fetched as pre-padded &'static str to avoid formatting.
        let shortcuts = KeyBindings::shortcuts();
        let items: Vec<ListItem> = shortcuts
            .iter()
            .map(|(key, desc)| {
                let line = Line::from(vec![
                    Span::raw("  "),
                    Span::styled(*key, theme.info_style()),
                    Span::raw(" "),
                    Span::styled(*desc, theme.normal_style()),
                ]);
                ListItem::new(line)
            })
            .collect();

        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border_style(false))
                .title(" Keyboard Shortcuts "),
        );
        f.render_widget(list, chunks[1]);

        let footer = Paragraph::new(vec![
            Line::from(Span::styled(
                "RUNE is a compact, human-readable format for passing structured data to LLMs",
                theme.normal_style(),
            )),
            Line::from(vec![
                Span::styled("Repository: ", theme.line_number_style()),
                Span::styled(
                    "github.com/toon-format/toon-rust",
                    theme.info_style(),
                ),
            ]),
        ])
        .alignment(Alignment::Center);
        f.render_widget(footer, chunks[2]);
    }
}