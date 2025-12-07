/* src/tui/components/confirmation_dialog.rs */
//! Terminal UI confirmation dialog component for user interactions.
//!
//! # TOON-RUNE – Confirmation Dialog Component
//!▫~•◦-----------------------------------------‣
//!
//! This module provides a modal confirmation dialog component for the TOON-RUNE
//! terminal user interface, handling user confirmations for destructive actions.
//!
//! ### Key Capabilities
//! - **Modal Display**: Renders centered confirmation dialogs with styled borders and content.
//! - **Action Variants**: Supports different confirmation types (New File, Quit, Delete File).
//! - **Keyboard Navigation**: Visual cues for Y/N/Esc key bindings.
//! - **Responsive Layout**: Automatically centers and sizes dialog within terminal viewport.
//!
//! ### Architectural Notes
//! This component integrates with the `AppState` type's `ConfirmationAction` enum and
//! works alongside other TUI components like editors and file browsers. Dialog rendering
//! uses Ratatui's layout system for consistent positioning and styling.
//!
//! ### Example
//! ```rust
//! use rune_format::tui::components::confirmation_dialog::ConfirmationDialog;
//! use rune_format::tui::state::app_state::ConfirmationAction;
//! use ratatui::{Frame, layout::Rect};
//!
//! let action = ConfirmationAction::DeleteFile;
//! // In your TUI rendering loop:
//! // confirmation_dialog::render(&mut frame, dialog_area, action);
//!
//! // The dialog renders with appropriate title and message for the delete action.
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};

use crate::tui::state::app_state::ConfirmationAction;

pub struct ConfirmationDialog;

impl ConfirmationDialog {
    pub fn render(frame: &mut Frame, area: Rect, action: ConfirmationAction) {
        let (title, message) = match action {
            ConfirmationAction::NewFile => (
                "New File",
                "Current file has unsaved changes. Create new file anyway?",
            ),
            ConfirmationAction::Quit => ("Quit", "Current file has unsaved changes. Quit anyway?"),
            ConfirmationAction::DeleteFile => (
                "Delete File",
                "Are you sure you want to delete this file? This cannot be undone.",
            ),
            ConfirmationAction::None => return,
        };

        // Create centered modal
        let popup_area = Self::centered_rect(50, 30, area);

        // Clear the area
        frame.render_widget(Clear, popup_area);

        // Create layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Title
                Constraint::Min(3),    // Message
                Constraint::Length(3), // Buttons
            ])
            .split(popup_area);

        // Render border
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow))
            .title(title)
            .title_alignment(Alignment::Center);
        frame.render_widget(block, popup_area);

        // Render message
        let message_paragraph = Paragraph::new(message)
            .style(Style::default().fg(Color::White))
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true });
        frame.render_widget(message_paragraph, chunks[1]);

        // Render buttons
        let buttons = Line::from(vec![
            Span::styled(
                "[Y]",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" Yes    "),
            Span::styled(
                "[N]",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" No    "),
            Span::styled("[ESC]", Style::default().fg(Color::Gray)),
            Span::raw(" Cancel"),
        ]);
        let buttons_paragraph = Paragraph::new(buttons).alignment(Alignment::Center);
        frame.render_widget(buttons_paragraph, chunks[2]);
    }

    fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
        let popup_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage((100 - percent_y) / 2),
                Constraint::Percentage(percent_y),
                Constraint::Percentage((100 - percent_y) / 2),
            ])
            .split(r);

        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage((100 - percent_x) / 2),
                Constraint::Percentage(percent_x),
                Constraint::Percentage((100 - percent_x) / 2),
            ])
            .split(popup_layout[1])[1]
    }
}
