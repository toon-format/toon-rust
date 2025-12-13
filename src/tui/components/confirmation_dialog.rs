/* src/tui/components/confirmation_dialog.rs */
//!▫~•◦-------------------------------‣
//! # Terminal UI confirmation dialog component for user interactions.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module provides a modal confirmation dialog component for the RUNE
//! terminal user interface, handling user confirmations for destructive actions.
//!
//! ## Key Capabilities
//! - **Modal Display**: Renders centered confirmation dialogs with styled borders and content.
//! - **Action Variants**: Supports different confirmation types (New File, Quit, Delete File).
//! - **Keyboard Navigation**: Visual cues for Y/N/Esc key bindings.
//! - **Responsive Layout**: Automatically centers and sizes dialog within terminal viewport.
//!
//! ### Architectural Notes
//! This component integrates with the `AppState` type's `ConfirmationAction` enum and
//! works alongside other TUI components like editors and file browsers. Dialog rendering
//! uses Ratatui's layout system for consistent positioning and styling. The implementation
//! is zero-copy for all text content, ensuring high rendering performance.
//!
//! #### Example
//! ```rust
//! // This is a conceptual example, as a real implementation requires a full TUI loop.
//! use rune_xero::tui::components::confirmation_dialog::ConfirmationDialog;
//! use rune_xero::tui::state::app_state::ConfirmationAction;
//! use ratatui::{Frame, layout::Rect};
//!
//! fn render_a_dialog(frame: &mut Frame, area: Rect) {
//!     let action = ConfirmationAction::DeleteFile;
//!     // In your TUI rendering loop, you would call:
//!     ConfirmationDialog::render(frame, area, action);
//!
//!     // The dialog renders with the appropriate title and message for the delete action.
//! }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

use crate::tui::state::app_state::ConfirmationAction;

/// A stateless component responsible for rendering a confirmation dialog.
pub struct ConfirmationDialog;

impl ConfirmationDialog {
    /// Renders the confirmation dialog onto the frame.
    ///
    /// This function is highly optimized, using zero-copy for all text content by
    /// leveraging `&'static str` and `ratatui`'s borrow-based widget APIs.
    pub fn render(frame: &mut Frame, area: Rect, action: ConfirmationAction) {
        let (title, message): (&'static str, &'static str) = match action {
            ConfirmationationAction::NewFile => (
                "New File",
                "Current file has unsaved changes. Create new file anyway?",
            ),
            ConfirmationAction::Quit => ("Quit", "Current file has unsaved changes. Quit anyway?"),
            ConfirmationAction::DeleteFile => (
                "Delete File",
                "Are you sure you want to delete this file? This cannot be undone.",
            ),
            // If there's no action, we render nothing.
            ConfirmationAction::None => return,
        };

        // Create a centered rectangle for the modal.
        let popup_area = Self::centered_rect(50, 30, area);

        // Clear the space where the dialog will be rendered to ensure no old UI shows through.
        frame.render_widget(Clear, popup_area);

        // Define the layout chunks for title, message, and buttons.
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Space for top border and title padding
                Constraint::Min(3),    // Flexible space for the message
                Constraint::Length(3), // Space for bottom border and buttons
            ])
            .split(popup_area);

        // Render the main dialog block with a border and title.
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow))
            .title(title)
            .title_alignment(Alignment::Center);
        frame.render_widget(block, popup_area);

        // Render the confirmation message, centered and wrapped.
        let message_paragraph = Paragraph::new(message)
            .style(Style::default().fg(Color::White))
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true });
        frame.render_widget(message_paragraph, chunks[1]);

        // Render the button hints. All text here is `&'static str` (zero-copy).
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
        // We render in the bottom-most part of the inner dialog area.
        frame.render_widget(buttons_paragraph, chunks[2]);
    }

    /// Helper function to create a centered rectangle within a given area.
    ///
    /// Percentages are used to define the size of the inner rectangle relative
    /// to the containing area `r`.
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