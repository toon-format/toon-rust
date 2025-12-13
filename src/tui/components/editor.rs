/* src/tui/components/editor.rs */
//!▫~•◦-------------------------------‣
//! # Input and output editor panel components for the RUNE TUI.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module provides the `EditorComponent`, which is responsible for rendering the
//! primary input and output text areas using the `tui-textarea` widget.
//!
//! ## Key Capabilities
//! - **Dual Panel Rendering**: Manages the display of both the input and output text editors.
//! - **Active State Styling**: Dynamically updates border styles and titles to indicate
//!   which panel is currently focused.
//! - **Theme Integration**: Applies styles from the central `Theme` for a consistent look.
//! - **Performance-Optimized**: Renders with zero heap allocations for titles, ensuring
//!   a fluid user experience.
//!
//! ### Architectural Notes
//! The component is stateless and mutates the `AppState`'s `tui_textarea::TextArea`
//! widgets directly to apply styling and render them. Titles are composed from
//! static `Span`s to avoid string formatting in the hot render loop.
//!
//! #### Example
//! ```rust
//! // This is a conceptual example, as a real implementation requires a full TUI loop.
//! use rune_xero::tui::{state::AppState, theme::Theme, components::editor::EditorComponent};
//! use ratatui::{Frame, layout::Rect};
//!
//! fn render_editors(
//!     frame: &mut Frame,
//!     input_area: Rect,
//!     output_area: Rect,
//!     app: &mut AppState,
//!     theme: &Theme
//! ) {
//!     // In your TUI rendering loop, you would call:
//!     EditorComponent::render(frame, input_area, output_area, app, theme);
//! }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use ratatui::{
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Borders},
    Frame,
};

use crate::tui::{state::AppState, theme::Theme};

/// A stateless component for rendering the input and output editor panels.
pub struct EditorComponent;

impl EditorComponent {
    /// Renders the input and output editors onto the frame.
    pub fn render(
        f: &mut Frame,
        input_area: Rect,
        output_area: Rect,
        app: &mut AppState,
        theme: &Theme,
    ) {
        // --- Input Panel ---
        let input_active = app.editor.is_input_active();
        let lang_str = match app.mode {
            crate::tui::state::app_state::Mode::Encode => "JSON",
            crate::tui::state::app_state::Mode::Decode => "RUNE",
            crate::tui::state::app_state::Mode::Rune => "RUNE",
        };
        let active_indicator = if input_active { " ●" } else { "" };

        // Construct the title from static spans to avoid String allocation.
        let input_title = Line::from(vec![
            Span::raw(" Input ("),
            Span::raw(lang_str),
            Span::raw(")"),
            Span::raw(active_indicator),
            Span::raw(" "),
        ]);

        let input_block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(input_active))
            .title(input_title)
            .style(theme.normal_style());

        // Apply styles and render the input TextArea widget.
        app.editor.input.set_block(input_block);
        app.editor
            .input
            .set_cursor_line_style(theme.selection_style());
        app.editor.input.set_style(theme.normal_style());
        f.render_widget(app.editor.input.widget(), input_area);

        // --- Output Panel ---
        let output_active = app.editor.is_output_active();
        let lang_str = match app.mode {
            crate::tui::state::app_state::Mode::Encode => "RUNE",
            crate::tui::state::app_state::Mode::Decode => "JSON",
            crate::tui::state::app_state::Mode::Rune => "Results",
        };
        let active_indicator = if output_active { " ●" } else { "" };

        // Construct the title from static spans to avoid String allocation.
        let output_title = Line::from(vec![
            Span::raw(" Output ("),
            Span::raw(lang_str),
            Span::raw(")"),
            Span::raw(active_indicator),
            Span::raw(" "),
        ]);

        let output_block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(output_active))
            .title(output_title)
            .style(theme.normal_style());

        // Apply styles and render the output TextArea widget.
        app.editor.output.set_block(output_block);
        app.editor
            .output
            .set_cursor_line_style(theme.selection_style());
        app.editor.output.set_style(theme.normal_style());
        f.render_widget(app.editor.output.widget(), output_area);
    }
}