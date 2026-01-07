//! Input and output editor panels.

use ratatui::{
    layout::Rect,
    widgets::{Block, Borders},
    Frame,
};

use crate::tui::{state::AppState, theme::Theme};

pub struct EditorComponent;

impl EditorComponent {
    pub fn render(
        f: &mut Frame,
        input_area: Rect,
        output_area: Rect,
        app: &mut AppState,
        theme: &Theme,
    ) {
        let input_active = app.editor.is_input_active();
        let input_title = format!(
            " Input ({}) {} ",
            match app.mode {
                crate::tui::state::app_state::Mode::Encode => "JSON",
                crate::tui::state::app_state::Mode::Decode => "TOON",
            },
            if input_active { "●" } else { "" }
        );

        let input_block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(input_active))
            .title(input_title)
            .style(theme.normal_style());

        app.editor.input.set_block(input_block);
        app.editor
            .input
            .set_cursor_line_style(theme.selection_style());
        app.editor.input.set_style(theme.normal_style());

        f.render_widget(&app.editor.input, input_area);

        let output_active = app.editor.is_output_active();
        let output_title = format!(
            " Output ({}) {} ",
            match app.mode {
                crate::tui::state::app_state::Mode::Encode => "TOON",
                crate::tui::state::app_state::Mode::Decode => "JSON",
            },
            if output_active { "●" } else { "" }
        );

        let output_block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(output_active))
            .title(output_title)
            .style(theme.normal_style());

        app.editor.output.set_block(output_block);
        app.editor
            .output
            .set_cursor_line_style(theme.selection_style());
        app.editor.output.set_style(theme.normal_style());

        f.render_widget(&app.editor.output, output_area);
    }
}
