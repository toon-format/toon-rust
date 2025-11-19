//! Side-by-side diff viewer for input/output comparison.

use ratatui::{
    layout::{
        Alignment,
        Constraint,
        Direction,
        Layout,
        Rect,
    },
    text::{
        Line,
        Span,
    },
    widgets::{
        Block,
        Borders,
        Paragraph,
        Wrap,
    },
    Frame,
};

use crate::tui::{
    state::AppState,
    theme::Theme,
};

pub struct DiffViewer;

impl DiffViewer {
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

        let input_text = app.editor.get_input();
        let input_title = match app.mode {
            crate::tui::state::app_state::Mode::Encode => "JSON Input",
            crate::tui::state::app_state::Mode::Decode => "TOON Input",
        };

        let input_lines: Vec<Line> = input_text
            .lines()
            .enumerate()
            .map(|(idx, line)| {
                Line::from(vec![
                    Span::styled(format!("{:4} ", idx + 1), theme.line_number_style()),
                    Span::styled(line, theme.normal_style()),
                ])
            })
            .collect();

        let input_para = Paragraph::new(input_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border_style(false))
                    .title(format!(" {input_title} ")),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(input_para, chunks[0]);

        let output_text = app.editor.get_output();
        let output_title = match app.mode {
            crate::tui::state::app_state::Mode::Encode => "TOON Output",
            crate::tui::state::app_state::Mode::Decode => "JSON Output",
        };

        let output_lines: Vec<Line> = output_text
            .lines()
            .enumerate()
            .map(|(idx, line)| {
                Line::from(vec![
                    Span::styled(format!("{:4} ", idx + 1), theme.line_number_style()),
                    Span::styled(line, theme.normal_style()),
                ])
            })
            .collect();

        let output_para = Paragraph::new(output_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border_style(false))
                    .title(format!(" {output_title} ")),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(output_para, chunks[1]);
    }
}
