use ratatui::{
    layout::{Constraint, Direction, Layout, Margin, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap},
    Frame,
};

use crate::tui::state::{AppState, ReplLineKind};

pub struct ReplPanel;

impl ReplPanel {
    pub fn render(f: &mut Frame, area: Rect, app: &mut AppState) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(10),
                Constraint::Length(3),
            ])
            .split(area);

        Self::render_output(f, chunks[0], app);
        Self::render_input(f, chunks[1], app);
    }

    fn render_output(f: &mut Frame, area: Rect, app: &AppState) {
        let lines: Vec<Line> = app.repl.output
            .iter()
            .skip(app.repl.scroll_offset)
            .map(|line| {
                let style = match line.kind {
                    ReplLineKind::Prompt => Style::default().fg(Color::Cyan),
                    ReplLineKind::Success => Style::default().fg(Color::Green),
                    ReplLineKind::Error => Style::default().fg(Color::Red),
                    ReplLineKind::Info => Style::default().fg(Color::Yellow),
                };
                Line::from(Span::styled(&line.content, style))
            })
            .collect();

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(" REPL Session (Ctrl+R to toggle, Esc to close) ");

        let paragraph = Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false });

        f.render_widget(paragraph, area);

        if app.repl.output.len() > (area.height as usize - 2) {
            let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("↑"))
                .end_symbol(Some("↓"));
            
            let mut scrollbar_state = ScrollbarState::new(app.repl.output.len())
                .position(app.repl.scroll_offset);
            
            f.render_stateful_widget(
                scrollbar,
                area.inner(Margin {
                    vertical: 1,
                    horizontal: 0,
                }),
                &mut scrollbar_state,
            );
        }
    }

    fn render_input(f: &mut Frame, area: Rect, app: &AppState) {
        let prompt = Span::styled(
            "> ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );

        let input_text = Span::raw(&app.repl.input);
        let cursor = Span::styled("█", Style::default().fg(Color::White));

        let line = Line::from(vec![prompt, input_text, cursor]);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));

        let paragraph = Paragraph::new(line).block(block);

        f.render_widget(paragraph, area);
    }
}

