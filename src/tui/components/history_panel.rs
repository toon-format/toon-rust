//! Conversion history panel.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

use crate::tui::{state::AppState, theme::Theme};

pub struct HistoryPanel;

impl HistoryPanel {
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
            .constraints([Constraint::Length(2), Constraint::Min(10)])
            .split(inner);

        let title = Paragraph::new(Line::from(Span::styled(
            format!("Total conversions: {}", app.file_state.history.len()),
            theme.info_style(),
        )))
        .alignment(Alignment::Center);
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
                    let time_str = entry.timestamp.format("%H:%M:%S").to_string();
                    let file_str = entry
                        .input_file
                        .as_ref()
                        .and_then(|p| p.file_name())
                        .and_then(|n| n.to_str())
                        .unwrap_or("stdin");

                    ListItem::new(Line::from(vec![
                        Span::styled(format!("  {time_str} "), theme.line_number_style()),
                        Span::styled(format!("[{}] ", entry.mode), theme.info_style()),
                        Span::styled(file_str, theme.normal_style()),
                        Span::styled(
                            format!(" → {:.1}% saved", entry.token_savings),
                            if entry.token_savings > 0.0 {
                                theme.success_style()
                            } else {
                                theme.warning_style()
                            },
                        ),
                    ]))
                })
                .collect();

            let list = List::new(items);
            f.render_widget(list, chunks[1]);
        }
    }
}
