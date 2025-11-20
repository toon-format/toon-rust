//! Help screen showing keyboard shortcuts.

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
        List,
        ListItem,
        Paragraph,
    },
    Frame,
};

use crate::tui::{
    keybindings::KeyBindings,
    theme::Theme,
};

pub struct HelpScreen;

impl HelpScreen {
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
                Constraint::Length(5),
            ])
            .split(inner);

        let title = Paragraph::new(vec![
            Line::from(Span::styled(
                "TOON Format - Interactive TUI",
                theme.title_style(),
            )),
            Line::from(Span::styled(
                "Token-Oriented Object Notation",
                theme.info_style(),
            )),
        ])
        .alignment(Alignment::Center);
        f.render_widget(title, chunks[0]);

        let shortcuts = KeyBindings::shortcuts();
        let items: Vec<ListItem> = shortcuts
            .iter()
            .map(|(key, desc)| {
                ListItem::new(Line::from(vec![
                    Span::styled(format!("  {key:18} "), theme.info_style()),
                    Span::styled(*desc, theme.normal_style()),
                ]))
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
                "TOON is a compact, human-readable format for passing structured data to LLMs",
                theme.normal_style(),
            )),
            Line::from(vec![
                Span::styled("Repository: ", theme.line_number_style()),
                Span::styled("github.com/toon-format/toon-rust", theme.info_style()),
            ]),
        ])
        .alignment(Alignment::Center);
        f.render_widget(footer, chunks[2]);
    }
}
