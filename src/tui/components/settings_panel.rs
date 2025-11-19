//! Settings panel for configuring encode/decode options.

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

use crate::{
    tui::{
        state::AppState,
        theme::Theme,
    },
    types::{
        Delimiter,
        Indent,
        KeyFoldingMode,
        PathExpansionMode,
    },
};

pub struct SettingsPanel;

impl SettingsPanel {
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
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(3),
            ])
            .split(inner);

        let title = Paragraph::new(Line::from(Span::styled(
            format!("Current Mode: {}", app.mode.as_str()),
            theme.title_style(),
        )))
        .alignment(Alignment::Center);
        f.render_widget(title, chunks[0]);

        let mut items = vec![];

        items.push(ListItem::new(Line::from(Span::styled(
            "═══ Encode Settings (JSON → TOON) ═══",
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

        let Indent::Spaces(indent_spaces) = app.encode_options.indent;
        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Indentation:     ", theme.info_style()),
            Span::styled(format!("{indent_spaces} spaces"), theme.normal_style()),
            Span::styled("  [+/- to adjust]", theme.line_number_style()),
        ])));

        let fold_keys = match app.encode_options.key_folding {
            KeyFoldingMode::Off => "Off",
            KeyFoldingMode::Safe => "On (Safe)",
        };
        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Key Folding:     ", theme.info_style()),
            Span::styled(fold_keys, theme.normal_style()),
            Span::styled("  [Press 'f' to toggle]", theme.line_number_style()),
        ])));

        if app.encode_options.key_folding != KeyFoldingMode::Off {
            items.push(ListItem::new(Line::from(vec![
                Span::styled("  Flatten Depth:   ", theme.info_style()),
                Span::styled(
                    if app.encode_options.flatten_depth == usize::MAX {
                        "Unlimited".to_string()
                    } else {
                        format!("{}", app.encode_options.flatten_depth)
                    },
                    theme.normal_style(),
                ),
                Span::styled(
                    "  [[/] to adjust, [u] for unlimited]",
                    theme.line_number_style(),
                ),
            ])));
        }

        items.push(ListItem::new(Line::from("")));

        items.push(ListItem::new(Line::from(Span::styled(
            "═══ Decode Settings (TOON → JSON) ═══",
            theme.title_style(),
        ))));

        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Strict Mode:     ", theme.info_style()),
            Span::styled(
                if app.decode_options.strict {
                    "On"
                } else {
                    "Off"
                },
                theme.normal_style(),
            ),
            Span::styled("  [Press 's' to toggle]", theme.line_number_style()),
        ])));

        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Type Coercion:   ", theme.info_style()),
            Span::styled(
                if app.decode_options.coerce_types {
                    "On"
                } else {
                    "Off"
                },
                theme.normal_style(),
            ),
            Span::styled("  [Press 'c' to toggle]", theme.line_number_style()),
        ])));

        let expand_paths = match app.decode_options.expand_paths {
            PathExpansionMode::Off => "Off",
            PathExpansionMode::Safe => "On (Safe)",
        };
        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Path Expansion:  ", theme.info_style()),
            Span::styled(expand_paths, theme.normal_style()),
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
