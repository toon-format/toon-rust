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
    },
    Frame,
};

use super::{
    components::{
        DiffViewer,
        EditorComponent,
        FileBrowser,
        HelpScreen,
        HistoryPanel,
        ReplPanel,
        SettingsPanel,
        StatsBar,
        StatusBar,
    },
    state::AppState,
    theme::Theme,
};

/// Main render function - orchestrates all UI components.
pub fn render(f: &mut Frame, app: &mut AppState, file_browser: &mut FileBrowser) {
    let theme = app.theme;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(4),
            Constraint::Length(3),
        ])
        .split(f.area());

    render_header(f, chunks[0], app);

    // REPL takes full screen (except header)
    if app.repl.active {
        let repl_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(10)])
            .split(f.area())[1];

        ReplPanel::render(f, repl_area, app);
        return;
    } else if app.show_help {
        HelpScreen::render(f, chunks[1], &theme);
    } else if app.show_file_browser {
        file_browser.render(f, chunks[1], app, &theme);
    } else if app.show_history {
        HistoryPanel::render(f, chunks[1], app, &theme);
    } else if app.show_diff {
        DiffViewer::render(f, chunks[1], app, &theme);
    } else if app.show_settings {
        SettingsPanel::render(f, chunks[1], app, &theme);
    } else {
        let editor_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(48),
                Constraint::Percentage(4),
                Constraint::Percentage(48),
            ])
            .split(chunks[1]);

        EditorComponent::render(f, editor_chunks[0], editor_chunks[2], app, &theme);
        render_arrow(f, editor_chunks[1], app, &theme);
    }

    StatsBar::render(f, chunks[2], app, &theme);
    StatusBar::render(f, chunks[3], app, &theme);
}

/// Render conversion arrow and round-trip button between panels.
fn render_arrow(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let arrow_symbol = match app.mode {
        crate::tui::state::app_state::Mode::Encode => "→",
        crate::tui::state::app_state::Mode::Decode => "←",
    };

    let arrow_text = vec![
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled(arrow_symbol, theme.info_style())),
        Line::from(""),
        Line::from(Span::styled("Ctrl+B", theme.line_number_style())),
        Line::from(Span::styled("Round", theme.line_number_style())),
        Line::from(Span::styled("Trip", theme.line_number_style())),
    ];

    let arrow_para = Paragraph::new(arrow_text).alignment(Alignment::Center);

    f.render_widget(arrow_para, area);
}

/// Render header with title, mode, and current settings.
fn render_header(f: &mut Frame, area: Rect, app: &AppState) {
    let theme = app.theme;

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(34),
            Constraint::Percentage(33),
        ])
        .split(area);

    let title = Paragraph::new(Line::from(vec![
        Span::styled("📋 ", theme.normal_style()),
        Span::styled("TOON", theme.title_style()),
        Span::styled(" Format", theme.info_style()),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    let mode_text = Paragraph::new(Line::from(vec![Span::styled(
        app.mode.as_str(),
        theme.highlight_style(),
    )]))
    .alignment(Alignment::Center)
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(mode_text, chunks[1]);

    // Show relevant settings based on current mode
    let settings_line = match app.mode {
        crate::tui::state::app_state::Mode::Encode => {
            let delimiter = match app.encode_options.delimiter {
                crate::Delimiter::Comma => "comma",
                crate::Delimiter::Tab => "tab",
                crate::Delimiter::Pipe => "pipe",
            };

            let indent = match app.encode_options.indent {
                crate::Indent::Spaces(n) => format!("{n}sp"),
            };

            vec![
                Span::styled("Delim:", theme.line_number_style()),
                Span::styled(format!(" {delimiter}"), theme.info_style()),
                Span::styled(" | Indent:", theme.line_number_style()),
                Span::styled(format!(" {indent}"), theme.info_style()),
            ]
        }
        crate::tui::state::app_state::Mode::Decode => {
            let strict = if app.decode_options.strict {
                "on"
            } else {
                "off"
            };

            vec![
                Span::styled("Strict:", theme.line_number_style()),
                Span::styled(format!(" {strict}"), theme.info_style()),
            ]
        }
    };

    let settings = Paragraph::new(Line::from(settings_line))
        .alignment(Alignment::Right)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(settings, chunks[2]);
}
