//! Status bar showing mode, file, and key commands.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::tui::{state::AppState, theme::Theme};

/// Status bar rendering.
///
/// # Examples
/// ```no_run
/// use ratatui::{backend::TestBackend, Terminal};
/// use toon_format::tui::{components::StatusBar, state::AppState, theme::Theme};
///
/// let backend = TestBackend::new(80, 24);
/// let mut terminal = Terminal::new(backend).unwrap();
/// let app = AppState::new();
/// let theme = Theme::default();
/// terminal
///     .draw(|f| StatusBar::render(f, f.area(), &app, &theme))
///     .unwrap();
/// ```
pub struct StatusBar;

impl StatusBar {
    /// Render the status bar.
    ///
    /// # Examples
    /// ```no_run
    /// use ratatui::{backend::TestBackend, Terminal};
    /// use toon_format::tui::{components::StatusBar, state::AppState, theme::Theme};
    ///
    /// let backend = TestBackend::new(80, 24);
    /// let mut terminal = Terminal::new(backend).unwrap();
    /// let app = AppState::new();
    /// let theme = Theme::default();
    /// terminal
    ///     .draw(|f| StatusBar::render(f, f.area(), &app, &theme))
    ///     .unwrap();
    /// ```
    pub fn render(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(area);

        let mut left_spans = vec![];

        left_spans.push(Span::styled(
            format!("{} ", app.mode.short_name()),
            theme.info_style(),
        ));

        left_spans.push(Span::raw("| "));

        if let Some(ref path) = app.file_state.current_file {
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("Untitled");
            left_spans.push(Span::styled(file_name, theme.normal_style()));

            if app.file_state.is_modified {
                left_spans.push(Span::styled(" [Modified]", theme.warning_style()));
            }
        } else {
            left_spans.push(Span::styled("No file", theme.line_number_style()));
        }

        left_spans.push(Span::raw(" | "));

        if let Some(ref error) = app.error_message {
            left_spans.push(Span::styled(format!("✗ {error} "), theme.error_style()));
        } else if let Some(ref status) = app.status_message {
            left_spans.push(Span::styled(format!("✓ {status} "), theme.success_style()));
        } else {
            left_spans.push(Span::styled("Ready ", theme.normal_style()));
        }

        left_spans.push(Span::raw("| "));
        let theme_name = match theme {
            Theme::Dark => "Dark",
            Theme::Light => "Light",
        };
        left_spans.push(Span::styled(
            theme_name.to_string(),
            theme.line_number_style(),
        ));

        let left_line = Line::from(left_spans);
        let left_paragraph =
            Paragraph::new(left_line).block(Block::default().borders(Borders::ALL));

        let key_commands = vec![
            Span::styled("F1", theme.info_style()),
            Span::raw(" Help | "),
            Span::styled("Ctrl+C", theme.info_style()),
            Span::raw(" Quit"),
        ];

        let right_line = Line::from(key_commands);
        let right_paragraph = Paragraph::new(right_line)
            .block(Block::default().borders(Borders::ALL))
            .alignment(Alignment::Right);

        f.render_widget(left_paragraph, chunks[0]);
        f.render_widget(right_paragraph, chunks[1]);
    }
}
