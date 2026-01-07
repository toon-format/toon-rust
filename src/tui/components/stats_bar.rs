//! Statistics bar showing token and byte savings.

use ratatui::{
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::tui::{state::AppState, theme::Theme};

/// Statistics bar rendering.
///
/// # Examples
/// ```no_run
/// use ratatui::{backend::TestBackend, Terminal};
/// use toon_format::tui::{components::StatsBar, state::AppState, theme::Theme};
///
/// let backend = TestBackend::new(80, 24);
/// let mut terminal = Terminal::new(backend).unwrap();
/// let app = AppState::new();
/// let theme = Theme::default();
/// terminal
///     .draw(|f| StatsBar::render(f, f.area(), &app, &theme))
///     .unwrap();
/// ```
pub struct StatsBar;

impl StatsBar {
    /// Render the statistics bar.
    ///
    /// # Examples
    /// ```no_run
    /// use ratatui::{backend::TestBackend, Terminal};
    /// use toon_format::tui::{components::StatsBar, state::AppState, theme::Theme};
    ///
    /// let backend = TestBackend::new(80, 24);
    /// let mut terminal = Terminal::new(backend).unwrap();
    /// let app = AppState::new();
    /// let theme = Theme::default();
    /// terminal
    ///     .draw(|f| StatsBar::render(f, f.area(), &app, &theme))
    ///     .unwrap();
    /// ```
    pub fn render(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
        if let Some(ref stats) = app.stats {
            let spans = vec![
                Span::styled(" Stats: ", theme.title_style()),
                Span::raw("Tokens: "),
                Span::styled(
                    format!("{}→{}", stats.json_tokens, stats.toon_tokens),
                    theme.info_style(),
                ),
                Span::styled(
                    format!(" ({:.1}%)", stats.token_savings),
                    if stats.token_savings > 0.0 {
                        theme.success_style()
                    } else {
                        theme.error_style()
                    },
                ),
                Span::raw(" | Bytes: "),
                Span::styled(
                    format!("{}→{}", stats.json_bytes, stats.toon_bytes),
                    theme.info_style(),
                ),
                Span::styled(
                    format!(" ({:.1}%)", stats.byte_savings),
                    if stats.byte_savings > 0.0 {
                        theme.success_style()
                    } else {
                        theme.error_style()
                    },
                ),
            ];

            let line = Line::from(spans);
            let paragraph = Paragraph::new(line).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border_style(false))
                    .title(" Statistics "),
            );

            f.render_widget(paragraph, area);
        } else {
            let paragraph = Paragraph::new(Line::from(vec![Span::styled(
                " No statistics available yet ",
                theme.line_number_style(),
            )]))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border_style(false))
                    .title(" Statistics "),
            );

            f.render_widget(paragraph, area);
        }
    }
}
