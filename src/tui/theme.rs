//! Color themes for the TUI.

use ratatui::style::{Color, Modifier, Style};

/// Available color themes.
///
/// # Examples
/// ```
/// use toon_format::tui::theme::Theme;
///
/// let theme = Theme::Dark;
/// let _ = theme;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Theme {
    #[default]
    Dark,
    Light,
}

impl Theme {
    /// Switch between dark and light themes.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark.toggle();
    /// let _ = theme;
    /// ```
    pub fn toggle(&self) -> Self {
        match self {
            Theme::Dark => Theme::Light,
            Theme::Light => Theme::Dark,
        }
    }

    /// Background color for the theme.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.background();
    /// ```
    pub fn background(&self) -> Color {
        match self {
            Theme::Dark => Color::Black,
            Theme::Light => Color::White,
        }
    }

    /// Foreground color for the theme.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.foreground();
    /// ```
    pub fn foreground(&self) -> Color {
        match self {
            Theme::Dark => Color::White,
            Theme::Light => Color::Black,
        }
    }

    /// Border color for inactive blocks.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.border();
    /// ```
    pub fn border(&self) -> Color {
        match self {
            Theme::Dark => Color::Cyan,
            Theme::Light => Color::Blue,
        }
    }

    /// Border color for active blocks.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.border_active();
    /// ```
    pub fn border_active(&self) -> Color {
        match self {
            Theme::Dark => Color::Green,
            Theme::Light => Color::Green,
        }
    }

    /// Title color.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.title();
    /// ```
    pub fn title(&self) -> Color {
        match self {
            Theme::Dark => Color::Yellow,
            Theme::Light => Color::Blue,
        }
    }

    /// Success color.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.success();
    /// ```
    pub fn success(&self) -> Color {
        Color::Green
    }

    /// Error color.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.error();
    /// ```
    pub fn error(&self) -> Color {
        Color::Red
    }

    /// Warning color.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.warning();
    /// ```
    pub fn warning(&self) -> Color {
        Color::Yellow
    }

    /// Info color.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.info();
    /// ```
    pub fn info(&self) -> Color {
        Color::Cyan
    }

    /// Highlight color.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.highlight();
    /// ```
    pub fn highlight(&self) -> Color {
        match self {
            Theme::Dark => Color::Blue,
            Theme::Light => Color::LightBlue,
        }
    }

    /// Selection color.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.selection();
    /// ```
    pub fn selection(&self) -> Color {
        match self {
            Theme::Dark => Color::DarkGray,
            Theme::Light => Color::LightYellow,
        }
    }

    /// Line number color.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.line_number();
    /// ```
    pub fn line_number(&self) -> Color {
        match self {
            Theme::Dark => Color::DarkGray,
            Theme::Light => Color::Gray,
        }
    }

    /// Style for normal text.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.normal_style();
    /// ```
    pub fn normal_style(&self) -> Style {
        Style::default().fg(self.foreground()).bg(self.background())
    }

    /// Get border style, highlighted if active.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.border_style(true);
    /// ```
    pub fn border_style(&self, active: bool) -> Style {
        Style::default().fg(if active {
            self.border_active()
        } else {
            self.border()
        })
    }

    /// Style for titles.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.title_style();
    /// ```
    pub fn title_style(&self) -> Style {
        Style::default()
            .fg(self.title())
            .add_modifier(Modifier::BOLD)
    }

    /// Style for highlighted text.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.highlight_style();
    /// ```
    pub fn highlight_style(&self) -> Style {
        Style::default().fg(self.foreground()).bg(self.highlight())
    }

    /// Style for selection text.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.selection_style();
    /// ```
    pub fn selection_style(&self) -> Style {
        Style::default()
            .fg(self.foreground())
            .bg(self.selection())
            .add_modifier(Modifier::BOLD)
    }

    /// Style for error text.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.error_style();
    /// ```
    pub fn error_style(&self) -> Style {
        Style::default()
            .fg(self.error())
            .add_modifier(Modifier::BOLD)
    }

    /// Style for success text.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.success_style();
    /// ```
    pub fn success_style(&self) -> Style {
        Style::default()
            .fg(self.success())
            .add_modifier(Modifier::BOLD)
    }

    /// Style for warning text.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.warning_style();
    /// ```
    pub fn warning_style(&self) -> Style {
        Style::default()
            .fg(self.warning())
            .add_modifier(Modifier::BOLD)
    }

    /// Style for informational text.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.info_style();
    /// ```
    pub fn info_style(&self) -> Style {
        Style::default().fg(self.info())
    }

    /// Style for line numbers.
    ///
    /// # Examples
    /// ```
    /// use toon_format::tui::theme::Theme;
    ///
    /// let theme = Theme::Dark;
    /// let _ = theme.line_number_style();
    /// ```
    pub fn line_number_style(&self) -> Style {
        Style::default().fg(self.line_number())
    }
}
