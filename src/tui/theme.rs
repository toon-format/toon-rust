//! Color themes for the TUI.

use ratatui::style::{
    Color,
    Modifier,
    Style,
};

/// Available color themes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Theme {
    #[default]
    Dark,
    Light,
}

impl Theme {
    /// Switch between dark and light themes.
    pub fn toggle(&self) -> Self {
        match self {
            Theme::Dark => Theme::Light,
            Theme::Light => Theme::Dark,
        }
    }

    pub fn background(&self) -> Color {
        match self {
            Theme::Dark => Color::Black,
            Theme::Light => Color::White,
        }
    }

    pub fn foreground(&self) -> Color {
        match self {
            Theme::Dark => Color::White,
            Theme::Light => Color::Black,
        }
    }

    pub fn border(&self) -> Color {
        match self {
            Theme::Dark => Color::Cyan,
            Theme::Light => Color::Blue,
        }
    }

    pub fn border_active(&self) -> Color {
        match self {
            Theme::Dark => Color::Green,
            Theme::Light => Color::Green,
        }
    }

    pub fn title(&self) -> Color {
        match self {
            Theme::Dark => Color::Yellow,
            Theme::Light => Color::Blue,
        }
    }

    pub fn success(&self) -> Color {
        Color::Green
    }

    pub fn error(&self) -> Color {
        Color::Red
    }

    pub fn warning(&self) -> Color {
        Color::Yellow
    }

    pub fn info(&self) -> Color {
        Color::Cyan
    }

    pub fn highlight(&self) -> Color {
        match self {
            Theme::Dark => Color::Blue,
            Theme::Light => Color::LightBlue,
        }
    }

    pub fn selection(&self) -> Color {
        match self {
            Theme::Dark => Color::DarkGray,
            Theme::Light => Color::LightYellow,
        }
    }

    pub fn line_number(&self) -> Color {
        match self {
            Theme::Dark => Color::DarkGray,
            Theme::Light => Color::Gray,
        }
    }

    pub fn normal_style(&self) -> Style {
        Style::default().fg(self.foreground()).bg(self.background())
    }

    /// Get border style, highlighted if active.
    pub fn border_style(&self, active: bool) -> Style {
        Style::default().fg(if active {
            self.border_active()
        } else {
            self.border()
        })
    }

    pub fn title_style(&self) -> Style {
        Style::default()
            .fg(self.title())
            .add_modifier(Modifier::BOLD)
    }

    pub fn highlight_style(&self) -> Style {
        Style::default().fg(self.foreground()).bg(self.highlight())
    }

    pub fn selection_style(&self) -> Style {
        Style::default()
            .fg(self.foreground())
            .bg(self.selection())
            .add_modifier(Modifier::BOLD)
    }

    pub fn error_style(&self) -> Style {
        Style::default()
            .fg(self.error())
            .add_modifier(Modifier::BOLD)
    }

    pub fn success_style(&self) -> Style {
        Style::default()
            .fg(self.success())
            .add_modifier(Modifier::BOLD)
    }

    pub fn warning_style(&self) -> Style {
        Style::default()
            .fg(self.warning())
            .add_modifier(Modifier::BOLD)
    }

    pub fn info_style(&self) -> Style {
        Style::default().fg(self.info())
    }

    pub fn line_number_style(&self) -> Style {
        Style::default().fg(self.line_number())
    }
}
