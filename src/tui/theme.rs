/* src/tui/theme.rs */
//!▫~•◦-------------------------------‣
//! # Rune color themes for the RUNE TUI.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module defines the `Theme` enum and provides a comprehensive set of
//! styling functions for the entire terminal user interface. The color palettes
//! are inspired by historical and modern interpretations of Viking Age aesthetics.
//!
//! ## Key Capabilities
//! - **Thematic Palettes**: Provides `Dark` (Runestone) and `Light` (Birch Bark) themes.
//! - **Style Composition**: Offers a suite of high-performance, zero-allocation
//!   functions that return `ratatui::style::Style` structs for all UI components.
//! - **Contextual Styling**: Methods like `border_style` return different styles
//!   based on application state (e.g., active vs. inactive panels).
//!
//! ### Architectural Notes
//! All style-generating functions are pure and return `Copy` types (`Color`, `Style`),
//! ensuring they can be called in the hot render loop without any performance penalty.
//! The color choices evoke a sense of authenticity, with reds for emphasis, deep
//! blues for importance, and natural tones for backgrounds.
//!
//! #### Example
//! ```rust
//! use rune_xero::tui::theme::Theme;
//! use ratatui::style::Style;
//!
//! let theme = Theme::Dark;
//! let active_border_style: Style = theme.border_style(true);
//! let normal_text_style: Style = theme.normal_style();
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use ratatui::style::{Color, Modifier, Style};

/// Available color themes, inspired by Viking Age aesthetics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Theme {
    #[default]
    /// Runestone: Dark stone, bright runes.
    Dark,
    /// Birch Bark: Light wood, dark carvings.
    Light,
}

impl Theme {
    /// Switches between the available themes.
    pub fn toggle(&self) -> Self {
        match self {
            Theme::Dark => Theme::Light,
            Theme::Light => Theme::Dark,
        }
    }

    // --- Core Palette ---

    pub fn background(&self) -> Color {
        match self {
            Theme::Dark => Color::Rgb(20, 20, 25), // Dark slate/stone
            Theme::Light => Color::Rgb(245, 240, 230), // Aged birch bark
        }
    }

    pub fn foreground(&self) -> Color {
        match self {
            Theme::Dark => Color::Rgb(220, 220, 220), // Off-white, like chalk
            Theme::Light => Color::Rgb(40, 35, 30),   // Dark wood carving
        }
    }

    pub fn border(&self) -> Color {
        match self {
            Theme::Dark => Color::Rgb(70, 80, 90), // Unfocused stone border
            Theme::Light => Color::Rgb(140, 130, 120), // Unfocused wood carving
        }
    }

    pub fn border_active(&self) -> Color {
        // Odin's Blue - a color of wisdom and importance.
        Color::Rgb(0, 110, 180)
    }

    pub fn title(&self) -> Color {
        match self {
            Theme::Dark => Color::Rgb(255, 200, 0), // Gilded gold
            Theme::Light => Color::Rgb(0, 90, 150),   // Deep blue ink
        }
    }

    // --- Semantic Colors ---

    pub fn success(&self) -> Color {
        // A hopeful, natural green (Malachite).
        Color::Rgb(0, 160, 110)
    }

    pub fn error(&self) -> Color {
        // Thor's Red - a powerful, unmissable color for warnings.
        Color::Rgb(210, 40, 40)
    }

    pub fn warning(&self) -> Color {
        // Ochre yellow, a common natural pigment.
        Color::Rgb(230, 160, 0)
    }

    pub fn info(&self) -> Color {
        // Azurite blue, a valuable pigment for important info.
        Color::Rgb(0, 130, 200)
    }

    pub fn highlight(&self) -> Color {
        match self {
            Theme::Dark => Color::Rgb(0, 80, 130), // Deep water blue
            Theme::Light => Color::Rgb(180, 210, 230), // Light sky blue
        }
    }

    pub fn selection(&self) -> Color {
        match self {
            Theme::Dark => Color::Rgb(55, 55, 65), // Darker selected stone
            Theme::Light => Color::Rgb(255, 230, 180), // Sun-bleached wood
        }
    }

    pub fn line_number(&self) -> Color {
        match self {
            Theme::Dark => Color::Rgb(90, 90, 90), // Faded carving
            Theme::Light => Color::Rgb(160, 150, 140), // Lighter carving
        }
    }

    // --- Style Compositions ---

    pub fn normal_style(&self) -> Style {
        Style::default().fg(self.foreground()).bg(self.background())
    }

    /// Gets the border style, highlighted if the component is active.
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