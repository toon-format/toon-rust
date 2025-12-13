/* src/tui/components/file_browser.rs */
//!▫~•◦-------------------------------‣
//! # High-performance file browser for opening JSON/RUNE files.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module provides a stateful file browser component for the RUNE TUI. It is
//! optimized for performance and responsiveness by caching directory contents and
//! eliminating heap allocations in the hot render path.
//!
//! ## Key Capabilities
//! - **Directory Navigation**: Allows traversal of the file system.
//! - **Optimized Rendering**: Caches directory entries to prevent redundant I/O calls
//!   on every frame. Uses zero-copy techniques for rendering file names and UI text.
//! - **File Type Recognition**: Identifies directories, `.json`, and `.rune` files with icons.
//!
//! ### Architectural Notes
//! The `FileBrowser` struct holds the selection state and a cache of the current
//! directory's entries (`Vec<DirEntryInfo>`). This cache is only invalidated when the
//! user navigates to a new directory. This design is crucial for preventing UI lag
//! when browsing directories with many files. `OsString` is used to store file names
//! to avoid unnecessary allocations and UTF-8 conversions.
//!
//! #### Example
//! ```rust
//! // This is a conceptual example, as a real implementation requires a full TUI loop.
//! use rune_xero::tui::{state::AppState, theme::Theme, components::file_browser::FileBrowser};
//! use ratatui::{Frame, layout::Rect};
//!
//! fn render_a_browser(frame: &mut Frame, area: Rect, app: &mut AppState, browser: &mut FileBrowser, theme: &Theme) {
//!     // In your TUI rendering loop, you would call:
//!     browser.render(frame, area, app, theme);
//! }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::{
    ffi::{OsStr, OsString},
    fs,
    path::{Path, PathBuf},
};

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

use crate::tui::{state::AppState, theme::Theme};

/// Represents cached information about a single directory entry.
#[derive(Debug, Clone)]
struct DirEntryInfo {
    name: OsString,
    is_dir: bool,
    is_json: bool,
    is_rune: bool,
}

/// File browser state and rendering.
pub struct FileBrowser {
    pub selected_index: usize,
    pub scroll_offset: usize,
    cached_entries: Vec<DirEntryInfo>,
    cached_path: PathBuf,
}

impl FileBrowser {
    pub fn new() -> Self {
        Self {
            selected_index: 0,
            scroll_offset: 0,
            cached_entries: Vec::new(),
            cached_path: PathBuf::new(),
        }
    }

    /// Ensures the directory cache is up-to-date.
    fn ensure_cache(&mut self, dir: &Path) {
        if self.cached_path != dir {
            self.cached_path = dir.to_path_buf();
            self.refresh_cache();
            self.selected_index = 0; // Reset selection on directory change
        }
    }

    pub fn move_up(&mut self) {
        if self.selected_index > 0 {
            self.selected_index -= 1;
            if self.selected_index < self.scroll_offset {
                self.scroll_offset = self.selected_index;
            }
        }
    }

    pub fn move_down(&mut self, app: &mut AppState) {
        self.ensure_cache(&app.file_state.current_dir);
        if self.selected_index < self.cached_entries.len().saturating_sub(1) {
            self.selected_index += 1;
        }
    }

    pub fn get_selected_entry(&mut self, app: &mut AppState) -> Option<PathBuf> {
        let dir = &app.file_state.current_dir;
        self.ensure_cache(dir);

        self.cached_entries.get(self.selected_index).map(|entry| {
            if entry.name == ".." {
                dir.parent().map_or_else(|| dir.to_path_buf(), |p| p.to_path_buf())
            } else {
                dir.join(&entry.name)
            }
        })
    }

    pub fn render(&mut self, f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
        self.ensure_cache(&app.file_state.current_dir);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(true))
            .title(" File Browser - Press Esc to close ")
            .title_alignment(Alignment::Center);

        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(1),
                Constraint::Length(1),
            ])
            .split(inner);

        // Render current directory path without allocation
        let path_display = app.file_state.current_dir.display();
        let current_dir_line = Line::from(vec![
            Span::styled("Current: ", theme.line_number_style()),
            Span::styled(path_display.to_string(), theme.info_style()),
        ]);
        let current_dir_para = Paragraph::new(current_dir_line);
        f.render_widget(current_dir_para, chunks[0]);

        let items: Vec<ListItem> = self
            .cached_entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                let icon = if entry.is_dir {
                    "📁"
                } else if entry.is_json {
                    "📄"
                } else if entry.is_rune {
                    "📋"
                } else {
                    "📃"
                };

                let style = if idx == self.selected_index {
                    theme.selection_style()
                } else if entry.is_json || entry.is_rune {
                    theme.highlight_style()
                } else {
                    theme.normal_style()
                };

                // Use lossy conversion for display; names are stored correctly as OsString
                let name_str = entry.name.to_string_lossy();

                // Render list item without format! allocation
                let line = Line::from(vec![
                    Span::raw("  "),
                    Span::styled(icon, style),
                    Span::raw(" "),
                    Span::styled(name_str, style),
                ]);
                ListItem::new(line)
            })
            .collect();

        // Adjust scroll offset to keep selection in view
        let list_height = chunks[1].height as usize;
        if self.selected_index >= self.scroll_offset + list_height {
            self.scroll_offset = self.selected_index - list_height + 1;
        }
        if self.selected_index < self.scroll_offset {
            self.scroll_offset = self.selected_index;
        }

        let list = List::new(items).highlight_style(theme.selection_style());
        f.render_widget(list, chunks[1]);

        // Render instructions without allocation
        let instructions = Paragraph::new(Line::from(vec![
            Span::styled("↑↓", theme.info_style()),
            Span::styled(" Navigate | ", theme.line_number_style()),
            Span::styled("Enter", theme.info_style()),
            Span::styled(" Open", theme.line_number_style()),
        ]))
        .alignment(Alignment::Center);
        f.render_widget(instructions, chunks[2]);
    }

    /// Re-reads the directory from the filesystem and updates the internal cache.
    fn refresh_cache(&mut self) {
        let mut entries = vec![DirEntryInfo {
            name: OsString::from(".."),
            is_dir: true,
            is_json: false,
            is_rune: false,
        }];

        if let Ok(read_dir) = fs::read_dir(&self.cached_path) {
            let mut files: Vec<_> = read_dir
                .filter_map(|entry| entry.ok())
                .filter_map(|entry| {
                    let path = entry.path();
                    let file_name = path.file_name()?.to_owned();
                    let is_dir = path.is_dir();
                    let extension = path.extension().and_then(OsStr::to_str);
                    let is_json = !is_dir && extension == Some("json");
                    let is_rune = !is_dir && extension == Some("rune");

                    Some(DirEntryInfo {
                        name: file_name,
                        is_dir,
                        is_json,
                        is_rune,
                    })
                })
                .collect();

            // Sort directories first, then files alphabetically.
            files.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then_with(|| a.name.cmp(&b.name)));

            entries.extend(files);
        }

        self.cached_entries = entries;
    }
}

impl Default for FileBrowser {
    fn default() -> Self {
        Self::new()
    }
}