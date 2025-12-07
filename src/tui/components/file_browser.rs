//! File browser for opening JSON/TOON files.

use std::fs;

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};

use crate::tui::{state::AppState, theme::Theme};

/// File browser state and rendering.
pub struct FileBrowser {
    pub selected_index: usize,
    pub scroll_offset: usize,
}

impl FileBrowser {
    pub fn new() -> Self {
        Self {
            selected_index: 0,
            scroll_offset: 0,
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

    pub fn move_down(&mut self, max: usize) {
        if self.selected_index < max.saturating_sub(1) {
            self.selected_index += 1;
        }
    }

    pub fn get_selected_entry(&self, dir: &std::path::Path) -> Option<std::path::PathBuf> {
        let entries = self.get_directory_entries(dir);
        if self.selected_index < entries.len() {
            let (name, _is_dir, _, _) = &entries[self.selected_index];
            if name == ".." {
                dir.parent().map(|p| p.to_path_buf())
            } else {
                Some(dir.join(name))
            }
        } else {
            None
        }
    }

    pub fn get_entry_count(&self, dir: &std::path::Path) -> usize {
        self.get_directory_entries(dir).len()
    }

    pub fn render(&mut self, f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
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
                Constraint::Length(2),
                Constraint::Min(10),
                Constraint::Length(3),
            ])
            .split(inner);

        let current_dir = Paragraph::new(Line::from(vec![
            Span::styled("Current: ", theme.line_number_style()),
            Span::styled(
                app.file_state.current_dir.display().to_string(),
                theme.info_style(),
            ),
        ]));
        f.render_widget(current_dir, chunks[0]);

        let entries = self.get_directory_entries(&app.file_state.current_dir);
        let items: Vec<ListItem> = entries
            .iter()
            .enumerate()
            .map(|(idx, (name, is_dir, is_json, is_toon))| {
                let icon = if *is_dir {
                    "📁"
                } else if *is_json {
                    "📄"
                } else if *is_toon {
                    "📋"
                } else {
                    "📃"
                };

                let style = if idx == self.selected_index {
                    theme.selection_style()
                } else if *is_json || *is_toon {
                    theme.highlight_style()
                } else {
                    theme.normal_style()
                };

                ListItem::new(Line::from(vec![
                    Span::styled(format!("  {icon} "), style),
                    Span::styled(name, style),
                ]))
            })
            .collect();

        let list = List::new(items);
        f.render_widget(list, chunks[1]);

        let instructions = Paragraph::new(Line::from(vec![
            Span::styled("↑↓", theme.info_style()),
            Span::styled(" Navigate | ", theme.line_number_style()),
            Span::styled("Enter", theme.info_style()),
            Span::styled(" Open | ", theme.line_number_style()),
            Span::styled("Space", theme.info_style()),
            Span::styled(" Select | ", theme.line_number_style()),
            Span::styled("Esc", theme.info_style()),
            Span::styled(" Close", theme.line_number_style()),
        ]))
        .alignment(Alignment::Center);
        f.render_widget(instructions, chunks[2]);
    }

    fn get_directory_entries(&self, dir: &std::path::Path) -> Vec<(String, bool, bool, bool)> {
        let mut entries = vec![("..".to_string(), true, false, false)];

        if let Ok(read_dir) = fs::read_dir(dir) {
            let mut files: Vec<_> = read_dir
                .filter_map(|entry| entry.ok())
                .filter_map(|entry| {
                    let path = entry.path();
                    let name = path.file_name()?.to_str()?.to_string();
                    let is_dir = path.is_dir();
                    let is_json =
                        !is_dir && path.extension().and_then(|e| e.to_str()) == Some("json");
                    let is_toon =
                        !is_dir && path.extension().and_then(|e| e.to_str()) == Some("toon");
                    Some((name, is_dir, is_json, is_toon))
                })
                .collect();

            files.sort_by(|a, b| {
                if a.1 == b.1 {
                    a.0.cmp(&b.0)
                } else {
                    b.1.cmp(&a.1)
                }
            });

            entries.extend(files);
        }

        entries
    }
}

impl Default for FileBrowser {
    fn default() -> Self {
        Self::new()
    }
}
