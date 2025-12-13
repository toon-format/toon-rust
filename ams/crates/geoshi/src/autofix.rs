/* crates/geoshi/src/autofix.rs */
//! Automated fixer for compiler diagnostics.
//!
//! # ArcMoon Studios – Geoshi Autofix Module
//!▫~•◦------------------------------------------------‣
//!
//! Bridges rustc/cargo JSON diagnostics into machine-applicable edits. This module
//! focuses on single-line, machine-applicable suggestions (e.g., unused imports,
//! redundant qualifiers) to keep edits predictable and reversible. Complex or
//! multi-line suggestions are surfaced but not applied automatically.
//!
//! ### Key Capabilities
//! - **Diagnostic Ingestion:** Parses `cargo --message-format=json` output.
//! - **Safe Edits:** Applies only machine-applicable, single-line suggestions.
//! - **Batching:** Groups edits per file and applies in reverse order to avoid
//!   shifting spans.
//! - **Reporting:** Returns applied/skipped counts for upstream shells.
//!
//! ### Example
//! ```rust
//! use cargo_metadata::Message;
//! use geoshi::{AutofixConfig, AutofixEngine};
//! #
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let messages: Vec<Message> = Vec::new(); // filled from cargo check output
//! let engine = AutofixEngine::new(std::env::current_dir()?, AutofixConfig::default());
//! let edits = engine.plan(&messages);
//! let outcome = engine.apply(&edits)?;
//! println!("Applied {} edits", outcome.applied);
//! # Ok(()) }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use cargo_metadata::{
    Message,
    diagnostic::{Applicability, DiagnosticCode, DiagnosticSpan},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use thiserror::Error;
use tracing::{debug, warn};

/// Configuration for the autofix engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutofixConfig {
    /// Apply edits immediately. If false, only planning is performed.
    pub apply: bool,
    /// Maximum passes the caller will iterate; stored for reporting only.
    pub max_passes: usize,
}

impl Default for AutofixConfig {
    fn default() -> Self {
        Self {
            apply: true,
            max_passes: 3,
        }
    }
}

/// Machine-applicable edit planned from a diagnostic.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlannedEdit {
    pub file: PathBuf,
    pub start: usize,
    pub end: usize,
    pub replacement: String,
    pub label: Option<String>,
    pub code: Option<String>,
}

/// High-level suggestion containing the edits and the originating message.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutofixSuggestion {
    pub edits: Vec<PlannedEdit>,
    pub message: String,
    pub code: Option<String>,
}

/// Application outcome for a batch of edits.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AutofixOutcome {
    pub applied: usize,
    pub skipped: usize,
}

/// Errors that can occur during planning or application.
#[derive(Debug, Error)]
pub enum AutofixError {
    #[error("Failed to read source file {path}: {source}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Failed to write source file {path}: {source}")]
    WriteFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Invalid span for file {0}")]
    InvalidSpan(PathBuf),
}

/// Autofix engine that turns cargo diagnostics into safe edits.
#[derive(Clone, Debug)]
pub struct AutofixEngine {
    root: PathBuf,
    config: AutofixConfig,
}

impl AutofixEngine {
    /// Create a new engine rooted at the workspace path.
    pub fn new(root: impl AsRef<Path>, config: AutofixConfig) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            config,
        }
    }

    /// Plan edits from a sequence of cargo metadata messages.
    pub fn plan(&self, messages: &[Message]) -> Vec<PlannedEdit> {
        let mut edits = Vec::new();
        let mut file_cache: BTreeMap<PathBuf, String> = BTreeMap::new();

        for message in messages {
            let Message::CompilerMessage(diag) = message else {
                continue;
            };

            let code = diag.message.code.as_ref().map(code_to_string);
            let label = diag.message.message.clone();
            let target_src = diag.target.src_path.as_std_path();

            for span in diag.message.spans.iter() {
                if span.suggested_replacement.is_none() {
                    continue;
                }
                if span.line_start != span.line_end {
                    // Skip multi-line suggestions; keep changes surgical.
                    continue;
                }

                if !matches!(
                    span.suggestion_applicability,
                    Some(Applicability::MachineApplicable)
                ) {
                    continue;
                }

                let Some((start, end)) = self.span_range(span, &mut file_cache, Some(target_src))
                else {
                    warn!("Skipping span with missing source: {:?}", span.file_name);
                    continue;
                };

                let file = self
                    .resolve_source_path(span, Some(target_src))
                    .unwrap_or_else(|| self.root.join(&span.file_name));

                edits.push(PlannedEdit {
                    file,
                    start,
                    end,
                    replacement: span.suggested_replacement.clone().unwrap_or_default(),
                    label: Some(label.clone()),
                    code: code.clone(),
                });
            }
        }

        edits
    }

    /// Apply the given edits to disk, grouped per file.
    pub fn apply(&self, edits: &[PlannedEdit]) -> Result<AutofixOutcome, AutofixError> {
        let mut grouped: BTreeMap<PathBuf, Vec<&PlannedEdit>> = BTreeMap::new();
        for edit in edits {
            grouped.entry(edit.file.clone()).or_default().push(edit);
        }

        let mut outcome = AutofixOutcome::default();
        for (file, mut file_edits) in grouped {
            if !self.config.apply {
                outcome.skipped += file_edits.len();
                continue;
            }

            let source = fs::read_to_string(&file).map_err(|source| AutofixError::ReadFile {
                path: file.clone(),
                source,
            })?;

            file_edits.sort_by(|a, b| b.start.cmp(&a.start));

            let mut updated = source;
            for edit in file_edits {
                if edit.start > updated.len() || edit.end > updated.len() || edit.start > edit.end {
                    outcome.skipped += 1;
                    continue;
                }
                debug!("Applying edit {:?} {}..{}", file, edit.start, edit.end);
                updated.replace_range(edit.start..edit.end, &edit.replacement);
                outcome.applied += 1;
            }

            fs::write(&file, updated).map_err(|source| AutofixError::WriteFile {
                path: file.clone(),
                source,
            })?;
        }

        Ok(outcome)
    }

    fn resolve_source_path(
        &self,
        span: &DiagnosticSpan,
        target_src: Option<&Path>,
    ) -> Option<PathBuf> {
        let span_path = PathBuf::from(&span.file_name);
        if span_path.is_absolute() {
            return Some(span_path);
        }

        // Try relative to target source directory first.
        if let Some(parent) = target_src.and_then(Path::parent) {
            let cand = parent.join(&span_path);
            if cand.exists() {
                return Some(cand);
            }
        }

        let root_join = self.root.join(&span_path);
        Some(root_join)
    }

    fn span_range(
        &self,
        span: &DiagnosticSpan,
        cache: &mut BTreeMap<PathBuf, String>,
        target_src: Option<&Path>,
    ) -> Option<(usize, usize)> {
        let file_path = self
            .resolve_source_path(span, target_src)
            .unwrap_or_else(|| self.root.join(&span.file_name));

        let source = if let Some(src) = cache.get(&file_path) {
            src.clone()
        } else {
            let contents = fs::read_to_string(&file_path).ok()?;
            cache.insert(file_path.clone(), contents.clone());
            contents
        };

        single_line_span_offsets(span, &source)
    }
}

fn single_line_span_offsets(span: &DiagnosticSpan, source: &str) -> Option<(usize, usize)> {
    let target_line = span.line_start.checked_sub(1)?;
    let mut offset = 0usize;

    for (idx, line) in source.split_inclusive('\n').enumerate() {
        if idx == target_line {
            let start_col = span.column_start.saturating_sub(1);
            let end_col = span.column_end.saturating_sub(1);
            let start = offset + start_col.min(line.len());
            let end = offset + end_col.min(line.len());
            return Some((start, end));
        }
        offset += line.len();
    }

    None
}

fn code_to_string(code: &DiagnosticCode) -> String {
    code.code.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cargo_metadata::Message;
    use std::fs;
    use std::path::Path;
    use tempfile::tempdir;

    fn sample_message(file_path: &Path) -> Message {
        let path = file_path.display().to_string().replace('\\', "\\\\");

        let json = format!(
            r#"{{
  "reason": "compiler-message",
  "package_id": "geoshi 0.1.0 (path+file:///tmp/geoshi)",
  "target": {{"kind": ["lib"], "crate_types": ["lib"], "name": "geoshi", "src_path": "{path}", "edition": "2024", "doctest": true, "test": true}},
  "message": {{
    "message": "unused import: `std::fmt`",
    "code": {{"code": "unused-imports", "explanation": null}},
    "level": "warning",
    "spans": [{{
      "file_name": "{path}",
      "byte_start": 0,
      "byte_end": 12,
      "line_start": 1,
      "line_end": 1,
      "column_start": 1,
      "column_end": 13,
      "is_primary": true,
      "text": [],
      "suggested_replacement": "",
      "suggestion_applicability": "MachineApplicable",
      "label": null,
      "expansion": null
    }}],
    "children": [],
    "rendered": null
  }}
}}"#,
            path = path
        );

        serde_json::from_str(&json).expect("valid message json")
    }

    #[test]
    fn plans_and_applies_machine_applicable_fix() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("sample.rs");
        fs::write(&file_path, "use std::fmt;\nfn main() {}\n").unwrap();

        let messages = vec![sample_message(&file_path)];
        let engine = AutofixEngine::new(dir.path(), AutofixConfig::default());
        let edits = engine.plan(&messages);
        assert_eq!(edits.len(), 1);

        let outcome = engine.apply(&edits).unwrap();
        assert_eq!(outcome.applied, 1);

        let updated = fs::read_to_string(&file_path).unwrap();
        assert!(!updated.contains("use std::fmt"));
        assert!(updated.contains("fn main"));
    }

    #[test]
    fn skips_when_apply_disabled() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("sample.rs");
        fs::write(&file_path, "use std::fmt;\nfn main() {}\n").unwrap();

        let messages = vec![sample_message(&file_path)];
        let engine = AutofixEngine::new(
            dir.path(),
            AutofixConfig {
                apply: false,
                max_passes: 1,
            },
        );
        let edits = engine.plan(&messages);
        let outcome = engine.apply(&edits).unwrap();
        assert_eq!(outcome.skipped, edits.len());

        let updated = fs::read_to_string(&file_path).unwrap();
        assert!(updated.starts_with("use std::fmt"));
    }
}
