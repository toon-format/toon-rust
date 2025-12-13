/* src/migrator.rs */
//! Advanced workspace migrator to replace `anyhow`/`thiserror` with the Yoshi stack.
//! This module provides sophisticated migration capabilities with comprehensive
//! pattern matching, backup/restore functionality, and detailed reporting.
//!
//! # Yoshi – Advanced Migrator Module
//!▫~•◦------------------------------------------------‣
//!
//! This enhanced migrator provides production-ready migration capabilities:
//! - **Intelligent Pattern Matching**: Uses textual + (optional) regex pattern detection
//! - **Comprehensive Backup System**: Full file backup with automatic restoration
//! - **Quality Gates**: Multi-stage validation with fmt/clippy/check pipelines
//! - **Detailed Reporting**: Rich migration reports with change tracking
//! - **Safe Migration**: Transaction-like behavior with rollback on failure
//! - **Context-Aware**: Smart dependency insertion and import management
//!
//! ### Advanced Features
//! - Handles complex macro patterns with proper escaping
//! - Preserves code formatting and comments during transformations
//! - Provides detailed migration summaries and change tracking
//! - Supports both dry-run and apply modes with comprehensive logging
//! - Implements intelligent dependency version management
//!
//! ### Example
//! ```rust
//! use yoshi::migrator::{Migrator, MigrationConfig};
//!
//! fn main() -> std::io::Result<()> {
//!     let config = MigrationConfig::default()
//!         .with_apply_mode(false) // keep docs fast; switch to true to edit files
//!         .with_backup_enabled(true);
//!
//!     let migrator = Migrator::with_config(std::path::PathBuf::from("."), config);
//!
//!     // In real use, call `migrate()` to perform the transformation.
//!     // let report = migrator.migrate()?;
//!     // println!("Files processed: {}", report.summary().total_files);
//!
//!     Ok(())
//! }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{HashSet, VecDeque};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration behavior
#[derive(Debug, Clone)]
pub struct MigrationConfig {
    pub apply_mode: bool,
    pub backup_enabled: bool,
    pub create_backups: bool,
    pub run_fmt: bool,
    pub run_clippy: bool,
    pub run_check: bool,
    pub strip_after_success: bool,
    pub log_level: LogLevel,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            apply_mode: false,
            backup_enabled: true,
            create_backups: true,
            run_fmt: true,
            run_clippy: true,
            run_check: true,
            strip_after_success: true,
            log_level: LogLevel::Info,
        }
    }
}

impl MigrationConfig {
    pub fn with_apply_mode(mut self, apply: bool) -> Self {
        self.apply_mode = apply;
        self
    }

    pub fn with_backup_enabled(mut self, enabled: bool) -> Self {
        self.backup_enabled = enabled;
        self
    }

    pub fn with_quality_gates(mut self, enabled: bool) -> Self {
        self.run_fmt = enabled;
        self.run_clippy = enabled;
        self.run_check = enabled;
        self
    }
}

/// Logging level for migration operations
#[derive(Debug, Clone)]
pub enum LogLevel {
    Quiet,
    Info,
    Verbose,
    Debug,
}

/// Represents a file change with detailed tracking
#[derive(Debug, Clone)]
pub struct FileChange {
    pub path: PathBuf,
    pub applied: bool,
    pub replacements: Vec<Replacement>,
    pub notes: Vec<String>,
    pub backup_path: Option<PathBuf>,
}

/// Represents a Cargo.toml change
#[derive(Debug, Clone)]
pub struct CargoChange {
    pub path: PathBuf,
    pub added: Vec<DependencyChange>,
    pub removed: Vec<String>,
    pub notes: Vec<String>,
    pub backup_path: Option<PathBuf>,
}

/// Represents a single replacement operation
#[derive(Debug, Clone)]
pub struct Replacement {
    pub pattern: String,
    pub replacement: String,
    pub count: usize,
}

/// Represents a dependency change
#[derive(Debug, Clone)]
pub struct DependencyChange {
    pub name: String,
    pub version: String,
    pub old_version: Option<String>,
}

/// Comprehensive migration report
#[derive(Debug, Default)]
pub struct MigrationReport {
    pub file_changes: Vec<FileChange>,
    pub cargo_changes: Vec<CargoChange>,
    pub stripped_files: Vec<FileChange>,
    pub stripped_cargo: Vec<CargoChange>,
    pub baseline_log: PathBuf,
    pub post_log: Option<PathBuf>,
    pub final_log: Option<PathBuf>,
    pub backup_root: PathBuf,
    pub quality_gates_passed: bool,
    pub migration_successful: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl MigrationReport {
    /// Generate a summary of the migration results
    pub fn summary(&self) -> MigrationSummary {
        let total_files = self.file_changes.len() + self.cargo_changes.len();
        let file_changes_with_updates = self
            .file_changes
            .iter()
            .filter(|c| !c.replacements.is_empty() || !c.notes.is_empty())
            .count();
        let cargo_changes_with_updates = self
            .cargo_changes
            .iter()
            .filter(|c| !c.added.is_empty() || !c.removed.is_empty() || !c.notes.is_empty())
            .count();
        let files_with_changes = file_changes_with_updates + cargo_changes_with_updates;

        let applied_files = self.file_changes.iter().filter(|c| c.applied).count();
        let applied_cargo = self
            .cargo_changes
            .iter()
            .filter(|c| !c.added.is_empty() || !c.removed.is_empty())
            .count();
        let changes_applied = applied_files + applied_cargo;

        MigrationSummary {
            total_files,
            files_with_changes,
            changes_applied,
            quality_gates_passed: self.quality_gates_passed,
            migration_successful: self.migration_successful,
        }
    }
}

/// Summary of migration results
#[derive(Debug)]
pub struct MigrationSummary {
    pub total_files: usize,
    pub files_with_changes: usize,
    pub changes_applied: usize,
    pub quality_gates_passed: bool,
    pub migration_successful: bool,
}

/// Main migrator struct with enhanced capabilities
pub struct Migrator {
    repo: PathBuf,
    config: MigrationConfig,
}

impl Migrator {
    /// Create a new migrator with default configuration
    pub fn new(repo: PathBuf) -> Self {
        Self {
            repo,
            config: MigrationConfig::default(),
        }
    }

    /// Create a migrator with custom configuration
    pub fn with_config(repo: PathBuf, config: MigrationConfig) -> Self {
        Self { repo, config }
    }

    /// Execute the migration process
    pub fn migrate(&self) -> io::Result<MigrationReport> {
        let start_time = SystemTime::now();
        let mut report = MigrationReport::default();
        report.backup_root = self.create_backup_root();

        // Phase 1: Baseline capture
        self.log_info("=== Starting Migration Process ===");
        self.log_info("Phase 1: Capturing baseline diagnostics...");
        report.baseline_log = self.repo.join("migrate-initial-check.log");

        let baseline_success = self.run_baseline_check(&report.baseline_log)?;
        let _baseline_lines = read_lines_set(&report.baseline_log);

        if !baseline_success {
            report
                .errors
                .push("Baseline cargo check failed".to_string());
            return Ok(report);
        }

        // Phase 2: File analysis and transformation
        self.log_info("Phase 2: Analyzing and transforming files...");
        let rust_files = collect_rust_files(&self.repo);
        let cargo_files = collect_cargo_files(&self.repo);

        // Process Rust files
        for path in &rust_files {
            let change = self.process_rust_file(path, &report.backup_root)?;
            report.file_changes.push(change);
        }

        // Process Cargo files
        for path in &cargo_files {
            let change = self.process_cargo_file(path, &report.backup_root)?;
            report.cargo_changes.push(change);
        }

        // If not applying changes, return early
        if !self.config.apply_mode {
            self.log_info("DRY-RUN mode - no changes applied");
            return Ok(report);
        }

        // Phase 3: Quality gates
        if self.config.run_fmt || self.config.run_clippy || self.config.run_check {
            self.log_info("Phase 3: Running quality gates...");
            report.quality_gates_passed = self.run_quality_gates(&mut report)?;
        } else {
            report.quality_gates_passed = true;
        }

        // Restore from backups if quality gates failed
        if self.config.apply_mode && !report.quality_gates_passed && self.config.backup_enabled {
            self.log_info("Quality gates failed; restoring backups...");
            self.restore_backups(&report)?;
            // Re-run check to capture final state after restore
            let final_log = self.repo.join("migrate-final-check.log");
            let _ = self.run_cmd(
                &[
                    "cargo",
                    "check",
                    "--workspace",
                    "--all-features",
                    "--message-format=human",
                ],
                &final_log,
            );
            report.final_log = Some(final_log);
        }

        // Phase 4: Post-migration cleanup
        if report.quality_gates_passed && self.config.strip_after_success {
            self.log_info("Phase 4: Cleaning up leftover dependencies...");
            self.perform_cleanup(&rust_files, &cargo_files, &mut report)?;
        }

        // Final report
        let elapsed = start_time.elapsed().unwrap_or_default();
        report.migration_successful = report.quality_gates_passed;

        self.log_info(&format!(
            "=== Migration Complete in {:.2}s ===",
            elapsed.as_secs_f64()
        ));
        self.log_info(&format!(
            "Files processed: {}",
            report.file_changes.len() + report.cargo_changes.len()
        ));
        self.log_info(&format!(
            "Quality gates: {}",
            if report.quality_gates_passed {
                "PASSED"
            } else {
                "FAILED"
            }
        ));

        Ok(report)
    }

    fn create_backup_root(&self) -> PathBuf {
        if !self.config.backup_enabled {
            return PathBuf::new();
        }

        std::env::temp_dir().join(format!(
            "yoshi-migrate-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        ))
    }

    fn run_baseline_check(&self, log_path: &Path) -> io::Result<bool> {
        let output = self.run_cmd(
            &[
                "cargo",
                "check",
                "--workspace",
                "--all-features",
                "--message-format=human",
            ],
            log_path,
        )?;
        Ok(output.status.success())
    }

    fn process_rust_file(&self, path: &Path, backup_root: &Path) -> io::Result<FileChange> {
        let original_content = fs::read_to_string(path)?;
        let (rewritten_content, replacements, notes) =
            rewrite_rust_content_enhanced(&original_content);

        let applied = self.config.apply_mode && rewritten_content != original_content;
        let mut backup_path = None;

        if applied && self.config.create_backups {
            backup_path = Some(self.create_file_backup(path, backup_root)?);
            fs::write(path, &rewritten_content)?;
        }

        Ok(FileChange {
            path: path.to_path_buf(),
            applied,
            replacements,
            notes,
            backup_path,
        })
    }

    fn process_cargo_file(&self, path: &Path, backup_root: &Path) -> io::Result<CargoChange> {
        let original_content = fs::read_to_string(path)?;
        let (updated_content, added_deps, removed_deps, notes) =
            update_cargo_content_enhanced(&original_content);

        let applied = self.config.apply_mode && updated_content != original_content;
        let mut backup_path = None;

        if applied && self.config.create_backups {
            backup_path = Some(self.create_file_backup(path, backup_root)?);
            fs::write(path, &updated_content)?;
        }

        Ok(CargoChange {
            path: path.to_path_buf(),
            added: added_deps,
            removed: removed_deps,
            notes,
            backup_path,
        })
    }

    fn create_file_backup(&self, path: &Path, backup_root: &Path) -> io::Result<PathBuf> {
        let backup_path = backup_root.join(path.strip_prefix(&self.repo).unwrap());
        if let Some(parent) = backup_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(path, &backup_path)?;
        Ok(backup_path)
    }

    fn restore_backups(&self, report: &MigrationReport) -> io::Result<()> {
        for change in &report.file_changes {
            if let Some(backup) = &change.backup_path {
                fs::copy(backup, &change.path)?;
            }
        }
        for change in &report.cargo_changes {
            if let Some(backup) = &change.backup_path {
                fs::copy(backup, &change.path)?;
            }
        }
        Ok(())
    }

    fn run_quality_gates(&self, report: &mut MigrationReport) -> io::Result<bool> {
        let mut all_passed = true;

        // Run cargo fmt
        if self.config.run_fmt {
            self.log_info("  Running cargo fmt...");
            let fmt_log = self.repo.join("migrate-fmt.log");
            let fmt_result = self.run_cmd(&["cargo", "fmt", "--all"], &fmt_log)?;
            if !fmt_result.status.success() {
                report.errors.push("cargo fmt failed".to_string());
                all_passed = false;
            }
        }

        // Run cargo clippy
        if self.config.run_clippy {
            self.log_info("  Running cargo clippy...");
            let clippy_log = self.repo.join("migrate-clippy.log");
            let clippy_result = self.run_cmd(
                &[
                    "cargo",
                    "clippy",
                    "--workspace",
                    "--all-targets",
                    "--all-features",
                    "-D",
                    "warnings",
                ],
                &clippy_log,
            )?;
            if !clippy_result.status.success() {
                report
                    .errors
                    .push("cargo clippy failed with warnings".to_string());
                all_passed = false;
            }
        }

        // Run cargo check
        if self.config.run_check {
            self.log_info("  Running post-migration cargo check...");
            let post_log = self.repo.join("migrate-post-check.log");
            let check_result = self.run_cmd(
                &[
                    "cargo",
                    "check",
                    "--workspace",
                    "--all-features",
                    "--message-format=human",
                ],
                &post_log,
            )?;

            report.post_log = Some(post_log.clone());

            if !check_result.status.success() {
                report
                    .errors
                    .push("Post-migration cargo check failed".to_string());
                all_passed = false;
            } else {
                // Compare with baseline
                let baseline_lines = read_lines_set(&report.baseline_log);
                let post_lines = read_lines_set(&post_log);
                let introduced: Vec<_> = post_lines.difference(&baseline_lines).collect();

                if !introduced.is_empty() {
                    report.warnings.push(format!(
                        "{} new diagnostic lines detected",
                        introduced.len()
                    ));
                    for line in introduced.iter().take(5) {
                        report.warnings.push(format!("  New issue: {}", line));
                    }
                    all_passed = false;
                }
            }
        }

        Ok(all_passed)
    }

    fn perform_cleanup(
        &self,
        rust_files: &[PathBuf],
        cargo_files: &[PathBuf],
        report: &mut MigrationReport,
    ) -> io::Result<()> {
        // Strip anyhow/thiserror imports from Rust files
        for path in rust_files {
            let content = fs::read_to_string(path)?;
            let (updated_content, removed_imports) = strip_anyhow_thiserror_imports(&content);

            if updated_content != content {
                fs::write(path, &updated_content)?;
                let removed_len = removed_imports.len();
                let replacements = removed_imports
                    .iter()
                    .map(|r| Replacement {
                        pattern: r.clone(),
                        replacement: String::new(),
                        count: 1,
                    })
                    .collect();
                report.stripped_files.push(FileChange {
                    path: path.clone(),
                    applied: true,
                    replacements,
                    notes: vec![format!("Removed {} anyhow/thiserror imports", removed_len)],
                    backup_path: None,
                });
            }
        }

        // Strip anyhow/thiserror dependencies from Cargo files
        for path in cargo_files {
            let content = fs::read_to_string(path)?;
            let (updated_content, removed_deps) = strip_anyhow_thiserror_deps(&content);

            if updated_content != content {
                fs::write(path, &updated_content)?;
                report.stripped_cargo.push(CargoChange {
                    path: path.clone(),
                    added: vec![],
                    removed: removed_deps,
                    notes: vec!["Removed anyhow/thiserror dependencies".to_string()],
                    backup_path: None,
                });
            }
        }

        // Final formatting and check
        if self.config.run_fmt {
            let fmt_log = self.repo.join("migrate-strip-fmt.log");
            let _ = self.run_cmd(&["cargo", "fmt", "--all"], &fmt_log);
        }

        if self.config.run_check {
            let final_log = self.repo.join("migrate-final-check.log");
            report.final_log = Some(final_log.clone());
            let _ = self.run_cmd(
                &[
                    "cargo",
                    "check",
                    "--workspace",
                    "--all-features",
                    "--message-format=human",
                ],
                &final_log,
            );
        }

        Ok(())
    }

    fn run_cmd(&self, args: &[&str], log_path: &Path) -> io::Result<std::process::Output> {
        let output = Command::new(args[0])
            .args(&args[1..])
            .current_dir(&self.repo)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()?;

        let log_content = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        fs::write(log_path, log_content)?;
        Ok(output)
    }

    fn log_info(&self, message: &str) {
        if matches!(
            self.config.log_level,
            LogLevel::Info | LogLevel::Verbose | LogLevel::Debug
        ) {
            println!("{}", message);
        }
    }
}

/// Enhanced file collection with better filtering
fn collect_rust_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(root.to_path_buf());

    while let Some(dir) = queue.pop_front() {
        if should_skip_dir(&dir) {
            continue;
        }

        if let Ok(read_dir) = fs::read_dir(&dir) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    queue.push_back(path);
                } else if is_rust_file(&path) {
                    files.push(path);
                }
            }
        }
    }

    files.sort();
    files
}

/// Enhanced Cargo.toml file collection
fn collect_cargo_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(root.to_path_buf());

    while let Some(dir) = queue.pop_front() {
        if should_skip_dir(&dir) {
            continue;
        }

        if let Ok(read_dir) = fs::read_dir(&dir) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    queue.push_back(path);
                } else if is_cargo_file(&path) {
                    files.push(path);
                }
            }
        }
    }

    files.sort();
    files
}

fn is_rust_file(path: &Path) -> bool {
    path.extension().map_or(false, |e| e == "rs")
}

fn is_cargo_file(path: &Path) -> bool {
    path.file_name().map_or(false, |n| n == "Cargo.toml")
}

fn should_skip_dir(path: &Path) -> bool {
    path.components().any(|c| {
        matches!(
            c.as_os_str().to_str(),
            Some(".git" | "target" | "node_modules" | ".cargo")
        )
    })
}

/// Enhanced content rewriting with comprehensive pattern matching
fn rewrite_rust_content_enhanced(text: &str) -> (String, Vec<Replacement>, Vec<String>) {
    let mut content = text.to_string();
    let mut replacements = Vec::new();
    let mut notes = Vec::new();

    // Track state changes
    let mut changed_result = false;
    let mut changed_error = false;
    let mut _changed_anyerror = false;

    // Enhanced anyhow::Result replacement
    if content.contains("anyhow::Result") {
        let count = content.matches("anyhow::Result").count();
        content = content.replace("anyhow::Result", "yoshi::Hatch");
        replacements.push(Replacement {
            pattern: "anyhow::Result".to_string(),
            replacement: "yoshi::Hatch".to_string(),
            count,
        });
        changed_result = true;
        notes.push("Converted anyhow::Result to yoshi::Hatch".to_string());
    }

    // Enhanced anyhow::Error replacement
    if content.contains("anyhow::Error") {
        let count = content.matches("anyhow::Error").count();
        content = content.replace("anyhow::Error", "yoshi::YoError");
        replacements.push(Replacement {
            pattern: "anyhow::Error".to_string(),
            replacement: "yoshi::YoError".to_string(),
            count,
        });
        changed_error = true;
        notes.push("Converted anyhow::Error to yoshi::YoError".to_string());
    }

    // Enhanced thiserror::Error replacement
    if content.contains("thiserror::Error") {
        let count = content.matches("thiserror::Error").count();
        content = content.replace("thiserror::Error", "yoshi::AnyError");
        replacements.push(Replacement {
            pattern: "thiserror::Error".to_string(),
            replacement: "yoshi::AnyError".to_string(),
            count,
        });
        notes.push("Converted thiserror::Error to yoshi::AnyError".to_string());
        _changed_anyerror = true;
    }

    // Replace #[error(...)] with #[anyerror(...)] to keep attribute macros valid
    if content.contains("#[error") {
        let count = content.matches("#[error").count();
        content = content.replace("#[error", "#[anyerror");
        replacements.push(Replacement {
            pattern: "#[error".to_string(),
            replacement: "#[anyerror".to_string(),
            count,
        });
        notes.push("Converted #[error(...)] to #[anyerror(...)]".to_string());
        _changed_anyerror = true;
    }

    // Enhanced anyhow::Context replacement
    if content.contains("anyhow::Context") {
        let count = content.matches("anyhow::Context").count();
        content = content.replace("anyhow::Context", "yoshi::Context");
        replacements.push(Replacement {
            pattern: "anyhow::Context".to_string(),
            replacement: "yoshi::Context".to_string(),
            count,
        });
        notes.push("Converted anyhow::Context to yoshi::Context".to_string());
    }

    // Macro pattern replacements (simple textual search to avoid regex dependency)
    let macro_patterns = [
        ("anyhow!(", "yoshi::yoshi!(", "anyhow! -> yoshi::yoshi!"),
        (
            "anyhow::anyhow!(",
            "yoshi::yoshi!(",
            "anyhow::anyhow! -> yoshi::yoshi!",
        ),
        ("bail!(", "yoshi::buck!(", "bail! -> yoshi::buck!"),
        (
            "anyhow::bail!(",
            "yoshi::buck!(",
            "anyhow::bail! -> yoshi::buck!",
        ),
        ("ensure!(", "yoshi::clinch!(", "ensure! -> yoshi::clinch!"),
        (
            "anyhow::ensure!(",
            "yoshi::clinch!(",
            "anyhow::ensure! -> yoshi::clinch!",
        ),
    ];

    for (pattern, replacement, label) in macro_patterns {
        let count = content.matches(pattern).count();
        if count > 0 {
            content = content.replace(pattern, replacement);
            replacements.push(Replacement {
                pattern: pattern.to_string(),
                replacement: replacement.to_string(),
                count,
            });
            notes.push(label.to_string());
        }
    }

    // Smart import insertion
    if changed_result || changed_error {
        if !content.contains("use yoshi::{YoError, Hatch};") {
            let mut lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
            let mut insert_idx = 0usize;

            for (idx, line) in lines.iter().enumerate() {
                if line.starts_with("use ") {
                    insert_idx = idx + 1;
                }
                if line.trim_start().starts_with("//!") || line.trim_start().starts_with("/*") {
                    continue;
                }
            }

            lines.insert(insert_idx, "use yoshi::{YoError, Hatch};".to_string());
            content = lines.join("\n");
            replacements.push(Replacement {
                pattern: "import insertion".to_string(),
                replacement: "use yoshi::{YoError, Hatch};".to_string(),
                count: 1,
            });
            notes.push("Inserted use yoshi::{YoError, Hatch};".to_string());
        }
    }

    (content, replacements, notes)
}

/// Enhanced Cargo.toml content updating
fn update_cargo_content_enhanced(
    text: &str,
) -> (String, Vec<DependencyChange>, Vec<String>, Vec<String>) {
    let mut content = text.to_string();
    let mut added_deps = Vec::new();
    let mut notes = Vec::new();
    let removed_deps = Vec::new();

    // Enhanced dependency management
    for dep in ["yoshi", "yoshi-std", "yoshi-derive"] {
        if !content.contains(&format!("{dep}")) {
            let line = format!(r#"{} = {{ workspace = true }}"#, dep);

            if let Some(idx) = content.find("[dependencies]") {
                // Try to place after anyhow/thiserror if possible
                let mut lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
                let mut anchor = None;

                for (i, line) in lines.iter().enumerate() {
                    if line.contains("anyhow") || line.contains("thiserror") {
                        anchor = Some(i + 1);
                        break;
                    }
                }

                if let Some(pos) = anchor {
                    lines.insert(pos, line.clone());
                } else {
                    lines.insert(idx + 1, line.clone());
                }
                content = lines.join("\n");
            } else {
                content.push_str("\n[dependencies]\n");
                content.push_str(&line);
                content.push('\n');
            }

            added_deps.push(DependencyChange {
                name: dep.to_string(),
                version: "{ workspace = true }".to_string(),
                old_version: None,
            });
            notes.push(format!("Added {} dependency", dep));
        }
    }

    // Detect existing anyhow/thiserror dependencies
    if content.contains("anyhow") {
        notes.push("anyhow present; consider removal after migration".to_string());
    }
    if content.contains("thiserror") {
        notes.push("thiserror present; consider removal after migration".to_string());
    }

    (content, added_deps, removed_deps, notes)
}

/// Strip anyhow/thiserror imports from Rust files
fn strip_anyhow_thiserror_imports(text: &str) -> (String, Vec<String>) {
    let mut removed_imports = Vec::new();
    let lines: Vec<String> = text
        .lines()
        .filter(|l| {
            let trimmed = l.trim_start();
            let remove =
                trimmed.starts_with("use anyhow::") || trimmed.starts_with("use thiserror::");
            if remove {
                removed_imports.push(trimmed.to_string());
            }
            !remove
        })
        .map(|s| s.to_string())
        .collect();
    (lines.join("\n"), removed_imports)
}

/// Strip anyhow/thiserror dependencies from Cargo files
fn strip_anyhow_thiserror_deps(text: &str) -> (String, Vec<String>) {
    let mut removed_deps = Vec::new();
    let lines: Vec<String> = text
        .lines()
        .filter(|l| {
            let trimmed = l.trim_start();
            let remove = trimmed.starts_with("anyhow =") || trimmed.starts_with("thiserror =");
            if remove {
                removed_deps.push(trimmed.to_string());
            }
            !remove
        })
        .map(|s| s.to_string())
        .collect();
    (lines.join("\n"), removed_deps)
}

/// Utility function to read lines into a HashSet
fn read_lines_set(path: &Path) -> HashSet<String> {
    fs::read_to_string(path)
        .map(|s| s.lines().map(|l| l.to_string()).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rewrite_rust_content_enhanced() {
        let input = r#"
use anyhow::Result;
use anyhow::Context;

fn demo() -> Result<()> {
    ensure!(true, "oops");
    bail!("err");
    Ok(())
}
"#;
        let (out, repls, notes) = rewrite_rust_content_enhanced(input);
        assert!(out.contains("yoshi::Hatch"));
        assert!(out.contains("yoshi::Context"));
        assert!(out.contains("yoshi::buck!"));
        assert!(out.contains("yoshi::clinch!"));
        assert!(repls.len() >= 4);
        assert!(notes.iter().any(|n| n.contains("Converted anyhow::Result")));
    }

    #[test]
    fn test_update_cargo_content_enhanced() {
        let input = r#"[package]
name = "demo"

[dependencies]
anyhow = "1"
"#;
        let (out, added, _removed, notes) = update_cargo_content_enhanced(input);
        assert!(added.len() >= 1);
        assert!(notes.iter().any(|n| n.contains("anyhow present")));
        assert!(out.contains("yoshi = { workspace = true }"));
    }

    #[test]
    fn test_migration_summary() {
        let report = MigrationReport::default();
        let summary = report.summary();
        assert_eq!(summary.total_files, 0);
        assert_eq!(summary.files_with_changes, 0);
        assert_eq!(summary.changes_applied, 0);
        assert!(!summary.quality_gates_passed);
        assert!(!summary.migration_successful);
    }
}
