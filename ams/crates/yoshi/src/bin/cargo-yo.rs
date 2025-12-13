/* crates/yoshi/src/bin/cargo-yo.rs */
//! Cargo subcommand for automated fixes using Geoshi.
//!
//! # ArcMoon Studios – cargo-yo Tool
//!▫~•◦------------------------------------------------‣
//!
//! Streams `cargo check --message-format=json` diagnostics into `geoshi::AutofixEngine`
//! and applies machine-safe edits. Designed for tight IDE loops: run once for
//! suggestions or repeat for a handful of passes until clean.
//!
//! ### Key Capabilities
//! - Runs `cargo check` with JSON diagnostics.
//! - Plans safe, single-line, machine-applicable edits.
//! - Applies edits in-place (or suggest-only mode).
//! - Repeats for `--max-passes` to converge without manual intervention.
//!
//! ### Example
//! ```bash
//! # Plan and apply safe fixes
//! cargo yo
//!
//! # Suggest only
//! cargo yo --suggest-only
//!
//! # Pass extra args to cargo check
//! cargo yo -- --features gpu
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier: MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use cargo_metadata::Message;
use clap::{ArgAction, Parser};
use geoshi::{AutofixConfig, AutofixEngine};
use std::{
    env, io,
    path::PathBuf,
    process::{Command, Stdio},
};
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(author, version, about = "Automated fixer driven by Geoshi")]
struct Args {
    /// Do not write changes; only print planned edits.
    #[arg(long, action = ArgAction::SetTrue)]
    suggest_only: bool,
    /// Maximum passes to attempt.
    #[arg(long, default_value_t = 3)]
    max_passes: usize,
    /// Additional args forwarded to `cargo check` after `--`.
    #[arg(last = true)]
    cargo_args: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();

    let args = Args::parse();
    let root = env::current_dir()?;

    let mut passes = 0usize;
    loop {
        passes += 1;
        let (messages, status_success) = run_cargo_check(&args.cargo_args)?;
        let config = AutofixConfig {
            apply: !args.suggest_only,
            max_passes: args.max_passes,
        };
        let engine = AutofixEngine::new(PathBuf::from(&root), config);
        let edits = engine.plan(&messages[..]);

        if edits.is_empty() {
            if status_success {
                info!("No autofix suggestions; workspace is clean.");
                return Ok(());
            }
            warn!("Build failed without machine-applicable suggestions; exiting with failure.");
            std::process::exit(1);
        }

        if args.suggest_only {
            print_suggestions(&edits);
            return Ok(());
        }

        let outcome = engine.apply(&edits)?;
        info!(
            "Applied {} edits (skipped {}).",
            outcome.applied, outcome.skipped
        );
        if outcome.applied == 0 {
            warn!("No edits applied; stopping to avoid an infinite loop.");
            break;
        }

        if passes >= args.max_passes {
            warn!("Reached max passes ({})", args.max_passes);
            break;
        }
    }

    Ok(())
}

fn run_cargo_check(args: &[String]) -> Result<(Vec<Message>, bool), Box<dyn std::error::Error>> {
    let mut cmd = Command::new("cargo");
    cmd.arg("check")
        .arg("--message-format=json")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !stderr.trim().is_empty() {
        eprint!("{stderr}");
    }

    let mut messages = Vec::new();
    for line in stdout.lines() {
        if let Ok(message) = serde_json::from_str::<Message>(line) {
            if let Message::CompilerMessage(ref diag) = message
                && let Some(rendered) = &diag.message.rendered
            {
                print!("{rendered}");
            }
            messages.push(message);
        } else {
            // passthrough non-JSON output
            println!("{line}");
        }
    }

    Ok((messages, output.status.success()))
}

fn print_suggestions(edits: &[geoshi::PlannedEdit]) {
    println!("Planned edits (suggest-only mode):");
    for edit in edits {
        println!(
            "{}:{}..{} -> {:?}",
            edit.file.display(),
            edit.start,
            edit.end,
            edit.replacement
        );
    }
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(io::stderr)
        .try_init();
}
