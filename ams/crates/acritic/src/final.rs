use clap::Parser;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(author, version, about = "Run full workspace health check and summarize to Final.md")]
struct Args {
    /// Skip benchmarks (useful when benches are slow or absent)
    #[arg(long)]
    skip_bench: bool,
    /// Skip clippy checks
    #[arg(long)]
    skip_clippy: bool,
    /// Push to remote if all steps succeed (requires clean git state)
    #[arg(long)]
    push: bool,
    /// Feature list to pass to cargo (comma-separated). Defaults to all features when omitted.
    #[arg(long)]
    features: Option<String>,
    /// Use all workspace features (defaults to true when no --features/--no-default-features provided)
    #[arg(long)]
    all_features: bool,
    /// Disable default features
    #[arg(long)]
    no_default_features: bool,
    /// Target directory to isolate builds/tests from the running binary
    #[arg(long, default_value = "target/final-check")]
    target_dir: PathBuf,
    /// Path to write the summary file
    #[arg(long, default_value = "Final.md")]
    output: PathBuf,
    /// Custom commit message for the final summary (used with --push)
    #[arg(long, default_value = "cargo final: Automated health check summary")]
    commit_message: String,
}

#[derive(Debug)]
struct StepResult {
    name: &'static str,
    success: bool,
    duration: Duration,
    stdout: String,
    stderr: String,
}

fn main() -> yoshi::Hatch<()> {
    let args = Args::parse();
    let mut results = Vec::new();
    let workspace_root = workspace_root()?;
    let target_dir = resolve_target_dir(&args.target_dir)?;
    let feature_flags = feature_flags(&args);

    results.push(run_step(
        "build",
        &["build", "--workspace", "--all-targets", "--release"],
        &workspace_root,
        &target_dir,
        &feature_flags,
    )?);

    results.push(run_step(
        "test",
        &["test", "--workspace", "--all-targets"],
        &workspace_root,
        &target_dir,
        &feature_flags,
    )?);

    if !args.skip_clippy {
        // For clippy, we need to insert the feature flags before the "--" separator
        // because "--" separates cargo options from clippy/rustc options
        let mut clippy_args = vec!["clippy", "--workspace", "--all-targets"];
        clippy_args.extend(feature_flags.iter().map(|s| s.as_str()));
        clippy_args.extend(&["--", "-D", "warnings"]);
        results.push(run_step(
            "clippy",
            &clippy_args,
            &workspace_root,
            &target_dir,
            &[], // Empty because we already included them above
        )?);
    }

    if !args.skip_bench {
        results.push(run_step(
            "bench",
            &["bench", "--workspace"],
            &workspace_root,
            &target_dir,
            &feature_flags,
        )?);
    }

    write_summary(&args.output, &results)?;

    if args.push {
        ensure_clean_git()?;
        git_commit_and_push(&args.output, &args.commit_message)?;
    }

    if results.iter().all(|r| r.success) {
        Ok(())
    } else {
        yoshi::buck!("One or more steps failed")
    }
}

fn run_step(
    name: &'static str,
    base_args: &[&str],
    workspace_root: &PathBuf,
    target_dir: &PathBuf,
    feature_flags: &[String],
) -> yoshi::Hatch<StepResult> {
    let start = Instant::now();
    let mut args: Vec<String> = base_args.iter().map(|s| s.to_string()).collect();
    args.extend(feature_flags.iter().cloned());
    let output = Command::new("cargo")
        .current_dir(workspace_root)
        .args(args)
        .env("CARGO_TARGET_DIR", target_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?
        .wait_with_output()?;
    let duration = start.elapsed();

    let success = output.status.success();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !success {
        eprintln!("[{name}] failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
    }

    Ok(StepResult {
        name,
        success,
        duration,
        stdout,
        stderr,
    })
}

fn write_summary(path: &PathBuf, results: &[StepResult]) -> yoshi::Hatch<()> {
    // Determine overall status
    let overall_success = results.iter().all(|r| r.success);
    let overall_status_emoji = if overall_success { "✅ PASS" } else { "❌ FAIL" };

    let mut lines = Vec::new();
    lines.push(format!("# Final Health Check: {}", overall_status_emoji));
    lines.push(String::new());

    // Add System Context
    lines.push("## System Context".to_string());
    lines.push(format!("- **OS:** {} {}", std::env::consts::OS, std::env::consts::ARCH));
    // Get Rust version
    let rustc_version_output = Command::new("rustc").arg("--version").output()?;
    lines.push(format!(
        "- **Rust Version:** {}",
        String::from_utf8_lossy(&rustc_version_output.stdout).trim()
    ));
    // Get Git commit hash
    let git_commit_output = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    lines.push(format!(
        "- **Git Commit:** {}",
        String::from_utf8_lossy(&git_commit_output.stdout).trim()
    ));
    lines.push(format!("- **Timestamp:** {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
    lines.push(String::new());

    // Add Step Results
    lines.push("## Step Results".to_string());
    for r in results {
        let status = if r.success { "✅" } else { "❌" };
        lines.push(format!(
            "- {} {} ({:.2?})",
            r.name.to_uppercase(),
            status,
            r.duration
        ));
    }

    lines.push(String::new());
    lines.push("## Details".to_string());
    for r in results {
        lines.push(format!("### {}", r.name.to_uppercase()));
        if !r.stdout.trim().is_empty() {
            lines.push("```\n".to_string() + r.stdout.trim() + "\n```");
        }
        if !r.stderr.trim().is_empty() {
            lines.push("stderr:".to_string());
            lines.push("```\n".to_string() + r.stderr.trim() + "\n```");
        }
        lines.push(String::new());
    }

    fs::write(path, lines.join("\n"))?;
    println!("Summary written to {}", path.display());
    Ok(())
}

fn resolve_target_dir(path: &PathBuf) -> yoshi::Hatch<PathBuf> {
    let target_dir = if path.is_absolute() {
        path.clone()
    } else {
        env::current_dir()?.join(path)
    };
    fs::create_dir_all(&target_dir)?;
    Ok(target_dir)
}

fn feature_flags(args: &Args) -> Vec<String> {
    // Preserve previous default (all features) unless user explicitly narrows.
    if args.all_features {
        return vec!["--all-features".to_string()];
    }

    let mut flags = Vec::new();

    if args.no_default_features {
        flags.push("--no-default-features".to_string());
    }

    if let Some(f) = &args.features {
        flags.push("--features".to_string());
        flags.push(f.clone());
    }

    if flags.is_empty() {
        flags.push("--all-features".to_string());
    }

    flags
}

fn workspace_root() -> yoshi::Hatch<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    Ok(manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .ok_or_else(|| yoshi::yoshi!("Unable to resolve workspace root from manifest"))?)
}

fn ensure_clean_git() -> yoshi::Hatch<()> {
    let status = Command::new("git")
        .args(["status", "--porcelain"])
        .output()?;
    let dirty = !status.stdout.is_empty();
    if dirty {
        yoshi::buck!("Git tree is dirty; not pushing. Please commit or clean first.");
    }
    Ok(())
}

fn git_commit_and_push(output_path: &PathBuf, commit_message: &str) -> yoshi::Hatch<()> {
    // Stage only the output file
    Command::new("git")
        .args(["add", output_path.to_str().ok_or_else(|| yoshi::yoshi!("Invalid output path"))?])
        .status()?
        .success()
        .then_some(())
        .ok_or_else(|| yoshi::yoshi!("git add failed"))?;

    Command::new("git")
        .args(["commit", "-m", commit_message])
        .status()?
        .success()
        .then_some(())
        .ok_or_else(|| yoshi::yoshi!("git commit failed"))?;

    Command::new("git")
        .arg("push")
        .status()?
        .success()
        .then_some(())
        .ok_or_else(|| yoshi::yoshi!("git push failed"))?;

    Ok(())
}
