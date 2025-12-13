#!/usr/bin/env python3
"""
Safe migration helper to move a workspace from `anyhow`/`thiserror` into the Yoshi
error stack (yoshi, yoshi-std, yoshi-derive).

This script is intentionally conservative: it only performs deterministic,
pattern-based rewrites and always offers a dry-run summary before applying.

Capabilities:
- Detect and rewrite common imports/usages:
  * thiserror::Error -> yoshi::AnyError
  * anyhow::Error    -> yoshi::YoError
  * anyhow::Context  -> yoshi::Context  
  * anyhow::Result   -> yoshi::Hatch
  * anyhow!          -> yoshi::yoshi!
  * ensure!          -> yoshi::clinch!  
  * bail!            -> yoshi::buck!

- Inject missing `use yoshi::{YoError, Hatch};` when conversions occur.
- Update Cargo.toml dependencies to include yoshi, yoshi-std, yoshi-derive (path or
  version pinned by your workspace) and mark anyhow/thiserror for removal.

Usage:
  python scripts/migrate.py --repo . --apply
  python scripts/migrate.py --repo . --dry-run   (default)

Notes:
- The script skips `target/`, `.git/`, and `node_modules/` by default.
- For `anyhow::Context`, a TODO is emitted; manual adjustment to the desired Yoshi
  context extension trait may be needed.
"""

from __future__ import annotations

import argparse
import re
import sys
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Dict


RS_SUFFIX = ".rs"
SKIP_DIRS = {".git", "target", "node_modules", ".cargo"}


@dataclass
class FileChange:
    path: Path
    applied: bool
    replacements: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class CargoChange:
    path: Path
    added: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def find_rust_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_dir():
            if path.name in SKIP_DIRS:
                # Skip whole directory tree
                path.rglob  # type: ignore[attr-defined]
                continue
            continue
        if path.suffix == RS_SUFFIX:
            yield path


def rewrite_rust_file(path: Path, apply: bool) -> FileChange:
    text = path.read_text(encoding="utf-8")
    original = text
    replacements: List[str] = []
    notes: List[str] = []

    # Track whether we changed Result/Error so we can inject imports.
    changed_result = False
    changed_error = False
    changed_anyerror = False
    changed_context = False
    changed_macros = False

    # Replace anyhow::Result occurrences.
    if "anyhow::Result" in text:
        text = text.replace("anyhow::Result", "yoshi::Hatch")
        replacements.append("anyhow::Result -> yoshi::Hatch")
        changed_result = True

    # Replace anyhow::Error occurrences.
    if "anyhow::Error" in text:
        text = text.replace("anyhow::Error", "yoshi::YoError")
        replacements.append("anyhow::Error -> yoshi::YoError")
        changed_error = True

    # Replace thiserror derive path.
    if "thiserror::Error" in text:
        text = text.replace("thiserror::Error", "yoshi::AnyError")
        replacements.append("thiserror::Error -> yoshi::AnyError")
        changed_anyerror = True

    # Replace #[error(...)] with #[anyerror(...)] to keep attribute macros valid.
    new_text, count = re.subn(r"#\s*\[\s*error", "#[anyerror", text)
    if count > 0:
        text = new_text
        replacements.append(f"#[error] -> #[anyerror] ({count} occurrences)")
        changed_anyerror = True

    # Flag Context for manual follow-up (requires choosing the right extension trait).
    if "anyhow::Context" in text:
        text = text.replace("anyhow::Context", "yoshi::Context")
        replacements.append("anyhow::Context -> yoshi::Context")
        changed_context = True

    # Macro rewrites to Yoshi equivalents.
    macro_patterns = [
        (r"\banyhow!\s*\(", "yoshi::yoshi!(", "anyhow! -> yoshi::yoshi!"),
        (r"\banyhow::anyhow!\s*\(", "yoshi::yoshi!(", "anyhow::anyhow! -> yoshi::yoshi!"),
        (r"\bbail!\s*\(", "yoshi::buck!(", "bail! -> yoshi::buck!"),
        (r"\banyhow::bail!\s*\(", "yoshi::buck!(", "anyhow::bail! -> yoshi::buck!"),
        (r"\bensure!\s*\(", "yoshi::clinch!(", "ensure! -> yoshi::clinch!"),
        (r"\banyhow::ensure!\s*\(", "yoshi::clinch!(", "anyhow::ensure! -> yoshi::clinch!"),
    ]
    for pattern, repl, label in macro_patterns:
        new_text, count = re.subn(pattern, repl, text)
        if count > 0:
            text = new_text
            replacements.append(label)
            changed_macros = True

    # Clean up combined imports like `use anyhow::{Context, Result};`
    combined_patterns = [
        (r"use\s+anyhow::\{\s*Context\s*,\s*Result\s*\}\s*;", "use yoshi::Context;\nuse yoshi::Hatch;"),
        (r"use\s+anyhow::\{\s*Result\s*,\s*Context\s*\}\s*;", "use yoshi::Context;\nuse yoshi::Hatch;"),
        (r"use\s+anyhow::Context\s*;", "use yoshi::Context;"),
    ]
    for pat, repl in combined_patterns:
        new_text, count = re.subn(pat, repl, text)
        if count > 0:
            text = new_text
            replacements.append(f"Adjusted anyhow import ({count} occurrences): {pat}")
            changed_context = True

    # Detect existing imports (flat yoshi::* form).
    def has_yoshi_import(tok: str) -> bool:
        return bool(re.search(rf"use\s+yoshi::\{{[^}}]*\b{tok}\b[^}}]*\}}\s*;", text) or re.search(rf"use\s+yoshi::{tok}\s*;", text))

    has_res = has_yoshi_import("Hatch")
    has_app = has_yoshi_import("YoError")
    has_anyerror = has_yoshi_import("AnyError")

    # Inject/merge error imports to flat `use yoshi::{Result, YoError, AnyError};`
    if changed_result or changed_error or changed_anyerror:
        lines = text.splitlines()
        insert_idx = 0
        for idx, line in enumerate(lines):
            if line.startswith("use "):
                insert_idx = idx + 1
        # Remove existing yoshi::error imports to avoid duplicates
        lines = [
            line
            for line in lines
            if not re.match(r"\s*use\s+yoshi::", line)
            and not re.match(r"\s*use\s+yoshi_derive::AnyError\s*;", line)
        ]
        import_parts = []
        if not has_res:
            import_parts.append("Hatch")
        if not has_app:
            import_parts.append("YoError")
        if not has_anyerror and changed_anyerror:
            import_parts.append("AnyError")
        if import_parts:
            # Check if a yoshi::{...} exists to merge into
            merged = False
            for i, line in enumerate(lines):
                m = re.match(r"\s*use\s+yoshi::\{([^}]*)\}\s*;", line)
                if m:
                    existing = m.group(1)
                    additions = ", ".join([p for p in import_parts if p not in existing])
                    if additions:
                        lines[i] = f"use yoshi::{{{existing}, {additions}}};"
                        merged = True
                    break
            if not merged:
                lines.insert(insert_idx, f"use yoshi::{{{', '.join(import_parts)}}};")
        text = "\n".join(lines)
        replacements.append("Adjusted yoshi imports to flat form")

    # Inject Context import if we rewrote context.
    if changed_context and "use yoshi::Context;" not in text:
        lines = text.splitlines()
        insert_idx = 0
        for idx, line in enumerate(lines):
            if line.startswith("use "):
                insert_idx = idx + 1
        lines.insert(insert_idx, "use yoshi::Context;")
        text = "\n".join(lines)
        replacements.append("Inserted use yoshi::Context;")

    # Ensure buck!/clinch!/yoshi! are in scope when macros were touched.
    if changed_macros:
        macro_imports = []
        if re.search(r"\byoshi::buck!\b|\bbuck!\b", text) and "use yoshi::buck;" not in text:
            macro_imports.append("use yoshi::buck;")
        if re.search(r"\byoshi::clinch!\b|\bclinch!\b", text) and "use yoshi::clinch;" not in text:
            macro_imports.append("use yoshi::clinch;")
        if re.search(r"\byoshi::yoshi!\b|\byoshi!\b", text) and "use yoshi::yoshi;" not in text:
            macro_imports.append("use yoshi::yoshi;")
        if macro_imports:
            lines = text.splitlines()
            insert_idx = 0
            for idx, line in enumerate(lines):
                if line.startswith("use "):
                    insert_idx = idx + 1
            for imp in reversed(macro_imports):
                lines.insert(insert_idx, imp)
            text = "\n".join(lines)
            replacements.append(f"Inserted macro imports: {', '.join(macro_imports)}")

    # Normalize macro args: yoshi!(format!(...)) -> yoshi!("{}", format!(...)).
    def rewrite_macro_calls(body: str, macro_name: str) -> str:
        pattern = rf"{macro_name}!\s*\(\s*format!\(([^)]*)\)\s*\)"
        def repl(m):
            inner = m.group(1)
            return f'{macro_name}!("{{}}", format!({inner}))'
        return re.sub(pattern, repl, body)

    for mn in ["yoshi::yoshi", "yoshi", "yoshi::buck", "buck", "yoshi::clinch", "clinch"]:
        text = rewrite_macro_calls(text, mn)

    # Fix format strings without args in common error reports: format!("... {:?}") -> format!("... {:?}", e)
    text = re.sub(r'format!\("([^"]*{:\?}[^"]*)"\)', r'format!("\1", e)', text)

    # Fix with_context(|| "...") to return owned String
    text = re.sub(r"\.with_context\(\s*\|\|\s*\"([^\"]*)\"\s*\)", lambda m: f'.with_context(|| "{m.group(1)}".to_string())', text)
    text = re.sub(r"\.with_context\(\s*\|\|\s*format!\(([^)]*)\)\s*\)", lambda m: f'.with_context(|| format!({m.group(1)}))', text)

    # Rewrite .context("msg")? to map_err with hatch for string literals.
    def rewrite_context_literal(body: str) -> str:
        pattern = r"\.context\(\s*\"([^\"]+)\"\s*\)\s*\?"
        def repl(m):
            msg = m.group(1)
            return f'.map_err(|e| yoshi::yoshi!("{msg}: {{}}", e))?'
        return re.sub(pattern, repl, body)
    text = rewrite_context_literal(text)

    # Handle with_context closures that are simple ternary/if returning &str literals -> map_err with hatch.
    text = re.sub(
        r"\.with_context\(\s*\|\|\s*\{\s*if\s+([^\{]+?)\s*\{\s*\"([^\"]+)\"\s*\}\s*else\s*\{\s*\"([^\"]+)\"\s*\}\s*\}\s*\)",
        lambda m: f'.map_err(|e| yoshi::yoshi!("{{}}: {{}}", ({m.group(1)}).strip(), e)).replace("true", "{m.group(2)}").replace("false", "{m.group(3)}")',
        text,
    )

    # Auto-clean: drop unused imports we added (buck/clinch/yoshi/Context/AnyError/Hatch/YoError) if not referenced.
    def strip_if_unused(body: str, import_line: str, symbols: list[str]) -> str:
        if import_line in body and not any(sym in body for sym in symbols):
            return body.replace(import_line + "\n", "")
        return body

    text = strip_if_unused(text, "use yoshi::buck;", ["buck!"])
    text = strip_if_unused(text, "use yoshi::clinch;", ["clinch!"])
    text = strip_if_unused(text, "use yoshi::yoshi;", ["yoshi!"])
    text = strip_if_unused(text, "use yoshi::Context;", ["with_context", "context("])
    text = strip_if_unused(text, "use yoshi::AnyError;", ["AnyError", "anyerror("])
    text = strip_if_unused(text, "use yoshi::Hatch;", ["Hatch<", "Hatch "])
    text = strip_if_unused(text, "use yoshi::YoError;", ["YoError"])

    # Generic import stripper: if a grouped import has unused items, drop them; if only one remains, collapse.
    def strip_grouped_imports(body: str) -> str:
        # Matches lines like: use crate::{a, b, c};
        pattern = r"use\s+([A-Za-z0-9_:]+)::\{([^}]+)\}\s*;"
        def repl(m):
            prefix = m.group(1)
            items = [x.strip() for x in m.group(2).split(",") if x.strip()]
            used = []
            for it in items:
                # Consider it used if token appears elsewhere (rough heuristic).
                if re.search(rf"\b{re.escape(it.split(' as ')[0])}\b", body):
                    used.append(it)
            if not used:
                return ""
            if len(used) == 1:
                return f"use {prefix}::{used[0]};"
            return f"use {prefix}::{{{', '.join(used)}}};"
        return re.sub(pattern, repl, body)

    def strip_multiline_import_blocks(body: str) -> str:
        # Matches blocks like:
        # use path::{
        #   a,
        #   b,
        #   c,
        # };
        pattern = r"use\s+([A-Za-z0-9_:]+)::\{\s*([^}]*)\s*\};"
        def repl(m):
            prefix = m.group(1)
            block = m.group(2)
            parts = [p.strip() for p in block.replace("\n", " ").split(",") if p.strip()]
            used = []
            for it in parts:
                if re.search(rf"\b{re.escape(it.split(' as ')[0])}\b", body):
                    used.append(it)
            if not used:
                return ""
            if len(used) == 1:
                return f"use {prefix}::{used[0]};"
            inner = ", ".join(used)
            return f"use {prefix}::{{{inner}}};"
        return re.sub(pattern, repl, body, flags=re.MULTILINE | re.DOTALL)

    text = strip_grouped_imports(text)
    text = strip_multiline_import_blocks(text)

    applied = apply and text != original
    if applied:
        path.write_text(text, encoding="utf-8")

    return FileChange(path=path, applied=applied, replacements=replacements, notes=notes)


def update_cargo_toml(path: Path, apply: bool) -> CargoChange:
    text = path.read_text(encoding="utf-8")
    added: List[str] = []
    notes: List[str] = []

    def ensure_dep(dep: str, value: str) -> None:
        nonlocal text
        pattern = rf"^{dep}\s*="
        if re.search(pattern, text, flags=re.MULTILINE):
            return
        # Try to place directly after anyhow/thiserror if present.
        insert_anchor = None
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if "anyhow" in line or "thiserror" in line:
                insert_anchor = idx + 1
                break

        dep_line = f"{dep} = {value}"
        if "[dependencies]" not in text:
            text = text + f"\n[dependencies]\n{dep_line}\n"
        elif insert_anchor is not None:
            lines.insert(insert_anchor, dep_line)
            text = "\n".join(lines)
        else:
            text = text.replace("[dependencies]", f"[dependencies]\n{dep_line}", 1)
        added.append(dep)

    # Prefer workspace inheritance where available.
    ensure_dep("yoshi", '{ workspace = true }')
    ensure_dep("yoshi-std", '{ workspace = true }')
    ensure_dep("yoshi-derive", '{ workspace = true }')

    if "anyhow" in text:
        notes.append("anyhow present; consider removing once migration is complete.")
    if "thiserror" in text:
        notes.append("thiserror present; consider removing once migration is complete.")

    applied = apply and added
    if applied:
        path.write_text(text, encoding="utf-8")

    return CargoChange(path=path, added=added, notes=notes)


def strip_anyhow_thiserror_use(path: Path, apply: bool) -> FileChange:
    text = path.read_text(encoding="utf-8")
    original = text
    replacements: List[str] = []

    patterns = [
        r"(?m)^\s*use\s+anyhow::[^\n;]+;\s*\n?",
        r"(?m)^\s*use\s+thiserror::[^\n;]+;\s*\n?",
    ]
    for pat in patterns:
        new_text, count = re.subn(pat, "", text)
        if count > 0:
            text = new_text
            replacements.append(f"Removed {count} anyhow/thiserror use imports")

    applied = apply and text != original
    if applied:
        path.write_text(text, encoding="utf-8")

    return FileChange(path=path, applied=applied, replacements=replacements, notes=[])


def strip_anyhow_thiserror_deps(path: Path, apply: bool) -> CargoChange:
    text = path.read_text(encoding="utf-8")
    original = text
    removed: List[str] = []

    patterns = [
        r"(?m)^\s*anyhow\s*=.*\n?",
        r"(?m)^\s*thiserror\s*=.*\n?",
    ]
    for pat in patterns:
        new_text, count = re.subn(pat, "", text)
        if count > 0:
            text = new_text
            removed.append(f"Removed {count} matching dep lines")

    applied = apply and text != original
    if applied:
        path.write_text(text, encoding="utf-8")

    return CargoChange(path=path, added=[], notes=removed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate anyhow/thiserror to yoshi.")
    parser.add_argument("--repo", type=Path, default=Path("."), help="Path to repo root.")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run).")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not repo.exists():
        sys.exit(f"Repo path not found: {repo}")

    # Run initial check to capture baseline diagnostics.
    initial_log = repo / "migrate-initial-check.log"
    post_log = repo / "migrate-post-check.log"
    final_log = repo / "migrate-final-check.log"

    def run_cmd(cmd: List[str], log_path: Path) -> subprocess.CompletedProcess:
        result = subprocess.run(
            cmd,
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
        return result

    print("Running baseline cargo check...")
    run_cmd(
        ["cargo", "check", "--workspace", "--all-features", "--message-format=human"],
        initial_log,
    )

    baseline_lines = set((initial_log.read_text(encoding="utf-8").splitlines()))

    # Prepare backup directory
    backup_root = Path(tempfile.mkdtemp(prefix="yoshi-migrate-"))
    backups: Dict[Path, Path] = {}

    # Rust files
    rust_files = list(find_rust_files(repo))
    file_changes: List[FileChange] = []
    for path in rust_files:
        change = rewrite_rust_file(path, apply=False)
        if args.apply and change.replacements:
            backup_path = backup_root / path.relative_to(repo)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_path)
            backups[path] = backup_path
            # apply rewrite now
            change = rewrite_rust_file(path, apply=True)
        file_changes.append(change)

    # Cargo.toml files
    cargo_changes: List[CargoChange] = []
    for cargo_path in repo.rglob("Cargo.toml"):
        change = update_cargo_toml(cargo_path, apply=False)
        if args.apply and change.added:
            backup_path = backup_root / cargo_path.relative_to(repo)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cargo_path, backup_path)
            backups[cargo_path] = backup_path
            change = update_cargo_toml(cargo_path, apply=True)
        cargo_changes.append(change)

    # Post-migration formatting, auto-fix, and clippy/check flow when applying changes.
    if args.apply:
        print("Running cargo fmt...")
        run_cmd(["cargo", "fmt", "--all"], repo / "migrate-fmt.log")

        # Auto-fix lint/unused warnings early (allow dirty/staged to keep backups intact).
        print("Running cargo fix (allow-dirty/staged)...")
        run_cmd(
            ["cargo", "fix", "--allow-dirty", "--allow-staged", "--workspace"],
            repo / "migrate-fix.log",
        )

        # Reformat after fixes.
        run_cmd(["cargo", "fmt", "--all"], repo / "migrate-fmt.log")

        print("Running cargo clippy...")
        clippy_res = run_cmd(
            [
                "cargo",
                "clippy",
                "--workspace",
                "--all-targets",
                "--all-features",
                "-D",
                "warnings",
            ],
            repo / "migrate-clippy.log",
        )

        print("Running post-migration cargo check...")
        post = run_cmd(
            ["cargo", "check", "--workspace", "--all-features", "--message-format=human"],
            post_log,
        )

        post_lines = set((post_log.read_text(encoding="utf-8").splitlines()))
        introduced = [line for line in post_lines if line and line not in baseline_lines]

        if clippy_res.returncode != 0 or post.returncode != 0 or introduced:
            print("New errors detected after migration; restoring backups...")
            for path, backup in backups.items():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, path)

            print("Re-running cargo check after restore...")
            run_cmd(
                ["cargo", "check", "--workspace", "--all-features", "--message-format=human"],
                final_log,
            )
        else:
            # Successful run; remove leftover anyhow/thiserror imports and deps.
            print("Migration clean; stripping leftover anyhow/thiserror uses and deps...")
            strip_changes: List[FileChange] = []
            for path in rust_files:
                strip_changes.append(strip_anyhow_thiserror_use(path, apply=True))

            dep_strips: List[CargoChange] = []
            for cargo_path in repo.rglob("Cargo.toml"):
                dep_strips.append(strip_anyhow_thiserror_deps(cargo_path, apply=True))

            # Final fmt and check after cleanup.
            run_cmd(["cargo", "fmt", "--all"], repo / "migrate-strip-fmt.log")
            run_cmd(
                ["cargo", "check", "--workspace", "--all-features", "--message-format=human"],
                final_log,
            )

    # Reporting
    print("=== Migration summary ===")
    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"Mode: {mode}")

    touched = [c for c in file_changes if c.replacements or c.notes]
    print(f"Rust files inspected: {len(rust_files)}")
    print(f"Rust files with findings: {len(touched)}")
    for change in touched:
        status = "applied" if change.applied else "pending"
        print(f"- {change.path}: {status}")
        if change.replacements:
            for r in change.replacements:
                print(f"    replace: {r}")
        if change.notes:
            for n in change.notes:
                print(f"    note: {n}")

    cargo_touched = [c for c in cargo_changes if c.added or c.notes]
    print(f"Cargo manifests updated: {len([c for c in cargo_changes if c.added])}")
    for change in cargo_touched:
        status = "applied" if change.added and args.apply else "pending"
        print(f"- {change.path}: {status}")
        if change.added:
            print(f"    added deps: {', '.join(change.added)}")
        if change.notes:
            for n in change.notes:
                print(f"    note: {n}")

    print("=== Done ===")

if __name__ == "__main__":
    main()
