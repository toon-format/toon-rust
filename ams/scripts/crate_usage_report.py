"""
Quick, dependency-oriented crate audit for Rust workspaces.

What it does (best-effort, heuristic):
1) Walks the repo to find Cargo.toml files (skips target/.git/node_modules).
2) Lists each crate's declared dependencies (normal/dev/build).
3) Scans the crate's Rust sources for references to each dependency name
   (`use dep::`, `dep::`, `dep!`, `extern crate dep`).
4) Flags dependencies that are never referenced in source (possible unused).
5) Reads Cargo.lock (if present) and reports crates that appear in multiple
   versions (candidates for consolidation).

Limitations:
- String/format usage, proc-macro-only deps, and feature-gated/optional deps
  may show up as false positives/negatives.
- Crate detection is purely textual; it does not parse Rust AST.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for older Pythons
    import tomli as tomllib  # type: ignore


EXCLUDE_DIRS = {".git", "target", "node_modules", ".idea", ".vscode", ".ruff_cache"}
RUST_FILE_SUFFIXES = {".rs"}
MANIFEST_NAME = "Cargo.toml"


@dataclass
class CrateDependencies:
    name: str
    manifest_path: Path
    root_dir: Path
    deps: Dict[str, str] = field(default_factory=dict)
    unused: List[str] = field(default_factory=list)


def find_manifests(root: Path) -> Iterable[Path]:
    for path in root.rglob(MANIFEST_NAME):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


def load_manifest(path: Path) -> Optional[dict]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - guardrails
        print(f"Warning: could not parse {path}: {exc}", file=sys.stderr)
        return None


def collect_dependencies(manifest: dict) -> Dict[str, str]:
    deps: Dict[str, str] = {}
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        table = manifest.get(section, {})
        for key, value in table.items():
            # Value can be a string or table; table may rename via "package"
            if isinstance(value, str):
                crate_name = key
            elif isinstance(value, dict):
                crate_name = value.get("package", key)
            else:
                crate_name = key
            deps[key] = crate_name
    return deps


def gather_rust_files(crate_root: Path) -> List[Path]:
    files: List[Path] = []
    for path in crate_root.rglob("*"):
        if path.is_dir():
            if path.name in EXCLUDE_DIRS:
                continue
            else:
                continue
        if path.suffix in RUST_FILE_SUFFIXES:
            if any(part in EXCLUDE_DIRS for part in path.parts):
                continue
            files.append(path)
    return files


def build_usage_index(files: List[Path]) -> str:
    # Concatenate contents for cheap substring scans; fine for typical crates.
    contents: List[str] = []
    for file in files:
        try:
            contents.append(file.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
    return "\n".join(contents)


def to_identifier(crate_name: str) -> str:
    # `foo-bar` in Cargo.toml is referred to as `foo_bar` in Rust.
    return crate_name.replace("-", "_")


def find_unused(crate: CrateDependencies, usage_blob: str) -> List[str]:
    unused: List[str] = []
    for dep_key, dep_real_name in crate.deps.items():
        ident = to_identifier(dep_key)
        alt_ident = to_identifier(dep_real_name)
        pattern = re.compile(rf"\b({re.escape(ident)}|{re.escape(alt_ident)})\b(?=\s*::|\s*!|\s*$)")
        if not pattern.search(usage_blob):
            unused.append(dep_key)
    return unused


def read_lockfile_versions(lockfile: Path) -> Dict[str, Set[str]]:
    versions: Dict[str, Set[str]] = {}
    if not lockfile.exists():
        return versions
    data = tomllib.loads(lockfile.read_text(encoding="utf-8"))
    for pkg in data.get("package", []):
        name = pkg.get("name")
        version = pkg.get("version")
        if not name or not version:
            continue
        versions.setdefault(name, set()).add(version)
    return versions


def report_duplicate_versions(versions: Dict[str, Set[str]]) -> List[Tuple[str, List[str]]]:
    dupes: List[Tuple[str, List[str]]] = []
    for name, vers in sorted(versions.items()):
        if len(vers) > 1:
            dupes.append((name, sorted(vers)))
    return dupes


def main() -> int:
    parser = argparse.ArgumentParser(description="Heuristic crate usage + duplicate version report.")
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.root.resolve()
    lockfile = root / "Cargo.lock"

    manifests = list(find_manifests(root))
    crates: List[CrateDependencies] = []
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        if not manifest or "package" not in manifest:
            continue
        crate_root = manifest_path.parent
        deps = collect_dependencies(manifest)
        crate = CrateDependencies(
            name=manifest["package"].get("name", manifest_path.parent.name),
            manifest_path=manifest_path,
            root_dir=crate_root,
            deps=deps,
        )
        files = gather_rust_files(crate_root)
        usage_blob = build_usage_index(files)
        crate.unused = find_unused(crate, usage_blob)
        crates.append(crate)

    print("=== Possible unused dependencies (heuristic) ===")
    if not crates:
        print("No crates found.")
    for crate in crates:
        if crate.unused:
            print(f"\n{crate.name} -- {crate.manifest_path}")
            for dep in sorted(crate.unused):
                print(f"  - {dep}")

    versions = read_lockfile_versions(lockfile)
    dupes = report_duplicate_versions(versions)
    print("\n=== Crates with multiple versions in Cargo.lock ===")
    if not dupes:
        print("None detected.")
    else:
        for name, vers in dupes:
            joined = ", ".join(vers)
            print(f"- {name}: {joined}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
