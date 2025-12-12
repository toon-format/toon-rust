# rune-cros

Cross-format helpers for RUNE: small wrappers to convert between RUNE/JSON and other formats.

This crate is intended to be a compact utility for conversions and import/export pipelines.

Features:

- JSON is first-class (no feature flag): `json_to_rune_string`, `rune_string_to_json`, and helpers for serde_json::Value.
- `yaml`: YAML support via serde_yaml_ng
- `toml`: TOML support via toml
- `mpak`: MessagePack support via rmp-serde
- `arow`: Apache Arrow helpers (arrow2) — feature gated
- `prqt`: Parquet helpers (parquet2) — feature gated
- `npy`: NumPy `.npy` helpers — feature gated

Examples:

- `cargo test --manifest-path Cargo.toml --features "yaml toml mpak"` (run with combinations)
