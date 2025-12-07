# Changelog

<!-- markdownlint-disable MD024 -->
<!--
  Disabling the following rules:
  - MD024/no-duplicate-heading: Multiple headings with the same content
-->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Performance

- **Hydron Geometry Optimizations**: Addressed performance regressions in E8 geometric operations
  - Spherical entropy normalization: -67-68% improvement by eliminating vector allocation in single-pass computation
  - GF8 norm squared: Optimized scalar implementation for small vector operations
  - GF8 matrix-vector: Added AVX SIMD acceleration for matrix-vector multiplication
  - Hyperbolic geometry: Streamlined scalar fallbacks to reduce redundant calculations

## [Unreleased] - 2025-11-26

### Added

- document the TOON -> RUNE fork direction, highlighting the root-centric semantic operators, CLI/TUI supersets, and E8 geometry focus described in the README
- note that the project now exposes the `rune` module, the `rune` CLI, and extends the existing TOON workflow with geosemantic root-specific notation, prefixes, and glyph operators

### Other

- reinforce the attribution for both TOON and RUNE contributors and offer the new guiding philosophy for semantic flow in the ecosystem

---

## [0.0.1](https://github.com/toon-format/toon-rust/compare/v0.3.6...v0.3.7) - 2025-11-21

### Added

- add the new `src/rune` workspace with Hydron geometry helpers for GF8, spherical, and hyperbolic routines
- extend the TUI so rune can be selected as an execution target and load rune-specific commands

### Changed

- align documentation and CLI tooling with the new rune workflow so the TUI can operate rune through the same interface

---

## [0.4.0](https://github.com/toon-format/toon-rust/compare/v0.3.7...v0.4.0) - 2025-11-25

### Added

- update to TOON v3.0 spec (breaking change) ([#36](https://github.com/toon-format/toon-rust/pull/36))

### Fixed

- *(decode)* Handle unquoted strings with parentheses and preserve spacing ([#34](https://github.com/toon-format/toon-rust/pull/34))

### Other

- update TUI documentation ([#31](https://github.com/toon-format/toon-rust/pull/31))

## [0.3.6](https://github.com/toon-format/toon-rust/compare/v0.3.5...v0.3.6) - 2025-11-20

### Added

- implement tui with repl  ([#29](https://github.com/toon-format/toon-rust/pull/29))

## [0.3.5](https://github.com/toon-format/toon-rust/compare/v0.3.4...v0.3.5) - 2025-11-18

### Other

- incorrect delimiter parsing for non-active delimiters in arrays ([#24](https://github.com/toon-format/toon-rust/pull/24))

## [0.3.4](https://github.com/toon-format/toon-rust/compare/v0.3.3...v0.3.4) - 2025-11-17

### Other

- Update PR template and contributing guide for Rust ([#25](https://github.com/toon-format/toon-rust/pull/25))

## [0.3.3](https://github.com/toon-format/toon-rust/compare/v0.3.2...v0.3.3) - 2025-11-14

### Added

- implement generic encode and decode ([#22](https://github.com/toon-format/toon-rust/pull/22))

## [0.3.2](https://github.com/toon-format/toon-rust/compare/v0.3.1...v0.3.2) - 2025-11-13

### Other

- standardize license holders

## [0.3.1](https://github.com/toon-format/toon-rust/compare/v0.3.0...v0.3.1) - 2025-11-12

### Fixed

- *(decode)* prevent sibling fields from being added to nested objects
- *(encode)* correct indentation for nested objects in list items

### Other

- update inline doc
- update changelog

## [0.3.0](https://github.com/toon-format/toon-rust/compare/v0.2.4...v0.3.0) - 2025-11-11

### Added

- implement key folds and expansion in cli
- [**breaking**] implement TOON Spec v2.0 with v1.5 optional features

### Other

- update readme
- update readme

## [0.2.4](https://github.com/toon-format/toon-rust/compare/v0.2.3...v0.2.4) - 2025-11-11

### Fixed

- encoder handling arrays, tabular rows and objects
- validation and parser to handle all the edge cases

### Other

- update ci
- cargo lint
- add fixtures test
- update test assertions
- update string utils
- add spec fixtures

## [0.2.3](https://github.com/toon-format/toon-rust/compare/v0.2.2...v0.2.3) - 2025-11-07

### Fixed

- readme license section

## [0.2.2](https://github.com/toon-format/toon-rust/compare/v0.2.1...v0.2.2) - 2025-11-07

### Added

- implement toon cli and update readme

### Fixed

- cargo clippy
- quoting context update
- update strict validations

### Other

- update changelog

## [0.2.1](https://github.com/toon-format/toon-rust/compare/v0.2.0...v0.2.1) - 2025-11-06

### Other

- update release workflow ([#8](https://github.com/toon-format/toon-rust/pull/8))
