# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
