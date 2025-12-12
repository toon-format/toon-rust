# Task Progress: Fix rune-gate UI Compilation Errors

## Issue Analysis

The rune-gate crate has Bevy as an optional dependency with the "viewer" feature, but ui.rs is using Bevy types unconditionally, causing compilation errors when the feature is disabled.

## Task Plan

- [ ] Add feature gating to ui.rs for Bevy types
- [ ] Check other rune-gate source files for similar issues
- [ ] Update Cargo.toml if needed to enable viewer by default
- [ ] Test compilation with both feature states
- [ ] Verify all Bevy imports and types are properly gated

## Solution Strategy

1. Add `#[cfg(feature = "viewer")]` guards around Bevy-specific code in ui.rs
2. Create conditional compilation for UI types and functions
3. Ensure the crate compiles without the viewer feature enabled
4. Test that the viewer functionality works when enabled
