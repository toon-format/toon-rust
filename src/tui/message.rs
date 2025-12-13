/* src/tui/message.rs */
//!▫~•◦-------------------------------‣
//! # Application messages (Elm-style) for the RUNE TUI.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module defines the central `Msg` enum, which represents all user actions
//! and internal events that can modify the application's state. It is the core
//! of the TUI's event-driven, Elm-style architecture.
//!
//! ## Key Capabilities
//! - **Centralized Actions**: Provides a single, type-safe enum for all possible state changes.
//! - **Data-Carrying Variants**: Allows messages to carry payloads, such as file paths or
//!   status text.
//! - **Performance-Optimized**: Uses `Cow<'static, str>` for status and error message
//!   payloads. This allows the application to create messages from static string
//!   literals without any heap allocation, while still supporting dynamic `String`s
//!   when necessary.
//!
//! ### Architectural Notes
//! The use of `Cow<'static, str>` is a key performance optimization. Most status
//! messages in the application are fixed strings. By using `Cow`, we avoid countless
//! unnecessary `String` allocations, reducing pressure on the memory allocator and
//! improving overall responsiveness.
//!
//! #### Example
//! ```rust
//! use std::borrow::Cow;
//! use rune_xero::tui::message::Msg;
//!
//! // Creating a message from a static string (zero allocation):
//! let status_msg = Msg::SetStatus("Operation successful".into());
//!
//! // Creating a message from a dynamic string (one allocation):
//! let error_code = 404;
//! let error_msg = Msg::SetError(format!("Error code: {}", error_code).into());
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::borrow::Cow;
use std::path::PathBuf;

/// Represents all possible actions and events that can modify application state.
// By using `'static`, we state that any borrowed strings must live for the entire
// program, which is true for string literals. This avoids infecting the entire
// app with a shorter lifetime parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum Msg {
    // --- Application Control ---
    Quit,
    Tick, // For periodic updates, like clearing messages

    // --- Mode & View Toggles ---
    ToggleMode,
    ToggleSettings,
    ToggleHelp,
    ToggleFileBrowser,
    ToggleHistory,
    ToggleDiff,
    ToggleTheme,

    // --- File Operations ---
    OpenFile(PathBuf),
    SaveFile,
    NewFile,

    // --- Editor & Conversion ---
    Refresh,
    CopyOutput,
    CopySelection,
    PasteInput,
    RoundTrip,
    ClearInput,

    // --- REPL ---
    ExecuteRepl(String),

    // --- Status & Error Handling ---
    SetError(Cow<'static, str>),
    SetStatus(Cow<'static, str>),
    ClearMessages,
}