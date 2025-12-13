/* src/tui/events.rs */
//!▫~•◦-------------------------------‣
//! # Event handling for terminal input and application ticks.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module provides the `EventHandler` struct and `Event` enum, which form
//! the core of the TUI's input and timing loop. It abstracts the underlying
//! `crossterm` event stream into a simpler, application-specific format.
//!
//! ## Key Capabilities
//! - **Event Abstraction**: Defines a simple `Event` enum (`Key`, `Tick`, `Resize`)
//!   for the application to handle.
//! - **Timed Polling**: Provides a non-blocking poll mechanism with a timeout, which
//!   generates `Tick` events for periodic updates.
//! - **Stateful Handling**: The `EventHandler` struct is designed to hold state for
//!   more complex event handling in the future (e.g., debouncing, batching).
//!
//! ### Architectural Notes
//! The `poll` method is the heart of the application's main loop. By using `crossterm::event::poll`,
//! it efficiently waits for user input without consuming CPU cycles in a busy-wait loop.
//! This is a fundamental pattern for building responsive, low-overhead terminal applications.
//!
//! #### Example
//! ```rust
//! // In the main application loop:
//! use std::time::Duration;
//! use rune_xero::tui::events::{EventHandler, Event};
//!
//! let mut event_handler = EventHandler::new();
//! loop {
//!     if let Some(event) = event_handler.poll(Duration::from_millis(100)).unwrap() {
//!         match event {
//!             Event::Key(key_event) => { /* handle key press */ },
//!             Event::Tick => { /* perform periodic updates */ },
//!             _ => {}
//!         }
//!     }
//!     // break condition
//! }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::time::Duration;

use ratatui::crossterm::event::{self, Event as CrosstermEvent, KeyEvent};
use yoshi::Hatch;

/// TUI events handled by the application.
#[derive(Debug, Clone, Copy)]
pub enum Event {
    Key(KeyEvent),
    Tick,
    Resize,
}

/// A stateful handler for managing terminal events.
///
/// While currently stateless, it is designed as a struct to accommodate
/// future features like event batching, debouncing, or custom tick management.
#[derive(Debug)]
pub struct EventHandler;

impl EventHandler {
    /// Constructs a new `EventHandler`.
    pub fn new() -> Self {
        Self {}
    }

    /// Polls for the next event with a specified timeout.
    ///
    /// If a terminal event (key press, resize) occurs within the timeout, it is
    /// returned. Otherwise, a `Tick` event is returned upon timeout.
    pub fn poll(&mut self, timeout: Duration) -> Hatch<Option<Event>> {
        if event::poll(timeout)? {
            match event::read()? {
                CrosstermEvent::Key(key) => Ok(Some(Event::Key(key))),
                CrosstermEvent::Resize(_, _) => Ok(Some(Event::Resize)),
                _ => Ok(None), // Ignore mouse, focus, etc. for now.
            }
        } else {
            // No event occurred within the timeout, so we emit a Tick.
            Ok(Some(Event::Tick))
        }
    }
}

impl Default for EventHandler {
    fn default() -> Self {
        Self::new()
    }
}