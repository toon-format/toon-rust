//! Event handling for terminal input.

use std::time::Duration;

use crossterm::event::{self, Event as CrosstermEvent, KeyEvent};

/// TUI events.
pub enum Event {
    Key(KeyEvent),
    Tick,
    Resize,
}

pub struct EventHandler;

impl EventHandler {
    /// Poll for next event with timeout.
    pub fn poll(timeout: Duration) -> std::io::Result<Option<Event>> {
        if event::poll(timeout)? {
            match event::read()? {
                CrosstermEvent::Key(key) => Ok(Some(Event::Key(key))),
                CrosstermEvent::Resize(_, _) => Ok(Some(Event::Resize)),
                _ => Ok(None),
            }
        } else {
            Ok(Some(Event::Tick))
        }
    }
}
