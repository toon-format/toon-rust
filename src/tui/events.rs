//! Event handling for terminal input.

use std::time::Duration;

use crossterm::event::{self, Event as CrosstermEvent, KeyEvent};

/// TUI events.
///
/// # Examples
/// ```no_run
/// use toon_format::tui::events::Event;
///
/// let _event = Event::Tick;
/// ```
pub enum Event {
    Key(KeyEvent),
    Tick,
    Resize,
}

/// Event polling helper for the TUI.
///
/// # Examples
/// ```no_run
/// use toon_format::tui::events::EventHandler;
/// use std::time::Duration;
///
/// let _ = EventHandler::poll(Duration::from_millis(10));
/// ```
pub struct EventHandler;

impl EventHandler {
    /// Poll for next event with timeout.
    ///
    /// # Examples
    /// ```no_run
    /// use toon_format::tui::events::EventHandler;
    /// use std::time::Duration;
    ///
    /// let _ = EventHandler::poll(Duration::from_millis(10));
    /// ```
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
