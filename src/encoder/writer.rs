/* rune-xerosrc/encoder/writer.rs */
//!▫~•◦-----------------------------‣
//! # RUNE-Xero – Zero-Copy Writer Module
//!▫~•◦-----------------------------------‣
//!
//! Handles serialization of RUNE values directly to an output stream.
//! Uses `std::io::Write` to support streaming to files, sockets, or buffers
//! without creating intermediate string allocations.
//!
//! ## Key Capabilities
//! - **Streaming Output:** Writes directly to any `io::Write` implementation.
//! - **Zero-Allocation:** Avoids temporary strings for numbers and quoting.
//! - **Buffer reuse:** Caller controls the output buffer.
//! 
//! ### Architectural Notes
//! This module is designed to work with `crate::types::EncodeOptions`.
//! It replaces the legacy `String`-based writer with a generic `W: io::Write` implementation
//! to support the Zero-Copy architecture.
//!
//! #### Example
//! ```rust
//! use rune_xero::encoder::writer::Writer;
//! use rune_xero::types::EncodeOptions;
//!
//! let mut buffer = Vec::new();
//! let mut writer = Writer::new(&mut buffer, EncodeOptions::default());
//!
//! writer.write_str("key").unwrap();
//! writer.write_delimiter().unwrap();
//! writer.write_quoted_string("value").unwrap();
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::{
    types::{Delimiter, EncodeOptions, RuneError, RuneResult},
    utils::{
        QuotingContext,
        string::{is_valid_unquoted_key, needs_quoting},
    },
};
use std::io;

/// Writer that builds RUNE output directly to a stream.
pub struct Writer<W: io::Write> {
    writer: W,
    pub(crate) options: EncodeOptions,
    active_delimiters: Vec<Delimiter>,
}

impl<W: io::Write> Writer<W> {
    /// Create a new writer wrapping the given output stream.
    pub fn new(writer: W, options: EncodeOptions) -> Self {
        Self {
            writer,
            active_delimiters: vec![options.delimiter],
            options,
        }
    }

    /// Flush and return the inner writer.
    pub fn finish(mut self) -> RuneResult<W> {
        self.writer.flush().map_err(|e| RuneError::IoError(e.to_string()))?;
        Ok(self.writer)
    }

    pub fn write_str(&mut self, s: &str) -> RuneResult<()> {
        self.writer.write_all(s.as_bytes()).map_err(|e| RuneError::IoError(e.to_string()))
    }

    pub fn write_char(&mut self, ch: char) -> RuneResult<()> {
        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        self.writer.write_all(s.as_bytes()).map_err(|e| RuneError::IoError(e.to_string()))
    }

    pub fn write_newline(&mut self) -> RuneResult<()> {
        self.writer.write_all(b"\n").map_err(|e| RuneError::IoError(e.to_string()))
    }

    pub fn write_indent(&mut self, depth: usize) -> RuneResult<()> {
        // Zero-alloc indentation: repeat writes instead of allocating string
        if depth == 0 { return Ok(()); }
        
        let indent_char = if self.options.indent.use_tabs { b"\t" } else { b" " };
        let count = if self.options.indent.use_tabs { depth } else { depth * self.options.indent.size };
        
        for _ in 0..count {
            self.writer.write_all(indent_char).map_err(|e| RuneError::IoError(e.to_string()))?;
        }
        Ok(())
    }

    pub fn write_delimiter(&mut self) -> RuneResult<()> {
        let mut buf = [0u8; 4];
        let ch = self.options.delimiter.as_char();
        let s = ch.encode_utf8(&mut buf);
        self.writer.write_all(s.as_bytes()).map_err(|e| RuneError::IoError(e.to_string()))
    }

    pub fn write_key(&mut self, key: &str) -> RuneResult<()> {
        if is_valid_unquoted_key(key) {
            self.write_str(key)
        } else {
            self.write_quoted_string(key)
        }
    }

    /// Write an array header.
    /// Changed `fields` to `&[&str]` to allow zero-copy slices.
    pub fn write_array_header(
        &mut self,
        key: Option<&str>,
        length: usize,
        fields: Option<&[&str]>,
        depth: usize,
    ) -> RuneResult<()> {
        if let Some(k) = key {
            if depth > 0 {
                self.write_indent(depth)?;
            }
            self.write_key(k)?;
        }

        self.write_char('[')?;
        // Use itoa to write integer directly to buffer without allocation
        self.write_str(itoa::Buffer::new().format(length))?;

        if self.options.delimiter != Delimiter::Comma {
            self.write_delimiter()?;
        }

        self.write_char(']')?;

        if let Some(field_list) = fields {
            self.write_char('{')?;
            for (i, field) in field_list.iter().enumerate() {
                if i > 0 {
                    self.write_delimiter()?;
                }
                self.write_key(field)?;
            }
            self.write_char('}')?;
        }

        self.write_char(':')
    }

    pub fn write_empty_array_with_key(
        &mut self,
        key: Option<&str>,
        depth: usize,
    ) -> RuneResult<()> {
        if let Some(k) = key {
            if depth > 0 {
                self.write_indent(depth)?;
            }
            self.write_key(k)?;
        }
        self.write_str("[0")?;

        if self.options.delimiter != Delimiter::Comma {
            self.write_delimiter()?;
        }

        self.write_str("]:")
    }

    pub fn needs_quoting(&self, s: &str, context: QuotingContext) -> bool {
        let delim_char = match context {
            QuotingContext::ObjectValue => self.get_document_delimiter_char(),
            QuotingContext::ArrayValue => self.get_active_delimiter_char(),
        };
        needs_quoting(s, delim_char)
    }

    pub fn write_quoted_string(&mut self, s: &str) -> RuneResult<()> {
        self.write_char('"')?;
        
        // Zero-alloc quoting: stream chars and escape on the fly
        let mut start = 0;
        for (i, ch) in s.char_indices() {
            let escape = match ch {
                '"' => Some("\\\""),
                '\\' => Some("\\\\"),
                '\n' => Some("\\n"),
                '\r' => Some("\\r"),
                '\t' => Some("\\t"),
                _ => None,
            };

            if let Some(esc) = escape {
                if start < i {
                    self.write_str(&s[start..i])?;
                }
                self.write_str(esc)?;
                start = i + ch.len_utf8();
            }
        }
        if start < s.len() {
            self.write_str(&s[start..])?;
        }
        
        self.write_char('"')
    }

    pub fn write_value(&mut self, s: &str, context: QuotingContext) -> RuneResult<()> {
        if self.needs_quoting(s, context) {
            self.write_quoted_string(s)
        } else {
            self.write_str(s)
        }
    }

    pub fn push_active_delimiter(&mut self, delim: Delimiter) {
        self.active_delimiters.push(delim);
    }

    pub fn pop_active_delimiter(&mut self) {
        if self.active_delimiters.len() > 1 {
            self.active_delimiters.pop();
        }
    }

    fn get_active_delimiter_char(&self) -> char {
        self.active_delimiters
            .last()
            .unwrap_or(&self.options.delimiter)
            .as_char()
    }

    fn get_document_delimiter_char(&self) -> char {
        self.options.delimiter.as_char()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to extract string from Writer<Vec<u8>>
    fn get_output(writer: Writer<Vec<u8>>) -> String {
        let vec = writer.finish().unwrap();
        String::from_utf8(vec).unwrap()
    }

    #[test]
    fn test_writer_basic() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(Vec::new(), opts);

        writer.write_str("hello").unwrap();
        writer.write_str(" ").unwrap();
        writer.write_str("world").unwrap();

        assert_eq!(get_output(writer), "hello world");
    }

    #[test]
    fn test_write_delimiter() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(Vec::new(), opts.clone());

        writer.write_str("a").unwrap();
        writer.write_delimiter().unwrap();
        writer.write_str("b").unwrap();

        assert_eq!(get_output(writer), "a,b");
    }

    #[test]
    fn test_write_array_header() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(Vec::new(), opts);

        writer
            .write_array_header(Some("items"), 3, None, 0)
            .unwrap();
        assert_eq!(get_output(writer), "items[3]:");
    }

    #[test]
    fn test_write_quoted_string_streaming() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(Vec::new(), opts);

        writer.write_quoted_string("hello \"world\"").unwrap();
        assert_eq!(get_output(writer), r#""hello \"world\"""#);
    }
}
