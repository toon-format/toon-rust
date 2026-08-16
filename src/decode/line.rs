//! Line scanning: the lexical pre-pass of §5.1 and §12.
//!
//! Turns raw input into a stream of [`ParsedLine`]s with byte-order mark
//! removal, CRLF acceptance, trailing-space stripping, comment-line removal,
//! blank-line tracking, and strict-mode indentation checks.

use crate::types::{
    ToonError,
    ToonResult,
};

/// One structural line of a TOON document, after the lexical pre-pass.
#[derive(Debug, Clone)]
pub(crate) struct ParsedLine {
    /// Raw line content (after BOM/CR exclusion), including indentation.
    pub raw: String,
    /// Content after indentation, with trailing spaces stripped.
    pub content: String,
    /// Indentation depth in `indent_size` units.
    pub depth: usize,
    /// 1-based line number in the source document.
    pub line_number: usize,
}

/// Streams [`ParsedLine`]s with one line of lookahead, so strict-mode line
/// errors surface in document order relative to structural errors.
pub(crate) struct LineReader<'s> {
    lines: std::str::Split<'s, char>,
    buffered: Option<ParsedLine>,
    done: bool,
    line_number: usize,
    /// Line numbers of blank lines seen so far (§12 header-span checks).
    pub blank_lines: Vec<usize>,
    /// Line number of the most recently consumed line.
    pub last_consumed_line: usize,
    indent_size: usize,
    strict: bool,
}

impl<'s> LineReader<'s> {
    pub fn new(input: &'s str, indent_size: usize, strict: bool) -> Self {
        Self {
            lines: input.split('\n'),
            buffered: None,
            done: false,
            line_number: 0,
            blank_lines: Vec::new(),
            last_consumed_line: 0,
            indent_size,
            strict,
        }
    }

    /// Peek the next structural line without consuming it.
    pub fn peek(&mut self) -> ToonResult<Option<&ParsedLine>> {
        self.fill()?;
        Ok(self.buffered.as_ref())
    }

    /// Consume and return the next structural line.
    pub fn next(&mut self) -> ToonResult<Option<ParsedLine>> {
        self.fill()?;
        let line = self.buffered.take();
        if let Some(line) = &line {
            self.last_consumed_line = line.line_number;
        }
        Ok(line)
    }

    fn fill(&mut self) -> ToonResult<()> {
        while self.buffered.is_none() && !self.done {
            match self.lines.next() {
                None => self.done = true,
                Some(raw) => {
                    if let Some(line) = self.parse_line(raw)? {
                        self.buffered = Some(line);
                    }
                }
            }
        }
        Ok(())
    }

    /// The lexical pre-pass for one line. Returns `None` for comment and
    /// blank lines, which never reach structural interpretation.
    fn parse_line(&mut self, raw: &str) -> ToonResult<Option<ParsedLine>> {
        self.line_number += 1;
        let line_number = self.line_number;

        // A single leading U+FEFF is a byte-order mark, not content (§12).
        // Lines are split on LF alone, so exactly one trailing CR – the CR of
        // a CRLF terminator – is excluded here; a second CR is content.
        let raw = raw.strip_suffix('\r').unwrap_or(raw);
        let raw = if line_number == 1 {
            raw.strip_prefix('\u{FEFF}').unwrap_or(raw)
        } else {
            raw
        };

        let lead_len = raw
            .find(|c: char| c != ' ' && c != '\t')
            .unwrap_or(raw.len());
        let leading = &raw[..lead_len];
        let first_tab = leading.find('\t');

        // Strict mode rejects tab indentation below, so only the spaces before
        // the first tab count as indentation there. Non-strict input may
        // indent with tabs; each tab counts as one depth level.
        let indent = match first_tab {
            Some(tab_index) if self.strict => tab_index,
            _ => lead_len,
        };
        let tab_indent = if self.strict || first_tab.is_none() {
            0
        } else {
            leading.matches('\t').count()
        };

        // Trailing spaces are not content (§12): without this, `- ` would be
        // an item carrying an empty token instead of the bare marker.
        let content = raw[indent..].trim_end_matches(' ');

        // Only spaces may precede a comment marker, so a tab in the
        // indentation rules the line out (§5.1). Comment lines vanish before
        // blank tracking and strict validation.
        if first_tab.is_none() && content.starts_with('#') {
            return Ok(None);
        }

        if content.is_empty() {
            self.blank_lines.push(line_number);
            return Ok(None);
        }

        if self.strict {
            if first_tab.is_some() {
                return Err(ToonError::parse_error(
                    line_number,
                    1,
                    "Tabs are not allowed in indentation in strict mode",
                ));
            }

            if indent > 0 && indent % self.indent_size != 0 {
                return Err(ToonError::parse_error(
                    line_number,
                    1,
                    format!(
                        "Indentation must be exact multiple of {}, but found {indent} spaces",
                        self.indent_size
                    ),
                ));
            }
        }

        let depth = (indent - tab_indent) / self.indent_size + tab_indent;

        Ok(Some(ParsedLine {
            raw: raw.to_string(),
            content: content.to_string(),
            depth,
            line_number,
        }))
    }
}
