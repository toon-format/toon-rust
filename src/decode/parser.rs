//! Line-based TOON decoder implementing spec v4.1.
//!
//! Lines are classified per §5.2 on the comment-stripped sequence (§5.1);
//! headers are parsed per §6, including keyed tabular headers and nested
//! field groups; tokens follow the normative number grammar of §4 and the
//! key/quoted-token rules of §7.4.

use serde_json::{
    Map,
    Number,
    Value,
};

use crate::{
    constants::MAX_DEPTH,
    decode::line::{
        LineReader,
        ParsedLine,
    },
    types::{
        DecodeOptions,
        Delimiter,
        ErrorContext,
        ToonError,
        ToonResult,
    },
    utils::{
        unescape_string,
        validation::validate_depth,
    },
};

// #region String scanning helpers
//
// All scans are byte-based: every target is ASCII, and ASCII bytes never
// occur inside a multi-byte UTF-8 sequence, so byte positions are always
// char boundaries.

/// Trims surrounding ASCII spaces (exactly U+0020, §12) from a token.
fn trim_spaces(value: &str) -> &str {
    value.trim_matches(' ')
}

/// Finds the byte index of the closing double quote for the quote at
/// `start`, accounting for escape sequences.
fn find_closing_quote(content: &str, start: usize) -> Option<usize> {
    let bytes = content.as_bytes();
    let mut i = start + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if i + 1 < bytes.len() => i += 2,
            b'"' => return Some(i),
            _ => i += 1,
        }
    }
    None
}

/// Finds the byte index of an ASCII character outside of quoted sections.
fn find_unquoted_char(content: &str, target: u8, start: usize) -> Option<usize> {
    let bytes = content.as_bytes();
    let mut in_quotes = false;
    let mut i = start;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\\' && in_quotes && i + 1 < bytes.len() {
            i += 2;
            continue;
        }
        if b == b'"' {
            in_quotes = !in_quotes;
            i += 1;
            continue;
        }
        if b == target && !in_quotes {
            return Some(i);
        }
        i += 1;
    }
    None
}

// #endregion

// #region Literal parsing (§4)

fn is_boolean_or_null_literal(token: &str) -> bool {
    matches!(token, "true" | "false" | "null")
}

/// The normative decoder number grammar (§4):
/// `/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:e[+-]?\d+)?$/i` with ASCII digits only.
fn is_numeric_literal(token: &str) -> bool {
    let bytes = token.as_bytes();
    let mut i = 0;
    if bytes.first() == Some(&b'-') {
        i = 1;
    }

    let int_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == int_start {
        return false;
    }
    // Forbidden leading zeros in the integer part.
    if bytes[int_start] == b'0' && i - int_start > 1 {
        return false;
    }

    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == frac_start {
            return false;
        }
    }

    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        i += 1;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return false;
        }
    }

    i == bytes.len()
}

/// Decodes a numeric token to a JSON number.
///
/// Integral tokens preserve full `i64`/`u64` precision. Fractional and
/// exponent forms parse as `f64`; integer-valued results within the `i64`
/// domain normalize to integers so `-1E+03` decodes as `-1000`. A token whose
/// value is not finite in `f64` (e.g. `1e999`) decodes as a string, per the
/// documented out-of-range policy.
fn parse_number_token(token: &str) -> Value {
    if !token.contains(['.', 'e', 'E']) {
        if let Ok(i) = token.parse::<i64>() {
            return Value::Number(Number::from(i));
        }
        if let Ok(u) = token.parse::<u64>() {
            return Value::Number(Number::from(u));
        }
    }

    match token.parse::<f64>() {
        Ok(f) if f.is_finite() => {
            if f == 0.0 {
                // -0 decodes to 0 (§4).
                Value::Number(Number::from(0u64))
            } else if f.fract() == 0.0 && f.abs() < i64::MAX as f64 {
                Value::Number(Number::from(f as i64))
            } else {
                Number::from_f64(f).map_or_else(|| Value::String(token.to_string()), Value::Number)
            }
        }
        _ => Value::String(token.to_string()),
    }
}

/// Parses a quoted or unquoted token as a string, enforcing the
/// quoted-token boundary rule of §7.4.
fn parse_string_literal(token: &str) -> Result<String, String> {
    let trimmed = trim_spaces(token);

    if trimmed.starts_with('"') {
        let closing = find_closing_quote(trimmed, 0)
            .ok_or_else(|| "Unterminated string: missing closing quote".to_string())?;
        if closing != trimmed.len() - 1 {
            return Err("Unexpected characters after closing quote".to_string());
        }
        return unescape_string(&trimmed[1..closing]);
    }

    Ok(trimmed.to_string())
}

/// Parses one primitive token per §4: quoted strings, `true`/`false`/`null`,
/// the normative number grammar, and the string fallback.
fn parse_primitive_token(token: &str) -> Result<Value, String> {
    let trimmed = trim_spaces(token);

    if trimmed.is_empty() {
        return Ok(Value::String(String::new()));
    }

    if trimmed.starts_with('"') {
        return parse_string_literal(trimmed).map(Value::String);
    }

    if is_boolean_or_null_literal(trimmed) {
        return Ok(match trimmed {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => Value::Null,
        });
    }

    if is_numeric_literal(trimmed) {
        return Ok(parse_number_token(trimmed));
    }

    Ok(Value::String(trimmed.to_string()))
}

// #endregion

// #region Key parsing (§7.4)

/// Parses the key of a key-value line or entry row starting at byte 0.
/// Returns the decoded key and the byte offset just past the colon.
fn parse_key_token(content: &str) -> Result<(String, usize), String> {
    if content.as_bytes().first() == Some(&b'"') {
        let closing =
            find_closing_quote(content, 0).ok_or_else(|| "Unterminated quoted key".to_string())?;
        let key = unescape_string(&content[1..closing])?;
        let after = closing + 1;
        if content.as_bytes().get(after) != Some(&b':') {
            return Err("Missing colon after key".to_string());
        }
        Ok((key, after + 1))
    } else {
        let colon =
            find_unquoted_char(content, b':', 0).ok_or_else(|| "Missing colon after key".to_string())?;
        Ok((trim_spaces(&content[..colon]).to_string(), colon + 1))
    }
}

// #endregion

// #region Header parsing (§6)

/// One entry of a header's field list. A leaf field maps to one row cell; a
/// nested field group carries its subfields and materializes a nested object
/// per row (§9.3).
#[derive(Debug, Clone)]
pub(crate) struct FieldNode {
    pub name: String,
    pub children: Option<Vec<FieldNode>>,
}

#[derive(Debug, Clone)]
pub(crate) struct ArrayHeaderInfo {
    pub key: Option<String>,
    pub length: usize,
    pub delimiter: Delimiter,
    pub fields: Option<Vec<FieldNode>>,
    /// Keyed tabular header `[N:<delim?>]` – N declares the entry count (§9.5).
    pub keyed: bool,
}

enum HeaderParse {
    Header {
        header: ArrayHeaderInfo,
        inline_values: Option<String>,
        strict_error: Option<String>,
    },
    NotHeader,
    Invalid(String),
}

/// Detects and parses an array-header line, staying free of strict-mode
/// policy: callers decide how to treat `Invalid` and `strict_error`.
fn parse_array_header_line(content: &str) -> HeaderParse {
    let trimmed = content.trim_start();

    let bracket_start = if trimmed.starts_with('"') {
        let Some(closing) = find_closing_quote(trimmed, 0) else {
            return HeaderParse::NotHeader;
        };
        if trimmed.as_bytes().get(closing + 1) != Some(&b'[') {
            return HeaderParse::NotHeader;
        }
        let leading = content.len() - trimmed.len();
        leading + closing + 1
    } else {
        match find_unquoted_char(content, b'[', 0) {
            Some(i) => i,
            None => return HeaderParse::NotHeader,
        }
    };

    // A header key can't contain an unquoted colon, so this is a key-value line.
    if let Some(colon) = find_unquoted_char(content, b':', 0) {
        if colon < bracket_start {
            return HeaderParse::NotHeader;
        }
    }

    let Some(bracket_end) = find_unquoted_char(content, b']', bracket_start) else {
        return HeaderParse::NotHeader;
    };

    let mut brace_end = bracket_end + 1;
    let brace_start = find_unquoted_char(content, b'{', bracket_end);
    if let Some(brace_start) = brace_start {
        let colon_after_bracket = find_unquoted_char(content, b':', bracket_end);
        if colon_after_bracket.is_some_and(|c| brace_start < c) {
            let gap = &content[bracket_end + 1..brace_start];
            if !gap.is_empty() {
                let trimmed_gap = gap.trim();
                return HeaderParse::Invalid(if trimmed_gap.is_empty() {
                    "Unexpected whitespace between bracket segment and field list".to_string()
                } else {
                    format!(
                        "Unexpected content \"{trimmed_gap}\" between bracket segment and field \
                         list"
                    )
                });
            }

            if let Some(found) = find_matching_brace(content, brace_start) {
                brace_end = found + 1;
            }
        }
    }

    let Some(colon_index) = find_unquoted_char(content, b':', bracket_end.max(brace_end)) else {
        return HeaderParse::NotHeader;
    };

    let gap_start = (bracket_end + 1).max(brace_end);
    let gap = &content[gap_start..colon_index];
    if !gap.is_empty() {
        let trimmed_gap = gap.trim();
        return HeaderParse::Invalid(if trimmed_gap.is_empty() {
            "Unexpected whitespace between bracket segment and colon".to_string()
        } else {
            format!("Unexpected content \"{trimmed_gap}\" between bracket segment and colon")
        });
    }

    let key = if bracket_start > 0 {
        let raw_key = &content[..bracket_start];
        // Trimming here would silently turn `foo [2]:` into a header with key `foo`.
        if raw_key != raw_key.trim_end() {
            return HeaderParse::Invalid(
                "Unexpected whitespace between key and bracket segment".to_string(),
            );
        }
        match parse_string_literal(raw_key) {
            Ok(key) => Some(key),
            Err(reason) => return HeaderParse::Invalid(reason),
        }
    } else {
        None
    };

    let after_colon = trim_spaces(&content[colon_index + 1..]);
    let bracket_content = &content[bracket_start + 1..bracket_end];

    let (length, delimiter, keyed) = match parse_bracket_segment(bracket_content) {
        Ok(parsed) => parsed,
        Err(reason) => return HeaderParse::Invalid(reason),
    };

    let mut fields = None;
    if let Some(brace_start) = brace_start {
        if brace_start < colon_index {
            if let Some(found) = find_matching_brace(content, brace_start) {
                if found < colon_index {
                    let fields_content = &content[brace_start + 1..found];

                    if let Some(mismatched) =
                        find_unquoted_mismatched_delimiter(fields_content, delimiter)
                    {
                        return HeaderParse::Invalid(format!(
                            "Header delimiter mismatch: bracket declares \"{}\" but field list \
                             contains unquoted \"{}\"",
                            format_delimiter(delimiter),
                            format_delimiter(mismatched)
                        ));
                    }

                    match parse_field_entries(fields_content, delimiter) {
                        Ok(parsed) => fields = Some(parsed),
                        Err(reason) => return HeaderParse::Invalid(reason),
                    }
                }
            }
        }
    }

    // Duplicate field names are strict-only – non-strict resolves them via
    // LWW – so the reason rides along on an otherwise-valid header.
    let strict_error = fields
        .as_deref()
        .and_then(find_duplicate_field_name)
        .map(|name| format!("Duplicate field name \"{name}\" in field list"));

    if keyed && fields.is_none() {
        return HeaderParse::Invalid("Keyed header requires a field list".to_string());
    }

    // A fields-bearing header, keyed or not, carries no inline content.
    if fields.is_some() && !after_colon.is_empty() {
        return HeaderParse::Invalid(strict_error.unwrap_or_else(|| {
            "Unexpected content after fields-bearing header colon".to_string()
        }));
    }

    HeaderParse::Header {
        header: ArrayHeaderInfo {
            key,
            length,
            delimiter,
            fields,
            keyed,
        },
        inline_values: if after_colon.is_empty() {
            None
        } else {
            Some(after_colon.to_string())
        },
        strict_error,
    }
}

/// Parses a bracket segment `N`, `N<delim>`, `N:`, or `N:<delim>` (§6).
fn parse_bracket_segment(seg: &str) -> Result<(usize, Delimiter, bool), String> {
    let mut content = seg;

    let mut delimiter = Delimiter::Comma;
    if let Some(rest) = content.strip_suffix('\t') {
        delimiter = Delimiter::Tab;
        content = rest;
    } else if let Some(rest) = content.strip_suffix('|') {
        delimiter = Delimiter::Pipe;
        content = rest;
    }

    // Only a colon between the length and the optional delimiter symbol marks
    // a keyed header; any other placement fails the length check below.
    let mut keyed = false;
    if let Some(rest) = content.strip_suffix(':') {
        keyed = true;
        content = rest;
    }

    let valid_length = !content.is_empty()
        && content.bytes().all(|b| b.is_ascii_digit())
        && (content == "0" || !content.starts_with('0'));
    if !valid_length {
        return Err(format!(
            "Invalid array length: \"{seg}\" (expected non-negative integer with no leading zeros)"
        ));
    }

    let length = content
        .parse::<usize>()
        .map_err(|_| format!("Invalid array length: \"{seg}\" (value out of range)"))?;

    Ok((length, delimiter, keyed))
}

/// Parses the content of a field list into field entries, recursively
/// descending into nested field groups (`field{sub1,sub2}`).
fn parse_field_entries(fields_content: &str, delimiter: Delimiter) -> Result<Vec<FieldNode>, String> {
    split_field_entries(fields_content, delimiter)
        .into_iter()
        .map(|entry| {
            let trimmed = trim_spaces(&entry);
            if trimmed.is_empty() {
                return Err("Empty field name in field list".to_string());
            }

            let Some(group_start) = find_unquoted_char(trimmed, b'{', 0) else {
                return Ok(FieldNode {
                    name: parse_string_literal(trimmed)?,
                    children: None,
                });
            };

            let name_part = trim_spaces(&trimmed[..group_start]);
            if name_part.is_empty() {
                return Err("Missing field name before nested field group".to_string());
            }

            let group_end = find_matching_brace(trimmed, group_start)
                .ok_or_else(|| "Unmatched brace in field list".to_string())?;
            if group_end != trimmed.len() - 1 {
                return Err("Unexpected content after nested field group".to_string());
            }

            let children = parse_field_entries(&trimmed[group_start + 1..group_end], delimiter)?;
            Ok(FieldNode {
                name: parse_string_literal(name_part)?,
                children: Some(children),
            })
        })
        .collect()
}

/// Splits a field list on the active delimiter at brace depth zero,
/// respecting quoted names and escape sequences.
fn split_field_entries(content: &str, delimiter: Delimiter) -> Vec<String> {
    let delim = delimiter.as_char() as u8;
    let bytes = content.as_bytes();
    let mut entries = Vec::new();
    let mut entry_start = 0;
    let mut in_quotes = false;
    let mut brace_depth = 0usize;
    let mut i = 0;

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\\' && in_quotes && i + 1 < bytes.len() {
            i += 2;
            continue;
        }
        if b == b'"' {
            in_quotes = !in_quotes;
            i += 1;
            continue;
        }
        if !in_quotes {
            if b == b'{' {
                brace_depth += 1;
            } else if b == b'}' {
                brace_depth = brace_depth.saturating_sub(1);
            } else if b == delim && brace_depth == 0 {
                entries.push(content[entry_start..i].to_string());
                entry_start = i + 1;
                i += 1;
                continue;
            }
        }
        i += 1;
    }

    entries.push(content[entry_start..].to_string());
    entries
}

/// Finds the byte index of the closing brace matching the opening brace at
/// `brace_start`, ignoring braces inside quoted names.
fn find_matching_brace(content: &str, brace_start: usize) -> Option<usize> {
    let bytes = content.as_bytes();
    let mut in_quotes = false;
    let mut brace_depth = 0usize;
    let mut i = brace_start;

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\\' && in_quotes && i + 1 < bytes.len() {
            i += 2;
            continue;
        }
        if b == b'"' {
            in_quotes = !in_quotes;
            i += 1;
            continue;
        }
        if !in_quotes {
            if b == b'{' {
                brace_depth += 1;
            } else if b == b'}' {
                brace_depth -= 1;
                if brace_depth == 0 {
                    return Some(i);
                }
            }
        }
        i += 1;
    }

    None
}

fn find_duplicate_field_name(fields: &[FieldNode]) -> Option<String> {
    let mut seen = std::collections::HashSet::new();
    for field in fields {
        if !seen.insert(field.name.as_str()) {
            return Some(field.name.clone());
        }
        if let Some(children) = &field.children {
            if let Some(nested) = find_duplicate_field_name(children) {
                return Some(nested);
            }
        }
    }
    None
}

/// Counts the leaf fields of a field list: the number of cells each row
/// carries, via a depth-first walk of nested field groups.
fn count_leaf_fields(fields: &[FieldNode]) -> usize {
    fields
        .iter()
        .map(|field| field.children.as_deref().map_or(1, count_leaf_fields))
        .sum()
}

fn find_unquoted_mismatched_delimiter(
    content: &str,
    active_delimiter: Delimiter,
) -> Option<Delimiter> {
    [Delimiter::Comma, Delimiter::Tab, Delimiter::Pipe]
        .into_iter()
        .filter(|candidate| *candidate != active_delimiter)
        .find(|candidate| find_unquoted_char(content, candidate.as_char() as u8, 0).is_some())
}

fn format_delimiter(delimiter: Delimiter) -> &'static str {
    match delimiter {
        Delimiter::Comma => ",",
        Delimiter::Tab => "\\t",
        Delimiter::Pipe => "|",
    }
}

// #endregion

// #region Delimited value parsing (§11.2)

/// Parses a delimited cell sequence into raw value tokens, respecting quoted
/// strings and escape sequences; each token is space-trimmed.
fn parse_delimited_values(input: &str, delimiter: Delimiter) -> Vec<String> {
    let delim = delimiter.as_char() as u8;
    let bytes = input.as_bytes();
    let mut values = Vec::new();
    let mut value_start = 0;
    let mut in_quotes = false;
    let mut i = 0;

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\\' && in_quotes && i + 1 < bytes.len() {
            i += 2;
            continue;
        }
        if b == b'"' {
            in_quotes = !in_quotes;
            i += 1;
            continue;
        }
        if b == delim && !in_quotes {
            values.push(trim_spaces(&input[value_start..i]).to_string());
            value_start = i + 1;
            i += 1;
            continue;
        }
        i += 1;
    }

    let last = trim_spaces(&input[value_start..]);
    if !last.is_empty() || !values.is_empty() {
        values.push(last.to_string());
    }

    values
}

// #endregion

// #region Line classification helpers

fn is_list_item_content(content: &str) -> bool {
    content == "-" || content.starts_with("- ")
}

fn is_array_header_content(content: &str) -> bool {
    content.trim().starts_with('[') && find_unquoted_char(content, b':', 0).is_some()
}

fn is_key_value_content(content: &str) -> bool {
    find_unquoted_char(content, b':', 0).is_some()
}

/// Root-form key-value check (§5): quoted keys look past the closing quote.
fn is_key_value_line(content: &str) -> bool {
    if content.starts_with('"') {
        match find_closing_quote(content, 0) {
            Some(closing) => content[closing + 1..].contains(':'),
            None => false,
        }
    } else {
        content.contains(':')
    }
}

/// Row/key-value disambiguation at row depth (§9.3): first unquoted
/// delimiter vs first unquoted colon.
fn is_data_row(content: &str, delimiter: Delimiter) -> bool {
    let Some(colon_pos) = find_unquoted_char(content, b':', 0) else {
        return true;
    };
    match find_unquoted_char(content, delimiter.as_char() as u8, 0) {
        Some(delim_pos) => delim_pos < colon_pos,
        None => false,
    }
}

// #endregion

// #region Parser

/// Attaches line context to an error message.
fn err_at(line: &ParsedLine, message: impl Into<String>) -> ToonError {
    ToonError::parse_error(line.line_number, 1, message)
        .with_context(ErrorContext::new(line.raw.clone()))
}

fn over_indented_error(line: &ParsedLine, expected_depth: usize) -> ToonError {
    err_at(
        line,
        format!(
            "Over-indented line: expected depth {expected_depth}, but found {}",
            line.depth
        ),
    )
}

/// Both modes reject a bare token outside root primitive position (§5.2),
/// so it must not reach the non-strict paths that drop an over-indented
/// line.
fn assert_not_scalar_line(line: &ParsedLine) -> ToonResult<()> {
    if is_list_item_content(&line.content) || is_key_value_content(&line.content) {
        return Ok(());
    }
    Err(err_at(
        line,
        "Unexpected bare token line outside root primitive position",
    ))
}

struct ResolvedHeader {
    header: ArrayHeaderInfo,
    inline_values: Option<String>,
}

pub struct Parser<'s> {
    reader: LineReader<'s>,
    strict: bool,
    #[cfg(feature = "layout")]
    layout: Option<crate::decode::layout_builder::LayoutBuilder>,
}

impl<'s> Parser<'s> {
    pub fn new(input: &'s str, options: DecodeOptions) -> ToonResult<Self> {
        let indent_size = options.indent.get_spaces();
        if indent_size == 0 {
            return Err(ToonError::InvalidInput(
                "indentSize must be at least 1".to_string(),
            ));
        }

        Ok(Self {
            reader: LineReader::new(input, indent_size, options.strict),
            strict: options.strict,
            #[cfg(feature = "layout")]
            layout: None,
        })
    }

    #[cfg(feature = "layout")]
    pub fn with_layout(mut self) -> Self {
        self.layout = Some(crate::decode::layout_builder::LayoutBuilder::new());
        self
    }

    #[cfg(feature = "layout")]
    pub fn take_layout(&mut self) -> Option<crate::layout::Layout> {
        self.layout.take().map(|builder| builder.finish())
    }

    pub fn parse(&mut self) -> ToonResult<Value> {
        self.decode_document()
    }

    // #region Layout hooks

    #[cfg(feature = "layout")]
    fn layout_push(&mut self, segment: &str) {
        if let Some(builder) = &mut self.layout {
            builder.push(segment);
        }
    }

    #[cfg(not(feature = "layout"))]
    fn layout_push(&mut self, _segment: &str) {}

    #[cfg(feature = "layout")]
    fn layout_pop(&mut self) {
        if let Some(builder) = &mut self.layout {
            builder.pop();
        }
    }

    #[cfg(not(feature = "layout"))]
    fn layout_pop(&mut self) {}

    #[cfg(feature = "layout")]
    fn layout_record_array(&mut self, header: &ArrayHeaderInfo, inline: bool) {
        use crate::layout::{
            FieldDescriptor,
            NodeLayout,
        };

        let Some(builder) = &mut self.layout else {
            return;
        };

        let node = if let Some(fields) = &header.fields {
            NodeLayout::Tabular {
                declared_len: header.length,
                fields: fields
                    .iter()
                    .map(|field| FieldDescriptor::leaf(field.name.clone()))
                    .collect(),
                delimiter: header.delimiter,
            }
        } else if inline {
            NodeLayout::InlineArray {
                declared_len: header.length,
                delimiter: header.delimiter,
            }
        } else {
            NodeLayout::List {
                declared_len: header.length,
            }
        };
        builder.record(node);
    }

    #[cfg(not(feature = "layout"))]
    fn layout_record_array(&mut self, _header: &ArrayHeaderInfo, _inline: bool) {}

    // #endregion

    // #region Error helpers

    fn assert_no_depth_jump(&self, nested: &ParsedLine, parent_depth: usize) -> ToonResult<()> {
        if self.strict && nested.depth > parent_depth + 1 {
            return Err(err_at(
                nested,
                format!(
                    "Indentation depth jump: expected depth {}, but found {}",
                    parent_depth + 1,
                    nested.depth
                ),
            ));
        }
        Ok(())
    }

    fn assert_expected_count(
        &self,
        actual: usize,
        expected: usize,
        item_type: &str,
        line_number: usize,
    ) -> ToonResult<()> {
        if self.strict && actual != expected {
            return Err(ToonError::parse_error(
                line_number,
                1,
                format!("Expected {expected} {item_type}, but got {actual}"),
            ));
        }
        Ok(())
    }

    /// Strict decoding never silently discards input, so a line after the
    /// root form is an error (§5).
    fn assert_fully_consumed(&mut self) -> ToonResult<()> {
        if !self.strict {
            return Ok(());
        }
        if let Some(line) = self.reader.peek()? {
            let err = err_at(line, "Unexpected content after the document root");
            return Err(err);
        }
        Ok(())
    }

    fn assert_no_blank_lines_in_span(
        &self,
        start_line: Option<usize>,
        end_line: usize,
        context: &str,
    ) -> ToonResult<()> {
        let Some(start_line) = start_line else {
            return Ok(());
        };
        if !self.strict {
            return Ok(());
        }
        if let Some(blank) = self
            .reader
            .blank_lines
            .iter()
            .find(|n| **n > start_line && **n < end_line)
        {
            return Err(ToonError::parse_error(
                *blank,
                1,
                format!("Blank lines inside {context} are not allowed in strict mode"),
            ));
        }
        Ok(())
    }

    fn insert_entry(
        &self,
        map: &mut Map<String, Value>,
        key: String,
        value: Value,
        line: &ParsedLine,
    ) -> ToonResult<()> {
        if self.strict && map.contains_key(&key) {
            return Err(err_at(line, format!("Duplicate sibling key \"{key}\"")));
        }
        // Non-strict duplicates resolve via last-write-wins (§14.3).
        map.insert(key, value);
        Ok(())
    }

    // #endregion

    /// Resolves a header parse result under the current mode: strict throws
    /// on `Invalid` and on strict-only defects; non-strict falls through to
    /// key-value parsing.
    fn resolve_array_header(
        &self,
        content: &str,
        line: &ParsedLine,
    ) -> ToonResult<Option<ResolvedHeader>> {
        match parse_array_header_line(content) {
            HeaderParse::NotHeader => Ok(None),
            HeaderParse::Invalid(reason) => {
                if self.strict {
                    Err(err_at(line, reason))
                } else {
                    Ok(None)
                }
            }
            HeaderParse::Header {
                header,
                inline_values,
                strict_error,
            } => {
                if self.strict {
                    if let Some(reason) = strict_error {
                        return Err(err_at(line, reason));
                    }
                }
                Ok(Some(ResolvedHeader {
                    header,
                    inline_values,
                }))
            }
        }
    }

    // #region Document dispatch (§5)

    fn decode_document(&mut self) -> ToonResult<Value> {
        let Some(first) = self.reader.peek()? else {
            return Ok(Value::Object(Map::new()));
        };
        let first = first.clone();

        if trim_spaces(&first.content) == "[]" {
            self.reader.next()?;
            self.assert_fully_consumed()?;
            return Ok(Value::Array(Vec::new()));
        }

        if is_array_header_content(&first.content) {
            if let Some(resolved) = self.resolve_array_header(&first.content, &first)? {
                self.reader.next()?;
                let value = self.decode_array_from_header(resolved, 0, &first)?;
                self.assert_fully_consumed()?;
                return Ok(value);
            }
        }

        self.reader.next()?;
        let following_depth = self.reader.peek()?.map(|line| line.depth);

        if following_depth.is_none() && !is_key_value_line(&first.content) {
            return parse_primitive_token(&first.content).map_err(|e| err_at(&first, e));
        }

        if !is_key_value_line(&first.content) && following_depth == Some(0) {
            return Err(err_at(
                &first,
                "Top-level document must start with a key-value or array-header line",
            ));
        }

        let mut map = Map::new();
        self.decode_key_value_into(&first, 0, &mut map)?;

        loop {
            let Some(line) = self.reader.peek()? else {
                break;
            };

            if line.depth != 0 {
                if self.strict {
                    return Err(over_indented_error(line, 0));
                }
                assert_not_scalar_line(line)?;
                self.reader.next()?;
                continue;
            }

            let line = self.reader.next()?.expect("peeked line exists");
            self.decode_key_value_into(&line, 0, &mut map)?;
        }

        Ok(Value::Object(map))
    }

    // #endregion

    // #region Decode rules

    /// Decodes one key-value line (§5.2 class 3/4) into `map`.
    fn decode_key_value_into(
        &mut self,
        line: &ParsedLine,
        base_depth: usize,
        map: &mut Map<String, Value>,
    ) -> ToonResult<()> {
        validate_depth(base_depth, MAX_DEPTH)?;
        let content = &line.content;

        if let Some(resolved) = self.resolve_array_header(content, line)? {
            match resolved.header.key.clone() {
                Some(key) => {
                    self.layout_push(&key);
                    let value = self.decode_array_from_header(resolved, base_depth, line);
                    self.layout_pop();
                    return self.insert_entry(map, key, value?, line);
                }
                None => {
                    if self.strict {
                        return Err(if resolved.header.keyed {
                            err_at(line, "Keyless keyed header is only valid at the document root")
                        } else {
                            err_at(
                                line,
                                "Keyless array header is only valid at the document root or as a \
                                 list item",
                            )
                        });
                    }
                }
            }
        }

        let (key, end) = parse_key_token(content).map_err(|e| err_at(line, e))?;
        let rest = trim_spaces(&content[end..]);

        if rest.is_empty() {
            let nested = match self.reader.peek()? {
                Some(next) if next.depth > base_depth => Some(next.clone()),
                _ => None,
            };
            let value = if let Some(next) = nested {
                self.assert_no_depth_jump(&next, base_depth)?;
                self.layout_push(&key);
                let fields = self.decode_object_fields(base_depth + 1);
                self.layout_pop();
                Value::Object(fields?)
            } else {
                Value::Object(Map::new())
            };
            return self.insert_entry(map, key, value, line);
        }

        if rest == "[]" {
            return self.insert_entry(map, key, Value::Array(Vec::new()), line);
        }

        let value = parse_primitive_token(rest).map_err(|e| err_at(line, e))?;
        self.insert_entry(map, key, value, line)
    }

    /// Decodes the fields of a nested object scope (§8).
    fn decode_object_fields(&mut self, base_depth: usize) -> ToonResult<Map<String, Value>> {
        let mut computed_depth: Option<usize> = None;
        let mut map = Map::new();

        loop {
            let Some(line) = self.reader.peek()? else {
                break;
            };
            if line.depth < base_depth {
                break;
            }

            let depth = *computed_depth.get_or_insert(line.depth);

            if line.depth == depth {
                let line = self.reader.next()?.expect("peeked line exists");
                self.decode_key_value_into(&line, depth, &mut map)?;
            } else if line.depth > depth {
                if self.strict {
                    return Err(over_indented_error(line, depth));
                }
                assert_not_scalar_line(line)?;
                self.reader.next()?;
            } else {
                break;
            }
        }

        Ok(map)
    }

    fn decode_array_from_header(
        &mut self,
        resolved: ResolvedHeader,
        base_depth: usize,
        header_line: &ParsedLine,
    ) -> ToonResult<Value> {
        validate_depth(base_depth, MAX_DEPTH)?;
        let ResolvedHeader {
            header,
            inline_values,
        } = resolved;

        // A keyed tabular header decodes to an object, not an array (§9.5).
        if header.keyed {
            return self.decode_keyed_object(&header, base_depth, header_line);
        }

        if let Some(inline) = inline_values {
            self.layout_record_array(&header, true);
            return self.decode_inline_primitive_array(&header, &inline, header_line);
        }

        if header.fields.is_some() {
            self.layout_record_array(&header, false);
            return self.decode_tabular_array(&header, base_depth, header_line);
        }

        self.layout_record_array(&header, false);
        self.decode_list_array(&header, base_depth, header_line)
    }

    fn decode_inline_primitive_array(
        &mut self,
        header: &ArrayHeaderInfo,
        inline_values: &str,
        header_line: &ParsedLine,
    ) -> ToonResult<Value> {
        let values = parse_delimited_values(inline_values, header.delimiter);
        self.assert_expected_count(
            values.len(),
            header.length,
            "inline-form values",
            header_line.line_number,
        )?;

        values
            .iter()
            .map(|token| parse_primitive_token(token).map_err(|e| err_at(header_line, e)))
            .collect::<ToonResult<Vec<Value>>>()
            .map(Value::Array)
    }

    /// Decodes a keyed tabular object (§9.5).
    fn decode_keyed_object(
        &mut self,
        header: &ArrayHeaderInfo,
        base_depth: usize,
        header_line: &ParsedLine,
    ) -> ToonResult<Value> {
        let entry_depth = base_depth + 1;
        let fields = header.fields.as_deref().expect("keyed header carries a field list");
        let leaf_field_count = count_leaf_fields(fields);

        let mut map = Map::new();
        let mut entry_count = 0usize;
        let mut start_line: Option<usize> = None;
        let mut last_entry_line = header_line.line_number;

        // A keyed scope ends only by dedent or end of input, so every line at
        // entry depth carrying an unquoted colon is an entry row.
        loop {
            let Some(line) = self.reader.peek()? else {
                break;
            };
            if line.depth <= base_depth {
                break;
            }

            if line.depth > entry_depth {
                if self.strict {
                    return Err(
                        err_at(line, "Unexpected indentation inside keyed tabular object")
                    );
                }
                self.reader.next()?;
                continue;
            }

            if find_unquoted_char(&line.content, b':', 0).is_none() {
                if self.strict {
                    return Err(
                        err_at(line, "Expected entry row inside keyed tabular object")
                    );
                }
                self.reader.next()?;
                continue;
            }

            let line = self.reader.next()?.expect("peeked line exists");
            start_line.get_or_insert(line.line_number);
            last_entry_line = line.line_number;

            let (key, end) = parse_key_token(&line.content).map_err(|e| err_at(&line, e))?;

            let cells_content = trim_spaces(&line.content[end..]);
            let values = if cells_content.is_empty() {
                Vec::new()
            } else {
                parse_delimited_values(cells_content, header.delimiter)
            };
            self.assert_expected_count(
                values.len(),
                leaf_field_count,
                "keyed entry cells",
                line.line_number,
            )?;

            let primitives = values
                .iter()
                .map(|token| parse_primitive_token(token).map_err(|e| err_at(&line, e)))
                .collect::<ToonResult<Vec<Value>>>()?;
            let value = object_from_fields(fields, &primitives);

            self.insert_entry(&mut map, key, value, &line)?;
            entry_count += 1;
        }

        self.assert_expected_count(entry_count, header.length, "keyed entries", last_entry_line)?;
        self.assert_no_blank_lines_in_span(start_line, last_entry_line, "keyed tabular object")?;

        Ok(Value::Object(map))
    }

    /// Decodes a tabular array (§9.3).
    fn decode_tabular_array(
        &mut self,
        header: &ArrayHeaderInfo,
        base_depth: usize,
        header_line: &ParsedLine,
    ) -> ToonResult<Value> {
        let row_depth = base_depth + 1;
        let fields = header.fields.as_deref().expect("tabular header carries a field list");
        let leaf_field_count = count_leaf_fields(fields);

        let mut rows = Vec::new();
        let mut start_line: Option<usize> = None;
        let mut last_row_line = header_line.line_number;

        // Only strict stops at N, leaving the surplus to the extra-rows check
        // below; non-strict reads on so a declared [N] never truncates (§14.1).
        while !self.strict || rows.len() < header.length {
            let Some(line) = self.reader.peek()? else {
                break;
            };
            if line.depth != row_depth || !is_data_row(&line.content, header.delimiter) {
                break;
            }

            let line = self.reader.next()?.expect("peeked line exists");
            start_line.get_or_insert(line.line_number);
            last_row_line = line.line_number;

            let values = parse_delimited_values(&line.content, header.delimiter);
            self.assert_expected_count(
                values.len(),
                leaf_field_count,
                "tabular row values",
                line.line_number,
            )?;

            let primitives = values
                .iter()
                .map(|token| parse_primitive_token(token).map_err(|e| err_at(&line, e)))
                .collect::<ToonResult<Vec<Value>>>()?;
            rows.push(object_from_fields(fields, &primitives));
        }

        self.assert_expected_count(rows.len(), header.length, "tabular rows", last_row_line)?;
        self.assert_no_blank_lines_in_span(start_line, last_row_line, "tabular array")?;

        if self.strict {
            if let Some(next) = self.reader.peek()? {
                if next.depth == row_depth
                    && !next.content.starts_with("- ")
                    && is_data_row(&next.content, header.delimiter)
                {
                    let err = err_at(
                        next,
                        format!("Expected {} tabular rows, but found more", header.length),
                    );
                    return Err(err);
                }
            }
        }

        Ok(Value::Array(rows))
    }

    /// Decodes an array in list form (§9.2, §9.4).
    fn decode_list_array(
        &mut self,
        header: &ArrayHeaderInfo,
        base_depth: usize,
        header_line: &ParsedLine,
    ) -> ToonResult<Value> {
        let item_depth = base_depth + 1;
        let mut items = Vec::new();
        let mut start_line: Option<usize> = None;
        let mut last_item_line = header_line.line_number;

        // Only strict stops at N, leaving the surplus to the extra-items check
        // below; non-strict reads on so a declared [N] never truncates (§14.1).
        while !self.strict || items.len() < header.length {
            let Some(line) = self.reader.peek()? else {
                break;
            };
            if line.depth != item_depth || !is_list_item_content(&line.content) {
                break;
            }

            start_line.get_or_insert(line.line_number);

            let index = items.len().to_string();
            self.layout_push(&index);
            let item = self.decode_list_item(item_depth);
            self.layout_pop();
            items.push(item?);

            last_item_line = self.reader.last_consumed_line;
        }

        self.assert_expected_count(items.len(), header.length, "list-form items", last_item_line)?;
        self.assert_no_blank_lines_in_span(start_line, last_item_line, "list-form array")?;

        if self.strict {
            if let Some(next) = self.reader.peek()? {
                if next.depth == item_depth && next.content.starts_with("- ") {
                    let err = err_at(
                        next,
                        format!("Expected {} list-form items, but found more", header.length),
                    );
                    return Err(err);
                }
            }
        }

        Ok(Value::Array(items))
    }

    /// Decodes one list item (§9.2, §9.4, §10).
    fn decode_list_item(&mut self, base_depth: usize) -> ToonResult<Value> {
        let line = self.reader.next()?.expect("caller peeked a list item");

        if line.content == "-" {
            return Ok(Value::Object(Map::new()));
        }
        let after_hyphen = &line.content[2..];
        let after_trimmed = trim_spaces(after_hyphen);

        if after_trimmed.is_empty() {
            return Ok(Value::Object(Map::new()));
        }

        if after_trimmed == "[]" {
            return Ok(Value::Array(Vec::new()));
        }

        let item_line = ParsedLine {
            raw: line.raw.clone(),
            content: after_hyphen.to_string(),
            depth: line.depth,
            line_number: line.line_number,
        };

        // Keyless header forms: `- [M]:` is the list item itself (§9.4);
        // there is no keyless keyed or fields-bearing list-item form.
        if is_array_header_content(after_hyphen) {
            if let Some(resolved) = self.resolve_array_header(after_hyphen, &item_line)? {
                if resolved.header.keyed || resolved.header.fields.is_some() {
                    if self.strict {
                        return Err(if resolved.header.keyed {
                            err_at(
                                &item_line,
                                "Keyless keyed header is only valid at the document root",
                            )
                        } else {
                            err_at(
                                &item_line,
                                "Keyless header with a field list is only valid at the document \
                                 root",
                            )
                        });
                    }
                } else {
                    return self.decode_array_from_header(resolved, base_depth, &item_line);
                }
            }
        }

        // A tabular array or keyed tabular object as the first field sits on
        // the hyphen line with rows at depth +2 (§10).
        if let Some(resolved) = self.resolve_array_header(after_hyphen, &item_line)? {
            if resolved.header.key.is_some() && resolved.header.fields.is_some() {
                let key = resolved.header.key.clone().expect("checked above");
                let mut map = Map::new();

                self.layout_push(&key);
                let value = self.decode_array_from_header(resolved, base_depth + 1, &item_line);
                self.layout_pop();
                self.insert_entry(&mut map, key, value?, &item_line)?;

                self.follow_sibling_fields(base_depth + 1, &mut map)?;
                return Ok(Value::Object(map));
            }
        }

        if is_key_value_content(after_hyphen) {
            let mut map = Map::new();
            self.decode_key_value_into(&item_line, base_depth + 1, &mut map)?;
            self.follow_sibling_fields(base_depth + 1, &mut map)?;
            return Ok(Value::Object(map));
        }

        parse_primitive_token(after_hyphen).map_err(|e| err_at(&item_line, e))
    }

    /// Decodes the remaining fields of a list-item object at depth +1 under
    /// the hyphen line (§10).
    fn follow_sibling_fields(
        &mut self,
        follow_depth: usize,
        map: &mut Map<String, Value>,
    ) -> ToonResult<()> {
        loop {
            let Some(line) = self.reader.peek()? else {
                break;
            };
            if line.depth != follow_depth || line.content.starts_with("- ") {
                break;
            }

            let line = self.reader.next()?.expect("peeked line exists");
            self.decode_key_value_into(&line, follow_depth, map)?;
        }
        Ok(())
    }

    // #endregion
}

/// Materializes one row's object by walking the field list in header order:
/// a leaf field takes the next cell; a nested field group materializes an
/// object from its subfields, applied recursively (§9.3).
fn object_from_fields(fields: &[FieldNode], primitives: &[Value]) -> Value {
    fn walk(fields: &[FieldNode], primitives: &[Value], cell_index: &mut usize) -> Value {
        let mut map = Map::new();
        for field in fields {
            match &field.children {
                Some(children) => {
                    map.insert(field.name.clone(), walk(children, primitives, cell_index));
                }
                None => {
                    // A non-strict width mismatch leaves trailing leaf fields
                    // with no cell; they are absent, not null (§14.1).
                    if *cell_index < primitives.len() {
                        map.insert(field.name.clone(), primitives[*cell_index].clone());
                        *cell_index += 1;
                    }
                }
            }
        }
        Value::Object(map)
    }

    let mut cell_index = 0;
    walk(fields, primitives, &mut cell_index)
}

// #endregion
