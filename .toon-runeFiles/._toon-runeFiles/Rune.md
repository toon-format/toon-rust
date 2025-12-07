# Rune

## Concatenated Directory Files

### File: toon-rune.txt

```txt
File: rune\grammar.pest
=======================

File: constants.rs
==================
//! Constants
use crate::types::Delimiter;

/// Characters that have structural meaning in TOON format.
pub const STRUCTURAL_CHARS: &[char] = &['[', ']', '{', '}', ':', '-'];

/// TOON keywords that must be quoted when used as strings.
pub const KEYWORDS: &[&str] = &["null", "true", "false"];

/// Default indentation size (2 spaces).
pub const DEFAULT_INDENT: usize = 2;

/// Default delimiter (comma).
pub const DEFAULT_DELIMITER: Delimiter = Delimiter::Comma;

/// Maximum nesting depth to prevent stack overflow.
pub const MAX_DEPTH: usize = 256;

/// Internal marker prefix for quoted keys containing dots.
/// Used during path expansion to distinguish quoted keys (which should remain
/// literal) from unquoted keys (which may be expanded).
/// This marker is added during parsing and removed during expansion.
pub(crate) const QUOTED_KEY_MARKER: char = '\x00';

#[inline]
pub fn is_structural_char(ch: char) -> bool {
    STRUCTURAL_CHARS.contains(&ch)
}

#[inline]
pub fn is_keyword(s: &str) -> bool {
    KEYWORDS.contains(&s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_structural_char() {
        assert!(is_structural_char('['));
        assert!(is_structural_char(']'));
        assert!(is_structural_char('{'));
        assert!(is_structural_char('}'));
        assert!(is_structural_char(':'));
        assert!(is_structural_char('-'));
        assert!(!is_structural_char('a'));
        assert!(!is_structural_char(','));
    }

    #[test]
    fn test_is_keyword() {
        assert!(is_keyword("null"));
        assert!(is_keyword("true"));
        assert!(is_keyword("false"));
        assert!(!is_keyword("hello"));
        assert!(!is_keyword("TRUE"));
    }
}

File: lib.rs
============
#![warn(rustdoc::missing_crate_level_docs)]
//! # TOON Format for Rust
//!
//! Token-Oriented Object Notation (TOON) is a compact, human-readable format
//! designed for passing structured data to Large Language Models with
//! significantly reduced token usage.
//!
//! This crate reserves the `toon-format` namespace for the official Rust
//! implementation. Full implementation coming soon!
//!
//! ## Resources
//!
//! - [TOON Specification](https://github.com/johannschopplich/toon/blob/main/SPEC.md)
//! - [Main Repository](https://github.com/johannschopplich/toon)
//! - [Other Implementations](https://github.com/johannschopplich/toon#other-implementations)
//!
//! ## Example Usage
//! ```rust
//! use rune_format::{encode_default, decode_default};
//! use serde_json::json;
//!
//! let data = json!({"name": "Alice", "age": 30});
//! let toon_string = encode_default(&data).unwrap();
//! let decoded: serde_json::Value = decode_default(&toon_string).unwrap();
//! assert_eq!(decoded["name"], "Alice");
//! assert_eq!(decoded["age"], 30);
//! ```
//! TOKEN_FORMAT is Copyright (c) 2025-PRESENT Shreyas S Bhat, Johann Schopplich
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod constants;
pub mod decode;
pub mod encode;
pub mod rune;
pub mod tui;
pub mod types;
pub mod utils;

pub use decode::{
    decode, decode_default, decode_no_coerce, decode_no_coerce_with_options, decode_strict,
    decode_strict_with_options,
};
pub use encode::{encode, encode_array, encode_default, encode_object};
pub use types::{DecodeOptions, Delimiter, EncodeOptions, Indent, ToonError};
pub use utils::{
    literal::{is_keyword, is_literal_like},
    normalize,
    string::{escape_string, is_valid_unquoted_key, needs_quoting},
};

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::{
        constants::is_keyword,
        decode::{decode_default, decode_strict},
        encode::{encode, encode_default},
        types::{Delimiter, EncodeOptions},
        utils::{escape_string, is_literal_like, needs_quoting, normalize},
    };

    #[test]
    fn test_round_trip_simple() {
        let original = json!({"name": "Alice", "age": 30});
        let encoded = encode_default(&original).unwrap();
        let decoded: Value = decode_default(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_round_trip_array() {
        let original = json!({"tags": ["reading", "gaming", "coding"]});
        let encoded = encode_default(&original).unwrap();
        let decoded: Value = decode_default(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_round_trip_tabular() {
        let original = json!({
            "users": [
                {"id": 1, "name": "Alice", "role": "admin"},
                {"id": 2, "name": "Bob", "role": "user"}
            ]
        });
        let encoded = encode_default(&original).unwrap();
        let decoded: Value = decode_default(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_custom_delimiter() {
        let original = json!({"tags": ["a", "b", "c"]});
        let opts = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
        let encoded = encode(&original, &opts).unwrap();
        assert!(encoded.contains("|"));

        let decoded: Value = decode_default(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_decode_strict_helper() {
        let input = "items[2]: a,b";
        assert!(decode_strict::<Value>(input).is_ok());

        let input = "items[3]: a,b";
        assert!(decode_strict::<Value>(input).is_err());
    }

    #[test]
    fn test_normalize_exported() {
        let value = json!(f64::NAN);
        let normalized = normalize(value.into());
        assert_eq!(serde_json::Value::from(normalized), json!(null));
    }

    #[test]
    fn test_utilities_exported() {
        assert!(is_keyword("null"));
        assert!(is_literal_like("true"));
        assert_eq!(escape_string("hello\nworld"), "hello\\nworld");
        assert!(needs_quoting("true", Delimiter::Comma.as_char()));
    }

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestUser {
        name: String,
        age: u32,
        active: bool,
    }

    #[test]
    fn test_encode_decode_simple_struct() {
        use crate::{decode_default, encode_default};

        let user = TestUser {
            name: "Alice".to_string(),
            age: 30,
            active: true,
        };

        let toon = encode_default(&user).unwrap();
        assert!(toon.contains("name: Alice"));
        assert!(toon.contains("age: 30"));
        assert!(toon.contains("active: true"));

        let decoded: TestUser = decode_default(&toon).unwrap();
        assert_eq!(user, decoded);
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestProduct {
        id: u64,
        name: String,
        tags: Vec<String>,
    }

    #[test]
    fn test_encode_decode_with_array() {
        use crate::{decode_default, encode_default};

        let product = TestProduct {
            id: 42,
            name: "Widget".to_string(),
            tags: vec!["electronics".to_string(), "gadgets".to_string()],
        };

        let toon = encode_default(&product).unwrap();
        let decoded: TestProduct = decode_default(&toon).unwrap();
        assert_eq!(product, decoded);
    }

    #[test]
    fn test_encode_decode_vec_of_structs() {
        use crate::{decode_default, encode_default};

        let users = vec![
            TestUser {
                name: "Alice".to_string(),
                age: 30,
                active: true,
            },
            TestUser {
                name: "Bob".to_string(),
                age: 25,
                active: false,
            },
        ];

        let toon = encode_default(&users).unwrap();
        let decoded: Vec<TestUser> = decode_default(&toon).unwrap();
        assert_eq!(users, decoded);
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct Nested {
        outer: OuterStruct,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct OuterStruct {
        inner: InnerStruct,
        value: i32,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct InnerStruct {
        data: String,
    }

    #[test]
    fn test_encode_decode_nested_structs() {
        use crate::{decode_default, encode_default};

        let nested = Nested {
            outer: OuterStruct {
                inner: InnerStruct {
                    data: "test".to_string(),
                },
                value: 42,
            },
        };

        let toon = encode_default(&nested).unwrap();
        let decoded: Nested = decode_default(&toon).unwrap();
        assert_eq!(nested, decoded);
    }

    #[test]
    fn test_round_trip_list_item_tabular_v3() {
        use crate::{decode_default, encode_default};

        let original = json!({
            "items": [
                {
                    "users": [
                        {"id": 1, "name": "Alice", "role": "admin"},
                        {"id": 2, "name": "Bob", "role": "user"}
                    ],
                    "status": "active",
                    "count": 2
                }
            ]
        });

        let encoded = encode_default(&original).unwrap();
        let decoded: Value = decode_default(&encoded).unwrap();

        assert_eq!(original, decoded);
    }

    #[test]
    fn test_round_trip_complex_list_item_tabular_v3() {
        use crate::{decode_default, encode_default};

        let original = json!({
            "data": [
                {
                    "records": [
                        {"id": 1, "value": "x", "score": 100},
                        {"id": 2, "value": "y", "score": 200}
                    ],
                    "total": 2,
                    "status": "active"
                },
                {
                    "records": [
                        {"id": 3, "value": "z", "score": 300}
                    ],
                    "total": 1,
                    "status": "pending"
                }
            ]
        });

        let encoded = encode_default(&original).unwrap();
        let decoded: Value = decode_default(&encoded).unwrap();

        assert_eq!(original, decoded);
    }

    #[test]
    fn test_round_trip_mixed_list_items_v3() {
        use crate::{decode_default, encode_default};

        let original = json!({
            "entries": [
                {
                    "type": "simple",
                    "value": 42
                },
                {
                    "people": [
                        {"name": "Alice", "age": 30},
                        {"name": "Bob", "age": 25}
                    ],
                    "type": "complex"
                },
                {
                    "tags": ["a", "b", "c"],
                    "type": "array"
                }
            ]
        });

        let encoded = encode_default(&original).unwrap();
        let decoded: Value = decode_default(&encoded).unwrap();

        assert_eq!(original, decoded);
    }
}

File: cli\main.rs
=================
use std::{
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use comfy_table::Table;
use rune_format::{
    decode, encode,
    types::{DecodeOptions, Delimiter, EncodeOptions, Indent, KeyFoldingMode, PathExpansionMode},
};
use serde::Serialize;
use tiktoken_rs::cl100k_base;

#[derive(Parser, Debug)]
#[command(
    name = "toon",
    version = env!("CARGO_PKG_VERSION"),
    author = env!("CARGO_PKG_AUTHORS"),
    about = "Encode JSON to TOON or decode TOON to JSON",
    long_about = "TOON Format CLI - Token-efficient JSON alternative for LLMs

EXAMPLES:
  toon --interactive                       # Launch interactive TUI
  toon -i                                  # Short flag for interactive mode

  toon input.json -o output.toon
  toon input.toon --json-indent 2
  cat data.json | toon -e --stats
  toon input.json --delimiter pipe
  toon input.toon -d --no-coerce

  toon input.json --fold-keys              # Collapse {a:{b:1}} to a.b: 1
  toon input.json --fold-keys --flatten-depth 2
  toon input.toon --expand-paths           # Expand a.b:1 to {\"a\":{\"b\":1}}",
    disable_help_subcommand = true
)]
struct Cli {
    input: Option<String>,

    #[arg(short, long, help = "Launch interactive TUI mode")]
    interactive: bool,

    #[arg(short, long, help = "Output file path")]
    output: Option<PathBuf>,

    #[arg(short, long, help = "Force encode mode (JSON → TOON)")]
    encode: bool,

    #[arg(short, long, help = "Force decode mode (TOON → JSON)")]
    decode: bool,

    #[arg(long, help = "Show token count and savings")]
    stats: bool,

    #[arg(long, value_parser = parse_delimiter, help = "Delimiter: comma, tab, or pipe")]
    delimiter: Option<Delimiter>,

    #[arg(long, value_parser = parse_indent, help = "Indentation spaces")]
    indent: Option<usize>,

    #[arg(long, help = "Disable strict validation (decode)")]
    no_strict: bool,

    #[arg(long, help = "Disable type coercion (decode)")]
    no_coerce: bool,

    #[arg(long, help = "Indent output JSON with N spaces")]
    json_indent: Option<usize>,

    #[arg(
        long,
        help = "Enable key folding (encode): collapse {a:{b:1}} → a.b: 1"
    )]
    fold_keys: bool,

    #[arg(long, help = "Max depth for key folding (default: unlimited)")]
    flatten_depth: Option<usize>,

    #[arg(
        long,
        help = "Enable path expansion (decode): expand a.b:1 → {\"a\":{\"b\":1}}"
    )]
    expand_paths: bool,
}

#[derive(Debug, PartialEq)]
enum Operation {
    Encode,
    Decode,
}

fn parse_indent(s: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|_| format!("'{s}' is not a valid number"))
}

fn parse_delimiter(s: &str) -> Result<Delimiter, String> {
    match s.to_lowercase().as_str() {
        "comma" | "," => Ok(Delimiter::Comma),
        "tab" | "\t" => Ok(Delimiter::Tab),
        "pipe" | "|" => Ok(Delimiter::Pipe),
        _ => Err(format!(
            "'{s}' is not a valid delimiter. Use 'comma', 'tab', or 'pipe'",
        )),
    }
}

fn get_input(file_arg: Option<&str>) -> Result<String> {
    let mut input_str = String::new();
    let mut reader: Box<dyn Read> = match file_arg {
        Some(path_str) if path_str != "-" => {
            let path = Path::new(path_str);
            Box::new(
                fs::File::open(path)
                    .with_context(|| format!("Failed to open: {}", path.display()))?,
            )
        }
        _ => Box::new(io::stdin()),
    };
    reader
        .read_to_string(&mut input_str)
        .context("Failed to read input")?;
    Ok(input_str)
}

fn write_output(output_path: Option<PathBuf>, content: &str) -> Result<()> {
    let mut writer: Box<dyn Write> = match output_path {
        Some(path) => Box::new(
            fs::File::create(&path)
                .with_context(|| format!("Failed to create: {}", path.display()))?,
        ),
        None => Box::new(io::stdout()),
    };
    writer
        .write_all(content.as_bytes())
        .context("Failed to write output")?;
    Ok(())
}

fn run_encode(cli: &Cli, input: &str) -> Result<()> {
    if input.trim().is_empty() {
        bail!("Input is empty. Provide JSON data via file or stdin");
    }

    let json_value: serde_json::Value =
        serde_json::from_str(input).context("Failed to parse input as JSON")?;

    let mut opts = EncodeOptions::new();
    if let Some(d) = cli.delimiter {
        opts = opts.with_delimiter(d);
    }
    if let Some(i) = cli.indent {
        opts = opts.with_indent(Indent::Spaces(i));
    }

    if cli.fold_keys {
        opts = opts.with_key_folding(KeyFoldingMode::Safe);
        if let Some(depth) = cli.flatten_depth {
            opts = opts.with_flatten_depth(depth);
        }
    }

    let toon_str = encode(&json_value, &opts).context("Failed to encode to TOON")?;

    write_output(cli.output.clone(), &toon_str)?;

    if cli.output.is_none() && !toon_str.ends_with('\n') {
        io::stdout().write_all(b"\n")?;
    }

    if cli.stats {
        let json_bytes = input.len();
        let toon_bytes = toon_str.len();
        let size_savings = 100.0 * (1.0 - (toon_bytes as f64 / json_bytes as f64));

        let bpe = cl100k_base().context("Failed to load tokenizer")?;
        let json_tokens = bpe.encode_with_special_tokens(input).len();
        let toon_tokens = bpe.encode_with_special_tokens(&toon_str).len();
        let token_savings = 100.0 * (1.0 - (toon_tokens as f64 / json_tokens as f64));

        eprintln!("\nStats:");
        let mut table = Table::new();
        table.set_header(vec!["Metric", "JSON", "TOON", "Savings"]);

        table.add_row(vec![
            "Tokens",
            &json_tokens.to_string(),
            &toon_tokens.to_string(),
            &format!("{token_savings:.2}%"),
        ]);

        table.add_row(vec![
            "Size (bytes)",
            &json_bytes.to_string(),
            &toon_bytes.to_string(),
            &format!("{size_savings:.2}%"),
        ]);

        eprintln!("\n{table}\n");
    }

    Ok(())
}

fn run_decode(cli: &Cli, input: &str) -> Result<()> {
    if input.trim().is_empty() {
        write_output(cli.output.clone(), "{}\n")?;
        return Ok(());
    }

    let mut opts = DecodeOptions::new();
    if cli.no_strict {
        opts = opts.with_strict(false);
    }
    if cli.no_coerce {
        opts = opts.with_coerce_types(false);
    }

    if cli.expand_paths {
        opts = opts.with_expand_paths(PathExpansionMode::Safe);
    }

    let json_value: serde_json::Value = decode(input, &opts).context("Failed to decode TOON")?;

    let output_json = match cli.json_indent {
        Some(n) if n > 0 => {
            let mut buf = Vec::new();
            let indent_str = " ".repeat(n);
            let formatter = serde_json::ser::PrettyFormatter::with_indent(indent_str.as_bytes());
            let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
            json_value
                .serialize(&mut ser)
                .context("Failed to serialize JSON")?;
            String::from_utf8(buf).context("Invalid UTF-8 in JSON output")?
        }
        _ => serde_json::to_string(&json_value).context("Failed to serialize JSON")?,
    };

    write_output(cli.output.clone(), &output_json)?;
    if cli.output.is_none() && !output_json.ends_with('\n') {
        io::stdout().write_all(b"\n")?;
    }
    Ok(())
}

fn determine_operation(cli: &Cli) -> Result<(Operation, bool)> {
    let mut from_stdin = false;
    let mut operation: Option<Operation> = None;

    if cli.interactive {
        bail!("Interactive mode cannot be combined with other operations");
    }

    if cli.encode && cli.decode {
        bail!("Cannot use --encode and --decode simultaneously");
    }

    if cli.encode {
        operation = Some(Operation::Encode);
    } else if cli.decode {
        operation = Some(Operation::Decode);
    }

    let file_arg = cli.input.as_deref();

    match file_arg {
        None | Some("-") => {
            from_stdin = true;
            if operation.is_none() {
                operation = Some(Operation::Encode);
            }
        }
        Some(path_str) => {
            if operation.is_none() {
                let path = Path::new(path_str);
                let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                match ext {
                    "json" => operation = Some(Operation::Encode),
                    "toon" => operation = Some(Operation::Decode),
                    _ => bail!(
                        "Cannot auto-detect operation for file: {}\nUse -e to encode or -d to \
                         decode",
                        path.display()
                    ),
                }
            }
        }
    }

    Ok((operation.unwrap(), from_stdin))
}

fn validate_flags(cli: &Cli, operation: &Operation) -> Result<()> {
    match operation {
        Operation::Encode => {
            if cli.no_strict {
                bail!("--no-strict is only valid for decode mode");
            }
            if cli.no_coerce {
                bail!("--no-coerce is only valid for decode mode");
            }
            if cli.json_indent.is_some() {
                bail!("--json-indent is only valid for decode mode");
            }
            if cli.expand_paths {
                bail!("--expand-paths is only valid for decode mode");
            }
        }
        Operation::Decode => {
            if cli.delimiter.is_some() {
                bail!("--delimiter is only valid for encode mode");
            }
            if cli.stats {
                bail!("--stats is only valid for encode mode");
            }
            if cli.indent.is_some() {
                bail!("--indent is only valid for encode mode");
            }
            if cli.fold_keys {
                bail!("--fold-keys is only valid for encode mode");
            }
            if cli.flatten_depth.is_some() {
                bail!("--flatten-depth is only valid for encode mode (use with --fold-keys)");
            }
        }
    }

    // Additional validation: flatten-depth requires fold-keys
    if cli.flatten_depth.is_some() && !cli.fold_keys {
        bail!("--flatten-depth requires --fold-keys to be enabled");
    }

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Check if interactive mode is requested
    if cli.interactive {
        return run_interactive();
    }

    let (operation, from_stdin) = determine_operation(&cli)?;
    validate_flags(&cli, &operation)?;

    let input = get_input(cli.input.as_deref()).with_context(|| {
        if from_stdin {
            "Failed to read from stdin"
        } else {
            "Failed to read input file"
        }
    })?;

    match operation {
        Operation::Encode => run_encode(&cli, &input)?,
        Operation::Decode => run_decode(&cli, &input)?,
    }

    Ok(())
}

fn run_interactive() -> Result<()> {
    rune_format::tui::run().context("Failed to run interactive TUI")?;
    Ok(())
}

File: decode\expansion.rs
=========================
use indexmap::IndexMap;

use crate::{
    constants::QUOTED_KEY_MARKER,
    types::{JsonValue as Value, PathExpansionMode, ToonError, ToonResult, is_identifier_segment},
};

pub fn should_expand_key(key: &str, mode: PathExpansionMode) -> Option<Vec<String>> {
    match mode {
        PathExpansionMode::Off => None,
        PathExpansionMode::Safe => {
            // Quoted keys with dots shouldn't be expanded (they were explicitly quoted)
            if key.starts_with(QUOTED_KEY_MARKER) {
                return None;
            }

            if !key.contains('.') {
                return None;
            }

            let segments: Vec<String> = key.split('.').map(String::from).collect();

            if segments.len() < 2 {
                return None;
            }

            // Only expand if all segments are valid identifiers (safety requirement)
            if segments.iter().all(|s| is_identifier_segment(s)) {
                Some(segments)
            } else {
                None
            }
        }
    }
}

pub fn deep_merge_value(
    target: &mut IndexMap<String, Value>,
    segments: &[String],
    value: Value,
    strict: bool,
) -> ToonResult<()> {
    if segments.is_empty() {
        return Ok(());
    }

    if segments.len() == 1 {
        let key = &segments[0];

        // Check for conflicts at leaf level
        if let Some(existing) = target.get(key) {
            if strict {
                return Err(ToonError::DeserializationError(format!(
                    "Path expansion conflict: key '{key}' already exists with value: {existing:?}",
                )));
            }
        }

        target.insert(key.clone(), value);
        return Ok(());
    }

    let first_key = &segments[0];
    let remaining_segments = &segments[1..];

    // Get or create nested object, handling type conflicts
    let nested_obj = if let Some(existing_value) = target.get_mut(first_key) {
        match existing_value {
            Value::Object(obj) => obj,
            _ => {
                if strict {
                    return Err(ToonError::DeserializationError(format!(
                        "Path expansion conflict: key '{first_key}' exists as non-object: \
                         {existing_value:?}",
                    )));
                }
                // Replace non-object with empty object in non-strict mode
                *existing_value = Value::Object(IndexMap::new());
                match existing_value {
                    Value::Object(obj) => obj,
                    _ => unreachable!(),
                }
            }
        }
    } else {
        target.insert(first_key.clone(), Value::Object(IndexMap::new()));
        match target.get_mut(first_key).unwrap() {
            Value::Object(obj) => obj,
            _ => unreachable!(),
        }
    };

    // Recurse into nested object
    deep_merge_value(nested_obj, remaining_segments, value, strict)
}

pub fn expand_paths_in_object(
    obj: IndexMap<String, Value>,
    mode: PathExpansionMode,
    strict: bool,
) -> ToonResult<IndexMap<String, Value>> {
    let mut result = IndexMap::new();

    for (key, mut value) in obj {
        // Expand nested objects first (depth-first)
        if let Value::Object(nested_obj) = value {
            value = Value::Object(expand_paths_in_object(nested_obj, mode, strict)?);
        }

        // Strip marker from quoted keys
        let clean_key = if key.starts_with(QUOTED_KEY_MARKER) {
            key.strip_prefix(QUOTED_KEY_MARKER).unwrap().to_string()
        } else {
            key.clone()
        };

        if let Some(segments) = should_expand_key(&key, mode) {
            deep_merge_value(&mut result, &segments, value, strict)?;
        } else {
            // Check for conflicts with expanded keys
            if let Some(existing) = result.get(&clean_key) {
                if strict {
                    return Err(ToonError::DeserializationError(format!(
                        "Key '{clean_key}' conflicts with existing value: {existing:?}",
                    )));
                }
            }
            result.insert(clean_key, value);
        }
    }

    Ok(result)
}

pub fn expand_paths_recursive(
    value: Value,
    mode: PathExpansionMode,
    strict: bool,
) -> ToonResult<Value> {
    match value {
        Value::Object(obj) => {
            let expanded = expand_paths_in_object(obj, mode, strict)?;
            Ok(Value::Object(expanded))
        }
        Value::Array(arr) => {
            let expanded: Result<Vec<_>, _> = arr
                .into_iter()
                .map(|v| expand_paths_recursive(v, mode, strict))
                .collect();
            Ok(Value::Array(expanded?))
        }
        _ => Ok(value),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_should_expand_key_off_mode() {
        assert!(should_expand_key("a.b.c", PathExpansionMode::Off).is_none());
    }

    #[test]
    fn test_should_expand_key_safe_mode() {
        // Valid expansions
        assert_eq!(
            should_expand_key("a.b", PathExpansionMode::Safe),
            Some(vec!["a".to_string(), "b".to_string()])
        );
        assert_eq!(
            should_expand_key("a.b.c", PathExpansionMode::Safe),
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()])
        );

        // No dots
        assert!(should_expand_key("simple", PathExpansionMode::Safe).is_none());

        // Invalid segments (not IdentifierSegments)
        assert!(should_expand_key("a.bad-key", PathExpansionMode::Safe).is_none());
        assert!(should_expand_key("123.key", PathExpansionMode::Safe).is_none());
    }

    #[test]
    fn test_deep_merge_simple() {
        let mut target = IndexMap::new();
        deep_merge_value(
            &mut target,
            &["a".to_string(), "b".to_string()],
            Value::from(json!(1)),
            true,
        )
        .unwrap();

        let expected = json!({"a": {"b": 1}});
        assert_eq!(Value::Object(target), Value::from(expected));
    }

    #[test]
    fn test_deep_merge_multiple_paths() {
        let mut target = IndexMap::new();

        deep_merge_value(
            &mut target,
            &["a".to_string(), "b".to_string()],
            Value::from(json!(1)),
            true,
        )
        .unwrap();

        deep_merge_value(
            &mut target,
            &["a".to_string(), "c".to_string()],
            Value::from(json!(2)),
            true,
        )
        .unwrap();

        let expected = json!({"a": {"b": 1, "c": 2}});
        assert_eq!(Value::Object(target), Value::from(expected));
    }

    #[test]
    fn test_deep_merge_conflict_strict() {
        let mut target = IndexMap::new();
        target.insert("a".to_string(), Value::from(json!({"b": 1})));

        let result = deep_merge_value(
            &mut target,
            &["a".to_string(), "b".to_string()],
            Value::from(json!(2)),
            true,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_deep_merge_conflict_non_strict() {
        let mut target = IndexMap::new();
        target.insert("a".to_string(), Value::from(json!({"b": 1})));

        deep_merge_value(
            &mut target,
            &["a".to_string(), "b".to_string()],
            Value::from(json!(2)),
            false,
        )
        .unwrap();

        let expected = json!({"a": {"b": 2}});
        assert_eq!(Value::Object(target), Value::from(expected));
    }

    #[test]
    fn test_expand_paths_in_object() {
        let mut obj = IndexMap::new();
        obj.insert("a.b.c".to_string(), Value::from(json!(1)));
        obj.insert("simple".to_string(), Value::from(json!(2)));

        let result = expand_paths_in_object(obj, PathExpansionMode::Safe, true).unwrap();

        let expected = json!({"a": {"b": {"c": 1}}, "simple": 2});
        assert_eq!(Value::Object(result), Value::from(expected));
    }

    #[test]
    fn test_expand_paths_with_merge() {
        let mut obj = IndexMap::new();
        obj.insert("a.b".to_string(), Value::from(json!(1)));
        obj.insert("a.c".to_string(), Value::from(json!(2)));

        let result = expand_paths_in_object(obj, PathExpansionMode::Safe, true).unwrap();

        let expected = json!({"a": {"b": 1, "c": 2}});
        assert_eq!(Value::Object(result), Value::from(expected));
    }
}

File: decode\mod.rs
===================
//! Decoder Implementation
pub mod expansion;
pub mod parser;
pub mod scanner;
pub mod validation;

use serde_json::Value;

use crate::types::{DecodeOptions, ToonResult};

/// Decode a TOON string into any deserializable type.
///
/// This function accepts any type implementing `serde::Deserialize`, including:
/// - Custom structs with `#[derive(Deserialize)]`
/// - `serde_json::Value`
/// - Built-in types (Vec, HashMap, etc.)
///
/// # Examples
///
/// **With custom structs:**
/// ```
/// use serde::Deserialize;
/// use rune_format::{
///     decode,
///     DecodeOptions,
/// };
///
/// #[derive(Deserialize, Debug, PartialEq)]
/// struct User {
///     name: String,
///     age: u32,
/// }
///
/// let toon = "name: Alice\nage: 30";
/// let user: User = decode(toon, &DecodeOptions::default())?;
/// assert_eq!(user.name, "Alice");
/// assert_eq!(user.age, 30);
/// # Ok::<(), rune_format::ToonError>(())
/// ```
///
/// **With JSON values:**
/// ```
/// use serde_json::{
///     json,
///     Value,
/// };
/// use rune_format::{
///     decode,
///     DecodeOptions,
/// };
///
/// let input = "name: Alice\nage: 30";
/// let result: Value = decode(input, &DecodeOptions::default())?;
/// assert_eq!(result["name"], json!("Alice"));
/// # Ok::<(), rune_format::ToonError>(())
/// ```
pub fn decode<T: serde::de::DeserializeOwned>(
    input: &str,
    options: &DecodeOptions,
) -> ToonResult<T> {
    let mut parser = parser::Parser::new(input, options.clone())?;
    let value = parser.parse()?;

    // Apply path expansion if enabled (v1.5 feature)
    use crate::types::PathExpansionMode;
    let final_value = if options.expand_paths != PathExpansionMode::Off {
        let json_value = crate::types::JsonValue::from(value);
        let expanded =
            expansion::expand_paths_recursive(json_value, options.expand_paths, options.strict)?;
        Value::from(expanded)
    } else {
        value
    };

    serde_json::from_value(final_value)
        .map_err(|e| crate::types::ToonError::DeserializationError(e.to_string()))
}

/// Decode with strict validation enabled (validates array lengths,
/// indentation).
///
/// # Examples
///
/// ```
/// use serde_json::{
///     json,
///     Value,
/// };
/// use rune_format::decode_strict;
///
/// // Valid array length
/// let result: Value = decode_strict("items[2]: a,b")?;
/// assert_eq!(result["items"], json!(["a", "b"]));
///
/// // Invalid array length (will error)
/// assert!(decode_strict::<Value>("items[3]: a,b").is_err());
/// # Ok::<(), rune_format::ToonError>(())
/// ```
pub fn decode_strict<T: serde::de::DeserializeOwned>(input: &str) -> ToonResult<T> {
    decode(input, &DecodeOptions::new().with_strict(true))
}

/// Decode with strict validation and additional options.
///
/// # Examples
///
/// ```
/// use serde_json::{
///     json,
///     Value,
/// };
/// use rune_format::{
///     decode_strict_with_options,
///     DecodeOptions,
/// };
///
/// let options = DecodeOptions::new()
///     .with_strict(true)
///     .with_delimiter(rune_format::Delimiter::Pipe);
/// let result: Value = decode_strict_with_options("items[2|]: a|b", &options)?;
/// assert_eq!(result["items"], json!(["a", "b"]));
/// # Ok::<(), rune_format::ToonError>(())
/// ```
pub fn decode_strict_with_options<T: serde::de::DeserializeOwned>(
    input: &str,
    options: &DecodeOptions,
) -> ToonResult<T> {
    let opts = options.clone().with_strict(true);
    decode(input, &opts)
}

/// Decode without type coercion (strings remain strings).
///
/// # Examples
///
/// ```
/// use serde_json::{
///     json,
///     Value,
/// };
/// use rune_format::decode_no_coerce;
///
/// // Without coercion: quoted strings that look like numbers stay as strings
/// let result: Value = decode_no_coerce("value: \"123\"")?;
/// assert_eq!(result["value"], json!("123"));
///
/// // With default coercion: unquoted "true" becomes boolean
/// let result: Value = rune_format::decode_default("value: true")?;
/// assert_eq!(result["value"], json!(true));
/// # Ok::<(), rune_format::ToonError>(())
/// ```
pub fn decode_no_coerce<T: serde::de::DeserializeOwned>(input: &str) -> ToonResult<T> {
    decode(input, &DecodeOptions::new().with_coerce_types(false))
}

/// Decode without type coercion and with additional options.
///
/// # Examples
///
/// ```
/// use serde_json::{
///     json,
///     Value,
/// };
/// use rune_format::{
///     decode_no_coerce_with_options,
///     DecodeOptions,
/// };
///
/// let options = DecodeOptions::new()
///     .with_coerce_types(false)
///     .with_strict(false);
/// let result: Value = decode_no_coerce_with_options("value: \"123\"", &options)?;
/// assert_eq!(result["value"], json!("123"));
/// # Ok::<(), rune_format::ToonError>(())
/// ```
pub fn decode_no_coerce_with_options<T: serde::de::DeserializeOwned>(
    input: &str,
    options: &DecodeOptions,
) -> ToonResult<T> {
    let opts = options.clone().with_coerce_types(false);
    decode(input, &opts)
}

/// Decode with default options (strict mode, type coercion enabled).
///
/// Works with any type implementing `serde::Deserialize`.
///
/// # Examples
///
/// **With structs:**
/// ```
/// use serde::Deserialize;
/// use rune_format::decode_default;
///
/// #[derive(Deserialize)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// let input = "name: Alice\nage: 30";
/// let person: Person = decode_default(input)?;
/// assert_eq!(person.name, "Alice");
/// # Ok::<(), rune_format::ToonError>(())
/// ```
///
/// **With JSON values:**
/// ```
/// use serde_json::{
///     json,
///     Value,
/// };
/// use rune_format::decode_default;
///
/// let input = "tags[3]: reading,gaming,coding";
/// let result: Value = decode_default(input)?;
/// assert_eq!(result["tags"], json!(["reading", "gaming", "coding"]));
/// # Ok::<(), rune_format::ToonError>(())
/// ```
pub fn decode_default<T: serde::de::DeserializeOwned>(input: &str) -> ToonResult<T> {
    decode(input, &DecodeOptions::default())
}

#[cfg(test)]
mod tests {
    use core::f64;

    use serde_json::json;

    use super::*;

    #[test]
    fn test_decode_null() {
        assert_eq!(decode_default::<Value>("null").unwrap(), json!(null));
    }

    #[test]
    fn test_decode_bool() {
        assert_eq!(decode_default::<Value>("true").unwrap(), json!(true));
        assert_eq!(decode_default::<Value>("false").unwrap(), json!(false));
    }

    #[test]
    fn test_decode_number() {
        assert_eq!(decode_default::<Value>("42").unwrap(), json!(42));
        assert_eq!(
            decode_default::<Value>("3.141592653589793").unwrap(),
            json!(f64::consts::PI)
        );
        assert_eq!(decode_default::<Value>("-5").unwrap(), json!(-5));
    }

    #[test]
    fn test_decode_string() {
        assert_eq!(decode_default::<Value>("hello").unwrap(), json!("hello"));
        assert_eq!(
            decode_default::<Value>("\"hello world\"").unwrap(),
            json!("hello world")
        );
    }

    #[test]
    fn test_decode_simple_object() {
        let input = "name: Alice\nage: 30";
        let result: Value = decode_default(input).unwrap();
        assert_eq!(result["name"], json!("Alice"));
        assert_eq!(result["age"], json!(30));
    }

    #[test]
    fn test_decode_primitive_array() {
        let input = "tags[3]: reading,gaming,coding";
        let result: Value = decode_default(input).unwrap();
        assert_eq!(result["tags"], json!(["reading", "gaming", "coding"]));
    }

    #[test]
    fn test_decode_tabular_array() {
        let input = "users[2]{id,name,role}:\n  1,Alice,admin\n  2,Bob,user";
        let result: Value = decode_default(input).unwrap();
        assert_eq!(
            result["users"],
            json!([
                {"id": 1, "name": "Alice", "role": "admin"},
                {"id": 2, "name": "Bob", "role": "user"}
            ])
        );
    }

    #[test]
    fn test_decode_empty_array() {
        let input = "items[0]:";
        let result: Value = decode_default(input).unwrap();
        assert_eq!(result["items"], json!([]));
    }

    #[test]
    fn test_decode_quoted_strings() {
        let input = "tags[3]: \"true\",\"42\",\"-3.14\"";
        let result: Value = decode_default(input).unwrap();
        assert_eq!(result["tags"], json!(["true", "42", "-3.14"]));
    }
}

File: decode\parser.rs
======================
use serde_json::{Map, Number, Value};

use crate::{
    constants::{KEYWORDS, MAX_DEPTH, QUOTED_KEY_MARKER},
    decode::{
        scanner::{Scanner, Token},
        validation,
    },
    types::{DecodeOptions, Delimiter, ErrorContext, ToonError, ToonResult},
    utils::validation::validate_depth,
};

/// Context for parsing arrays to determine correct indentation depth.
///
/// Arrays as the first field of list-item objects require special indentation:
/// their content (rows for tabular, items for non-uniform) appears at depth +2
/// relative to the hyphen line, while arrays in other contexts use depth +1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrayParseContext {
    /// Normal array parsing context (content at depth +1)
    Normal,

    /// Array as first field of list-item object
    /// (content at depth +2 relative to hyphen line)
    ListItemFirstField,
}

/// Parser that builds JSON values from a sequence of tokens.
#[allow(unused)]
pub struct Parser<'a> {
    scanner: Scanner,
    current_token: Token,
    options: DecodeOptions,
    delimiter: Option<Delimiter>,
    input: &'a str,
}

impl<'a> Parser<'a> {
    /// Create a new parser with the given input and options.
    pub fn new(input: &'a str, options: DecodeOptions) -> ToonResult<Self> {
        let mut scanner = Scanner::new(input);
        let chosen_delim = options.delimiter;
        scanner.set_active_delimiter(chosen_delim);
        let current_token = scanner.scan_token()?;

        Ok(Self {
            scanner,
            current_token,
            delimiter: chosen_delim,
            options,
            input,
        })
    }

    /// Parse the input into a JSON value.
    pub fn parse(&mut self) -> ToonResult<Value> {
        if self.options.strict {
            self.validate_indentation(self.scanner.get_last_line_indent())?;
        }
        let value = self.parse_value()?;

        // In strict mode, check for trailing content at root level
        if self.options.strict {
            self.skip_newlines()?;
            if !matches!(self.current_token, Token::Eof) {
                return Err(self
                    .parse_error_with_context(
                        "Multiple values at root level are not allowed in strict mode",
                    )
                    .with_suggestion("Wrap multiple values in an object or array"));
            }
        }

        Ok(value)
    }

    fn advance(&mut self) -> ToonResult<()> {
        self.current_token = self.scanner.scan_token()?;
        Ok(())
    }

    fn skip_newlines(&mut self) -> ToonResult<()> {
        while matches!(self.current_token, Token::Newline) {
            self.advance()?;
        }
        Ok(())
    }

    fn parse_value(&mut self) -> ToonResult<Value> {
        self.parse_value_with_depth(0)
    }

    fn parse_value_with_depth(&mut self, depth: usize) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        let had_newline = matches!(self.current_token, Token::Newline);
        self.skip_newlines()?;

        match &self.current_token {
            Token::Null => {
                // Peek ahead to see if this is a key (followed by ':') or a value
                let next_char_is_colon = matches!(self.scanner.peek(), Some(':'));
                if next_char_is_colon {
                    let key = KEYWORDS[0].to_string();
                    self.advance()?;
                    self.parse_object_with_initial_key(key, depth)
                } else {
                    self.advance()?;
                    Ok(Value::Null)
                }
            }
            Token::Bool(b) => {
                let next_char_is_colon = matches!(self.scanner.peek(), Some(':'));
                if next_char_is_colon {
                    let key = if *b {
                        KEYWORDS[1].to_string()
                    } else {
                        KEYWORDS[2].to_string()
                    };
                    self.advance()?;
                    self.parse_object_with_initial_key(key, depth)
                } else {
                    let val = *b;
                    self.advance()?;
                    Ok(Value::Bool(val))
                }
            }
            Token::Integer(i) => {
                let next_char_is_colon = matches!(self.scanner.peek(), Some(':'));
                if next_char_is_colon {
                    let key = i.to_string();
                    self.advance()?;
                    self.parse_object_with_initial_key(key, depth)
                } else {
                    let val = *i;
                    self.advance()?;
                    Ok(serde_json::Number::from(val).into())
                }
            }
            Token::Number(n) => {
                let next_char_is_colon = matches!(self.scanner.peek(), Some(':'));
                if next_char_is_colon {
                    let key = n.to_string();
                    self.advance()?;
                    self.parse_object_with_initial_key(key, depth)
                } else {
                    let val = *n;
                    self.advance()?;
                    // Normalize floats that are actually integers
                    if val.is_finite() && val.fract() == 0.0 && val.abs() <= i64::MAX as f64 {
                        Ok(serde_json::Number::from(val as i64).into())
                    } else {
                        Ok(serde_json::Number::from_f64(val)
                            .ok_or_else(|| {
                                ToonError::InvalidInput(format!("Invalid number: {val}"))
                            })?
                            .into())
                    }
                }
            }
            Token::String(s, _) => {
                let first = s.clone();
                self.advance()?;

                match &self.current_token {
                    Token::Colon | Token::LeftBracket => {
                        self.parse_object_with_initial_key(first, depth)
                    }
                    _ => {
                        // Strings on new indented lines could be missing colons (keys) or values
                        // Only error in strict mode when we know it's a new line
                        if self.options.strict && depth > 0 && had_newline {
                            return Err(self
                                .parse_error_with_context(format!(
                                    "Expected ':' after '{first}' in object context"
                                ))
                                .with_suggestion(
                                    "Add ':' after the key, or place the value on the same line \
                                     as the parent key",
                                ));
                        }

                        // Root-level string value - join consecutive tokens
                        let mut accumulated = first;
                        while let Token::String(next, _) = &self.current_token {
                            if !accumulated.is_empty() {
                                accumulated.push(' ');
                            }
                            accumulated.push_str(next);
                            self.advance()?;
                        }
                        Ok(Value::String(accumulated))
                    }
                }
            }
            Token::LeftBracket => self.parse_root_array(depth),
            Token::Eof => Ok(Value::Object(Map::new())),
            _ => self.parse_object(depth),
        }
    }

    fn parse_object(&mut self, depth: usize) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        let mut obj = Map::new();
        // Track the indentation of the first key to ensure all keys align
        let mut base_indent: Option<usize> = None;

        loop {
            while matches!(self.current_token, Token::Newline) {
                self.advance()?;
            }

            if matches!(self.current_token, Token::Eof) {
                break;
            }

            let current_indent = self.scanner.get_last_line_indent();

            if self.options.strict {
                self.validate_indentation(current_indent)?;
            }

            // Once we've seen the first key, all subsequent keys must match its indent
            if let Some(expected) = base_indent {
                if current_indent != expected {
                    break;
                }
            } else {
                base_indent = Some(current_indent);
            }

            let key = match &self.current_token {
                Token::String(s, was_quoted) => {
                    // Mark quoted keys containing dots with a special prefix
                    // so path expansion can skip them
                    if *was_quoted && s.contains('.') {
                        format!("{QUOTED_KEY_MARKER}{s}")
                    } else {
                        s.clone()
                    }
                }
                _ => {
                    return Err(self
                        .parse_error_with_context(format!(
                            "Expected key, found {:?}",
                            self.current_token
                        ))
                        .with_suggestion("Object keys must be strings"));
                }
            };
            self.advance()?;

            let value = if matches!(self.current_token, Token::LeftBracket) {
                self.parse_array(depth)?
            } else {
                if !matches!(self.current_token, Token::Colon) {
                    return Err(self
                        .parse_error_with_context(format!(
                            "Expected ':' or '[', found {:?}",
                            self.current_token
                        ))
                        .with_suggestion("Use ':' for object values or '[' for arrays"));
                }
                self.advance()?;
                self.parse_field_value(depth)?
            };

            obj.insert(key, value);
        }

        Ok(Value::Object(obj))
    }

    fn parse_object_with_initial_key(&mut self, key: String, depth: usize) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        let mut obj = Map::new();
        let mut base_indent: Option<usize> = None;

        // Validate indentation for the initial key if in strict mode
        if self.options.strict {
            let current_indent = self.scanner.get_last_line_indent();
            self.validate_indentation(current_indent)?;
        }

        if matches!(self.current_token, Token::LeftBracket) {
            let value = self.parse_array(depth)?;
            obj.insert(key, value);
        } else {
            if !matches!(self.current_token, Token::Colon) {
                return Err(self.parse_error_with_context(format!(
                    "Expected ':', found {:?}",
                    self.current_token
                )));
            }
            self.advance()?;

            let value = self.parse_field_value(depth)?;
            obj.insert(key, value);
        }

        loop {
            // Skip newlines and check if the next line belongs to this object
            while matches!(self.current_token, Token::Newline) {
                self.advance()?;

                if !self.options.strict {
                    while matches!(self.current_token, Token::Newline) {
                        self.advance()?;
                    }
                }

                if matches!(self.current_token, Token::Newline) {
                    continue;
                }

                let next_indent = self.scanner.get_last_line_indent();

                // Check if the next line is at the right indentation level
                let should_continue = if let Some(expected) = base_indent {
                    next_indent == expected
                } else {
                    // First field: use depth-based expected indent
                    let current_depth_indent = self.options.indent.get_spaces() * depth;
                    next_indent == current_depth_indent
                };

                if !should_continue {
                    break;
                }
            }

            if matches!(self.current_token, Token::Eof) {
                break;
            }

            if !matches!(self.current_token, Token::String(_, _)) {
                break;
            }

            if matches!(self.current_token, Token::Eof) {
                break;
            }

            let current_indent = self.scanner.get_last_line_indent();

            if let Some(expected) = base_indent {
                if current_indent != expected {
                    break;
                }
            } else {
                // verify first additional field matches expected depth
                let expected_depth_indent = self.options.indent.get_spaces() * depth;
                if current_indent != expected_depth_indent {
                    break;
                }
            }

            if self.options.strict {
                self.validate_indentation(current_indent)?;
            }

            if base_indent.is_none() {
                base_indent = Some(current_indent);
            }

            let key = match &self.current_token {
                Token::String(s, was_quoted) => {
                    // Mark quoted keys containing dots with a special prefix
                    // so path expansion can skip them
                    if *was_quoted && s.contains('.') {
                        format!("{QUOTED_KEY_MARKER}{s}")
                    } else {
                        s.clone()
                    }
                }
                _ => break,
            };
            self.advance()?;

            let value = if matches!(self.current_token, Token::LeftBracket) {
                self.parse_array(depth)?
            } else {
                if !matches!(self.current_token, Token::Colon) {
                    break;
                }
                self.advance()?;
                self.parse_field_value(depth)?
            };

            obj.insert(key, value);
        }

        Ok(Value::Object(obj))
    }

    fn parse_field_value(&mut self, depth: usize) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        if matches!(self.current_token, Token::Newline | Token::Eof) {
            let has_children = if matches!(self.current_token, Token::Newline) {
                let current_depth_indent = self.options.indent.get_spaces() * (depth + 1);
                let next_indent = self.scanner.count_leading_spaces();
                next_indent >= current_depth_indent
            } else {
                false
            };

            if has_children {
                self.parse_value_with_depth(depth + 1)
            } else {
                Ok(Value::Object(Map::new()))
            }
        } else if matches!(self.current_token, Token::LeftBracket) {
            self.parse_value_with_depth(depth + 1)
        } else {
            // Check if there's more content after the current token
            let (rest, had_space) = self.scanner.read_rest_of_line_with_space_info();

            let result = if rest.is_empty() {
                // Single token - convert directly to avoid redundant parsing
                match &self.current_token {
                    Token::String(s, _) => Ok(Value::String(s.clone())),
                    Token::Integer(i) => Ok(serde_json::Number::from(*i).into()),
                    Token::Number(n) => {
                        let val = *n;
                        if val.is_finite() && val.fract() == 0.0 && val.abs() <= i64::MAX as f64 {
                            Ok(serde_json::Number::from(val as i64).into())
                        } else {
                            Ok(serde_json::Number::from_f64(val)
                                .ok_or_else(|| {
                                    ToonError::InvalidInput(format!("Invalid number: {val}"))
                                })?
                                .into())
                        }
                    }
                    Token::Bool(b) => Ok(Value::Bool(*b)),
                    Token::Null => Ok(Value::Null),
                    _ => Err(self.parse_error_with_context("Unexpected token after colon")),
                }
            } else {
                // Multi-token value - reconstruct and re-parse as complete string
                let mut value_str = String::new();

                match &self.current_token {
                    Token::String(s, true) => {
                        // Quoted strings need quotes preserved for re-parsing
                        value_str.push('"');
                        value_str.push_str(&crate::utils::escape_string(s));
                        value_str.push('"');
                    }
                    Token::String(s, false) => value_str.push_str(s),
                    Token::Integer(i) => value_str.push_str(&i.to_string()),
                    Token::Number(n) => value_str.push_str(&n.to_string()),
                    Token::Bool(b) => value_str.push_str(if *b { "true" } else { "false" }),
                    Token::Null => value_str.push_str("null"),
                    _ => {
                        return Err(self.parse_error_with_context("Unexpected token after colon"));
                    }
                }

                // Only add space if there was whitespace in the original input
                if had_space {
                    value_str.push(' ');
                }
                value_str.push_str(&rest);

                let token = self.scanner.parse_value_string(&value_str)?;
                match token {
                    Token::String(s, _) => Ok(Value::String(s)),
                    Token::Integer(i) => Ok(serde_json::Number::from(i).into()),
                    Token::Number(n) => {
                        if n.is_finite() && n.fract() == 0.0 && n.abs() <= i64::MAX as f64 {
                            Ok(serde_json::Number::from(n as i64).into())
                        } else {
                            Ok(serde_json::Number::from_f64(n)
                                .ok_or_else(|| {
                                    ToonError::InvalidInput(format!("Invalid number: {n}"))
                                })?
                                .into())
                        }
                    }
                    Token::Bool(b) => Ok(Value::Bool(b)),
                    Token::Null => Ok(Value::Null),
                    _ => Err(ToonError::InvalidInput("Unexpected token type".to_string())),
                }
            }?;

            self.current_token = self.scanner.scan_token()?;
            Ok(result)
        }
    }

    fn parse_root_array(&mut self, depth: usize) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        if !matches!(self.current_token, Token::LeftBracket) {
            return Err(self.parse_error_with_context("Expected '[' at the start of root array"));
        }

        self.parse_array(depth)
    }

    fn parse_array_header(
        &mut self,
    ) -> ToonResult<(usize, Option<Delimiter>, Option<Vec<String>>)> {
        if !matches!(self.current_token, Token::LeftBracket) {
            return Err(self.parse_error_with_context("Expected '['"));
        }
        self.advance()?;

        // Parse array length (plain integer only)
        // Supports formats: [N], [N|], [N\t] (no # marker)
        let length = if let Token::Integer(n) = &self.current_token {
            *n as usize
        } else if let Token::String(s, _) = &self.current_token {
            // Check if string starts with # - this marker is not supported
            if s.starts_with('#') {
                return Err(self
                    .parse_error_with_context(
                        "Length marker '#' is not supported. Use [N] format instead of [#N]",
                    )
                    .with_suggestion("Remove the '#' prefix from the array length"));
            }

            // Plain string that's a number: "3"
            s.parse::<usize>().map_err(|_| {
                self.parse_error_with_context(format!("Expected array length, found: {s}"))
            })?
        } else {
            return Err(self.parse_error_with_context(format!(
                "Expected array length, found {:?}",
                self.current_token
            )));
        };

        self.advance()?;

        // Check for optional delimiter after length
        let detected_delim = match &self.current_token {
            Token::Delimiter(d) => {
                let delim = *d;
                self.advance()?;
                Some(delim)
            }
            Token::String(s, _) if s == "," => {
                self.advance()?;
                Some(Delimiter::Comma)
            }
            Token::String(s, _) if s == "|" => {
                self.advance()?;
                Some(Delimiter::Pipe)
            }
            Token::String(s, _) if s == "\t" => {
                self.advance()?;
                Some(Delimiter::Tab)
            }
            _ => None,
        };

        // Default to comma if no delimiter specified
        let active_delim = detected_delim.or(Some(Delimiter::Comma));

        self.scanner.set_active_delimiter(active_delim);

        if !matches!(self.current_token, Token::RightBracket) {
            return Err(self.parse_error_with_context(format!(
                "Expected ']', found {:?}",
                self.current_token
            )));
        }
        self.advance()?;

        let fields = if matches!(self.current_token, Token::LeftBrace) {
            self.advance()?;
            let mut fields = Vec::new();

            loop {
                match &self.current_token {
                    Token::String(s, _) => {
                        fields.push(s.clone());
                        self.advance()?;

                        if matches!(self.current_token, Token::RightBrace) {
                            break;
                        }

                        if matches!(self.current_token, Token::Delimiter(_)) {
                            self.advance()?;
                        } else {
                            return Err(self.parse_error_with_context(format!(
                                "Expected delimiter or '}}', found {:?}",
                                self.current_token
                            )));
                        }
                    }
                    Token::RightBrace => break,
                    _ => {
                        return Err(self.parse_error_with_context(format!(
                            "Expected field name, found {:?}",
                            self.current_token
                        )));
                    }
                }
            }

            self.advance()?;
            Some(fields)
        } else {
            None
        };

        if !matches!(self.current_token, Token::Colon) {
            return Err(self.parse_error_with_context("Expected ':' after array header"));
        }
        self.advance()?;

        Ok((length, detected_delim, fields))
    }

    fn parse_array(&mut self, depth: usize) -> ToonResult<Value> {
        self.parse_array_with_context(depth, ArrayParseContext::Normal)
    }

    fn parse_array_with_context(
        &mut self,
        depth: usize,
        context: ArrayParseContext,
    ) -> ToonResult<Value> {
        validate_depth(depth, MAX_DEPTH)?;

        let (length, _detected_delim, fields) = self.parse_array_header()?;

        if let Some(fields) = fields {
            validation::validate_field_list(&fields)?;
            self.parse_tabular_array(length, fields, depth, context)
        } else {
            // Non-tabular arrays as first field of list items require depth adjustment
            // (items at depth +2 relative to hyphen, not the usual +1)
            let adjusted_depth = match context {
                ArrayParseContext::Normal => depth,
                ArrayParseContext::ListItemFirstField => depth + 1,
            };
            self.parse_regular_array(length, adjusted_depth)
        }
    }

    fn parse_tabular_array(
        &mut self,
        length: usize,
        fields: Vec<String>,
        depth: usize,
        context: ArrayParseContext,
    ) -> ToonResult<Value> {
        let mut rows = Vec::new();

        if !matches!(self.current_token, Token::Newline) {
            return Err(self
                .parse_error_with_context("Expected newline after tabular array header")
                .with_suggestion("Tabular arrays must have rows on separate lines"));
        }
        self.skip_newlines()?;

        for row_index in 0..length {
            if matches!(self.current_token, Token::Eof) {
                if self.options.strict {
                    return Err(self.parse_error_with_context(format!(
                        "Expected {} rows, but got {} before EOF",
                        length,
                        rows.len()
                    )));
                }
                break;
            }

            let current_indent = self.scanner.get_last_line_indent();

            // Tabular arrays as first field of list-item objects require rows at depth +2
            // (relative to hyphen), while normal tabular arrays use depth +1
            let row_depth_offset = match context {
                ArrayParseContext::Normal => 1,
                ArrayParseContext::ListItemFirstField => 2,
            };
            let expected_indent = self.options.indent.get_spaces() * (depth + row_depth_offset);

            if self.options.strict {
                self.validate_indentation(current_indent)?;

                if current_indent != expected_indent {
                    return Err(self.parse_error_with_context(format!(
                        "Invalid indentation for tabular row: expected {expected_indent} spaces, \
                         found {current_indent}"
                    )));
                }
            }

            let mut row = Map::new();

            for (field_index, field) in fields.iter().enumerate() {
                // Skip delimiter before each field except the first
                if field_index > 0 {
                    if matches!(self.current_token, Token::Delimiter(_)) {
                        self.advance()?;
                    } else {
                        return Err(self
                            .parse_error_with_context(format!(
                                "Expected delimiter, found {:?}",
                                self.current_token
                            ))
                            .with_suggestion(format!(
                                "Tabular row {} field {} needs a delimiter",
                                row_index + 1,
                                field_index + 1
                            )));
                    }
                }

                // Empty values show up as delimiters or newlines
                let value = if matches!(self.current_token, Token::Delimiter(_))
                    || matches!(self.current_token, Token::Newline | Token::Eof)
                {
                    Value::String(String::new())
                } else {
                    self.parse_tabular_field_value()?
                };

                row.insert(field.clone(), value);

                // Validate row completeness
                if field_index < fields.len() - 1 {
                    // Not the last field - shouldn't hit newline yet
                    if matches!(self.current_token, Token::Newline | Token::Eof) {
                        if self.options.strict {
                            return Err(self
                                .parse_error_with_context(format!(
                                    "Tabular row {}: expected {} values, but found only {}",
                                    row_index + 1,
                                    fields.len(),
                                    field_index + 1
                                ))
                                .with_suggestion(format!(
                                    "Row {} should have exactly {} values",
                                    row_index + 1,
                                    fields.len()
                                )));
                        } else {
                            // Fill remaining fields with null in non-strict mode
                            for field in fields.iter().skip(field_index + 1) {
                                row.insert(field.clone(), Value::Null);
                            }
                            break;
                        }
                    }
                } else if !matches!(self.current_token, Token::Newline | Token::Eof)
                    && matches!(self.current_token, Token::Delimiter(_))
                {
                    // Last field but there's another delimiter - too many values
                    return Err(self
                        .parse_error_with_context(format!(
                            "Tabular row {}: expected {} values, but found extra values",
                            row_index + 1,
                            fields.len()
                        ))
                        .with_suggestion(format!(
                            "Row {} should have exactly {} values",
                            row_index + 1,
                            fields.len()
                        )));
                }
            }

            if !self.options.strict && row.len() < fields.len() {
                for field in fields.iter().skip(row.len()) {
                    row.insert(field.clone(), Value::Null);
                }
            }

            rows.push(Value::Object(row));

            if matches!(self.current_token, Token::Eof) {
                break;
            }

            if !matches!(self.current_token, Token::Newline) {
                if !self.options.strict {
                    while !matches!(self.current_token, Token::Newline | Token::Eof) {
                        self.advance()?;
                    }
                    if matches!(self.current_token, Token::Eof) {
                        break;
                    }
                } else {
                    return Err(self.parse_error_with_context(format!(
                        "Expected newline after tabular row {}",
                        row_index + 1
                    )));
                }
            }

            if row_index + 1 < length {
                self.advance()?;
                if self.options.strict && matches!(self.current_token, Token::Newline) {
                    return Err(self.parse_error_with_context(
                        "Blank lines are not allowed inside tabular arrays in strict mode",
                    ));
                }

                self.skip_newlines()?;
            } else if matches!(self.current_token, Token::Newline) {
                // After the last row, check if there are extra rows
                self.advance()?;
                self.skip_newlines()?;

                let expected_indent = self.options.indent.get_spaces() * (depth + 1);
                let actual_indent = self.scanner.get_last_line_indent();

                // If something at the same indent level, it might be a new row (error)
                // unless it's a key-value pair (which belongs to parent)
                if actual_indent == expected_indent && !matches!(self.current_token, Token::Eof) {
                    let is_key_value = matches!(self.current_token, Token::String(_, _))
                        && matches!(self.scanner.peek(), Some(':'));

                    if !is_key_value {
                        return Err(self.parse_error_with_context(format!(
                            "Array length mismatch: expected {length} rows, but more rows found",
                        )));
                    }
                }
            }
        }

        validation::validate_array_length(length, rows.len())?;

        Ok(Value::Array(rows))
    }

    fn parse_regular_array(&mut self, length: usize, depth: usize) -> ToonResult<Value> {
        let mut items = Vec::new();

        match &self.current_token {
            Token::Newline => {
                self.skip_newlines()?;

                let expected_indent = self.options.indent.get_spaces() * (depth + 1);

                for i in 0..length {
                    let current_indent = self.scanner.get_last_line_indent();
                    if self.options.strict {
                        self.validate_indentation(current_indent)?;

                        if current_indent != expected_indent {
                            return Err(self.parse_error_with_context(format!(
                                "Invalid indentation for list item: expected {expected_indent} \
                                 spaces, found {current_indent}"
                            )));
                        }
                    }
                    if !matches!(self.current_token, Token::Dash) {
                        return Err(self
                            .parse_error_with_context(format!(
                                "Expected '-' for list item, found {:?}",
                                self.current_token
                            ))
                            .with_suggestion(format!(
                                "List arrays need '-' prefix for each item (item {} of {})",
                                i + 1,
                                length
                            )));
                    }
                    self.advance()?;

                    let value = if matches!(self.current_token, Token::Newline | Token::Eof) {
                        Value::Object(Map::new())
                    } else if matches!(self.current_token, Token::LeftBracket) {
                        self.parse_array(depth + 1)?
                    } else if let Token::String(s, _) = &self.current_token {
                        let key = s.clone();
                        self.advance()?;

                        if matches!(self.current_token, Token::Colon | Token::LeftBracket) {
                            // This is an object: key followed by colon or array bracket
                            // First field of list-item object may be an array requiring special
                            // indentation
                            let first_value = if matches!(self.current_token, Token::LeftBracket) {
                                // Array directly after key (e.g., "- key[N]:")
                                // Use ListItemFirstField context to apply correct indentation
                                self.parse_array_with_context(
                                    depth + 1,
                                    ArrayParseContext::ListItemFirstField,
                                )?
                            } else {
                                self.advance()?;
                                // Handle nested arrays: "key: [2]: ..."
                                if matches!(self.current_token, Token::LeftBracket) {
                                    // Array after colon - not directly on hyphen line, use normal
                                    // context
                                    self.parse_array(depth + 2)?
                                } else {
                                    self.parse_field_value(depth + 2)?
                                }
                            };

                            let mut obj = Map::new();
                            obj.insert(key, first_value);

                            let field_indent = self.options.indent.get_spaces() * (depth + 2);

                            // Check if there are more fields at the same indentation level
                            let should_parse_more_fields =
                                if matches!(self.current_token, Token::Newline) {
                                    let next_indent = self.scanner.count_leading_spaces();

                                    if next_indent < field_indent {
                                        false
                                    } else {
                                        self.advance()?;

                                        if !self.options.strict {
                                            self.skip_newlines()?;
                                        }
                                        true
                                    }
                                } else if matches!(self.current_token, Token::String(_, _)) {
                                    // When already positioned at a field key, check its indent
                                    let current_indent = self.scanner.get_last_line_indent();
                                    current_indent == field_indent
                                } else {
                                    false
                                };

                            // Parse additional fields if they're at the right indentation
                            if should_parse_more_fields {
                                while !matches!(self.current_token, Token::Eof) {
                                    let current_indent = self.scanner.get_last_line_indent();

                                    if current_indent < field_indent {
                                        break;
                                    }

                                    if current_indent != field_indent && self.options.strict {
                                        break;
                                    }

                                    // Stop if we hit the next list item
                                    if matches!(self.current_token, Token::Dash) {
                                        break;
                                    }

                                    let field_key = match &self.current_token {
                                        Token::String(s, _) => s.clone(),
                                        _ => break,
                                    };
                                    self.advance()?;

                                    let field_value =
                                        if matches!(self.current_token, Token::LeftBracket) {
                                            self.parse_array(depth + 2)?
                                        } else if matches!(self.current_token, Token::Colon) {
                                            self.advance()?;
                                            if matches!(self.current_token, Token::LeftBracket) {
                                                self.parse_array(depth + 2)?
                                            } else {
                                                self.parse_field_value(depth + 2)?
                                            }
                                        } else {
                                            break;
                                        };

                                    obj.insert(field_key, field_value);

                                    if matches!(self.current_token, Token::Newline) {
                                        let next_indent = self.scanner.count_leading_spaces();
                                        if next_indent < field_indent {
                                            break;
                                        }
                                        self.advance()?;
                                        if !self.options.strict {
                                            self.skip_newlines()?;
                                        }
                                    } else {
                                        break;
                                    }
                                }
                            }

                            Value::Object(obj)
                        } else if matches!(self.current_token, Token::LeftBracket) {
                            // Array as object value: "key[2]: ..."
                            let array_value = self.parse_array(depth + 1)?;
                            let mut obj = Map::new();
                            obj.insert(key, array_value);
                            Value::Object(obj)
                        } else {
                            // Plain string value
                            Value::String(key)
                        }
                    } else {
                        self.parse_primitive()?
                    };

                    items.push(value);

                    if items.len() < length {
                        if matches!(self.current_token, Token::Newline) {
                            self.advance()?;

                            if self.options.strict && matches!(self.current_token, Token::Newline) {
                                return Err(self.parse_error_with_context(
                                    "Blank lines are not allowed inside list arrays in strict mode",
                                ));
                            }

                            self.skip_newlines()?;
                        } else if !matches!(self.current_token, Token::Dash) {
                            return Err(self.parse_error_with_context(format!(
                                "Expected newline or next list item after list item {}",
                                i + 1
                            )));
                        }
                    } else if matches!(self.current_token, Token::Newline) {
                        // After the last item, check for extra items
                        self.advance()?;
                        self.skip_newlines()?;

                        let list_indent = self.options.indent.get_spaces() * (depth + 1);
                        let actual_indent = self.scanner.get_last_line_indent();
                        // If we see another dash at the same indent, there are too many items
                        if actual_indent == list_indent && matches!(self.current_token, Token::Dash)
                        {
                            return Err(self.parse_error_with_context(format!(
                                "Array length mismatch: expected {length} items, but more items \
                                 found",
                            )));
                        }
                    }
                }
            }
            _ => {
                for i in 0..length {
                    if i > 0 {
                        if matches!(self.current_token, Token::Delimiter(_)) {
                            self.advance()?;
                        } else {
                            return Err(self
                                .parse_error_with_context(format!(
                                    "Expected delimiter, found {:?}",
                                    self.current_token
                                ))
                                .with_suggestion(format!(
                                    "Expected delimiter between items (item {} of {})",
                                    i + 1,
                                    length
                                )));
                        }
                    }

                    let value = if matches!(self.current_token, Token::Delimiter(_))
                        || (matches!(self.current_token, Token::Eof | Token::Newline) && i < length)
                    {
                        Value::String(String::new())
                    } else if matches!(self.current_token, Token::LeftBracket) {
                        self.parse_array(depth + 1)?
                    } else {
                        self.parse_primitive()?
                    };

                    items.push(value);
                }
            }
        }

        validation::validate_array_length(length, items.len())?;

        if self.options.strict && matches!(self.current_token, Token::Delimiter(_)) {
            return Err(self.parse_error_with_context(format!(
                "Array length mismatch: expected {length} items, but more items found",
            )));
        }

        Ok(Value::Array(items))
    }

    fn parse_tabular_field_value(&mut self) -> ToonResult<Value> {
        match &self.current_token {
            Token::Null => {
                self.advance()?;
                Ok(Value::Null)
            }
            Token::Bool(b) => {
                let val = *b;
                self.advance()?;
                Ok(Value::Bool(val))
            }
            Token::Integer(i) => {
                let val = *i;
                self.advance()?;
                Ok(Number::from(val).into())
            }
            Token::Number(n) => {
                let val = *n;
                self.advance()?;
                // If the float is actually an integer, represent it as such
                if val.is_finite() && val.fract() == 0.0 && val.abs() <= i64::MAX as f64 {
                    Ok(Number::from(val as i64).into())
                } else {
                    Ok(Number::from_f64(val)
                        .ok_or_else(|| ToonError::InvalidInput(format!("Invalid number: {val}")))?
                        .into())
                }
            }
            Token::String(s, _) => {
                // Tabular fields can have multiple string tokens joined with spaces
                let mut accumulated = s.clone();
                self.advance()?;

                while let Token::String(next, _) = &self.current_token {
                    if !accumulated.is_empty() {
                        accumulated.push(' ');
                    }
                    accumulated.push_str(next);
                    self.advance()?;
                }

                Ok(Value::String(accumulated))
            }
            _ => Err(self.parse_error_with_context(format!(
                "Expected primitive value, found {:?}",
                self.current_token
            ))),
        }
    }

    fn parse_primitive(&mut self) -> ToonResult<Value> {
        match &self.current_token {
            Token::Null => {
                self.advance()?;
                Ok(Value::Null)
            }
            Token::Bool(b) => {
                let val = *b;
                self.advance()?;
                Ok(Value::Bool(val))
            }
            Token::Integer(i) => {
                let val = *i;
                self.advance()?;
                Ok(Number::from(val).into())
            }
            Token::Number(n) => {
                let val = *n;
                self.advance()?;

                if val.is_finite() && val.fract() == 0.0 && val.abs() <= i64::MAX as f64 {
                    Ok(Number::from(val as i64).into())
                } else {
                    Ok(Number::from_f64(val)
                        .ok_or_else(|| ToonError::InvalidInput(format!("Invalid number: {val}")))?
                        .into())
                }
            }
            Token::String(s, _) => {
                let val = s.clone();
                self.advance()?;
                Ok(Value::String(val))
            }
            _ => Err(self.parse_error_with_context(format!(
                "Expected primitive value, found {:?}",
                self.current_token
            ))),
        }
    }

    fn parse_error_with_context(&self, message: impl Into<String>) -> ToonError {
        let (line, column) = self.scanner.current_position();
        let message = message.into();

        let context = self.get_error_context(line, column);

        ToonError::ParseError {
            line,
            column,
            message,
            context: Some(Box::new(context)),
        }
    }

    fn get_error_context(&self, line: usize, column: usize) -> ErrorContext {
        let lines: Vec<&str> = self.input.lines().collect();

        let source_line = if line > 0 && line <= lines.len() {
            lines[line - 1].to_string()
        } else {
            String::new()
        };

        let preceding_lines: Vec<String> = if line > 1 {
            lines[line.saturating_sub(3)..line - 1]
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        };

        let following_lines: Vec<String> = if line < lines.len() {
            lines[line..line.saturating_add(2).min(lines.len())]
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        };

        let indicator = if column > 0 {
            Some(format!("{:width$}^", "", width = column - 1))
        } else {
            None
        };

        ErrorContext {
            source_line,
            preceding_lines,
            following_lines,
            suggestion: None,
            indicator,
        }
    }

    fn validate_indentation(&self, indent_amount: usize) -> ToonResult<()> {
        if !self.options.strict {
            return Ok(());
        }

        let indent_size = self.options.indent.get_spaces();
        // In strict mode, indentation must be a multiple of the configured indent size
        if indent_size > 0 && indent_amount > 0 && !indent_amount.is_multiple_of(indent_size) {
            Err(self.parse_error_with_context(format!(
                "Invalid indentation: found {indent_amount} spaces, but must be a multiple of \
                 {indent_size}"
            )))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64;

    use serde_json::json;

    use super::*;

    fn parse(input: &str) -> ToonResult<Value> {
        let mut parser = Parser::new(input, DecodeOptions::default())?;
        parser.parse()
    }

    #[test]
    fn test_parse_primitives() {
        assert_eq!(parse("null").unwrap(), json!(null));
        assert_eq!(parse("true").unwrap(), json!(true));
        assert_eq!(parse("false").unwrap(), json!(false));
        assert_eq!(parse("42").unwrap(), json!(42));
        assert_eq!(parse("3.141592653589793").unwrap(), json!(f64::consts::PI));
        assert_eq!(parse("hello").unwrap(), json!("hello"));
    }

    #[test]
    fn test_parse_simple_object() {
        let result = parse("name: Alice\nage: 30").unwrap();
        assert_eq!(result["name"], json!("Alice"));
        assert_eq!(result["age"], json!(30));
    }

    #[test]
    fn test_parse_primitive_array() {
        let result = parse("tags[3]: a,b,c").unwrap();
        assert_eq!(result["tags"], json!(["a", "b", "c"]));
    }

    #[test]
    fn test_parse_empty_array() {
        let result = parse("items[0]:").unwrap();
        assert_eq!(result["items"], json!([]));
    }

    #[test]
    fn test_parse_tabular_array() {
        let result = parse("users[2]{id,name}:\n  1,Alice\n  2,Bob").unwrap();
        assert_eq!(
            result["users"],
            json!([
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ])
        );
    }

    #[test]
    fn test_empty_tokens() {
        let result = parse("items[3]: a,,c").unwrap();
        assert_eq!(result["items"], json!(["a", "", "c"]));
    }

    #[test]
    fn test_empty_nested_object() {
        let result = parse("user:").unwrap();
        assert_eq!(result, json!({"user": {}}));
    }

    #[test]
    fn test_list_item_object() {
        let result =
            parse("items[2]:\n  - id: 1\n    name: First\n  - id: 2\n    name: Second").unwrap();
        assert_eq!(
            result["items"],
            json!([
                {"id": 1, "name": "First"},
                {"id": 2, "name": "Second"}
            ])
        );
    }

    #[test]
    fn test_nested_array_in_list_item() {
        let result = parse("items[1]:\n  - tags[3]: a,b,c").unwrap();
        assert_eq!(result["items"], json!([{"tags": ["a", "b", "c"]}]));
    }

    #[test]
    fn test_two_level_siblings() {
        let input = "x:\n  y: 1\n  z: 2";
        let opts = DecodeOptions::default();
        let mut parser = Parser::new(input, opts).unwrap();
        let result = parser.parse().unwrap();

        let x = result.as_object().unwrap().get("x").unwrap();
        let x_obj = x.as_object().unwrap();

        assert_eq!(x_obj.len(), 2, "x should have 2 keys");
        assert_eq!(x_obj.get("y").unwrap(), &serde_json::json!(1));
        assert_eq!(x_obj.get("z").unwrap(), &serde_json::json!(2));
    }

    #[test]
    fn test_nested_object_with_sibling() {
        let input = "a:\n  b:\n    c: 1\n  d: 2";
        let opts = DecodeOptions::default();
        let mut parser = Parser::new(input, opts).unwrap();
        let result = parser.parse().unwrap();

        let a = result.as_object().unwrap().get("a").unwrap();
        let a_obj = a.as_object().unwrap();

        assert_eq!(a_obj.len(), 2, "a should have 2 keys (b and d)");
        assert!(a_obj.contains_key("b"), "a should have key 'b'");
        assert!(a_obj.contains_key("d"), "a should have key 'd'");

        let b = a_obj.get("b").unwrap().as_object().unwrap();
        assert_eq!(b.len(), 1, "b should have only 1 key (c)");
        assert!(b.contains_key("c"), "b should have key 'c'");
        assert!(!b.contains_key("d"), "b should NOT have key 'd'");
    }

    #[test]
    fn test_field_value_with_parentheses() {
        let result = parse("msg: Mostly Functions (3 of 3)").unwrap();
        assert_eq!(result, json!({"msg": "Mostly Functions (3 of 3)"}));

        let result = parse("val: (hello)").unwrap();
        assert_eq!(result, json!({"val": "(hello)"}));

        let result = parse("test: a (b) c (d)").unwrap();
        assert_eq!(result, json!({"test": "a (b) c (d)"}));
    }

    #[test]
    fn test_field_value_number_with_parentheses() {
        let result = parse("code: 0(f)").unwrap();
        assert_eq!(result, json!({"code": "0(f)"}));

        let result = parse("val: 5(test)").unwrap();
        assert_eq!(result, json!({"val": "5(test)"}));

        let result = parse("msg: test 123)").unwrap();
        assert_eq!(result, json!({"msg": "test 123)"}));
    }

    #[test]
    fn test_field_value_single_token_optimization() {
        let result = parse("name: hello").unwrap();
        assert_eq!(result, json!({"name": "hello"}));

        let result = parse("age: 42").unwrap();
        assert_eq!(result, json!({"age": 42}));

        let result = parse("active: true").unwrap();
        assert_eq!(result, json!({"active": true}));

        let result = parse("value: null").unwrap();
        assert_eq!(result, json!({"value": null}));
    }

    #[test]
    fn test_field_value_multi_token() {
        let result = parse("msg: hello world").unwrap();
        assert_eq!(result, json!({"msg": "hello world"}));

        let result = parse("msg: test 123 end").unwrap();
        assert_eq!(result, json!({"msg": "test 123 end"}));
    }

    #[test]
    fn test_field_value_spacing_preserved() {
        let result = parse("val: hello world").unwrap();
        assert_eq!(result, json!({"val": "hello world"}));

        let result = parse("val: 0(f)").unwrap();
        assert_eq!(result, json!({"val": "0(f)"}));
    }

    #[test]
    fn test_round_trip_parentheses() {
        use crate::{decode::decode_default, encode::encode_default};

        let original = json!({
            "message": "Mostly Functions (3 of 3)",
            "code": "0(f)",
            "simple": "(hello)",
            "mixed": "test 123)"
        });

        let encoded = encode_default(&original).unwrap();
        let decoded: Value = decode_default(&encoded).unwrap();

        assert_eq!(original, decoded);
    }

    #[test]
    fn test_multiple_fields_with_edge_cases() {
        let input = r#"message: Mostly Functions (3 of 3)
sone: (hello)
hello: 0(f)"#;

        let result = parse(input).unwrap();
        assert_eq!(
            result,
            json!({
                "message": "Mostly Functions (3 of 3)",
                "sone": "(hello)",
                "hello": "0(f)"
            })
        );
    }

    #[test]
    fn test_decode_list_item_tabular_array_v3() {
        // Tabular arrays as first field of list items
        // Rows must be at depth +2 relative to hyphen (6 spaces from root)
        let input = r#"items[1]:
  - users[2]{id,name}:
      1,Ada
      2,Bob
    status: active"#;

        let result = parse(input).unwrap();

        assert_eq!(
            result,
            json!({
                "items": [
                    {
                        "users": [
                            {"id": 1, "name": "Ada"},
                            {"id": 2, "name": "Bob"}
                        ],
                        "status": "active"
                    }
                ]
            })
        );
    }

    #[test]
    fn test_decode_list_item_tabular_array_multiple_items() {
        // Multiple list items each with tabular array as first field
        let input = r#"data[2]:
  - records[1]{id,val}:
      1,x
    count: 1
  - records[1]{id,val}:
      2,y
    count: 1"#;

        let result = parse(input).unwrap();

        assert_eq!(
            result,
            json!({
                "data": [
                    {
                        "records": [{"id": 1, "val": "x"}],
                        "count": 1
                    },
                    {
                        "records": [{"id": 2, "val": "y"}],
                        "count": 1
                    }
                ]
            })
        );
    }

    #[test]
    fn test_decode_list_item_tabular_array_with_multiple_fields() {
        // List item with tabular array first and multiple sibling fields
        let input = r#"entries[1]:
  - people[2]{name,age}:
      Alice,30
      Bob,25
    total: 2
    category: staff"#;

        let result = parse(input).unwrap();

        assert_eq!(
            result,
            json!({
                "entries": [
                    {
                        "people": [
                            {"name": "Alice", "age": 30},
                            {"name": "Bob", "age": 25}
                        ],
                        "total": 2,
                        "category": "staff"
                    }
                ]
            })
        );
    }

    #[test]
    fn test_decode_list_item_non_tabular_array_unchanged() {
        // Non-tabular arrays as first field should work normally
        let input = r#"items[1]:
  - tags[3]: a,b,c
    name: test"#;

        let result = parse(input).unwrap();

        assert_eq!(
            result,
            json!({
                "items": [
                    {
                        "tags": ["a", "b", "c"],
                        "name": "test"
                    }
                ]
            })
        );
    }

    #[test]
    fn test_decode_strict_rejects_v2_tabular_indent() {
        use crate::decode::decode_strict;

        // Old format: rows at depth +1 (4 spaces from root)
        // Strict mode should reject this incorrect indentation
        let input_v2 = r#"items[1]:
  - users[2]{id,name}:
    1,Ada
    2,Bob"#;

        let result = decode_strict::<Value>(input_v2);

        // Should error due to incorrect indentation
        assert!(
            result.is_err(),
            "Old format with incorrect indentation should be rejected in strict mode"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("indentation") || err_msg.contains("Invalid indentation"),
            "Error should mention indentation. Got: {}",
            err_msg
        );
    }

    #[test]
    fn test_decode_tabular_array_not_in_list_item_unchanged() {
        // Regular tabular arrays (not in list items) should still use depth +1
        let input = r#"users[2]{id,name}:
  1,Ada
  2,Bob"#;

        let result = parse(input).unwrap();

        assert_eq!(
            result,
            json!({
                "users": [
                    {"id": 1, "name": "Ada"},
                    {"id": 2, "name": "Bob"}
                ]
            })
        );
    }

    #[test]
    fn test_decode_nested_tabular_not_first_field() {
        // Tabular array as a subsequent field (not first) should use normal depth
        let input = r#"items[1]:
  - name: test
    data[2]{id,val}:
      1,x
      2,y"#;

        let result = parse(input).unwrap();

        assert_eq!(
            result,
            json!({
                "items": [
                    {
                        "name": "test",
                        "data": [
                            {"id": 1, "val": "x"},
                            {"id": 2, "val": "y"}
                        ]
                    }
                ]
            })
        );
    }
}

File: decode\validation.rs
==========================
use crate::types::{ToonError, ToonResult};

/// Validate that array length matches expected value.
pub fn validate_array_length(expected: usize, actual: usize) -> ToonResult<()> {
    // Array length mismatches should always error, regardless of strict mode
    if expected != actual {
        return Err(ToonError::length_mismatch(expected, actual));
    }
    Ok(())
}

/// Validate field list for tabular arrays (no duplicates, non-empty names).
pub fn validate_field_list(fields: &[String]) -> ToonResult<()> {
    if fields.is_empty() {
        return Err(ToonError::InvalidInput(
            "Field list cannot be empty for tabular arrays".to_string(),
        ));
    }

    // Check for duplicate field names
    for i in 0..fields.len() {
        for j in (i + 1)..fields.len() {
            if fields[i] == fields[j] {
                return Err(ToonError::InvalidInput(format!(
                    "Duplicate field name: '{}'",
                    fields[i]
                )));
            }
        }
    }

    for field in fields {
        if field.is_empty() {
            return Err(ToonError::InvalidInput(
                "Field name cannot be empty".to_string(),
            ));
        }
    }

    Ok(())
}

/// Validate that a tabular row has the expected number of values.
pub fn validate_row_length(
    row_index: usize,
    expected_fields: usize,
    actual_values: usize,
) -> ToonResult<()> {
    if expected_fields != actual_values {
        return Err(ToonError::InvalidStructure(format!(
            "Row {row_index} has {actual_values} values but expected {expected_fields} fields"
        )));
    }
    Ok(())
}

/// Validate that detected and expected delimiters match.
pub fn validate_delimiter_consistency(
    detected: Option<crate::types::Delimiter>,
    expected: Option<crate::types::Delimiter>,
) -> ToonResult<()> {
    if let (Some(detected), Some(expected)) = (detected, expected) {
        if detected != expected {
            return Err(ToonError::InvalidDelimiter(format!(
                "Detected delimiter {detected} but expected {expected}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_array_length() {
        assert!(validate_array_length(5, 3).is_err());
        assert!(validate_array_length(3, 5).is_err());
        assert!(validate_array_length(5, 5).is_ok());
    }

    #[test]
    fn test_validate_field_list() {
        assert!(validate_field_list(&["id".to_string(), "name".to_string()]).is_ok());
        assert!(validate_field_list(&["field1".to_string()]).is_ok());

        assert!(validate_field_list(&[]).is_err());

        assert!(
            validate_field_list(&["id".to_string(), "name".to_string(), "id".to_string()]).is_err()
        );

        assert!(
            validate_field_list(&["id".to_string(), "".to_string(), "name".to_string()]).is_err()
        );
    }

    #[test]
    fn test_validate_row_length() {
        assert!(validate_row_length(0, 3, 3).is_ok());
        assert!(validate_row_length(1, 5, 5).is_ok());

        assert!(validate_row_length(0, 3, 2).is_err());
        assert!(validate_row_length(1, 3, 4).is_err());
    }

    #[test]
    fn test_validate_delimiter_consistency() {
        use crate::types::Delimiter;

        assert!(
            validate_delimiter_consistency(Some(Delimiter::Comma), Some(Delimiter::Comma)).is_ok()
        );

        assert!(
            validate_delimiter_consistency(Some(Delimiter::Comma), Some(Delimiter::Pipe)).is_err()
        );

        assert!(validate_delimiter_consistency(None, Some(Delimiter::Comma)).is_ok());
        assert!(validate_delimiter_consistency(Some(Delimiter::Comma), None).is_ok());
        assert!(validate_delimiter_consistency(None, None).is_ok());
    }
}

File: decode\scanner.rs
=======================
use crate::types::{Delimiter, ToonError, ToonResult};

/// Tokens produced by the scanner during lexical analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Colon,
    Dash,
    Newline,
    String(String, bool),
    Number(f64),
    Integer(i64),
    Bool(bool),
    Null,
    Delimiter(Delimiter),
    Eof,
}

/// Scanner that tokenizes TOON input into a sequence of tokens.
pub struct Scanner {
    input: Vec<char>,
    position: usize,
    line: usize,
    column: usize,
    active_delimiter: Option<Delimiter>,
    last_line_indent: usize,
}

impl Scanner {
    /// Create a new scanner for the given input string.
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            position: 0,
            line: 1,
            column: 1,
            active_delimiter: None,
            last_line_indent: 0,
        }
    }

    /// Set the active delimiter for tokenizing array elements.
    pub fn set_active_delimiter(&mut self, delimiter: Option<Delimiter>) {
        self.active_delimiter = delimiter;
    }

    /// Get the current position (line, column).
    pub fn current_position(&self) -> (usize, usize) {
        (self.line, self.column)
    }

    pub fn get_line(&self) -> usize {
        self.line
    }

    pub fn get_column(&self) -> usize {
        self.column
    }

    pub fn peek(&self) -> Option<char> {
        self.input.get(self.position).copied()
    }

    pub fn count_leading_spaces(&self) -> usize {
        let mut idx = self.position;
        let mut count = 0;
        while let Some(&ch) = self.input.get(idx) {
            if ch == ' ' {
                count += 1;
                idx += 1;
            } else {
                break;
            }
        }
        count
    }

    pub fn count_spaces_after_newline(&self) -> usize {
        let mut idx = self.position;
        if self.input.get(idx) != Some(&'\n') {
            return 0;
        }
        idx += 1;
        let mut count = 0;
        while let Some(&ch) = self.input.get(idx) {
            if ch == ' ' {
                count += 1;
                idx += 1;
            } else {
                break;
            }
        }
        count
    }

    pub fn peek_ahead(&self, offset: usize) -> Option<char> {
        self.input.get(self.position + offset).copied()
    }

    pub fn advance(&mut self) -> Option<char> {
        if let Some(ch) = self.input.get(self.position) {
            self.position += 1;
            if *ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            Some(*ch)
        } else {
            None
        }
    }

    pub fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch == ' ' {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Scan the next token from the input.
    pub fn scan_token(&mut self) -> ToonResult<Token> {
        if self.column == 1 {
            let mut count = 0;
            let mut idx = self.position;

            while let Some(&ch) = self.input.get(idx) {
                if ch == ' ' {
                    count += 1;
                    idx += 1;
                } else {
                    if ch == '\t' {
                        let (line, col) = self.current_position();
                        return Err(ToonError::parse_error(
                            line,
                            col + count,
                            "Tabs are not allowed in indentation",
                        ));
                    }
                    break;
                }
            }
            self.last_line_indent = count;
        }

        self.skip_whitespace();

        match self.peek() {
            None => Ok(Token::Eof),
            Some('\n') => {
                self.advance();
                Ok(Token::Newline)
            }
            Some('[') => {
                self.advance();
                Ok(Token::LeftBracket)
            }
            Some(']') => {
                self.advance();
                Ok(Token::RightBracket)
            }
            Some('{') => {
                self.advance();
                Ok(Token::LeftBrace)
            }
            Some('}') => {
                self.advance();
                Ok(Token::RightBrace)
            }
            Some(':') => {
                self.advance();
                Ok(Token::Colon)
            }
            Some('-') => {
                self.advance();
                if let Some(ch) = self.peek() {
                    if ch.is_ascii_digit() {
                        let num_str = self.scan_number_string(true)?;
                        return self.parse_number(&num_str);
                    }
                }
                Ok(Token::Dash)
            }
            Some(',') => {
                // Delimiter only when active, otherwise part of unquoted string
                if matches!(self.active_delimiter, Some(Delimiter::Comma)) {
                    self.advance();
                    Ok(Token::Delimiter(Delimiter::Comma))
                } else {
                    self.scan_unquoted_string()
                }
            }
            Some('|') => {
                if matches!(self.active_delimiter, Some(Delimiter::Pipe)) {
                    self.advance();
                    Ok(Token::Delimiter(Delimiter::Pipe))
                } else {
                    self.scan_unquoted_string()
                }
            }
            Some('\t') => {
                if matches!(self.active_delimiter, Some(Delimiter::Tab)) {
                    self.advance();
                    Ok(Token::Delimiter(Delimiter::Tab))
                } else {
                    self.scan_unquoted_string()
                }
            }
            Some('"') => self.scan_quoted_string(),
            Some(ch) if ch.is_ascii_digit() => {
                let num_str = self.scan_number_string(false)?;
                self.parse_number(&num_str)
            }
            Some(_) => self.scan_unquoted_string(),
        }
    }

    fn scan_quoted_string(&mut self) -> ToonResult<Token> {
        self.advance();

        let mut value = String::new();
        let mut escaped = false;

        while let Some(ch) = self.advance() {
            if escaped {
                match ch {
                    'n' => value.push('\n'),
                    'r' => value.push('\r'),
                    't' => value.push('\t'),
                    '"' => value.push('"'),
                    '\\' => value.push('\\'),
                    _ => {
                        let (line, col) = self.current_position();
                        return Err(ToonError::parse_error(
                            line,
                            col - 1,
                            format!("Invalid escape sequence: \\{ch}"),
                        ));
                    }
                }
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                return Ok(Token::String(value, true));
            } else {
                value.push(ch);
            }
        }

        Err(ToonError::UnexpectedEof)
    }

    fn scan_unquoted_string(&mut self) -> ToonResult<Token> {
        let mut value = String::new();

        while let Some(ch) = self.peek() {
            if ch == '\n'
                || ch == ' '
                || ch == ':'
                || ch == '['
                || ch == ']'
                || ch == '{'
                || ch == '}'
            {
                break;
            }

            // Active delimiters stop the string; otherwise they're part of it
            if let Some(active) = self.active_delimiter {
                if (active == Delimiter::Comma && ch == ',')
                    || (active == Delimiter::Pipe && ch == '|')
                    || (active == Delimiter::Tab && ch == '\t')
                {
                    break;
                }
            }
            value.push(ch);
            self.advance();
        }

        // Single-char delimiters kept as-is, others trimmed
        let value = if value.len() == 1 && (value == "," || value == "|" || value == "\t") {
            value
        } else {
            value.trim_end().to_string()
        };

        match value.as_str() {
            "null" => Ok(Token::Null),
            "true" => Ok(Token::Bool(true)),
            "false" => Ok(Token::Bool(false)),
            _ => Ok(Token::String(value, false)),
        }
    }

    pub fn get_last_line_indent(&self) -> usize {
        self.last_line_indent
    }

    fn scan_number_string(&mut self, negative: bool) -> ToonResult<String> {
        let mut num_str = if negative {
            String::from("-")
        } else {
            String::new()
        };

        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() || ch == '.' || ch == 'e' || ch == 'E' || ch == '+' || ch == '-'
            {
                num_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        Ok(num_str)
    }

    fn parse_number(&self, s: &str) -> ToonResult<Token> {
        // Number followed immediately by other chars like "0(f)" should be a string
        if let Some(next_ch) = self.peek() {
            if next_ch != ' '
                && next_ch != '\n'
                && next_ch != ':'
                && next_ch != '['
                && next_ch != ']'
                && next_ch != '{'
                && next_ch != '}'
                && !matches!(
                    (self.active_delimiter, next_ch),
                    (Some(Delimiter::Comma), ',')
                        | (Some(Delimiter::Pipe), '|')
                        | (Some(Delimiter::Tab), '\t')
                )
            {
                return Ok(Token::String(s.to_string(), false));
            }
        }

        // Leading zeros like "05" are strings, but "0", "0.5", "-0" are numbers
        if s.starts_with('0') && s.len() > 1 {
            let second_char = s.chars().nth(1).unwrap();
            if second_char.is_ascii_digit() {
                return Ok(Token::String(s.to_string(), false));
            }
        }

        if s.contains('.') || s.contains('e') || s.contains('E') {
            if let Ok(f) = s.parse::<f64>() {
                Ok(Token::Number(f))
            } else {
                Ok(Token::String(s.to_string(), false))
            }
        } else if let Ok(i) = s.parse::<i64>() {
            Ok(Token::Integer(i))
        } else {
            Ok(Token::String(s.to_string(), false))
        }
    }

    /// Read the rest of the current line (until newline or EOF).
    /// Returns the content with a flag indicating if it started with
    /// whitespace.
    pub fn read_rest_of_line_with_space_info(&mut self) -> (String, bool) {
        let had_leading_space = matches!(self.peek(), Some(' '));
        self.skip_whitespace();

        let mut result = String::new();
        while let Some(ch) = self.peek() {
            if ch == '\n' {
                break;
            }
            result.push(ch);
            self.advance();
        }

        (result.trim_end().to_string(), had_leading_space)
    }

    /// Read the rest of the current line (until newline or EOF).
    pub fn read_rest_of_line(&mut self) -> String {
        self.read_rest_of_line_with_space_info().0
    }

    /// Parse a complete value string into a token.
    pub fn parse_value_string(&self, s: &str) -> ToonResult<Token> {
        let trimmed = s.trim();

        if trimmed.is_empty() {
            return Ok(Token::String(String::new(), false));
        }

        if trimmed.starts_with('"') {
            let mut value = String::new();
            let mut escaped = false;
            let chars: Vec<char> = trimmed.chars().collect();
            let mut i = 1;

            while i < chars.len() {
                let ch = chars[i];
                if escaped {
                    match ch {
                        'n' => value.push('\n'),
                        'r' => value.push('\r'),
                        't' => value.push('\t'),
                        '"' => value.push('"'),
                        '\\' => value.push('\\'),
                        _ => {
                            return Err(ToonError::parse_error(
                                self.line,
                                self.column,
                                format!("Invalid escape sequence: \\{ch}"),
                            ));
                        }
                    }
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    if i != chars.len() - 1 {
                        return Err(ToonError::parse_error(
                            self.line,
                            self.column,
                            "Unexpected characters after closing quote",
                        ));
                    }
                    return Ok(Token::String(value, true));
                } else {
                    value.push(ch);
                }
                i += 1;
            }

            return Err(ToonError::parse_error(
                self.line,
                self.column,
                "Unterminated string: missing closing quote",
            ));
        }

        match trimmed {
            "true" => return Ok(Token::Bool(true)),
            "false" => return Ok(Token::Bool(false)),
            "null" => return Ok(Token::Null),
            _ => {}
        }

        if trimmed.starts_with('-') || trimmed.chars().next().unwrap().is_ascii_digit() {
            // Leading zeros like "05" are strings
            if trimmed.starts_with('0') && trimmed.len() > 1 {
                let second_char = trimmed.chars().nth(1).unwrap();
                if second_char.is_ascii_digit() {
                    return Ok(Token::String(trimmed.to_string(), false));
                }
            }

            if trimmed.contains('.') || trimmed.contains('e') || trimmed.contains('E') {
                if let Ok(f) = trimmed.parse::<f64>() {
                    let normalized = if f == -0.0 { 0.0 } else { f };
                    return Ok(Token::Number(normalized));
                }
            } else if let Ok(i) = trimmed.parse::<i64>() {
                return Ok(Token::Integer(i));
            }
        }

        Ok(Token::String(trimmed.to_string(), false))
    }

    pub fn detect_delimiter(&mut self) -> Option<Delimiter> {
        let saved_pos = self.position;

        while let Some(ch) = self.peek() {
            match ch {
                ',' => {
                    self.position = saved_pos;
                    return Some(Delimiter::Comma);
                }
                '|' => {
                    self.position = saved_pos;
                    return Some(Delimiter::Pipe);
                }
                '\t' => {
                    self.position = saved_pos;
                    return Some(Delimiter::Tab);
                }
                '\n' | ':' | '[' | ']' | '{' | '}' => break,
                _ => {
                    self.advance();
                }
            }
        }

        self.position = saved_pos;
        None
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;

    #[test]
    fn test_scan_structural_tokens() {
        let mut scanner = Scanner::new("[]{}:-");
        assert_eq!(scanner.scan_token().unwrap(), Token::LeftBracket);
        assert_eq!(scanner.scan_token().unwrap(), Token::RightBracket);
        assert_eq!(scanner.scan_token().unwrap(), Token::LeftBrace);
        assert_eq!(scanner.scan_token().unwrap(), Token::RightBrace);
        assert_eq!(scanner.scan_token().unwrap(), Token::Colon);
        assert_eq!(scanner.scan_token().unwrap(), Token::Dash);
    }

    #[test]
    fn test_scan_numbers() {
        let mut scanner = Scanner::new("42 3.141592653589793 -5");
        assert_eq!(scanner.scan_token().unwrap(), Token::Integer(42));
        assert_eq!(
            scanner.scan_token().unwrap(),
            Token::Number(f64::consts::PI)
        );
        assert_eq!(scanner.scan_token().unwrap(), Token::Integer(-5));
    }

    #[test]
    fn test_scan_booleans() {
        let mut scanner = Scanner::new("true false");
        assert_eq!(scanner.scan_token().unwrap(), Token::Bool(true));
        assert_eq!(scanner.scan_token().unwrap(), Token::Bool(false));
    }

    #[test]
    fn test_scan_null() {
        let mut scanner = Scanner::new("null");
        assert_eq!(scanner.scan_token().unwrap(), Token::Null);
    }

    #[test]
    fn test_scan_quoted_string() {
        let mut scanner = Scanner::new(r#""hello world""#);
        assert_eq!(
            scanner.scan_token().unwrap(),
            Token::String("hello world".to_string(), true)
        );
    }

    #[test]
    fn test_scan_escaped_string() {
        let mut scanner = Scanner::new(r#""hello\nworld""#);
        assert_eq!(
            scanner.scan_token().unwrap(),
            Token::String("hello\nworld".to_string(), true)
        );
    }

    #[test]
    fn test_scan_unquoted_string() {
        let mut scanner = Scanner::new("hello");
        assert_eq!(
            scanner.scan_token().unwrap(),
            Token::String("hello".to_string(), false)
        );
    }

    #[test]
    fn test_detect_delimiter() {
        let mut scanner = Scanner::new("a,b,c");
        assert_eq!(scanner.detect_delimiter(), Some(Delimiter::Comma));

        let mut scanner = Scanner::new("a|b|c");
        assert_eq!(scanner.detect_delimiter(), Some(Delimiter::Pipe));

        let mut scanner = Scanner::new("a\tb\tc");
        assert_eq!(scanner.detect_delimiter(), Some(Delimiter::Tab));
    }

    #[test]
    fn test_read_rest_of_line_with_space_info() {
        let mut scanner = Scanner::new(" world");
        let (content, had_space) = scanner.read_rest_of_line_with_space_info();
        assert_eq!(content, "world");
        assert!(had_space);

        let mut scanner = Scanner::new("world");
        let (content, had_space) = scanner.read_rest_of_line_with_space_info();
        assert_eq!(content, "world");
        assert!(!had_space);

        let mut scanner = Scanner::new("(hello)");
        let (content, had_space) = scanner.read_rest_of_line_with_space_info();
        assert_eq!(content, "(hello)");
        assert!(!had_space);

        let mut scanner = Scanner::new("");
        let (content, had_space) = scanner.read_rest_of_line_with_space_info();
        assert_eq!(content, "");
        assert!(!had_space);
    }

    #[test]
    fn test_parse_value_string() {
        let scanner = Scanner::new("");
        assert_eq!(
            scanner.parse_value_string("hello").unwrap(),
            Token::String("hello".to_string(), false)
        );

        assert_eq!(
            scanner.parse_value_string("(hello)").unwrap(),
            Token::String("(hello)".to_string(), false)
        );

        assert_eq!(
            scanner
                .parse_value_string("Mostly Functions (3 of 3)")
                .unwrap(),
            Token::String("Mostly Functions (3 of 3)".to_string(), false)
        );
        assert_eq!(
            scanner.parse_value_string("0(f)").unwrap(),
            Token::String("0(f)".to_string(), false)
        );

        assert_eq!(
            scanner.parse_value_string("42").unwrap(),
            Token::Integer(42)
        );

        assert_eq!(
            scanner.parse_value_string("true").unwrap(),
            Token::Bool(true)
        );
        assert_eq!(
            scanner.parse_value_string("false").unwrap(),
            Token::Bool(false)
        );
        assert_eq!(scanner.parse_value_string("null").unwrap(), Token::Null);

        assert_eq!(
            scanner.parse_value_string(r#""hello world""#).unwrap(),
            Token::String("hello world".to_string(), true)
        );
    }

    #[test]
    fn test_number_followed_by_parenthesis() {
        let mut scanner = Scanner::new("0(f)");
        let num_token = scanner.scan_number_string(false).unwrap();
        let token = scanner.parse_number(&num_token).unwrap();

        assert_eq!(token, Token::String("0".to_string(), false));
    }
}

File: encode\folding.rs
=======================
use crate::types::{JsonValue as Value, KeyFoldingMode, is_identifier_segment};

/// Result of chain analysis for folding.
pub struct FoldableChain {
    /// The folded key path (e.g., "a.b.c")
    pub folded_key: String,
    /// The leaf value at the end of the chain
    pub leaf_value: Value,
    /// Number of segments that were folded
    pub depth_folded: usize,
}

/// Check if a value is a single-key object suitable for folding.
fn is_single_key_object(value: &Value) -> Option<(&String, &Value)> {
    if let Value::Object(obj) = value {
        if obj.len() == 1 {
            return obj.iter().next();
        }
    }
    None
}

/// Analyze if a key-value pair can be folded into dotted notation.
pub fn analyze_foldable_chain(
    key: &str,
    value: &Value,
    flatten_depth: usize,
    existing_keys: &[&String],
) -> Option<FoldableChain> {
    if !is_identifier_segment(key) {
        return None;
    }

    let mut segments = vec![key.to_string()];
    let mut current_value = value;

    // Follow single-key object chain until we hit a multi-key object or leaf
    while let Some((next_key, next_value)) = is_single_key_object(current_value) {
        if segments.len() >= flatten_depth {
            break;
        }

        if !is_identifier_segment(next_key) {
            break;
        }

        segments.push(next_key.clone());
        current_value = next_value;
    }

    // Must fold at least 2 segments to be worthwhile
    if segments.len() < 2 {
        return None;
    }

    let folded_key = segments.join(".");

    // Don't fold if it would collide with an existing key
    if existing_keys.contains(&&folded_key) {
        return None;
    }

    Some(FoldableChain {
        folded_key,
        leaf_value: current_value.clone(),
        depth_folded: segments.len(),
    })
}

pub fn should_fold(mode: KeyFoldingMode, chain: &Option<FoldableChain>) -> bool {
    match mode {
        KeyFoldingMode::Off => false,
        KeyFoldingMode::Safe => chain.is_some(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_is_single_key_object() {
        let val = Value::from(json!({"a": 1}));
        assert!(is_single_key_object(&val).is_some());

        let val = Value::from(json!({"a": 1, "b": 2}));
        assert!(is_single_key_object(&val).is_none());

        let val = Value::from(json!(42));
        assert!(is_single_key_object(&val).is_none());
    }

    #[test]
    fn test_analyze_simple_chain() {
        let val = Value::from(json!({"b": {"c": 1}}));
        let existing: Vec<&String> = vec![];

        let result = analyze_foldable_chain("a", &val, usize::MAX, &existing);
        assert!(result.is_some());

        let chain = result.unwrap();
        assert_eq!(chain.folded_key, "a.b.c");
        assert_eq!(chain.depth_folded, 3);
        assert_eq!(chain.leaf_value, Value::from(json!(1)));
    }

    #[test]
    fn test_analyze_with_flatten_depth() {
        let val = Value::from(json!({"b": {"c": {"d": 1}}}));
        let existing: Vec<&String> = vec![];

        let result = analyze_foldable_chain("a", &val, 2, &existing);
        assert!(result.is_some());

        let chain = result.unwrap();
        assert_eq!(chain.folded_key, "a.b");
        assert_eq!(chain.depth_folded, 2);
    }

    #[test]
    fn test_analyze_stops_at_multi_key() {
        let val = Value::from(json!({"b": {"c": 1, "d": 2}}));
        let existing: Vec<&String> = vec![];

        let result = analyze_foldable_chain("a", &val, usize::MAX, &existing);
        assert!(result.is_some());

        let chain = result.unwrap();
        assert_eq!(chain.folded_key, "a.b");
        assert_eq!(chain.depth_folded, 2);
    }

    #[test]
    fn test_analyze_rejects_non_identifier() {
        let val = Value::from(json!({"c": 1}));
        let existing: Vec<&String> = vec![];

        let result = analyze_foldable_chain("bad-key", &val, usize::MAX, &existing);
        assert!(result.is_none());
    }

    #[test]
    fn test_analyze_detects_collision() {
        let val = Value::from(json!({"b": 1}));
        let existing_key = String::from("a.b");
        let existing: Vec<&String> = vec![&existing_key];

        let result = analyze_foldable_chain("a", &val, usize::MAX, &existing);
        assert!(result.is_none());
    }

    #[test]
    fn test_analyze_too_short_chain() {
        let val = Value::from(json!(42));
        let existing: Vec<&String> = vec![];

        let result = analyze_foldable_chain("a", &val, usize::MAX, &existing);
        assert!(result.is_none());
    }
}

File: encode\mod.rs
===================
//! Encoder Implementation
pub mod folding;
pub mod primitives;
pub mod writer;
use indexmap::IndexMap;

use crate::{
    constants::MAX_DEPTH,
    types::{
        EncodeOptions, IntoJsonValue, JsonValue as Value, KeyFoldingMode, ToonError, ToonResult,
    },
    utils::{QuotingContext, format_canonical_number, normalize, validation::validate_depth},
};

/// Encode any serializable value to TOON format.
///
/// This function accepts any type implementing `serde::Serialize`, including:
/// - Custom structs with `#[derive(Serialize)]`
/// - `serde_json::Value`
/// - Built-in types (Vec, HashMap, etc.)
///
/// # Examples
///
/// **With custom structs:**
/// ```
/// use serde::Serialize;
/// use rune_format::{
///     encode,
///     EncodeOptions,
/// };
///
/// #[derive(Serialize)]
/// struct User {
///     name: String,
///     age: u32,
/// }
///
/// let user = User {
///     name: "Alice".to_string(),
///     age: 30,
/// };
/// let toon = encode(&user, &EncodeOptions::default())?;
/// assert!(toon.contains("name: Alice"));
/// # Ok::<(), rune_format::ToonError>(())
/// ```
///
/// **With JSON values:**
/// ```
/// use rune_format::{encode, EncodeOptions, Delimiter};
/// use serde_json::json;
///
/// let data = json!({"tags": ["a", "b", "c"]});
/// let options = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
/// let toon = encode(&data, &options)?;
/// assert!(toon.contains("|"));
/// # Ok::<(), rune_format::ToonError>(())
/// ```
pub fn encode<T: serde::Serialize>(value: &T, options: &EncodeOptions) -> ToonResult<String> {
    let json_value =
        serde_json::to_value(value).map_err(|e| ToonError::SerializationError(e.to_string()))?;
    let json_value: Value = json_value.into();
    encode_impl(&json_value, options)
}

fn encode_impl(value: &Value, options: &EncodeOptions) -> ToonResult<String> {
    let normalized: Value = normalize(value.clone());
    let mut writer = writer::Writer::new(options.clone());

    match &normalized {
        Value::Array(arr) => {
            write_array(&mut writer, None, arr, 0)?;
        }
        Value::Object(obj) => {
            write_object(&mut writer, obj, 0)?;
        }
        _ => {
            write_primitive_value(&mut writer, &normalized, QuotingContext::ObjectValue)?;
        }
    }

    Ok(writer.finish())
}

/// Encode with default options (2-space indent, comma delimiter).
///
/// Works with any type implementing `serde::Serialize`.
///
/// # Examples
///
/// **With structs:**
/// ```
/// use serde::Serialize;
/// use rune_format::encode_default;
///
/// #[derive(Serialize)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// let person = Person {
///     name: "Alice".to_string(),
///     age: 30,
/// };
/// let toon = encode_default(&person)?;
/// assert!(toon.contains("name: Alice"));
/// # Ok::<(), rune_format::ToonError>(())
/// ```
///
/// **With JSON values:**
/// ```
/// use rune_format::encode_default;
/// use serde_json::json;
///
/// let data = json!({"tags": ["reading", "gaming", "coding"]});
/// let toon = encode_default(&data)?;
/// assert_eq!(toon, "tags[3]: reading,gaming,coding");
/// # Ok::<(), rune_format::ToonError>(())
/// ```
pub fn encode_default<T: serde::Serialize>(value: &T) -> ToonResult<String> {
    encode(value, &EncodeOptions::default())
}

/// Encode a JSON object to TOON format (errors if not an object).
///
/// This function accepts either `JsonValue` or `serde_json::Value` and converts
/// automatically.
///
/// # Examples
///
/// ```
/// use rune_format::{encode_object, EncodeOptions};
/// use serde_json::json;
///
/// let data = json!({"name": "Alice", "age": 30});
/// let toon = encode_object(&data, &EncodeOptions::default())?;
/// assert!(toon.contains("name: Alice"));
///
/// // Will error if not an object
/// assert!(encode_object(&json!(42), &EncodeOptions::default()).is_err());
/// # Ok::<(), rune_format::ToonError>(())
/// ```
pub fn encode_object<V: IntoJsonValue>(value: V, options: &EncodeOptions) -> ToonResult<String> {
    let json_value = value.into_json_value();
    if !json_value.is_object() {
        return Err(ToonError::TypeMismatch {
            expected: "object".to_string(),
            found: value_type_name(&json_value).to_string(),
        });
    }
    encode_impl(&json_value, options)
}

/// Encode a JSON array to TOON format (errors if not an array).
///
/// This function accepts either `JsonValue` or `serde_json::Value` and converts
/// automatically.
///
/// # Examples
///
/// ```
/// use rune_format::{encode_array, EncodeOptions};
/// use serde_json::json;
///
/// let data = json!(["a", "b", "c"]);
/// let toon = encode_array(&data, &EncodeOptions::default())?;
/// assert_eq!(toon, "[3]: a,b,c");
///
/// // Will error if not an array
/// assert!(encode_array(&json!({"key": "value"}), &EncodeOptions::default()).is_err());
/// # Ok::<(), rune_format::ToonError>(())
/// ```
pub fn encode_array<V: IntoJsonValue>(value: V, options: &EncodeOptions) -> ToonResult<String> {
    let json_value = value.into_json_value();
    if !json_value.is_array() {
        return Err(ToonError::TypeMismatch {
            expected: "array".to_string(),
            found: value_type_name(&json_value).to_string(),
        });
    }
    encode_impl(&json_value, options)
}

fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn write_object(
    writer: &mut writer::Writer,
    obj: &IndexMap<String, Value>,
    depth: usize,
) -> ToonResult<()> {
    write_object_impl(writer, obj, depth, false)
}

fn write_object_impl(
    writer: &mut writer::Writer,
    obj: &IndexMap<String, Value>,
    depth: usize,
    disable_folding: bool,
) -> ToonResult<()> {
    validate_depth(depth, MAX_DEPTH)?;

    let keys: Vec<&String> = obj.keys().collect();

    for (i, key) in keys.iter().enumerate() {
        if i > 0 {
            writer.write_newline()?;
        }

        let value = &obj[*key];

        // Check if this key-value pair can be folded (v1.5 feature)
        // Don't fold if any sibling key is a dotted path starting with this key
        // (e.g., don't fold inside "data" if "data.meta.items" exists as a sibling)
        let has_conflicting_sibling = keys
            .iter()
            .any(|k| k.starts_with(&format!("{key}.")) || (k.contains('.') && k == key));

        let folded = if !disable_folding
            && writer.options.key_folding == KeyFoldingMode::Safe
            && !has_conflicting_sibling
        {
            folding::analyze_foldable_chain(key, value, writer.options.flatten_depth, &keys)
        } else {
            None
        };

        if let Some(chain) = folded {
            // Write folded key-value pair
            if depth > 0 {
                writer.write_indent(depth)?;
            }

            // Write the leaf value
            match &chain.leaf_value {
                Value::Array(arr) => {
                    // For arrays, pass the folded key to write_array so it generates the header
                    // correctly
                    write_array(writer, Some(&chain.folded_key), arr, 0)?;
                }
                Value::Object(nested_obj) => {
                    // Write the folded key (e.g., "a.b.c")
                    writer.write_key(&chain.folded_key)?;
                    writer.write_char(':')?;
                    if !nested_obj.is_empty() {
                        writer.write_newline()?;
                        // After folding a chain, disable folding for the leaf object
                        // This respects flattenDepth and prevents over-folding
                        write_object_impl(writer, nested_obj, depth + 1, true)?;
                    }
                }
                _ => {
                    // Write the folded key (e.g., "a.b.c")
                    writer.write_key(&chain.folded_key)?;
                    writer.write_char(':')?;
                    writer.write_char(' ')?;
                    write_primitive_value(writer, &chain.leaf_value, QuotingContext::ObjectValue)?;
                }
            }
        } else {
            // Standard (non-folded) encoding
            match value {
                Value::Array(arr) => {
                    write_array(writer, Some(key), arr, depth)?;
                }
                Value::Object(nested_obj) => {
                    if depth > 0 {
                        writer.write_indent(depth)?;
                    }
                    writer.write_key(key)?;
                    writer.write_char(':')?;
                    if !nested_obj.is_empty() {
                        writer.write_newline()?;
                        // If this key has a conflicting sibling, disable folding for its nested
                        // objects
                        let nested_disable_folding = disable_folding || has_conflicting_sibling;
                        write_object_impl(writer, nested_obj, depth + 1, nested_disable_folding)?;
                    }
                }
                _ => {
                    if depth > 0 {
                        writer.write_indent(depth)?;
                    }
                    writer.write_key(key)?;
                    writer.write_char(':')?;
                    writer.write_char(' ')?;
                    write_primitive_value(writer, value, QuotingContext::ObjectValue)?;
                }
            }
        }
    }

    Ok(())
}

fn write_array(
    writer: &mut writer::Writer,
    key: Option<&str>,
    arr: &[Value],
    depth: usize,
) -> ToonResult<()> {
    validate_depth(depth, MAX_DEPTH)?;

    if arr.is_empty() {
        writer.write_empty_array_with_key(key, depth)?;
        return Ok(());
    }

    // Select format based on array content: tabular (uniform objects) > inline
    // primitives > nested list
    if let Some(keys) = is_tabular_array(arr) {
        encode_tabular_array(writer, key, arr, &keys, depth)?;
    } else if is_primitive_array(arr) {
        encode_primitive_array(writer, key, arr, depth)?;
    } else {
        encode_nested_array(writer, key, arr, depth)?;
    }

    Ok(())
}

/// Check if an array can be encoded as tabular format (uniform objects with
/// primitive values).
fn is_tabular_array(arr: &[Value]) -> Option<Vec<String>> {
    if arr.is_empty() {
        return None;
    }

    let first = arr.first()?;
    if !first.is_object() {
        return None;
    }

    let first_obj = first.as_object()?;
    let keys: Vec<String> = first_obj.keys().cloned().collect();

    // First object must have only primitive values
    for value in first_obj.values() {
        if !is_primitive(value) {
            return None;
        }
    }

    // All remaining objects must match: same keys and all primitive values
    for val in arr.iter().skip(1) {
        if let Some(obj) = val.as_object() {
            if obj.len() != keys.len() {
                return None;
            }
            // Verify all keys from first object exist (order doesn't matter)
            for key in &keys {
                if !obj.contains_key(key) {
                    return None;
                }
            }
            // All values must be primitives
            for value in obj.values() {
                if !is_primitive(value) {
                    return None;
                }
            }
        } else {
            return None;
        }
    }

    Some(keys)
}

/// Check if a value is a primitive (not array or object).
fn is_primitive(value: &Value) -> bool {
    matches!(
        value,
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
    )
}

/// Check if all array elements are primitives.
fn is_primitive_array(arr: &[Value]) -> bool {
    arr.iter().all(is_primitive)
}

fn encode_primitive_array(
    writer: &mut writer::Writer,
    key: Option<&str>,
    arr: &[Value],
    depth: usize,
) -> ToonResult<()> {
    writer.write_array_header(key, arr.len(), None, depth)?;
    writer.write_char(' ')?;
    // Set delimiter context for array values (affects quoting decisions)
    writer.push_active_delimiter(writer.options.delimiter);

    for (i, val) in arr.iter().enumerate() {
        if i > 0 {
            writer.write_delimiter()?;
        }
        write_primitive_value(writer, val, QuotingContext::ArrayValue)?;
    }
    writer.pop_active_delimiter();

    Ok(())
}

fn write_primitive_value(
    writer: &mut writer::Writer,
    value: &Value,
    context: QuotingContext,
) -> ToonResult<()> {
    match value {
        Value::Null => writer.write_str("null"),
        Value::Bool(b) => writer.write_str(&b.to_string()),
        Value::Number(n) => {
            // Format in canonical TOON form (no exponents, no trailing zeros)
            let num_str = format_canonical_number(n);
            writer.write_str(&num_str)
        }
        Value::String(s) => {
            if writer.needs_quoting(s, context) {
                writer.write_quoted_string(s)
            } else {
                writer.write_str(s)
            }
        }
        _ => Err(ToonError::InvalidInput(
            "Expected primitive value".to_string(),
        )),
    }
}

fn encode_tabular_array(
    writer: &mut writer::Writer,
    key: Option<&str>,
    arr: &[Value],
    keys: &[String],
    depth: usize,
) -> ToonResult<()> {
    writer.write_array_header(key, arr.len(), Some(keys), depth)?;
    writer.write_newline()?;

    writer.push_active_delimiter(writer.options.delimiter);

    // Write each row with values separated by delimiters
    for (row_index, obj_val) in arr.iter().enumerate() {
        if let Some(obj) = obj_val.as_object() {
            writer.write_indent(depth + 1)?;

            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    writer.write_delimiter()?;
                }

                // Missing fields become null
                if let Some(val) = obj.get(key) {
                    write_primitive_value(writer, val, QuotingContext::ArrayValue)?;
                } else {
                    writer.write_str("null")?;
                }
            }

            if row_index < arr.len() - 1 {
                writer.write_newline()?;
            }
        }
    }

    Ok(())
}

/// Encode a tabular array as the first field of a list-item object.
///
/// Tabular rows appear at depth +2 relative to the hyphen line when the array
/// is the first field of a list-item object. This function handles that special
/// indentation requirement.
///
/// Note: The array header is written separately before calling this function.
fn encode_list_item_tabular_array(
    writer: &mut writer::Writer,
    arr: &[Value],
    keys: &[String],
    depth: usize,
) -> ToonResult<()> {
    // Write array header without key (key already written on hyphen line)
    writer.write_char('[')?;
    writer.write_str(&arr.len().to_string())?;

    if writer.options.delimiter != crate::types::Delimiter::Comma {
        writer.write_char(writer.options.delimiter.as_char())?;
    }

    writer.write_char(']')?;

    // Write field list for tabular arrays: {field1,field2}
    writer.write_char('{')?;
    for (i, field) in keys.iter().enumerate() {
        if i > 0 {
            writer.write_char(writer.options.delimiter.as_char())?;
        }
        writer.write_key(field)?;
    }
    writer.write_char('}')?;
    writer.write_char(':')?;
    writer.write_newline()?;

    writer.push_active_delimiter(writer.options.delimiter);

    // Write rows at depth + 2 (relative to hyphen line)
    // The hyphen line is at depth, so rows appear at depth + 2
    for (row_index, obj_val) in arr.iter().enumerate() {
        if let Some(obj) = obj_val.as_object() {
            writer.write_indent(depth + 2)?;

            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    writer.write_delimiter()?;
                }

                // Missing fields become null
                if let Some(val) = obj.get(key) {
                    write_primitive_value(writer, val, QuotingContext::ArrayValue)?;
                } else {
                    writer.write_str("null")?;
                }
            }

            if row_index < arr.len() - 1 {
                writer.write_newline()?;
            }
        }
    }

    writer.pop_active_delimiter();

    Ok(())
}

fn encode_nested_array(
    writer: &mut writer::Writer,
    key: Option<&str>,
    arr: &[Value],
    depth: usize,
) -> ToonResult<()> {
    writer.write_array_header(key, arr.len(), None, depth)?;
    writer.write_newline()?;
    writer.push_active_delimiter(writer.options.delimiter);

    for (i, val) in arr.iter().enumerate() {
        writer.write_indent(depth + 1)?;
        writer.write_char('-')?;

        match val {
            Value::Array(inner_arr) => {
                writer.write_char(' ')?;
                write_array(writer, None, inner_arr, depth + 1)?;
            }
            Value::Object(obj) => {
                // Objects in list items: first field on same line as "- ", rest indented
                // For empty objects, write only the hyphen (no space)
                let keys: Vec<&String> = obj.keys().collect();
                if let Some(first_key) = keys.first() {
                    writer.write_char(' ')?;
                    let first_val = &obj[*first_key];

                    match first_val {
                        Value::Array(arr) => {
                            // Arrays as first field of list items require special indentation
                            // (depth +2 relative to hyphen) for their nested content
                            // (rows for tabular, items for non-uniform)
                            writer.write_key(first_key)?;

                            if let Some(keys) = is_tabular_array(arr) {
                                // Tabular array: write inline with correct indentation
                                encode_list_item_tabular_array(writer, arr, &keys, depth + 1)?;
                            } else {
                                // Non-tabular array: write with depth offset
                                // (items at depth +2 instead of depth +1)
                                write_array(writer, None, arr, depth + 2)?;
                            }
                        }
                        Value::Object(nested_obj) => {
                            writer.write_key(first_key)?;
                            writer.write_char(':')?;
                            if !nested_obj.is_empty() {
                                writer.write_newline()?;
                                write_object(writer, nested_obj, depth + 3)?;
                            }
                        }
                        _ => {
                            writer.write_key(first_key)?;
                            writer.write_char(':')?;
                            writer.write_char(' ')?;
                            write_primitive_value(writer, first_val, QuotingContext::ObjectValue)?;
                        }
                    }

                    // Remaining fields on separate lines with proper indentation
                    for key in keys.iter().skip(1) {
                        writer.write_newline()?;
                        writer.write_indent(depth + 2)?;

                        let value = &obj[*key];
                        match value {
                            Value::Array(arr) => {
                                writer.write_key(key)?;
                                write_array(writer, None, arr, depth + 1)?;
                            }
                            Value::Object(nested_obj) => {
                                writer.write_key(key)?;
                                writer.write_char(':')?;
                                if !nested_obj.is_empty() {
                                    writer.write_newline()?;
                                    write_object(writer, nested_obj, depth + 3)?;
                                }
                            }
                            _ => {
                                writer.write_key(key)?;
                                writer.write_char(':')?;
                                writer.write_char(' ')?;
                                write_primitive_value(writer, value, QuotingContext::ObjectValue)?;
                            }
                        }
                    }
                }
            }
            _ => {
                writer.write_char(' ')?;
                write_primitive_value(writer, val, QuotingContext::ArrayValue)?;
            }
        }

        if i < arr.len() - 1 {
            writer.write_newline()?;
        }
    }
    writer.pop_active_delimiter();

    Ok(())
}

#[cfg(test)]
mod tests {
    use core::f64;

    use serde_json::json;

    use super::*;

    #[test]
    fn test_encode_null() {
        let value = json!(null);
        assert_eq!(encode_default(&value).unwrap(), "null");
    }

    #[test]
    fn test_encode_bool() {
        assert_eq!(encode_default(&json!(true)).unwrap(), "true");
        assert_eq!(encode_default(&json!(false)).unwrap(), "false");
    }

    #[test]
    fn test_encode_number() {
        assert_eq!(encode_default(&json!(42)).unwrap(), "42");
        assert_eq!(
            encode_default(&json!(f64::consts::PI)).unwrap(),
            "3.141592653589793"
        );
        assert_eq!(encode_default(&json!(-5)).unwrap(), "-5");
    }

    #[test]
    fn test_encode_string() {
        assert_eq!(encode_default(&json!("hello")).unwrap(), "hello");
        assert_eq!(
            encode_default(&json!("hello world")).unwrap(),
            "hello world"
        );
    }

    #[test]
    fn test_encode_simple_object() {
        let obj = json!({"name": "Alice", "age": 30});
        let result = encode_default(&obj).unwrap();
        assert!(result.contains("name: Alice"));
        assert!(result.contains("age: 30"));
    }

    #[test]
    fn test_encode_primitive_array() {
        let obj = json!({"tags": ["reading", "gaming", "coding"]});
        let result = encode_default(&obj).unwrap();
        assert_eq!(result, "tags[3]: reading,gaming,coding");
    }

    #[test]
    fn test_encode_tabular_array() {
        let obj = json!({
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ]
        });
        let result = encode_default(&obj).unwrap();
        assert!(result.contains("users[2]{id,name}:"));
        assert!(result.contains("1,Alice"));
        assert!(result.contains("2,Bob"));
    }

    #[test]
    fn test_encode_empty_array() {
        let obj = json!({"items": []});
        let result = encode_default(&obj).unwrap();
        assert_eq!(result, "items[0]:");
    }

    #[test]
    fn test_encode_nested_object() {
        let obj = json!({
            "user": {
                "name": "Alice",
                "age": 30
            }
        });
        let result = encode_default(&obj).unwrap();
        assert!(result.contains("user:"));
        assert!(result.contains("name: Alice"));
        assert!(result.contains("age: 30"));
    }

    #[test]
    fn test_encode_list_item_tabular_array_v3() {
        let obj = json!({
            "items": [
                {
                    "users": [
                        {"id": 1, "name": "Ada"},
                        {"id": 2, "name": "Bob"}
                    ],
                    "status": "active"
                }
            ]
        });

        let result = encode_default(&obj).unwrap();

        assert!(
            result.contains("  - users[2]{id,name}:"),
            "Header should be on hyphen line"
        );

        assert!(
            result.contains("      1,Ada"),
            "First row should be at 6 spaces (depth +2 from hyphen). Got:\n{}",
            result
        );
        assert!(
            result.contains("      2,Bob"),
            "Second row should be at 6 spaces (depth +2 from hyphen). Got:\n{}",
            result
        );

        assert!(
            result.contains("    status: active"),
            "Sibling field should be at 4 spaces (depth +1 from hyphen). Got:\n{}",
            result
        );
    }

    #[test]
    fn test_encode_list_item_tabular_array_multiple_items() {
        let obj = json!({
            "data": [
                {
                    "records": [
                        {"id": 1, "val": "x"}
                    ],
                    "count": 1
                },
                {
                    "records": [
                        {"id": 2, "val": "y"}
                    ],
                    "count": 1
                }
            ]
        });

        let result = encode_default(&obj).unwrap();

        let lines: Vec<&str> = result.lines().collect();

        let row_lines: Vec<&str> = lines
            .iter()
            .filter(|line| line.trim().starts_with(char::is_numeric))
            .copied()
            .collect();

        for row in row_lines {
            let spaces = row.len() - row.trim_start().len();
            assert_eq!(
                spaces, 6,
                "Tabular rows should be at 6 spaces. Found {} spaces in: {}",
                spaces, row
            );
        }
    }

    #[test]
    fn test_encode_list_item_non_tabular_array_unchanged() {
        let obj = json!({
            "items": [
                {
                    "tags": ["a", "b", "c"],
                    "name": "test"
                }
            ]
        });

        let result = encode_default(&obj).unwrap();

        assert!(
            result.contains("  - tags[3]: a,b,c"),
            "Inline array should be on hyphen line. Got:\n{}",
            result
        );

        assert!(
            result.contains("    name: test"),
            "Sibling field should be at 4 spaces. Got:\n{}",
            result
        );
    }

    #[test]
    fn test_encode_list_item_tabular_array_with_nested_fields() {
        let obj = json!({
            "entries": [
                {
                    "people": [
                        {"name": "Alice", "age": 30},
                        {"name": "Bob", "age": 25}
                    ],
                    "total": 2,
                    "category": "staff"
                }
            ]
        });

        let result = encode_default(&obj).unwrap();

        assert!(result.contains("  - people[2]{name,age}:"));

        assert!(result.contains("      Alice,30"));
        assert!(result.contains("      Bob,25"));

        assert!(result.contains("    total: 2"));
        assert!(result.contains("    category: staff"));
    }
}

File: encode\primitives.rs
==========================
pub fn is_primitive(value: &serde_json::Value) -> bool {
    matches!(
        value,
        serde_json::Value::Null
            | serde_json::Value::Bool(_)
            | serde_json::Value::Number(_)
            | serde_json::Value::String(_)
    )
}

pub fn all_primitives(values: &[serde_json::Value]) -> bool {
    values.iter().all(is_primitive)
}

/// Recursively normalize JSON values.
pub fn normalize_value(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Null => serde_json::Value::Null,
        serde_json::Value::Bool(b) => serde_json::Value::Bool(b),
        serde_json::Value::Number(n) => serde_json::Value::Number(n),
        serde_json::Value::String(s) => serde_json::Value::String(s),
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(normalize_value).collect())
        }
        serde_json::Value::Object(obj) => serde_json::Value::Object(
            obj.into_iter()
                .map(|(k, v)| (k, normalize_value(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_is_primitive() {
        assert!(is_primitive(&json!(null)));
        assert!(is_primitive(&json!(true)));
        assert!(is_primitive(&json!(42)));
        assert!(is_primitive(&json!("hello")));
        assert!(!is_primitive(&json!([])));
        assert!(!is_primitive(&json!({})));
    }

    #[test]
    fn test_all_primitives() {
        assert!(all_primitives(&[json!(1), json!(2), json!(3)]));
        assert!(all_primitives(&[json!("a"), json!("b")]));
        assert!(all_primitives(&[json!(null), json!(true), json!(42)]));
        assert!(!all_primitives(&[json!(1), json!([]), json!(3)]));
        assert!(!all_primitives(&[json!({})]));
    }

    #[test]
    fn test_normalize_value() {
        assert_eq!(normalize_value(json!(null)), json!(null));
        assert_eq!(normalize_value(json!(42)), json!(42));
        assert_eq!(normalize_value(json!("hello")), json!("hello"));

        let normalized = normalize_value(json!({"a": 1, "b": [2, 3]}));
        assert_eq!(normalized, json!({"a": 1, "b": [2, 3]}));
    }
}

File: rune\ast.rs
=================
/* src/rune/ast.rs */
//! RUNE Abstract Syntax Tree (AST) definitions.
//!
//! # TOON-RUNE – RUNE AST Module
//!▫~•◦---------------------------‣
//!
//! This module defines the core expression tree structures for RUNE:
//! identifiers, literals, terms, and expressions built on `RuneOp`.
//! It also includes statement-level constructs for TOON blocks and root declarations.
//!
//! The AST is intentionally minimal and expression-centric. Higher-level
//! constructs (definitions, constraints, blocks) can be layered on top
//! without changing the fundamental expression nodes.
//!
//! ### Key Types
//! - [`Literal`] – Numeric values.
//! - [`Ident`]   – Symbolic names (types, tensors, nodes, roots).
//! - [`Term`]    – Basic units: identifiers, literals, grouped expressions.
//! - [`Expr`]    – Recursive expression tree parameterized by [`RuneOp`].
//! - [`Stmt`]    – Top-level statements: root declarations, TOON blocks, expressions.
//!
//! ### Example
//! ```rust
//! use rune_format::rune::{Stmt, Expr};
//! use rune_format::rune::RuneOp;
//!
//! let root_stmt = Stmt::root("continuum");
//! let expr_stmt = Stmt::expr(
//!     Expr::binary(
//!         Expr::ident("users"),
//!         RuneOp::Descendant,
//!         Expr::ident("0"),
//!     )
//! );
//! ```
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::rune::ops::{MathOp, RuneOp};
use std::fmt;

/// A symbolic identifier in RUNE.
///
/// This covers type symbols (`T`, `Gf8`, `XUID`), nodes, roots,
/// fields, and any named entities.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ident(pub String);

impl Ident {
    pub fn new<S: Into<String>>(s: S) -> Self {
        Ident(s.into())
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<S: Into<String>> From<S> for Ident {
    fn from(s: S) -> Self {
        Ident::new(s)
    }
}

/// A semantic identifier with a single-letter namespace prefix.
///
/// Examples: T:Gf8, V:vector, R:continuum, Q:e32l
/// The prefix is always a single uppercase letter (A-Z).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticIdent {
    /// The semantic prefix (A-Z)
    pub prefix: char,
    /// The identifier name
    pub name: Ident,
}

impl SemanticIdent {
    pub fn new(prefix: char, name: impl Into<String>) -> Self {
        SemanticIdent {
            prefix,
            name: Ident::new(name),
        }
    }
}

impl fmt::Display for SemanticIdent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.prefix, self.name)
    }
}

/// Literal values in RUNE expressions.
///
/// Currently supports numeric literals, strings, and arrays.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// Numeric literal (parsed as f64).
    Number(f64),
    /// String literal.
    String(String),
    /// Array literal: [1,2,3] or [a,b,c]
    Array(Vec<Expr>),
}

impl Literal {
    pub fn number<N: Into<f64>>(n: N) -> Self {
        Literal::Number(n.into())
    }

    pub fn string<S: Into<String>>(s: S) -> Self {
        Literal::String(s.into())
    }

    pub fn array(elements: Vec<Expr>) -> Self {
        Literal::Array(elements)
    }
}

/// Arithmetic expressions within `[...]` value blocks.
///
/// These support traditional math with operators: `+ - * /`.
/// Isolated from glyph operators for clean separation.
#[derive(Debug, Clone, PartialEq)]
pub enum MathExpr {
    /// A single math atom (identifier, number, or grouped math).
    Atom(MathAtom),

    /// A binary math operation `lhs op rhs`.
    Binary {
        left: Box<MathExpr>,
        op: MathOp,
        right: Box<MathExpr>,
    },

    /// A unary math operation `op expr` (e.g., `-x`, `+5`).
    Unary {
        op: MathUnaryOp,
        operand: Box<MathExpr>,
    },
}

/// Unary operators in arithmetic expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathUnaryOp {
    /// Negation `-x`.
    Negate,
    /// Positive `+x` (typically a no-op).
    Plus,
}

impl MathUnaryOp {
    pub fn as_str(self) -> &'static str {
        match self {
            MathUnaryOp::Negate => "-",
            MathUnaryOp::Plus => "+",
        }
    }
}

/// Atoms in arithmetic expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum MathAtom {
    /// Numeric literal.
    Number(f64),
    /// Variable identifier.
    Ident(Ident),
    /// Grouped sub-expression `(math)` (for precedence).
    Group(Box<MathExpr>),
    /// Array literal inside math block `[expr, expr, ...]`.
    Array(Vec<MathExpr>),
}

impl MathExpr {
    /// Create a math atom expression.
    pub fn atom(atom: MathAtom) -> Self {
        MathExpr::Atom(atom)
    }

    /// Create a binary math expression.
    pub fn binary(left: MathExpr, op: MathOp, right: MathExpr) -> Self {
        MathExpr::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    /// Create a unary math expression.
    pub fn unary(op: MathUnaryOp, operand: MathExpr) -> Self {
        MathExpr::Unary {
            op,
            operand: Box::new(operand),
        }
    }
}

impl fmt::Display for MathExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathExpr::Atom(atom) => match atom {
                MathAtom::Number(n) => write!(f, "{}", n),
                MathAtom::Ident(id) => write!(f, "{}", id),
                MathAtom::Group(inner) => write!(f, "({})", inner),
                MathAtom::Array(elements) => {
                    write!(f, "[")?;
                    for (i, elem) in elements.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", elem)?;
                    }
                    write!(f, "]")
                }
            },
            MathExpr::Binary { left, op, right } => {
                // Add parens for clarity in nested operations
                write!(f, "{} {} {}", left, op, right)
            }
            MathExpr::Unary { op, operand } => {
                write!(f, "{}{}", op.as_str(), operand)
            }
        }
    }
}

/// Atomic terms in a RUNE expression.
///
/// These are the building blocks that operators connect.
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// A named symbol (identifier).
    Ident(Ident),
    /// A semantic identifier with namespace prefix (e.g., T:Gf8, V:vector).
    SemanticIdent(SemanticIdent),
    /// A literal value.
    Literal(Literal),
    /// A grouped sub-expression `(expr)`.
    Group(Box<Expr>),
    /// Arithmetic within `[...]` value blocks.
    Math(Box<MathExpr>),
}

impl Term {
    pub fn ident<S: Into<String>>(s: S) -> Self {
        Term::Ident(Ident::new(s))
    }

    pub fn semantic_ident(prefix: char, name: impl Into<String>) -> Self {
        Term::SemanticIdent(SemanticIdent::new(prefix, name))
    }

    pub fn literal<N: Into<f64>>(n: N) -> Self {
        Term::Literal(Literal::number(n))
    }

    pub fn group(expr: Expr) -> Self {
        Term::Group(Box::new(expr))
    }

    pub fn math(math: MathExpr) -> Self {
        Term::Math(Box::new(math))
    }
}

/// A full RUNE expression.
///
/// This is the node-level representation that a Pratt parser will
/// construct from a token stream (`Term`s and `RuneOp`s).
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A single term (identifier, literal, or grouped expression).
    Term(Term),

    /// A binary expression `lhs op rhs`.
    Binary {
        left: Box<Expr>,
        op: RuneOp,
        right: Box<Expr>,
    },
}

impl Expr {
    /// Construct a term expression from an identifier.
    pub fn ident<S: Into<String>>(s: S) -> Self {
        Expr::Term(Term::ident(s))
    }

    /// Construct a term expression from a numeric literal.
    pub fn literal<N: Into<f64>>(n: N) -> Self {
        Expr::Term(Term::literal(n))
    }

    /// Construct a grouped expression `(expr)`.
    pub fn group(expr: Expr) -> Self {
        Expr::Term(Term::group(expr))
    }

    /// Construct a binary expression `left op right`.
    pub fn binary(left: Expr, op: RuneOp, right: Expr) -> Self {
        Expr::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Term(t) => match t {
                Term::Ident(id) => write!(f, "{}", id),
                Term::SemanticIdent(sid) => write!(f, "{}", sid),
                Term::Literal(Literal::Number(n)) => write!(f, "{}", n),
                Term::Literal(Literal::String(s)) => write!(f, "\"{}\"", s),
                Term::Literal(Literal::Array(elements)) => {
                    write!(f, "[")?;
                    for (i, elem) in elements.iter().enumerate() {
                        if i > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "{}", elem)?;
                    }
                    write!(f, "]")
                }
                Term::Group(inner) => write!(f, "({})", inner),
                Term::Math(math) => write!(f, "[{}]", math),
            },
            Expr::Binary { left, op, right } => {
                // Don't add spaces around :: (namespace operator)
                if *op == RuneOp::Namespace {
                    write!(f, "{}::{}", left, right)
                } else {
                    write!(f, "{} {} {}", left, op, right)
                }
            }
        }
    }
}

/// Top-level RUNE statements.
///
/// These are the syntactic units parsed from RUNE files:
/// root declarations anchor contexts, TOON blocks provide raw data,
/// and expressions allow symbolic computations over that data.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// A root declaration: `root: name`
    /// Anchors the semantic context of the document.
    RootDecl(Ident),

    /// A TOON block: `name ~TOON:\n  content`
    /// Raw TOON data preserved verbatim for later parsing by the TOON library.
    ToonBlock { name: Ident, content: String },

    /// A RUNE expression statement.
    /// Typically constraints, definitions, or relations over TOON data.
    Expr(Expr),
}

impl Stmt {
    /// Create a root declaration statement.
    pub fn root<S: Into<String>>(name: S) -> Self {
        Stmt::RootDecl(Ident::new(name))
    }

    /// Create a TOON block statement.
    pub fn toon_block<S: Into<String>>(name: S, content: String) -> Self {
        Stmt::ToonBlock {
            name: Ident::new(name),
            content,
        }
    }

    /// Create an expression statement.
    pub fn expr(expr: Expr) -> Self {
        Stmt::Expr(expr)
    }
}

impl fmt::Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stmt::RootDecl(name) => write!(f, "root: {}", name),
            Stmt::ToonBlock { name, content } => {
                writeln!(f, "{} ~TOON:", name)?;
                for line in content.lines() {
                    writeln!(f, "  {}", line)?;
                }
                Ok(())
            }
            Stmt::Expr(expr) => write!(f, "{}", expr),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rune::ops::RuneOp;

    #[test]
    fn test_expr_binary() {
        let left = Expr::ident("users");
        let right = Expr::literal(0.0);
        let expr = Expr::binary(left, RuneOp::Descendant, right);
        assert_eq!(format!("{}", expr), "users / 0");
    }

    #[test]
    fn test_stmt_root() {
        let stmt = Stmt::root("continuum");
        assert_eq!(format!("{}", stmt), "root: continuum");
    }

    #[test]
    fn test_stmt_toon_block() {
        let content = "items[2]{id,name}:\n  1,hello\n  2,world".to_string();
        let stmt = Stmt::toon_block("data", content);
        let output = format!("{}", stmt);
        assert!(output.contains("data ~TOON:"));
        assert!(output.contains("  items[2]"));
    }
}

File: rune\mod.rs
=================
/* src/rune/hydron/mod.rs */
//! RUNE (Root-Unified Notation Encoding) is a semantic extension built on top of TOON.
//! Where TOON provides token-efficient data serialization, RUNE adds:
//!
//! - **Root-oriented semantics**: Everything revolves around hierarchical roots
//! - **Operator calculus**: Glyphs and tokens for describing relationships, flow, and structure
//! - **E8-awareness**: Geometric and identity-aware operators
//! - **Composability**: Mix RUNE semantics with TOON data blocks seamlessly
//!
//! ## Overview
//!
//! RUNE files can contain:
//! - **TOON blocks**: Raw TOON data (preserved verbatim)
//! - **RUNE operators**: Relations, constraints, transformations over TOON data
//! - **Root declarations**: Anchor points in your E8 ecosystem
//!
//! ## Example RUNE File
//! ```rune
//! root: continuum
//!
//! data ~TOON:
//!   users[3]{id,name,role}:
//!     1,Ada,admin
//!     2,Bob,user
//!     3,Eve,viewer
//!
//! # RUNE semantics over TOON data
//! users / 0 -> role := admin
//! users / * -> name ~ ValidString()
//! ```
//!
//! This crate leverages the TOON format as foundational data representation
//! while adding symbolic operator layers for E8 ecosystems.
//!
//! TOKEN_FORMAT is Copyright (c) 2025-PRESENT Shreyas S Bhat, Johann Schopplich
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

mod ast;
mod ops;
mod parser;

#[cfg(feature = "hydron")]
pub mod hydron;

pub use ast::*;
pub use ops::*;
pub use parser::*;

// Re-export common types for convenience
pub type RuneParser = parser::ParseError;

/// Parse a RUNE source string into a list of statements.
pub fn parse_rune(input: &str) -> Result<Vec<Stmt>, ParseError> {
    parser::parse(input)
}

/// Encode TOON data blocks within RUNE files as raw strings.
pub fn encode_rune(statements: &[Stmt]) -> String {
    let mut output = String::new();
    for stmt in statements {
        output.push_str(&format!("{}\n", stmt));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_toon_block() {
        let input = r#"
root: test_root

data ~TOON:
  items[2]{id,name}:
    1,hello
    2,world
"#;
        let stmts = parse_rune(input).unwrap();
        assert_eq!(stmts.len(), 2);
        // First statement is root declaration
        if let Stmt::RootDecl(root) = &stmts[0] {
            assert_eq!(root.0.as_str(), "test_root");
        } else {
            panic!("Expected root declaration");
        }
        // Second is TOON block
        if let Stmt::ToonBlock { name, content } = &stmts[1] {
            assert_eq!(name.0.as_str(), "data");
            assert!(content.contains("items[2]"));
        } else {
            panic!("Expected TOON block");
        }
    }

    #[test]
    fn test_operator_expression() {
        let input = r#"
items / 0 -> name := hello
"#;
        let stmts = parse_rune(input).unwrap();
        assert_eq!(stmts.len(), 1);
        if let Stmt::Expr(expr) = &stmts[0] {
            // Check it's a binary expression with -> operator (lower precedence)
            // Parses as: items / 0 -> (name := hello)
            if let Expr::Binary { op, left, right } = expr {
                assert_eq!(*op, RuneOp::FlowRight);
                assert_eq!(format!("{}", left), "items / 0");
                assert_eq!(format!("{}", right), "name := hello");
            } else {
                panic!("Expected binary expression");
            }
        }
    }
}

File: encode\writer.rs
======================
use crate::{
    types::{Delimiter, EncodeOptions, ToonResult},
    utils::{
        QuotingContext,
        string::{is_valid_unquoted_key, needs_quoting, quote_string},
    },
};

/// Writer that builds TOON output string from JSON values.
pub struct Writer {
    buffer: String,
    pub(crate) options: EncodeOptions,
    active_delimiters: Vec<Delimiter>,
}

impl Writer {
    /// Create a new writer with the given options.
    pub fn new(options: EncodeOptions) -> Self {
        Self {
            buffer: String::new(),
            active_delimiters: vec![options.delimiter],
            options,
        }
    }

    /// Finish writing and return the complete TOON string.
    pub fn finish(self) -> String {
        self.buffer
    }

    pub fn write_str(&mut self, s: &str) -> ToonResult<()> {
        self.buffer.push_str(s);
        Ok(())
    }

    pub fn write_char(&mut self, ch: char) -> ToonResult<()> {
        self.buffer.push(ch);
        Ok(())
    }

    pub fn write_newline(&mut self) -> ToonResult<()> {
        self.buffer.push('\n');
        Ok(())
    }

    pub fn write_indent(&mut self, depth: usize) -> ToonResult<()> {
        let indent_string = self.options.indent.get_string(depth);
        if !indent_string.is_empty() {
            self.buffer.push_str(&indent_string);
        }
        Ok(())
    }

    pub fn write_delimiter(&mut self) -> ToonResult<()> {
        self.buffer.push(self.options.delimiter.as_char());
        Ok(())
    }

    pub fn write_key(&mut self, key: &str) -> ToonResult<()> {
        if is_valid_unquoted_key(key) {
            self.write_str(key)
        } else {
            self.write_quoted_string(key)
        }
    }

    /// Write an array header with key, length, and optional field list.
    pub fn write_array_header(
        &mut self,
        key: Option<&str>,
        length: usize,
        fields: Option<&[String]>,
        depth: usize,
    ) -> ToonResult<()> {
        if let Some(k) = key {
            if depth > 0 {
                self.write_indent(depth)?;
            }
            self.write_key(k)?;
        }

        self.write_char('[')?;
        self.write_str(&length.to_string())?;

        // Only write delimiter in header if it's not comma (comma is default/implied)
        if self.options.delimiter != Delimiter::Comma {
            self.write_delimiter()?;
        }

        self.write_char(']')?;

        // Write field list for tabular arrays: {field1,field2}
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

    /// Write an empty array header.
    pub fn write_empty_array_with_key(
        &mut self,
        key: Option<&str>,
        depth: usize,
    ) -> ToonResult<()> {
        if let Some(k) = key {
            if depth > 0 {
                self.write_indent(depth)?;
            }
            self.write_key(k)?;
        }
        self.write_char('[')?;
        self.write_str("0")?;

        if self.options.delimiter != Delimiter::Comma {
            self.write_delimiter()?;
        }

        self.write_char(']')?;
        self.write_char(':')
    }

    pub fn needs_quoting(&self, s: &str, context: QuotingContext) -> bool {
        // Use active delimiter for array values, document delimiter for object values
        let delim_char = match context {
            QuotingContext::ObjectValue => self.get_document_delimiter_char(),
            QuotingContext::ArrayValue => self.get_active_delimiter_char(),
        };
        needs_quoting(s, delim_char)
    }

    pub fn write_quoted_string(&mut self, s: &str) -> ToonResult<()> {
        self.write_str(&quote_string(s))
    }

    pub fn write_value(&mut self, s: &str, context: QuotingContext) -> ToonResult<()> {
        if self.needs_quoting(s, context) {
            self.write_quoted_string(s)
        } else {
            self.write_str(s)
        }
    }

    /// Push a new delimiter onto the stack (for nested arrays with different
    /// delimiters).
    pub fn push_active_delimiter(&mut self, delim: Delimiter) {
        self.active_delimiters.push(delim);
    }
    /// Pop the active delimiter, keeping at least one (the document default).
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

    #[test]
    fn test_writer_basic() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer.write_str("hello").unwrap();
        writer.write_str(" ").unwrap();
        writer.write_str("world").unwrap();

        assert_eq!(writer.finish(), "hello world");
    }

    #[test]
    fn test_write_delimiter() {
        let mut opts = EncodeOptions::default();
        let mut writer = Writer::new(opts.clone());

        writer.write_str("a").unwrap();
        writer.write_delimiter().unwrap();
        writer.write_str("b").unwrap();

        assert_eq!(writer.finish(), "a,b");

        opts = opts.with_delimiter(Delimiter::Pipe);
        let mut writer = Writer::new(opts);

        writer.write_str("a").unwrap();
        writer.write_delimiter().unwrap();
        writer.write_str("b").unwrap();

        assert_eq!(writer.finish(), "a|b");
    }

    #[test]
    fn test_write_indent() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer.write_indent(0).unwrap();
        writer.write_str("a").unwrap();
        writer.write_newline().unwrap();

        writer.write_indent(1).unwrap();
        writer.write_str("b").unwrap();
        writer.write_newline().unwrap();

        writer.write_indent(2).unwrap();
        writer.write_str("c").unwrap();

        assert_eq!(writer.finish(), "a\n  b\n    c");
    }

    #[test]
    fn test_write_array_header() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer
            .write_array_header(Some("items"), 3, None, 0)
            .unwrap();
        assert_eq!(writer.finish(), "items[3]:");

        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);
        let fields = vec!["id".to_string(), "name".to_string()];

        writer
            .write_array_header(Some("users"), 2, Some(&fields), 0)
            .unwrap();
        assert_eq!(writer.finish(), "users[2]{id,name}:");
    }

    #[test]
    fn test_write_array_header_with_pipe_delimiter() {
        let opts = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
        let mut writer = Writer::new(opts);

        writer
            .write_array_header(Some("items"), 3, None, 0)
            .unwrap();
        assert_eq!(writer.finish(), "items[3|]:");

        let opts = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
        let mut writer = Writer::new(opts);
        let fields = vec!["id".to_string(), "name".to_string()];

        writer
            .write_array_header(Some("users"), 2, Some(&fields), 0)
            .unwrap();
        assert_eq!(writer.finish(), "users[2|]{id|name}:");
    }

    #[test]
    fn test_write_key_with_special_chars() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer.write_key("normal_key").unwrap();
        assert_eq!(writer.finish(), "normal_key");

        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer.write_key("key:with:colons").unwrap();
        assert_eq!(writer.finish(), "\"key:with:colons\"");
    }

    #[test]
    fn test_write_quoted_string() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer.write_quoted_string("hello world").unwrap();
        assert_eq!(writer.finish(), "\"hello world\"");

        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer.write_quoted_string("say \"hi\"").unwrap();
        assert_eq!(writer.finish(), r#""say \"hi\"""#);
    }

    #[test]
    fn test_needs_quoting() {
        let opts = EncodeOptions::default();
        let writer = Writer::new(opts);
        let ctx = QuotingContext::ObjectValue;

        assert!(!writer.needs_quoting("hello", ctx));
        assert!(writer.needs_quoting("hello,world", ctx));
        assert!(writer.needs_quoting("true", ctx));
        assert!(writer.needs_quoting("false", ctx));
        assert!(writer.needs_quoting("null", ctx));
        assert!(writer.needs_quoting("123", ctx));
        assert!(writer.needs_quoting("", ctx));
        assert!(writer.needs_quoting("hello:world", ctx));
    }

    #[test]
    fn test_write_empty_array() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer.write_empty_array_with_key(Some("items"), 0).unwrap();
        assert_eq!(writer.finish(), "items[0]:");
    }
}

File: rune\ops.rs
=================
//! Core Operator Registry and Definitions for RUNE.
//!
//! # e8 Notation – RUNE Operators
//!▫~•◦----------------------------‣
//!
//! This module defines the strict, closed registry of valid RUNE operators.
//! It maps the text representations (from the grammar) to strongly-typed
//! Rust enums, ensuring that no "illegal" or "fused" operators can represent
//! a valid state in the AST.
//!
//! ### Key Capabilities
//! - **Closed Registry:** `RuneOp` enum exhaustively lists every allowed operator.
//! - **Category Safety:** Distinguishes between `Glyph` (Topological), `Relation` (Directed), and `Math` (Value).
//! - **Precedence Logic:** Defines binding power for Pratt parsing (e.g., `*` binds tighter than `+`, which binds tighter than `->`).
//!
//! ### Example
//! ```rust
//! use rune_format::rune::RuneOp;
//! use std::str::FromStr;
//!
//! let op = RuneOp::from_str("->").unwrap();
//! assert_eq!(op, RuneOp::FlowRight);
//! assert_eq!(op.category(), rune_format::rune::OpCategory::Relation);
//! ```
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::fmt;
use std::str::FromStr;

/// Categories of operators in RUNE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCategory {
    /// Topological shapes (e.g., `/\`, `\|/`).
    Glyph,
    /// Structural relations (e.g., `->`, `:`, `:=`).
    Relation,
    /// Value comparisons (e.g., `<`, `>`).
    Compare,
    /// Arithmetic operations (e.g., `+`, `*`).
    Math,
}

/// The Closed Registry of all valid RUNE operators.
///
/// Any sequence of characters not matching one of these variants
/// is syntactically invalid in RUNE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuneOp {
    // --- 1. Glyph Operators (Topology/Shape) ---
    /// `/\` : Branch then converge (Split -> Join).
    SplitJoin,
    /// `\/` : Converge then branch (Join -> Split).
    JoinSplit,
    /// `|/` : Stable lineage then branch away (Descend from Anchor).
    AnchorDescend,
    /// `/|` : Branch away then stabilize (Branch -> Stabilize).
    BranchStabilize,
    /// `\|` : Converge to root then stabilize.
    RootStabilize,
    /// `|\` : Stabilize then converge to root.
    StabilizeRoot,
    /// `\|/` : Symmetric split from a stable center.
    SymmetricSplit,
    /// `/|\` : Branch, Anchor, Branch (Composite).
    BranchAnchorBranch,

    // --- 2. Token Operators (Relations) ---
    /// `:` : Bind / Key-Value / Annotation.
    Bind,
    /// `::` : Namespace / Type Tag.
    Namespace,
    /// `:=` : Definition / Assignment.
    Define,
    /// `=` : Equality / Constraint (Invariant).
    Equal,
    /// `->` : Directed Edge (Flow Right / Rootwards).
    FlowRight,
    /// `<-` : Reverse Edge (Flow Left).
    FlowLeft,
    /// `/` : Descendant / Under (Structural Context).
    Descendant,
    /// `\` : Ancestor / Parent (Sugar for `->` in some contexts).
    Ancestor,
    /// `|` : Alias / Equivalence.
    Alias,
    /// `||` : Parallel / Siblings.
    Parallel,
    /// `~` : Transform / View.
    Transform,

    // --- 4. Comparison ---
    /// `<` : Less / Precedes / Deeper.
    Less,
    /// `<=` : Less than or equal.
    LessEqual,
    /// `>` : Greater / Succeeds / Higher.
    Greater,
    /// `>=` : Greater than or equal.
    GreaterEqual,
}

impl RuneOp {
    /// Returns the semantic category of the operator.
    pub fn category(&self) -> OpCategory {
        match self {
            Self::SplitJoin
            | Self::JoinSplit
            | Self::AnchorDescend
            | Self::BranchStabilize
            | Self::RootStabilize
            | Self::StabilizeRoot
            | Self::SymmetricSplit
            | Self::BranchAnchorBranch => OpCategory::Glyph,

            Self::Bind
            | Self::Namespace
            | Self::Define
            | Self::Equal
            | Self::FlowRight
            | Self::FlowLeft
            | Self::Descendant
            | Self::Ancestor
            | Self::Alias
            | Self::Parallel
            | Self::Transform => OpCategory::Relation,

            Self::Less | Self::LessEqual | Self::Greater | Self::GreaterEqual => {
                OpCategory::Compare
            }
        }
    }

    /// Returns the textual representation of the operator.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SplitJoin => "/\\",
            Self::JoinSplit => "\\/",
            Self::AnchorDescend => "|/",
            Self::BranchStabilize => "/|",
            Self::RootStabilize => "\\|",
            Self::StabilizeRoot => "|\\",
            Self::SymmetricSplit => "\\|/",
            Self::BranchAnchorBranch => "/|\\",

            Self::Bind => ":",
            Self::Namespace => "::",
            Self::Define => ":=",
            Self::Equal => "=",
            Self::FlowRight => "->",
            Self::FlowLeft => "<-",
            Self::Descendant => "/",
            Self::Ancestor => "\\",
            Self::Alias => "|",
            Self::Parallel => "||",
            Self::Transform => "~",

            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
        }
    }

    /// Binding Power for Pratt Parsing (Precedence).
    ///
    /// Higher numbers bind tighter.
    /// - Namespace/Path: Structural binding
    /// - Flow / Glyphs / Transform: mid-tier
    /// - Comparison: lower
    /// - Bind / Define: lowest (top-level)
    pub fn binding_power(&self) -> (u8, u8) {
        match self {
            // Namespace / Path / Hierarchy
            Self::Namespace => (70, 71),
            Self::Descendant | Self::Ancestor => (60, 61),

            // Flow / Graph Edges / Glyphs / Transform
            Self::FlowRight
            | Self::FlowLeft
            | Self::SplitJoin
            | Self::JoinSplit
            | Self::SymmetricSplit
            | Self::BranchAnchorBranch
            | Self::Transform
            | Self::AnchorDescend
            | Self::BranchStabilize
            | Self::RootStabilize
            | Self::StabilizeRoot => (50, 51),

            // Comparison
            Self::Less | Self::LessEqual | Self::Greater | Self::GreaterEqual | Self::Equal => {
                (40, 41)
            }

            // Loose Structure
            Self::Parallel | Self::Alias => (30, 31),

            // Definition / Assignment / Bind: Lowest
            Self::Bind | Self::Define => (10, 11),
        }
    }
}

/// Parsing error for invalid operator strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidOpError(pub String);

impl fmt::Display for InvalidOpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid RUNE operator literal: '{}'", self.0)
    }
}

impl std::error::Error for InvalidOpError {}

impl FromStr for RuneOp {
    type Err = InvalidOpError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            // Glyphs (3-char)
            "\\|/" => Ok(Self::SymmetricSplit),
            "/|\\" => Ok(Self::BranchAnchorBranch),

            // Glyphs (2-char)
            "/\\" => Ok(Self::SplitJoin),
            "\\/" => Ok(Self::JoinSplit),
            "|/" => Ok(Self::AnchorDescend),
            "/|" => Ok(Self::BranchStabilize),
            "\\|" => Ok(Self::RootStabilize),
            "|\\" => Ok(Self::StabilizeRoot),

            // Tokens (2-char)
            "::" => Ok(Self::Namespace),
            ":=" => Ok(Self::Define),
            "->" => Ok(Self::FlowRight),
            "<-" => Ok(Self::FlowLeft),
            "<=" => Ok(Self::LessEqual),
            ">=" => Ok(Self::GreaterEqual),
            "||" => Ok(Self::Parallel),

            // Tokens (1-char)
            ":" => Ok(Self::Bind),
            "=" => Ok(Self::Equal),
            "<" => Ok(Self::Less),
            ">" => Ok(Self::Greater),
            "/" => Ok(Self::Descendant),
            "\\" => Ok(Self::Ancestor),
            "|" => Ok(Self::Alias),
            "~" => Ok(Self::Transform),

            _ => Err(InvalidOpError(s.to_string())),
        }
    }
}

/// Arithmetic operators within math blocks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MathOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power, // ^ operator
    Modulo,
    Root, // R operator: n-th root
}

impl MathOp {
    pub fn precedence(self) -> u8 {
        match self {
            MathOp::Add | MathOp::Subtract => 1,                     // + -
            MathOp::Multiply | MathOp::Divide | MathOp::Modulo => 2, // * / %
            MathOp::Power | MathOp::Root => 3,                       // ^ R
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            MathOp::Add => "+",
            MathOp::Subtract => "-",
            MathOp::Multiply => "*",
            MathOp::Divide => "/",
            MathOp::Power => "^",
            MathOp::Modulo => "%",
            MathOp::Root => "R",
        }
    }
}

impl fmt::Display for MathOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Display for RuneOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_from_str() {
        assert_eq!(RuneOp::from_str("->").unwrap(), RuneOp::FlowRight);
        assert_eq!(RuneOp::from_str("/\\").unwrap(), RuneOp::SplitJoin);
        assert_eq!(RuneOp::from_str(":=").unwrap(), RuneOp::Define);
    }

    #[test]
    fn test_invalid_operator() {
        assert!(RuneOp::from_str("=>").is_err());
        assert!(RuneOp::from_str("/->").is_err());
        assert!(RuneOp::from_str(":|").is_err());
    }

    #[test]
    fn test_binding_power() {
        assert!(RuneOp::FlowRight.binding_power() > RuneOp::Define.binding_power());
    }
}

File: rune\parser.rs
====================
/* src/rune/parser.rs */
//!
//! # e8 Notation – RUNE Parser
//!▫~•◦-------------------------‣
//!
//! This module provides parsing functionality for RUNE source code,
//! converting text into `Stmt` structures with proper operator precedence.
//! It uses Pest for lexical analysis and implements expression parsing
//! driven by the grammar’s precedence layering.
//!
//! The parser handles:
//! - **Operator precedence** via grammar layers:
//!   - `mul`   → `*` level
//!   - `add_sub` → `+` / `-`
//!   - `access` / `relation_expr` / `expr` → structural / relation ops
//! - **TOON blocks**: Raw content preservation for later TOON library parsing
//! - **Root declarations**: Semantic anchors for E8 contexts
//! - **Expression trees**: Recursive binary structures respecting precedence
//!
//! ### Implementation Details
//! - Uses grammar-encoded precedence (`term (op term)*` per layer).
//! - Preserves TOON blocks as raw strings without internal parsing.
//! - Validates all operators against the closed `RuneOp` registry.
//!
//! ### Error Handling
//! Parser errors include:
//! - Invalid operators (not in registry)
//! - Mismatched parentheses
//! - Malformed TOON blocks
//! - Unexpected tokens
//!
//! ### Example
//! ```rust
//! use rune_format::rune::parse_rune;
//!
//! let input = r#"
//! root: continuum
//! data ~TOON:
//!   users[2]{id,name}:
//!     1,Ada
//!     2,Bob
//! users / 1 := Bob
//! "#;
//!
//! let stmts = parse_rune(input).unwrap();
//! // stmts contains RootDecl, ToonBlock, and ExprStmt
//! ```
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use std::str::FromStr;
use thiserror::Error;

use crate::rune::ast::*;
use crate::rune::ops::*;

// Pest grammar reference
#[derive(Parser)]
#[grammar = "rune/grammar.pest"]
pub struct RuneParser;

/// Root error type for parsing RUNE source code.
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Pest parse error: {0}")]
    Pest(Box<pest::error::Error<Rule>>),
    #[error("Invalid operator '{0}' not in registry")]
    InvalidOperator(String),
    #[error("Expected identifier, found: {0}")]
    ExpectedIdent(String),
    #[error("Expected number, found: {0}")]
    ExpectedNumber(String),
    #[error("Parse tree error: {0}")]
    ParseTree(String),
}

/// Parse RUNE source code into a list of statements.
pub fn parse(input: &str) -> Result<Vec<Stmt>, ParseError> {
    let pairs = RuneParser::parse(Rule::file, input).map_err(|e| ParseError::Pest(Box::new(e)))?;
    let mut stmts = Vec::new();

    for pair in pairs {
        if pair.as_rule() == Rule::file {
            for inner_pair in pair.into_inner() {
                match inner_pair.as_rule() {
                    Rule::WHITESPACE | Rule::COMMENT => {} // skip
                    Rule::stmt => {
                        if let Some(stmt_pair) = inner_pair.into_inner().next() {
                            stmts.push(parse_stmt(stmt_pair)?);
                        }
                    }
                    Rule::root_decl | Rule::toon_block | Rule::stmt_expr => {
                        stmts.push(parse_stmt(inner_pair)?);
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(stmts)
}

/// Parse a statement pair into a Stmt.
fn parse_stmt(pair: Pair<Rule>) -> Result<Stmt, ParseError> {
    let rule = pair.as_rule();
    match rule {
        Rule::root_decl => parse_root_decl(pair),
        Rule::toon_block => parse_toon_block(pair),
        Rule::stmt_expr => {
            let expr_pair = pair.into_inner().next().unwrap();
            Ok(Stmt::expr(parse_expr(expr_pair)?))
        }
        _ => Err(ParseError::ParseTree(format!(
            "Unexpected statement rule: {:?}",
            rule
        ))),
    }
}

/// Parse root declaration: `root: name` or `root: e8::continuum`
fn parse_root_decl(pair: Pair<Rule>) -> Result<Stmt, ParseError> {
    let inner = pair.into_inner();

    let mut segments = Vec::new();

    // Collect all identifier segments (with :: separators)
    for pair in inner {
        if pair.as_rule() == Rule::ident {
            segments.push(pair.as_str());
        }
    }

    if segments.is_empty() {
        return Err(ParseError::ParseTree(
            "root declaration missing identifier".to_string(),
        ));
    }

    // Join segments with :: to create the full name
    let name = segments.join("::");
    Ok(Stmt::root(&name))
}

/// Parse TOON block: `name ~TOON:\n  content\n  content`
fn parse_toon_block(pair: Pair<Rule>) -> Result<Stmt, ParseError> {
    let mut inner = pair.into_inner();
    let ident_pair = inner.next().unwrap();
    let name = ident_pair.as_str();

    // The next pair is toon_content (atomic capture of all lines)
    let content_pair = inner.next().unwrap();
    let content = content_pair.as_str();

    // Split into lines and dedent (remove common leading whitespace)
    let lines: Vec<&str> = content.lines().collect();

    if lines.is_empty() {
        return Ok(Stmt::toon_block(name, String::new()));
    }

    // Find minimum indentation (excluding empty lines)
    let min_indent = lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);

    // Remove the common indentation from all lines
    let dedented: Vec<String> = lines
        .iter()
        .map(|line| {
            if line.trim().is_empty() {
                String::new()
            } else {
                line[min_indent..].to_string()
            }
        })
        .collect();

    let final_content = dedented.join("\n");

    Ok(Stmt::toon_block(name, final_content))
}

/// Parse expression using grammar-driven precedence.
///
/// We rely on the Pest grammar to encode precedence via nested rules:
/// - `flow_expr` wraps `struct_expr` (lower precedence)
/// - `struct_expr` wraps `access` (higher precedence)
/// - `access` wraps `term`
///
/// Each non-terminal is parsed as:
///   sub_expr (op sub_expr)*
fn parse_expr(pair: Pair<Rule>) -> Result<Expr, ParseError> {
    match pair.as_rule() {
        // Expression layers for structural operators
        Rule::relation_expr | Rule::flow_expr | Rule::struct_expr | Rule::access => {
            let mut inner = pair.into_inner();

            // First element is always a sub-expression or term.
            let first = inner
                .next()
                .ok_or_else(|| ParseError::ParseTree("Empty expression".to_string()))?;

            let mut left = match first.as_rule() {
                Rule::relation_expr | Rule::flow_expr | Rule::struct_expr | Rule::access => {
                    parse_expr(first)?
                }
                Rule::term => parse_term(first)?,
                _ => parse_term(first)?,
            };

            // Then we expect zero or more (op, rhs) pairs.
            while let Some(op_pair) = inner.next() {
                // Determine what kind of pair this is
                let (op, right) = match op_pair.as_rule() {
                    // If it's one of the named operator rules, parse it
                    Rule::relation_op | Rule::flow_op | Rule::struct_op | Rule::path_op => {
                        let op = parse_operator(op_pair)?;
                        let rhs_pair = inner.next().ok_or_else(|| {
                            ParseError::ParseTree("Missing right operand".to_string())
                        })?;
                        let right = match rhs_pair.as_rule() {
                            Rule::relation_expr
                            | Rule::flow_expr
                            | Rule::struct_expr
                            | Rule::access => parse_expr(rhs_pair)?,
                            Rule::term => parse_term(rhs_pair)?,
                            _ => parse_term(rhs_pair)?,
                        };
                        (op, right)
                    }
                    // If it's another expression layer, something is wrong
                    Rule::access | Rule::struct_expr | Rule::flow_expr | Rule::relation_expr => {
                        return Err(ParseError::ParseTree(format!(
                            "Unexpected expression node where operator expected: {:?}",
                            op_pair.as_rule()
                        )));
                    }
                    _ => {
                        // Fallback: treat as operator by text
                        let op = parse_operator(op_pair)?;
                        let rhs_pair = inner.next().ok_or_else(|| {
                            ParseError::ParseTree("Missing right operand".to_string())
                        })?;
                        let right = match rhs_pair.as_rule() {
                            Rule::relation_expr
                            | Rule::flow_expr
                            | Rule::struct_expr
                            | Rule::access => parse_expr(rhs_pair)?,
                            Rule::term => parse_term(rhs_pair)?,
                            _ => parse_term(rhs_pair)?,
                        };
                        (op, right)
                    }
                };

                left = Expr::binary(left, op, right);
            }

            Ok(left)
        }
        // Direct term -> literal / ident / grouped expr
        Rule::term => parse_term(pair),
        _ => Err(ParseError::ParseTree(format!(
            "Unexpected expression rule: {:?}",
            pair.as_rule()
        ))),
    }
}

/// Parse a term: identifier, number, string, array, grouped expression, or math block.
fn parse_term(pair: Pair<Rule>) -> Result<Expr, ParseError> {
    match pair.as_rule() {
        Rule::term => {
            // Term is a composite rule, get its inner content
            let inner = pair
                .into_inner()
                .next()
                .ok_or_else(|| ParseError::ParseTree("Empty term".to_string()))?;
            parse_term(inner) // Recursively parse the inner rule
        }
        Rule::array_literal => {
            // Parse array literal: [expr, expr, ...]
            let inner = pair.into_inner();
            let mut elements = Vec::new();

            for expr_pair in inner {
                elements.push(parse_expr(expr_pair)?);
            }

            Ok(Expr::Term(Term::Literal(Literal::Array(elements))))
        }
        Rule::semantic_ident => {
            // Parse semantic identifier: prefix:name
            let mut inner = pair.into_inner();
            let prefix_pair = inner.next().unwrap();
            let name_pair = inner.next().unwrap();

            // Extract prefix character (first char before the colon)
            let prefix_str = prefix_pair.as_str();
            let prefix = prefix_str.chars().next().unwrap();

            let name = name_pair.as_str();
            Ok(Expr::Term(Term::semantic_ident(prefix, name)))
        }
        Rule::ident => Ok(Expr::ident(pair.as_str())),
        Rule::number => {
            let num: f64 = pair
                .as_str()
                .parse()
                .map_err(|_| ParseError::ExpectedNumber(pair.as_str().to_string()))?;
            Ok(Expr::literal(num))
        }
        Rule::string => {
            // Parse string, handling escape sequences
            let raw = pair.as_str();
            // Remove surrounding quotes
            let content = &raw[1..raw.len() - 1];
            // Unescape common sequences
            let unescaped = content
                .replace("\\\"", "\"")
                .replace("\\\\", "\\")
                .replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t");
            Ok(Expr::Term(Term::Literal(Literal::String(unescaped))))
        }
        Rule::relation_expr | Rule::flow_expr | Rule::struct_expr | Rule::access => {
            // These are expression nodes that can appear as terms (e.g., in parentheses)
            parse_expr(pair)
        }
        Rule::math_block => {
            // Math blocks are at the term level, parse content as math expression
            let math_expr = parse_math_expr(pair)?;
            Ok(Expr::Term(Term::Math(Box::new(math_expr))))
        }
        _ => Err(ParseError::ParseTree(format!(
            "Unexpected term rule: {:?}",
            pair.as_rule()
        ))),
    }
}

/// Parse math expression from a math block `[...]`.
/// Handles arithmetic operators with proper precedence.
fn parse_math_expr(pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
    // The pair is a math_block, we need the inner math_expr
    let math_expr_pair = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::ParseTree("Empty math block".to_string()))?;

    parse_math_expr_inner(math_expr_pair)
}

/// Internal math expression parser.
fn parse_math_expr_inner(pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
    match pair.as_rule() {
        Rule::math_expr | Rule::math_add | Rule::math_mul | Rule::math_exp => {
            let mut inner = pair.into_inner();

            let first = inner
                .next()
                .ok_or_else(|| ParseError::ParseTree("Empty math expression".to_string()))?;

            let mut left = match first.as_rule() {
                Rule::math_expr | Rule::math_add | Rule::math_mul | Rule::math_exp => {
                    parse_math_expr_inner(first)?
                }
                Rule::math_unary => parse_math_unary(first)?,
                Rule::math_atom => parse_math_atom(first)?,
                _ => parse_math_expr_inner(first)?,
            };

            while let Some(op_pair) = inner.next() {
                let op = parse_math_operator(op_pair)?;

                let rhs_pair = inner.next().ok_or_else(|| {
                    ParseError::ParseTree("Missing right operand in math".to_string())
                })?;

                let right = match rhs_pair.as_rule() {
                    Rule::math_expr | Rule::math_add | Rule::math_mul | Rule::math_exp => {
                        parse_math_expr_inner(rhs_pair)?
                    }
                    Rule::math_unary => parse_math_unary(rhs_pair)?,
                    Rule::math_atom => parse_math_atom(rhs_pair)?,
                    _ => parse_math_atom(rhs_pair)?,
                };

                left = MathExpr::binary(left, op, right);
            }

            Ok(left)
        }
        Rule::math_unary => parse_math_unary(pair),
        Rule::math_atom => parse_math_atom(pair),
        _ => Err(ParseError::ParseTree(format!(
            "Unexpected math expression rule: {:?}",
            pair.as_rule()
        ))),
    }
}

/// Parse a math atom: number, identifier, or grouped expression.
fn parse_math_atom(pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::ParseTree("Empty math atom".to_string()))?;

    match inner.as_rule() {
        Rule::number => {
            let num: f64 = inner
                .as_str()
                .parse()
                .map_err(|_| ParseError::ExpectedNumber(inner.as_str().to_string()))?;
            Ok(MathExpr::atom(MathAtom::Number(num)))
        }
        Rule::ident => Ok(MathExpr::atom(MathAtom::Ident(Ident::new(inner.as_str())))),
        Rule::semantic_ident => {
            // Parse semantic identifier inside math blocks - treat as regular identifier for now
            Ok(MathExpr::atom(MathAtom::Ident(Ident::new(inner.as_str()))))
        }
        Rule::math_array_literal => {
            // Parse array literal inside math blocks
            let elements: Result<Vec<MathExpr>, ParseError> =
                inner.into_inner().map(parse_math_expr_inner).collect();
            Ok(MathExpr::atom(MathAtom::Array(elements?)))
        }
        Rule::math_expr => Ok(MathExpr::atom(MathAtom::Group(Box::new(
            parse_math_expr_inner(inner)?,
        )))),
        _ => Err(ParseError::ParseTree(format!(
            "Unexpected math atom rule: {:?}",
            inner.as_rule()
        ))),
    }
}

/// Parse unary expression: optional prefix operator followed by atom.
fn parse_math_unary(pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
    let mut inner = pair.into_inner();

    let first = inner
        .next()
        .ok_or_else(|| ParseError::ParseTree("Empty unary expression".to_string()))?;

    // Check if first is a unary operator
    match first.as_rule() {
        Rule::math_unary_op => {
            let op = parse_math_unary_operator(first)?;
            let operand_pair = inner.next().ok_or_else(|| {
                ParseError::ParseTree("Missing operand after unary operator".to_string())
            })?;

            let operand = match operand_pair.as_rule() {
                Rule::math_atom => parse_math_atom(operand_pair)?,
                Rule::math_unary => parse_math_unary(operand_pair)?,
                _ => {
                    return Err(ParseError::ParseTree(format!(
                        "Unexpected unary operand rule: {:?}",
                        operand_pair.as_rule()
                    )));
                }
            };

            Ok(MathExpr::unary(op, operand))
        }
        Rule::math_atom => parse_math_atom(first),
        _ => Err(ParseError::ParseTree(format!(
            "Unexpected unary rule: {:?}",
            first.as_rule()
        ))),
    }
}

/// Parse math operator into MathOp.
fn parse_math_operator(pair: Pair<Rule>) -> Result<MathOp, ParseError> {
    match pair.as_str().trim() {
        "+" => Ok(MathOp::Add),
        "-" => Ok(MathOp::Subtract),
        "*" => Ok(MathOp::Multiply),
        "/" => Ok(MathOp::Divide),
        "%" => Ok(MathOp::Modulo),
        "^" => Ok(MathOp::Power),
        "R" => Ok(MathOp::Root),
        op => Err(ParseError::InvalidOperator(format!(
            "Unknown math operator: {}",
            op
        ))),
    }
}

/// Parse unary operator into MathUnaryOp.
fn parse_math_unary_operator(pair: Pair<Rule>) -> Result<MathUnaryOp, ParseError> {
    match pair.as_str().trim() {
        "-" => Ok(MathUnaryOp::Negate),
        "+" => Ok(MathUnaryOp::Plus),
        op => Err(ParseError::InvalidOperator(format!(
            "Unknown unary operator: {}",
            op
        ))),
    }
}

/// Parse operator token into RuneOp.
///
/// We defensively trim whitespace so that rules which
/// include incidental spaces around operators do not
/// accidentally produce `"+"`, `"1 "` or `"b * c"` as a
/// single operator token.
fn parse_operator(pair: Pair<Rule>) -> Result<RuneOp, ParseError> {
    let text = pair.as_str().trim();
    RuneOp::from_str(text).map_err(|_| ParseError::InvalidOperator(text.to_string()))
}

File: tui\keybindings.rs
========================
//! Keyboard shortcuts and action mapping.

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// Actions that can be triggered by keyboard shortcuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Quit,
    ToggleMode,
    SwitchPanel,
    OpenFile,
    SaveFile,
    Refresh,
    ToggleSettings,
    ToggleHelp,
    ToggleFileBrowser,
    ToggleHistory,
    ToggleDiff,
    ToggleTheme,
    CopyOutput,
    CopySelection,
    PasteInput,
    ClearInput,
    NewFile,
    RoundTrip,
    OpenRepl,
    None,
}

pub struct KeyBindings;

impl KeyBindings {
    /// Map key event to action.
    pub fn handle(key: KeyEvent) -> Action {
        match (key.code, key.modifiers) {
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => Action::Quit,
            (KeyCode::Char('q'), KeyModifiers::CONTROL) => Action::Quit,
            (KeyCode::Char('e'), KeyModifiers::CONTROL) => Action::ToggleMode,
            (KeyCode::Char('m'), KeyModifiers::CONTROL) => Action::ToggleMode,
            (KeyCode::Tab, KeyModifiers::NONE) => Action::SwitchPanel,
            (KeyCode::Char('o'), KeyModifiers::CONTROL) => Action::OpenFile,
            (KeyCode::Char('s'), KeyModifiers::CONTROL) => Action::SaveFile,
            (KeyCode::Char('n'), KeyModifiers::CONTROL) => Action::NewFile,
            (KeyCode::Char('p'), KeyModifiers::CONTROL) => Action::ToggleSettings,
            (KeyCode::F(1), KeyModifiers::NONE) => Action::ToggleHelp,
            (KeyCode::Char('f'), KeyModifiers::CONTROL) => Action::ToggleFileBrowser,
            (KeyCode::Char('h'), KeyModifiers::CONTROL) => Action::ToggleHistory,
            (KeyCode::Char('d'), KeyModifiers::CONTROL) => Action::ToggleDiff,
            (KeyCode::Char('t'), KeyModifiers::CONTROL) => Action::ToggleTheme,
            (KeyCode::Char('y'), KeyModifiers::CONTROL) => Action::CopyOutput,
            (KeyCode::Char('k'), KeyModifiers::CONTROL) => Action::CopySelection,
            (KeyCode::Char('v'), KeyModifiers::CONTROL) => Action::PasteInput,
            (KeyCode::Char('b'), KeyModifiers::CONTROL) => Action::RoundTrip,
            (KeyCode::Char('r'), KeyModifiers::CONTROL) => Action::OpenRepl,
            (KeyCode::Char('l'), KeyModifiers::CONTROL) => Action::ClearInput,

            _ => Action::None,
        }
    }

    /// Get list of shortcuts for help display.
    pub fn shortcuts() -> Vec<(&'static str, &'static str)> {
        vec![
            ("Ctrl+C/Q", "Quit"),
            ("Ctrl+E/M", "Toggle Mode"),
            ("Tab", "Switch Panel"),
            ("Ctrl+R", "Open REPL"),
            ("Ctrl+O", "Open File"),
            ("Ctrl+S", "Save File"),
            ("Ctrl+N", "New File"),
            ("Ctrl+P", "Settings"),
            ("F1", "Help"),
            ("Ctrl+F", "File Browser"),
            ("Ctrl+H", "History"),
            ("Ctrl+D", "Diff View"),
            ("Ctrl+T", "Toggle Theme"),
            ("Ctrl+Y", "Copy All Output"),
            ("Ctrl+K", "Copy Selection"),
            ("Ctrl+V", "Paste Input"),
            ("Ctrl+B", "Round Trip Test"),
            ("Ctrl+L", "Clear Input"),
        ]
    }
}

File: tui\app.rs
================
use std::{fs, path::PathBuf, time::Duration};

use anyhow::{Context, Result};
use chrono::Local;
use ratatui::crossterm::event::{KeyCode, KeyEvent};
use tiktoken_rs::cl100k_base;

use crate::{
    decode, encode,
    tui::{
        components::FileBrowser,
        events::{Event, EventHandler},
        keybindings::{Action, KeyBindings},
        repl_command::ReplCommand,
        state::{AppState, ConversionHistory, app_state::ConversionStats},
        ui,
    },
};

/// Main TUI application managing state, events, and rendering.
pub struct TuiApp<'a> {
    pub app_state: AppState<'a>,
    pub file_browser: FileBrowser,
}

impl<'a> TuiApp<'a> {
    pub fn new() -> Self {
        Self {
            app_state: AppState::new(),
            file_browser: FileBrowser::new(),
        }
    }

    pub fn run<B: ratatui::backend::Backend>(
        &mut self,
        terminal: &mut ratatui::Terminal<B>,
    ) -> Result<()> {
        loop {
            terminal.draw(|f| ui::render(f, &mut self.app_state, &mut self.file_browser))?;

            if let Some(event) = EventHandler::poll(Duration::from_millis(100))? {
                self.handle_event(event)?;
            }

            if self.app_state.should_quit {
                break;
            }
        }
        Ok(())
    }

    fn handle_event(&mut self, event: Event) -> Result<()> {
        match event {
            Event::Key(key) => self.handle_key_event(key)?,
            Event::Resize => {}
            Event::Tick => {}
        }
        Ok(())
    }

    fn handle_key_event(&mut self, key: KeyEvent) -> Result<()> {
        // Confirmation dialog takes highest priority
        if self.app_state.show_confirmation {
            return self.handle_confirmation_key(key);
        }

        // REPL takes priority when active
        if self.app_state.repl.active {
            return self.handle_repl_key(key);
        }

        // Handle overlay panels (help, file browser, settings, etc.)
        if self.app_state.show_help
            || self.app_state.show_file_browser
            || self.app_state.show_history
            || self.app_state.show_diff
            || self.app_state.show_settings
        {
            match key.code {
                KeyCode::Esc => {
                    self.app_state.show_help = false;
                    self.app_state.show_file_browser = false;
                    self.app_state.show_history = false;
                    self.app_state.show_diff = false;
                    self.app_state.show_settings = false;
                    return Ok(());
                }
                KeyCode::F(1) if self.app_state.show_help => {
                    self.app_state.show_help = false;
                    return Ok(());
                }
                _ => {}
            }

            if self.app_state.show_file_browser {
                match key.code {
                    KeyCode::Up => {
                        self.file_browser.move_up();
                        return Ok(());
                    }
                    KeyCode::Down => {
                        let count = self
                            .file_browser
                            .get_entry_count(&self.app_state.file_state.current_dir);
                        self.file_browser.move_down(count);
                        return Ok(());
                    }
                    KeyCode::Enter => {
                        self.handle_file_selection()?;
                        return Ok(());
                    }
                    KeyCode::Char(' ') => {
                        self.handle_file_toggle_selection()?;
                        return Ok(());
                    }
                    _ => {}
                }
            }

            if self.app_state.show_settings {
                match key.code {
                    KeyCode::Esc => {
                        self.app_state.show_settings = false;
                        return Ok(());
                    }
                    KeyCode::Char('d') => {
                        self.app_state.cycle_delimiter();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        self.app_state.increase_indent();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('-') | KeyCode::Char('_') => {
                        self.app_state.decrease_indent();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('f') => {
                        self.app_state.toggle_fold_keys();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('p') => {
                        self.app_state.toggle_expand_paths();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('s') => {
                        self.app_state.toggle_strict();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('c') => {
                        self.app_state.toggle_coerce_types();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('[') | KeyCode::Char('{') => {
                        self.app_state.decrease_flatten_depth();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char(']') | KeyCode::Char('}') => {
                        self.app_state.increase_flatten_depth();
                        self.perform_conversion();
                        return Ok(());
                    }
                    KeyCode::Char('u') => {
                        self.app_state.toggle_flatten_depth();
                        self.perform_conversion();
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }

        let action = KeyBindings::handle(key);
        match action {
            Action::Quit => self.app_state.quit(),
            Action::ToggleMode => {
                self.app_state.toggle_mode();
                self.perform_conversion();
            }
            Action::SwitchPanel => {
                self.app_state.editor.toggle_active();
            }
            Action::OpenFile => {
                self.open_file_dialog()?;
            }
            Action::SaveFile => {
                self.save_output()?;
            }
            Action::NewFile => {
                self.new_file();
            }
            Action::Refresh => {
                self.perform_conversion();
            }
            Action::ToggleSettings => {
                self.app_state.toggle_settings();
            }
            Action::ToggleHelp => {
                self.app_state.toggle_help();
            }
            Action::ToggleFileBrowser => {
                self.app_state.toggle_file_browser();
            }
            Action::ToggleHistory => {
                self.app_state.toggle_history();
            }
            Action::ToggleDiff => {
                self.app_state.toggle_diff();
            }
            Action::ToggleTheme => {
                self.app_state.toggle_theme();
            }
            Action::CopyOutput => {
                self.copy_to_clipboard()?;
            }
            Action::OpenRepl => {
                self.app_state.repl.activate();
            }
            Action::CopySelection => {
                self.copy_selection_to_clipboard()?;
            }
            Action::PasteInput => {
                self.paste_from_clipboard()?;
            }
            Action::RoundTrip => {
                self.perform_round_trip()?;
            }
            Action::ClearInput => {
                self.app_state.editor.clear_input();
                self.app_state.editor.clear_output();
                self.app_state.stats = None;
            }
            Action::None => {
                if self.app_state.editor.is_input_active() {
                    self.app_state.editor.input.input(key);
                    self.app_state.file_state.mark_modified();
                    self.perform_conversion();
                } else if self.app_state.editor.is_output_active() {
                    // Output is read-only, only allow navigation
                    match key.code {
                        KeyCode::Up
                        | KeyCode::Down
                        | KeyCode::Left
                        | KeyCode::Right
                        | KeyCode::PageUp
                        | KeyCode::PageDown
                        | KeyCode::Home
                        | KeyCode::End => {
                            self.app_state.editor.output.input(key);
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }

    /// Convert input based on current mode (encode/decode).
    fn perform_conversion(&mut self) {
        let input = self.app_state.editor.get_input();
        if input.trim().is_empty() {
            self.app_state.editor.clear_output();
            self.app_state.stats = None;
            self.app_state.clear_error();
            return;
        }

        self.app_state.clear_error();

        match self.app_state.mode {
            crate::tui::state::app_state::Mode::Encode => {
                self.encode_input(&input);
            }
            crate::tui::state::app_state::Mode::Decode => {
                self.decode_input(&input);
            }
            crate::tui::state::app_state::Mode::Rune => {
                self.parse_rune_input(&input);
            }
        }
    }

    fn encode_input(&mut self, input: &str) {
        self.app_state.editor.clear_output();

        match serde_json::from_str::<serde_json::Value>(input) {
            Ok(json_value) => match encode(&json_value, &self.app_state.encode_options) {
                Ok(toon_str) => {
                    self.app_state.editor.set_output(toon_str.clone());
                    self.app_state.clear_error();

                    if let Ok(bpe) = cl100k_base() {
                        let json_tokens = bpe.encode_with_special_tokens(input).len();
                        let toon_tokens = bpe.encode_with_special_tokens(&toon_str).len();
                        let json_bytes = input.len();
                        let toon_bytes = toon_str.len();

                        let token_savings =
                            100.0 * (1.0 - (toon_tokens as f64 / json_tokens as f64));
                        let byte_savings = 100.0 * (1.0 - (toon_bytes as f64 / json_bytes as f64));

                        self.app_state.stats = Some(ConversionStats {
                            json_tokens,
                            toon_tokens,
                            json_bytes,
                            toon_bytes,
                            token_savings,
                            byte_savings,
                        });

                        self.app_state.file_state.add_to_history(ConversionHistory {
                            timestamp: Local::now(),
                            mode: "Encode".to_string(),
                            input_file: self.app_state.file_state.current_file.clone(),
                            output_file: None,
                            token_savings,
                            byte_savings,
                        });
                    }
                }
                Err(e) => {
                    self.app_state.set_error(format!("Encode error: {e}"));
                }
            },
            Err(e) => {
                self.app_state.set_error(format!("Invalid JSON: {e}"));
            }
        }
    }

    fn decode_input(&mut self, input: &str) {
        self.app_state.editor.clear_output();

        match decode::<serde_json::Value>(input, &self.app_state.decode_options) {
            Ok(json_value) => match serde_json::to_string_pretty(&json_value) {
                Ok(json_str) => {
                    self.app_state.editor.set_output(json_str.clone());
                    self.app_state.clear_error();

                    if let Ok(bpe) = cl100k_base() {
                        let toon_tokens = bpe.encode_with_special_tokens(input).len();
                        let json_tokens = bpe.encode_with_special_tokens(&json_str).len();
                        let toon_bytes = input.len();
                        let json_bytes = json_str.len();

                        let token_savings =
                            100.0 * (1.0 - (toon_tokens as f64 / json_tokens as f64));
                        let byte_savings = 100.0 * (1.0 - (toon_bytes as f64 / json_bytes as f64));

                        self.app_state.stats = Some(ConversionStats {
                            json_tokens,
                            toon_tokens,
                            json_bytes,
                            toon_bytes,
                            token_savings,
                            byte_savings,
                        });

                        self.app_state.file_state.add_to_history(ConversionHistory {
                            timestamp: Local::now(),
                            mode: "Decode".to_string(),
                            input_file: self.app_state.file_state.current_file.clone(),
                            output_file: None,
                            token_savings,
                            byte_savings,
                        });
                    }
                }
                Err(e) => {
                    self.app_state
                        .set_error(format!("JSON serialization error: {e}"));
                }
            },
            Err(e) => {
                self.app_state.set_error(format!("Decode error: {e}"));
            }
        }
    }

    fn open_file_dialog(&mut self) -> Result<()> {
        self.app_state.toggle_file_browser();
        Ok(())
    }

    fn parse_rune_input(&mut self, input: &str) {
        self.app_state.editor.clear_output();

        match crate::rune::parse_rune(input) {
            Ok(statements) => {
                let mut output = String::new();

                for stmt in &statements {
                    match stmt {
                        crate::rune::Stmt::RootDecl(root) => {
                            output.push_str(&format!("Root: {}\n", root));
                        }
                        crate::rune::Stmt::ToonBlock { name, content } => {
                            output.push_str(&format!("TOON Block '{}':\n", name));
                            // Try to decode as TOON and show JSON
                            match crate::decode::decode::<serde_json::Value>(
                                content,
                                &self.app_state.decode_options,
                            ) {
                                Ok(json) => {
                                    if let Ok(pretty) = serde_json::to_string_pretty(&json) {
                                        output.push_str(&format!("Decoded JSON:\n{}\n", pretty));
                                    }
                                }
                                Err(_) => {
                                    output.push_str(&format!("Raw TOON:\n{}\n", content));
                                }
                            }
                        }
                        crate::rune::Stmt::Expr(expr) => {
                            output.push_str(&format!("Expression: {}\n", expr));
                        }
                    }
                }

                self.app_state.editor.set_output(output);
                self.app_state.clear_error();

                // Count statements as "complexity"
                self.app_state.stats = Some(ConversionStats {
                    json_tokens: statements.len(),
                    toon_tokens: 0,
                    json_bytes: input.len(),
                    toon_bytes: 0,
                    token_savings: 0.0,
                    byte_savings: 0.0,
                });
            }
            Err(e) => {
                self.app_state.set_error(format!("RUNE parse error: {e}"));
            }
        }
    }

    fn save_output(&mut self) -> Result<()> {
        let output = self.app_state.editor.get_output();
        if output.trim().is_empty() {
            self.app_state.set_error("Nothing to save".to_string());
            return Ok(());
        }

        let extension = match self.app_state.mode {
            crate::tui::state::app_state::Mode::Encode => "toon",
            crate::tui::state::app_state::Mode::Decode => "json",
            crate::tui::state::app_state::Mode::Rune => "rune",
        };

        let path = if let Some(current) = &self.app_state.file_state.current_file {
            current.with_extension(extension)
        } else {
            PathBuf::from(format!("output.{extension}"))
        };

        fs::write(&path, output).context("Failed to save file")?;
        self.app_state
            .set_status(format!("Saved to {}", path.display()));
        self.app_state.file_state.is_modified = false;

        Ok(())
    }

    fn new_file(&mut self) {
        if self.app_state.file_state.is_modified {
            self.app_state.show_confirmation = true;
            self.app_state.confirmation_action =
                crate::tui::state::app_state::ConfirmationAction::NewFile;
            return;
        }
        self.app_state.editor.clear_input();
        self.app_state.editor.clear_output();
        self.app_state.file_state.clear_current_file();
        self.app_state.stats = None;
        self.app_state.set_status("New file created".to_string());
    }

    fn copy_to_clipboard(&mut self) -> Result<()> {
        let output = self.app_state.editor.get_output();
        if output.trim().is_empty() {
            self.app_state.set_error("Nothing to copy".to_string());
            return Ok(());
        }

        #[cfg(not(target_os = "unknown"))]
        {
            use arboard::Clipboard;
            let mut clipboard = Clipboard::new()?;
            clipboard.set_text(output)?;
            self.app_state.set_status("Copied to clipboard".to_string());
        }

        #[cfg(target_os = "unknown")]
        {
            self.app_state
                .set_error("Clipboard not supported on this platform".to_string());
        }

        Ok(())
    }

    fn paste_from_clipboard(&mut self) -> Result<()> {
        #[cfg(not(target_os = "unknown"))]
        {
            use arboard::Clipboard;
            let mut clipboard = Clipboard::new()?;
            let text = clipboard.get_text()?;
            self.app_state.editor.set_input(text);
            self.app_state.file_state.mark_modified();
            self.perform_conversion();
            self.app_state
                .set_status("Pasted from clipboard".to_string());
        }

        #[cfg(target_os = "unknown")]
        {
            self.app_state
                .set_error("Clipboard not supported on this platform".to_string());
        }

        Ok(())
    }

    fn handle_confirmation_key(&mut self, key: KeyEvent) -> Result<()> {
        use crate::tui::state::app_state::ConfirmationAction;

        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                // User confirmed - perform action
                match self.app_state.confirmation_action {
                    ConfirmationAction::NewFile => {
                        self.app_state.editor.clear_input();
                        self.app_state.editor.clear_output();
                        self.app_state.file_state.clear_current_file();
                        self.app_state.set_status("New file created".to_string());
                    }
                    ConfirmationAction::Quit => {
                        self.app_state.should_quit = true;
                    }
                    ConfirmationAction::DeleteFile => {
                        if let Some(current_file) = &self.app_state.file_state.current_file {
                            if let Err(e) = std::fs::remove_file(current_file) {
                                self.app_state.set_error(format!("Delete failed: {e}"));
                            } else {
                                self.app_state.set_status("File deleted".to_string());
                            }
                        }
                    }
                    ConfirmationAction::None => {}
                }
                self.app_state.show_confirmation = false;
                self.app_state.confirmation_action = ConfirmationAction::None;
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                // User cancelled
                self.app_state.show_confirmation = false;
                self.app_state.confirmation_action = ConfirmationAction::None;
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_file_selection(&mut self) -> Result<()> {
        let current_dir = self.app_state.file_state.current_dir.clone();
        if let Some(selected_path) = self.file_browser.get_selected_entry(&current_dir) {
            if selected_path.is_dir() {
                // Navigate into directory
                self.app_state.file_state.current_dir = selected_path;
                self.file_browser.selected_index = 0;
                self.app_state.set_status(format!(
                    "Navigated to {}",
                    self.app_state.file_state.current_dir.display()
                ));
            } else if selected_path.is_file() {
                // Open file
                match fs::read_to_string(&selected_path) {
                    Ok(content) => {
                        self.app_state.editor.set_input(content);
                        self.app_state
                            .file_state
                            .set_current_file(selected_path.clone());

                        // Auto-detect mode based on extension
                        if let Some(ext) = selected_path.extension().and_then(|e| e.to_str()) {
                            match ext {
                                "json" => {
                                    self.app_state.mode =
                                        crate::tui::state::app_state::Mode::Encode;
                                }
                                "toon" => {
                                    self.app_state.mode =
                                        crate::tui::state::app_state::Mode::Decode;
                                }
                                "rune" => {
                                    self.app_state.mode = crate::tui::state::app_state::Mode::Rune;
                                }
                                _ => {}
                            }
                        }

                        self.perform_conversion();
                        self.app_state.show_file_browser = false;
                        self.app_state
                            .set_status(format!("Opened {}", selected_path.display()));
                    }
                    Err(e) => {
                        self.app_state
                            .set_error(format!("Failed to read file: {e}"));
                    }
                }
            }
        }
        Ok(())
    }

    fn handle_file_toggle_selection(&mut self) -> Result<()> {
        let current_dir = self.app_state.file_state.current_dir.clone();
        if let Some(selected_path) = self.file_browser.get_selected_entry(&current_dir) {
            if selected_path.is_file() {
                self.app_state
                    .file_state
                    .toggle_file_selection(selected_path.clone());
                let is_selected = self.app_state.file_state.is_selected(&selected_path);
                let action = if is_selected {
                    "Selected"
                } else {
                    "Deselected"
                };
                self.app_state
                    .set_status(format!("{} {}", action, selected_path.display()));
            }
        }
        Ok(())
    }

    fn copy_selection_to_clipboard(&mut self) -> Result<()> {
        let text = if self.app_state.editor.is_input_active() {
            self.app_state.editor.input.yank_text()
        } else {
            self.app_state.editor.output.yank_text()
        };

        if text.is_empty() {
            self.app_state.set_error("Nothing to copy".to_string());
            return Ok(());
        }

        #[cfg(not(target_os = "unknown"))]
        {
            use arboard::Clipboard;
            let mut clipboard = Clipboard::new()?;
            clipboard.set_text(text)?;
            self.app_state
                .set_status("Copied selection to clipboard".to_string());
        }

        #[cfg(target_os = "unknown")]
        {
            self.app_state
                .set_error("Clipboard not supported on this platform".to_string());
        }

        Ok(())
    }

    /// Round-trip test: convert output back to input and verify.
    fn perform_round_trip(&mut self) -> Result<()> {
        let output = self.app_state.editor.get_output();
        if output.trim().is_empty() {
            self.app_state
                .set_error("No output to round-trip test. Convert something first!".to_string());
            return Ok(());
        }

        let original_input = self.app_state.editor.get_input();
        self.app_state.editor.set_input(output.clone());
        self.app_state.toggle_mode();
        self.perform_conversion();

        let roundtrip_output = self.app_state.editor.get_output();

        if roundtrip_output.trim().is_empty() {
            self.app_state.set_error(
                "Round-trip failed! Conversion produced no output. Check for errors.".to_string(),
            );
            return Ok(());
        }

        let matches = self.compare_data(&original_input, &roundtrip_output);

        if matches {
            self.app_state
                .set_status("✓ Round-trip successful! Output matches original.".to_string());
        } else {
            self.app_state.set_error(format!(
                "⚠ Round-trip mismatch! Original had {} chars, round-trip has {} chars.",
                original_input.len(),
                roundtrip_output.len()
            ));
        }

        Ok(())
    }

    /// Compare data semantically, trying JSON parse first.
    fn compare_data(&self, original: &str, roundtrip: &str) -> bool {
        // Try JSON comparison for accuracy
        if let (Ok(orig_json), Ok(rt_json)) = (
            serde_json::from_str::<serde_json::Value>(original),
            serde_json::from_str::<serde_json::Value>(roundtrip),
        ) {
            return orig_json == rt_json;
        }

        let original_normalized: String = original.split_whitespace().collect();
        let roundtrip_normalized: String = roundtrip.split_whitespace().collect();
        original_normalized == roundtrip_normalized
    }

    /// Handle keyboard input when REPL is active.
    fn handle_repl_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Esc => {
                self.app_state.repl.deactivate();
            }
            KeyCode::Char('r')
                if key
                    .modifiers
                    .contains(ratatui::crossterm::event::KeyModifiers::CONTROL) =>
            {
                self.app_state.repl.deactivate();
            }
            KeyCode::Enter => {
                let cmd_input = self.app_state.repl.input.clone();
                if !cmd_input.trim().is_empty() {
                    self.app_state.repl.add_prompt(&cmd_input);
                    self.app_state.repl.add_to_history(cmd_input.clone());

                    if let Err(e) = self.execute_repl_command(&cmd_input) {
                        self.app_state.repl.add_error(format!("{e}"));
                    }

                    self.app_state.repl.input.clear();
                    self.app_state.repl.scroll_to_bottom();
                }
            }
            KeyCode::Up => {
                self.app_state.repl.history_up();
            }
            KeyCode::Down => {
                self.app_state.repl.history_down();
            }
            KeyCode::PageUp => {
                self.app_state.repl.scroll_up();
            }
            KeyCode::PageDown => {
                self.app_state.repl.scroll_down(20);
            }
            KeyCode::Char(c) => {
                self.app_state.repl.input.push(c);
            }
            KeyCode::Backspace => {
                self.app_state.repl.input.pop();
            }
            _ => {}
        }
        Ok(())
    }

    /// Execute parsed REPL command and update state.
    fn execute_repl_command(&mut self, input: &str) -> Result<()> {
        let cmd = ReplCommand::parse(input)?;

        match cmd.name.as_str() {
            "encode" | "e" => {
                let mut data = cmd
                    .inline_data
                    .as_ref()
                    .map(|s| s.to_string())
                    .unwrap_or_else(String::new);

                data = self.substitute_variables(&data);

                if data.is_empty() {
                    self.app_state
                        .repl
                        .add_error("Usage: encode {\"data\": true} or encode $var".to_string());
                    return Ok(());
                }

                match serde_json::from_str::<serde_json::Value>(&data) {
                    Ok(json_value) => match encode(&json_value, &self.app_state.encode_options) {
                        Ok(toon_str) => {
                            self.app_state.repl.add_success(toon_str.clone());
                            self.app_state.repl.last_result = Some(toon_str);
                        }
                        Err(e) => {
                            self.app_state.repl.add_error(format!("Encode error: {e}"));
                        }
                    },
                    Err(e) => {
                        self.app_state.repl.add_error(format!("Invalid JSON: {e}"));
                    }
                }
            }
            "decode" | "d" => {
                let mut data = cmd
                    .inline_data
                    .as_ref()
                    .map(|s| s.to_string())
                    .unwrap_or_else(String::new);

                data = self.substitute_variables(&data);

                if data.is_empty() {
                    self.app_state
                        .repl
                        .add_error("Usage: decode name: Alice or decode $var".to_string());
                    return Ok(());
                }

                match decode::<serde_json::Value>(&data, &self.app_state.decode_options) {
                    Ok(json_value) => match serde_json::to_string_pretty(&json_value) {
                        Ok(json_str) => {
                            self.app_state.repl.add_success(json_str.clone());
                            self.app_state.repl.last_result = Some(json_str);
                        }
                        Err(e) => {
                            self.app_state.repl.add_error(format!("JSON error: {e}"));
                        }
                    },
                    Err(e) => {
                        self.app_state.repl.add_error(format!("Decode error: {e}"));
                    }
                }
            }
            "rune" | "r" => {
                let mut data = cmd
                    .inline_data
                    .as_ref()
                    .map(|s| s.to_string())
                    .unwrap_or_else(String::new);

                data = self.substitute_variables(&data);

                if data.is_empty() {
                    self.app_state.repl.add_error(
                        "Usage: rune root: example\ndata ~TOON:\n  items: value\n\nor rune $var"
                            .to_string(),
                    );
                    return Ok(());
                }

                match crate::rune::parse_rune(&data) {
                    Ok(statements) => {
                        let mut output = String::new();

                        for stmt in &statements {
                            match stmt {
                                crate::rune::Stmt::RootDecl(root) => {
                                    output.push_str(&format!("✓ Root: {}\n", root));
                                }
                                crate::rune::Stmt::ToonBlock { name, content } => {
                                    output.push_str(&format!(
                                        "✓ TOON Block '{}': {} chars\n",
                                        name,
                                        content.len()
                                    ));
                                    // Try to show decoded result briefly
                                    if let Ok(json) = crate::decode::decode::<serde_json::Value>(
                                        content,
                                        &self.app_state.decode_options,
                                    ) {
                                        if let Ok(json_str) = serde_json::to_string(&json) {
                                            if json_str.len() < 200 {
                                                output.push_str(&format!("  {}", json_str));
                                            } else {
                                                output.push_str(&format!(
                                                    "  ({} items)",
                                                    if json.is_array() {
                                                        json.as_array().unwrap().len()
                                                    } else if json.is_object() {
                                                        json.as_object().unwrap().len()
                                                    } else {
                                                        1
                                                    }
                                                ));
                                            }
                                            output.push('\n');
                                        }
                                    }
                                }
                                crate::rune::Stmt::Expr(expr) => {
                                    output.push_str(&format!("✓ Expression: {}\n", expr));
                                }
                            }
                        }
                        self.app_state.repl.add_success(output);
                        self.app_state.repl.last_result = Some(statements.len().to_string());
                    }
                    Err(e) => {
                        self.app_state
                            .repl
                            .add_error(format!("RUNE parse error: {e}"));
                    }
                }
            }
            "let" => {
                let parts: Vec<&str> = input.splitn(2, '=').collect();
                if parts.len() == 2 {
                    let var_part = parts[0].trim().trim_start_matches("let").trim();
                    let data_part = parts[1].trim();

                    if !var_part.is_empty() && !data_part.is_empty() {
                        let var_name = var_part.trim_start_matches('$');
                        self.app_state
                            .repl
                            .variables
                            .insert(var_name.to_string(), data_part.to_string());
                        self.app_state
                            .repl
                            .add_info(format!("Stored in ${var_name}"));
                        self.app_state.repl.last_result = Some(data_part.to_string());
                    } else {
                        self.app_state
                            .repl
                            .add_error("Usage: let $var = {\"data\": true}".to_string());
                    }
                } else {
                    self.app_state
                        .repl
                        .add_error("Usage: let $var = {\"data\": true}".to_string());
                }
            }
            "vars" => {
                if self.app_state.repl.variables.is_empty() {
                    self.app_state
                        .repl
                        .add_info("No variables defined".to_string());
                } else {
                    let vars: Vec<String> = self
                        .app_state
                        .repl
                        .variables
                        .keys()
                        .map(|k| format!("${k}"))
                        .collect();
                    for var in vars {
                        self.app_state.repl.add_info(var);
                    }
                }
            }
            "clear" => {
                self.app_state.repl.output.clear();
                self.app_state
                    .repl
                    .output
                    .push(crate::tui::state::ReplLine {
                        kind: crate::tui::state::ReplLineKind::Info,
                        content: "Cleared".to_string(),
                    });
            }
            "help" | "h" => {
                self.app_state
                    .repl
                    .add_info("📖 REPL Commands:".to_string());
                self.app_state.repl.add_info("".to_string());
                self.app_state
                    .repl
                    .add_info("  encode {\"data\": true}  - Encode JSON to TOON".to_string());
                self.app_state
                    .repl
                    .add_info("  decode name: Alice      - Decode TOON to JSON".to_string());
                self.app_state
                    .repl
                    .add_info("  rune root: continuum   - Parse and evaluate RUNE".to_string());
                self.app_state
                    .repl
                    .add_info("  let $var = {...}        - Store data in variable".to_string());
                self.app_state
                    .repl
                    .add_info("  vars                    - List all variables".to_string());
                self.app_state
                    .repl
                    .add_info("  clear                   - Clear session".to_string());
                self.app_state
                    .repl
                    .add_info("  help                    - Show this help".to_string());
                self.app_state
                    .repl
                    .add_info("  exit                    - Close REPL".to_string());
                self.app_state.repl.add_info("".to_string());
                self.app_state
                    .repl
                    .add_info("Press ↑/↓ for history, Esc to close".to_string());
            }
            "exit" | "quit" | "q" => {
                self.app_state.repl.add_info("Closing REPL...".to_string());
                self.app_state.repl.deactivate();
            }
            _ => {
                self.app_state
                    .repl
                    .add_error(format!("Unknown command: {}. Type 'help'", cmd.name));
            }
        }

        Ok(())
    }

    /// Replace $var and $_ with their stored values.
    fn substitute_variables(&self, text: &str) -> String {
        let mut result = text.to_string();

        // $_ is the last result
        if let Some(last) = &self.app_state.repl.last_result {
            result = result.replace("$_", last);
        }

        // Variables are stored without $, add it for matching
        for (var_name, var_value) in &self.app_state.repl.variables {
            let pattern = format!("${var_name}");
            result = result.replace(&pattern, var_value);
        }

        result
    }
}

impl<'a> Default for TuiApp<'a> {
    fn default() -> Self {
        Self::new()
    }
}

File: tui\events.rs
===================
//! Event handling for terminal input.

use std::time::Duration;

use ratatui::crossterm::event::{self, Event as CrosstermEvent, KeyEvent};

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

File: tui\repl_command.rs
=========================
//! REPL command parser with inline data support

use anyhow::{Result, bail};

/// Parsed REPL command with inline data
#[derive(Debug, Clone)]
pub struct ReplCommand {
    pub name: String,
    pub inline_data: Option<String>,
    pub args: Vec<String>,
}

impl ReplCommand {
    /// Parse command input, extracting inline data if present.
    ///
    /// Handles patterns like:
    /// - `encode {"data": true}` - JSON inline
    /// - `decode name: Alice` - TOON inline
    /// - `encode $var` - Variable reference
    pub fn parse(input: &str) -> Result<Self> {
        let input = input.trim();
        if input.is_empty() {
            bail!("Empty command");
        }

        let parts: Vec<&str> = input.splitn(2, ' ').collect();
        let cmd_name = parts[0].to_string();

        let (inline_data, remaining_args) = if parts.len() > 1 {
            let rest = parts[1].trim();

            // Check if input looks like data rather than flags/args
            if rest.starts_with('{')
                || rest.starts_with('"')
                || rest.starts_with('$')
                || rest.contains(':')
            {
                let data_end = if rest.starts_with('{') {
                    find_matching_brace(rest) // Handle nested braces
                } else if rest.starts_with('$') {
                    rest.find(' ').unwrap_or(rest.len()) // Variable name
                } else {
                    rest.find(" --").unwrap_or(rest.len()) // Until flag or end
                };

                let data = rest[..data_end].trim().to_string();
                let remaining = rest[data_end..].trim();

                (
                    Some(data),
                    if remaining.is_empty() {
                        vec![]
                    } else {
                        remaining
                            .split_whitespace()
                            .map(|s| s.to_string())
                            .collect()
                    },
                )
            } else {
                (
                    None,
                    rest.split_whitespace().map(|s| s.to_string()).collect(),
                )
            }
        } else {
            (None, vec![])
        };

        Ok(ReplCommand {
            name: cmd_name,
            inline_data,
            args: remaining_args,
        })
    }

    pub fn has_flag(&self, flag: &str) -> bool {
        self.args.iter().any(|a| a == flag)
    }

    pub fn get_option(&self, option: &str) -> Option<&str> {
        self.args
            .iter()
            .position(|a| a == option)
            .and_then(|i| self.args.get(i + 1))
            .map(|s| s.as_str())
    }
}

fn find_matching_brace(s: &str) -> usize {
    let mut depth = 0;
    for (i, ch) in s.chars().enumerate() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
    }
    s.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inline_json() {
        let cmd = ReplCommand::parse(r#"encode {"test": true}"#).unwrap();
        assert_eq!(cmd.name, "encode");
        assert_eq!(cmd.inline_data, Some(r#"{"test": true}"#.to_string()));
    }

    #[test]
    fn test_inline_toon() {
        let cmd = ReplCommand::parse("decode name: Alice").unwrap();
        assert_eq!(cmd.name, "decode");
        assert_eq!(cmd.inline_data, Some("name: Alice".to_string()));
    }

    #[test]
    fn test_with_flags() {
        let cmd = ReplCommand::parse(r#"encode {"test": true} --fold-keys"#).unwrap();
        assert_eq!(cmd.name, "encode");
        assert!(cmd.inline_data.is_some());
        assert!(cmd.has_flag("--fold-keys"));
    }
}

File: tui\mod.rs
================
//! Terminal User Interface for TOON format conversion.
//!
//! Provides an interactive TUI with real-time conversion, REPL, and settings
//! panels.

pub mod app;
pub mod components;
pub mod events;
pub mod keybindings;
pub mod repl_command;
pub mod state;
pub mod theme;
pub mod ui;

use std::io;

use anyhow::Result;
pub use app::TuiApp;
use crossterm::{
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};

/// Initialize and run the TUI application.
///
/// Sets up terminal in raw mode, runs the app, then restores terminal state.
pub fn run() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = TuiApp::new();
    let res = app.run(&mut terminal);

    // Always restore terminal, even on error
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    res
}

File: tui\theme.rs
==================
//! Color themes for the TUI.

use ratatui::style::{Color, Modifier, Style};

/// Available color themes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Theme {
    #[default]
    Dark,
    Light,
}

impl Theme {
    /// Switch between dark and light themes.
    pub fn toggle(&self) -> Self {
        match self {
            Theme::Dark => Theme::Light,
            Theme::Light => Theme::Dark,
        }
    }

    pub fn background(&self) -> Color {
        match self {
            Theme::Dark => Color::Black,
            Theme::Light => Color::White,
        }
    }

    pub fn foreground(&self) -> Color {
        match self {
            Theme::Dark => Color::White,
            Theme::Light => Color::Black,
        }
    }

    pub fn border(&self) -> Color {
        match self {
            Theme::Dark => Color::Cyan,
            Theme::Light => Color::Blue,
        }
    }

    pub fn border_active(&self) -> Color {
        match self {
            Theme::Dark => Color::Green,
            Theme::Light => Color::Green,
        }
    }

    pub fn title(&self) -> Color {
        match self {
            Theme::Dark => Color::Yellow,
            Theme::Light => Color::Blue,
        }
    }

    pub fn success(&self) -> Color {
        Color::Green
    }

    pub fn error(&self) -> Color {
        Color::Red
    }

    pub fn warning(&self) -> Color {
        Color::Yellow
    }

    pub fn info(&self) -> Color {
        Color::Cyan
    }

    pub fn highlight(&self) -> Color {
        match self {
            Theme::Dark => Color::Blue,
            Theme::Light => Color::LightBlue,
        }
    }

    pub fn selection(&self) -> Color {
        match self {
            Theme::Dark => Color::DarkGray,
            Theme::Light => Color::LightYellow,
        }
    }

    pub fn line_number(&self) -> Color {
        match self {
            Theme::Dark => Color::DarkGray,
            Theme::Light => Color::Gray,
        }
    }

    pub fn normal_style(&self) -> Style {
        Style::default().fg(self.foreground()).bg(self.background())
    }

    /// Get border style, highlighted if active.
    pub fn border_style(&self, active: bool) -> Style {
        Style::default().fg(if active {
            self.border_active()
        } else {
            self.border()
        })
    }

    pub fn title_style(&self) -> Style {
        Style::default()
            .fg(self.title())
            .add_modifier(Modifier::BOLD)
    }

    pub fn highlight_style(&self) -> Style {
        Style::default().fg(self.foreground()).bg(self.highlight())
    }

    pub fn selection_style(&self) -> Style {
        Style::default()
            .fg(self.foreground())
            .bg(self.selection())
            .add_modifier(Modifier::BOLD)
    }

    pub fn error_style(&self) -> Style {
        Style::default()
            .fg(self.error())
            .add_modifier(Modifier::BOLD)
    }

    pub fn success_style(&self) -> Style {
        Style::default()
            .fg(self.success())
            .add_modifier(Modifier::BOLD)
    }

    pub fn warning_style(&self) -> Style {
        Style::default()
            .fg(self.warning())
            .add_modifier(Modifier::BOLD)
    }

    pub fn info_style(&self) -> Style {
        Style::default().fg(self.info())
    }

    pub fn line_number_style(&self) -> Style {
        Style::default().fg(self.line_number())
    }
}

File: tui\ui.rs
===============
use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

use super::{
    components::{
        ConfirmationDialog, DiffViewer, EditorComponent, FileBrowser, HelpScreen, HistoryPanel,
        ReplPanel, SettingsPanel, StatsBar, StatusBar,
    },
    state::AppState,
    theme::Theme,
};
use crate::types::{KeyFoldingMode, PathExpansionMode};

/// Main render function - orchestrates all UI components.
pub fn render(f: &mut Frame, app: &mut AppState, file_browser: &mut FileBrowser) {
    let theme = app.theme;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(4),
            Constraint::Length(3),
        ])
        .split(f.area());

    render_header(f, chunks[0], app);

    // REPL takes full screen (except header)
    if app.repl.active {
        let repl_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(10)])
            .split(f.area())[1];

        ReplPanel::render(f, repl_area, app);
        return;
    } else if app.show_help {
        HelpScreen::render(f, chunks[1], &theme);
    } else if app.show_file_browser {
        file_browser.render(f, chunks[1], app, &theme);
    } else if app.show_history {
        HistoryPanel::render(f, chunks[1], app, &theme);
    } else if app.show_diff {
        DiffViewer::render(f, chunks[1], app, &theme);
    } else if app.show_settings {
        SettingsPanel::render(f, chunks[1], app, &theme);
    } else {
        let editor_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(48),
                Constraint::Percentage(4),
                Constraint::Percentage(48),
            ])
            .split(chunks[1]);

        EditorComponent::render(f, editor_chunks[0], editor_chunks[2], app, &theme);
        render_arrow(f, editor_chunks[1], app, &theme);
    }

    StatsBar::render(f, chunks[2], app, &theme);
    StatusBar::render(f, chunks[3], app, &theme);

    // Render confirmation dialog on top if active
    if app.show_confirmation {
        ConfirmationDialog::render(f, f.area(), app.confirmation_action);
    }
}

/// Render conversion arrow and round-trip button between panels.
fn render_arrow(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let arrow_symbol = match app.mode {
        crate::tui::state::app_state::Mode::Encode => "→",
        crate::tui::state::app_state::Mode::Decode => "←",
        crate::tui::state::app_state::Mode::Rune => "🪄",
    };

    let arrow_text = vec![
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled(arrow_symbol, theme.info_style())),
        Line::from(""),
        Line::from(Span::styled("Ctrl+B", theme.line_number_style())),
        Line::from(Span::styled("Round", theme.line_number_style())),
        Line::from(Span::styled("Trip", theme.line_number_style())),
    ];

    let arrow_para = Paragraph::new(arrow_text).alignment(Alignment::Center);

    f.render_widget(arrow_para, area);
}

/// Render header with title, mode, and current settings.
fn render_header(f: &mut Frame, area: Rect, app: &AppState) {
    let theme = app.theme;

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(34),
            Constraint::Percentage(33),
        ])
        .split(area);

    let title = Paragraph::new(Line::from(vec![
        Span::styled("📋 ", theme.normal_style()),
        Span::styled("TOON", theme.title_style()),
        Span::styled(" Format", theme.info_style()),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    let mode_text = Paragraph::new(Line::from(vec![Span::styled(
        app.mode.as_str(),
        theme.highlight_style(),
    )]))
    .alignment(Alignment::Center)
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(mode_text, chunks[1]);

    // Show relevant settings based on current mode
    let settings_line = match app.mode {
        crate::tui::state::app_state::Mode::Encode => {
            let delimiter = match app.encode_options.delimiter {
                crate::Delimiter::Comma => "comma",
                crate::Delimiter::Tab => "tab",
                crate::Delimiter::Pipe => "pipe",
            };

            let indent = match app.encode_options.indent {
                crate::Indent::Spaces(n) => format!("{n}sp"),
            };

            let mut spans = vec![
                Span::styled("Delim:", theme.line_number_style()),
                Span::styled(format!(" {delimiter}"), theme.info_style()),
                Span::styled(" | Indent:", theme.line_number_style()),
                Span::styled(format!(" {indent}"), theme.info_style()),
            ];

            // Show folding depth only when folding is enabled
            match app.encode_options.key_folding {
                KeyFoldingMode::Off => {}
                KeyFoldingMode::Safe => {
                    spans.push(Span::styled(" | fold:", theme.line_number_style()));
                    spans.push(Span::styled("on", theme.info_style()));

                    // ∞ for unlimited, number for specific depth
                    let depth_str = if app.encode_options.flatten_depth == usize::MAX {
                        "∞".to_string()
                    } else {
                        format!("{}", app.encode_options.flatten_depth)
                    };
                    spans.push(Span::styled(" (", theme.line_number_style()));
                    spans.push(Span::styled(depth_str, theme.info_style()));
                    spans.push(Span::styled(")", theme.line_number_style()));
                }
            }

            spans
        }
        crate::tui::state::app_state::Mode::Decode => {
            let strict = if app.decode_options.strict {
                "on"
            } else {
                "off"
            };
            let coerce = if app.decode_options.coerce_types {
                "on"
            } else {
                "off"
            };
            let expand = match app.decode_options.expand_paths {
                PathExpansionMode::Off => "",
                PathExpansionMode::Safe => " | expand:on",
            };

            vec![
                Span::styled("Strict:", theme.line_number_style()),
                Span::styled(format!(" {strict}"), theme.info_style()),
                Span::styled(" | Coerce:", theme.line_number_style()),
                Span::styled(format!(" {coerce}"), theme.info_style()),
                Span::styled(expand, theme.line_number_style()),
            ]
        }
        crate::tui::state::app_state::Mode::Rune => {
            vec![
                Span::styled("RUNE:", theme.line_number_style()),
                Span::styled(" Geometric", theme.info_style()),
                Span::styled(" | Operators:", theme.line_number_style()),
                Span::styled(" 21", theme.info_style()),
            ]
        }
    };

    let settings = Paragraph::new(Line::from(settings_line))
        .alignment(Alignment::Right)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(settings, chunks[2]);
}

File: types\delimeter.rs
========================
use std::fmt;

use serde::{Deserialize, Serialize};

/// Delimiter character used to separate array elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Delimiter {
    #[default]
    Comma,
    Tab,
    Pipe,
}

impl Delimiter {
    /// Get the character representation of this delimiter.
    pub fn as_char(&self) -> char {
        match self {
            Delimiter::Comma => ',',
            Delimiter::Tab => '\t',
            Delimiter::Pipe => '|',
        }
    }

    /// Get the string representation for metadata (empty for comma, char for
    /// others).
    pub fn as_metadata_str(&self) -> &'static str {
        match self {
            Delimiter::Comma => "",
            Delimiter::Tab => "\t",
            Delimiter::Pipe => "|",
        }
    }

    /// Parse a delimiter from a character.
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            ',' => Some(Delimiter::Comma),
            '\t' => Some(Delimiter::Tab),
            '|' => Some(Delimiter::Pipe),
            _ => None,
        }
    }

    /// Check if the delimiter character appears in the string.
    pub fn contains_in(&self, s: &str) -> bool {
        s.contains(self.as_char())
    }
}

impl fmt::Display for Delimiter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_char())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delimiter_conversion() {
        assert_eq!(Delimiter::Comma.as_char(), ',');
        assert_eq!(Delimiter::Tab.as_char(), '\t');
        assert_eq!(Delimiter::Pipe.as_char(), '|');
    }

    #[test]
    fn test_delimiter_from_char() {
        assert_eq!(Delimiter::from_char(','), Some(Delimiter::Comma));
        assert_eq!(Delimiter::from_char('\t'), Some(Delimiter::Tab));
        assert_eq!(Delimiter::from_char('|'), Some(Delimiter::Pipe));
        assert_eq!(Delimiter::from_char('x'), None);
    }

    #[test]
    fn test_delimiter_contains() {
        assert!(Delimiter::Comma.contains_in("a,b,c"));
        assert!(Delimiter::Tab.contains_in("a\tb\tc"));
        assert!(Delimiter::Pipe.contains_in("a|b|c"));
        assert!(!Delimiter::Comma.contains_in("abc"));
    }
}

File: types\errors.rs
=====================
use thiserror::Error;

/// Result type alias for TOON operations.
pub type ToonResult<T> = std::result::Result<T, ToonError>;

/// Errors that can occur during TOON encoding or decoding.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ToonError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Parse error at line {line}, column {column}: {message}")]
    ParseError {
        line: usize,
        column: usize,
        message: String,
        #[source]
        context: Option<Box<ErrorContext>>,
    },

    #[error("Invalid character '{char}' at position {position}")]
    InvalidCharacter { char: char, position: usize },

    #[error("Unexpected end of input")]
    UnexpectedEof,

    #[error("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    #[error("Invalid delimiter: {0}")]
    InvalidDelimiter(String),

    #[error("Array length mismatch: expected {expected}, found {found}")]
    LengthMismatch {
        expected: usize,
        found: usize,
        #[source]
        context: Option<Box<ErrorContext>>,
    },

    #[error("Invalid structure: {0}")]
    InvalidStructure(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),
}

/// Contextual information for error reporting, including source location
/// and suggestions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorContext {
    pub source_line: String,
    pub preceding_lines: Vec<String>,
    pub following_lines: Vec<String>,
    pub suggestion: Option<String>,
    pub indicator: Option<String>,
}

impl std::fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\nContext:")?;

        for line in &self.preceding_lines {
            writeln!(f, "  {line}")?;
        }

        writeln!(f, "> {}", self.source_line)?;

        if let Some(indicator) = &self.indicator {
            writeln!(f, "  {indicator}")?;
        }

        for line in &self.following_lines {
            writeln!(f, "  {line}")?;
        }

        if let Some(suggestion) = &self.suggestion {
            writeln!(f, "\nSuggestion: {suggestion}")?;
        }

        Ok(())
    }
}

impl std::error::Error for ErrorContext {}

impl ErrorContext {
    /// Create a new error context with a source line.
    pub fn new(source_line: impl Into<String>) -> Self {
        Self {
            source_line: source_line.into(),
            preceding_lines: Vec::new(),
            following_lines: Vec::new(),
            suggestion: None,
            indicator: None,
        }
    }

    /// Add preceding context lines.
    pub fn with_preceding_lines(mut self, lines: Vec<String>) -> Self {
        self.preceding_lines = lines;
        self
    }

    /// Add following context lines.
    pub fn with_following_lines(mut self, lines: Vec<String>) -> Self {
        self.following_lines = lines;
        self
    }

    /// Add a suggestion message to help fix the error.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Add a column indicator (caret) pointing to the error position.
    pub fn with_indicator(mut self, column: usize) -> Self {
        let indicator = format!("{}^", " ".repeat(column));
        self.indicator = Some(indicator);
        self
    }

    /// Create error context from input string with automatic context
    /// extraction.
    pub fn from_input(
        input: &str,
        line: usize,
        column: usize,
        context_lines: usize,
    ) -> Option<Self> {
        let lines: Vec<&str> = input.lines().collect();

        if line == 0 || line > lines.len() {
            return None;
        }

        let line_idx = line - 1;
        let source_line = lines.get(line_idx)?.to_string();

        let start_line = line_idx.saturating_sub(context_lines);
        let end_line = (line_idx + context_lines + 1).min(lines.len());

        let preceding_lines = lines[start_line..line_idx]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let following_lines = lines[(line_idx + 1)..end_line]
            .iter()
            .map(|s| s.to_string())
            .collect();

        Some(Self {
            source_line,
            preceding_lines,
            following_lines,
            suggestion: None,
            indicator: Some(format!("{}^", " ".repeat(column.saturating_sub(1)))),
        })
    }
}

impl ToonError {
    /// Create a parse error at the given position.
    pub fn parse_error(line: usize, column: usize, message: impl Into<String>) -> Self {
        ToonError::ParseError {
            line,
            column,
            message: message.into(),
            context: None,
        }
    }

    /// Create a parse error with additional context information.
    pub fn parse_error_with_context(
        line: usize,
        column: usize,
        message: impl Into<String>,
        context: ErrorContext,
    ) -> Self {
        ToonError::ParseError {
            line,
            column,
            message: message.into(),
            context: Some(Box::new(context)),
        }
    }

    /// Create an error for an invalid character.
    pub fn invalid_char(char: char, position: usize) -> Self {
        ToonError::InvalidCharacter { char, position }
    }

    /// Create an error for a type mismatch.
    pub fn type_mismatch(expected: impl Into<String>, found: impl Into<String>) -> Self {
        ToonError::TypeMismatch {
            expected: expected.into(),
            found: found.into(),
        }
    }

    /// Create an error for array length mismatch.
    pub fn length_mismatch(expected: usize, found: usize) -> Self {
        ToonError::LengthMismatch {
            expected,
            found,
            context: None,
        }
    }

    /// Create an array length mismatch error with context.
    pub fn length_mismatch_with_context(
        expected: usize,
        found: usize,
        context: ErrorContext,
    ) -> Self {
        ToonError::LengthMismatch {
            expected,
            found,
            context: Some(Box::new(context)),
        }
    }

    /// Add context to an error if it supports it.
    pub fn with_context(self, context: ErrorContext) -> Self {
        match self {
            ToonError::ParseError {
                line,
                column,
                message,
                ..
            } => ToonError::ParseError {
                line,
                column,
                message,
                context: Some(Box::new(context)),
            },
            ToonError::LengthMismatch {
                expected, found, ..
            } => ToonError::LengthMismatch {
                expected,
                found,
                context: Some(Box::new(context)),
            },
            other => other,
        }
    }

    /// Add a suggestion to help fix the error.
    pub fn with_suggestion(self, suggestion: impl Into<String>) -> Self {
        let suggestion = suggestion.into();
        match self {
            ToonError::ParseError {
                line,
                column,
                message,
                context,
            } => {
                let new_context = context
                    .map(|c| Box::new(c.with_suggestion(suggestion.clone())))
                    .or_else(|| Some(Box::new(ErrorContext::new("").with_suggestion(suggestion))));
                ToonError::ParseError {
                    line,
                    column,
                    message,
                    context: new_context,
                }
            }
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_creation() {
        let ctx = ErrorContext::new("test line")
            .with_suggestion("Try using quotes")
            .with_indicator(5);

        assert_eq!(ctx.source_line, "test line");
        assert_eq!(ctx.suggestion, Some("Try using quotes".to_string()));
        assert!(ctx.indicator.is_some());
    }

    #[test]
    fn test_error_context_from_input() {
        let input = "line 1\nline 2 with error\nline 3";
        let ctx = ErrorContext::from_input(input, 2, 6, 1);

        assert!(ctx.is_some());
        let ctx = ctx.unwrap();
        assert_eq!(ctx.source_line, "line 2 with error");
        assert_eq!(ctx.preceding_lines, vec!["line 1"]);
        assert_eq!(ctx.following_lines, vec!["line 3"]);
    }

    #[test]
    fn test_parse_error_with_context() {
        let ctx =
            ErrorContext::new("invalid: value").with_suggestion("Did you mean 'value: invalid'?");

        let err = ToonError::parse_error_with_context(1, 8, "Unexpected token", ctx);

        match err {
            ToonError::ParseError {
                line,
                column,
                message,
                context,
            } => {
                assert_eq!(line, 1);
                assert_eq!(column, 8);
                assert_eq!(message, "Unexpected token");
                assert!(context.is_some());
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_error_with_suggestion() {
        let err = ToonError::parse_error(1, 5, "Invalid syntax")
            .with_suggestion("Use quotes around string values");

        match err {
            ToonError::ParseError { context, .. } => {
                assert!(context.is_some());
                let ctx = context.unwrap();
                assert_eq!(
                    ctx.suggestion,
                    Some("Use quotes around string values".to_string())
                );
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_length_mismatch_with_context() {
        let ctx = ErrorContext::new("items[3]: a,b").with_suggestion(
            "Expected 3 items but found 2. Add another item or fix the length marker.",
        );

        let err = ToonError::length_mismatch_with_context(3, 2, ctx);

        match err {
            ToonError::LengthMismatch {
                expected,
                found,
                context,
            } => {
                assert_eq!(expected, 3);
                assert_eq!(found, 2);
                assert!(context.is_some());
            }
            _ => panic!("Wrong error type"),
        }
    }
}

File: types\options.rs
======================
use crate::{
    constants::DEFAULT_INDENT,
    types::{Delimiter, KeyFoldingMode, PathExpansionMode},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Indent {
    Spaces(usize),
}

impl Default for Indent {
    fn default() -> Self {
        Indent::Spaces(DEFAULT_INDENT)
    }
}

impl Indent {
    pub fn get_string(&self, depth: usize) -> String {
        if depth == 0 {
            return String::new();
        }

        match self {
            Indent::Spaces(count) => {
                if *count > 0 {
                    " ".repeat(*count * depth)
                } else {
                    String::new()
                }
            }
        }
    }

    pub fn get_spaces(&self) -> usize {
        match self {
            Indent::Spaces(count) => *count,
        }
    }
}

/// Options for encoding JSON values to TOON format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeOptions {
    pub delimiter: Delimiter,
    pub indent: Indent,
    pub key_folding: KeyFoldingMode,
    pub flatten_depth: usize,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            delimiter: Delimiter::Comma,
            indent: Indent::default(),
            key_folding: KeyFoldingMode::Off,
            flatten_depth: usize::MAX,
        }
    }
}

impl EncodeOptions {
    /// Create new encoding options with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the delimiter for array elements.
    pub fn with_delimiter(mut self, delimiter: Delimiter) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set the indentation string for nested structures.
    pub fn with_indent(mut self, style: Indent) -> Self {
        self.indent = style;
        self
    }

    /// Set indentation to a specific number of spaces.
    pub fn with_spaces(mut self, count: usize) -> Self {
        self.indent = Indent::Spaces(count);
        self
    }

    /// Enable key folding (v1.5 feature).
    ///
    /// When set to `Safe`, single-key object chains will be folded into
    /// dotted-path notation if all safety requirements are met.
    ///
    /// Default: `Off`
    pub fn with_key_folding(mut self, mode: KeyFoldingMode) -> Self {
        self.key_folding = mode;
        self
    }

    /// Set maximum depth for key folding.
    ///
    /// Controls how many segments will be folded. A value of 2 folds
    /// only two-segment chains: `{a: {b: val}}` → `a.b: val`.
    ///
    /// Default: `usize::MAX` (fold entire eligible chains)
    pub fn with_flatten_depth(mut self, depth: usize) -> Self {
        self.flatten_depth = depth;
        self
    }
}

/// Options for decoding TOON format to JSON values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeOptions {
    pub delimiter: Option<Delimiter>,
    pub strict: bool,
    pub coerce_types: bool,
    pub indent: Indent,
    pub expand_paths: PathExpansionMode,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            delimiter: None,
            strict: true,
            coerce_types: true,
            indent: Indent::default(),
            expand_paths: PathExpansionMode::Off,
        }
    }
}

impl DecodeOptions {
    /// Create new decoding options with defaults (strict mode enabled).
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable strict mode (validates array lengths, indentation,
    /// etc.).
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Set the expected delimiter (auto-detected if None).
    pub fn with_delimiter(mut self, delimiter: Delimiter) -> Self {
        self.delimiter = Some(delimiter);
        self
    }

    /// Enable or disable type coercion (strings like "123" -> numbers).
    pub fn with_coerce_types(mut self, coerce: bool) -> Self {
        self.coerce_types = coerce;
        self
    }

    pub fn with_indent(mut self, style: Indent) -> Self {
        self.indent = style;
        self
    }

    /// Enable path expansion (v1.5 feature).
    ///
    /// When set to `Safe`, dotted keys will be expanded into nested objects
    /// if all segments are IdentifierSegments.
    ///
    /// Conflict handling:
    /// - `strict=true`: Errors on conflicts
    /// - `strict=false`: Last-write-wins
    ///
    /// Default: `Off`
    pub fn with_expand_paths(mut self, mode: PathExpansionMode) -> Self {
        self.expand_paths = mode;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_options_indent() {
        let opts = EncodeOptions::new().with_spaces(4);
        assert_eq!(opts.indent, Indent::Spaces(4));

        let opts = EncodeOptions::new().with_indent(Indent::Spaces(2));
        assert_eq!(opts.indent, Indent::Spaces(2));
    }

    #[test]
    fn test_decode_options_coerce_types() {
        let opts = DecodeOptions::new();
        assert!(opts.coerce_types);

        let opts = DecodeOptions::new().with_coerce_types(false);
        assert!(!opts.coerce_types);

        let opts = DecodeOptions::new().with_coerce_types(true);
        assert!(opts.coerce_types);
    }
}

File: types\mod.rs
==================
mod delimeter;
mod errors;
mod folding;
mod options;
mod value;

pub use delimeter::Delimiter;
pub use errors::{ErrorContext, ToonError, ToonResult};
pub use folding::{KeyFoldingMode, PathExpansionMode, is_identifier_segment};
pub use options::{DecodeOptions, EncodeOptions, Indent};
pub use value::{IntoJsonValue, JsonValue, Number};

File: types\folding.rs
======================
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeyFoldingMode {
    /// No folding performed. All objects use standard nesting.
    #[default]
    Off,
    /// Fold eligible chains according to safety rules.
    Safe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathExpansionMode {
    /// Dotted keys are treated as literal keys. No expansion.
    #[default]
    Off,
    /// Expand eligible dotted keys according to safety rules.
    Safe,
}

/// Check if a key segment is a valid IdentifierSegment (stricter than unquoted
/// keys).
pub fn is_identifier_segment(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let mut chars = s.chars();

    // First character must be letter or underscore
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };

    if !first.is_alphabetic() && first != '_' {
        return false;
    }

    // Remaining characters: letters, digits, or underscore (NO dots)
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_folding_mode_default() {
        assert_eq!(KeyFoldingMode::default(), KeyFoldingMode::Off);
    }

    #[test]
    fn test_path_expansion_mode_default() {
        assert_eq!(PathExpansionMode::default(), PathExpansionMode::Off);
    }

    #[test]
    fn test_is_identifier_segment() {
        // Valid segments
        assert!(is_identifier_segment("a"));
        assert!(is_identifier_segment("_private"));
        assert!(is_identifier_segment("userName"));
        assert!(is_identifier_segment("user_name"));
        assert!(is_identifier_segment("user123"));
        assert!(is_identifier_segment("_123"));

        // Invalid segments
        assert!(!is_identifier_segment(""));
        assert!(!is_identifier_segment("123"));
        assert!(!is_identifier_segment("user-name"));
        assert!(!is_identifier_segment("user.name")); // Contains dot
        assert!(!is_identifier_segment("user name")); // Contains space
        assert!(!is_identifier_segment("user:name")); // Contains colon
        assert!(!is_identifier_segment(".name")); // Starts with dot
    }

    #[test]
    fn test_identifier_segment_vs_general_key() {
        // These are valid unquoted keys but NOT IdentifierSegments
        assert!(!is_identifier_segment("a.b")); // Contains dot
        assert!(!is_identifier_segment("a.b.c")); // Contains dots

        // These are valid for both
        assert!(is_identifier_segment("abc"));
        assert!(is_identifier_segment("_private"));
        assert!(is_identifier_segment("key123"));
    }
}

File: utils\literal.rs
======================
use crate::constants;

/// Check if a string looks like a keyword or number (needs quoting).
pub fn is_literal_like(s: &str) -> bool {
    is_keyword(s) || is_numeric_like(s)
}

#[inline]
pub fn is_keyword(s: &str) -> bool {
    constants::is_keyword(s)
}

#[inline]
pub fn is_structural_char(ch: char) -> bool {
    constants::is_structural_char(ch)
}

/// Check if a string looks like a number (starts with digit, no leading zeros).
pub fn is_numeric_like(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    if chars[i] == '-' {
        i += 1;
    }

    if i >= chars.len() {
        return false;
    }

    if !chars[i].is_ascii_digit() {
        return false;
    }

    if chars[i] == '0' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
        return false;
    }

    let has_valid_chars = chars[i..].iter().all(|c| {
        c.is_ascii_digit() || *c == '.' || *c == 'e' || *c == 'E' || *c == '+' || *c == '-'
    });

    has_valid_chars
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_literal_like() {
        assert!(is_literal_like("null"));
        assert!(is_literal_like("true"));
        assert!(is_literal_like("false"));
        assert!(is_literal_like("123"));
        assert!(is_literal_like("-456"));
        assert!(is_literal_like("3.14"));
        assert!(!is_literal_like("hello"));
        assert!(!is_literal_like(""));
    }

    #[test]
    fn test_is_keyword() {
        assert!(is_keyword("null"));
        assert!(is_keyword("true"));
        assert!(is_keyword("false"));
        assert!(!is_keyword("TRUE"));
        assert!(!is_keyword("hello"));
    }

    #[test]
    fn test_is_structural_char() {
        assert!(is_structural_char('['));
        assert!(is_structural_char('{'));
        assert!(is_structural_char(':'));
        assert!(!is_structural_char('a'));
    }

    #[test]
    fn test_is_numeric_like() {
        assert!(is_numeric_like("123"));
        assert!(is_numeric_like("-456"));
        assert!(is_numeric_like("0"));
        assert!(is_numeric_like("3.14"));
        assert!(is_numeric_like("1e10"));
        assert!(is_numeric_like("1.5e-3"));

        assert!(!is_numeric_like(""));
        assert!(!is_numeric_like("-"));
        assert!(!is_numeric_like("abc"));
        assert!(!is_numeric_like("01"));
        assert!(!is_numeric_like("00"));
    }
}

File: utils\mod.rs
==================
pub mod literal;
pub mod number;
pub mod string;
pub mod validation;

use indexmap::IndexMap;
pub use literal::{is_keyword, is_literal_like, is_numeric_like, is_structural_char};
pub use number::format_canonical_number;
pub use string::{
    escape_string, is_valid_unquoted_key, needs_quoting, quote_string, unescape_string,
};

use crate::types::{JsonValue as Value, Number};

/// Context for determining when quoting is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotingContext {
    ObjectValue,
    ArrayValue,
}

/// Normalize a JSON value (converts NaN/Infinity to null, -0 to 0).
pub fn normalize(value: Value) -> Value {
    match value {
        Value::Number(n) => {
            // Handle NegInt(0) case - convert to PosInt(0)
            if let Number::NegInt(0) = n {
                Value::Number(Number::from(0u64))
            } else if let Some(f) = n.as_f64() {
                if f.is_nan() || f.is_infinite() {
                    Value::Null
                } else if f == 0.0 && f.is_sign_negative() {
                    Value::Number(Number::from(0u64))
                } else {
                    Value::Number(n)
                }
            } else {
                Value::Number(n)
            }
        }
        Value::Object(obj) => {
            let normalized: IndexMap<String, Value> =
                obj.into_iter().map(|(k, v)| (k, normalize(v))).collect();
            Value::Object(normalized)
        }
        Value::Array(arr) => {
            let normalized: Vec<Value> = arr.into_iter().map(normalize).collect();
            Value::Array(normalized)
        }
        _ => value,
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use serde_json::json;

    use super::*;

    #[test]
    fn test_normalize_nan() {
        let value = Value::from(json!(f64::NAN));
        let normalized = normalize(value);
        assert_eq!(normalized, Value::from(json!(null)));
    }

    #[test]
    fn test_normalize_infinity() {
        let value = Value::from(json!(f64::INFINITY));
        let normalized = normalize(value);
        assert_eq!(normalized, Value::from(json!(null)));

        let value = Value::from(json!(f64::NEG_INFINITY));
        let normalized = normalize(value);
        assert_eq!(normalized, Value::from(json!(null)));
    }

    #[test]
    fn test_normalize_negative_zero() {
        let value = Value::from(json!(-0.0));
        let normalized = normalize(value);
        assert_eq!(normalized, Value::from(json!(0)));
    }

    #[test]
    fn test_normalize_nested() {
        let value = Value::from(json!({
            "a": f64::NAN,
            "b": {
                "c": f64::INFINITY
            },
            "d": [1, f64::NAN, 3]
        }));

        let normalized = normalize(value);
        assert_eq!(
            normalized,
            Value::from(json!({
                "a": null,
                "b": {
                    "c": null
                },
                "d": [1, null, 3]
            }))
        );
    }

    #[test]
    fn test_normalize_normal_values() {
        let value = Value::from(json!({
            "name": "Alice",
            "age": 30,
            "score": f64::consts::PI
        }));

        let normalized = normalize(value.clone());
        assert_eq!(normalized, value);
    }
}

File: types\value.rs
====================
use std::{
    fmt,
    ops::{Index, IndexMut},
};

use indexmap::IndexMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Number {
    PosInt(u64),
    NegInt(i64),
    Float(f64),
}

impl Number {
    pub fn from_f64(f: f64) -> Option<Self> {
        if f.is_finite() {
            Some(Number::Float(f))
        } else {
            None
        }
    }

    pub fn is_i64(&self) -> bool {
        match self {
            Number::NegInt(_) => true,
            Number::PosInt(u) => *u <= i64::MAX as u64,
            Number::Float(f) => {
                let i = *f as i64;
                i as f64 == *f && i != i64::MAX
            }
        }
    }

    pub fn is_u64(&self) -> bool {
        match self {
            Number::PosInt(_) => true,
            Number::NegInt(_) => false,
            Number::Float(f) => {
                let u = *f as u64;
                u as f64 == *f
            }
        }
    }

    pub fn is_f64(&self) -> bool {
        matches!(self, Number::Float(_))
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Number::PosInt(u) => {
                if *u <= i64::MAX as u64 {
                    Some(*u as i64)
                } else {
                    None
                }
            }
            Number::NegInt(i) => Some(*i),
            Number::Float(f) => {
                let i = *f as i64;
                if i as f64 == *f { Some(i) } else { None }
            }
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Number::PosInt(u) => Some(*u),
            Number::NegInt(_) => None,
            Number::Float(f) => {
                if *f >= 0.0 {
                    let u = *f as u64;
                    if u as f64 == *f { Some(u) } else { None }
                } else {
                    None
                }
            }
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Number::PosInt(u) => Some(*u as f64),
            Number::NegInt(i) => Some(*i as f64),
            Number::Float(f) => Some(*f),
        }
    }

    pub fn is_integer(&self) -> bool {
        match self {
            Number::PosInt(_) | Number::NegInt(_) => true,
            Number::Float(f) => f.fract() == 0.0,
        }
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s_json_num = match self {
            Number::PosInt(u) => serde_json::Number::from(*u),
            Number::NegInt(i) => serde_json::Number::from(*i),
            Number::Float(fl) => {
                serde_json::Number::from_f64(*fl).unwrap_or_else(|| serde_json::Number::from(0))
            }
        };
        write!(f, "{s_json_num}")
    }
}

impl From<i8> for Number {
    fn from(n: i8) -> Self {
        Number::NegInt(n as i64)
    }
}

impl From<i16> for Number {
    fn from(n: i16) -> Self {
        Number::NegInt(n as i64)
    }
}

impl From<i32> for Number {
    fn from(n: i32) -> Self {
        Number::NegInt(n as i64)
    }
}

impl From<i64> for Number {
    fn from(n: i64) -> Self {
        if n >= 0 {
            Number::PosInt(n as u64)
        } else {
            Number::NegInt(n)
        }
    }
}

impl From<isize> for Number {
    fn from(n: isize) -> Self {
        Number::from(n as i64)
    }
}

impl From<u8> for Number {
    fn from(n: u8) -> Self {
        Number::PosInt(n as u64)
    }
}

impl From<u16> for Number {
    fn from(n: u16) -> Self {
        Number::PosInt(n as u64)
    }
}

impl From<u32> for Number {
    fn from(n: u32) -> Self {
        Number::PosInt(n as u64)
    }
}

impl From<u64> for Number {
    fn from(n: u64) -> Self {
        Number::PosInt(n)
    }
}

impl From<usize> for Number {
    fn from(n: usize) -> Self {
        Number::PosInt(n as u64)
    }
}

impl From<f32> for Number {
    fn from(n: f32) -> Self {
        Number::Float(n as f64)
    }
}

impl From<f64> for Number {
    fn from(n: f64) -> Self {
        Number::Float(n)
    }
}

pub type Object = IndexMap<String, JsonValue>;

#[derive(Clone, Debug, PartialEq, Default)]
pub enum JsonValue {
    #[default]
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<JsonValue>),
    Object(Object),
}

impl JsonValue {
    pub const fn is_null(&self) -> bool {
        matches!(self, JsonValue::Null)
    }

    pub const fn is_bool(&self) -> bool {
        matches!(self, JsonValue::Bool(_))
    }

    pub const fn is_number(&self) -> bool {
        matches!(self, JsonValue::Number(_))
    }

    pub const fn is_string(&self) -> bool {
        matches!(self, JsonValue::String(_))
    }

    pub const fn is_array(&self) -> bool {
        matches!(self, JsonValue::Array(_))
    }

    pub const fn is_object(&self) -> bool {
        matches!(self, JsonValue::Object(_))
    }

    /// Returns true if the value is a number that can be represented as i64
    pub fn is_i64(&self) -> bool {
        match self {
            JsonValue::Number(n) => n.is_i64(),
            _ => false,
        }
    }

    /// Returns true if the value is a number that can be represented as u64
    pub fn is_u64(&self) -> bool {
        match self {
            JsonValue::Number(n) => n.is_u64(),
            _ => false,
        }
    }

    pub fn is_f64(&self) -> bool {
        match self {
            JsonValue::Number(n) => n.is_f64(),
            _ => false,
        }
    }

    /// If the value is a Bool, returns the associated bool. Returns None
    /// otherwise.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// If the value is a number, represent it as i64 if possible. Returns None
    /// otherwise.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            JsonValue::Number(n) => n.as_i64(),
            _ => None,
        }
    }

    /// If the value is a number, represent it as u64 if possible. Returns None
    /// otherwise.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            JsonValue::Number(n) => n.as_u64(),
            _ => None,
        }
    }

    /// If the value is a number, represent it as f64 if possible. Returns None
    /// otherwise.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => n.as_f64(),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<JsonValue>> {
        match self {
            JsonValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    pub fn as_array_mut(&mut self) -> Option<&mut Vec<JsonValue>> {
        match self {
            JsonValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&Object> {
        match self {
            JsonValue::Object(obj) => Some(obj),
            _ => None,
        }
    }

    pub fn as_object_mut(&mut self) -> Option<&mut Object> {
        match self {
            JsonValue::Object(obj) => Some(obj),
            _ => None,
        }
    }

    /// Takes the value, leaving Null in its place.
    pub fn take(&mut self) -> JsonValue {
        std::mem::replace(self, JsonValue::Null)
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            JsonValue::Null => "null",
            JsonValue::Bool(_) => "boolean",
            JsonValue::Number(_) => "number",
            JsonValue::String(_) => "string",
            JsonValue::Array(_) => "array",
            JsonValue::Object(_) => "object",
        }
    }
}

impl fmt::Display for JsonValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonValue::Null => write!(f, "null"),
            JsonValue::Bool(b) => write!(f, "{b}"),
            JsonValue::Number(n) => write!(f, "{n}"),
            JsonValue::String(s) => write!(f, "\"{s}\""),
            JsonValue::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            JsonValue::Object(obj) => {
                write!(f, "{{")?;
                for (i, (k, v)) in obj.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "\"{k}\": {v}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

impl Index<usize> for JsonValue {
    type Output = JsonValue;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            JsonValue::Array(arr) => &arr[index],
            _ => panic!("cannot index into non-array value with usize"),
        }
    }
}

impl IndexMut<usize> for JsonValue {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            JsonValue::Array(arr) => &mut arr[index],
            _ => panic!("cannot index into non-array value with usize"),
        }
    }
}

impl Index<&str> for JsonValue {
    type Output = JsonValue;

    fn index(&self, key: &str) -> &Self::Output {
        match self {
            JsonValue::Object(obj) => obj
                .get(key)
                .unwrap_or_else(|| panic!("key '{key}' not found in object")),
            _ => panic!("cannot index into non-object value with &str"),
        }
    }
}

impl IndexMut<&str> for JsonValue {
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        match self {
            JsonValue::Object(obj) => obj
                .get_mut(key)
                .unwrap_or_else(|| panic!("key '{key}' not found in object")),
            _ => panic!("cannot index into non-object value with &str"),
        }
    }
}

impl Index<String> for JsonValue {
    type Output = JsonValue;

    fn index(&self, key: String) -> &Self::Output {
        self.index(key.as_str())
    }
}

impl IndexMut<String> for JsonValue {
    fn index_mut(&mut self, key: String) -> &mut Self::Output {
        self.index_mut(key.as_str())
    }
}

impl From<serde_json::Value> for JsonValue {
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => JsonValue::Null,
            serde_json::Value::Bool(b) => JsonValue::Bool(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    JsonValue::Number(Number::from(i))
                } else if let Some(u) = n.as_u64() {
                    JsonValue::Number(Number::from(u))
                } else if let Some(f) = n.as_f64() {
                    JsonValue::Number(Number::from(f))
                } else {
                    JsonValue::Null
                }
            }
            serde_json::Value::String(s) => JsonValue::String(s),
            serde_json::Value::Array(arr) => {
                JsonValue::Array(arr.into_iter().map(JsonValue::from).collect())
            }
            serde_json::Value::Object(obj) => {
                let mut new_obj = Object::new();
                for (k, v) in obj {
                    new_obj.insert(k, JsonValue::from(v));
                }
                JsonValue::Object(new_obj)
            }
        }
    }
}

impl From<&serde_json::Value> for JsonValue {
    fn from(value: &serde_json::Value) -> Self {
        value.clone().into()
    }
}

impl From<JsonValue> for serde_json::Value {
    fn from(value: JsonValue) -> Self {
        match value {
            JsonValue::Null => serde_json::Value::Null,
            JsonValue::Bool(b) => serde_json::Value::Bool(b),
            JsonValue::Number(n) => {
                if let Some(i) = n.as_i64() {
                    serde_json::Value::Number(i.into())
                } else if let Some(u) = n.as_u64() {
                    serde_json::Value::Number(u.into())
                } else if let Some(f) = n.as_f64() {
                    serde_json::Number::from_f64(f)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null)
                } else {
                    serde_json::Value::Null
                }
            }
            JsonValue::String(s) => serde_json::Value::String(s),
            JsonValue::Array(arr) => {
                serde_json::Value::Array(arr.into_iter().map(Into::into).collect())
            }
            JsonValue::Object(obj) => {
                let mut new_obj = serde_json::Map::new();
                for (k, v) in obj {
                    new_obj.insert(k, v.into());
                }
                serde_json::Value::Object(new_obj)
            }
        }
    }
}

impl From<&JsonValue> for serde_json::Value {
    fn from(value: &JsonValue) -> Self {
        value.clone().into()
    }
}

pub trait IntoJsonValue {
    fn into_json_value(self) -> JsonValue;
}

impl IntoJsonValue for &JsonValue {
    fn into_json_value(self) -> JsonValue {
        self.clone()
    }
}

impl IntoJsonValue for JsonValue {
    fn into_json_value(self) -> JsonValue {
        self
    }
}

impl IntoJsonValue for &serde_json::Value {
    fn into_json_value(self) -> JsonValue {
        self.into()
    }
}

impl IntoJsonValue for serde_json::Value {
    fn into_json_value(self) -> JsonValue {
        (&self).into()
    }
}

File: utils\number.rs
=====================
use crate::types::Number;

/// Format a number in TOON canonical form (no exponents, no trailing zeros).
pub fn format_canonical_number(n: &Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }

    if let Some(u) = n.as_u64() {
        return u.to_string();
    }

    if let Some(f) = n.as_f64() {
        return format_f64_canonical(f);
    }

    n.to_string()
}

fn format_f64_canonical(f: f64) -> String {
    // Normalize integer-valued floats to integers
    if f.is_finite() && f.fract() == 0.0 && f.abs() <= i64::MAX as f64 {
        return format!("{}", f as i64);
    }

    let default_format = format!("{f}");

    // Handle cases where Rust would use exponential notation
    if default_format.contains('e') || default_format.contains('E') {
        format_without_exponent(f)
    } else {
        remove_trailing_zeros(&default_format)
    }
}

fn format_without_exponent(f: f64) -> String {
    if !f.is_finite() {
        return "0".to_string();
    }

    if f.abs() >= 1.0 {
        let abs_f = f.abs();
        let int_part = abs_f.trunc();
        let frac_part = abs_f.fract();

        if frac_part == 0.0 {
            format!("{}{}", if f < 0.0 { "-" } else { "" }, int_part as i64)
        } else {
            // High precision to avoid exponent, then trim trailing zeros
            let result = format!("{f:.17}");
            remove_trailing_zeros(&result)
        }
    } else if f == 0.0 {
        "0".to_string()
    } else {
        // Small numbers: use high precision to avoid exponent
        let result = format!("{f:.17}",);
        remove_trailing_zeros(&result)
    }
}

fn remove_trailing_zeros(s: &str) -> String {
    if !s.contains('.') {
        // No decimal point, return as-is
        return s.to_string();
    }

    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 2 {
        return s.to_string();
    }

    let int_part = parts[0];
    let mut frac_part = parts[1].to_string();

    frac_part = frac_part.trim_end_matches('0').to_string();

    if frac_part.is_empty() {
        // All zeros removed, return as integer
        int_part.to_string()
    } else {
        format!("{int_part}.{frac_part}")
    }
}

#[cfg(test)]
mod tests {
    use std::f64;

    use serde_json::json;

    use super::*;

    #[test]
    fn test_format_canonical_integers() {
        let n = Number::from(42i64);
        assert_eq!(format_canonical_number(&n), "42");

        let n = Number::from(-123i64);
        assert_eq!(format_canonical_number(&n), "-123");

        let n = Number::from(0i64);
        assert_eq!(format_canonical_number(&n), "0");
    }

    #[test]
    fn test_format_canonical_floats() {
        // Integer-valued floats
        let n = Number::from_f64(1.0).unwrap();
        assert_eq!(format_canonical_number(&n), "1");

        let n = Number::from_f64(42.0).unwrap();
        assert_eq!(format_canonical_number(&n), "42");

        // Non-integer floats
        let n = Number::from_f64(1.5).unwrap();
        assert_eq!(format_canonical_number(&n), "1.5");

        let n = Number::from_f64(f64::consts::PI).unwrap();
        let result = format_canonical_number(&n);
        assert!(result.starts_with("3.141592653589793"));
        assert!(!result.contains('e'));
        assert!(!result.contains('E'));
    }

    #[test]
    fn test_remove_trailing_zeros() {
        assert_eq!(remove_trailing_zeros("1.5000"), "1.5");
        assert_eq!(remove_trailing_zeros("1.0"), "1");
        assert_eq!(remove_trailing_zeros("1.500"), "1.5");
        assert_eq!(remove_trailing_zeros("42"), "42");
        assert_eq!(remove_trailing_zeros("0.0"), "0");
        assert_eq!(remove_trailing_zeros("1.23"), "1.23");
    }

    #[test]
    fn test_large_numbers_no_exponent() {
        // 1e6 should become 1000000
        let n = Number::from_f64(1_000_000.0).unwrap();
        let result = format_canonical_number(&n);
        assert_eq!(result, "1000000");
        assert!(!result.contains('e'));

        // 1e9
        let n = Number::from_f64(1_000_000_000.0).unwrap();
        let result = format_canonical_number(&n);
        assert_eq!(result, "1000000000");
        assert!(!result.contains('e'));
    }

    #[test]
    fn test_small_numbers_no_exponent() {
        // 1e-6 should become 0.000001
        let n = Number::from_f64(0.000001).unwrap();
        let result = format_canonical_number(&n);
        assert!(result.starts_with("0.000001"));
        assert!(!result.contains('e'));
        assert!(!result.contains('E'));

        // 1e-3
        let n = Number::from_f64(0.001).unwrap();
        let result = format_canonical_number(&n);
        assert_eq!(result, "0.001");
    }

    #[test]
    fn test_pi_formatting() {
        let n = Number::from_f64(std::f64::consts::PI).unwrap();
        let result = format_canonical_number(&n);

        // Should not have exponent
        assert!(!result.contains('e'));
        assert!(!result.contains('E'));

        // Should start with 3.14159...
        assert!(result.starts_with("3.14159"));
    }

    #[test]
    fn test_from_json_values() {
        // Test with actual JSON values
        let val = json!(1000000);
        if let Some(n) = val.as_i64() {
            let num = Number::from(n);
            assert_eq!(format_canonical_number(&num), "1000000");
        }

        let val = json!(1.5000);
        if let Some(f) = val.as_f64() {
            let num = Number::from_f64(f).unwrap();
            assert_eq!(format_canonical_number(&num), "1.5");
        }
    }

    #[test]
    fn test_negative_numbers() {
        let n = Number::from_f64(-1.5).unwrap();
        assert_eq!(format_canonical_number(&n), "-1.5");

        let n = Number::from(-42i64);
        assert_eq!(format_canonical_number(&n), "-42");

        let n = Number::from_f64(-1000000.0).unwrap();
        assert_eq!(format_canonical_number(&n), "-1000000");
    }
}

File: utils\validation.rs
=========================
use serde_json::Value;

use crate::types::{ToonError, ToonResult};

/// Validate that nesting depth doesn't exceed the maximum.
pub fn validate_depth(depth: usize, max_depth: usize) -> ToonResult<()> {
    if depth > max_depth {
        return Err(ToonError::InvalidStructure(
            "Maximum nesting depth of {max_depth} exceeded".to_string(),
        ));
    }
    Ok(())
}

/// Validate that a field name is not empty.
pub fn validate_field_name(name: &str) -> ToonResult<()> {
    if name.is_empty() {
        return Err(ToonError::InvalidInput(
            "Field name cannot be empty".to_string(),
        ));
    }
    Ok(())
}

/// Recursively validate a JSON value and all nested fields.
pub fn validate_value(value: &Value) -> ToonResult<()> {
    match value {
        Value::Object(obj) => {
            for (key, val) in obj.iter() {
                validate_field_name(key)?;
                validate_value(val)?;
            }
        }
        Value::Array(arr) => {
            for val in arr.iter() {
                validate_value(val)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_validate_depth() {
        assert!(validate_depth(0, 10).is_ok());
        assert!(validate_depth(5, 10).is_ok());
        assert!(validate_depth(10, 10).is_ok());
        assert!(validate_depth(11, 10).is_err());
    }

    #[test]
    fn test_validate_field_name() {
        assert!(validate_field_name("name").is_ok());
        assert!(validate_field_name("user_id").is_ok());
        assert!(validate_field_name("").is_err());
    }

    #[test]
    fn test_validate_value() {
        assert!(validate_value(&json!(null)).is_ok());
        assert!(validate_value(&json!(123)).is_ok());
        assert!(validate_value(&json!("hello")).is_ok());
        assert!(validate_value(&json!({"name": "Alice"})).is_ok());
        assert!(validate_value(&json!([1, 2, 3])).is_ok());

        let bad_obj = json!({"": "value"});
        assert!(validate_value(&bad_obj).is_err());
    }
}

File: utils\string.rs
=====================
use crate::{types::Delimiter, utils::literal};

/// Escape special characters in a string for quoted output.
pub fn escape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());

    for ch in s.chars() {
        match ch {
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            _ => result.push(ch),
        }
    }

    result
}

/// Unescape special characters in a quoted string.
///
/// Per TOON spec §7.1, only these escape sequences are valid:
/// - `\\` → `\`
/// - `\"` → `"`
/// - `\n` → newline
/// - `\r` → carriage return
/// - `\t` → tab
///
/// Any other escape sequence MUST cause an error.
///
/// # Errors
///
/// Returns an error if the string contains an invalid escape sequence
/// or if a backslash appears at the end of the string.
pub fn unescape_string(s: &str) -> Result<String, String> {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    let mut position = 0;

    while let Some(ch) = chars.next() {
        position += 1;

        if ch == '\\' {
            if let Some(&next) = chars.peek() {
                match next {
                    'n' => {
                        result.push('\n');
                        chars.next(); // consume the 'n'
                        position += 1;
                    }
                    'r' => {
                        result.push('\r');
                        chars.next();
                        position += 1;
                    }
                    't' => {
                        result.push('\t');
                        chars.next();
                        position += 1;
                    }
                    '"' => {
                        result.push('"');
                        chars.next();
                        position += 1;
                    }
                    '\\' => {
                        result.push('\\');
                        chars.next();
                        position += 1;
                    }
                    _ => {
                        return Err(format!(
                            "Invalid escape sequence '\\{next}' at position {position}. Only \
                             \\\\, \\\", \\n, \\r, \\t are valid",
                        ));
                    }
                }
            } else {
                return Err(format!(
                    "Unterminated escape sequence at end of string (position {position})",
                ));
            }
        } else {
            result.push(ch);
        }
    }

    Ok(result)
}

/// Check if a key can be written without quotes (alphanumeric, underscore,
/// dot).
pub fn is_valid_unquoted_key(key: &str) -> bool {
    if key.is_empty() {
        return false;
    }

    let mut chars = key.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };

    if !first.is_alphabetic() && first != '_' {
        return false;
    }

    chars.all(|c| c.is_alphanumeric() || c == '_' || c == '.')
}

/// Determine if a string needs quoting based on content and delimiter.
pub fn needs_quoting(s: &str, delimiter: char) -> bool {
    if s.is_empty() {
        return true;
    }

    if literal::is_literal_like(s) {
        return true;
    }

    if s.chars().any(literal::is_structural_char) {
        return true;
    }

    if s.contains('\\') || s.contains('"') {
        return true;
    }

    if s.contains(delimiter) {
        return true;
    }

    if s.contains('\n') || s.contains('\r') || s.contains('\t') {
        return true;
    }

    if s.starts_with(char::is_whitespace) || s.ends_with(char::is_whitespace) {
        return true;
    }

    if s.starts_with("-") {
        return true;
    }

    // Check for leading zeros (e.g., "05", "007", "0123")
    // Numbers with leading zeros must be quoted
    if s.starts_with('0') && s.len() > 1 && s.chars().nth(1).is_some_and(|c| c.is_ascii_digit()) {
        return true;
    }

    false
}

/// Quote and escape a string.
pub fn quote_string(s: &str) -> String {
    format!("\"{}\"", escape_string(s))
}

pub fn split_by_delimiter(s: &str, delimiter: Delimiter) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let chars = s.chars().peekable();
    let delim_char = delimiter.as_char();

    for ch in chars {
        if ch == '"' && (current.is_empty() || !current.ends_with('\\')) {
            in_quotes = !in_quotes;
            current.push(ch);
        } else if ch == delim_char && !in_quotes {
            result.push(current.trim().to_string());
            current.clear();
        } else {
            current.push(ch);
        }
    }

    if !current.is_empty() {
        result.push(current.trim().to_string());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_string() {
        assert_eq!(escape_string("hello"), "hello");
        assert_eq!(escape_string("hello\nworld"), "hello\\nworld");
        assert_eq!(escape_string("say \"hi\""), "say \\\"hi\\\"");
        assert_eq!(escape_string("back\\slash"), "back\\\\slash");
    }

    #[test]
    fn test_unescape_string() {
        // Valid escapes
        assert_eq!(unescape_string("hello").unwrap(), "hello");
        assert_eq!(unescape_string("hello\\nworld").unwrap(), "hello\nworld");
        assert_eq!(unescape_string("say \\\"hi\\\"").unwrap(), "say \"hi\"");
        assert_eq!(unescape_string("back\\\\slash").unwrap(), "back\\slash");
        assert_eq!(unescape_string("tab\\there").unwrap(), "tab\there");
        assert_eq!(unescape_string("return\\rhere").unwrap(), "return\rhere");
    }

    #[test]
    fn test_unescape_string_invalid_escapes() {
        // Invalid escape sequences should error
        assert!(unescape_string("bad\\xescape").is_err());
        assert!(unescape_string("bad\\uescape").is_err());
        assert!(unescape_string("bad\\0escape").is_err());
        assert!(unescape_string("bad\\aescape").is_err());

        // Unterminated escape at end
        assert!(unescape_string("ends\\").is_err());
    }

    #[test]
    fn test_unescape_string_error_messages() {
        let result = unescape_string("bad\\x");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Invalid escape sequence"));
        assert!(err.contains("\\x"));
    }

    #[test]
    fn test_needs_quoting() {
        let comma = Delimiter::Comma.as_char();

        assert!(needs_quoting("", comma));

        assert!(needs_quoting("true", comma));
        assert!(needs_quoting("false", comma));
        assert!(needs_quoting("null", comma));
        assert!(needs_quoting("123", comma));

        assert!(needs_quoting("hello[world]", comma));
        assert!(needs_quoting("key:value", comma));

        assert!(needs_quoting("a,b", comma));
        assert!(!needs_quoting("a,b", Delimiter::Pipe.as_char()));

        assert!(!needs_quoting("hello world", comma));
        assert!(needs_quoting(" hello", comma));
        assert!(needs_quoting("hello ", comma));

        assert!(!needs_quoting("hello", comma));
        assert!(!needs_quoting("world", comma));
        assert!(!needs_quoting("helloworld", comma));
    }

    #[test]
    fn test_quote_string() {
        assert_eq!(quote_string("hello"), "\"hello\"");
        assert_eq!(quote_string("hello\nworld"), "\"hello\\nworld\"");
    }

    #[test]
    fn test_split_by_delimiter() {
        let comma = Delimiter::Comma;

        assert_eq!(split_by_delimiter("a,b,c", comma), vec!["a", "b", "c"]);

        assert_eq!(split_by_delimiter("a, b, c", comma), vec!["a", "b", "c"]);

        assert_eq!(split_by_delimiter("\"a,b\",c", comma), vec!["\"a,b\"", "c"]);
    }

    #[test]
    fn test_is_valid_unquoted_key() {
        // Valid keys (should return true)
        assert!(is_valid_unquoted_key("normal_key"));
        assert!(is_valid_unquoted_key("key123"));
        assert!(is_valid_unquoted_key("key.value"));
        assert!(is_valid_unquoted_key("_private"));
        assert!(is_valid_unquoted_key("KeyName"));
        assert!(is_valid_unquoted_key("key_name"));
        assert!(is_valid_unquoted_key("key.name.sub"));
        assert!(is_valid_unquoted_key("a"));
        assert!(is_valid_unquoted_key("_"));
        assert!(is_valid_unquoted_key("key_123.value"));

        assert!(!is_valid_unquoted_key(""));
        assert!(!is_valid_unquoted_key("123"));
        assert!(!is_valid_unquoted_key("key:value"));
        assert!(!is_valid_unquoted_key("key-value"));
        assert!(!is_valid_unquoted_key("key value"));
        assert!(!is_valid_unquoted_key(".key"));
        assert!(is_valid_unquoted_key("key.value.sub."));
        assert!(is_valid_unquoted_key("key."));
        assert!(!is_valid_unquoted_key("key[value]"));
        assert!(!is_valid_unquoted_key("key{value}"));
    }
}

File: rune\hydron\eval.rs
=========================
//! RUNE Expression Evaluator
//!
//! Evaluates RUNE expressions with semantic prefixes, array literals, and operators.
//! Supports mathematical operations, semantic type checking, and E8 geometry primitives.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use super::values::{EvalContext, EvalError, Value};
use crate::rune::ops::MathOp;
use crate::rune::{
    Expr, Ident, Literal, MathAtom, MathExpr, MathUnaryOp, RuneOp, SemanticIdent, Stmt, Term,
};
use std::collections::HashMap;

/// RUNE expression evaluator with semantic type support
pub struct Evaluator {
    /// Variable bindings (name -> value)
    variables: HashMap<String, Value>,
    /// Semantic namespace bindings (T:name -> value)
    semantic_vars: HashMap<String, Value>,
}

impl Evaluator {
    /// Create a new evaluator with empty context
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            semantic_vars: HashMap::new(),
        }
    }

    /// Create evaluator with pre-populated context
    pub fn with_context(ctx: EvalContext) -> Self {
        Self {
            variables: ctx.variables,
            semantic_vars: ctx.semantic_vars,
        }
    }

    /// Set a variable value
    pub fn set_var(&mut self, name: impl Into<String>, value: Value) {
        self.variables.insert(name.into(), value);
    }

    /// Set a semantic variable value (e.g., T:Gf8)
    pub fn set_semantic(&mut self, prefix: char, name: impl Into<String>, value: Value) {
        let key = format!("{}:{}", prefix, name.into());
        self.semantic_vars.insert(key, value);
    }

    /// Print SIMD capabilities (diagnostic)
    #[cfg(feature = "simd")]
    pub fn print_simd_info(&self) {
        use super::values::{get_available_f32_256_intrinsics, print_simd_capabilities};
        print_simd_capabilities();
        let intrinsics = get_available_f32_256_intrinsics();
        println!("Available f32x256 intrinsics: {:?}", intrinsics);
    }

    /// Get a variable value
    pub fn get_var(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }

    /// Get a semantic variable value
    pub fn get_semantic(&self, prefix: char, name: &str) -> Option<&Value> {
        let key = format!("{}:{}", prefix, name);
        self.semantic_vars.get(&key)
    }

    /// Evaluate a statement
    pub fn eval_stmt(&mut self, stmt: &Stmt) -> Result<Value, EvalError> {
        match stmt {
            Stmt::RootDecl(root) => {
                // Root declarations don't produce values, but we can store them as context
                Ok(Value::String(root.to_string()))
            }
            Stmt::ToonBlock { name, content } => {
                // TOON blocks are data, not computation - return the raw content
                Ok(Value::String(format!(
                    "TOON block '{}': {} chars",
                    name,
                    content.len()
                )))
            }
            Stmt::Expr(expr) => self.eval_expr(expr),
        }
    }

    /// Evaluate an expression
    pub fn eval_expr(&mut self, expr: &Expr) -> Result<Value, EvalError> {
        match expr {
            Expr::Term(term) => self.eval_term(term),
            Expr::Binary { left, op, right } => {
                let left_val = self.eval_expr(left)?;
                let right_val = self.eval_expr(right)?;
                self.eval_binary_op(&left_val, op, &right_val)
            }
        }
    }

    /// Evaluate a term
    fn eval_term(&self, term: &Term) -> Result<Value, EvalError> {
        match term {
            Term::Literal(lit) => self.eval_literal(lit),
            Term::Ident(ident) => self.eval_ident(ident),
            Term::SemanticIdent(sem) => self.eval_semantic_ident(sem),
            Term::Group(expr) => {
                // Group expressions are used for math blocks [expr]
                // For now, just evaluate the inner expression
                let mut temp_eval = Self {
                    variables: self.variables.clone(),
                    semantic_vars: self.semantic_vars.clone(),
                };
                temp_eval.eval_expr(expr)
            }
            Term::Math(math_expr) => {
                // Math blocks contain MathExpr which needs evaluation
                self.eval_math_expr(math_expr)
            }
        }
    }

    /// Evaluate a math expression
    fn eval_math_expr(&self, math: &MathExpr) -> Result<Value, EvalError> {
        match math {
            MathExpr::Atom(atom) => self.eval_math_atom(atom),
            MathExpr::Binary { left, op, right } => {
                let left_val = self.eval_math_expr(left)?;
                let right_val = self.eval_math_expr(right)?;
                self.eval_math_op(&left_val, op, &right_val)
            }
            MathExpr::Unary { op, operand } => {
                let val = self.eval_math_expr(operand)?;
                self.eval_math_unary_op(op, &val)
            }
        }
    }

    /// Evaluate a math atom
    fn eval_math_atom(&self, atom: &MathAtom) -> Result<Value, EvalError> {
        match atom {
            MathAtom::Number(n) => Ok(Value::Float(*n)),
            MathAtom::Ident(ident) => {
                // Check if it's a semantic identifier (contains ':')
                if ident.0.contains(':') {
                    let parts: Vec<&str> = ident.0.split(':').collect();
                    if parts.len() == 2 && parts[0].len() == 1 {
                        let prefix = parts[0].chars().next().unwrap();
                        self.get_semantic(prefix, parts[1])
                            .cloned()
                            .ok_or_else(|| EvalError::UndefinedVariable(ident.0.clone()))
                    } else {
                        self.eval_ident(ident)
                    }
                } else {
                    self.eval_ident(ident)
                }
            }
            MathAtom::Group(math) => self.eval_math_expr(math),
            MathAtom::Array(elements) => {
                // Evaluate array literal inside math block
                let mut values = Vec::new();
                for elem in elements {
                    values.push(self.eval_math_expr(elem)?);
                }
                Ok(Value::Array(values))
            }
        }
    }

    /// Evaluate a math binary operation
    fn eval_math_op(&self, left: &Value, op: &MathOp, right: &Value) -> Result<Value, EvalError> {
        match op {
            MathOp::Add => left.add(right),
            MathOp::Subtract => left.sub(right),
            MathOp::Multiply => left.mul(right),
            MathOp::Divide => left.div(right),
            MathOp::Power => left.pow(right),
            MathOp::Modulo => left.modulo(right),
            MathOp::Root => Err(EvalError::UnsupportedOperation(
                "Root operator not yet implemented".into(),
            )),
        }
    }

    /// Evaluate a math unary operation
    fn eval_math_unary_op(&self, op: &MathUnaryOp, val: &Value) -> Result<Value, EvalError> {
        match op {
            MathUnaryOp::Negate => val.negate(),
            MathUnaryOp::Plus => Ok(val.clone()),
        }
    }

    /// Evaluate a literal value
    fn eval_literal(&self, lit: &Literal) -> Result<Value, EvalError> {
        match lit {
            Literal::Number(n) => Ok(Value::Float(*n)),
            Literal::String(s) => Ok(Value::String(s.clone())),
            Literal::Array(exprs) => {
                let mut values = Vec::new();
                let mut temp_eval = Self {
                    variables: self.variables.clone(),
                    semantic_vars: self.semantic_vars.clone(),
                };
                for expr in exprs {
                    values.push(temp_eval.eval_expr(expr)?);
                }
                Ok(Value::Array(values))
            }
        }
    }

    /// Evaluate an identifier (variable lookup)
    fn eval_ident(&self, ident: &Ident) -> Result<Value, EvalError> {
        self.variables
            .get(&ident.0)
            .cloned()
            .ok_or_else(|| EvalError::UndefinedVariable(ident.0.clone()))
    }

    /// Evaluate a semantic identifier (T:name, V:velocity, etc.)
    fn eval_semantic_ident(&self, sem: &SemanticIdent) -> Result<Value, EvalError> {
        let key = format!("{}:{}", sem.prefix, sem.name.0);
        self.semantic_vars
            .get(&key)
            .cloned()
            .ok_or_else(|| EvalError::UndefinedVariable(key))
    }

    /// Evaluate a binary operation using RuneOp (structural operations only)
    /// Arithmetic operations are handled by MathOp within math blocks `[]`
    fn eval_binary_op(&self, left: &Value, op: &RuneOp, right: &Value) -> Result<Value, EvalError> {
        use RuneOp::*;

        match op {
            // Comparison operators
            Less => left.lt(right),
            LessEqual => left.le(right),
            Greater => left.gt(right),
            GreaterEqual => left.ge(right),
            Equal => Ok(Value::Bool(left == right)),

            // Structural operators (not for computation) - arithmetic handled by MathOp
            Descendant | Ancestor | Define | FlowRight | FlowLeft | Bind | Namespace | Alias
            | Parallel | Transform | SplitJoin | JoinSplit | AnchorDescend | BranchStabilize
            | RootStabilize | StabilizeRoot | SymmetricSplit | BranchAnchorBranch => {
                Err(EvalError::UnsupportedOperation(format!(
                    "Structural operator {:?} not implemented for computation. Use math blocks `[]` for arithmetic.",
                    op
                )))
            }
        }
    }

    /// Export current context
    pub fn context(&self) -> EvalContext {
        let mut ctx = EvalContext::new();
        for (name, value) in &self.variables {
            ctx.bind(name.clone(), value.clone());
        }
        // Note: semantic_vars would need to be stored differently in EvalContext
        ctx
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rune::parse;

    #[test]
    fn test_eval_literal_number() {
        let mut eval = Evaluator::new();
        let stmts = parse("42").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(42.0));
    }

    #[test]
    fn test_eval_arithmetic() {
        let mut eval = Evaluator::new();

        // Simple addition in math block
        let stmts = parse("[2 + 3]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(5.0));

        // Multiplication in math block
        let stmts = parse("[4 * 5]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(20.0));

        // Complex expression with precedence
        let stmts = parse("[2 + 3 * 4]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(14.0)); // Respects precedence

        // Division in math block
        let stmts = parse("[10 / 2]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(5.0));

        // Power in math block
        let stmts = parse("[2 ^ 3]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(8.0));

        // Modulo in math block
        let stmts = parse("[10 % 3]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(1.0));
    }

    #[test]
    fn test_eval_array_literal() {
        let mut eval = Evaluator::new();
        let stmts = parse("[1, 2, 3]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();

        match result {
            Value::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0], Value::Float(1.0));
                assert_eq!(arr[1], Value::Float(2.0));
                assert_eq!(arr[2], Value::Float(3.0));
            }
            _ => panic!("Expected array value"),
        }
    }

    #[test]
    fn test_eval_array_operations() {
        let mut eval = Evaluator::new();

        // Array addition (element-wise) in math block
        let stmts = parse("[[1, 2, 3] + [4, 5, 6]]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();

        match result {
            Value::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0], Value::Float(5.0));
                assert_eq!(arr[1], Value::Float(7.0));
                assert_eq!(arr[2], Value::Float(9.0));
            }
            _ => panic!("Expected array value"),
        }
    }

    #[test]
    fn test_eval_semantic_prefix() {
        let mut eval = Evaluator::new();

        // Set semantic variable
        eval.set_semantic('T', "Gf8", Value::Float(2.5));

        // Evaluate semantic expression in math block
        let stmts = parse("[T:Gf8 * 3]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(7.5));
    }

    #[test]
    fn test_eval_variables() {
        let mut eval = Evaluator::new();

        // Set variable
        eval.set_var("x", Value::Float(10.0));

        // Use in expression within math block
        let stmts = parse("[x + 5]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(15.0));
    }

    #[test]
    fn test_eval_nested_math() {
        let mut eval = Evaluator::new();

        // Math block with nested operations
        let stmts = parse("[[3, 3, 3] * [2, 2, 2]]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();

        match result {
            Value::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0], Value::Float(6.0));
                assert_eq!(arr[1], Value::Float(6.0));
                assert_eq!(arr[2], Value::Float(6.0));
            }
            _ => panic!("Expected array value"),
        }
    }

    #[test]
    fn test_eval_comparison() {
        let mut eval = Evaluator::new();

        // Comparisons work with RuneOp outside math blocks
        let stmts = parse("5 > 3").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(true));

        let stmts = parse("2 = 2").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_eval_unary_minus() {
        let mut eval = Evaluator::new();

        // Unary minus in math block
        let stmts = parse("[-5]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(-5.0));
    }

    #[test]
    fn test_eval_comparison_operators() {
        let mut eval = Evaluator::new();

        // Test less than or equal
        let stmts = parse("3 <= 5").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(true));

        let stmts = parse("5 <= 5").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(true));

        let stmts = parse("7 <= 5").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(false));

        // Test greater than or equal
        let stmts = parse("5 >= 3").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(true));

        let stmts = parse("5 >= 5").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(true));

        let stmts = parse("3 >= 5").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(false));
    }
}

File: rune\hydron\hyperbolic.rs
===============================
//! Hyperbolic H8 Geometry Layer - Poincaré Ball Model
//!
//! Implements hyperbolic geometry in 8 dimensions using the Poincaré ball model.
//! Points lie inside the unit ball with hyperbolic distance metric.
//!
//! Key operations:
//! - Projection to Poincaré ball interior
//! - Hyperbolic distance using arcosh formula
//! - Möbius addition for hyperbolic translation
//! - Geodesic interpolation
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// Hyperbolic H8 layer using Poincaré ball model
pub struct HyperbolicLayer;

impl HyperbolicLayer {
    /// Project an E8 coordinate into the Poincaré ball interior
    /// Ensures ||x|| < 1 by normalization
    pub fn project(coords: &[f32; 8]) -> [f32; 8] {
        #[cfg(feature = "simd")]
        {
            use super::gf8::gf8_norm2_simd;
            let norm_sq = gf8_norm2_simd(coords);

            if norm_sq < 1e-8 {
                return [0.0; 8]; // Origin
            }

            let norm = norm_sq.sqrt();

            // Scale to fit in ball with margin
            let scale = if norm >= 0.95 { 0.95 / norm } else { 1.0 };

            coords.map(|x| x * scale)
        }
        #[cfg(not(feature = "simd"))]
        {
            let norm_sq: f32 = coords.iter().map(|x| x * x).sum();

            if norm_sq < 1e-8 {
                return [0.0; 8]; // Origin
            }

            let norm = norm_sq.sqrt();

            // Scale to fit in ball with margin
            let scale = if norm >= 0.95 { 0.95 / norm } else { 1.0 };

            coords.map(|x| x * scale)
        }
    }

    /// Compute hyperbolic distance in Poincaré ball
    /// d_H(x, y) = arcosh(1 + 2||x - y||² / ((1 - ||x||²)(1 - ||y||²)))
    pub fn distance(x: &[f32; 8], y: &[f32; 8]) -> f32 {
        #[cfg(feature = "simd")]
        {
            use super::gf8::{gf8_norm2_simd, gf8_sub_simd};
            let diff = gf8_sub_simd(x, y);
            let diff_sq = gf8_norm2_simd(&diff);
            let x_norm_sq = gf8_norm2_simd(x);
            let y_norm_sq = gf8_norm2_simd(y);

            // Hyperbolic distance formula
            let numerator = 2.0 * diff_sq;
            let denominator = (1.0 - x_norm_sq) * (1.0 - y_norm_sq);

            if denominator < 1e-8 {
                return f32::INFINITY; // Points on boundary
            }

            let ratio = 1.0 + numerator / denominator;
            ratio.max(1.0).acosh()
        }
        #[cfg(not(feature = "simd"))]
        {
            // Optimized scalar computation - avoid intermediate powi(2)
            let mut diff_sq = 0.0;
            let mut x_norm_sq = 0.0;
            let mut y_norm_sq = 0.0;

            for i in 0..8 {
                let diff = x[i] - y[i];
                diff_sq += diff * diff;
                x_norm_sq += x[i] * x[i];
                y_norm_sq += y[i] * y[i];
            }

            // Hyperbolic distance formula
            let numerator = 2.0 * diff_sq;
            let denominator = (1.0 - x_norm_sq) * (1.0 - y_norm_sq);

            if denominator < 1e-8 {
                return f32::INFINITY; // Points on boundary
            }

            let ratio = 1.0 + numerator / denominator;
            ratio.max(1.0).acosh()
        }
    }

    /// Möbius addition: a ⊕ b (hyperbolic translation)
    /// a ⊕ b = ((1 + 2⟨a,b⟩ + ||b||²)a + (1 - ||a||²)b) / (1 + 2⟨a,b⟩ + ||a||²||b||²)
    pub fn mobius_add(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
        #[cfg(feature = "simd")]
        let (a_norm_sq, b_norm_sq, dot_ab) = {
            use super::gf8::{gf8_dot_simd, gf8_norm2_simd};
            (gf8_norm2_simd(a), gf8_norm2_simd(b), gf8_dot_simd(a, b))
        };
        #[cfg(not(feature = "simd"))]
        let (a_norm_sq, b_norm_sq, dot_ab) = {
            let a_norm_sq: f32 = a.iter().map(|x| x * x).sum();
            let b_norm_sq: f32 = b.iter().map(|x| x * x).sum();
            let dot_ab: f32 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
            (a_norm_sq, b_norm_sq, dot_ab)
        };

        let numerator_a_coeff = 1.0 + 2.0 * dot_ab + b_norm_sq;
        let numerator_b_coeff = 1.0 - a_norm_sq;
        let denominator = 1.0 + 2.0 * dot_ab + a_norm_sq * b_norm_sq;

        if denominator.abs() < 1e-8 {
            return [0.0; 8]; // Degenerate case
        }

        let mut result = [0.0f32; 8];
        for i in 0..8 {
            result[i] = (numerator_a_coeff * a[i] + numerator_b_coeff * b[i]) / denominator;
        }

        result
    }

    /// Compute hyperbolic distance from origin
    pub fn norm(x: &[f32; 8]) -> f32 {
        let origin = [0.0f32; 8];
        Self::distance(&origin, x)
    }

    /// Interpolate along hyperbolic geodesic between two points
    /// Uses parameter t ∈ [0, 1]
    pub fn interpolate(x: &[f32; 8], y: &[f32; 8], t: f32) -> [f32; 8] {
        let t_clamped = t.clamp(0.0, 1.0);

        // Hyperbolic geodesic: use Möbius addition with scaling
        // First translate to make x the origin, interpolate, then translate back
        let neg_x = x.map(|xi| -xi);

        // Translate y to origin frame: y' = (-x) ⊕ y
        let y_translated = Self::mobius_add(&neg_x, y);

        // Scale y' by t
        let y_scaled = y_translated.map(|yi| yi * t_clamped);

        // Translate back: x ⊕ (t * y')
        Self::mobius_add(x, &y_scaled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_project() {
        // Test projection keeps points inside ball
        let coords = [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        let projected = HyperbolicLayer::project(&coords);

        let norm_sq: f32 = projected.iter().map(|x| x * x).sum();
        assert!(norm_sq < 1.0, "Projected point should be inside unit ball");
    }

    #[test]
    fn test_hyperbolic_distance() {
        // Distance from origin
        let x = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let origin = [0.0; 8];

        let dist = HyperbolicLayer::distance(&origin, &x);
        assert!(dist > 0.0);
        assert!(dist.is_finite());
    }

    #[test]
    fn test_hyperbolic_distance_symmetry() {
        let x = [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y = [0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let dist_xy = HyperbolicLayer::distance(&x, &y);
        let dist_yx = HyperbolicLayer::distance(&y, &x);

        assert!(
            (dist_xy - dist_yx).abs() < 1e-6,
            "Distance should be symmetric"
        );
    }

    #[test]
    fn test_mobius_add_identity() {
        let x = [0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let origin = [0.0; 8];

        // x ⊕ 0 = x
        let result = HyperbolicLayer::mobius_add(&x, &origin);
        for i in 0..8 {
            assert!((result[i] - x[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_hyperbolic_interpolate() {
        let x = [0.0; 8];
        let y = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // At t=0, should be x
        let interp_0 = HyperbolicLayer::interpolate(&x, &y, 0.0);
        for i in 0..8 {
            assert!((interp_0[i] - x[i]).abs() < 1e-6);
        }

        // At t=1, should be y
        let interp_1 = HyperbolicLayer::interpolate(&x, &y, 1.0);
        for i in 0..8 {
            assert!((interp_1[i] - y[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_hyperbolic_norm() {
        let x = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let norm = HyperbolicLayer::norm(&x);

        assert!(norm > 0.0);
        assert!(norm.is_finite());

        // Norm at origin should be 0
        let origin = [0.0; 8];
        let norm_origin = HyperbolicLayer::norm(&origin);
        assert!(norm_origin.abs() < 1e-6);
    }
}

File: rune\hydron\fisher.rs
===========================
//! Fisher Information Layer - Information Geometry
//!
//! Implements Fisher information geometry and statistical manifolds.
//! The Fisher information matrix measures the amount of information
//! that observable data carries about unknown parameters in a statistical model.
//!
//! Key operations:
//! - Fisher matrix computation from probability distributions
//! - Uncertainty quantification
//! - KL divergence for statistical distance
//! - Entropy calculations
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// Fisher information layer for information geometry
pub struct FisherLayer;

impl FisherLayer {
    /// Compute Fisher information matrix from resonance distribution
    /// F_ij = E[∂log(p)/∂θ_i * ∂log(p)/∂θ_j]
    /// Simplified: uses variance as proxy
    pub fn fisher_matrix(resonance: &[u32; 240]) -> [[f32; 8]; 8] {
        let mut fisher = [[0.0f32; 8]; 8];

        // Normalize resonance to probability distribution
        let total: u32 = resonance.iter().sum();
        if total == 0 {
            return fisher; // Zero matrix if no resonance
        }

        let total_f = total as f32;
        let probs: Vec<f32> = resonance.iter().map(|&r| r as f32 / total_f).collect();

        // Compute Fisher matrix elements using simplified covariance
        // Map 240 roots to 8D coordinates and compute covariance
        for (i, row) in fisher.iter_mut().enumerate() {
            for j in 0..8 {
                let mut sum = 0.0f32;

                // Use simplified mapping: each coordinate contributes to 30 roots
                let start = i * 30;
                let end = (start + 30).min(240);

                for p in probs.iter().take(end).skip(start).copied() {
                    if p > 1e-8 {
                        // Fisher information: 1/p (for single parameter)
                        sum += 1.0 / p;
                    }
                }

                row[j] = if i == j { sum / 30.0 } else { 0.0 };
            }
        }

        fisher
    }

    /// Compute uncertainty from Fisher matrix
    /// Uncertainty = 1 / sqrt(trace(F))
    pub fn uncertainty(fisher_matrix: &[[f32; 8]; 8]) -> f32 {
        let trace: f32 = (0..8).map(|i| fisher_matrix[i][i]).sum();

        if trace < 1e-8 {
            return f32::INFINITY; // Maximum uncertainty
        }

        1.0 / trace.sqrt()
    }

    /// Compute Kullback-Leibler divergence: KL(P || Q) = Σ P(i) log(P(i)/Q(i))
    /// Measures statistical distance between distributions
    pub fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
        if p.len() != q.len() {
            return f32::INFINITY; // Invalid comparison
        }

        let mut kl = 0.0f32;

        for i in 0..p.len() {
            if p[i] > 1e-8 && q[i] > 1e-8 {
                kl += p[i] * (p[i] / q[i]).ln();
            }
        }

        kl.max(0.0) // KL divergence is non-negative
    }

    /// Compute information metric from Fisher matrix
    /// Metric = log(1 + trace(F))
    pub fn information_metric(fisher_matrix: &[[f32; 8]; 8]) -> f32 {
        let trace: f32 = (0..8).map(|i| fisher_matrix[i][i]).sum();
        (1.0 + trace).ln()
    }

    /// Compute entropy from distribution
    /// H = -Σ p_i log(p_i)
    pub fn entropy(distribution: &[f32]) -> f32 {
        let sum: f32 = distribution.iter().sum();

        if sum < 1e-8 {
            return 0.0;
        }

        let normalized: Vec<f32> = distribution.iter().map(|x| x / sum).collect();

        normalized
            .iter()
            .filter(|&&p| p > 1e-8)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Compute Fisher information metric distance
    /// d_F(θ1, θ2)² ≈ (θ1 - θ2)ᵀ F (θ1 - θ2)
    pub fn fisher_distance(
        theta1: &[f32; 8],
        theta2: &[f32; 8],
        fisher_matrix: &[[f32; 8]; 8],
    ) -> f32 {
        let mut diff = [0.0f32; 8];
        for i in 0..8 {
            diff[i] = theta1[i] - theta2[i];
        }

        // Compute diff^T * F * diff
        let mut result = 0.0f32;
        for i in 0..8 {
            for j in 0..8 {
                result += diff[i] * fisher_matrix[i][j] * diff[j];
            }
        }

        result.sqrt().max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_matrix() {
        let mut resonance = [0u32; 240];
        // Create peaked distribution
        for i in 0..240 {
            resonance[i] = if i < 30 { 10 } else { 1 };
        }

        let fisher = FisherLayer::fisher_matrix(&resonance);

        // Fisher matrix should be diagonal and positive
        let trace: f32 = (0..8).map(|i| fisher[i][i]).sum();
        assert!(trace > 0.0, "Fisher matrix should have positive trace");
    }

    #[test]
    fn test_uncertainty() {
        let mut fisher = [[0.0f32; 8]; 8];
        for i in 0..8 {
            fisher[i][i] = 1.0;
        }

        let uncertainty = FisherLayer::uncertainty(&fisher);

        // Uncertainty should be 1/sqrt(8) ≈ 0.35
        assert!((uncertainty - 0.353).abs() < 0.01);
    }

    #[test]
    fn test_kl_divergence() {
        let p = [0.5, 0.5];
        let q = [0.5, 0.5];

        // KL(P || P) = 0
        let kl = FisherLayer::kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-6);

        // Different distributions
        let q2 = [0.8, 0.2];
        let kl2 = FisherLayer::kl_divergence(&p, &q2);
        assert!(kl2 > 0.0);
    }

    #[test]
    fn test_information_metric() {
        let mut fisher = [[0.0f32; 8]; 8];
        for i in 0..8 {
            fisher[i][i] = 2.0;
        }

        let metric = FisherLayer::information_metric(&fisher);

        // metric = ln(1 + 8*2) = ln(17) ≈ 2.83
        assert!((metric - 2.833).abs() < 0.01);
    }

    #[test]
    fn test_entropy() {
        // Uniform distribution has maximum entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let h_uniform = FisherLayer::entropy(&uniform);

        // Peaked distribution has lower entropy
        let peaked = vec![0.8, 0.1, 0.05, 0.05];
        let h_peaked = FisherLayer::entropy(&peaked);

        assert!(h_uniform > h_peaked);
    }

    #[test]
    fn test_fisher_distance() {
        let theta1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let theta2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let mut fisher = [[0.0f32; 8]; 8];
        for i in 0..8 {
            fisher[i][i] = 1.0;
        }

        let distance = FisherLayer::fisher_distance(&theta1, &theta2, &fisher);

        // Distance should be sqrt(2) ≈ 1.414
        assert!((distance - 1.414).abs() < 0.01);
    }
}

File: rune\hydron\gf8.rs
========================
/* src/rune/hydron/gf8.rs */
//! A foundational 8-dimensional geometric float gf8, inspired by E₈ lattice properties.
//!
//! # e8 Primitives – Gf8 Module
//!▫~•◦-----------------------------‣
//!
//! This module provides the `Gf8` type, a core numeric gf8 for the e8 ecosystem.
//! It is designed to replace standard floating-point numbers in contexts where geometric
//! stability, intrinsic normalization, and binary-addressable states are paramount.
//!
//! ### Key Capabilities
//! - **Geometric Representation:** `Gf8` represents a value as a normalized 8D vector on the unit sphere (S⁷).
//! - **Binary Encoding:** Provides a constructor from 8 bits that maps to a unique, stable direction in 8D space, enforcing an E₈-like even parity constraint.
//! - **Geometric Arithmetic:** All arithmetic operations (add, sub) are geometric, preserving the unit-norm constraint by re-projecting results onto the sphere.
//! - **Tensor-like API:** Implements `Deref` and a `Gf8Tensor` trait, allowing it to be used seamlessly as a small, fixed-size tensor.
//!
//! ### Architectural Notes
//! `Gf8` is the cornerstone of the e8 compute and data model. Its fixed dimensionality is a perfect
//! match for 256-bit SIMD registers (e.g., AVX), enabling highly efficient hardware acceleration.
//! It serves as the basis for E8B codes, E8DB keys, and the E8 LLM's numerical representation.
//!
//! ### Example
//! ```rust
//! use rune_format::rune::hydron::gf8::{Gf8, Gf8Tensor};
//!
//! // Create a Gf8 from a binary pattern (0b10101010)
//! let bits = [0, 1, 0, 1, 0, 1, 0, 1];
//! let a = Gf8::from_bits_even_parity(bits);
//!
//! // Create another Gf8 from a different pattern
//! let b = Gf8::from_scalar(-0.5);
//!
//! // Compute the dot product (cosine similarity)
//! let similarity = a.dot(b.coords());
//!
//! // `Gf8` can be treated like a slice
//! println!("Gf8 'a' has {} dimensions.", a.as_slice().len());
//! println!("Similarity between a and b: {}", similarity);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::ops::{Add, AddAssign, Deref, DerefMut, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::rune::hydron::{
    fisher::FisherLayer,
    hyperbolic::HyperbolicLayer,
    lorentzian::{LorentzianLayer, SpacetimePoint},
    quaternion::QuaternionOps,
    spherical::SphericalLayer,
    symplectic::SymplecticLayer,
};

/// A tiny tensor-like trait for GF8.
///
/// This provides an explicit contract for types that can be viewed as a slice of floats,
/// intended for use in generic, tensor-aware code.
pub trait Gf8Tensor {
    /// Returns the underlying data as an immutable slice.
    fn as_slice(&self) -> &[f32];
    /// Returns the underlying data as a mutable slice.
    fn as_mut_slice(&mut self) -> &mut [f32];
}

/// A GF8 (GeoFloat8), an 8-dimensional geometric float gf8.
///
/// It is internally represented by an array of 8 `f32`s, which is always
/// normalized to have a unit L2 norm (i.e., it lies on the surface of an
/// 8D hypersphere). This property provides intrinsic stability and makes it suitable
/// for representing directions, rotations, and normalized semantic states.
///
/// The only exception to the unit-norm rule is the zero vector, which has a norm of 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gf8 {
    coords: [f32; 8],
}

impl Gf8 {
    /// The zero vector, representing a neutral or null state.
    pub const ZERO: Self = Self { coords: [0.0; 8] };

    /// Constructs a `Gf8` from raw 8D coordinates, normalizing them to unit length.
    ///
    /// If the input vector has a magnitude of zero, the zero `Gf8` is returned.
    #[inline]
    pub fn new(coords: [f32; 8]) -> Self {
        let mut v = Self { coords };
        v.renormalize();
        v
    }

    /// Constructs a `Gf8` from raw 8D coordinates.
    ///
    /// This is an alias for [`Gf8::new`], provided for clarity when working in
    /// math-heavy code where "from_coords" more clearly expresses intent than "new".
    #[inline]
    pub fn from_coords(coords: [f32; 8]) -> Self {
        Self::new(coords)
    }

    /// Constructs a `Gf8` from 8 bits, mapping them to an E₈-like ±1 pattern.
    ///
    /// The mapping is `0 -> +1.0` and `1 -> -1.0`. To satisfy an E₈-like constraint,
    /// the number of `-1.0` entries is forced to be even by flipping the sign of
    /// the last coordinate if necessary. The resulting vector is then normalized
    /// to unit length.
    pub fn from_bits_even_parity(bits: [u8; 8]) -> Self {
        let mut coords = [0.0f32; 8];
        let mut neg_count = 0usize;

        for (i, &b) in bits.iter().enumerate() {
            if b == 0 {
                coords[i] = 1.0;
            } else {
                coords[i] = -1.0;
                neg_count += 1;
            }
        }

        if neg_count % 2 == 1 {
            // Flip the sign of the last coordinate to enforce even parity.
            coords[7] = -coords[7];
        }

        // Normalize the resulting vector to place it on the unit sphere.
        // A pure ±1 vector has a norm of sqrt(8).
        Self::new(coords)
    }

    /// Constructs a `Gf8` by embedding a scalar along the first axis.
    ///
    /// The resulting `Gf8` will be `[signum(x), 0.0, ..., 0.0]`. This provides a simple
    /// way to represent scalar magnitudes directionally.
    pub fn from_scalar(x: f32) -> Self {
        let mut coords = [0.0; 8];
        coords[0] = x;
        Self::new(coords)
    }

    /// Retrieves the raw coordinate data as a slice.
    #[inline]
    pub fn coords(&self) -> &[f32; 8] {
        &self.coords
    }

    /// Approximates a scalar value by projecting the `Gf8` onto the first axis.
    ///
    /// Since `Gf8` is a unit vector, this value will be in the range `[-1.0, 1.0]`.
    #[inline]
    pub fn to_scalar(&self) -> f32 {
        self.coords[0]
    }

    /// Computes the dot product with another 8D vector.
    ///
    /// For two unit vectors, this is equivalent to their cosine similarity.
    /// This method is backed by a runtime-dispatching SIMD implementation
    /// for maximum performance.
    #[inline(always)]
    pub fn dot(&self, other: &[f32; 8]) -> f32 {
        #[cfg(feature = "simd")]
        {
            self::simd::dot_product(self.coords, *other)
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut sum = 0.0f32;
            for i in 0..8 {
                sum += self.coords[i] * other[i];
            }
            sum
        }
    }

    /// Computes the squared L2 norm. For a valid `Gf8`, this is always `1.0` (or `0.0` for zero).
    #[inline]
    pub fn norm2(&self) -> f32 {
        #[cfg(feature = "simd")]
        {
            self::simd::norm2_scalar(&self.coords)
        }
        #[cfg(not(feature = "simd"))]
        {
            self.coords.iter().map(|x| x * x).sum::<f32>()
        }
    }

    /// Computes the L2 norm. For a valid `Gf8`, this is always `1.0` (or `0.0` for zero).
    #[inline]
    pub fn norm(&self) -> f32 {
        self.norm2().sqrt()
    }

    /// Re-normalizes the `Gf8` in-place to ensure it remains a unit vector.
    /// This is useful after performing arithmetic operations that may alter the magnitude.
    pub fn renormalize(&mut self) {
        let n2 = self.norm2();
        if n2 > 0.0 {
            let inv_norm = 1.0 / n2.sqrt();
            for x in &mut self.coords {
                *x *= inv_norm;
            }
        }
    }

    // --- Geometry Operations ---

    /// Spherical geometry: computes geodesic distance to another Gf8 (unit sphere).
    pub fn spherical_distance_to(&self, other: &Self) -> f32 {
        SphericalLayer::distance(self.coords(), other.coords())
    }

    /// Spherical geometry: spherical linear interpolation between two Gf8 values.
    pub fn spherical_slerp(&self, other: &Self, t: f32) -> Self {
        Self::new(SphericalLayer::slerp(self.coords(), other.coords(), t))
    }

    /// Spherical geometry: computes the antipodal (opposite) point on the sphere.
    pub fn spherical_antipodal(&self) -> Self {
        Self {
            coords: SphericalLayer::antipodal(self.coords()),
        }
    }

    /// Hyperbolic geometry: computes distance in the Poincaré ball model.
    /// Since Gf8 coords are on the unit sphere, we map them to the interior ball first.
    pub fn hyperbolic_distance_to(&self, other: &Self) -> f32 {
        // Map sphere coords to ball interior by scaling down
        let self_ball = self.coords.map(|x| x * 0.95);
        let other_ball = other.coords.map(|x| x * 0.95);
        HyperbolicLayer::distance(&self_ball, &other_ball)
    }

    /// Hyperbolic geometry: Möbius addition in Poincaré ball.
    pub fn hyperbolic_mobius_add(&self, other: &Self) -> Self {
        let self_ball = self.coords.map(|x| x * 0.95);
        let other_ball = other.coords.map(|x| x * 0.95);
        let result_ball = HyperbolicLayer::mobius_add(&self_ball, &other_ball);
        // Map back to sphere? Or return as ball coords?
        Self::new(result_ball)
    }

    /// Fisher information geometry: distance with Fisher information matrix.
    pub fn fisher_distance_to(&self, other: &Self, fisher_matrix: &[[f32; 8]; 8]) -> f32 {
        FisherLayer::fisher_distance(self.coords(), other.coords(), fisher_matrix)
    }

    /// Fisher information: uncertainty from Fisher matrix.
    pub fn fisher_uncertainty(fisher_matrix: &[[f32; 8]; 8]) -> f32 {
        FisherLayer::uncertainty(fisher_matrix)
    }

    /// Quaternion operations: convert Gf8 to quaternion.
    pub fn to_quaternion(&self) -> [f32; 4] {
        QuaternionOps::from_e8_spinor(self.coords())
    }

    /// Lorentzian geometry: check if other point is in past light cone.
    pub fn lorentzian_in_past_light_cone(&self, other: &Self) -> bool {
        let p1_coords: [f64; 8] = self.coords.map(|x| x as f64);
        let p2_coords: [f64; 8] = other.coords.map(|x| x as f64);
        let p1 = SpacetimePoint::new(p1_coords);
        let p2 = SpacetimePoint::new(p2_coords);

        // Create a temporary LorentzianLayer to check causality
        let layer = LorentzianLayer::new();
        layer.in_past_light_cone(&p1, &p2)
    }

    /// Symplectic geometry: compute Hamiltonian treating Gf8 as position, momentum as zero.
    pub fn symplectic_hamiltonian(&self) -> f32 {
        let zero_momentum = [0.0f32; 8];
        SymplecticLayer::new().hamiltonian(self.coords(), &zero_momentum)
    }
}

impl Gf8 {
    // --- Associated Geometry Functions ---

    /// Spherical geometry: compute Fréchet mean of multiple Gf8 values.
    pub fn spherical_mean(points: &[Self]) -> Self {
        let coords: Vec<[f32; 8]> = points.iter().map(|p| *p.coords()).collect();
        let mean_coords = SphericalLayer::mean(&coords);
        Self::new(mean_coords)
    }

    /// Fisher information: compute Fisher matrix from resonance data.
    pub fn fisher_matrix_from_resonance(resonance: &[u32; 240]) -> [[f32; 8]; 8] {
        FisherLayer::fisher_matrix(resonance)
    }

    /// Quaternion: SLERP between two quaternions derived from Gf8 values.
    pub fn quaternion_slerp_from_gf8(a: &Self, b: &Self, t: f32) -> Self {
        let qa = a.to_quaternion();
        let qb = b.to_quaternion();
        let slerped = QuaternionOps::slerp(&qa, &qb, t);
        // Map quaternion back to Gf8 somehow - this is a bit hacky
        // Use the first 4 quaternion components as coords, pad with 0s, normalize
        let mut coords = [0.0f32; 8];
        coords[0..4].copy_from_slice(&slerped);
        Self::new(coords)
    }
}

impl Gf8Tensor for Gf8 {
    #[inline]
    fn as_slice(&self) -> &[f32] {
        &self.coords
    }
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.coords
    }
}

impl Default for Gf8 {
    /// The default `Gf8` is the zero vector.
    fn default() -> Self {
        Self::ZERO
    }
}

impl Deref for Gf8 {
    type Target = [f32; 8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.coords
    }
}

impl DerefMut for Gf8 {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.coords
    }
}

/// Geometric addition: performs element-wise vector addition and then
/// re-normalizes the result, projecting it back onto the unit sphere.
impl Add for Gf8 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut coords = [0.0f32; 8];
        for (i, (&a, &b)) in self.coords.iter().zip(rhs.coords.iter()).enumerate() {
            coords[i] = a + b;
        }
        Self::new(coords)
    }
}

impl AddAssign for Gf8 {
    fn add_assign(&mut self, rhs: Self) {
        for (i, &v) in rhs.coords.iter().enumerate() {
            self.coords[i] += v;
        }
        self.renormalize();
    }
}

/// Geometric subtraction: performs element-wise vector subtraction and then
/// re-normalizes the result, projecting it back onto the unit sphere.
impl Sub for Gf8 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut coords = [0.0f32; 8];
        for (i, (&a, &b)) in self.coords.iter().zip(rhs.coords.iter()).enumerate() {
            coords[i] = a - b;
        }
        Self::new(coords)
    }
}

impl SubAssign for Gf8 {
    fn sub_assign(&mut self, rhs: Self) {
        for (i, &v) in rhs.coords.iter().enumerate() {
            self.coords[i] -= v;
        }
        self.renormalize();
    }
}

/// Scalar multiplication. The result is re-normalized, so this operation primarily
/// affects the vector's direction (flipping it if the scalar is negative).
impl Mul<f32> for Gf8 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        let mut coords = self.coords;
        for x in &mut coords {
            *x *= rhs;
        }
        Self::new(coords)
    }
}

impl MulAssign<f32> for Gf8 {
    fn mul_assign(&mut self, rhs: f32) {
        for x in &mut self.coords {
            *x *= rhs;
        }
        self.renormalize();
    }
}

/// Negation: flips the direction of the vector. The norm remains unchanged.
impl Neg for Gf8 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut coords = self.coords;
        for x in &mut coords {
            *x = -*x;
        }
        Self { coords }
    }
}

/// Embedded SIMD module
#[cfg(feature = "simd")]
mod simd {
    use crate::rune::hydron::intrinsics_for_f32_width;

    // Gate architecture-specific modules.
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64::*;

    /// Prints a summary of available SIMD capabilities for debugging.
    #[allow(dead_code)]
    pub fn print_simd_capabilities() {
        println!("--- SIMD Capabilities ---");
        #[cfg(target_arch = "x86_64")]
        {
            println!("Architecture: x86_64");
            println!("AVX enabled: {}", is_x86_feature_detected!("avx"));
            println!("AVX2 enabled: {}", is_x86_feature_detected!("avx2"));
            println!("FMA enabled: {}", is_x86_feature_detected!("fma"));
        }
        #[cfg(target_arch = "aarch64")]
        {
            println!("Architecture: aarch64");
            println!("NEON enabled: {}", is_aarch64_feature_detected!("neon"));
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            println!("Architecture: Not x86_64 or aarch64. Scalar fallback only.");
        }
        println!("-------------------------");
    }

    /// Returns a list of available 256-bit f32 intrinsic names for analysis.
    #[allow(dead_code)]
    pub fn get_available_f32_256_intrinsics() -> Vec<&'static str> {
        #[cfg(target_arch = "x86_64")]
        {
            return intrinsics_for_f32_width(256)
                .filter(|i| {
                    let tech = i.technology;
                    (tech.contains("AVX2") && is_x86_feature_detected!("avx2"))
                        || (tech.contains("AVX") && is_x86_feature_detected!("avx"))
                        || (tech.contains("FMA") && is_x86_feature_detected!("fma"))
                })
                .map(|i| i.name)
                .collect();
        }
        // Return an empty vector for non-x86 architectures.
        #[cfg(not(target_arch = "x86_64"))]
        {
            Vec::new()
        }
    }

    /// Performs SIMD-accelerated addition of two raw f32 arrays.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn gf8_add_simd(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe {
                    let va = _mm256_loadu_ps(a.as_ptr());
                    let vb = _mm256_loadu_ps(b.as_ptr());
                    let sum = _mm256_add_ps(va, vb);

                    let mut result = [0.0f32; 8];
                    _mm256_storeu_ps(result.as_mut_ptr(), sum);
                    return result;
                }
            }
        }
        // Fallback
        let mut result = [0.0f32; 8];
        for i in 0..8 {
            result[i] = a[i] + b[i];
        }
        result
    }

    /// Performs SIMD-accelerated subtraction of two raw f32 arrays.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn gf8_sub_simd(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe {
                    let va = _mm256_loadu_ps(a.as_ptr());
                    let vb = _mm256_loadu_ps(b.as_ptr());
                    let diff = _mm256_sub_ps(va, vb);

                    let mut result = [0.0f32; 8];
                    _mm256_storeu_ps(result.as_mut_ptr(), diff);
                    return result;
                }
            }
        }
        // Fallback
        let mut result = [0.0f32; 8];
        for i in 0..8 {
            result[i] = a[i] - b[i];
        }
        result
    }

    /// Computes the dot product of two raw f32 arrays using SIMD.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn gf8_dot_simd(a: &[f32; 8], b: &[f32; 8]) -> f32 {
        dot_product(*a, *b)
    }

    /// Computes the squared L2 norm of a raw f32 array using SIMD.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn gf8_norm2_simd(a: &[f32; 8]) -> f32 {
        dot_product(*a, *a)
    }

    /// Performs SIMD-accelerated in-place addition for raw f32 arrays: `dst[i] += src[i]`.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn gf8_add_inplace_slice_simd(dst: &mut [f32; 8], src: &[f32; 8]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe {
                    let vdst = _mm256_loadu_ps(dst.as_ptr());
                    let vsrc = _mm256_loadu_ps(src.as_ptr());
                    let sum = _mm256_add_ps(vdst, vsrc);
                    _mm256_storeu_ps(dst.as_mut_ptr(), sum);
                }
                return;
            }
        }
        // Fallback to scalar
        for i in 0..8 {
            dst[i] += src[i];
        }
    }

    /// SIMD-accelerated matrix-vector multiplication for raw arrays.
    #[cfg(feature = "simd")]
    pub fn gf8_matvec_simd(mat: &[[f32; 8]; 8], vec: &[f32; 8]) -> [f32; 8] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                return matvec_simd_avx(mat, vec);
            }
        }
        // Fallback to using dot product
        let mut result = [0.0f32; 8];
        for (i, row) in mat.iter().enumerate() {
            result[i] = dot_product(*row, *vec);
        }
        result
    }

    // --- Private Implementation Details ---

    /// The primary, runtime-dispatching dot product implementation.
    ///
    /// This function is the single source of truth for dot products. It checks for CPU
    /// features at runtime and calls the most optimal available kernel.
    #[inline]
    pub fn dot_product(a: [f32; 8], b: [f32; 8]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("fma") {
                // FMA is fastest on modern CPUs that support it (implies AVX/AVX2).
                return unsafe { dot_product_fma(a, b) };
            }
            if is_x86_feature_detected!("avx") {
                return unsafe { dot_product_avx(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("neon") {
                return unsafe { dot_product_neon(a, b) };
            }
        }

        // Scalar fallback for all other cases.
        dot_product_scalar(a, b)
    }

    /// Scalar dot product implementation (fallback).
    #[inline]
    fn dot_product_scalar(a: [f32; 8], b: [f32; 8]) -> f32 {
        let mut sum = 0.0;
        for i in 0..8 {
            sum += a[i] * b[i];
        }
        sum
    }

    /// NEON implementation for dot product on aarch64.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn dot_product_neon(a: [f32; 8], b: [f32; 8]) -> f32 {
        let a1 = vld1q_f32(a.as_ptr());
        let a2 = vld1q_f32(a.as_ptr().add(4));
        let b1 = vld1q_f32(b.as_ptr());
        let b2 = vld1q_f32(b.as_ptr().add(4));
        let acc1 = vmulq_f32(a1, b1);
        let acc2 = vmulq_f32(a2, b2);
        let sum = vaddq_f32(acc1, acc2);
        vaddvq_f32(sum)
    }

    /// AVX implementation for dot product on x86_64.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn dot_product_avx(a: [f32; 8], b: [f32; 8]) -> f32 {
        unsafe {
            let va = _mm256_loadu_ps(a.as_ptr());
            let vb = _mm256_loadu_ps(b.as_ptr());
            // Use the `_mm256_dp_ps` intrinsic for a combined dot product.
            // The 0xf1 mask means: multiply lanes 0-3, sum them, and place in lane 0;
            // multiply lanes 4-7, sum them, and place in lane 4.
            let prod = _mm256_dp_ps(va, vb, 0xf1);
            let lo = _mm256_castps256_ps128(prod); // Low 128 bits
            let hi = _mm256_extractf128_ps(prod, 1); // High 128 bits
            let sum = _mm_add_ss(lo, hi); // Add the two sums
            _mm_cvtss_f32(sum)
        }
    }

    /// AVX+FMA implementation for dot product on x86_64.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "fma")]
    #[inline]
    unsafe fn dot_product_fma(a: [f32; 8], b: [f32; 8]) -> f32 {
        unsafe {
            let va = _mm256_loadu_ps(a.as_ptr());
            let vb = _mm256_loadu_ps(b.as_ptr());
            // This is identical to the AVX version but allows the compiler to use FMA.
            // The `_mm256_dp_ps` is often the most efficient way to do this.
            let prod = _mm256_dp_ps(va, vb, 0xf1);
            let lo = _mm256_castps256_ps128(prod);
            let hi = _mm256_extractf128_ps(prod, 1);
            let sum = _mm_add_ss(lo, hi);
            _mm_cvtss_f32(sum)
        }
    }

    /// Scalar norm2 implementation for small arrays where SIMD overhead is not worth it.
    #[inline]
    pub fn norm2_scalar(a: &[f32; 8]) -> f32 {
        a.iter().map(|x| x * x).sum()
    }

    /// AVX implementation for matrix-vector multiplication.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    #[inline]
    #[allow(dead_code)]
    unsafe fn matvec_simd_avx(mat: &[[f32; 8]; 8], vec: &[f32; 8]) -> [f32; 8] {
        let mut result = [0.0f32; 8];
        let vvec = unsafe { _mm256_loadu_ps(vec.as_ptr()) };

        // Process each row
        for i in 0..8 {
            let row = &mat[i];
            let vrow = unsafe { _mm256_loadu_ps(row.as_ptr()) };
            let prod = _mm256_dp_ps(vrow, vvec, 0xf1); // Dot product with mask 0xf1
            let lo = _mm256_castps256_ps128(prod);
            let hi = _mm256_extractf128_ps(prod, 1);
            let sum = _mm_add_ss(lo, hi);
            result[i] = _mm_cvtss_f32(sum);
        }

        result
    }
}

// Re-export SIMD functions when feature is enabled
#[cfg(feature = "simd")]
pub use simd::{
    get_available_f32_256_intrinsics, gf8_add_inplace_slice_simd, gf8_add_simd, gf8_dot_simd,
    gf8_matvec_simd, gf8_norm2_simd, gf8_sub_simd, print_simd_capabilities,
};

File: rune\hydron\mod.rs
========================
//! Hydron - E8 Geometric Mathematics Engine
//!
//! Pure mathematical implementations of E8 lattice geometry with multi-geometric layers:
//! - Fisher information geometry (statistical manifolds)
//! - Symplectic T*E8 geometry (Hamiltonian dynamics)
//! - Hyperbolic H8 geometry (Poincaré ball model)
//! - Topological analysis (persistent homology)
//! - Lorentzian geometry (spacetime metrics)
//! - Quaternion algebra (rotations, SLERP)
//! - Spherical S7 geometry (unit sphere)
//!
//! All modules provide pure geometric operations. Application-specific extensions
//! (e.g., causal DAGs, event systems) are clearly separated.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod eval;
pub mod fisher;
pub mod gf8;
pub mod hyperbolic;
pub mod intrinsics;
pub mod lorentzian;
pub mod quaternion;
pub mod spherical;
pub mod symplectic;
pub mod topological;
pub mod values;

pub use eval::Evaluator;
pub use fisher::FisherLayer;
pub use gf8::{Gf8, Gf8Tensor};
pub use hyperbolic::HyperbolicLayer;
pub use intrinsics::intrinsics_for_f32_width;
pub use lorentzian::{
    CausalDAG, CausalNode, CausalRelation, EventType, LorentzianCausalLayer, LorentzianLayer,
    SpacetimePoint, Worldline,
};
pub use quaternion::QuaternionOps;
pub use spherical::SphericalLayer;
pub use symplectic::SymplecticLayer;
pub use topological::{PersistencePair, TopologicalLayer};
pub use values::{EvalContext, EvalError, Octonion, Value};

File: rune\hydron\quaternion.rs
===============================
//! Quaternion Operations for Phase Transitions
//!
//! Implements quaternion algebra for rotations and phase transitions in E8.
//! Quaternions are used for smooth interpolation (SLERP) and representing
//! rotational symmetries in the E8 lattice.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// Quaternion operations
pub struct QuaternionOps;

impl QuaternionOps {
    /// Normalize a quaternion to unit length
    pub fn normalize(q: &[f32; 4]) -> [f32; 4] {
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        if norm < 1e-8 {
            [1.0, 0.0, 0.0, 0.0] // Identity quaternion
        } else {
            [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
        }
    }

    /// Multiply two quaternions (non-commutative)
    /// q1 * q2 = [w1*w2 - v1·v2, w1*v2 + w2*v1 + v1×v2]
    pub fn multiply(q1: &[f32; 4], q2: &[f32; 4]) -> [f32; 4] {
        let w1 = q1[0];
        let v1 = [q1[1], q1[2], q1[3]];
        let w2 = q2[0];
        let v2 = [q2[1], q2[2], q2[3]];

        // Scalar part: w1*w2 - v1·v2
        let w = w1 * w2 - (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]);

        // Vector part: w1*v2 + w2*v1 + v1×v2
        let i = w1 * v2[0] + w2 * v1[0] + (v1[1] * v2[2] - v1[2] * v2[1]);
        let j = w1 * v2[1] + w2 * v1[1] + (v1[2] * v2[0] - v1[0] * v2[2]);
        let k = w1 * v2[2] + w2 * v1[2] + (v1[0] * v2[1] - v1[1] * v2[0]);

        [w, i, j, k]
    }

    /// Compute quaternion conjugate: q* = [w, -v]
    pub fn conjugate(q: &[f32; 4]) -> [f32; 4] {
        [q[0], -q[1], -q[2], -q[3]]
    }

    /// Compute quaternion inverse: q⁻¹ = q* / |q|²
    pub fn inverse(q: &[f32; 4]) -> [f32; 4] {
        let norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        if norm_sq < 1e-8 {
            [0.0, 0.0, 0.0, 0.0] // Undefined
        } else {
            let conj = Self::conjugate(q);
            [
                conj[0] / norm_sq,
                conj[1] / norm_sq,
                conj[2] / norm_sq,
                conj[3] / norm_sq,
            ]
        }
    }

    /// Spherical linear interpolation (SLERP)
    /// Smoothly interpolate between q1 and q2 by parameter t ∈ [0, 1]
    pub fn slerp(q1: &[f32; 4], q2: &[f32; 4], t: f32) -> [f32; 4] {
        let q1_norm = Self::normalize(q1);
        let mut q2_norm = Self::normalize(q2);

        // Compute dot product
        let mut dot = q1_norm[0] * q2_norm[0]
            + q1_norm[1] * q2_norm[1]
            + q1_norm[2] * q2_norm[2]
            + q1_norm[3] * q2_norm[3];

        // If dot is negative, negate one quaternion to take shorter path
        if dot < 0.0 {
            q2_norm = [-q2_norm[0], -q2_norm[1], -q2_norm[2], -q2_norm[3]];
            dot = -dot;
        }

        // Clamp dot to [-1, 1] for numerical stability
        dot = dot.clamp(-1.0, 1.0);

        // If quaternions are very close, use linear interpolation
        if dot > 0.9995 {
            let result = [
                q1_norm[0] + t * (q2_norm[0] - q1_norm[0]),
                q1_norm[1] + t * (q2_norm[1] - q1_norm[1]),
                q1_norm[2] + t * (q2_norm[2] - q1_norm[2]),
                q1_norm[3] + t * (q2_norm[3] - q1_norm[3]),
            ];
            return Self::normalize(&result);
        }

        // Compute angle between quaternions
        let theta = dot.acos();
        let sin_theta = theta.sin();

        // SLERP formula: (sin((1-t)*θ) * q1 + sin(t*θ) * q2) / sin(θ)
        let scale1 = ((1.0 - t) * theta).sin() / sin_theta;
        let scale2 = (t * theta).sin() / sin_theta;

        [
            scale1 * q1_norm[0] + scale2 * q2_norm[0],
            scale1 * q1_norm[1] + scale2 * q2_norm[1],
            scale1 * q1_norm[2] + scale2 * q2_norm[2],
            scale1 * q1_norm[3] + scale2 * q2_norm[3],
        ]
    }

    /// Create quaternion from axis-angle representation
    /// q = [cos(θ/2), sin(θ/2) * axis]
    pub fn from_axis_angle(axis: &[f32; 3], angle: f32) -> [f32; 4] {
        let half_angle = angle * 0.5;
        let sin_half = half_angle.sin();

        // Normalize axis
        let axis_norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if axis_norm < 1e-8 {
            return [1.0, 0.0, 0.0, 0.0]; // Identity
        }

        let ax = axis[0] / axis_norm;
        let ay = axis[1] / axis_norm;
        let az = axis[2] / axis_norm;

        [
            half_angle.cos(),
            sin_half * ax,
            sin_half * ay,
            sin_half * az,
        ]
    }

    /// Rotate a 3D vector by a quaternion
    /// v' = q * v * q⁻¹ (treating v as quaternion [0, v])
    pub fn rotate_vector(q: &[f32; 4], v: &[f32; 3]) -> [f32; 3] {
        let q_norm = Self::normalize(q);
        let v_quat = [0.0, v[0], v[1], v[2]];

        // Compute q * v
        let qv = Self::multiply(&q_norm, &v_quat);

        // Compute (q * v) * q⁻¹
        let q_inv = Self::conjugate(&q_norm); // For unit quaternions, conjugate = inverse
        let result = Self::multiply(&qv, &q_inv);

        [result[1], result[2], result[3]]
    }

    /// Quaternion dot product
    pub fn dot(q1: &[f32; 4], q2: &[f32; 4]) -> f32 {
        q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]
    }

    /// Convert E8 spinor root (indices 112-239) to quaternion
    /// Uses simplified mapping from 8D E8 coordinates to 4D quaternion
    pub fn from_e8_spinor(e8_coords: &[f32; 8]) -> [f32; 4] {
        // Simple projection: map 8D to 4D by averaging pairs
        let w = (e8_coords[0] + e8_coords[1]) * 0.5;
        let i = (e8_coords[2] + e8_coords[3]) * 0.5;
        let j = (e8_coords[4] + e8_coords[5]) * 0.5;
        let k = (e8_coords[6] + e8_coords[7]) * 0.5;

        Self::normalize(&[w, i, j, k])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternion_normalize() {
        let q = [1.0, 1.0, 1.0, 1.0];
        let q_norm = QuaternionOps::normalize(&q);
        let norm = (q_norm[0] * q_norm[0]
            + q_norm[1] * q_norm[1]
            + q_norm[2] * q_norm[2]
            + q_norm[3] * q_norm[3])
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quaternion_multiply() {
        let q1 = [1.0, 0.0, 0.0, 0.0]; // Identity
        let q2 = [0.0, 1.0, 0.0, 0.0]; // i
        let result = QuaternionOps::multiply(&q1, &q2);
        assert_eq!(result, [0.0, 1.0, 0.0, 0.0]);

        // i * i = -1
        let i = [0.0, 1.0, 0.0, 0.0];
        let result = QuaternionOps::multiply(&i, &i);
        assert!((result[0] + 1.0).abs() < 1e-6); // Scalar part should be -1
    }

    #[test]
    fn test_quaternion_conjugate() {
        let q = [1.0, 2.0, 3.0, 4.0];
        let conj = QuaternionOps::conjugate(&q);
        assert_eq!(conj, [1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn test_quaternion_slerp() {
        let q1 = [1.0, 0.0, 0.0, 0.0]; // Identity
        let q2 = [0.0, 1.0, 0.0, 0.0]; // 180° rotation around x-axis

        let mid = QuaternionOps::slerp(&q1, &q2, 0.5);
        // Should be halfway rotation
        let expected = QuaternionOps::normalize(&[0.5, 0.5, 0.0, 0.0]);
        for i in 0..4 {
            assert!((mid[i] - expected[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_quaternion_from_axis_angle() {
        use std::f32::consts::PI;

        // 90° rotation around z-axis
        let q = QuaternionOps::from_axis_angle(&[0.0, 0.0, 1.0], PI / 2.0);

        // Rotate vector [1, 0, 0]
        let v = [1.0, 0.0, 0.0];
        let rotated = QuaternionOps::rotate_vector(&q, &v);

        // Should rotate to [0, 1, 0]
        assert!((rotated[0] - 0.0).abs() < 1e-5);
        assert!((rotated[1] - 1.0).abs() < 1e-5);
        assert!((rotated[2] - 0.0).abs() < 1e-5);
    }
}

File: rune\hydron\spherical.rs
==============================
//! Spherical S7 Geometry Layer - Unit 7-Sphere
//!
//! Implements spherical geometry on the unit 7-sphere in 8-dimensional space.
//! All points satisfy the constraint ||x|| = 1.
//!
//! Key operations:
//! - Projection to unit sphere
//! - Geodesic distance using arccos
//! - SLERP (Spherical Linear Interpolation)
//! - Antipodal points and mean computation
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// Spherical S7 layer - unit 7-sphere in 8D
pub struct SphericalLayer;

impl SphericalLayer {
    /// Project coordinates onto the unit 7-sphere (||x|| = 1)
    pub fn project(coords: &[f32; 8]) -> [f32; 8] {
        let norm_sq: f32 = coords.iter().map(|x| x * x).sum();

        if norm_sq < 1e-8 {
            // If at origin, project to arbitrary point on sphere
            let mut result = [0.0f32; 8];
            result[0] = 1.0;
            return result;
        }

        let norm = norm_sq.sqrt();
        coords.map(|x| x / norm)
    }

    /// Compute geodesic distance on S7 using arccos
    /// d(x, y) = arccos(x · y) for unit vectors
    pub fn distance(x: &[f32; 8], y: &[f32; 8]) -> f32 {
        #[cfg(feature = "simd")]
        {
            use super::gf8::gf8_dot_simd;
            let dot = gf8_dot_simd(x, y).clamp(-1.0, 1.0);
            dot.acos()
        }
        #[cfg(not(feature = "simd"))]
        {
            let dot: f32 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
            let dot_clamped = dot.clamp(-1.0, 1.0);
            dot_clamped.acos()
        }
    }

    /// Spherical linear interpolation (SLERP)
    /// Smoothly interpolate along great circle between x and y
    /// t ∈ [0, 1]
    pub fn slerp(x: &[f32; 8], y: &[f32; 8], t: f32) -> [f32; 8] {
        let t_clamped = t.clamp(0.0, 1.0);

        // Compute angle between vectors
        #[cfg(feature = "simd")]
        let dot = {
            use super::gf8::gf8_dot_simd;
            gf8_dot_simd(x, y)
        };
        #[cfg(not(feature = "simd"))]
        let dot: f32 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();

        let dot_clamped = dot.clamp(-1.0, 1.0);

        // If vectors are very close, use linear interpolation
        if dot_clamped > 0.9995 {
            let mut result = [0.0f32; 8];
            for i in 0..8 {
                result[i] = x[i] + t_clamped * (y[i] - x[i]);
            }
            return Self::project(&result); // Renormalize
        }

        let theta = dot_clamped.acos();
        let sin_theta = theta.sin();

        if sin_theta.abs() < 1e-8 {
            return *x; // Degenerate case
        }

        // SLERP formula: (sin((1-t)*θ) * x + sin(t*θ) * y) / sin(θ)
        let scale1 = ((1.0 - t_clamped) * theta).sin() / sin_theta;
        let scale2 = (t_clamped * theta).sin() / sin_theta;

        let mut result = [0.0f32; 8];
        for i in 0..8 {
            result[i] = scale1 * x[i] + scale2 * y[i];
        }

        result
    }

    /// Compute normalized entropy from probability distribution
    /// Returns value in [0, 1] where 0 = uniform distribution, 1 = concentrated
    /// Optimized to avoid vector allocation and compute in single pass
    pub fn normalized_entropy(distribution: &[f32]) -> f32 {
        let sum: f32 = distribution.iter().sum();

        if sum < 1e-8 {
            return 0.0; // No information
        }

        let n = distribution.len() as f32;
        if n < 1e-8 {
            return 0.0;
        }

        // Compute entropy in single pass without intermediate vector allocation
        let mut entropy = 0.0;
        for &x in distribution {
            if x > 1e-8 {
                let p = x / sum;
                entropy -= p * p.ln();
            }
        }

        // Max entropy for n elements is ln(n)
        let max_entropy = n.ln();

        // Return normalized entropy (0 = uniform, 1 = concentrated)
        1.0 - (entropy / max_entropy)
    }

    /// Get antipodal point (opposite point on sphere)
    pub fn antipodal(x: &[f32; 8]) -> [f32; 8] {
        x.map(|xi| -xi)
    }

    /// Compute Fréchet mean (average on sphere) for multiple points
    /// Uses iterative optimization
    pub fn mean(points: &[[f32; 8]]) -> [f32; 8] {
        if points.is_empty() {
            return Self::project(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }

        if points.len() == 1 {
            return points[0];
        }

        // Start with Euclidean mean, then project
        let mut mean = [0.0f32; 8];
        for point in points {
            for i in 0..8 {
                mean[i] += point[i];
            }
        }

        Self::project(&mean)
    }

    /// Compute geodesic from point in direction (tangent vector)
    /// Returns point at distance t along geodesic
    pub fn geodesic(point: &[f32; 8], direction: &[f32; 8], t: f32) -> [f32; 8] {
        // Project direction to tangent space (remove component parallel to point)
        let dot: f32 = point
            .iter()
            .zip(direction.iter())
            .map(|(pi, di)| pi * di)
            .sum();

        let mut tangent = [0.0f32; 8];
        for i in 0..8 {
            tangent[i] = direction[i] - dot * point[i];
        }

        // Normalize tangent
        let tangent_norm_sq: f32 = tangent.iter().map(|x| x * x).sum();
        if tangent_norm_sq < 1e-8 {
            return *point; // No movement
        }

        let tangent_norm = tangent_norm_sq.sqrt();
        for t in &mut tangent {
            *t /= tangent_norm;
        }

        // Geodesic: point * cos(t) + tangent * sin(t)
        let mut result = [0.0f32; 8];
        let cos_t = t.cos();
        let sin_t = t.sin();
        for i in 0..8 {
            result[i] = point[i] * cos_t + tangent[i] * sin_t;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spherical_project() {
        let coords = [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        let projected = SphericalLayer::project(&coords);

        // Check that result is on unit sphere
        let norm_sq: f32 = projected.iter().map(|x| x * x).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-6,
            "Point should be on unit sphere"
        );
    }

    #[test]
    fn test_spherical_distance() {
        let x = SphericalLayer::project(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = SphericalLayer::project(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let dist = SphericalLayer::distance(&x, &y);

        // Should be π/2 (90 degrees)
        use std::f32::consts::FRAC_PI_2;
        assert!((dist - FRAC_PI_2).abs() < 1e-5);
    }

    #[test]
    fn test_spherical_slerp() {
        let x = SphericalLayer::project(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = SphericalLayer::project(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // At t=0, should be x
        let interp_0 = SphericalLayer::slerp(&x, &y, 0.0);
        for i in 0..8 {
            assert!((interp_0[i] - x[i]).abs() < 1e-6);
        }

        // At t=1, should be y
        let interp_1 = SphericalLayer::slerp(&x, &y, 1.0);
        for i in 0..8 {
            assert!((interp_1[i] - y[i]).abs() < 1e-5);
        }

        // Midpoint should be on unit sphere
        let mid = SphericalLayer::slerp(&x, &y, 0.5);
        let norm_sq: f32 = mid.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_spherical_normalized_entropy() {
        // Uniform distribution has high entropy (returns ~0)
        let uniform = vec![1.0; 10];
        let entropy_uniform = SphericalLayer::normalized_entropy(&uniform);
        assert!(entropy_uniform < 0.1);

        // Peaked distribution has low entropy (returns ~1)
        let peaked = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        let entropy_peaked = SphericalLayer::normalized_entropy(&peaked);
        assert!(entropy_peaked > 0.8);
    }

    #[test]
    fn test_spherical_antipodal() {
        let x = SphericalLayer::project(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let antipodal = SphericalLayer::antipodal(&x);

        // Distance should be π (180 degrees)
        use std::f32::consts::PI;
        let dist = SphericalLayer::distance(&x, &antipodal);
        assert!((dist - PI).abs() < 1e-5);
    }

    #[test]
    fn test_spherical_mean() {
        let x = SphericalLayer::project(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = SphericalLayer::project(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let mean = SphericalLayer::mean(&[x, y]);

        // Mean should be on unit sphere
        let norm_sq: f32 = mean.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6);
    }
}

File: rune\hydron\symplectic.rs
===============================
//! Symplectic T*E8 Layer - Hamiltonian Dynamics
//!
//! Implements symplectic phase space over E8 for Hamiltonian dynamics.
//! Phase space has 16 dimensions: 8 positions (q) + 8 momenta (p).
//!
//! Key operations:
//! - Hamiltonian computation (H = T + V)
//! - Symplectic evolution (Velocity Verlet integrator)
//! - Möbius kicks and drifts
//! - Poisson brackets
//! - Phase space conservation
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// Symplectic phase space layer for Hamiltonian dynamics
pub struct SymplecticLayer {
    /// Symplectic 2-form ω (16×16 for 8 positions + 8 momenta)
    /// Standard form: ω[i][i+8] = 1, ω[i+8][i] = -1
    pub omega: [[f32; 16]; 16],
}

impl SymplecticLayer {
    /// Create new symplectic layer with standard symplectic form
    pub fn new() -> Self {
        let mut omega = [[0.0f32; 16]; 16];

        // Standard symplectic form: ω = Σ dp_i ∧ dq_i
        for i in 0..8 {
            omega[i][i + 8] = 1.0; // dq_i ∧ dp_i = 1
            omega[i + 8][i] = -1.0; // dp_i ∧ dq_i = -1
        }

        Self { omega }
    }

    /// Compute Hamiltonian: H = kinetic + potential
    /// H = ½ Σ p_i² + V(q)
    /// where V(q) is a simple harmonic potential
    pub fn hamiltonian(&self, _q: &[f32; 8], p: &[f32; 8]) -> f32 {
        // Kinetic energy: T = ½ Σ p_i² (use SIMD norm2)
        #[cfg(feature = "simd")]
        let kinetic = {
            use super::gf8::gf8_norm2_simd;
            gf8_norm2_simd(p) * 0.5
        };
        #[cfg(not(feature = "simd"))]
        let kinetic: f32 = p.iter().map(|&pi| pi * pi).sum::<f32>() * 0.5;

        // Potential energy: V = ½ k Σ q_i² (harmonic oscillator)
        let k = 0.1;
        #[cfg(feature = "simd")]
        let potential = {
            use super::gf8::gf8_norm2_simd;
            gf8_norm2_simd(_q) * 0.5 * k
        };
        #[cfg(not(feature = "simd"))]
        let potential: f32 = _q.iter().map(|&qi| qi * qi).sum::<f32>() * 0.5 * k;

        kinetic + potential
    }

    /// Compute force F = -∂V/∂q for harmonic potential
    fn compute_force(&self, q: &[f32; 8]) -> [f32; 8] {
        let k = 0.1;
        q.map(|qi| -k * qi)
    }

    /// Evolve system using Velocity Verlet integrator (symplectic)
    /// Preserves phase space volume and energy
    pub fn evolve(&self, q: &mut [f32; 8], p: &mut [f32; 8], dt: f32) {
        #[cfg(feature = "simd")]
        {
            use super::gf8::gf8_add_inplace_slice_simd;
            // Half-step momentum update: p(t+dt/2) = p(t) + (dt/2) * F(t)
            let force = self.compute_force(q);
            let scaled_force = force.map(|fi| 0.5 * dt * fi);
            gf8_add_inplace_slice_simd(p, &scaled_force);

            // Full-step position update: q(t+dt) = q(t) + dt * p(t+dt/2)
            let scaled_p = p.map(|pi| dt * pi);
            gf8_add_inplace_slice_simd(q, &scaled_p);

            // Recompute force at new position
            let force_new = self.compute_force(q);

            // Complete momentum update: p(t+dt) = p(t+dt/2) + (dt/2) * F(t+dt)
            let scaled_force_new = force_new.map(|fi| 0.5 * dt * fi);
            gf8_add_inplace_slice_simd(p, &scaled_force_new);
        }
        #[cfg(not(feature = "simd"))]
        {
            // Half-step momentum update: p(t+dt/2) = p(t) + (dt/2) * F(t)
            let force = self.compute_force(q);
            for i in 0..8 {
                p[i] += 0.5 * dt * force[i];
            }

            // Full-step position update: q(t+dt) = q(t) + dt * p(t+dt/2)
            for i in 0..8 {
                q[i] += dt * p[i];
            }

            // Recompute force at new position
            let force_new = self.compute_force(q);

            // Complete momentum update: p(t+dt) = p(t+dt/2) + (dt/2) * F(t+dt)
            for i in 0..8 {
                p[i] += 0.5 * dt * force_new[i];
            }
        }
    }

    /// Apply symplectic kick (instantaneous momentum change)
    pub fn kick(&self, p: &mut [f32; 8], force: &[f32; 8], dt: f32) {
        #[cfg(feature = "simd")]
        {
            use super::gf8::gf8_add_inplace_slice_simd;
            let scaled_force = force.map(|fi| fi * dt);
            gf8_add_inplace_slice_simd(p, &scaled_force);
        }
        #[cfg(not(feature = "simd"))]
        {
            for i in 0..8 {
                p[i] += force[i] * dt;
            }
        }
    }

    /// Apply symplectic drift (position update from momentum)
    pub fn drift(&self, q: &mut [f32; 8], p: &[f32; 8], dt: f32) {
        #[cfg(feature = "simd")]
        {
            use super::gf8::gf8_add_inplace_slice_simd;
            let scaled_p = p.map(|pi| pi * dt);
            gf8_add_inplace_slice_simd(q, &scaled_p);
        }
        #[cfg(not(feature = "simd"))]
        {
            for i in 0..8 {
                q[i] += p[i] * dt;
            }
        }
    }

    /// Convert position and momentum to phase space coordinates
    pub fn to_phase_space(&self, q: &[f32; 8], p: &[f32; 8]) -> [f32; 16] {
        let mut phase = [0.0f32; 16];
        phase[..8].copy_from_slice(q);
        phase[8..].copy_from_slice(p);
        phase
    }

    /// Extract position and momentum from phase space coordinates
    pub fn from_phase_space(&self, phase: &[f32; 16]) -> ([f32; 8], [f32; 8]) {
        let mut q = [0.0f32; 8];
        let mut p = [0.0f32; 8];
        q.copy_from_slice(&phase[..8]);
        p.copy_from_slice(&phase[8..]);
        (q, p)
    }

    /// Compute Poisson bracket {f, g} for coordinate functions
    /// {q_i, p_i} = 1, {p_i, q_i} = -1, others = 0
    pub fn poisson_bracket(&self, i: usize, j: usize) -> f32 {
        if (0..8).contains(&i) && (8..16).contains(&j) && i + 8 == j {
            1.0 // {q_i, p_i} = 1
        } else if (8..16).contains(&i) && (0..8).contains(&j) && i - 8 == j {
            -1.0 // {p_i, q_i} = -1
        } else {
            0.0 // All other brackets vanish
        }
    }

    /// Verify energy conservation (approximately)
    pub fn verify_energy_conservation(
        &self,
        before: &([f32; 8], [f32; 8]),
        after: &([f32; 8], [f32; 8]),
    ) -> bool {
        let h_before = self.hamiltonian(&before.0, &before.1);
        let h_after = self.hamiltonian(&after.0, &after.1);

        let relative_error = (h_before - h_after).abs() / h_before.abs().max(1e-8);
        relative_error < 0.1 // Allow 10% error
    }
}

impl Default for SymplecticLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symplectic_form() {
        let sym = SymplecticLayer::new();

        // Check antisymmetry
        for i in 0..16 {
            for j in 0..16 {
                assert_eq!(sym.omega[i][j], -sym.omega[j][i]);
            }
        }

        // Check structure
        for i in 0..8 {
            assert_eq!(sym.omega[i][i + 8], 1.0);
            assert_eq!(sym.omega[i + 8][i], -1.0);
        }
    }

    #[test]
    fn test_hamiltonian() {
        let sym = SymplecticLayer::new();

        let q = [0.0; 8];
        let p = [1.0; 8];

        let h = sym.hamiltonian(&q, &p);
        // H = ½ * 8 * 1.0 = 4.0 (kinetic only)
        assert!((h - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_phase_space_conversion() {
        let sym = SymplecticLayer::new();

        let q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let phase = sym.to_phase_space(&q, &p);
        let (q_back, p_back) = sym.from_phase_space(&phase);

        for i in 0..8 {
            assert!((q[i] - q_back[i]).abs() < 1e-6);
            assert!((p[i] - p_back[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_poisson_bracket() {
        let sym = SymplecticLayer::new();

        // {q_0, p_0} = 1
        assert_eq!(sym.poisson_bracket(0, 8), 1.0);

        // {p_0, q_0} = -1
        assert_eq!(sym.poisson_bracket(8, 0), -1.0);

        // {q_0, q_1} = 0
        assert_eq!(sym.poisson_bracket(0, 1), 0.0);

        // {p_0, p_1} = 0
        assert_eq!(sym.poisson_bracket(8, 9), 0.0);
    }

    #[test]
    fn test_symplectic_evolution() {
        let sym = SymplecticLayer::new();

        let mut q = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut p = [0.0; 8];

        let before = (q, p);

        // Evolve for small time step
        sym.evolve(&mut q, &mut p, 0.1);

        let after = (q, p);

        // Energy should be approximately conserved
        assert!(sym.verify_energy_conservation(&before, &after));
    }

    #[test]
    fn test_kick_and_drift() {
        let sym = SymplecticLayer::new();

        let mut q = [0.0; 8];
        let mut p = [1.0; 8];

        // Apply drift: q = q + p * dt
        sym.drift(&mut q, &p, 0.1);
        for i in 0..8 {
            assert!((q[i] - 0.1).abs() < 1e-6);
        }

        // Apply kick: p = p + F * dt
        let force = [1.0; 8];
        sym.kick(&mut p, &force, 0.1);
        for i in 0..8 {
            assert!((p[i] - 1.1).abs() < 1e-6);
        }
    }
}

File: rune\hydron\topological.rs
================================
//! Topological Analysis Layer - Persistent Homology
//!
//! Implements persistent homology for analyzing topological features in point clouds.
//! Computes Betti numbers (β₀, β₁, β₂) representing:
//! - β₀: Connected components
//! - β₁: Loops/cycles
//! - β₂: Voids/cavities
//!
//! Topological features are invariant under continuous deformation,
//! making them robust descriptors of data structure.
//!
//! Key operations:
//! - Topological signature generation
//! - Persistence diagram computation
//! - Filtration-based analysis
//! - Betti number tracking
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashSet;

/// Persistence diagram entry: (birth, death) for a topological feature
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersistencePair {
    pub birth: f32,
    pub death: f32,
    pub dimension: usize,
}

impl PersistencePair {
    /// Lifetime of the topological feature
    pub fn persistence(&self) -> f32 {
        self.death - self.birth
    }
}

/// Topological layer for persistent homology
pub struct TopologicalLayer {
    /// Persistence diagrams by dimension (0, 1, 2)
    pub diagrams: [Vec<PersistencePair>; 3],

    /// Current Betti numbers [β₀, β₁, β₂]
    pub betti: [u32; 3],

    /// Point cloud data for simplicial complex construction
    points: Vec<[f32; 8]>,
}

impl TopologicalLayer {
    /// Create new topological layer
    pub fn new() -> Self {
        Self {
            diagrams: [Vec::new(), Vec::new(), Vec::new()],
            betti: [1, 0, 0], // Start with single connected component
            points: Vec::new(),
        }
    }

    /// Add point to point cloud
    pub fn add_point(&mut self, point: [f32; 8]) {
        self.points.push(point);
    }

    /// Clear all points
    pub fn clear_points(&mut self) {
        self.points.clear();
    }

    /// Compute distance matrix for point cloud
    fn distance_matrix(&self) -> Vec<Vec<f32>> {
        let n = self.points.len();
        let mut distances = vec![vec![0.0f32; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = euclidean_distance(&self.points[i], &self.points[j]);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        distances
    }

    /// Build Vietoris-Rips filtration up to given radius
    /// Returns edges that appear at each threshold
    fn vietoris_rips_filtration(
        &self,
        max_radius: f32,
        steps: usize,
    ) -> Vec<(f32, Vec<(usize, usize)>)> {
        let n = self.points.len();
        if n < 2 {
            return Vec::new();
        }

        let distances = self.distance_matrix();
        let mut filtration = Vec::new();

        let step_size = max_radius / steps as f32;

        for step in 0..=steps {
            let threshold = step as f32 * step_size;
            let mut edges = Vec::new();

            for i in 0..n {
                for j in (i + 1)..n {
                    if distances[i][j] <= threshold {
                        edges.push((i, j));
                    }
                }
            }

            filtration.push((threshold, edges));
        }

        filtration
    }

    /// Compute Betti numbers using Union-Find for β₀
    pub fn compute_betti_numbers(&mut self, max_radius: f32, steps: usize) {
        let n = self.points.len();

        if n == 0 {
            self.betti = [0, 0, 0];
            return;
        }

        let filtration = self.vietoris_rips_filtration(max_radius, steps);

        // Initialize β₀ = number of points (all disconnected)
        let mut uf = UnionFind::new(n);
        let mut beta0_history = Vec::new();

        // Track β₀ through filtration
        for (threshold, edges) in &filtration {
            for &(i, j) in edges {
                uf.union(i, j);
            }
            beta0_history.push((*threshold, uf.count_components()));
        }

        // Current β₀ is final component count
        self.betti[0] = uf.count_components() as u32;

        // Estimate β₁ (loops) from Euler characteristic
        // χ = β₀ - β₁ + β₂
        // For simplicial complexes: χ = V - E + F
        if let Some((_, final_edges)) = filtration.last() {
            let v = n as i32;
            let e = final_edges.len() as i32;

            // Count triangles (2-simplices) for better β₁ estimation
            let mut triangle_count = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        // Check if edges (i,j), (j,k), (k,i) all exist
                        let has_ij = final_edges.contains(&(i, j));
                        let has_jk = final_edges.contains(&(j, k));
                        let has_ki = final_edges.contains(&(i, k));

                        if has_ij && has_jk && has_ki {
                            triangle_count += 1;
                        }
                    }
                }
            }

            let f = triangle_count;
            let chi = v - e + f;
            let beta1_estimate = (self.betti[0] as i32 - chi).max(0);
            self.betti[1] = beta1_estimate as u32;

            // β₂ (voids) estimation using Euler characteristic
            // For closed surfaces: χ = 2 - 2g where g is genus
            // For 3D complexes: β₂ relates to enclosed voids

            // Count tetrahedra (3-simplices) for void detection
            let mut tetrahedron_count = 0;
            if n >= 4 {
                for i in 0..n {
                    for j in (i + 1)..n {
                        for k in (j + 1)..n {
                            for l in (k + 1)..n {
                                // Check if all 6 edges and 4 faces exist
                                let edges_exist = final_edges.contains(&(i, j))
                                    && final_edges.contains(&(i, k))
                                    && final_edges.contains(&(i, l))
                                    && final_edges.contains(&(j, k))
                                    && final_edges.contains(&(j, l))
                                    && final_edges.contains(&(k, l));

                                if edges_exist {
                                    tetrahedron_count += 1;
                                }
                            }
                        }
                    }
                }
            }

            // χ = β₀ - β₁ + β₂ for 2D surfaces
            // For 3D: χ = V - E + F - T (where T is tetrahedra)
            // β₂ = χ - β₀ + β₁
            if tetrahedron_count > 0 {
                let chi_3d = v - e + f - tetrahedron_count;
                let beta2_estimate = (chi_3d - self.betti[0] as i32 + self.betti[1] as i32).max(0);
                self.betti[2] = beta2_estimate as u32;
            } else {
                // No 3D structure detected, β₂ = 0
                self.betti[2] = 0;
            }
        }
    }

    /// Generate topological signature (hash of Betti numbers and persistence)
    pub fn signature(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.betti[0].hash(&mut hasher);
        self.betti[1].hash(&mut hasher);
        self.betti[2].hash(&mut hasher);

        // Include persistence information
        for dim in 0..3 {
            for pair in &self.diagrams[dim] {
                let persistence = (pair.persistence() * 1000.0) as u64;
                persistence.hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Compute persistence diagram for dimension 0 (connected components)
    pub fn compute_persistence_diagram_dim0(&mut self, max_radius: f32, steps: usize) {
        let n = self.points.len();
        if n < 2 {
            return;
        }

        let filtration = self.vietoris_rips_filtration(max_radius, steps);
        let mut uf = UnionFind::new(n);
        let birth_times = vec![0.0f32; n];
        let mut alive = vec![true; n];

        self.diagrams[0].clear();

        for (threshold, edges) in &filtration {
            for &(i, j) in edges {
                let root_i = uf.find(i);
                let root_j = uf.find(j);

                if root_i != root_j {
                    // Merge components: older component survives
                    let (_survivor, victim) = if birth_times[root_i] <= birth_times[root_j] {
                        (root_i, root_j)
                    } else {
                        (root_j, root_i)
                    };

                    if alive[victim] {
                        // Record death of victim component
                        self.diagrams[0].push(PersistencePair {
                            birth: birth_times[victim],
                            death: *threshold,
                            dimension: 0,
                        });
                        alive[victim] = false;
                    }

                    uf.union(root_i, root_j);
                }
            }
        }

        // Surviving components have infinite death time (we use max_radius as proxy)
        for i in 0..n {
            let root = uf.find(i);
            if alive[root] && uf.is_root(i) {
                self.diagrams[0].push(PersistencePair {
                    birth: birth_times[root],
                    death: f32::INFINITY,
                    dimension: 0,
                });
            }
        }
    }

    /// Get persistence pairs with lifetime above threshold
    pub fn significant_features(&self, min_persistence: f32) -> Vec<PersistencePair> {
        let mut features = Vec::new();

        for dim in 0..3 {
            for &pair in &self.diagrams[dim] {
                if pair.persistence() >= min_persistence {
                    features.push(pair);
                }
            }
        }

        features.sort_by(|a, b| b.persistence().partial_cmp(&a.persistence()).unwrap());
        features
    }

    /// Compute total persistence (sum of all feature lifetimes)
    pub fn total_persistence(&self) -> f32 {
        let mut total = 0.0f32;

        for dim in 0..3 {
            for pair in &self.diagrams[dim] {
                let pers = pair.persistence();
                if pers.is_finite() {
                    total += pers;
                }
            }
        }

        total
    }

    /// Check if two point clouds are topologically similar
    pub fn is_similar(&self, other: &TopologicalLayer, tolerance: u32) -> bool {
        for i in 0..3 {
            if self.betti[i].abs_diff(other.betti[i]) > tolerance {
                return false;
            }
        }
        true
    }
}

impl Default for TopologicalLayer {
    fn default() -> Self {
        Self::new()
    }
}

/// Union-Find data structure for connected components
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x != root_y {
            // Union by rank
            if self.rank[root_x] < self.rank[root_y] {
                self.parent[root_x] = root_y;
            } else if self.rank[root_x] > self.rank[root_y] {
                self.parent[root_y] = root_x;
            } else {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }
    }

    fn count_components(&mut self) -> usize {
        let mut roots = HashSet::new();
        for i in 0..self.parent.len() {
            roots.insert(self.find(i));
        }
        roots.len()
    }

    fn is_root(&self, x: usize) -> bool {
        self.parent[x] == x
    }
}

/// Compute Euclidean distance between two 8D points
fn euclidean_distance(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    #[cfg(feature = "simd")]
    {
        use super::gf8::{gf8_norm2_simd, gf8_sub_simd};
        let diff = gf8_sub_simd(a, b);
        gf8_norm2_simd(&diff).sqrt()
    }
    #[cfg(not(feature = "simd"))]
    {
        let sum_sq: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum();
        sum_sq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_layer_creation() {
        let layer = TopologicalLayer::new();
        assert_eq!(layer.betti, [1, 0, 0]);
    }

    #[test]
    fn test_add_points() {
        let mut layer = TopologicalLayer::new();
        layer.add_point([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        layer.add_point([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        assert_eq!(layer.points.len(), 2);
    }

    #[test]
    fn test_betti_numbers_disconnected() {
        let mut layer = TopologicalLayer::new();

        // Add two widely separated points
        layer.add_point([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        layer.add_point([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // With small radius, should have 2 components
        layer.compute_betti_numbers(1.0, 10);
        assert_eq!(layer.betti[0], 2);
    }

    #[test]
    fn test_betti_numbers_connected() {
        let mut layer = TopologicalLayer::new();

        // Add two close points
        layer.add_point([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        layer.add_point([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // With large radius, should have 1 component
        layer.compute_betti_numbers(10.0, 10);
        assert_eq!(layer.betti[0], 1);
    }

    #[test]
    fn test_signature() {
        let layer1 = TopologicalLayer::new();
        let layer2 = TopologicalLayer::new();

        // Same Betti numbers should give same signature
        assert_eq!(layer1.signature(), layer2.signature());
    }

    #[test]
    fn test_persistence_diagram() {
        let mut layer = TopologicalLayer::new();

        // Add triangle of points
        layer.add_point([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        layer.add_point([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        layer.add_point([0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        layer.compute_persistence_diagram_dim0(5.0, 20);

        // Should have persistence pairs
        assert!(!layer.diagrams[0].is_empty());
    }

    #[test]
    fn test_significant_features() {
        let mut layer = TopologicalLayer::new();

        layer.diagrams[0].push(PersistencePair {
            birth: 0.0,
            death: 2.0,
            dimension: 0,
        });
        layer.diagrams[0].push(PersistencePair {
            birth: 0.0,
            death: 0.1,
            dimension: 0,
        });

        let features = layer.significant_features(1.0);
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].persistence(), 2.0);
    }

    #[test]
    fn test_total_persistence() {
        let mut layer = TopologicalLayer::new();

        layer.diagrams[0].push(PersistencePair {
            birth: 0.0,
            death: 2.0,
            dimension: 0,
        });
        layer.diagrams[1].push(PersistencePair {
            birth: 1.0,
            death: 3.0,
            dimension: 1,
        });

        let total = layer.total_persistence();
        assert_eq!(total, 4.0);
    }

    #[test]
    fn test_is_similar() {
        let mut layer1 = TopologicalLayer::new();
        layer1.betti = [2, 1, 0];

        let mut layer2 = TopologicalLayer::new();
        layer2.betti = [2, 1, 0];

        assert!(layer1.is_similar(&layer2, 0));

        let mut layer3 = TopologicalLayer::new();
        layer3.betti = [5, 1, 0];

        assert!(!layer1.is_similar(&layer3, 1));
    }
}

File: rune\hydron\intrinsics.rs
===============================
/* src/rune/hydron/intrinsics.rs */
//! A queryable registry of x86 SIMD intrinsics for backend code generation.
//!
//! # e8 Primitives – Gf8 Intrinsics Module
//!▫~•◦-----------------------------------------‣
//!
//! This module contains a comprehensive, static list of x86 intrinsics, auto-generated
//! from external documentation. It is designed to be used by the `simd` backend
//! and future procedural code generators to reason about available hardware instructions.
//!
//! ### Key Capabilities
//! - **Static Registry:** Provides `GF8_INTRINSICS`, a constant slice of `Gf8Intrinsic` structs.
//! - **Queryable API:** Offers helper functions to filter and find intrinsics by name, technology, or SIMD width.
//! - **Metadata Rich:** Each entry includes the intrinsic's name, required technology (e.g., AVX2), header, and C prototype.
//!
//! ### Architectural Notes
//! This module acts as a "database" for the compiler backend. Instead of hard-coding
//! intrinsic names, higher-level modules can query this registry to make dynamic
//! decisions about which instructions to use, enabling more flexible and future-proof
//! code generation.
//!
//! ### Example
//! ```rust
//! // This example assumes this module is part of the e8_gf8 crate.
//! // use e8_gf8::intrinsics::{find_intrinsic_by_name, intrinsics_for_f32_width};
//!
//! // fn main() {
//!     // Find a specific intrinsic by name
//!     // if let Some(intrinsic) = find_intrinsic_by_name("_mm256_add_ps") {
//!     //     println!("Found AVX add for f32: {}", intrinsic.prototype);
//!     // }
//!
//!     // Find all 256-bit f32 intrinsics
//!     // let avx_f32_intrinsics = intrinsics_for_f32_width(256).count();
//!     // println!("There are {} relevant 256-bit f32 intrinsics.", avx_f32_intrinsics);
//! // }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// Represents the metadata for a single x86 hardware intrinsic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Gf8Intrinsic {
    /// The C/C++ name of the intrinsic function (e.g., `_mm256_add_ps`).
    pub name: &'static str,
    /// The required CPU feature flag or technology (e.g., "AVX2", "SSE4.1").
    pub technology: &'static str,
    /// The C header file where the intrinsic is typically defined (e.g., "immintrin.h").
    pub header: &'static str,
    /// The C function prototype for the intrinsic.
    pub prototype: &'static str,
}

impl Gf8Intrinsic {
    /// Returns `true` if this intrinsic's prototype suggests it operates on `f32` vectors.
    pub fn is_f32_vector(&self) -> bool {
        self.prototype.contains("__m128")
            || self.prototype.contains("__m256")
            || self.prototype.contains("__m512")
            || self.prototype.contains("ps") // Packed Single
    }

    /// Returns `true` if this intrinsic's prototype suggests it operates on `f64` vectors.
    pub fn is_f64_vector(&self) -> bool {
        self.prototype.contains("__m128d")
            || self.prototype.contains("__m256d")
            || self.prototype.contains("__m512d")
            || self.prototype.contains("pd") // Packed Double
    }

    /// Returns the SIMD vector width in bits, if it can be inferred from the prototype.
    pub fn simd_width_bits(&self) -> Option<u32> {
        if self.prototype.contains("__m512") {
            Some(512)
        } else if self.prototype.contains("__m256") {
            Some(256)
        } else if self.prototype.contains("__m128") {
            Some(128)
        } else if self.prototype.contains("__m64") {
            Some(64)
        } else {
            None
        }
    }
}

/// A static, compile-time registry of all known x86 intrinsics from the source file.
pub const GF8_INTRINSICS: &[Gf8Intrinsic] = &[
    Gf8Intrinsic {
        name: "_m_from_float",
        technology: "3DNOW",
        header: "intrin.h",
        prototype: "__m64 _m_from_float(float);",
    },
    Gf8Intrinsic {
        name: "_m_from_int",
        technology: "MMX",
        header: "intrin.h",
        prototype: "__m64 _m_from_int(int);",
    },
    Gf8Intrinsic {
        name: "_m_maskmovq",
        technology: "SSE",
        header: "intrin.h",
        prototype: "void _m_maskmovq(__m64, __m64, char*);",
    },
    Gf8Intrinsic {
        name: "_mm_abs_epi16",
        technology: "SSSE3",
        header: "intrin.h",
        prototype: "__m128i _mm_abs_epi16(__m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_abs_epi32",
        technology: "SSSE3",
        header: "intrin.h",
        prototype: "__m128i _mm_abs_epi32(__m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_abs_epi8",
        technology: "SSSE3",
        header: "intrin.h",
        prototype: "__m128i _mm_abs_epi8(__m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_add_epi16",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128i _mm_add_epi16(__m128i, __m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_add_epi32",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128i _mm_add_epi32(__m128i, __m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_add_epi64",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128i _mm_add_epi64(__m128i, __m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_add_epi8",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128i _mm_add_epi8(__m128i, __m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_add_pd",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128d _mm_add_pd(__m128d, __m128d);",
    },
    Gf8Intrinsic {
        name: "_mm_add_ps",
        technology: "SSE",
        header: "intrin.h",
        prototype: "__m128 _mm_add_ps(__m128, __m128);",
    },
    Gf8Intrinsic {
        name: "_mm_add_sd",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128d _mm_add_sd(__m128d, __m128d);",
    },
    Gf8Intrinsic {
        name: "_mm_add_ss",
        technology: "SSE",
        header: "intrin.h",
        prototype: "__m128 _mm_add_ss(__m128, __m128);",
    },
    Gf8Intrinsic {
        name: "_mm_adds_epi16",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128i _mm_adds_epi16(__m128i, __m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_adds_epi8",
        technology: "SSE2",
        header: "intrin.h",
        prototype: "__m128i _mm_adds_epi8(__m128i, __m128i);",
    },
    Gf8Intrinsic {
        name: "_mm_addsub_pd",
        technology: "SSE3",
        header: "intrin.h",
        prototype: "__m128d _mm_addsub_pd(__m128d, __m128d);",
    },
    Gf8Intrinsic {
        name: "_mm_addsub_ps",
        technology: "SSE3",
        header: "intrin.h",
        prototype: "__m128 _mm_addsub_ps(__m128, __m128);",
    },
    Gf8Intrinsic {
        name: "_mm_aesdec_si128",
        technology: "AESNI",
        header: "immintrin.h",
        prototype: "__m128i _mm_aesdec_si128(__m128i, __m128i);",
    },
    Gf8Intrinsic {
        name: "_mm256_add_pd",
        technology: "AVX",
        header: "immintrin.h",
        prototype: "__m256d _mm256_add_pd(__m256d, __m256d);",
    },
    Gf8Intrinsic {
        name: "_mm256_add_ps",
        technology: "AVX",
        header: "immintrin.h",
        prototype: "__m256 _mm256_add_ps(__m256, __m256);",
    },
    Gf8Intrinsic {
        name: "_mm256_sub_ps",
        technology: "AVX",
        header: "immintrin.h",
        prototype: "__m256 _mm256_sub_ps(__m256, __m256);",
    },
    Gf8Intrinsic {
        name: "_mm256_dp_ps",
        technology: "AVX",
        header: "immintrin.h",
        prototype: "__m256 _mm256_dp_ps(__m256, __m256, const int);",
    },
    // Add only essential AVX intrinsics for gf8
];

/// Look up an intrinsic by exact name (e.g. "_mm256_add_ps").
pub fn find_intrinsic_by_name(name: &str) -> Option<&'static Gf8Intrinsic> {
    GF8_INTRINSICS.iter().find(|i| i.name == name)
}

/// All intrinsics for a given technology (e.g. "AVX2", "AVX-512F").
pub fn intrinsics_by_technology(tech: &str) -> impl Iterator<Item = &'static Gf8Intrinsic> {
    GF8_INTRINSICS.iter().filter(move |i| i.technology == tech)
}

/// All intrinsics that look like f32 SIMD of a particular width (128/256/512).
pub fn intrinsics_for_f32_width(width_bits: u32) -> impl Iterator<Item = &'static Gf8Intrinsic> {
    GF8_INTRINSICS
        .iter()
        .filter(move |i| i.is_f32_vector() && i.simd_width_bits() == Some(width_bits))
}

/// All intrinsics that look like f64 SIMD of a particular width (128/256/512).
pub fn intrinsics_for_f64_width(width_bits: u32) -> impl Iterator<Item = &'static Gf8Intrinsic> {
    GF8_INTRINSICS
        .iter()
        .filter(move |i| i.is_f64_vector() && i.simd_width_bits() == Some(width_bits))
}

File: rune\hydron\lorentzian.rs
===============================
//! Lorentzian Geometry Layer - Spacetime Metrics and Geodesics
//!
//! Implements Lorentzian (pseudo-Riemannian) geometry with signature (-,+,+,+,+,+,+,+).
//! Core mathematical operations:
//! - Minkowski metric: ds² = -dt² + dx₁² + dx₂² + ... + dx₇²
//! - Proper time along timelike worldlines
//! - Geodesic computations
//! - Light cone structure (null surfaces)
//! - Lorentz transformations and boosts
//! - Causal relationships (timelike, spacelike, lightlike separation)
//!
//! Extension: Causal DAG for event ordering (optional, game-specific logic)
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{HashMap, HashSet, VecDeque};

/// Spacetime point in 7+1 dimensional Lorentzian manifold
#[derive(Clone, Debug, PartialEq)]
pub struct SpacetimePoint {
    /// Coordinates [t, x₁, x₂, x₃, x₄, x₅, x₆, x₇]
    pub coords: [f64; 8],
}

impl SpacetimePoint {
    /// Create new spacetime point
    pub fn new(coords: [f64; 8]) -> Self {
        Self { coords }
    }

    /// Time coordinate
    pub fn time(&self) -> f64 {
        self.coords[0]
    }

    /// Spatial coordinates as slice
    pub fn spatial(&self) -> &[f64] {
        &self.coords[1..8]
    }

    /// Compute Minkowski metric interval to another point
    /// ds² = -dt² + Σ(dxᵢ²)
    pub fn minkowski_interval(&self, other: &SpacetimePoint) -> f64 {
        let dt = self.coords[0] - other.coords[0];
        let spatial_dist_sq: f64 = (1..8)
            .map(|i| {
                let dx = self.coords[i] - other.coords[i];
                dx * dx
            })
            .sum();

        -dt * dt + spatial_dist_sq
    }

    /// Compute proper time (for timelike curves)
    /// τ = √(-ds²) if ds² < 0
    pub fn proper_time(&self, other: &SpacetimePoint) -> Option<f64> {
        let interval = self.minkowski_interval(other);
        if interval < 0.0 {
            Some((-interval).sqrt())
        } else {
            None // Not timelike separated
        }
    }

    /// Check if two points are timelike separated (ds² < 0)
    pub fn is_timelike(&self, other: &SpacetimePoint) -> bool {
        self.minkowski_interval(other) < 0.0
    }

    /// Check if two points are spacelike separated (ds² > 0)
    pub fn is_spacelike(&self, other: &SpacetimePoint) -> bool {
        self.minkowski_interval(other) > 0.0
    }

    /// Check if two points are lightlike/null separated (ds² = 0)
    pub fn is_lightlike(&self, other: &SpacetimePoint) -> bool {
        self.minkowski_interval(other).abs() < 1e-10
    }

    /// Determine causal relationship based on interval and time ordering
    pub fn causal_relation(&self, other: &SpacetimePoint) -> CausalRelation {
        let interval = self.minkowski_interval(other);
        let dt = self.coords[0] - other.coords[0];

        if interval < -1e-10 {
            if dt > 0.0 {
                CausalRelation::Future
            } else {
                CausalRelation::Past
            }
        } else if interval > 1e-10 {
            CausalRelation::Spacelike
        } else {
            // Lightlike
            if dt > 0.0 {
                CausalRelation::LightlikeFuture
            } else if dt < 0.0 {
                CausalRelation::LightlikePast
            } else {
                CausalRelation::Coincident
            }
        }
    }
}

/// Causal relationship between two spacetime points
#[derive(Clone, Debug, PartialEq)]
pub enum CausalRelation {
    /// Timelike future (this event is in the causal future)
    Future,
    /// Timelike past (this event is in the causal past)
    Past,
    /// Spacelike separated (no causal relationship)
    Spacelike,
    /// Null/lightlike future
    LightlikeFuture,
    /// Null/lightlike past
    LightlikePast,
    /// Same event
    Coincident,
}

/// Worldline - timelike curve through spacetime
#[derive(Clone, Debug)]
pub struct Worldline {
    /// Points along the worldline (must be timelike connected)
    pub points: Vec<SpacetimePoint>,
}

impl Worldline {
    /// Create new worldline
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Add point to worldline (validates timelike connection)
    pub fn add_point(&mut self, point: SpacetimePoint) -> Result<(), &'static str> {
        if let Some(last) = self.points.last() {
            if !last.is_timelike(&point) || point.time() <= last.time() {
                return Err("Point must be timelike future-connected to previous point");
            }
        }
        self.points.push(point);
        Ok(())
    }

    /// Compute total proper time along worldline
    pub fn proper_time(&self) -> f64 {
        let mut tau = 0.0;
        for i in 1..self.points.len() {
            if let Some(dtau) = self.points[i - 1].proper_time(&self.points[i]) {
                tau += dtau;
            }
        }
        tau
    }

    /// Get velocity 4-vector at index (forward difference)
    pub fn four_velocity(&self, index: usize) -> Option<[f64; 8]> {
        if index + 1 >= self.points.len() {
            return None;
        }

        let p1 = &self.points[index];
        let p2 = &self.points[index + 1];

        let dtau = p1.proper_time(p2)?;
        let mut v = [0.0; 8];

        for i in 0..8 {
            v[i] = (p2.coords[i] - p1.coords[i]) / dtau;
        }

        Some(v)
    }
}

impl Default for Worldline {
    fn default() -> Self {
        Self::new()
    }
}

/// Lorentzian geometry layer
pub struct LorentzianLayer {
    /// Worldlines tracked in this layer
    pub worldlines: Vec<Worldline>,

    /// Metric signature (-1, +1, +1, +1, +1, +1, +1, +1)
    pub signature: [i8; 8],
}

impl LorentzianLayer {
    /// Create new Lorentzian layer with standard signature
    pub fn new() -> Self {
        Self {
            worldlines: Vec::new(),
            signature: [-1, 1, 1, 1, 1, 1, 1, 1],
        }
    }

    /// Add worldline to layer
    pub fn add_worldline(&mut self, worldline: Worldline) {
        self.worldlines.push(worldline);
    }

    /// Compute geodesic distance between two points (proper time for timelike)
    pub fn geodesic_distance(&self, p1: &SpacetimePoint, p2: &SpacetimePoint) -> Option<f64> {
        p1.proper_time(p2)
    }

    /// Check if point is in past light cone of another
    pub fn in_past_light_cone(&self, point: &SpacetimePoint, reference: &SpacetimePoint) -> bool {
        let interval = point.minkowski_interval(reference);
        let dt = reference.time() - point.time();
        interval <= 0.0 && dt > 0.0
    }

    /// Check if point is in future light cone of another
    pub fn in_future_light_cone(&self, point: &SpacetimePoint, reference: &SpacetimePoint) -> bool {
        let interval = point.minkowski_interval(reference);
        let dt = point.time() - reference.time();
        interval <= 0.0 && dt > 0.0
    }

    /// Lorentz boost along x₁ axis
    /// γ = 1/√(1 - v²)
    pub fn lorentz_boost(&self, point: &SpacetimePoint, velocity: f64) -> SpacetimePoint {
        if velocity.abs() >= 1.0 {
            return point.clone(); // Invalid velocity
        }

        let gamma = 1.0 / (1.0 - velocity * velocity).sqrt();
        let mut boosted = point.coords;

        let t = point.coords[0];
        let x = point.coords[1];

        boosted[0] = gamma * (t - velocity * x);
        boosted[1] = gamma * (x - velocity * t);

        SpacetimePoint::new(boosted)
    }
}

impl Default for LorentzianLayer {
    fn default() -> Self {
        Self::new()
    }
}

//====================================================================================
// EXTENSION: CAUSAL DAG (Game-Specific Event Ordering)
//====================================================================================

/// Event types in the causal graph (game/application specific)
#[derive(Clone, Debug, PartialEq)]
pub enum EventType {
    /// Movement between E8 roots
    Move { from_root: usize, to_root: usize },

    /// Combat encounter
    Combat { monster_id: String, outcome: bool },

    /// Death event
    Death { cause: String },

    /// Concept emergence
    Emergence { concept_id: u64 },

    /// Goal transition
    GoalSwitch { old_root: usize, new_root: usize },

    /// Healing event
    Heal { amount: f32 },

    /// Trap trigger
    Trap { damage: f32 },

    /// Generic action
    Action { description: String },
}

/// Causal node representing an event in spacetime (extends SpacetimePoint)
#[derive(Clone, Debug)]
pub struct CausalNode {
    /// Spacetime location
    pub location: SpacetimePoint,

    /// Unique event identifier
    pub event_id: u64,

    /// Spatial position (E8 root index, application-specific)
    pub e8_root: usize,

    /// Event type (application-specific)
    pub event_type: EventType,
}

impl CausalNode {
    /// Create new causal node
    pub fn new(
        event_id: u64,
        location: SpacetimePoint,
        e8_root: usize,
        event_type: EventType,
    ) -> Self {
        Self {
            location,
            event_id,
            e8_root,
            event_type,
        }
    }

    /// Compute spacetime interval to another event (delegates to SpacetimePoint)
    pub fn spacetime_interval(&self, other: &CausalNode) -> f64 {
        self.location.minkowski_interval(&other.location)
    }

    /// Check if this event is in the causal future of another
    pub fn is_causally_after(&self, other: &CausalNode) -> bool {
        matches!(
            self.location.causal_relation(&other.location),
            CausalRelation::Future | CausalRelation::LightlikeFuture
        )
    }

    /// Check if this event is spacelike separated from another
    pub fn is_spacelike_separated(&self, other: &CausalNode) -> bool {
        self.location.is_spacelike(&other.location)
    }
}

/// Causal directed acyclic graph (DAG) - Extension for event ordering
#[derive(Clone, Debug)]
pub struct CausalDAG {
    /// All events in the causal history
    pub nodes: Vec<CausalNode>,

    /// Causal edges: (cause_id, effect_id)
    pub edges: Vec<(u64, u64)>,

    /// Adjacency list for efficient traversal: event_id -> [effect_ids]
    adjacency: HashMap<u64, Vec<u64>>,

    /// Reverse adjacency: event_id -> [cause_ids]
    reverse_adjacency: HashMap<u64, Vec<u64>>,
}

impl CausalDAG {
    /// Create empty causal DAG
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
            reverse_adjacency: HashMap::new(),
        }
    }

    /// Add event with causal dependencies
    pub fn add_event(&mut self, node: CausalNode, causes: &[u64]) -> u64 {
        let event_id = node.event_id;

        // Add node
        self.nodes.push(node);

        // Add edges
        for &cause_id in causes {
            self.edges.push((cause_id, event_id));

            // Update adjacency lists
            self.adjacency.entry(cause_id).or_default().push(event_id);

            self.reverse_adjacency
                .entry(event_id)
                .or_default()
                .push(cause_id);
        }

        event_id
    }

    /// Get past light cone (all causal ancestors)
    pub fn past_light_cone(&self, event_id: u64) -> Vec<u64> {
        let mut past = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(event_id);
        visited.insert(event_id);

        while let Some(current) = queue.pop_front() {
            if let Some(causes) = self.reverse_adjacency.get(&current) {
                for &cause_id in causes {
                    if !visited.contains(&cause_id) {
                        past.push(cause_id);
                        queue.push_back(cause_id);
                        visited.insert(cause_id);
                    }
                }
            }
        }

        past
    }

    /// Get future light cone (all causal descendants)
    pub fn future_light_cone(&self, event_id: u64) -> Vec<u64> {
        let mut future = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(event_id);
        visited.insert(event_id);

        while let Some(current) = queue.pop_front() {
            if let Some(effects) = self.adjacency.get(&current) {
                for &effect_id in effects {
                    if !visited.contains(&effect_id) {
                        future.push(effect_id);
                        queue.push_back(effect_id);
                        visited.insert(effect_id);
                    }
                }
            }
        }

        future
    }

    /// Check if event A is in causal past of event B
    pub fn is_causal_past(&self, a: u64, b: u64) -> bool {
        self.past_light_cone(b).contains(&a)
    }

    /// Check if event A is in causal future of event B
    pub fn is_causal_future(&self, a: u64, b: u64) -> bool {
        self.future_light_cone(b).contains(&a)
    }

    /// Find node by event ID
    pub fn get_node(&self, event_id: u64) -> Option<&CausalNode> {
        self.nodes.iter().find(|n| n.event_id == event_id)
    }

    /// Verify causal consistency (no cycles)
    pub fn verify_consistency(&self) -> bool {
        // Use topological sort to detect cycles
        let mut in_degree: HashMap<u64, usize> = HashMap::new();

        // Initialize in-degrees
        for node in &self.nodes {
            in_degree.insert(node.event_id, 0);
        }

        for &(_, effect) in &self.edges {
            *in_degree.get_mut(&effect).unwrap() += 1;
        }

        // Find all nodes with in-degree 0
        let mut queue: VecDeque<u64> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut sorted_count = 0;

        while let Some(node_id) = queue.pop_front() {
            sorted_count += 1;

            if let Some(effects) = self.adjacency.get(&node_id) {
                for &effect_id in effects {
                    let deg = in_degree.get_mut(&effect_id).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(effect_id);
                    }
                }
            }
        }

        // If all nodes were processed, no cycles exist
        sorted_count == self.nodes.len()
    }

    /// Get topological ordering of events
    pub fn topological_order(&self) -> Option<Vec<u64>> {
        if !self.verify_consistency() {
            return None;
        }

        let mut in_degree: HashMap<u64, usize> = HashMap::new();

        for node in &self.nodes {
            in_degree.insert(node.event_id, 0);
        }

        for &(_, effect) in &self.edges {
            *in_degree.get_mut(&effect).unwrap() += 1;
        }

        let mut queue: VecDeque<u64> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut order = Vec::new();

        while let Some(node_id) = queue.pop_front() {
            order.push(node_id);

            if let Some(effects) = self.adjacency.get(&node_id) {
                for &effect_id in effects {
                    let deg = in_degree.get_mut(&effect_id).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(effect_id);
                    }
                }
            }
        }

        Some(order)
    }
}

impl Default for CausalDAG {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined layer with both Lorentzian geometry and causal DAG extension
pub struct LorentzianCausalLayer {
    /// Core Lorentzian geometry
    pub geometry: LorentzianLayer,

    /// Causal DAG extension
    pub dag: CausalDAG,

    /// Current proper time
    pub proper_time: f64,

    /// Next event ID
    next_event_id: u64,
}

impl LorentzianCausalLayer {
    /// Create new combined layer
    pub fn new() -> Self {
        Self {
            geometry: LorentzianLayer::new(),
            dag: CausalDAG::new(),
            proper_time: 0.0,
            next_event_id: 0,
        }
    }

    /// Add event to causal graph
    pub fn add_event(&mut self, e8_root: usize, event_type: EventType, causes: &[u64]) -> u64 {
        let event_id = self.next_event_id;
        self.next_event_id += 1;
        self.proper_time += 1.0;

        // Create spacetime coordinates [t, x1, ..., x7]
        let mut coordinates = [0.0f64; 8];
        coordinates[0] = self.proper_time;
        // Spatial coordinates encode E8 root (simplified mapping)
        for (i, coord) in coordinates.iter_mut().enumerate().skip(1) {
            *coord = ((e8_root >> i) & 1) as f64;
        }

        let location = SpacetimePoint::new(coordinates);
        let node = CausalNode::new(event_id, location, e8_root, event_type);

        self.dag.add_event(node, causes);
        event_id
    }

    /// Get past light cone of an event
    pub fn past_light_cone(&self, event_id: u64) -> Vec<u64> {
        self.dag.past_light_cone(event_id)
    }

    /// Get future light cone of an event
    pub fn future_light_cone(&self, event_id: u64) -> Vec<u64> {
        self.dag.future_light_cone(event_id)
    }

    /// Check causal ordering
    pub fn is_causal_past(&self, a: u64, b: u64) -> bool {
        self.dag.is_causal_past(a, b)
    }

    /// Check causal future
    pub fn is_causal_future(&self, a: u64, b: u64) -> bool {
        self.dag.is_causal_future(a, b)
    }

    /// Verify causal consistency
    pub fn verify_consistency(&self) -> bool {
        self.dag.verify_consistency()
    }

    /// Get event by ID
    pub fn get_event(&self, event_id: u64) -> Option<&CausalNode> {
        self.dag.get_node(event_id)
    }
}

impl Default for LorentzianCausalLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Core Lorentzian geometry tests
    #[test]
    fn test_minkowski_interval() {
        let p1 = SpacetimePoint::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p2 = SpacetimePoint::new([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let interval = p1.minkowski_interval(&p2);
        // ds² = -dt² + dx² = -(1)² + (1)² = 0 (lightlike)
        assert_eq!(interval, 0.0);
    }

    #[test]
    fn test_timelike_separation() {
        let p1 = SpacetimePoint::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p2 = SpacetimePoint::new([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        assert!(p1.is_timelike(&p2));
        assert_eq!(p1.proper_time(&p2), Some((3.0_f64).sqrt()));
    }

    #[test]
    fn test_spacelike_separation() {
        let p1 = SpacetimePoint::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p2 = SpacetimePoint::new([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        assert!(p1.is_spacelike(&p2));
        assert_eq!(p1.proper_time(&p2), None);
    }

    #[test]
    fn test_causal_relation() {
        let p1 = SpacetimePoint::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p2 = SpacetimePoint::new([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        assert_eq!(p2.causal_relation(&p1), CausalRelation::Future);
        assert_eq!(p1.causal_relation(&p2), CausalRelation::Past);
    }

    #[test]
    fn test_worldline() {
        let mut wl = Worldline::new();

        let p1 = SpacetimePoint::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p2 = SpacetimePoint::new([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p3 = SpacetimePoint::new([4.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        assert!(wl.add_point(p1).is_ok());
        assert!(wl.add_point(p2).is_ok());
        assert!(wl.add_point(p3).is_ok());

        assert!(wl.proper_time() > 0.0);
    }

    #[test]
    fn test_lorentz_boost() {
        let layer = LorentzianLayer::new();
        let p = SpacetimePoint::new([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = 0.5;

        let boosted = layer.lorentz_boost(&p, v);

        // Check that time and x1 are transformed
        assert_ne!(boosted.coords[0], p.coords[0]);
        assert_ne!(boosted.coords[1], p.coords[1]);
        // Other coordinates unchanged
        assert_eq!(boosted.coords[2], p.coords[2]);
    }

    // Extension: Causal DAG tests
    #[test]
    fn test_lorentzian_causal_layer_creation() {
        let layer = LorentzianCausalLayer::new();
        assert_eq!(layer.proper_time, 0.0);
        assert_eq!(layer.next_event_id, 0);
    }

    #[test]
    fn test_add_event() {
        let mut layer = LorentzianCausalLayer::new();

        let event1 = layer.add_event(
            0,
            EventType::Action {
                description: "Start".to_string(),
            },
            &[],
        );
        assert_eq!(event1, 0);
        assert_eq!(layer.proper_time, 1.0);

        let event2 = layer.add_event(
            1,
            EventType::Move {
                from_root: 0,
                to_root: 1,
            },
            &[event1],
        );
        assert_eq!(event2, 1);
        assert_eq!(layer.proper_time, 2.0);
    }

    #[test]
    fn test_causal_ordering() {
        let mut layer = LorentzianCausalLayer::new();

        let e1 = layer.add_event(
            0,
            EventType::Action {
                description: "A".to_string(),
            },
            &[],
        );
        let e2 = layer.add_event(
            1,
            EventType::Action {
                description: "B".to_string(),
            },
            &[e1],
        );
        let e3 = layer.add_event(
            2,
            EventType::Action {
                description: "C".to_string(),
            },
            &[e2],
        );

        assert!(layer.is_causal_past(e1, e2));
        assert!(layer.is_causal_past(e1, e3));
        assert!(layer.is_causal_past(e2, e3));

        assert!(!layer.is_causal_past(e2, e1));
        assert!(!layer.is_causal_past(e3, e1));
    }

    #[test]
    fn test_light_cones() {
        let mut layer = LorentzianCausalLayer::new();

        let e1 = layer.add_event(
            0,
            EventType::Action {
                description: "Root".to_string(),
            },
            &[],
        );
        let e2 = layer.add_event(
            1,
            EventType::Action {
                description: "Child1".to_string(),
            },
            &[e1],
        );
        let e3 = layer.add_event(
            2,
            EventType::Action {
                description: "Child2".to_string(),
            },
            &[e1],
        );
        let e4 = layer.add_event(
            3,
            EventType::Action {
                description: "Grandchild".to_string(),
            },
            &[e2, e3],
        );

        let past = layer.past_light_cone(e4);
        assert!(past.contains(&e1));
        assert!(past.contains(&e2));
        assert!(past.contains(&e3));

        let future = layer.future_light_cone(e1);
        assert!(future.contains(&e2));
        assert!(future.contains(&e3));
        assert!(future.contains(&e4));
    }

    #[test]
    fn test_causal_consistency() {
        let mut layer = LorentzianCausalLayer::new();

        let e1 = layer.add_event(
            0,
            EventType::Action {
                description: "A".to_string(),
            },
            &[],
        );
        let e2 = layer.add_event(
            1,
            EventType::Action {
                description: "B".to_string(),
            },
            &[e1],
        );
        let _e3 = layer.add_event(
            2,
            EventType::Action {
                description: "C".to_string(),
            },
            &[e2],
        );

        assert!(layer.verify_consistency());
    }

    #[test]
    fn test_topological_order() {
        let mut layer = LorentzianCausalLayer::new();

        let e1 = layer.add_event(
            0,
            EventType::Action {
                description: "A".to_string(),
            },
            &[],
        );
        let e2 = layer.add_event(
            1,
            EventType::Action {
                description: "B".to_string(),
            },
            &[e1],
        );
        let e3 = layer.add_event(
            2,
            EventType::Action {
                description: "C".to_string(),
            },
            &[e1],
        );

        let order = layer.dag.topological_order().unwrap();

        // e1 must come before e2 and e3
        let pos1 = order.iter().position(|&x| x == e1).unwrap();
        let pos2 = order.iter().position(|&x| x == e2).unwrap();
        let pos3 = order.iter().position(|&x| x == e3).unwrap();

        assert!(pos1 < pos2);
        assert!(pos1 < pos3);
    }
}

File: rune\hydron\values.rs
===========================
//! RUNE Evaluation Engine - Runtime Value System
//!
//! Provides runtime evaluation for RUNE expressions, including:
//! - E8 geometric types (vectors, octonions)
//! - GF(8) Galois field arithmetic
//! - Context-aware evaluation based on root declarations
//! - Built-in operations that bridge RUNE into Hydron geometry layers
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

// Hydron geometry layers
use super::quaternion::QuaternionOps;
use super::spherical::SphericalLayer;
use super::symplectic::SymplecticLayer;
use super::topological::TopologicalLayer;

/// Runtime value types in the E8 ecosystem
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Boolean value
    Bool(bool),

    /// Floating-point number (for RUNE expressions)
    Float(f64),

    /// String value
    String(String),

    /// Array of values
    Array(Vec<Value>),

    /// Scalar numeric value (f32)
    Scalar(f32),

    /// 8-dimensional geometric float (canonical Gf8)
    Gf8(super::gf8::Gf8),

    /// 8-dimensional vector in E8 lattice
    Vec8([f32; 8]),

    /// 16-dimensional phase space vector (position + momentum)
    Vec16([f32; 16]),

    /// Octonion (8-dimensional non-associative algebra)
    Octonion(Octonion),

    /// Quaternion (4D rotation)
    Quaternion([f32; 4]),

    /// Symbolic reference (unevaluated)
    Symbol(String),

    /// 8x8 matrix (Fisher information, etc.)
    Matrix8x8([[f32; 8]; 8]),

    /// Betti numbers (topological invariants)
    Betti([u32; 3]),

    /// Collection of Vec8 points (for point clouds)
    PointCloud(Vec<[f32; 8]>),

    /// Error value
    Error(String),
}

/// Octonion representation: (scalar, 7 imaginary units)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Octonion {
    pub scalar: f32,
    pub i: [f32; 7], // e1, e2, e3, e4, e5, e6, e7
}

impl Octonion {
    /// Create a new octonion
    pub fn new(scalar: f32, i: [f32; 7]) -> Self {
        Self { scalar, i }
    }

    /// Create a real octonion (pure scalar)
    pub fn real(scalar: f32) -> Self {
        Self {
            scalar,
            i: [0.0; 7],
        }
    }

    /// Octonion multiplication (non-associative!)
    ///
    /// Implements full Fano-plane based multiplication:
    /// (a0 + a·e) * (b0 + b·e) =
    ///   (a0*b0 - a·b) + (a0*b + b0*a + a × b),
    /// where a × b is the G₂-invariant 7D cross product induced by the Fano plane.
    pub fn mul(&self, other: &Octonion) -> Octonion {
        let a0 = self.scalar;
        let b0 = other.scalar;
        let a = &self.i;
        let b = &other.i;

        // Scalar part: a0*b0 - a·b
        let mut scalar = a0 * b0;
        for k in 0..7 {
            scalar -= a[k] * b[k];
        }

        // Imaginary part: a0*b + b0*a + a × b
        let mut imag = [0.0f32; 7];

        // Linear terms a0*b + b0*a
        for k in 0..7 {
            imag[k] += a0 * b[k] + b0 * a[k];
        }

        // Cross product term a × b via Fano plane structure constants
        //
        // We encode the oriented Fano triples for the imaginary units e1..e7.
        // Indices 0..6 correspond to e1..e7.
        //
        // The triples below define:
        //   e_i * e_j =  e_k  if (i,j,k) in oriented triple
        //   e_j * e_i = -e_k  (anti-commutativity)
        //
        // The chosen convention is one standard G₂ / octonion orientation:
        //   (1,2,3), (1,4,5), (1,6,7),
        //   (2,4,6), (2,5,7), (3,4,7), (3,5,6)
        const FANO_TRIPLES: &[(usize, usize, usize)] = &[
            (0, 1, 2),
            (0, 3, 4),
            (0, 5, 6),
            (1, 3, 5),
            (1, 4, 6),
            (2, 3, 6),
            (2, 4, 5),
        ];

        // Helper: product of basis elements e_(i+1) * e_(j+1)
        // Returns (scalar_part, imag_basis) where imag_basis[k] is the coefficient of e_(k+1).
        fn basis_mul(i: usize, j: usize) -> (f32, [f32; 7]) {
            debug_assert!(i < 7 && j < 7);
            if i == j {
                // e_i * e_i = -1
                return (-1.0, [0.0; 7]);
            }

            for &(a, b, c) in FANO_TRIPLES.iter() {
                // e_a * e_b =  e_c, e_b * e_a = -e_c
                if i == a && j == b {
                    let mut v = [0.0f32; 7];
                    v[c] = 1.0;
                    return (0.0, v);
                }
                if i == b && j == a {
                    let mut v = [0.0f32; 7];
                    v[c] = -1.0;
                    return (0.0, v);
                }

                // e_b * e_c =  e_a, e_c * e_b = -e_a
                if i == b && j == c {
                    let mut v = [0.0f32; 7];
                    v[a] = 1.0;
                    return (0.0, v);
                }
                if i == c && j == b {
                    let mut v = [0.0f32; 7];
                    v[a] = -1.0;
                    return (0.0, v);
                }

                // e_c * e_a =  e_b, e_a * e_c = -e_b
                if i == c && j == a {
                    let mut v = [0.0f32; 7];
                    v[b] = 1.0;
                    return (0.0, v);
                }
                if i == a && j == c {
                    let mut v = [0.0f32; 7];
                    v[b] = -1.0;
                    return (0.0, v);
                }
            }

            // This should never be reached if FANO_TRIPLES covers all oriented pairs.
            (0.0, [0.0; 7])
        }

        // Accumulate a × b via bilinearity:
        // (∑ a_i e_i) * (∑ b_j e_j) = ∑_{i,j} a_i b_j (e_i * e_j)
        // We already handled the i == j scalar contribution above,
        // so here we only need i != j and only add imaginary parts.
        for i in 0..7 {
            if a[i] == 0.0 {
                continue;
            }
            for j in 0..7 {
                if b[j] == 0.0 || i == j {
                    continue;
                }
                let (_s_part, basis_vec) = basis_mul(i, j);
                let coeff = a[i] * b[j];

                // Only imaginary contributions are expected here (_s_part is 0.0 for i != j).
                for k in 0..7 {
                    imag[k] += coeff * basis_vec[k];
                }
            }
        }

        Octonion { scalar, i: imag }
    }

    /// Conjugate of octonion
    pub fn conjugate(&self) -> Octonion {
        let mut neg_i = self.i;
        for x in &mut neg_i {
            *x = -*x;
        }
        Octonion {
            scalar: self.scalar,
            i: neg_i,
        }
    }

    /// Norm (magnitude) of octonion
    pub fn norm(&self) -> f32 {
        let mut sum = self.scalar * self.scalar;
        for &x in &self.i {
            sum += x * x;
        }
        sum.sqrt()
    }
}

impl fmt::Display for Octonion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.scalar)?;
        for (i, &val) in self.i.iter().enumerate() {
            if val != 0.0 {
                write!(f, " + {}e{}", val, i + 1)?;
            }
        }
        Ok(())
    }
}

// Re-export canonical Gf8 from gf8 module
pub use super::gf8::Gf8;

// Re-export SIMD functions when feature is enabled
#[cfg(feature = "simd")]
pub use super::gf8::{
    get_available_f32_256_intrinsics, gf8_add_inplace_slice_simd, gf8_add_simd, gf8_dot_simd,
    gf8_matvec_simd, gf8_norm2_simd, gf8_sub_simd, print_simd_capabilities,
};

impl Value {
    /// Add two values
    pub fn add(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a + b)),

            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(format!(
                        "Cannot add arrays of different lengths: {} and {}",
                        a.len(),
                        b.len()
                    )));
                }
                let mut result = Vec::new();
                for (va, vb) in a.iter().zip(b.iter()) {
                    result.push(va.add(vb)?);
                }
                Ok(Value::Array(result))
            }

            (Value::Vec8(a), Value::Vec8(b)) => {
                #[cfg(feature = "simd")]
                {
                    use super::gf8::gf8_add_inplace_slice_simd;
                    let mut result = *a;
                    gf8_add_inplace_slice_simd(&mut result, b);
                    Ok(Value::Vec8(result))
                }
                #[cfg(not(feature = "simd"))]
                {
                    let mut result = [0.0; 8];
                    for i in 0..8 {
                        result[i] = a[i] + b[i];
                    }
                    Ok(Value::Vec8(result))
                }
            }

            (Value::Gf8(a), Value::Gf8(b)) => {
                #[cfg(feature = "simd")]
                {
                    let result_coords = super::gf8::gf8_add_simd(a.coords(), b.coords());
                    Ok(Value::Gf8(super::gf8::Gf8::new(result_coords)))
                }
                #[cfg(not(feature = "simd"))]
                {
                    Ok(Value::Gf8(*a + *b))
                }
            }

            (Value::Octonion(a), Value::Octonion(b)) => {
                let mut result_i = [0.0f32; 7];
                for (i, result) in result_i.iter_mut().enumerate() {
                    *result = a.i[i] + b.i[i];
                }
                Ok(Value::Octonion(Octonion {
                    scalar: a.scalar + b.scalar,
                    i: result_i,
                }))
            }

            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot add {:?} and {:?}",
                self, other
            ))),
        }
    }

    /// Multiply two values
    pub fn mul(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a * b)),

            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(format!(
                        "Cannot multiply arrays of different lengths: {} and {}",
                        a.len(),
                        b.len()
                    )));
                }
                let mut result = Vec::new();
                for (va, vb) in a.iter().zip(b.iter()) {
                    result.push(va.mul(vb)?);
                }
                Ok(Value::Array(result))
            }

            (Value::Scalar(s), Value::Vec8(v)) | (Value::Vec8(v), Value::Scalar(s)) => {
                let mut result = [0.0; 8];
                for i in 0..8 {
                    result[i] = v[i] * s;
                }
                Ok(Value::Vec8(result))
            }

            (Value::Gf8(a), Value::Scalar(s)) | (Value::Scalar(s), Value::Gf8(a)) => {
                Ok(Value::Gf8(*a * *s))
            }

            (Value::Octonion(a), Value::Octonion(b)) => Ok(Value::Octonion(a.mul(b))),

            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot multiply {:?} and {:?}",
                self, other
            ))),
        }
    }

    /// Subtract two values
    pub fn sub(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a - b)),

            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(format!(
                        "Cannot subtract arrays of different lengths: {} and {}",
                        a.len(),
                        b.len()
                    )));
                }
                let mut result = Vec::new();
                for (va, vb) in a.iter().zip(b.iter()) {
                    result.push(va.sub(vb)?);
                }
                Ok(Value::Array(result))
            }

            (Value::Vec8(a), Value::Vec8(b)) => {
                let mut result = [0.0; 8];
                for i in 0..8 {
                    result[i] = a[i] - b[i];
                }
                Ok(Value::Vec8(result))
            }

            (Value::Gf8(a), Value::Gf8(b)) => {
                #[cfg(feature = "simd")]
                {
                    let result_coords = super::gf8::gf8_sub_simd(a.coords(), b.coords());
                    Ok(Value::Gf8(super::gf8::Gf8::new(result_coords)))
                }
                #[cfg(not(feature = "simd"))]
                {
                    Ok(Value::Gf8(*a - *b))
                }
            }

            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot subtract {:?} and {:?}",
                self, other
            ))),
        }
    }

    /// Divide two values
    pub fn div(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => {
                if *b == 0.0 {
                    return Err(EvalError::DivisionByZero);
                }
                Ok(Value::Float(a / b))
            }

            (Value::Scalar(a), Value::Scalar(b)) => {
                if *b == 0.0 {
                    return Err(EvalError::DivisionByZero);
                }
                Ok(Value::Scalar(a / b))
            }

            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Err(EvalError::TypeMismatch(format!(
                        "Cannot divide arrays of different lengths: {} and {}",
                        a.len(),
                        b.len()
                    )));
                }
                let mut result = Vec::new();
                for (va, vb) in a.iter().zip(b.iter()) {
                    result.push(va.div(vb)?);
                }
                Ok(Value::Array(result))
            }

            (Value::Vec8(v), Value::Scalar(s)) => {
                if *s == 0.0 {
                    return Err(EvalError::DivisionByZero);
                }
                let mut result = [0.0; 8];
                for i in 0..8 {
                    result[i] = v[i] / s;
                }
                Ok(Value::Vec8(result))
            }

            (Value::Gf8(_a), Value::Gf8(_b)) => {
                // Division for geometric Gf8 not directly supported
                Err(EvalError::TypeMismatch(
                    "Division not supported for Gf8 geometric types".to_string(),
                ))
            }

            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot divide {:?} by {:?}",
                self, other
            ))),
        }
    }

    /// Power operation
    pub fn pow(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.powf(*b))),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a.powf(*b))),

            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot raise {:?} to power {:?}",
                self, other
            ))),
        }
    }

    /// Modulo operation
    pub fn modulo(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => {
                if *b == 0.0 {
                    return Err(EvalError::DivisionByZero);
                }
                Ok(Value::Float(a % b))
            }

            (Value::Scalar(a), Value::Scalar(b)) => {
                if *b == 0.0 {
                    return Err(EvalError::DivisionByZero);
                }
                Ok(Value::Scalar(a % b))
            }

            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot compute {:?} mod {:?}",
                self, other
            ))),
        }
    }

    /// Negate a value (unary minus)
    pub fn negate(&self) -> Result<Value, EvalError> {
        match self {
            Value::Float(a) => Ok(Value::Float(-a)),
            Value::Scalar(a) => Ok(Value::Scalar(-a)),

            Value::Array(a) => {
                let mut result = Vec::new();
                for val in a.iter() {
                    result.push(val.negate()?);
                }
                Ok(Value::Array(result))
            }

            Value::Vec8(v) => {
                let mut result = [0.0; 8];
                for i in 0..8 {
                    result[i] = -v[i];
                }
                Ok(Value::Vec8(result))
            }

            Value::Gf8(g) => Ok(Value::Gf8(-*g)),

            Value::Octonion(o) => Ok(Value::Octonion(Octonion {
                scalar: -o.scalar,
                i: o.i.map(|x| -x),
            })),

            _ => Err(EvalError::TypeMismatch(format!("Cannot negate {:?}", self))),
        }
    }

    /// Less than comparison
    pub fn lt(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Bool(a < b)),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot compare {:?} < {:?}",
                self, other
            ))),
        }
    }

    /// Less than or equal comparison
    pub fn le(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Bool(a <= b)),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot compare {:?} <= {:?}",
                self, other
            ))),
        }
    }

    /// Greater than comparison
    pub fn gt(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Bool(a > b)),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot compare {:?} > {:?}",
                self, other
            ))),
        }
    }

    /// Greater than or equal comparison
    pub fn ge(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Bool(a >= b)),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot compare {:?} >= {:?}",
                self, other
            ))),
        }
    }

    /// Logical AND
    pub fn and(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a && *b)),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot apply AND to {:?} and {:?}",
                self, other
            ))),
        }
    }

    /// Logical OR
    pub fn or(&self, other: &Value) -> Result<Value, EvalError> {
        match (self, other) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a || *b)),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot apply OR to {:?} and {:?}",
                self, other
            ))),
        }
    }

    /// Compute dot product (SIMD-accelerated when available)
    #[cfg(feature = "simd")]
    pub fn dot_simd(&self, other: &Value) -> Result<f32, EvalError> {
        match (self, other) {
            (Value::Gf8(a), Value::Gf8(b)) => Ok(super::gf8::gf8_dot_simd(a.coords(), b.coords())),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot compute dot product of {:?} and {:?}",
                self, other
            ))),
        }
    }

    /// Compute squared norm (SIMD-accelerated when available)
    #[cfg(feature = "simd")]
    pub fn norm2_simd(&self) -> Result<f32, EvalError> {
        match self {
            Value::Gf8(a) => Ok(super::gf8::gf8_norm2_simd(a.coords())),
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot compute norm of {:?}",
                self
            ))),
        }
    }

    /// Apply matrix transformation (SIMD-accelerated when available)
    #[cfg(feature = "simd")]
    pub fn matrix_transform(&self, matrix: &[[f32; 8]; 8]) -> Result<Value, EvalError> {
        match self {
            Value::Gf8(a) => {
                let result_coords = super::gf8::gf8_matvec_simd(matrix, a.coords());
                Ok(Value::Gf8(super::gf8::Gf8::new(result_coords)))
            }
            _ => Err(EvalError::TypeMismatch(format!(
                "Cannot apply matrix transform to {:?}",
                self
            ))),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(b) => write!(f, "{}", b),
            Value::Float(v) => write!(f, "{}", v),
            Value::String(s) => write!(f, "{}", s),
            Value::Array(arr) => {
                write!(f, "[")?;
                for (i, val) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", val)?;
                }
                write!(f, "]")
            }
            Value::Scalar(v) => write!(f, "{}", v),
            Value::Gf8(g) => write!(f, "Gf8({})", g.to_scalar()),
            Value::Vec8(v) => write!(
                f,
                "Vec8[{}, {}, {}, {}, {}, {}, {}, {}]",
                v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]
            ),
            Value::Vec16(v) => write!(
                f,
                "Vec16[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]",
                v[0],
                v[1],
                v[2],
                v[3],
                v[4],
                v[5],
                v[6],
                v[7],
                v[8],
                v[9],
                v[10],
                v[11],
                v[12],
                v[13],
                v[14],
                v[15]
            ),
            Value::Octonion(o) => write!(f, "{}", o),
            Value::Quaternion(q) => write!(f, "Quat[{}, {}, {}, {}]", q[0], q[1], q[2], q[3]),
            Value::Symbol(s) => write!(f, "{}", s),
            Value::Matrix8x8(_) => write!(f, "Matrix8x8[...]"),
            Value::Betti(b) => write!(f, "Betti[{}, {}, {}]", b[0], b[1], b[2]),
            Value::PointCloud(points) => write!(f, "PointCloud[{} points]", points.len()),
            Value::Error(e) => write!(f, "Error: {}", e),
        }
    }
}

/// Evaluation context with variable bindings and root context
#[derive(Debug, Clone)]
pub struct EvalContext {
    /// Variable bindings
    pub variables: HashMap<String, Value>,

    /// Semantic variable bindings (prefix:name -> value)
    pub semantic_vars: HashMap<String, Value>,

    /// Current root context (affects interpretation)
    root: Option<String>,
}

impl EvalContext {
    /// Create a new evaluation context
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            semantic_vars: HashMap::new(),
            root: None,
        }
    }

    /// Set the root context
    pub fn set_root(&mut self, root: String) {
        self.root = Some(root);
    }

    /// Get the current root context
    pub fn root(&self) -> Option<&str> {
        self.root.as_deref()
    }

    /// Bind a variable to a value
    pub fn bind(&mut self, name: String, value: Value) {
        self.variables.insert(name, value);
    }

    /// Look up a variable
    pub fn lookup(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }
}

impl Default for EvalContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluation errors
#[derive(Debug, Error)]
pub enum EvalError {
    #[error("Type mismatch: {0}")]
    TypeMismatch(String),

    #[error("Undefined variable: {0}")]
    UndefinedVariable(String),

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

// ===================================
// From trait implementations - automatic Value wrapping
// ===================================

impl From<[f32; 8]> for Value {
    fn from(arr: [f32; 8]) -> Self {
        Value::Vec8(arr)
    }
}

impl From<[f32; 16]> for Value {
    fn from(arr: [f32; 16]) -> Self {
        Value::Vec16(arr)
    }
}

impl From<[f32; 4]> for Value {
    fn from(arr: [f32; 4]) -> Self {
        Value::Quaternion(arr)
    }
}

impl From<[u32; 3]> for Value {
    fn from(arr: [u32; 3]) -> Self {
        Value::Betti(arr)
    }
}

// ===================================
// RuneBuiltin - Geometric Operation Dispatch
// ===================================

/// Built-in geometric operations that bridge RUNE into Hydron
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuneBuiltin {
    // Spherical (S7) operations
    S7Project,  // [f32;8] → [f32;8]
    S7Distance, // [f32;8], [f32;8] → f32
    S7Slerp,    // [f32;8], [f32;8], f32 → [f32;8]

    // Quaternion operations
    QuatSlerp, // [f32;4], [f32;4], f32 → [f32;4]

    // Symplectic operations
    SymHamiltonian, // [f32;16] → f32
    SymEvolveStep,  // [f32;16], f32 → [f32;16]

    // Topological operations
    TopoBetti,     // [[f32;8]] → [u32;3]
    TopoSignature, // [[f32;8]] → symbol
}

impl EvalContext {
    /// Apply a built-in geometric operation
    ///
    /// This is the bridge layer that makes RUNE expressions actually drive Hydron geometry.
    pub fn apply_builtin(&self, op: RuneBuiltin, args: &[Value]) -> Result<Value, EvalError> {
        match op {
            // Spherical S7 operations
            RuneBuiltin::S7Project => {
                let v = expect_vec8(args.first())?;
                let projected = SphericalLayer::project(&v);
                Ok(Value::Vec8(projected))
            }

            RuneBuiltin::S7Distance => {
                let a = expect_vec8(args.first())?;
                let b = expect_vec8(args.get(1))?;
                let dist = SphericalLayer::distance(&a, &b);
                Ok(Value::Scalar(dist))
            }

            RuneBuiltin::S7Slerp => {
                let a = expect_vec8(args.first())?;
                let b = expect_vec8(args.get(1))?;
                let t = expect_scalar(args.get(2))?;
                let result = SphericalLayer::slerp(&a, &b, t);
                Ok(Value::Vec8(result))
            }

            // Quaternion operations
            RuneBuiltin::QuatSlerp => {
                let a = expect_quat(args.first())?;
                let b = expect_quat(args.get(1))?;
                let t = expect_scalar(args.get(2))?;
                let result = QuaternionOps::slerp(&a, &b, t);
                Ok(Value::Quaternion(result))
            }

            // Symplectic operations
            RuneBuiltin::SymHamiltonian => {
                let state = expect_vec16(args.first())?;
                let (q, p) = split_phase_space(&state);
                let layer = SymplecticLayer::new();
                let h = layer.hamiltonian(&q, &p);
                Ok(Value::Scalar(h))
            }

            RuneBuiltin::SymEvolveStep => {
                let state = expect_vec16(args.first())?;
                let dt = expect_scalar(args.get(1))?;
                let (mut q, mut p) = split_phase_space(&state);
                let layer = SymplecticLayer::new();
                layer.evolve(&mut q, &mut p, dt);
                let evolved = merge_phase_space(&q, &p);
                Ok(Value::Vec16(evolved))
            }

            // Topological operations
            RuneBuiltin::TopoBetti => {
                let points = extract_point_cloud(args)?;
                let mut layer = TopologicalLayer::new();
                for point in points {
                    layer.add_point(point);
                }
                layer.compute_betti_numbers(2.0, 10); // max_radius=2.0, steps=10
                Ok(Value::Betti(layer.betti))
            }

            RuneBuiltin::TopoSignature => {
                let points = extract_point_cloud(args)?;
                let mut layer = TopologicalLayer::new();
                for point in points {
                    layer.add_point(point);
                }
                layer.compute_betti_numbers(2.0, 10);
                let sig = format!("β={:?}", layer.betti);
                Ok(Value::Symbol(sig))
            }
        }
    }
}

// ===================================
// Helper functions for type extraction
// ===================================

fn expect_vec8(val: Option<&Value>) -> Result<[f32; 8], EvalError> {
    match val {
        Some(Value::Vec8(v)) => Ok(*v),
        Some(other) => Err(EvalError::TypeMismatch(format!(
            "Expected Vec8, got {}",
            match other {
                Value::Scalar(_) => "Scalar",
                Value::Vec16(_) => "Vec16",
                Value::Quaternion(_) => "Quaternion",
                Value::Gf8(_) => "Gf8",
                Value::Octonion(_) => "Octonion",
                Value::Symbol(_) => "Symbol",
                Value::Matrix8x8(_) => "Matrix8x8",
                Value::Betti(_) => "Betti",
                _ => "unknown",
            }
        ))),
        None => Err(EvalError::InvalidOperation("Missing argument".to_string())),
    }
}

fn expect_vec16(val: Option<&Value>) -> Result<[f32; 16], EvalError> {
    match val {
        Some(Value::Vec16(v)) => Ok(*v),
        Some(_) => Err(EvalError::TypeMismatch("Expected Vec16".to_string())),
        None => Err(EvalError::InvalidOperation("Missing argument".to_string())),
    }
}

fn expect_quat(val: Option<&Value>) -> Result<[f32; 4], EvalError> {
    match val {
        Some(Value::Quaternion(q)) => Ok(*q),
        Some(_) => Err(EvalError::TypeMismatch("Expected Quaternion".to_string())),
        None => Err(EvalError::InvalidOperation("Missing argument".to_string())),
    }
}

fn expect_scalar(val: Option<&Value>) -> Result<f32, EvalError> {
    match val {
        Some(Value::Scalar(s)) => Ok(*s),
        Some(_) => Err(EvalError::TypeMismatch("Expected Scalar".to_string())),
        None => Err(EvalError::InvalidOperation("Missing argument".to_string())),
    }
}

fn extract_point_cloud(args: &[Value]) -> Result<Vec<[f32; 8]>, EvalError> {
    // Handle multiple argument formats:
    // 1. Single PointCloud value
    // 2. Single Vec16 (two packed points)
    // 3. Multiple Vec8 arguments

    if args.is_empty() {
        return Err(EvalError::InvalidOperation(
            "No points provided".to_string(),
        ));
    }

    // Case 1: PointCloud value
    if args.len() == 1 {
        if let Value::PointCloud(points) = &args[0] {
            return Ok(points.clone());
        }

        // Case 2: Vec16 (two packed points)
        if let Value::Vec16(v16) = &args[0] {
            let p1 = [
                v16[0], v16[1], v16[2], v16[3], v16[4], v16[5], v16[6], v16[7],
            ];
            let p2 = [
                v16[8], v16[9], v16[10], v16[11], v16[12], v16[13], v16[14], v16[15],
            ];
            return Ok(vec![p1, p2]);
        }
    }

    // Case 3: Multiple Vec8 arguments
    let mut points = Vec::new();
    for arg in args {
        match arg {
            Value::Vec8(v) => points.push(*v),
            Value::PointCloud(pc) => points.extend_from_slice(pc),
            _ => {
                return Err(EvalError::TypeMismatch(
                    "Expected Vec8, Vec16, or PointCloud for point cloud".to_string(),
                ));
            }
        }
    }

    if points.is_empty() {
        return Err(EvalError::InvalidOperation("Empty point cloud".to_string()));
    }

    Ok(points)
}

/// Split Vec16 phase space into position and momentum
fn split_phase_space(state: &[f32; 16]) -> ([f32; 8], [f32; 8]) {
    let mut q = [0.0f32; 8];
    let mut p = [0.0f32; 8];
    q.copy_from_slice(&state[..8]);
    p.copy_from_slice(&state[8..]);
    (q, p)
}

/// Merge position and momentum into Vec16 phase space
fn merge_phase_space(q: &[f32; 8], p: &[f32; 8]) -> [f32; 16] {
    let mut state = [0.0f32; 16];
    state[..8].copy_from_slice(q);
    state[8..].copy_from_slice(p);
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_arithmetic() {
        let a = Value::Scalar(5.0);
        let b = Value::Scalar(3.0);

        assert_eq!(a.add(&b).unwrap(), Value::Scalar(8.0));
        assert_eq!(a.mul(&b).unwrap(), Value::Scalar(15.0));
        assert_eq!(a.sub(&b).unwrap(), Value::Scalar(2.0));
    }

    #[test]
    fn test_gf8_arithmetic() {
        use crate::rune::hydron::gf8::Gf8;

        // Test Gf8 addition (geometric addition on unit sphere)
        let gf_a = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let gf_b = Gf8::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let a = Value::Gf8(gf_a);
        let b = Value::Gf8(gf_b);

        // Geometric Gf8 addition
        let result = a.add(&b).unwrap();
        assert!(matches!(result, Value::Gf8(_)));
    }

    #[test]
    fn test_octonion_multiplication() {
        let a = Octonion::real(2.0);
        let b = Octonion::real(3.0);
        let c = a.mul(&b);

        assert_eq!(c.scalar, 6.0);
    }

    #[test]
    fn test_vec8_operations() {
        let a = Value::Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = Value::Vec8([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let result = a.add(&b).unwrap();
        if let Value::Vec8(v) = result {
            assert_eq!(v[0], 2.0);
            assert_eq!(v[7], 9.0);
        }
    }

    // ===================================
    // Integration Tests: RUNE → Hydron Geometry
    // ===================================

    #[test]
    fn test_rune_drives_spherical_geometry() {
        let ctx = EvalContext::new();

        // Test S7 projection
        let v = Value::Vec8([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let result = ctx.apply_builtin(RuneBuiltin::S7Project, &[v]).unwrap();

        if let Value::Vec8(projected) = result {
            // Should be normalized to unit sphere
            let norm: f32 = projected.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "S7 projection should normalize");
        } else {
            panic!("Expected Vec8 result from S7Project");
        }

        // Test S7 distance
        let a = Value::Vec8([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = Value::Vec8([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let dist = ctx
            .apply_builtin(RuneBuiltin::S7Distance, &[a.clone(), b.clone()])
            .unwrap();

        if let Value::Scalar(d) = dist {
            assert!(
                d > 0.0,
                "Distance between distinct points should be positive"
            );
        }

        // Test S7 slerp
        let t = Value::Scalar(0.5);
        let interp = ctx.apply_builtin(RuneBuiltin::S7Slerp, &[a, b, t]).unwrap();

        assert!(matches!(interp, Value::Vec8(_)), "Slerp should return Vec8");
    }

    #[test]
    fn test_rune_drives_symplectic_geometry() {
        let ctx = EvalContext::new();

        // Create a symplectic state (position + momentum)
        let state = Value::Vec16([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // position
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // momentum
        ]);

        // Test Hamiltonian computation
        let h = ctx
            .apply_builtin(RuneBuiltin::SymHamiltonian, &[state.clone()])
            .unwrap();

        if let Value::Scalar(energy) = h {
            assert!(energy >= 0.0, "Hamiltonian should be non-negative");
        } else {
            panic!("Expected Scalar from SymHamiltonian");
        }

        // Test symplectic evolution
        let dt = Value::Scalar(0.1);
        let evolved = ctx
            .apply_builtin(RuneBuiltin::SymEvolveStep, &[state, dt])
            .unwrap();

        assert!(
            matches!(evolved, Value::Vec16(_)),
            "Symplectic evolution should return Vec16"
        );
    }

    #[test]
    fn test_rune_drives_topological_analysis() {
        let ctx = EvalContext::new();

        // Create a point cloud (2 points packed into Vec16)
        let points = Value::Vec16([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // point 1
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // point 2
        ]);

        // Test Betti number computation
        let betti = ctx
            .apply_builtin(RuneBuiltin::TopoBetti, &[points.clone()])
            .unwrap();

        if let Value::Betti([b0, b1, b2]) = betti {
            assert!(b0 > 0, "Should have at least one connected component");
            // b1, b2 depend on point cloud structure
            let _ = (b1, b2);
        } else {
            panic!("Expected Betti from TopoBetti");
        }

        // Test topological signature
        let sig = ctx
            .apply_builtin(RuneBuiltin::TopoSignature, &[points])
            .unwrap();

        assert!(
            matches!(sig, Value::Symbol(_)),
            "Topological signature should return Symbol"
        );
    }

    #[test]
    fn test_from_trait_conversions() {
        // Test automatic Value wrapping
        let v8: Value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].into();
        assert!(matches!(v8, Value::Vec8(_)));

        let v16: Value = [0.0; 16].into();
        assert!(matches!(v16, Value::Vec16(_)));

        let quat: Value = [1.0, 0.0, 0.0, 0.0].into();
        assert!(matches!(quat, Value::Quaternion(_)));

        let betti: Value = [1, 0, 0].into();
        assert!(matches!(betti, Value::Betti(_)));
    }
}

File: tui\components\editor.rs
==============================
//! Input and output editor panels.

use ratatui::{
    Frame,
    layout::Rect,
    widgets::{Block, Borders},
};

use crate::tui::{state::AppState, theme::Theme};

pub struct EditorComponent;

impl EditorComponent {
    pub fn render(
        f: &mut Frame,
        input_area: Rect,
        output_area: Rect,
        app: &mut AppState,
        theme: &Theme,
    ) {
        let input_active = app.editor.is_input_active();
        let input_title = format!(
            " Input ({}) {} ",
            match app.mode {
                crate::tui::state::app_state::Mode::Encode => "JSON",
                crate::tui::state::app_state::Mode::Decode => "TOON",
                crate::tui::state::app_state::Mode::Rune => "RUNE",
            },
            if input_active { "●" } else { "" }
        );

        let input_block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(input_active))
            .title(input_title)
            .style(theme.normal_style());

        app.editor.input.set_block(input_block);
        app.editor
            .input
            .set_cursor_line_style(theme.selection_style());
        app.editor.input.set_style(theme.normal_style());

        f.render_widget(&app.editor.input, input_area);

        let output_active = app.editor.is_output_active();
        let output_title = format!(
            " Output ({}) {} ",
            match app.mode {
                crate::tui::state::app_state::Mode::Encode => "TOON",
                crate::tui::state::app_state::Mode::Decode => "JSON",
                crate::tui::state::app_state::Mode::Rune => "Results",
            },
            if output_active { "●" } else { "" }
        );

        let output_block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(output_active))
            .title(output_title)
            .style(theme.normal_style());

        app.editor.output.set_block(output_block);
        app.editor
            .output
            .set_cursor_line_style(theme.selection_style());
        app.editor.output.set_style(theme.normal_style());

        f.render_widget(&app.editor.output, output_area);
    }
}

File: tui\components\confirmation_dialog.rs
===========================================
/* src/tui/components/confirmation_dialog.rs */
//! Terminal UI confirmation dialog component for user interactions.
//!
//! # TOON-RUNE – Confirmation Dialog Component
//!▫~•◦-----------------------------------------‣
//!
//! This module provides a modal confirmation dialog component for the TOON-RUNE
//! terminal user interface, handling user confirmations for destructive actions.
//!
//! ### Key Capabilities
//! - **Modal Display**: Renders centered confirmation dialogs with styled borders and content.
//! - **Action Variants**: Supports different confirmation types (New File, Quit, Delete File).
//! - **Keyboard Navigation**: Visual cues for Y/N/Esc key bindings.
//! - **Responsive Layout**: Automatically centers and sizes dialog within terminal viewport.
//!
//! ### Architectural Notes
//! This component integrates with the `AppState` type's `ConfirmationAction` enum and
//! works alongside other TUI components like editors and file browsers. Dialog rendering
//! uses Ratatui's layout system for consistent positioning and styling.
//!
//! ### Example
//! ```rust
//! use rune_format::tui::components::confirmation_dialog::ConfirmationDialog;
//! use rune_format::tui::state::app_state::ConfirmationAction;
//! use ratatui::{Frame, layout::Rect};
//!
//! let action = ConfirmationAction::DeleteFile;
//! // In your TUI rendering loop:
//! // confirmation_dialog::render(&mut frame, dialog_area, action);
//!
//! // The dialog renders with appropriate title and message for the delete action.
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};

use crate::tui::state::app_state::ConfirmationAction;

pub struct ConfirmationDialog;

impl ConfirmationDialog {
    pub fn render(frame: &mut Frame, area: Rect, action: ConfirmationAction) {
        let (title, message) = match action {
            ConfirmationAction::NewFile => (
                "New File",
                "Current file has unsaved changes. Create new file anyway?",
            ),
            ConfirmationAction::Quit => ("Quit", "Current file has unsaved changes. Quit anyway?"),
            ConfirmationAction::DeleteFile => (
                "Delete File",
                "Are you sure you want to delete this file? This cannot be undone.",
            ),
            ConfirmationAction::None => return,
        };

        // Create centered modal
        let popup_area = Self::centered_rect(50, 30, area);

        // Clear the area
        frame.render_widget(Clear, popup_area);

        // Create layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Title
                Constraint::Min(3),    // Message
                Constraint::Length(3), // Buttons
            ])
            .split(popup_area);

        // Render border
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow))
            .title(title)
            .title_alignment(Alignment::Center);
        frame.render_widget(block, popup_area);

        // Render message
        let message_paragraph = Paragraph::new(message)
            .style(Style::default().fg(Color::White))
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true });
        frame.render_widget(message_paragraph, chunks[1]);

        // Render buttons
        let buttons = Line::from(vec![
            Span::styled(
                "[Y]",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" Yes    "),
            Span::styled(
                "[N]",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" No    "),
            Span::styled("[ESC]", Style::default().fg(Color::Gray)),
            Span::raw(" Cancel"),
        ]);
        let buttons_paragraph = Paragraph::new(buttons).alignment(Alignment::Center);
        frame.render_widget(buttons_paragraph, chunks[2]);
    }

    fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
        let popup_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage((100 - percent_y) / 2),
                Constraint::Percentage(percent_y),
                Constraint::Percentage((100 - percent_y) / 2),
            ])
            .split(r);

        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage((100 - percent_x) / 2),
                Constraint::Percentage(percent_x),
                Constraint::Percentage((100 - percent_x) / 2),
            ])
            .split(popup_layout[1])[1]
    }
}

File: tui\components\diff_viewer.rs
===================================
//! Side-by-side diff viewer for input/output comparison.

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

use crate::tui::{state::AppState, theme::Theme};

pub struct DiffViewer;

impl DiffViewer {
    pub fn render(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(true))
            .title(" Side-by-Side Comparison - Press Esc to close ")
            .title_alignment(Alignment::Center);

        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(inner);

        let input_text = app.editor.get_input();
        let input_title = match app.mode {
            crate::tui::state::app_state::Mode::Encode => "JSON Input",
            crate::tui::state::app_state::Mode::Decode => "TOON Input",
            crate::tui::state::app_state::Mode::Rune => "RUNE Input",
        };

        let input_lines: Vec<Line> = input_text
            .lines()
            .enumerate()
            .map(|(idx, line)| {
                Line::from(vec![
                    Span::styled(format!("{:4} ", idx + 1), theme.line_number_style()),
                    Span::styled(line, theme.normal_style()),
                ])
            })
            .collect();

        let input_para = Paragraph::new(input_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border_style(false))
                    .title(format!(" {input_title} ")),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(input_para, chunks[0]);

        let output_text = app.editor.get_output();
        let output_title = match app.mode {
            crate::tui::state::app_state::Mode::Encode => "TOON Output",
            crate::tui::state::app_state::Mode::Decode => "JSON Output",
            crate::tui::state::app_state::Mode::Rune => "Parsed Results",
        };

        let output_lines: Vec<Line> = output_text
            .lines()
            .enumerate()
            .map(|(idx, line)| {
                Line::from(vec![
                    Span::styled(format!("{:4} ", idx + 1), theme.line_number_style()),
                    Span::styled(line, theme.normal_style()),
                ])
            })
            .collect();

        let output_para = Paragraph::new(output_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border_style(false))
                    .title(format!(" {output_title} ")),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(output_para, chunks[1]);
    }
}

File: tui\components\file_browser.rs
====================================
//! File browser for opening JSON/TOON files.

use std::fs;

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};

use crate::tui::{state::AppState, theme::Theme};

/// File browser state and rendering.
pub struct FileBrowser {
    pub selected_index: usize,
    pub scroll_offset: usize,
}

impl FileBrowser {
    pub fn new() -> Self {
        Self {
            selected_index: 0,
            scroll_offset: 0,
        }
    }

    pub fn move_up(&mut self) {
        if self.selected_index > 0 {
            self.selected_index -= 1;
            if self.selected_index < self.scroll_offset {
                self.scroll_offset = self.selected_index;
            }
        }
    }

    pub fn move_down(&mut self, max: usize) {
        if self.selected_index < max.saturating_sub(1) {
            self.selected_index += 1;
        }
    }

    pub fn get_selected_entry(&self, dir: &std::path::Path) -> Option<std::path::PathBuf> {
        let entries = self.get_directory_entries(dir);
        if self.selected_index < entries.len() {
            let (name, _is_dir, _, _) = &entries[self.selected_index];
            if name == ".." {
                dir.parent().map(|p| p.to_path_buf())
            } else {
                Some(dir.join(name))
            }
        } else {
            None
        }
    }

    pub fn get_entry_count(&self, dir: &std::path::Path) -> usize {
        self.get_directory_entries(dir).len()
    }

    pub fn render(&mut self, f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(true))
            .title(" File Browser - Press Esc to close ")
            .title_alignment(Alignment::Center);

        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Min(10),
                Constraint::Length(3),
            ])
            .split(inner);

        let current_dir = Paragraph::new(Line::from(vec![
            Span::styled("Current: ", theme.line_number_style()),
            Span::styled(
                app.file_state.current_dir.display().to_string(),
                theme.info_style(),
            ),
        ]));
        f.render_widget(current_dir, chunks[0]);

        let entries = self.get_directory_entries(&app.file_state.current_dir);
        let items: Vec<ListItem> = entries
            .iter()
            .enumerate()
            .map(|(idx, (name, is_dir, is_json, is_toon))| {
                let icon = if *is_dir {
                    "📁"
                } else if *is_json {
                    "📄"
                } else if *is_toon {
                    "📋"
                } else {
                    "📃"
                };

                let style = if idx == self.selected_index {
                    theme.selection_style()
                } else if *is_json || *is_toon {
                    theme.highlight_style()
                } else {
                    theme.normal_style()
                };

                ListItem::new(Line::from(vec![
                    Span::styled(format!("  {icon} "), style),
                    Span::styled(name, style),
                ]))
            })
            .collect();

        let list = List::new(items);
        f.render_widget(list, chunks[1]);

        let instructions = Paragraph::new(Line::from(vec![
            Span::styled("↑↓", theme.info_style()),
            Span::styled(" Navigate | ", theme.line_number_style()),
            Span::styled("Enter", theme.info_style()),
            Span::styled(" Open | ", theme.line_number_style()),
            Span::styled("Space", theme.info_style()),
            Span::styled(" Select | ", theme.line_number_style()),
            Span::styled("Esc", theme.info_style()),
            Span::styled(" Close", theme.line_number_style()),
        ]))
        .alignment(Alignment::Center);
        f.render_widget(instructions, chunks[2]);
    }

    fn get_directory_entries(&self, dir: &std::path::Path) -> Vec<(String, bool, bool, bool)> {
        let mut entries = vec![("..".to_string(), true, false, false)];

        if let Ok(read_dir) = fs::read_dir(dir) {
            let mut files: Vec<_> = read_dir
                .filter_map(|entry| entry.ok())
                .filter_map(|entry| {
                    let path = entry.path();
                    let name = path.file_name()?.to_str()?.to_string();
                    let is_dir = path.is_dir();
                    let is_json =
                        !is_dir && path.extension().and_then(|e| e.to_str()) == Some("json");
                    let is_toon =
                        !is_dir && path.extension().and_then(|e| e.to_str()) == Some("toon");
                    Some((name, is_dir, is_json, is_toon))
                })
                .collect();

            files.sort_by(|a, b| {
                if a.1 == b.1 {
                    a.0.cmp(&b.0)
                } else {
                    b.1.cmp(&a.1)
                }
            });

            entries.extend(files);
        }

        entries
    }
}

impl Default for FileBrowser {
    fn default() -> Self {
        Self::new()
    }
}

File: tui\components\history_panel.rs
=====================================
//! Conversion history panel.

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};

use crate::tui::{state::AppState, theme::Theme};

pub struct HistoryPanel;

impl HistoryPanel {
    pub fn render(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(true))
            .title(" Conversion History - Press Esc to close ")
            .title_alignment(Alignment::Center);

        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(2), Constraint::Min(10)])
            .split(inner);

        let title = Paragraph::new(Line::from(Span::styled(
            format!("Total conversions: {}", app.file_state.history.len()),
            theme.info_style(),
        )))
        .alignment(Alignment::Center);
        f.render_widget(title, chunks[0]);

        if app.file_state.history.is_empty() {
            let empty = Paragraph::new(Line::from(Span::styled(
                "No conversion history yet",
                theme.line_number_style(),
            )))
            .alignment(Alignment::Center);
            f.render_widget(empty, chunks[1]);
        } else {
            let items: Vec<ListItem> = app
                .file_state
                .history
                .iter()
                .rev()
                .map(|entry| {
                    let time_str = entry.timestamp.format("%H:%M:%S").to_string();
                    let file_str = entry
                        .input_file
                        .as_ref()
                        .and_then(|p| p.file_name())
                        .and_then(|n| n.to_str())
                        .unwrap_or("stdin");

                    ListItem::new(Line::from(vec![
                        Span::styled(format!("  {time_str} "), theme.line_number_style()),
                        Span::styled(format!("[{}] ", entry.mode), theme.info_style()),
                        Span::styled(file_str, theme.normal_style()),
                        Span::styled(
                            format!(" → {:.1}% saved", entry.token_savings),
                            if entry.token_savings > 0.0 {
                                theme.success_style()
                            } else {
                                theme.warning_style()
                            },
                        ),
                    ]))
                })
                .collect();

            let list = List::new(items);
            f.render_widget(list, chunks[1]);
        }
    }
}

File: tui\components\settings_panel.rs
======================================
//! Settings panel for configuring encode/decode options.

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};

use crate::{
    tui::{state::AppState, theme::Theme},
    types::{Delimiter, Indent, KeyFoldingMode, PathExpansionMode},
};

pub struct SettingsPanel;

impl SettingsPanel {
    pub fn render(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(true))
            .title(" Settings - Press Ctrl+P or Esc to close ")
            .title_alignment(Alignment::Center);

        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(3),
            ])
            .split(inner);

        let title = Paragraph::new(Line::from(Span::styled(
            format!("Current Mode: {}", app.mode.as_str()),
            theme.title_style(),
        )))
        .alignment(Alignment::Center);
        f.render_widget(title, chunks[0]);

        let mut items = vec![];

        items.push(ListItem::new(Line::from(Span::styled(
            "═══ Encode Settings (JSON → TOON) ═══",
            theme.title_style(),
        ))));

        let delimiter_str = match app.encode_options.delimiter {
            Delimiter::Comma => "Comma (,)",
            Delimiter::Tab => "Tab (\\t)",
            Delimiter::Pipe => "Pipe (|)",
        };
        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Delimiter:       ", theme.info_style()),
            Span::styled(delimiter_str, theme.normal_style()),
            Span::styled("  [Press 'd' to cycle]", theme.line_number_style()),
        ])));

        let Indent::Spaces(indent_spaces) = app.encode_options.indent;
        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Indentation:     ", theme.info_style()),
            Span::styled(format!("{indent_spaces} spaces"), theme.normal_style()),
            Span::styled("  [+/- to adjust]", theme.line_number_style()),
        ])));

        let fold_keys = match app.encode_options.key_folding {
            KeyFoldingMode::Off => "Off",
            KeyFoldingMode::Safe => "On (Safe)",
        };
        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Key Folding:     ", theme.info_style()),
            Span::styled(fold_keys, theme.normal_style()),
            Span::styled("  [Press 'f' to toggle]", theme.line_number_style()),
        ])));

        if app.encode_options.key_folding != KeyFoldingMode::Off {
            items.push(ListItem::new(Line::from(vec![
                Span::styled("  Flatten Depth:   ", theme.info_style()),
                Span::styled(
                    if app.encode_options.flatten_depth == usize::MAX {
                        "Unlimited".to_string()
                    } else {
                        format!("{}", app.encode_options.flatten_depth)
                    },
                    theme.normal_style(),
                ),
                Span::styled(
                    "  [[/] to adjust, [u] for unlimited]",
                    theme.line_number_style(),
                ),
            ])));
        }

        items.push(ListItem::new(Line::from("")));

        items.push(ListItem::new(Line::from(Span::styled(
            "═══ Decode Settings (TOON → JSON) ═══",
            theme.title_style(),
        ))));

        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Strict Mode:     ", theme.info_style()),
            Span::styled(
                if app.decode_options.strict {
                    "On"
                } else {
                    "Off"
                },
                theme.normal_style(),
            ),
            Span::styled("  [Press 's' to toggle]", theme.line_number_style()),
        ])));

        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Type Coercion:   ", theme.info_style()),
            Span::styled(
                if app.decode_options.coerce_types {
                    "On"
                } else {
                    "Off"
                },
                theme.normal_style(),
            ),
            Span::styled("  [Press 'c' to toggle]", theme.line_number_style()),
        ])));

        let expand_paths = match app.decode_options.expand_paths {
            PathExpansionMode::Off => "Off",
            PathExpansionMode::Safe => "On (Safe)",
        };
        items.push(ListItem::new(Line::from(vec![
            Span::styled("  Path Expansion:  ", theme.info_style()),
            Span::styled(expand_paths, theme.normal_style()),
            Span::styled("  [Press 'p' to toggle]", theme.line_number_style()),
        ])));

        let list = List::new(items);
        f.render_widget(list, chunks[1]);

        let instructions = Paragraph::new(Line::from(vec![
            Span::styled("Press ", theme.line_number_style()),
            Span::styled("Ctrl+E", theme.info_style()),
            Span::styled(" to toggle mode | ", theme.line_number_style()),
            Span::styled("Ctrl+R", theme.info_style()),
            Span::styled(" to refresh conversion", theme.line_number_style()),
        ]))
        .alignment(Alignment::Center);
        f.render_widget(instructions, chunks[2]);
    }
}

File: tui\components\mod.rs
===========================
//! UI components for the TUI.

pub mod confirmation_dialog;
pub mod diff_viewer;
pub mod editor;
pub mod file_browser;
pub mod help_screen;
pub mod history_panel;
pub mod repl_panel;
pub mod settings_panel;
pub mod stats_bar;
pub mod status_bar;

pub use confirmation_dialog::ConfirmationDialog;
pub use diff_viewer::DiffViewer;
pub use editor::EditorComponent;
pub use file_browser::FileBrowser;
pub use help_screen::HelpScreen;
pub use history_panel::HistoryPanel;
pub use repl_panel::ReplPanel;
pub use settings_panel::SettingsPanel;
pub use stats_bar::StatsBar;
pub use status_bar::StatusBar;

File: tui\components\repl_panel.rs
==================================
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Margin, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap},
};

use crate::tui::state::{AppState, ReplLineKind};

pub struct ReplPanel;

impl ReplPanel {
    pub fn render(f: &mut Frame, area: Rect, app: &mut AppState) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(10), Constraint::Length(3)])
            .split(area);

        Self::render_output(f, chunks[0], app);
        Self::render_input(f, chunks[1], app);
    }

    fn render_output(f: &mut Frame, area: Rect, app: &AppState) {
        let lines: Vec<Line> = app
            .repl
            .output
            .iter()
            .skip(app.repl.scroll_offset)
            .map(|line| {
                let style = match line.kind {
                    ReplLineKind::Prompt => Style::default().fg(Color::Cyan),
                    ReplLineKind::Success => Style::default().fg(Color::Green),
                    ReplLineKind::Error => Style::default().fg(Color::Red),
                    ReplLineKind::Info => Style::default().fg(Color::Yellow),
                };
                Line::from(Span::styled(&line.content, style))
            })
            .collect();

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(" REPL Session (Ctrl+R to toggle, Esc to close) ");

        let paragraph = Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false });

        f.render_widget(paragraph, area);

        if app.repl.output.len() > (area.height as usize - 2) {
            let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("↑"))
                .end_symbol(Some("↓"));

            let mut scrollbar_state =
                ScrollbarState::new(app.repl.output.len()).position(app.repl.scroll_offset);

            f.render_stateful_widget(
                scrollbar,
                area.inner(Margin {
                    vertical: 1,
                    horizontal: 0,
                }),
                &mut scrollbar_state,
            );
        }
    }

    fn render_input(f: &mut Frame, area: Rect, app: &AppState) {
        let prompt = Span::styled(
            "> ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );

        let input_text = Span::raw(&app.repl.input);
        let cursor = Span::styled("█", Style::default().fg(Color::White));

        let line = Line::from(vec![prompt, input_text, cursor]);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));

        let paragraph = Paragraph::new(line).block(block);

        f.render_widget(paragraph, area);
    }
}

File: tui\components\help_screen.rs
===================================
//! Help screen showing keyboard shortcuts.

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};

use crate::tui::{keybindings::KeyBindings, theme::Theme};

pub struct HelpScreen;

impl HelpScreen {
    pub fn render(f: &mut Frame, area: Rect, theme: &Theme) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style(true))
            .title(" Help - Press F1 or Esc to close ")
            .title_alignment(Alignment::Center);

        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(5),
            ])
            .split(inner);

        let title = Paragraph::new(vec![
            Line::from(Span::styled(
                "TOON Format - Interactive TUI",
                theme.title_style(),
            )),
            Line::from(Span::styled(
                "Token-Oriented Object Notation",
                theme.info_style(),
            )),
        ])
        .alignment(Alignment::Center);
        f.render_widget(title, chunks[0]);

        let shortcuts = KeyBindings::shortcuts();
        let items: Vec<ListItem> = shortcuts
            .iter()
            .map(|(key, desc)| {
                ListItem::new(Line::from(vec![
                    Span::styled(format!("  {key:18} "), theme.info_style()),
                    Span::styled(*desc, theme.normal_style()),
                ]))
            })
            .collect();

        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border_style(false))
                .title(" Keyboard Shortcuts "),
        );
        f.render_widget(list, chunks[1]);

        let footer = Paragraph::new(vec![
            Line::from(Span::styled(
                "TOON is a compact, human-readable format for passing structured data to LLMs",
                theme.normal_style(),
            )),
            Line::from(vec![
                Span::styled("Repository: ", theme.line_number_style()),
                Span::styled("github.com/toon-format/toon-rust", theme.info_style()),
            ]),
        ])
        .alignment(Alignment::Center);
        f.render_widget(footer, chunks[2]);
    }
}

File: tui\components\stats_bar.rs
=================================
//! Statistics bar showing token and byte savings.

use ratatui::{
    Frame,
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

use crate::tui::{state::AppState, theme::Theme};

pub struct StatsBar;

impl StatsBar {
    pub fn render(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
        if let Some(ref stats) = app.stats {
            let spans = vec![
                Span::styled(" Stats: ", theme.title_style()),
                Span::raw("Tokens: "),
                Span::styled(
                    format!("{}→{}", stats.json_tokens, stats.toon_tokens),
                    theme.info_style(),
                ),
                Span::styled(
                    format!(" ({:.1}%)", stats.token_savings),
                    if stats.token_savings > 0.0 {
                        theme.success_style()
                    } else {
                        theme.error_style()
                    },
                ),
                Span::raw(" | Bytes: "),
                Span::styled(
                    format!("{}→{}", stats.json_bytes, stats.toon_bytes),
                    theme.info_style(),
                ),
                Span::styled(
                    format!(" ({:.1}%)", stats.byte_savings),
                    if stats.byte_savings > 0.0 {
                        theme.success_style()
                    } else {
                        theme.error_style()
                    },
                ),
            ];

            let line = Line::from(spans);
            let paragraph = Paragraph::new(line).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border_style(false))
                    .title(" Statistics "),
            );

            f.render_widget(paragraph, area);
        } else {
            let paragraph = Paragraph::new(Line::from(vec![Span::styled(
                " No statistics available yet ",
                theme.line_number_style(),
            )]))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border_style(false))
                    .title(" Statistics "),
            );

            f.render_widget(paragraph, area);
        }
    }
}

File: tui\components\status_bar.rs
==================================
//! Status bar showing mode, file, and key commands.

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

use crate::tui::{state::AppState, theme::Theme};

pub struct StatusBar;

impl StatusBar {
    pub fn render(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(area);

        let mut left_spans = vec![];

        left_spans.push(Span::styled(
            format!("{} ", app.mode.short_name()),
            theme.info_style(),
        ));

        left_spans.push(Span::raw("| "));

        if let Some(ref path) = app.file_state.current_file {
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("Untitled");
            left_spans.push(Span::styled(file_name, theme.normal_style()));

            if app.file_state.is_modified {
                left_spans.push(Span::styled(" [Modified]", theme.warning_style()));
            }
        } else {
            left_spans.push(Span::styled("No file", theme.line_number_style()));
        }

        left_spans.push(Span::raw(" | "));

        if let Some(ref error) = app.error_message {
            left_spans.push(Span::styled(format!("✗ {error} "), theme.error_style()));
        } else if let Some(ref status) = app.status_message {
            left_spans.push(Span::styled(format!("✓ {status} "), theme.success_style()));
        } else {
            left_spans.push(Span::styled("Ready ", theme.normal_style()));
        }

        left_spans.push(Span::raw("| "));
        let theme_name = match theme {
            Theme::Dark => "Dark",
            Theme::Light => "Light",
        };
        left_spans.push(Span::styled(
            theme_name.to_string(),
            theme.line_number_style(),
        ));

        let left_line = Line::from(left_spans);
        let left_paragraph =
            Paragraph::new(left_line).block(Block::default().borders(Borders::ALL));

        let key_commands = vec![
            Span::styled("F1", theme.info_style()),
            Span::raw(" Help | "),
            Span::styled("Ctrl+C", theme.info_style()),
            Span::raw(" Quit"),
        ];

        let right_line = Line::from(key_commands);
        let right_paragraph = Paragraph::new(right_line)
            .block(Block::default().borders(Borders::ALL))
            .alignment(Alignment::Right);

        f.render_widget(left_paragraph, chunks[0]);
        f.render_widget(right_paragraph, chunks[1]);
    }
}

File: tui\state\app_state.rs
============================
//! Main application state.

use super::{EditorState, FileState, ReplState};
use crate::{
    tui::theme::Theme,
    types::{DecodeOptions, Delimiter, EncodeOptions, Indent, KeyFoldingMode, PathExpansionMode},
};

/// Conversion mode (encode/decode/parse).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Encode, // JSON → TOON
    Decode, // TOON → JSON
    Rune,   // RUNE → Parsed AST + TOON blocks
}

impl Mode {
    pub fn toggle(&self) -> Self {
        match self {
            Mode::Encode => Mode::Decode,
            Mode::Decode => Mode::Rune,
            Mode::Rune => Mode::Encode,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Mode::Encode => "Encode (JSON → TOON)",
            Mode::Decode => "Decode (TOON → JSON)",
            Mode::Rune => "Parse (RUNE → Results)",
        }
    }

    pub fn short_name(&self) -> &'static str {
        match self {
            Mode::Encode => "Encode",
            Mode::Decode => "Decode",
            Mode::Rune => "RUNE",
        }
    }
}

/// Statistics from the last conversion.
#[derive(Debug, Clone)]
pub struct ConversionStats {
    pub json_tokens: usize,
    pub toon_tokens: usize,
    pub json_bytes: usize,
    pub toon_bytes: usize,
    pub token_savings: f64,
    pub byte_savings: f64,
}

/// Central application state containing all UI and conversion state.
pub struct AppState<'a> {
    pub mode: Mode,
    pub editor: EditorState<'a>,
    pub file_state: FileState,
    pub repl: ReplState,
    pub theme: Theme,
    pub encode_options: EncodeOptions,
    pub decode_options: DecodeOptions,
    pub show_settings: bool,
    pub show_help: bool,
    pub show_file_browser: bool,
    pub show_history: bool,
    pub show_diff: bool,
    pub show_confirmation: bool,
    pub confirmation_action: ConfirmationAction,
    pub error_message: Option<String>,
    pub status_message: Option<String>,
    pub stats: Option<ConversionStats>,
    pub should_quit: bool,
}

/// Actions that require user confirmation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConfirmationAction {
    None,
    NewFile,
    Quit,
    DeleteFile,
}

impl<'a> AppState<'a> {
    pub fn new() -> Self {
        Self {
            mode: Mode::Encode,
            editor: EditorState::new(),
            file_state: FileState::new(),
            repl: ReplState::new(),
            theme: Theme::default(),

            encode_options: EncodeOptions::default(),
            decode_options: DecodeOptions::default(),

            show_settings: false,
            show_help: false,
            show_file_browser: false,
            show_history: false,
            show_diff: false,
            show_confirmation: false,
            confirmation_action: ConfirmationAction::None,

            error_message: None,
            status_message: None,
            stats: None,

            should_quit: false,
        }
    }

    pub fn toggle_mode(&mut self) {
        self.mode = self.mode.toggle();
        self.clear_error();
        self.clear_status();
    }

    pub fn toggle_theme(&mut self) {
        self.theme = self.theme.toggle();
        self.set_status("Theme toggled".to_string());
    }

    pub fn set_error(&mut self, msg: String) {
        self.error_message = Some(msg);
        self.status_message = None;
    }

    pub fn set_status(&mut self, msg: String) {
        self.status_message = Some(msg);
        self.error_message = None;
    }

    pub fn clear_error(&mut self) {
        self.error_message = None;
    }

    pub fn clear_status(&mut self) {
        self.status_message = None;
    }

    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    pub fn toggle_settings(&mut self) {
        self.show_settings = !self.show_settings;
        if self.show_settings {
            self.show_help = false;
            self.show_file_browser = false;
            self.show_history = false;
            self.show_diff = false;
        }
    }

    pub fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
        if self.show_help {
            self.show_settings = false;
            self.show_file_browser = false;
            self.show_history = false;
            self.show_diff = false;
        }
    }

    pub fn toggle_file_browser(&mut self) {
        self.show_file_browser = !self.show_file_browser;
        if self.show_file_browser {
            self.show_settings = false;
            self.show_help = false;
            self.show_history = false;
            self.show_diff = false;
        }
    }

    pub fn toggle_history(&mut self) {
        self.show_history = !self.show_history;
        if self.show_history {
            self.show_settings = false;
            self.show_help = false;
            self.show_file_browser = false;
            self.show_diff = false;
        }
    }

    pub fn toggle_diff(&mut self) {
        self.show_diff = !self.show_diff;
        if self.show_diff {
            self.show_settings = false;
            self.show_help = false;
            self.show_file_browser = false;
            self.show_history = false;
        }
    }

    pub fn cycle_delimiter(&mut self) {
        self.encode_options =
            self.encode_options
                .clone()
                .with_delimiter(match self.encode_options.delimiter {
                    Delimiter::Comma => Delimiter::Tab,
                    Delimiter::Tab => Delimiter::Pipe,
                    Delimiter::Pipe => Delimiter::Comma,
                });
    }

    pub fn increase_indent(&mut self) {
        let Indent::Spaces(current) = self.encode_options.indent;
        if current < 8 {
            self.encode_options = self
                .encode_options
                .clone()
                .with_indent(Indent::Spaces(current + 1));
        }
    }

    pub fn decrease_indent(&mut self) {
        let Indent::Spaces(current) = self.encode_options.indent;
        if current > 1 {
            self.encode_options = self
                .encode_options
                .clone()
                .with_indent(Indent::Spaces(current - 1));
        }
    }

    pub fn toggle_fold_keys(&mut self) {
        self.encode_options =
            self.encode_options
                .clone()
                .with_key_folding(match self.encode_options.key_folding {
                    KeyFoldingMode::Off => KeyFoldingMode::Safe,
                    KeyFoldingMode::Safe => KeyFoldingMode::Off,
                });
    }

    pub fn increase_flatten_depth(&mut self) {
        if self.encode_options.flatten_depth == usize::MAX {
            self.encode_options = self.encode_options.clone().with_flatten_depth(2);
        } else if self.encode_options.flatten_depth < 10 {
            self.encode_options = self
                .encode_options
                .clone()
                .with_flatten_depth(self.encode_options.flatten_depth + 1);
        }
    }

    pub fn decrease_flatten_depth(&mut self) {
        if self.encode_options.flatten_depth == 2 {
            self.encode_options = self.encode_options.clone().with_flatten_depth(usize::MAX);
        } else if self.encode_options.flatten_depth > 2
            && self.encode_options.flatten_depth != usize::MAX
        {
            self.encode_options = self
                .encode_options
                .clone()
                .with_flatten_depth(self.encode_options.flatten_depth - 1);
        }
    }

    pub fn toggle_flatten_depth(&mut self) {
        if self.encode_options.flatten_depth == usize::MAX {
            self.encode_options = self.encode_options.clone().with_flatten_depth(2);
        } else {
            self.encode_options = self.encode_options.clone().with_flatten_depth(usize::MAX);
        }
    }

    pub fn toggle_expand_paths(&mut self) {
        self.decode_options =
            self.decode_options
                .clone()
                .with_expand_paths(match self.decode_options.expand_paths {
                    PathExpansionMode::Off => PathExpansionMode::Safe,
                    PathExpansionMode::Safe => PathExpansionMode::Off,
                });
    }

    pub fn toggle_strict(&mut self) {
        let strict = !self.decode_options.strict;
        self.decode_options = self.decode_options.clone().with_strict(strict);
    }

    pub fn toggle_coerce_types(&mut self) {
        let coerce = !self.decode_options.coerce_types;
        self.decode_options = self.decode_options.clone().with_coerce_types(coerce);
    }
}

impl<'a> Default for AppState<'a> {
    fn default() -> Self {
        Self::new()
    }
}

File: tui\state\file_state.rs
=============================
//! File management and conversion history.

use std::path::PathBuf;

use chrono::{DateTime, Local};

/// A file or directory entry.
#[derive(Debug, Clone)]
pub struct FileEntry {
    pub path: PathBuf,
    pub is_dir: bool,
    pub size: u64,
    pub modified: Option<DateTime<Local>>,
}

impl FileEntry {
    pub fn name(&self) -> String {
        self.path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string()
    }

    pub fn is_json(&self) -> bool {
        !self.is_dir && self.path.extension().and_then(|e| e.to_str()) == Some("json")
    }

    pub fn is_toon(&self) -> bool {
        !self.is_dir && self.path.extension().and_then(|e| e.to_str()) == Some("toon")
    }
}

/// Record of a conversion operation.
#[derive(Debug, Clone)]
pub struct ConversionHistory {
    pub timestamp: DateTime<Local>,
    pub mode: String,
    pub input_file: Option<PathBuf>,
    pub output_file: Option<PathBuf>,
    pub token_savings: f64,
    pub byte_savings: f64,
}

/// File browser and conversion history state.
pub struct FileState {
    pub current_file: Option<PathBuf>,
    pub current_dir: PathBuf,
    pub selected_files: Vec<PathBuf>,
    pub history: Vec<ConversionHistory>,
    pub is_modified: bool,
}

impl FileState {
    pub fn new() -> Self {
        Self {
            current_file: None,
            current_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            selected_files: Vec::new(),
            history: Vec::new(),
            is_modified: false,
        }
    }

    pub fn set_current_file(&mut self, path: PathBuf) {
        self.current_file = Some(path.clone());
        self.current_dir = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        self.is_modified = false;
    }

    pub fn clear_current_file(&mut self) {
        self.current_file = None;
        self.is_modified = false;
    }

    pub fn mark_modified(&mut self) {
        self.is_modified = true;
    }

    pub fn add_to_history(&mut self, entry: ConversionHistory) {
        self.history.push(entry);
        if self.history.len() > 50 {
            self.history.remove(0);
        }
    }

    pub fn toggle_file_selection(&mut self, path: PathBuf) {
        if let Some(pos) = self.selected_files.iter().position(|p| p == &path) {
            self.selected_files.remove(pos);
        } else {
            self.selected_files.push(path);
        }
    }

    pub fn clear_selection(&mut self) {
        self.selected_files.clear();
    }

    pub fn is_selected(&self, path: &PathBuf) -> bool {
        self.selected_files.contains(path)
    }
}

impl Default for FileState {
    fn default() -> Self {
        Self::new()
    }
}

File: tui\state\editor_state.rs
===============================
//! Editor state for input/output text areas.

use tui_textarea::TextArea;

/// Which panel is currently active.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EditorMode {
    Input,
    Output,
}

/// State for input and output text areas.
pub struct EditorState<'a> {
    pub input: TextArea<'a>,
    pub output: TextArea<'a>,
    pub active: EditorMode,
}

impl<'a> EditorState<'a> {
    pub fn new() -> Self {
        let mut input = TextArea::default();
        input.set_placeholder_text("Enter JSON here or open a file (Ctrl+O)");

        let mut output = TextArea::default();
        output.set_placeholder_text("TOON output will appear here");

        Self {
            input,
            output,
            active: EditorMode::Input,
        }
    }

    pub fn set_input(&mut self, text: String) {
        let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        self.input = TextArea::from(lines);
    }

    pub fn set_output(&mut self, text: String) {
        let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        self.output = TextArea::from(lines);
    }

    pub fn get_input(&self) -> String {
        self.input.lines().join("\n")
    }

    pub fn get_output(&self) -> String {
        self.output.lines().join("\n")
    }

    pub fn clear_input(&mut self) {
        self.input = TextArea::default();
        self.input
            .set_placeholder_text("Enter JSON here or open a file (Ctrl+O)");
    }

    pub fn clear_output(&mut self) {
        self.output = TextArea::default();
        self.output
            .set_placeholder_text("TOON output will appear here");
    }

    pub fn toggle_active(&mut self) {
        self.active = match self.active {
            EditorMode::Input => EditorMode::Output,
            EditorMode::Output => EditorMode::Input,
        };
    }

    pub fn is_input_active(&self) -> bool {
        self.active == EditorMode::Input
    }

    pub fn is_output_active(&self) -> bool {
        self.active == EditorMode::Output
    }
}

impl<'a> Default for EditorState<'a> {
    fn default() -> Self {
        Self::new()
    }
}

File: tui\state\mod.rs
======================
//! Application state management.

pub mod app_state;
pub mod editor_state;
pub mod file_state;
pub mod repl_state;

pub use app_state::{AppState, ConversionStats, Mode};
pub use editor_state::{EditorMode, EditorState};
pub use file_state::{ConversionHistory, FileState};
pub use repl_state::{ReplLine, ReplLineKind, ReplState};

File: tui\state\repl_state.rs
=============================
//! REPL state - separate from command mode

use std::collections::HashMap;

/// REPL session state
#[derive(Debug, Clone)]
pub struct ReplState {
    /// Whether REPL is active
    pub active: bool,
    /// Current input line
    pub input: String,
    /// Session history (output lines)
    pub output: Vec<ReplLine>,
    /// Variables stored in session
    pub variables: HashMap<String, String>,
    /// Command history
    pub history: Vec<String>,
    /// History index for navigation
    pub history_index: Option<usize>,
    /// Last result (for _ variable)
    pub last_result: Option<String>,
    /// Scroll offset for output
    pub scroll_offset: usize,
}

/// A line in the REPL output
#[derive(Debug, Clone)]
pub struct ReplLine {
    pub kind: ReplLineKind,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReplLineKind {
    Prompt,
    Success,
    Error,
    Info,
}

impl ReplState {
    pub fn new() -> Self {
        Self {
            active: false,
            input: String::new(),
            output: vec![ReplLine {
                kind: ReplLineKind::Info,
                content: "TOON REPL - Type 'help' for commands, 'exit' to close".to_string(),
            }],
            variables: HashMap::new(),
            history: Vec::new(),
            history_index: None,
            last_result: None,
            scroll_offset: 0,
        }
    }

    pub fn activate(&mut self) {
        self.active = true;
        self.input.clear();
        self.history_index = None;
    }

    pub fn deactivate(&mut self) {
        self.active = false;
        self.input.clear();
        self.history_index = None;
    }

    pub fn add_prompt(&mut self, cmd: &str) {
        self.output.push(ReplLine {
            kind: ReplLineKind::Prompt,
            content: format!("> {cmd}"),
        });
    }

    pub fn add_success(&mut self, msg: String) {
        for line in msg.lines() {
            self.output.push(ReplLine {
                kind: ReplLineKind::Success,
                content: line.to_string(),
            });
        }
    }

    pub fn add_error(&mut self, msg: String) {
        self.output.push(ReplLine {
            kind: ReplLineKind::Error,
            content: format!("✗ {msg}"),
        });
    }

    pub fn add_info(&mut self, msg: String) {
        let content = if msg.is_empty() || msg.starts_with("  ") || msg.starts_with("📖") {
            msg
        } else {
            format!("✓ {msg}")
        };

        self.output.push(ReplLine {
            kind: ReplLineKind::Info,
            content,
        });
    }

    pub fn add_to_history(&mut self, cmd: String) {
        if cmd.trim().is_empty() {
            return;
        }
        if self.history.last() == Some(&cmd) {
            return;
        }
        self.history.push(cmd);
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }

    pub fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let new_index = match self.history_index {
            None => Some(self.history.len() - 1),
            Some(0) => Some(0),
            Some(i) => Some(i - 1),
        };
        if let Some(idx) = new_index {
            self.input = self.history[idx].clone();
            self.history_index = new_index;
        }
    }

    pub fn history_down(&mut self) {
        match self.history_index {
            None => (),
            Some(i) if i >= self.history.len() - 1 => {
                self.input.clear();
                self.history_index = None;
            }
            Some(i) => {
                let new_idx = i + 1;
                self.input = self.history[new_idx].clone();
                self.history_index = Some(new_idx);
            }
        }
    }

    pub fn scroll_up(&mut self) {
        if self.scroll_offset > 0 {
            self.scroll_offset -= 1;
        }
    }

    pub fn scroll_down(&mut self, visible_lines: usize) {
        let max_scroll = self.output.len().saturating_sub(visible_lines);
        if self.scroll_offset < max_scroll {
            self.scroll_offset += 1;
        }
    }

    pub fn scroll_to_bottom(&mut self) {
        if self.output.len() <= 30 {
            self.scroll_offset = 0;
        } else {
            self.scroll_offset = self.output.len().saturating_sub(30);
        }
    }
}

impl Default for ReplState {
    fn default() -> Self {
        Self::new()
    }
}
```

---
