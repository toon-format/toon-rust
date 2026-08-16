# TOON Format for Rust

[![Crates.io](https://img.shields.io/crates/v/toon-format.svg)](https://crates.io/crates/toon-format)
[![Documentation](https://docs.rs/toon-format/badge.svg)](https://docs.rs/toon-format)
[![Spec v4.1](https://img.shields.io/badge/spec-v4.1-brightgreen.svg)](https://github.com/toon-format/spec/blob/main/SPEC.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-%20passing-success.svg)]()

**Token-Oriented Object Notation (TOON)** is a compact, human-readable format designed for passing structured data to Large Language Models with significantly reduced token usage.

This crate provides the official, **spec-compliant Rust implementation** of TOON, offering both a library (`toon-format`) and a full-featured command-line tool (`toon`).

`toon-spec: 4.1` — this implementation targets [TOON Specification v4.1](https://github.com/toon-format/spec/blob/main/SPEC.md).

## Quick Example

**JSON** (16 tokens, 40 bytes):
```json
{
  "users": [
    { "id": 1, "name": "Alice" },
    { "id": 2, "name": "Bob" }
  ]
}
```

**TOON** (13 tokens, 28 bytes) - **18.75% token savings**:
```toon
users[2]{id,name}:
  1,Alice
  2,Bob
```

## Features

- **Generic API**: Works with any `Serialize`/`Deserialize` type - custom structs, enums, JSON values, and more
- **Spec-Compliant**: Fully compliant with [TOON Specification v4.1](https://github.com/toon-format/spec/blob/main/SPEC.md), including comment lines, keyed tabular form, and nested field groups
- **Safe & Performant**: Built with safe, fast Rust
- **Powerful CLI**: Full-featured command-line tool
- **Strict Validation**: Enforces all spec rules (configurable)
- **Well-Tested**: Comprehensive test suite with unit tests, spec fixtures, and real-world scenarios

### Experimental Cargo Features

- **`layout`** *(experimental, off by default)*: Exposes decoder layout metadata
  (tabular vs list vs inline, declared `[N]` lengths, field descriptors) via
  `decode_with_layout`. Scoped to independent exploration of schema and tooling
  use cases (validators, formatters, linters); **not part of the TOON
  specification** and may evolve independently of the core decoder.

## Installation

### As a Library

```bash
cargo add toon-format
```

### As a CLI Tool

```bash
cargo install toon-format
```

---

## Library Usage

### Basic Encode & Decode

The `encode` and `decode` functions work with any type implementing `Serialize`/`Deserialize`:

**With custom structs:**

```rust
use serde::{Serialize, Deserialize};
use toon_format::{encode_default, decode_default};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct User {
    name: String,
    age: u32,
    email: String,
}

fn main() -> Result<(), toon_format::ToonError> {
    let user = User {
        name: "Alice".to_string(),
        age: 30,
        email: "alice@example.com".to_string(),
    };

    // Encode to TOON
    let toon = encode_default(&user)?;
    println!("{}", toon);
    // Output:
    // name: Alice
    // age: 30
    // email: alice@example.com

    // Decode back to struct
    let decoded: User = decode_default(&toon)?;
    assert_eq!(user, decoded);

    Ok(())
}
```

**With JSON values:**

```rust
use serde_json::{json, Value};
use toon_format::{encode_default, decode_default};

fn main() -> Result<(), toon_format::ToonError> {
    let data = json!({
        "users": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"}
        ]
    });

    // Encode to TOON
    let toon_str = encode_default(&data)?;
    println!("{}", toon_str);
    // Output:
    // users[2]{id,name}:
    //   1,Alice
    //   2,Bob

    // Decode back to JSON
    let decoded: Value = decode_default(&toon_str)?;
    assert_eq!(decoded, data);

    Ok(())
}
```
---

## API Reference

### Encoding

#### `encode<T: Serialize>(&value, &options) -> Result<String, ToonError>`

Encode any serializable type to TOON format. Works with custom structs, enums, collections, and `serde_json::Value`.

```rust
use toon_format::{encode, EncodeOptions, Delimiter, Indent};
use serde_json::json;

let data = json!({"items": ["a", "b", "c"]});

// Default encoding
let toon = encode(&data, &EncodeOptions::default())?;
// items[3]: a,b,c

// Custom delimiter
let opts = EncodeOptions::new()
    .with_delimiter(Delimiter::Pipe);
let toon = encode(&data, &opts)?;
// items[3|]: a|b|c

// Custom indentation
let opts = EncodeOptions::new()
    .with_indent(Indent::Spaces(4));
let toon = encode(&data, &opts)?;
```

#### `EncodeOptions`

| Method | Description | Default |
|--------|-------------|---------|
| `with_delimiter(d)` | Set delimiter: `Comma`, `Tab`, or `Pipe` | `Comma` |
| `with_indent(i)` | Set indentation (spaces only) | `Spaces(2)` |
| `with_spaces(n)` | Shorthand for `Indent::Spaces(n)` | `2` |

#### `json_stream` Feature

The optional `json_stream` feature adds conveniences for encoding JSON from a
`Read` source to a `Write` target. Spec v4.1 selects the encoded form from a
value's whole shape (tabular and keyed tabular headers depend on every element
of their subtree), so the input is parsed in full before encoding; these
functions are I/O conveniences, not bounded-memory streaming.

```bash
cargo add toon-format --features json_stream
```

```rust
use std::io::Cursor;
use toon_format::{encode_json_stream_default};

let input = Cursor::new(br#"{"users":[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"}]}"#);
let mut output = Vec::new();
encode_json_stream_default(input, &mut output)?;
let toon = String::from_utf8(output)?;
assert!(toon.contains("users[2]{id,name}:"));
```

### Decoding

#### `decode<T: Deserialize>(&input, &options) -> Result<T, ToonError>`

Decode TOON format into any deserializable type. Works with custom structs, enums, collections, and `serde_json::Value`.

**With custom structs:**
```rust
use serde::Deserialize;
use toon_format::{decode, DecodeOptions};

#[derive(Deserialize)]
struct Config {
    host: String,
    port: u16,
}

let toon = "host: localhost\nport: 8080";
let config: Config = decode(toon, &DecodeOptions::default())?;
```

**With JSON values:**
```rust
use serde_json::Value;
use toon_format::{decode, DecodeOptions};

let toon = "name: Alice\nage: 30";

// Default (strict) decode
let json: Value = decode(toon, &DecodeOptions::default())?;

// Non-strict mode (relaxed validation)
let opts = DecodeOptions::new().with_strict(false);
let json: Value = decode(toon, &opts)?;
```

**Helper functions:**
- `encode_default<T>(&value)` - Encode with default options
- `decode_default<T>(&input)` - Decode with default options

#### `DecodeOptions`

| Method | Description | Default |
|--------|-------------|---------|
| `with_strict(b)` | Enable strict validation | `true` |
| `with_indent(i)` | Set spaces per indentation level | `Spaces(2)` |

---

## Spec v4.1 Highlights

### Comment Lines (Decoder)

Lines whose first non-space character is `#` are comments, removed before any
structural interpretation. Encoders never emit them.

```toon
# server inventory
servers[2]{host,port}:
  a.example.com,8080
  b.example.com,9090
```

### Keyed Tabular Form

An object whose values are uniform non-empty objects collapses into a keyed
header with one entry row per entry:

```toon
servers[2:]{host,port}:
  alpha: a.example.com,8080
  beta: b.example.com,9090
```

### Nested Field Groups

A uniform nested-object column collapses into the header, its leaf cells laid
out by a depth-first walk:

```toon
orders[2]{id,customer{name,country},total}:
  1,Ada,DK,99
  2,Bob,UK,149
```

### Empty Arrays

Empty arrays encode as `key: []` in field position and `[]` at the root; the
legacy `key[0]:` and `[0]:` forms are still accepted by the decoder.

---

## Interactive TUI

TOON includes a full-featured Terminal User Interface for interactive conversions!

```bash
# Launch interactive mode
toon --interactive
# or
toon -i
```

### Features:
- Real-time conversion as you type
- Live statistics (tokens, bytes, savings)
- Interactive settings - adjust all options on-the-fly
- File browser with visual navigation
- Side-by-side diff viewer
- Conversion history tracking
- File operations (open, save, new)
- Clipboard integration (copy/paste)
- REPL mode for command-line interaction
- Round-trip testing
- Theme support (Dark/Light)
- Built-in help with keyboard shortcuts

**Perfect for:**
- Learning TOON format interactively
- Testing conversions in real-time
- Experimenting with different settings
- Visual before/after comparisons
- Quick data transformations

See [docs/TUI.md](docs/TUI.md) for complete documentation and keyboard shortcuts!

---

## CLI Usage

### Basic Commands

```bash
# Auto-detect from extension
toon data.json        # Encode
toon data.toon        # Decode

# Force mode
toon -e data.txt      # Force encode
toon -d output.txt    # Force decode

# Pipe from stdin
cat data.json | toon
echo '{"name": "Alice"}' | toon -e
```

### Encode Options

```bash
# Custom delimiter
toon data.json --delimiter pipe
toon data.json --delimiter tab

# Custom indentation
toon data.json --indent 4

# Show statistics
toon data.json --stats
```

### Decode Options

```bash
# Pretty-print JSON
toon data.toon --json-indent 2

# Relaxed validation
toon data.toon --no-strict
```

### Full Example

```bash
$ echo '{"users":[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"}]}' | toon --stats

users[2]{id,name}:
  1,Alice
  2,Bob

Stats:

+--------------+------+------+---------+
| Metric       | JSON | TOON | Savings |
+======================================+
| Tokens       | 20   | 19   | 5.00%   |
|--------------+------+------+---------|
| Size (bytes) | 58   | 36   | 37.93%  |
+--------------+------+------+---------+
```

---

## Testing

The library includes a comprehensive test suite covering core functionality, edge cases, spec compliance, and real-world scenarios.

```bash
# Run all tests
cargo test

# Run specific test suites
cargo test --test spec_fixtures
cargo test --lib

# With output
cargo test -- --nocapture
```

## Error Handling

All operations return `Result<T, ToonError>` with descriptive error messages:

```rust
use serde_json::Value;
use toon_format::{decode_strict, ToonError};

match decode_strict::<Value>("items[3]: a,b") {
    Ok(value) => println!("Success: {:?}", value),
    Err(ToonError::ParseError { line, message, .. }) => {
        eprintln!("Parse error on line {}: {}", line, message);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

### Error Types

- `ParseError` - Syntax, structural, and strict-mode errors with line info
- `TypeMismatch` - Unexpected value type
- `InvalidStructure` - Malformed TOON structure
- `SerializationError` / `DeserializationError` - Conversion failures

---


## Examples
Run with `cargo run --example examples` to see all examples:
- `structs.rs` - Custom struct serialization
- `tabular.rs` - Tabular array formatting
- `arrays.rs` - Various array formats
- `arrays_of_arrays.rs` - Nested arrays
- `objects.rs` - Object encoding
- `mixed_arrays.rs` - Mixed-type arrays
- `delimiters.rs` - Custom delimiters
- `round_trip.rs` - Encode/decode round-trips
- `decode_strict.rs` - Strict validation
- `empty_and_root.rs` - Edge cases

---

## Resources

- 📖 [TOON Specification v4.1](https://github.com/toon-format/spec/blob/main/SPEC.md)
- 📦 [Crates.io Package](https://crates.io/crates/toon-format)
- 📚 [API Documentation](https://docs.rs/toon-format)
- 🔧 [Main Repository (JS/TS)](https://github.com/toon-format/toon)
- 🎯 [Benchmarks & Performance](https://github.com/toon-format/toon#benchmarks)

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development

```bash
# Clone the repository
git clone https://github.com/your-org/toon-rust.git
cd toon-rust

# Run tests
cargo test --all

# Run lints
cargo clippy -- -D warnings

# Format code
cargo fmt

# Build docs
cargo doc --open
```

---

## License

MIT License © 2025-PRESENT [Johann Schopplich](https://github.com/johannschopplich) and [Shreyas K S](https://github.com/shreyasbhat0)
