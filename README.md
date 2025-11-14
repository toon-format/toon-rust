# TOON Format for Rust

[![Crates.io](https://img.shields.io/crates/v/toon-format.svg)](https://crates.io/crates/toon-format)
[![Documentation](https://docs.rs/toon-format/badge.svg)](https://docs.rs/toon-format)
[![Spec v2.0](https://img.shields.io/badge/spec-v2.0-brightgreen.svg)](https://github.com/toon-format/spec/blob/main/SPEC.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-%20passing-success.svg)]()

**Token-Oriented Object Notation (TOON)** is a compact, human-readable format designed for passing structured data to Large Language Models with significantly reduced token usage.

This crate provides the official, **spec-compliant Rust implementation** of TOON v2.0 with v1.5 optional features, offering both a library (`toon-format`) and a full-featured command-line tool (`toon`).

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
- **Spec-Compliant**: Fully compliant with [TOON Specification v2.0](https://github.com/toon-format/spec/blob/main/SPEC.md)
- **v1.5 Optional Features**: Key folding and path expansion
- **Safe & Performant**: Built with safe, fast Rust
- **Powerful CLI**: Full-featured command-line tool
- **Strict Validation**: Enforces all spec rules (configurable)
- **Well-Tested**: Comprehensive test suite with unit tests, spec fixtures, and real-world scenarios

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
| `with_key_folding(mode)` | Enable key folding (v1.5) | `Off` |
| `with_flatten_depth(n)` | Set max folding depth | `usize::MAX` |

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

// Disable type coercion
let opts = DecodeOptions::new().with_coerce_types(false);
let json: Value = decode("active: true", &opts)?;
// With coercion: {"active": true}
// Without: {"active": "true"}
```

**Helper functions:**
- `encode_default<T>(&value)` - Encode with default options
- `decode_default<T>(&input)` - Decode with default options

#### `DecodeOptions`

| Method | Description | Default |
|--------|-------------|---------|
| `with_strict(b)` | Enable strict validation | `true` |
| `with_coerce_types(b)` | Auto-convert strings to types | `true` |
| `with_expand_paths(mode)` | Enable path expansion (v1.5) | `Off` |

---

## v1.5 Features

### Key Folding (Encoder)

**New in v1.5**: Collapse single-key object chains into dotted paths to reduce tokens.

**Standard nesting:**
```toon
data:
  metadata:
    items[2]: a,b
```

**With key folding:**
```toon
data.metadata.items[2]: a,b
```

**Example:**

```rust
use serde_json::json;
use toon_format::{encode, EncodeOptions, KeyFoldingMode};

let data = json!({
    "data": {
        "metadata": {
            "items": ["a", "b"]
        }
    }
});

// Enable key folding
let opts = EncodeOptions::new()
    .with_key_folding(KeyFoldingMode::Safe);

let toon = encode(&data, &opts)?;
// Output: data.metadata.items[2]: a,b
```

#### With Depth Control

```rust
let data = json!({"a": {"b": {"c": {"d": 1}}}});

// Fold only 2 levels
let opts = EncodeOptions::new()
    .with_key_folding(KeyFoldingMode::Safe)
    .with_flatten_depth(2);

let toon = encode(&data, &opts)?;
// Output:
// a.b:
//   c:
//     d: 1
```

#### Safety Features

Key folding only applies when:
- All segments are valid identifiers (`a-z`, `A-Z`, `0-9`, `_`)
- Each level contains exactly one key
- No collision with sibling literal keys
- Within the specified `flatten_depth`

Keys like `full-name`, `user.email` (if quoted), or numeric keys won't be folded.

### Path Expansion (Decoder)

**New in v1.5**: Automatically expand dotted keys into nested objects.

**Compact input:**
```toon
a.b.c: 1
a.b.d: 2
a.e: 3
```

**Expanded output:**
```json
{
  "a": {
    "b": {
      "c": 1,
      "d": 2
    },
    "e": 3
  }
}
```

**Example:**

```rust
use serde_json::Value;
use toon_format::{decode, DecodeOptions, PathExpansionMode};

let toon = "a.b.c: 1\na.b.d: 2";

// Enable path expansion
let opts = DecodeOptions::new()
    .with_expand_paths(PathExpansionMode::Safe);

let json: Value = decode(toon, &opts)?;
// {"a": {"b": {"c": 1, "d": 2}}}
```

**Round-Trip Example:**

```rust
use serde_json::{json, Value};
use toon_format::{encode, decode, EncodeOptions, DecodeOptions, KeyFoldingMode, PathExpansionMode};

let original = json!({
    "user": {
        "profile": {
            "name": "Alice"
        }
    }
});

// Encode with folding
let encode_opts = EncodeOptions::new()
    .with_key_folding(KeyFoldingMode::Safe);
let toon = encode(&original, &encode_opts)?;
// Output: "user.profile.name: Alice"

// Decode with expansion
let decode_opts = DecodeOptions::new()
    .with_expand_paths(PathExpansionMode::Safe);
let restored: Value = decode(&toon, &decode_opts)?;

assert_eq!(restored, original); // Perfect round-trip!
```

**Quoted Keys Remain Literal:**

```rust
use serde_json::Value;
use toon_format::{decode, DecodeOptions, PathExpansionMode};

let toon = r#"a.b: 1
"c.d": 2"#;

let opts = DecodeOptions::new()
    .with_expand_paths(PathExpansionMode::Safe);
let json: Value = decode(toon, &opts)?;
// {
//   "a": {"b": 1},
//   "c.d": 2        <- quoted key preserved
// }
```

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

# Key folding (v1.5)
toon data.json --fold-keys
toon data.json --fold-keys --flatten-depth 2

# Show statistics
toon data.json --stats
```

### Decode Options

```bash
# Pretty-print JSON
toon data.toon --json-indent 2

# Relaxed validation
toon data.toon --no-strict

# Disable type coercion
toon data.toon --no-coerce

# Path expansion (v1.5)
toon data.toon --expand-paths
```

### Full Example

```bash
$ echo '{"data":{"meta":{"items":["x","y"]}}}' | toon --fold-keys --stats

data.meta.items[2]: x,y

Stats:
+--------------+------+------+---------+
| Metric       | JSON | TOON | Savings |
+======================================+
| Tokens       | 13   | 8    | 38.46%  |
|--------------+------+------+---------|
| Size (bytes) | 38   | 23   | 39.47%  |
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
    Err(ToonError::LengthMismatch { expected, found, .. }) => {
        eprintln!("Array length mismatch: expected {}, found {}", expected, found);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

### Error Types

- `ParseError` - Syntax errors with line/column info
- `LengthMismatch` - Array length doesn't match header
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

- 📖 [TOON Specification v2.0](https://github.com/toon-format/spec/blob/main/SPEC.md)
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
