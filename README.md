# TOON Format for Rust

[![Crates.io](https://img.shields.io/crates/v/toon-format.svg)](https://crates.io/crates/toon-format)
[![Documentation](https://docs.rs/toon-format/badge.svg)](https://docs.rs/toon-format)
[![Spec v1.4](https://img.shields.io/badge/spec-v1.4-brightgreen.svg)](https://github.com/toon-format/spec/blob/main/SPEC.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

**Token-Oriented Object Notation (TOON)** is a compact, human-readable format designed for passing structured data to Large Language Models with significantly reduced token usage.

This crate provides the official, spec-compliant Rust implementation of TOON, offering both a library (`toon-format`) and a full-featured command-line tool (`toon`).

### Example

**JSON** (verbose):
```json
{
  "users": [
    { "id": 1, "name": "Alice", "role": "admin" },
    { "id": 2, "name": "Bob", "role": "user" }
  ]
}
````

**TOON** (compact):

```toon
users[2]{id,name,role}:
  1,Alice,admin
  2,Bob,user
```

## Features

  * **Spec-Compliant:** Fully compliant with [TOON Specification v1.4](https://github.com/toon-format/spec/blob/main/SPEC.md).
  * **Safe & Performant:** Built with safe, fast Rust.
  * **Serde Integration:** Natively serializes from and deserializes to `serde_json::Value` for easy integration.
  * **Powerful CLI:** A full-featured `toon` binary for command-line conversion.
  * **Validation:** Includes a strict mode decoder (on by default) to enforce all spec rules.


## Library Usage

Add `toon-format` to your Rust project:

```bash
cargo add toon-format
```

### Basic `encode` and `decode`

The library works directly with `serde_json::Value`.

```rust
use serde_json::json;
use toon_format::{encode_default, decode_default};

fn main() -> Result<(), toon_format::ToonError> {
    let data = json!({
      "users": [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"}
      ]
    });

    // Encode
    let toon_string = encode_default(&data)?;
    println!("{}", toon_string);
    // users[2]{id,name}:
    //   1,Alice
    //   2,Bob

    // Decode
    let decoded_value = decode_default(&toon_string)?;
    assert_eq!(decoded_value, data);
    
    Ok(())
}
```

### Options

You can customize the encoding and decoding process by passing `EncodeOptions` or `DecodeOptions`.

```rust
use toon_format::{encode, decode, EncodeOptions, DecodeOptions, Delimiter, Indent};
use serde_json::json;

// --- Encode with options ---
let data = json!({"tags": ["a", "b"]});

let encode_opts = EncodeOptions::new()
    .with_delimiter(Delimiter::Pipe) // Use '|' as delimiter
    .with_indent(Indent::Spaces(4))  // Use 4 spaces
    .with_length_marker('#');        // Add '#' to [N]

let toon_string = encode(&data, &encode_opts)?;
// tags[#2|]: a|b

// --- Decode with options ---
let toon_input = "items[3]: a,b"; // Mismatched length

// Default (strict) decode will fail
assert!(decode_default(toon_input).is_err());

// Non-strict decode will pass
let decode_opts = DecodeOptions::new()
    .with_strict(false); // Disable length validation

let decoded = decode(toon_input, &decode_opts)?;
assert_eq!(decoded, json!({"items": ["a", "b"]}));
```

## CLI Usage

You can also install the `toon` binary for command-line use.

### Installation

```bash
cargo install toon-format
```

### Basic Usage

The CLI auto-detects the operation based on the file extension.

  * `.json` input will be **encoded** to TOON.
  * `.toon` input will be **decoded** to JSON.

<!-- end list -->

```bash
# Encode JSON to TOON (auto-detected)
toon data.json

# Decode TOON to JSON (auto-detected)
toon data.toon
```

### Piping

The CLI reads from `stdin` if no input file is provided, or if `-` is used as the input file.

```bash
# Pipe from stdin (defaults to ENCODE)
cat data.json | toon
echo '{"name": "Ada"}' | toon

# Pipe from stdin and force DECODE
cat data.toon | toon -d
echo "name: Ada" | toon --decode
```

### Stats

Use --stats during encoding to see token (cl100k_base) and byte-size savings.

```bash
$ echo '{"users": [{"id": 1, "name": "Alice"}]}' | toon --stats
users[1]{id,name}:
  1,Alice

Stats:

+--------------+------+------+---------+
| Metric       | JSON | TOON | Savings |
+==============+======+======+=========+
| Tokens       | 16   | 13   | 13.33%  |
+--------------+------+------+---------+
| Size (bytes) | 40   | 28   | 27.50%  |
+--------------+------+------+---------+


```

### Options

| Option | Short | Description |
| :--- | :--- | :--- |
| `[input]` | | Input file path (reads from stdin if omitted or '-') |
| `--output <file>` | `-o` | Output file path (prints to stdout if omitted) |
| `--encode` | `-e` | Force encode mode (JSON -\> TOON) |
| `--decode` | `-d` | Force decode mode (TOON -\> JSON) |
| `--delimiter <val>` | | Array delimiter: `comma`, `tab`, or `pipe` (encode only) |
| `--indent <num>` | `-i` | Indentation size in spaces (default: 2) (encode only) |
| `--tabs` | | Use tabs for indentation (encode only) |
| `--length-marker` | | Add `#` prefix to array lengths (e.g., `items[#3]`) (encode only) |
| `--stats` | | Show byte size savings (encode only) |
| `--no-strict` | | Disable strict validation (decode only) |
| `--no-coerce` | | Disable type coercion (e.g., "true" -\> `true`) (decode only) |
| `--json-indent <num>` | | Indent output JSON with N spaces (decode only) |

### CLI Examples

```bash
# Encode a file to stdout
toon data.json

# Encode a file to another file
toon data.json -o data.toon

# Encode with tab delimiters and 4-space indent
toon data.json --delimiter tab -i 4

# Decode a file and pretty-print the JSON
toon data.toon --json-indent 2

# Decode from stdin
cat data.toon | toon -d

# Encode from stdin and save to file with stats
cat data.json | toon --stats -o data.toon
```

## Resources
  * [TOON Specification](https://github.com/toon-format/spec/blob/main/SPEC.md)
  * [Main Repository (JS/TS)](https://github.com/toon-format/toon)
  * [Benchmarks & Performance](https://github.com/toon-format/toon#benchmarks)
  * [Other Language Implementations](https://github.com/toon-format/toon#other-implementations)

## Contributing

Contributions are welcome\! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License

MIT License © 2025-PRESENT [Johann Schopplich](https://github.com/johannschopplich) and [Shreyas K S](https://github.com/shreyasbhat0)
