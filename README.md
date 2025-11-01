# TOON Format for Rust

[![Crates.io](https://img.shields.io/crates/v/toon-format.svg)](https://crates.io/crates/toon-format)
[![Documentation](https://docs.rs/toon-format/badge.svg)](https://docs.rs/toon-format)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

**Token-Oriented Object Notation** is a compact, human-readable format designed for passing structured data to Large Language Models with significantly reduced token usage.

## Status

🚧 **This crate is currently a namespace reservation.** Full implementation coming soon!

## What is TOON?

TOON achieves 30-60% token reduction compared to JSON by:

- **Eliminating repetitive keys** in arrays of objects
- **Using indentation** instead of braces
- **Declaring structure upfront** with explicit lengths and field names
- **Minimal quoting** for string values

### Example

**JSON** (verbose):
```json
{
  "users": [
    { "id": 1, "name": "Alice", "role": "admin" },
    { "id": 2, "name": "Bob", "role": "user" }
  ]
}
```

**TOON** (compact):
```
users[2]{id,name,role}:
  1,Alice,admin
  2,Bob,user
```

## Resources

- [TOON Specification](https://github.com/johannschopplich/toon/blob/main/SPEC.md)
- [Main Repository](https://github.com/johannschopplich/toon)
- [Benchmarks & Performance](https://github.com/johannschopplich/toon#benchmarks)
- [Other Language Implementations](https://github.com/johannschopplich/toon#other-implementations)

## Future Usage

Once implemented, the crate will provide:

```rust
use toon_format::{encode, decode};

let data = // your data structure
let toon_string = encode(data)?;
let decoded = decode(&toon_string)?;
```

## Contributing

Interested in implementing TOON for Rust? Check out the [specification](https://github.com/johannschopplich/toon/blob/main/SPEC.md) and feel free to contribute!

## License

MIT License © 2025-PRESENT [Johann Schopplich](https://github.com/johannschopplich)
