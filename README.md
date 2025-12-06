# RUNE: Root Unified Notation Encoding

[![Crates.io](https://img.shields.io/crates/v/toon-rune.svg)](https://crates.io/crates/toon-rune)
[![Documentation](https://docs.rs/toon-rune/badge.svg)](https://docs.rs/toon-rune)
[![Built on TOON](https://img.shields.io/badge/built%20on-TOON-purple.svg)](https://github.com/toon-format/toon)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

**RUNE (Root Unified Notation Encoding)** is a root-centric, semantic operator system for the E8 ecosystem. It wraps the **TOON** data format to provide geometric flow, hierarchical definitions, and explicit semantic wiring for LLMs and generative systems.

> **Attribution Note:**
> RUNE is built upon the foundational work of **[Token-Oriented Object Notation (TOON)](https://github.com/toon-format/toon)**.
> It extends the TOON serialization format with root-oriented operators (`/`, `->`, `:=`) and topological glyphs.
> We gratefully acknowledge Johann Schopplich and Shreyas S Bhat for their excellent work on the underlying data format.

---

## The RUNE Philosophy

While **TOON** solves the problem of *efficient data serialization* for LLMs, **RUNE** solves the problem of *semantic structure and flow*.

RUNE treats data blocks as **Nodes** in a root-oriented hierarchy. It introduces a notation to:
1.  **Define Roots:** Explicitly anchor data to a context (`root: context`).
2.  **Embed Data:** Use TOON syntax for efficient, token-cheap data payloads.
3.  **Define Flow:** Use directed operators (`->`) and glyphs (`\|/`) to describe relationships between data nodes.

### RUNE = Root + Operators + TOON Data

```rune
# 1. Define the Root Context
root: e8_continuum

# 2. Embed Data (using TOON syntax)
layers ~TOON:
  config[2]{id, type}:
    1,Lattice
    2,Projection

# 3. Define Semantic Flow (RUNE Operators)
layers / 1 -> type := Lattice
layers / 2 -> type := Projection

# 4. Topological Relations (Glyphs)
layers / 1 \|/ layers / 2   # Symmetric split relation
```

---

## Features

-   **Root-Centric Architecture**: Every document is anchored to a specific root/context.
-   **TOON Data Layer**: Uses the spec-compliant TOON format for all data blocks (see below).
-   **Semantic Operators**: First-class support for flow (`->`), definition (`:=`), and hierarchy (`/`).
-   **Token Efficiency**: Inherits TOON's 18-40% token savings over JSON.
-   **E8 Geometry**: Native support for E8 primitives (Gf8, XUID) in the AST.

---

## Installation

### As a Library

```bash
cargo add toon-rune
```

### As a CLI Tool

```bash
cargo install toon-rune
```

---

## Data Layer: TOON Format

*RUNE uses TOON v2.0 as its native data serialization format. The following documentation applies to data blocks within RUNE.*

### Quick Example (Data Layer)

**JSON** (16 tokens, 40 bytes):
```json
{
  "users": [
    { "id": 1, "name": "Alice" },
    { "id": 2, "name": "Bob" }
  ]
}
```

**RUNE/TOON** (13 tokens, 28 bytes) - **18.75% token savings**:
```rune
users[2]{id,name}:
  1,Alice
  2,Bob
```

### Benchmarks (Inherited from TOON)

RUNE inherits the performance characteristics of the underlying TOON parser:

| Format | Accuracy/1K Tok | Accuracy | Tokens |
|--------|-----------------|----------|--------|
| **TOON** | **26.9%** | **73.9%** | **2,744** |
| JSON Compact | 22.9% | 70.7% | 3,081 |
| YAML | 18.6% | 69.0% | 3,719 |
| JSON | 15.3% | 69.7% | 4,545 |

---

## Library Usage

The `rune` crate exposes APIs to parse full RUNE documents as well as raw TOON blocks.

### Parsing RUNE

```rust
use toon_rune::{RuneParser, Stmt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = r#"
        root: system_config
        
        users ~TOON:
          users[2]{id,role}:
            1,admin
            2,viewer
            
        users / 1 -> role := admin
    "#;

    let file = RuneParser::parse(input)?;
    
    for stmt in file.statements {
        match stmt {
            Stmt::RootDef(root) => println!("Root: {}", root),
            Stmt::ToonBlock(block) => {
                println!("Found data block '{}'", block.name);
                // Parse the inner content using the standard TOON decoder
                let data: serde_json::Value = toon_rune::decode(&block.raw_content)?;
                println!("Data: {:?}", data);
            }
            Stmt::Def { name, value } => println!("Defined {} := {}", name, value),
            _ => {}
        }
    }

    Ok(())
}
```

---

## v1.5 Features (Supported in Data Blocks)

RUNE supports all modern TOON features within `~TOON:` blocks.

### Key Folding

Collapse single-key object chains into dotted paths.

```rune
config ~TOON:
  data.metadata.items[2]: a,b
```

### Path Expansion

Automatically expand dotted keys into nested objects during decoding.

```rune
settings ~TOON:
  a.b.c: 1
  a.b.d: 2
```

Expands to:
```json
{"a": {"b": {"c": 1, "d": 2}}}
```

---

## CLI Usage

The `rune` CLI works as a superset of the `toon` CLI.

```bash
# Parse a RUNE file and output the resolved JSON tree
rune input.rune --json

# Interactive TUI mode (inherited from TOON)
rune -i
```

### Options

```bash
# Custom indentation for data blocks
rune data.rune --indent 4

# Key folding for data blocks
rune data.rune --fold-keys

# Show statistics
rune data.rune --stats
```

---

## License & Attribution

This project is a fork and extension of [toon-format](https://github.com/toon-format/toon).

-   **RUNE Extensions & Modifications**: Copyright © 2025 ArcMoon Studios
-   **Original TOON Format & Code**: Copyright © 2025-PRESENT Johann Schopplich and Shreyas S Bhat

Licensed under the **MIT License**.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
