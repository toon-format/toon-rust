# RUNE Semantic Prefix & Array Implementation

## Summary

Successfully implemented **single-letter semantic prefixes** (`Capital:name`) and **array literals** (`[expr,expr,...]`) for the RUNE language.

## Semantic Prefixes (A-Z)

### Syntax

- **Format**: `Capital:identifier`
- **Examples**: `T:Gf8`, `V:vector`, `R:continuum`, `Q:e32l`

### Complete Alphabet

| Letter | Meaning | Example Usage |
|--------|---------|---------------|
| A | Address, Axis | `A:pos` |
| B | Binary, Basis | `B:basis` |
| C | Compute, Cache, Cell | `C:cache` |
| D | Data, Delta, Dimension | `D:data` |
| E | Entity, Edge, Expression | `E:entity` |
| F | Function, Frame, Field | `F:func` |
| G | Geometry, Graph, Group | `G:geo` |
| H | Hash, Heap, Hyper | `H:hash` |
| I | Index, Instruction, Identity | `I:index` |
| J | Jump, Join | `J:join` |
| K | Key, Kernel | `K:key` |
| L | Lattice, Layer | `L:lattice` |
| M | Memory, Matrix, Module | `M:matrix` |
| N | Node, Number, Namespace | `N:node` |
| O | Object, Op, Offset | `O:obj` |
| P | Pointer, Process, Page | `P:ptr` |
| Q | Quantized, Query | `Q:e32l` |
| R | Root, Register, Reference | `R:continuum` |
| S | State, Stack, Scalar | `S:state` |
| T | Tensor, Type, Thread | `T:Gf8` |
| U | Unit, Unary, Universal | `U:unit` |
| V | Vector, Value, Vertex | `V:velocity` |
| W | Write, Word, Warp | `W:warp` |
| X | XUID, Cross, Transform | `X:session` |
| Y | Yield, YAML-like | `Y:result` |
| Z | Zero, Zone, Zenith | `Z:null` |

### Disambiguation Rules

- `T:Gf8` → Semantic prefix (Type namespace)
- `T::Gf8` → Namespace operator (double colon)
- `T` → Regular identifier (bare capital)
- `tensor` → Regular identifier (lowercase)

## Array Literals

### Syntax Example

- **Format**: `[expr,expr,expr]` (comma-separated, 2+ elements)
- **Examples**: `[1,2,3]`, `[a,b,c]`, `[T:Gf8, V:vec]`

### Array vs Math Block

| Syntax | Type | Meaning |
|--------|------|---------|
| `[1,2,3]` | Array | Three-element array |
| `[a,b,c]` | Array | Array of identifiers |
| `[1 + 2]` | Math Block | Arithmetic expression |
| `[-5]` | Math Block | Unary negation |
| `[a * b]` | Math Block | Multiplication |
| `[42]` | Math Block | Single value (not array) |

### Key Design Decisions

1. **Arrays require commas**: `[1,2,3]` is array, `[1]` is math block
2. **Math blocks use operators**: `[a + b]`, `[-x]`, `[2^3]`
3. **Nested operations**: `[[1,2], [3,4]]` = array of arrays
4. **No ambiguity**: Parser tries math block first, then array

## Implementation Details

### Grammar Changes

```pest
// Semantic prefix
semantic_prefix = @{ ASCII_ALPHA_UPPER ~ ":" }
semantic_ident = { semantic_prefix ~ ident }

// Array literal (requires comma = 2+ elements)
array_literal = { "[" ~ relation_expr ~ ("," ~ relation_expr)+ ~ "]" }

// Term ordering (math_block before array_literal)
term = {
      string
    | number
    | math_block      // Try first (has operators)
    | array_literal   // Then array (has commas)
    | semantic_ident
    | ident
    | "(" ~ relation_expr ~ ")"
}
```

### AST Changes

```rust
// New semantic identifier type
pub struct SemanticIdent {
    pub prefix: char,  // A-Z
    pub name: Ident,
}

// Extended Term enum
pub enum Term {
    Ident(Ident),
    SemanticIdent(SemanticIdent),  // NEW
    Literal(Literal),
    Group(Box<Expr>),
    Math(Box<MathExpr>),
}

// Extended Literal enum
pub enum Literal {
    Number(f64),
    String(String),
    Array(Vec<Expr>),  // NEW
}
```

## Usage Examples

### Semantic Prefixes in Expressions

```rune
# Type annotations
T:Gf8 -> vector := V:[1,2,3]

# Quantization modes
Q:e32l / L:lattice -> R:continuum

# XUID and state management
X:session -> S:state := "active"

# Geometry and transforms
G:space / A:origin -> V:position
```

### Arrays in Context

```rune
# Array assignment
data := [1,2,3,4,5]

# Array of expressions
paths := [items / 0, items / 1, items / 2]

# Array with semantic identifiers
types := [T:Gf8, T:E32L, T:Quant]

# Nested arrays
matrix := [[1,2,3], [4,5,6], [7,8,9]]
```

### Combined Features

```rune
# Root with semantic types
R:continuum

# Type flows to vector array
T:Gf8 -> vectors := [V:[1,0,0], V:[0,1,0], V:[0,0,1]]

# Quantized lattice data
Q:e32l / L:cells -> data := [
  [A:x1, A:y1],
  [A:x2, A:y2]
]
```

## Test Coverage

✅ **30 tests passing** (12 core + 6 semantic + 12 array)

### Core RUNE Tests (12)

- Basic parsing and TOON integration
- Operator precedence (flow vs struct)
- PEMDAS math expressions
- Root declarations with namespaces
- Round-trip encoding

### Semantic Prefix Tests (6)

- All 26 letters (A-Z)
- Disambiguation from regular identifiers
- Namespace operator distinction (`:` vs `::`)
- Complex expressions with multiple prefixes
- Invalid lowercase prefix rejection

### Array Literal Tests (12)

- Numeric, identifier, mixed arrays
- Arrays with expressions
- Array vs math block distinction
- Nested arrays
- Arrays in assignments
- Semantic identifiers in arrays
- String arrays
- Empty/single-element handling

## Notes

- **Operator precedence**: Flow operators (`->`, `<-`) bind looser than structural operators (`:=`, `::`, etc.)
- **TOON integration**: Semantic prefixes and arrays work seamlessly with TOON blocks
- **Math blocks**: Still use `[expr]` syntax for arithmetic operations
- **Future extensions**: Could add tuple syntax `(a,b)` vs grouped expr `(a + b)` if needed
