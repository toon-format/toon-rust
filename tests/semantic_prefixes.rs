/* tests/semantic_prefixes.rs */
//! Tests for RUNE single-letter semantic namespace prefixes.
//!
//! # TOON-RUNE – Semantic Prefix Tests
//!▫~•◦------------------------------------------------‣
//!
//! This test suite validates the complete single-letter semantic alphabet
//! for RUNE (A-Z), ensuring each prefix can be parsed and displayed correctly.
//!
//! ### Semantic Alphabet
//! - A: Address, Axis
//! - B: Binary, Basis
//! - C: Compute, Cache, Cell
//! - D: Data, Delta, Dimension
//! - E: Entity, Edge, Expression
//! - F: Function, Frame, Field
//! - G: Geometry, Graph, Group
//! - H: Hash, Heap, Hyper
//! - I: Index, Instruction, Identity
//! - J: Jump, Join
//! - K: Key, Kernel
//! - L: Lattice, Layer, Left
//! - M: Memory, Matrix, Module
//! - N: Node, Number, Namespace
//! - O: Object, Op, Offset
//! - P: Pointer, Process, Page
//! - Q: Quantized, Query
//! - R: Root, Register, Reference
//! - S: State, Stack, Scalar
//! - T: Tensor, Type, Thread
//! - U: Unit, Unary, Universal
//! - V: Vector, Value, Vertex
//! - W: Write, Word, Warp
//! - X: XUID, Cross, Transform
//! - Y: Yield, YAML-like
//! - Z: Zero, Zone, Zenith
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use rune_format::rune::*;

#[test]
fn test_all_semantic_prefixes() {
    // Test all 26 single-letter semantic prefixes
    let prefixes = [
        ('A', "addr"),    // Address
        ('B', "basis"),   // Binary/Basis
        ('C', "cache"),   // Compute/Cache
        ('D', "data"),    // Data/Delta
        ('E', "entity"),  // Entity/Edge
        ('F', "func"),    // Function
        ('G', "geo"),     // Geometry
        ('H', "hash"),    // Hash
        ('I', "index"),   // Index
        ('J', "join"),    // Join
        ('K', "key"),     // Key
        ('L', "lattice"), // Lattice
        ('M', "mem"),     // Memory
        ('N', "node"),    // Node
        ('O', "obj"),     // Object
        ('P', "ptr"),     // Pointer
        ('Q', "quant"),   // Quantized
        ('R', "root"),    // Root
        ('S', "state"),   // State
        ('T', "tensor"),  // Tensor/Type
        ('U', "unit"),    // Unit
        ('V', "vector"),  // Vector
        ('W', "warp"),    // Warp
        ('X', "xuid"),    // XUID
        ('Y', "yield"),   // Yield
        ('Z', "zero"),    // Zero
    ];

    for (prefix, name) in prefixes {
        let input = format!("{}:{}", prefix, name);
        let stmts = parse_rune(&input).unwrap();

        assert_eq!(stmts.len(), 1, "Failed to parse {}", input);

        if let Stmt::Expr(expr) = &stmts[0] {
            if let Expr::Term(Term::SemanticIdent(sid)) = expr {
                assert_eq!(sid.prefix, prefix);
                assert_eq!(sid.name.0, name);

                // Verify display round-trip
                let displayed = format!("{}", expr);
                assert_eq!(displayed, input);
            } else {
                panic!("Expected semantic ident for {}", input);
            }
        } else {
            panic!("Expected expression statement for {}", input);
        }
    }
}

#[test]
fn test_semantic_in_expressions() {
    // Test semantic identifiers in real expressions
    let test_cases = vec![
        (
            "T:Gf8 -> vec := V:[1,2,3]",
            "Type flows to vector definition",
        ),
        ("R:continuum", "Root declaration style"),
        ("Q:e32l -> compressed", "Quantization mode"),
        ("X:session -> uid", "XUID anchor"),
        ("L:lattice / G:cell", "Lattice path to geometry cell"),
        ("T:Tensor := M:matrix", "Type defined as matrix"),
        ("A:pos -> V:velocity", "Address to vector flow"),
    ];

    for (input, description) in test_cases {
        let result = parse_rune(input);
        assert!(result.is_ok(), "Failed to parse {}: {}", description, input);

        let stmts = result.unwrap();
        assert!(
            !stmts.is_empty(),
            "No statements parsed for: {}",
            description
        );

        // Verify round-trip through display
        let displayed = format!("{}", stmts[0]);
        assert!(
            displayed.contains(':'),
            "Display should preserve semantic prefix: {}",
            description
        );
    }
}

#[test]
fn test_semantic_vs_regular_ident() {
    // Semantic identifier
    let semantic = "T:Gf8";
    let stmts = parse_rune(semantic).unwrap();
    if let Stmt::Expr(Expr::Term(Term::SemanticIdent(sid))) = &stmts[0] {
        assert_eq!(sid.prefix, 'T');
        assert_eq!(sid.name.0, "Gf8");
    } else {
        panic!("Expected semantic ident");
    }

    // Regular identifier (lowercase)
    let regular = "tensor";
    let stmts = parse_rune(regular).unwrap();
    if let Stmt::Expr(Expr::Term(Term::Ident(id))) = &stmts[0] {
        assert_eq!(id.0, "tensor");
    } else {
        panic!("Expected regular ident");
    }

    // Capital letter without colon is still a regular identifier
    let capital = "T";
    let stmts = parse_rune(capital).unwrap();
    if let Stmt::Expr(Expr::Term(Term::Ident(id))) = &stmts[0] {
        assert_eq!(id.0, "T");
    } else {
        panic!("Expected regular ident for bare capital");
    }
}

#[test]
fn test_semantic_with_namespace_operator() {
    // T::Gf8 uses :: operator, not semantic prefix
    let input = "T::Gf8";
    let stmts = parse_rune(input).unwrap();

    if let Stmt::Expr(Expr::Binary { op, left, right }) = &stmts[0] {
        assert_eq!(*op, RuneOp::Namespace);
        assert_eq!(format!("{}", left), "T");
        assert_eq!(format!("{}", right), "Gf8");
    } else {
        panic!("Expected namespace operator expression");
    }

    // T:Gf8 is semantic prefix (single colon)
    let input2 = "T:Gf8";
    let stmts2 = parse_rune(input2).unwrap();

    if let Stmt::Expr(Expr::Term(Term::SemanticIdent(sid))) = &stmts2[0] {
        assert_eq!(sid.prefix, 'T');
        assert_eq!(sid.name.0, "Gf8");
    } else {
        panic!("Expected semantic ident");
    }
}

#[test]
fn test_complex_semantic_expression() {
    let input = r#"
T:Gf8 -> V:vector := "[+1,-1,+1,-1,+1,-1,+1,-1]"
Q:e32l / L:lattice -> R:continuum
"#;

    let stmts = parse_rune(input).unwrap();
    assert_eq!(stmts.len(), 2);

    // First expression: T:Gf8 -> V:vector := "..."
    if let Stmt::Expr(expr) = &stmts[0] {
        let displayed = format!("{}", expr);
        assert!(displayed.contains("T:Gf8"));
        assert!(displayed.contains("V:vector"));
    }

    // Second expression: Q:e32l / L:lattice -> R:continuum
    if let Stmt::Expr(expr) = &stmts[1] {
        let displayed = format!("{}", expr);
        assert!(displayed.contains("Q:e32l"));
        assert!(displayed.contains("L:lattice"));
        assert!(displayed.contains("R:continuum"));
    }
}

#[test]
fn test_semantic_prefix_invalid() {
    // Lowercase prefix should not work as semantic
    let invalid = "t:tensor";
    let result = parse_rune(invalid);
    // This should parse as 't', then fail on unexpected ':'
    // Or parse as ident 't' and fail on leftover ':tensor'
    assert!(
        result.is_err() || {
            // If it parses, it should NOT be a semantic ident
            if let Ok(stmts) = result {
                !matches!(stmts[0], Stmt::Expr(Expr::Term(Term::SemanticIdent(_))))
            } else {
                true
            }
        }
    );
}
