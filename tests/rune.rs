/* tests/rune.rs */
//! Comprehensive integration tests for the RUNE parsing engine.
//!
//! # TOON-RUNE – RUNE Integration Tests
//!▫~•◦------------------------------------------------‣
//! [NOTE: The underline above must underline the "Remarks:" line only.]
//!
//! This test suite validates the complete RUNE parsing, AST construction,
//! and integration with TOON data blocks for the E8 ecosystem.
//!
//! ### Key Capabilities
//! - **Parser Validation**: Tests complete RUNE source code parsing and AST generation.
//! - **TOON Integration**: Verifies TOON block extraction and round-trip compatibility.
//! - **Operator Precedence**: Validates mathematical and semantic operator precedence rules.
//! - **Root Semantics**: Tests root declaration parsing and namespace handling.
//! - **Round-trip Fidelity**: Ensures parse ↔ encode consistency.
//!
//! ### Architectural Notes
//! Tests cover the parser::ParsingError, AST types, and rune::encode_rune functionality.
//! Integration tests use serde_json for TOON data validation and ensure clean compilation.
//!
//! ### Example
//! ```rust
//! use rune_format::rune::*;
//!
//! let rune_code = "root: test\nitems / 0 -> name := value";
//! let statements = parse_rune(rune_code).unwrap();
//! assert_eq!(statements.len(), 2);
//!
//! // Tests validate that parsing produces expected AST structures.
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use rune_format::decode_default;
use rune_format::rune::*;
use serde_json::json;

#[test]
fn test_rune_parsing_basic() {
    let input = r#"
root: test_context

data ~TOON:
  items[2]{id,name}:
    1,hello
    2,world

items / 1 -> name := world
"#;

    let stmts = parse_rune(input).unwrap();

    // Should have 3 statements: root, toon_block, expr
    assert_eq!(stmts.len(), 3);

    match &stmts[0] {
        Stmt::RootDecl(root) => assert_eq!(root.0.as_str(), "test_context"),
        _ => panic!("Expected root declaration"),
    }

    match &stmts[1] {
        Stmt::ToonBlock { name, content } => {
            assert_eq!(name.0.as_str(), "data");
            // Should contain the TOON data
            assert!(content.contains("items[2]"));
        }
        _ => panic!("Expected TOON block"),
    }

    match &stmts[2] {
        Stmt::Expr(expr) => {
            // Should be items / 1 -> name := world
            // With precedence: flow_op (-> ) < struct_op (:=)
            // Parses as: items / 1 -> (name := world)
            match expr {
                Expr::Binary { op, left, right } => {
                    assert_eq!(*op, RuneOp::FlowRight);
                    assert_eq!(format!("{}", left), "items / 1");
                    assert_eq!(format!("{}", right), "name := world");
                }
                _ => panic!("Expected binary expression with ->"),
            }
        }
        _ => panic!("Expected expression statement"),
    }
}

#[test]
fn test_toon_integration() {
    let input = r#"
users ~TOON:
  list[3]{id,name,role}:
    1,"Alice",admin
    2,"Bob",user
    3,"Charlie",moderator

list / 0 -> role := admin
"#;

    let stmts = parse_rune(input).unwrap();
    assert_eq!(stmts.len(), 2);

    // Extract TOON content and verify it parses as valid TOON
    if let Stmt::ToonBlock { content, .. } = &stmts[0] {
        let decoded: serde_json::Value = decode_default(content).unwrap();
        let expected = json!({
            "list": [
                {"id": 1, "name": "Alice", "role": "admin"},
                {"id": 2, "name": "Bob", "role": "user"},
                {"id": 3, "name": "Charlie", "role": "moderator"}
            ]
        });
        assert_eq!(decoded, expected);
    }
}

#[test]
fn test_operator_precedence() {
    // Test math precedence: * before + (inside math block)
    let input = "[a + b * c]";
    let stmts = parse_rune(input).unwrap();

    if let Stmt::Expr(expr) = &stmts[0] {
        // Should be a math block term
        match expr {
            Expr::Term(Term::Math(math_expr)) => {
                // Inside should be addition at top level
                match math_expr.as_ref() {
                    MathExpr::Binary { op, left, right } => {
                        assert_eq!(*op, MathOp::Add);
                        // Left should be 'a'
                        match left.as_ref() {
                            MathExpr::Atom(MathAtom::Ident(id)) => assert_eq!(id.0, "a"),
                            _ => panic!("Expected identifier 'a'"),
                        }
                        // Right side should be (b * c)
                        match right.as_ref() {
                            MathExpr::Binary { op, left, right } => {
                                assert_eq!(*op, MathOp::Multiply);
                                match left.as_ref() {
                                    MathExpr::Atom(MathAtom::Ident(id)) => assert_eq!(id.0, "b"),
                                    _ => panic!("Expected identifier 'b'"),
                                }
                                match right.as_ref() {
                                    MathExpr::Atom(MathAtom::Ident(id)) => assert_eq!(id.0, "c"),
                                    _ => panic!("Expected identifier 'c'"),
                                }
                            }
                            _ => panic!("Expected nested multiplication"),
                        }
                    }
                    _ => panic!("Expected addition at top level"),
                }
            }
            _ => panic!("Expected math block"),
        }
    }
}

#[test]
fn test_complex_glyphs() {
    let input = r#"
fiber_net / hub /\ endpoint
data \|/ modes
"#;

    let stmts = parse_rune(input).unwrap();
    assert_eq!(stmts.len(), 2);

    // Check first expression: fiber_net / hub /\ endpoint
    if let Stmt::Expr(expr) = &stmts[0] {
        // Should parse fiber_net / hub first, then /\ operator
        println!("Parsed: {}", expr);
    }
}

#[test]
fn test_root_and_namespace_operators() {
    let input = r#"
root: e8::continuum
T::Gf8 -> vec := "[+1,-1,+1,-1,+1,-1,+1,-1]"
embeddings / 0 | input_vec
"#;

    let stmts = parse_rune(input).unwrap();
    assert_eq!(stmts.len(), 3);

    // First statement should be root declaration
    match &stmts[0] {
        Stmt::RootDecl(root) => assert_eq!(root.0.as_str(), "e8::continuum"),
        _ => panic!("Expected root declaration with namespace"),
    }

    // Check type annotation: T::Gf8 -> vec
    if let Stmt::Expr(expr) = &stmts[1] {
        match expr {
            Expr::Binary { op, left, right } => {
                assert_eq!(*op, RuneOp::FlowRight);
                assert_eq!(format!("{}", left), "T::Gf8");
                assert_eq!(format!("{}", right), "vec := \"[+1,-1,+1,-1,+1,-1,+1,-1]\"");
            }
            _ => panic!("Expected type -> value flow"),
        }
    }
}

#[test]
fn test_round_trip_encoding() {
    // Create some statements programmatically
    let original = vec![
        Stmt::root("test_root"),
        Stmt::expr(Expr::binary(
            Expr::ident("data"),
            RuneOp::FlowRight,
            Expr::literal(42.0),
        )),
    ];

    // Encode to string
    let encoded = encode_rune(&original);

    // Parse back
    let decoded = parse_rune(&encoded).unwrap();

    // Should be structurally equivalent
    assert_eq!(original.len(), decoded.len());

    // Check first statement (root)
    match (&original[0], &decoded[0]) {
        (Stmt::RootDecl(orig), Stmt::RootDecl(dec)) => assert_eq!(orig, dec),
        _ => panic!("Root declaration mismatch"),
    }

    // Check second statement (expression)
    match (&original[1], &decoded[1]) {
        (Stmt::Expr(orig), Stmt::Expr(dec)) => {
            // Both should contain data -> 42.0
            assert_eq!(format!("{}", orig), format!("{}", dec));
        }
        _ => panic!("Expression mismatch"),
    }
}

#[test]
fn test_invalid_operators_rejected() {
    // These should fail parsing due to operators not in registry
    let bad_inputs = vec![
        "a => b",      // => not a valid operator
        "a |-> b",     // |-> is invalid fusion
        "value :++ x", // ++ not valid
    ];

    for input in bad_inputs {
        assert!(
            parse_rune(input).is_err(),
            "Input '{}' should fail parsing",
            input
        );
    }
}

#[test]
fn test_pemdas_exponentiation() {
    // Test exponent operator has highest precedence
    let input = "[2 + 3 * 4 ^ 2]";
    let stmts = parse_rune(input).unwrap();

    if let Stmt::Expr(expr) = &stmts[0] {
        match expr {
            Expr::Term(Term::Math(math_expr)) => {
                // Should parse as: 2 + (3 * (4 ^ 2))
                match math_expr.as_ref() {
                    MathExpr::Binary { op, left, right } => {
                        assert_eq!(*op, MathOp::Add);
                        // Left is 2
                        match left.as_ref() {
                            MathExpr::Atom(MathAtom::Number(n)) => assert_eq!(*n, 2.0),
                            _ => panic!("Expected 2"),
                        }
                        // Right is (3 * (4 ^ 2))
                        match right.as_ref() {
                            MathExpr::Binary { op, left, right } => {
                                assert_eq!(*op, MathOp::Multiply);
                                // Left is 3
                                match left.as_ref() {
                                    MathExpr::Atom(MathAtom::Number(n)) => assert_eq!(*n, 3.0),
                                    _ => panic!("Expected 3"),
                                }
                                // Right is (4 ^ 2)
                                match right.as_ref() {
                                    MathExpr::Binary { op, left, right } => {
                                        assert_eq!(*op, MathOp::Power);
                                        match left.as_ref() {
                                            MathExpr::Atom(MathAtom::Number(n)) => {
                                                assert_eq!(*n, 4.0)
                                            }
                                            _ => panic!("Expected 4"),
                                        }
                                        match right.as_ref() {
                                            MathExpr::Atom(MathAtom::Number(n)) => {
                                                assert_eq!(*n, 2.0)
                                            }
                                            _ => panic!("Expected 2"),
                                        }
                                    }
                                    _ => panic!("Expected exponent operation"),
                                }
                            }
                            _ => panic!("Expected multiplication"),
                        }
                    }
                    _ => panic!("Expected addition at top level"),
                }
            }
            _ => panic!("Expected math block"),
        }
    } else {
        panic!("Expected expression statement");
    }
}

#[test]
fn test_pemdas_unary_operators() {
    // Test unary minus
    let input = "[-5]";
    let stmts = parse_rune(input).unwrap();

    if let Stmt::Expr(expr) = &stmts[0] {
        match expr {
            Expr::Term(Term::Math(math_expr)) => match math_expr.as_ref() {
                MathExpr::Unary { op, operand } => {
                    assert_eq!(*op, MathUnaryOp::Negate);
                    match operand.as_ref() {
                        MathExpr::Atom(MathAtom::Number(n)) => assert_eq!(*n, 5.0),
                        _ => panic!("Expected 5"),
                    }
                }
                _ => panic!("Expected unary expression"),
            },
            _ => panic!("Expected math block"),
        }
    } else {
        panic!("Expected expression statement");
    }

    // Test unary with identifier
    let input2 = "[-x + 5]";
    let stmts2 = parse_rune(input2).unwrap();

    if let Stmt::Expr(expr) = &stmts2[0] {
        match expr {
            Expr::Term(Term::Math(math_expr)) => {
                // Should be addition at top level
                match math_expr.as_ref() {
                    MathExpr::Binary { op, left, right } => {
                        assert_eq!(*op, MathOp::Add);
                        // Left should be unary minus x
                        match left.as_ref() {
                            MathExpr::Unary { op, operand } => {
                                assert_eq!(*op, MathUnaryOp::Negate);
                                match operand.as_ref() {
                                    MathExpr::Atom(MathAtom::Ident(id)) => assert_eq!(id.0, "x"),
                                    _ => panic!("Expected identifier x"),
                                }
                            }
                            _ => panic!("Expected unary expression"),
                        }
                        // Right should be 5
                        match right.as_ref() {
                            MathExpr::Atom(MathAtom::Number(n)) => assert_eq!(*n, 5.0),
                            _ => panic!("Expected 5"),
                        }
                    }
                    _ => panic!("Expected addition"),
                }
            }
            _ => panic!("Expected math block"),
        }
    } else {
        panic!("Expected expression statement");
    }
}

#[test]
fn test_pemdas_arithmetic_outside_brackets_fails() {
    // Arithmetic operators should NOT work outside math blocks
    // These should fail because +, *, ^ are not valid operators outside [...]
    let invalid_inputs = vec![
        "2 + 3", "a * b",
        "x ^ 2",
        // Note: "a R 2" is actually valid (R is parsed as identifier, not operator)
    ];

    for input in invalid_inputs {
        let result = parse_rune(input);
        assert!(
            result.is_err(),
            "Input '{}' should fail (arithmetic outside brackets)",
            input
        );
    }
}

#[test]
fn test_pemdas_root_operator() {
    // Test n-th root operator with R
    let input = "[x R n]";
    let stmts = parse_rune(input).unwrap();

    if let Stmt::Expr(expr) = &stmts[0] {
        match expr {
            Expr::Term(Term::Math(math_expr)) => match math_expr.as_ref() {
                MathExpr::Binary { op, left, right } => {
                    assert_eq!(*op, MathOp::Root);
                    match left.as_ref() {
                        MathExpr::Atom(MathAtom::Ident(id)) => assert_eq!(id.0, "x"),
                        _ => panic!("Expected identifier x"),
                    }
                    match right.as_ref() {
                        MathExpr::Atom(MathAtom::Ident(id)) => assert_eq!(id.0, "n"),
                        _ => panic!("Expected identifier n"),
                    }
                }
                _ => panic!("Expected binary root operation"),
            },
            _ => panic!("Expected math block"),
        }
    } else {
        panic!("Expected expression statement");
    }

    // Test another root operation
    let input2 = "[27 R 3]";
    let stmts2 = parse_rune(input2).unwrap();

    if let Stmt::Expr(expr) = &stmts2[0] {
        match expr {
            Expr::Term(Term::Math(math_expr)) => match math_expr.as_ref() {
                MathExpr::Binary { op, left, right } => {
                    assert_eq!(*op, MathOp::Root);
                    match left.as_ref() {
                        MathExpr::Atom(MathAtom::Number(n)) => assert_eq!(*n, 27.0),
                        _ => panic!("Expected 27"),
                    }
                    match right.as_ref() {
                        MathExpr::Atom(MathAtom::Number(n)) => assert_eq!(*n, 3.0),
                        _ => panic!("Expected 3"),
                    }
                }
                _ => panic!("Expected binary root operation"),
            },
            _ => panic!("Expected math block"),
        }
    } else {
        panic!("Expected expression statement");
    }
}

#[test]
fn test_pemdas_root_precedence() {
    // Root should have same precedence as exponentiation
    // Test: [2 + 8 R 3] should parse as 2 + (8 R 3)
    let input = "[2 + 8 R 3]";
    let stmts = parse_rune(input).unwrap();

    if let Stmt::Expr(expr) = &stmts[0] {
        match expr {
            Expr::Term(Term::Math(math_expr)) => {
                match math_expr.as_ref() {
                    MathExpr::Binary { op, left, right } => {
                        assert_eq!(*op, MathOp::Add);
                        // Left is 2
                        match left.as_ref() {
                            MathExpr::Atom(MathAtom::Number(n)) => assert_eq!(*n, 2.0),
                            _ => panic!("Expected 2"),
                        }
                        // Right is (8 R 3)
                        match right.as_ref() {
                            MathExpr::Binary { op, left, right } => {
                                assert_eq!(*op, MathOp::Root);
                                match left.as_ref() {
                                    MathExpr::Atom(MathAtom::Number(n)) => assert_eq!(*n, 8.0),
                                    _ => panic!("Expected 8"),
                                }
                                match right.as_ref() {
                                    MathExpr::Atom(MathAtom::Number(n)) => assert_eq!(*n, 3.0),
                                    _ => panic!("Expected 3"),
                                }
                            }
                            _ => panic!("Expected root operation"),
                        }
                    }
                    _ => panic!("Expected addition at top level"),
                }
            }
            _ => panic!("Expected math block"),
        }
    } else {
        panic!("Expected expression statement");
    }
}
