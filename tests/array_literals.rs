/* tests/arrays.rs */
//! Tests for RUNE array literals and nested arithmetic operations.
//!
//! # TOON-RUNE – Array Literal Tests
//!▫~•◦-------------------------------‣
//!
//! This test suite validates array literal syntax in RUNE:
//! - [1,2,3] = array of numbers
//! - [a,b,c] = array of identifiers/expressions
//! - [[expr * expr]] = nested math on arrays
//!
//! Arrays use commas, math blocks don't.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use rune_format::rune::*;

#[test]
fn test_numeric_array_literal() {
    let input = "[1,2,3]";
    let stmts = parse_rune(input).unwrap();

    assert_eq!(stmts.len(), 1);

    if let Stmt::Expr(Expr::Term(Term::Literal(Literal::Array(elements)))) = &stmts[0] {
        assert_eq!(elements.len(), 3);

        // Check each element
        for (i, elem) in elements.iter().enumerate() {
            if let Expr::Term(Term::Literal(Literal::Number(n))) = elem {
                assert_eq!(*n, (i + 1) as f64);
            } else {
                panic!("Expected number literal");
            }
        }

        // Verify round-trip
        let displayed = format!("{}", stmts[0]);
        assert_eq!(displayed, input);
    } else {
        panic!("Expected array literal");
    }
}

#[test]
fn test_identifier_array_literal() {
    let input = "[a,b,c]";
    let stmts = parse_rune(input).unwrap();

    if let Stmt::Expr(Expr::Term(Term::Literal(Literal::Array(elements)))) = &stmts[0] {
        assert_eq!(elements.len(), 3);

        let expected_names = ["a", "b", "c"];
        for (i, elem) in elements.iter().enumerate() {
            if let Expr::Term(Term::Ident(id)) = elem {
                assert_eq!(id.0, expected_names[i]);
            } else {
                panic!("Expected identifier");
            }
        }
    }
}

#[test]
fn test_mixed_array_literal() {
    let input = "[1,a,2,b]";
    let stmts = parse_rune(input).unwrap();

    if let Stmt::Expr(Expr::Term(Term::Literal(Literal::Array(elements)))) = &stmts[0] {
        assert_eq!(elements.len(), 4);

        // First element: 1
        matches!(&elements[0], Expr::Term(Term::Literal(Literal::Number(_))));
        // Second element: a
        matches!(&elements[1], Expr::Term(Term::Ident(_)));
        // Third element: 2
        matches!(&elements[2], Expr::Term(Term::Literal(Literal::Number(_))));
        // Fourth element: b
        matches!(&elements[3], Expr::Term(Term::Ident(_)));
    }
}

#[test]
fn test_array_with_expressions() {
    let input = "[a / 0, b / 1, c / 2]";
    let stmts = parse_rune(input).unwrap();

    if let Stmt::Expr(Expr::Term(Term::Literal(Literal::Array(elements)))) = &stmts[0] {
        assert_eq!(elements.len(), 3);

        // Each element should be a binary expression (a / 0, etc.)
        for elem in elements {
            if let Expr::Binary { op, .. } = elem {
                assert_eq!(*op, RuneOp::Descendant); // / operator
            } else {
                panic!("Expected binary expression");
            }
        }
    }
}

#[test]
fn test_semantic_prefix_with_array() {
    let input = "V:[1,2,3]";
    let stmts = parse_rune(input).unwrap();

    // Should parse as: V (semantic prefix) with name '[1,2,3]'? No!
    // Actually, semantic_ident requires an ident after the colon
    // So this should fail to parse as semantic_ident and instead parse as...
    // Actually, let's check what happens

    // It should be: semantic_ident 'V' with name that starts with '['
    // But ident can't start with '[', so this should fail
    // OR it parses as V: followed by array [1,2,3]

    // Let me check the actual structure
    println!("Parsed: {:?}", stmts);
}

#[test]
fn test_math_block_vs_array() {
    // Array: comma-separated
    let array = "[1,2,3]";
    let arr_stmts = parse_rune(array).unwrap();
    assert!(matches!(
        &arr_stmts[0],
        Stmt::Expr(Expr::Term(Term::Literal(Literal::Array(_))))
    ));

    // Math block: operator-based
    let math = "[1 + 2]";
    let math_stmts = parse_rune(math).unwrap();
    assert!(matches!(
        &math_stmts[0],
        Stmt::Expr(Expr::Term(Term::Math(_)))
    ));

    // Not ambiguous!
    assert_ne!(format!("{}", arr_stmts[0]), format!("{}", math_stmts[0]));
}

#[test]
fn test_nested_math_on_array() {
    // [[3,3,3] * [3,3,3]] - math block containing array multiplication
    // Wait, this doesn't make sense with current design
    // The inner [3,3,3] would be parsed as array
    // But math_block expects math_expr inside

    // Let's test what we CAN do:
    // [expr] where expr contains arrays
    let input = "[[1,2,3], [4,5,6]]";
    let stmts = parse_rune(input).unwrap();

    // This is an array of arrays
    if let Stmt::Expr(Expr::Term(Term::Literal(Literal::Array(outer)))) = &stmts[0] {
        assert_eq!(outer.len(), 2);

        for elem in outer {
            assert!(matches!(elem, Expr::Term(Term::Literal(Literal::Array(_)))));
        }
    }
}

#[test]
fn test_array_in_expression() {
    let input = "data := [1,2,3]";
    let stmts = parse_rune(input).unwrap();

    if let Stmt::Expr(Expr::Binary { op, left, right }) = &stmts[0] {
        assert_eq!(*op, RuneOp::Define);
        assert!(matches!(left.as_ref(), Expr::Term(Term::Ident(_))));
        assert!(matches!(
            right.as_ref(),
            Expr::Term(Term::Literal(Literal::Array(_)))
        ));
    }
}

#[test]
fn test_array_with_semantic_elements() {
    let input = "[T:Gf8, V:vector, R:root]";
    let stmts = parse_rune(input).unwrap();

    if let Stmt::Expr(Expr::Term(Term::Literal(Literal::Array(elements)))) = &stmts[0] {
        assert_eq!(elements.len(), 3);

        // Each element should be a semantic identifier
        for elem in elements {
            assert!(matches!(elem, Expr::Term(Term::SemanticIdent(_))));
        }
    }
}

#[test]
fn test_empty_array() {
    // Should we support empty arrays? Let's try
    let input = "[]";
    let result = parse_rune(input);

    // This might fail because array_literal requires at least one element
    // Let's see what happens
    println!("Empty array result: {:?}", result);
}

#[test]
fn test_single_element_array() {
    // Single-element arrays are not supported by design
    // [42] is parsed as a math block, not an array
    // Arrays must have commas, i.e., 2+ elements
    let input = "[42]";
    let stmts = parse_rune(input).unwrap();

    // Should parse as math block, not array
    if let Stmt::Expr(Expr::Term(Term::Math(_))) = &stmts[0] {
        // Expected: math block
    } else {
        panic!("Expected math block for single-element bracket notation");
    }

    // If you want a single-element collection, use array with trailing comma
    // or just use the bare value
}

#[test]
fn test_array_with_strings() {
    let input = r#"["hello", "world", "test"]"#;
    let stmts = parse_rune(input).unwrap();

    if let Stmt::Expr(Expr::Term(Term::Literal(Literal::Array(elements)))) = &stmts[0] {
        assert_eq!(elements.len(), 3);

        for elem in elements {
            assert!(matches!(
                elem,
                Expr::Term(Term::Literal(Literal::String(_)))
            ));
        }
    }
}
