//! RUNE Arithmetic Examples
//!
//! Demonstrates arithmetic operations in RUNE using math blocks.
//!
//! In RUNE, arithmetic happens inside [brackets] to separate
//! mathematical evaluation from structural/topological operations:
//! - Math blocks: [a + b * c]
//! - Operators: +, -, *, /
//! - Precedence: */ before +-
//! - Integration with RUNE: items / 0 := [base + offset]
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣
//! RUNE builds on TOON: Copyright © 2025 Shreyas S Bhat, Johann Schopplich (MIT License)

use rune_format::rune;

/// Demonstrate arithmetic operations in RUNE
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 RUNE Arithmetic Examples\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Example 1: Basic arithmetic in math blocks
    println!("📌 Basic Arithmetic in Math Blocks [...]:");
    let basic_examples = vec![
        "[a + b]",
        "[x - y]",
        "[width * height]",
        "[2 + 3]",
        "[10 - 7]",
        "[4 * 5]",
        "[a / b]",
    ];

    for expr in basic_examples {
        parse_and_display(expr);
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Example 2: Operator precedence
    println!("📌 Operator Precedence (*/ before +-):");
    let precedence_examples = vec![
        "[a + b * c]", // a + (b * c)
        "[x * y + z]", // (x * y) + z
        "[2 + 3 * 4]", // 2 + (3 * 4) = 14
        "[5 * 2 - 3]", // (5 * 2) - 3 = 7
        "[a - b + c]", // (a - b) + c
        "[x * y / z]", // (x * y) / z
        "[a / b * c]", // (a / b) * c
    ];

    for expr in precedence_examples {
        parse_and_display(expr);
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Example 3: Mixing structural operators with math blocks
    println!("📌 Structural Operations + Math Blocks:");
    let mixed_examples = vec![
        "items / 0 := [base + offset]",
        "matrix / row := [index * cols]",
        "x -> value := [a * 2 + b]",
        "vec / i ~ [x + y * scale]",
        "data / point := [x * x + y * y]",
    ];

    for expr in mixed_examples {
        parse_and_display(expr);
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Example 4: Complex expressions
    println!("📌 Complex Math Expressions:");
    let complex_examples = vec![
        "[a + b * c - d]",
        "[width * height / 2]",
        "[a + b * c - d / e]",
        "[(a + b) * c]",
        "result := [sum * 100 / total]",
    ];

    for expr in complex_examples {
        parse_and_display(expr);
    }

    println!("\n🔍 Key Concepts:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Math Blocks:");
    println!("  • Arithmetic uses [brackets]: [a + b * c]");
    println!("  • Separates math from structural ops");
    println!("  • Outside []: RUNE operators (/, \\, ->, :=, etc.)");
    println!("  • Inside []: Math operators (+, -, *, /)");
    println!();
    println!("Precedence within [...]:");
    println!("  1. *, / (multiply, divide) - highest");
    println!("  2. +, - (add, subtract) - lower");
    println!("  3. Parentheses () for grouping");
    println!();
    println!("Associativity:");
    println!("  • Left-to-right for same precedence");
    println!("  • [a + b + c] → [(a + b) + c]");
    println!("  • [x / y / z] → [(x / y) / z]");
    println!();
    println!("Integration:");
    println!("  • Use math blocks as values:");
    println!("    items / 0 := [base + offset]");
    println!("  • Combine navigation with computation:");
    println!("    data / point := [x * x + y * y]");

    Ok(())
}

/// Parse and display a RUNE expression
fn parse_and_display(expr: &str) {
    print!("  {} ", expr);

    match rune::parse_rune(expr) {
        Ok(stmts) => {
            if let Some(rune::Stmt::Expr(_)) = stmts.first() {
                println!("✓");
            } else {
                println!("✗ (unexpected statement type)");
            }
        }
        Err(e) => {
            println!("✗ {}", e);
        }
    }
}
