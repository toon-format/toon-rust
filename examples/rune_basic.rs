//! Basic RUNE Example
//!
//! This example demonstrates the core RUNE syntax: root declarations,
//! TOON blocks, and operator expressions over that data.
//! RUNE builds on TOON: Copyright © 2025 Shreyas S Bhat, Johann Schopplich (MIT License)
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use rune_format::rune;

/// Basic RUNE usage showing root declarations and TOON integration
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Sample RUNE source code
    let rune_source = r#"
# Declare the semantic root context for this data
root: continuum

# Embed raw TOON data with full fidelity
data ~TOON:
  users[3]{id,name,role}:
    1,Ada,admin
    2,Bob,user
    3,Eve,viewer

# RUNE operators work with TOON paths
users / 0 -> role := admin
users / 1 -> name ~ ValidString()
"#;

    // Parse the RUNE file into statements
    let statements = rune::parse_rune(rune_source)?;

    println!("Parsed {} statements:", statements.len());

    // Process each statement
    for stmt in &statements {
        match stmt {
            rune::Stmt::RootDecl(name) => {
                println!("✓ Root declared: {}", name);
            }
            rune::Stmt::ToonBlock { name, content } => {
                println!(
                    "✓ TOON block '{}': {} lines of raw data",
                    name,
                    content.lines().count()
                );
                // This raw TOON content can be passed to the TOON crate for decoding
                println!(
                    "  TOON content: \"{}\" (first 50 chars)",
                    content.chars().take(50).collect::<String>()
                );
            }
            rune::Stmt::RuneBlock { name, content } => {
                println!(
                    "✓ RUNE block '{}': {} lines of raw data",
                    name,
                    content.lines().count()
                );
                // This raw RUNE content can be processed by the RUNE crate
                println!(
                    "  RUNE content: \"{}\" (first 50 chars)",
                    content.chars().take(50).collect::<String>()
                );
            }
            rune::Stmt::Expr(expr) => {
                println!("✓ Expression: {}", expr);
            }
            rune::Stmt::KernelDecl { name, archetype } => {
                println!("✓ Kernel declared: {} := {}", name, archetype.name);
            }
        }
    }

    // Demonstrate re-encoding
    let encoded = rune::encode_rune(&statements);
    println!("\nRe-encoded RUNE:\n{}", encoded);

    Ok(())
}
