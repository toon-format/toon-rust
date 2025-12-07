//! RUNE Operator Examples
//!
//! Demonstrates the complete operator registry and E8-specific semantics.
//! Shows how glyph operators represent complex relationships.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣
//! RUNE builds on TOON: Copyright © 2025 Shreyas S Bhat, Johann Schopplich (MIT License)

use rune_format::rune::{self, OpCategory, RuneOp};

/// Display the complete operator catalog for RUNE
fn show_operator_catalog() {
    println!("🌌 RUNE Operator Catalog\n");

    let operators = vec![
        RuneOp::SplitJoin,          // /\
        RuneOp::JoinSplit,          // \/
        RuneOp::AnchorDescend,      // |/
        RuneOp::BranchStabilize,    // /|
        RuneOp::RootStabilize,      // \|
        RuneOp::StabilizeRoot,      // |\
        RuneOp::SymmetricSplit,     // \|/
        RuneOp::BranchAnchorBranch, // /|\
        RuneOp::Bind,               // :
        RuneOp::Namespace,          // ::
        RuneOp::Define,             // :=
        RuneOp::Equal,              // =
        RuneOp::FlowRight,          // ->
        RuneOp::FlowLeft,           // <-
        RuneOp::Descendant,         // /
        RuneOp::Ancestor,           // \
        RuneOp::Alias,              // |
        RuneOp::Parallel,           // ||
        RuneOp::Transform,          // ~
        RuneOp::Less,               // <
        RuneOp::Greater,            // >
    ];

    println!("📐 GLYPH OPERATORS (Topology):");
    for op in &operators {
        if op.category() == OpCategory::Glyph {
            let (bp1, bp2) = op.binding_power();
            println!(
                "  {:4} → {} (BP: {},{})",
                op.as_str(),
                format_category(op.category()),
                bp1,
                bp2
            );
        }
    }

    println!("\n🔗 RELATION OPERATORS:");
    for op in &operators {
        if op.category() == OpCategory::Relation {
            let (bp1, bp2) = op.binding_power();
            println!(
                "  {:3} → {} (BP: {},{})",
                op.as_str(),
                format_category(op.category()),
                bp1,
                bp2
            );
        }
    }

    println!("\n📊 MATH & COMPARE:");
    for op in &operators {
        if op.category() == OpCategory::Math || op.category() == OpCategory::Compare {
            let (bp1, bp2) = op.binding_power();
            println!(
                "  {:3} → {} (BP: {},{})",
                op.as_str(),
                format_category(op.category()),
                bp1,
                bp2
            );
        }
    }
}

fn format_category(cat: OpCategory) -> &'static str {
    match cat {
        OpCategory::Glyph => "Geometric Topology",
        OpCategory::Relation => "Structural Relations",
        OpCategory::Compare => "Value Comparison",
        OpCategory::Math => "Arithmetic",
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    show_operator_catalog();

    println!("\n🌟 RUNE Expression Examples:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Example RUNE source demonstrating operator precedence
    let examples = vec![
        "fiber_net / hub /\\ endpoint",     // Split topology
        "data /\\ validation ~ normalized", // Branch then transform
        "root \\|/ modes",                  // Symmetric splitting
        "users / 0 -> role := admin",       // Navigation + definition
        "vec_a + vec_b * 2",                // Math precedence
        "T::Gf8 terms * 2 + offset",        // Type annotation + math
    ];

    for example in examples {
        println!("Expression: {}", example);

        // Parse the expression (our parser focuses on full statements)
        if let Ok(stmts) = rune::parse_rune(&format!("dummy {}", example)) {
            if let Some(rune::Stmt::Expr(_)) = stmts.first() {
                println!("  ✓ Parsed successfully (statement-level)");
            }
        } else {
            println!("  → Parser note (expected for partial examples)");
        }
        println!();
    }

    println!("🔍 Operator Semantics:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Glyph Ops (Topology):");
    println!("  /\\  → Split then converge (detour + rejoin)");
    println!("  \\/  → Converge then split (merge + diverge)");
    println!("  |/   → Stable → branch away");
    println!("  /|   → Branch → stabilize");
    println!("  \\|  → Approach root → stabilize");
    println!("  |\\  → Stabilize → approach root");
    println!("  \\|/ → Symmetric fork from stable center");
    println!("  /|\\ → Complex branch-anchor-branch");
    println!();
    println!("Relations:");
    println!("  /   → Child/descendant (E8 lattice)");
    println!("  \\  → Parent/ancestor (root approach)");
    println!("  ->  → Flow/directed edge right-wards");
    println!("  <-  → Reverse flow left-wards");
    println!("  :=  → Definition/allocation");
    println!("  |   → Alias/equivalence");
    println!("  ||  → Parallel/peer relationship");
    println!("  ~   → Transform/view conversion");

    Ok(())
}
