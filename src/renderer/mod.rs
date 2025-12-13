
/* src/renderer/mod.rs */
//! Zero-Copy Renderer for RUNE.
//!
//! # Rune-Xero – Absolute Zero-Copy Rendering
//!▫~•◦-----------------------------‣
//!
//! This module renders the AST back into RUNE source text without allocating
//! intermediate strings for values or operators. All writes go directly into
//! the output buffer.
//!
//! ### Key Capabilities
//! - **Zero-Allocation for Values:** Custom `render_value()` avoids `to_string()`.
//! - **Operator Rendering:** Uses `as_str()` for enums instead of allocating.
//! - **Preserves Original Semantics:** AST slices are written as-is.
//!
//! ### Example
//! ```rust
//! use rune_xero::renderer::render;
//! use rune_xero::parser::parse;
//!
//! let input = r#"root: continuum"#;
//! let doc = parse(input).unwrap();
//! let output = render(&doc);
//! assert_eq!(output.trim(), input);
//! ```
//!
///▫~•◦------------------------------------------------------------------------------------‣
/// © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
///•------------------------------------------------------------------------------------‣

use crate::decoder::parser::ast::*;

pub fn render(doc: &Document<'_>) -> String {
    let mut out = String::new();
    for (i, item) in doc.items.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        match item {
            Item::Statement(stmt) => render_statement(stmt, &mut out),
            Item::Section(sec) => render_section(sec, &mut out),
        }
    }
    out
}

fn render_statement(stmt: &Statement<'_>, out: &mut String) {
    match stmt {
        Statement::RootDecl(name) => {
            out.push_str("root: ");
            out.push_str(name);
            out.push('\n');
        }
        Statement::KernelDecl { name, archetype } => {
            out.push_str(name);
            out.push_str(" := CUDA:Archetype:");
            out.push_str(archetype.name);
            out.push('(');
            for (i, (p_name, p_val)) in archetype.params.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(p_name);
                out.push_str(": ");
                render_value(p_val, out);
            }
            out.push_str(")\n");
        }
        Statement::Expr(expr) => {
            render_expression(expr, out);
            out.push('\n');
        }
    }
}

fn render_section(sec: &Section<'_>, out: &mut String) {
    out.push_str(sec.name);
    out.push(' ');
    match sec.kind {
        SectionKind::Toon => out.push_str("~TOON:\n"),
        SectionKind::Rune => out.push_str("~RUNE:\n"),
    }
    out.push_str(sec.content);
    if !sec.content.ends_with('\n') {
        out.push('\n');
    }
}

fn render_expression(expr: &Expression<'_>, out: &mut String) {
    match expr {
        Expression::Term(term) => render_term(term, out),
        Expression::Binary { left, op, right } => {
            render_expression(left, out);
            out.push(' ');
            out.push_str(op);
            out.push(' ');
            render_expression(right, out);
        }
    }
}

fn render_term(term: &Term<'_>, out: &mut String) {
    match term {
        Term::Ident(s) => out.push_str(s),
        Term::SemanticIdent { prefix, name } => {
            out.push(*prefix);
            out.push(':');
            out.push_str(name);
        }
        Term::Literal(val) => render_value(val, out),
        Term::Call { name, args } => {
            out.push_str(name);
            out.push('(');
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                render_expression(arg, out);
            }
            out.push(')');
        }
        Term::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                render_expression(item, out);
            }
            out.push(']');
        }
        Term::Object(entries) => {
            out.push('{');
            for (i, (k, v)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(k);
                out.push_str(": ");
                render_expression(v, out);
            }
            out.push('}');
        }
        Term::Math(math) => {
            out.push('[');
            render_math(math, out);
            out.push(']');
        }
        Term::Tabular(rows) => {
            // Render tabular data as a standard array of objects for safety
            out.push('[');
            for (i, row) in rows.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                render_value(row, out);
            }
            out.push(']');
        }
    }
}

fn render_value(val: &Value<'_>, out: &mut String) {
    match val {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Float(n) => {
            // Avoid allocation: write directly
            use std::fmt::Write;
            write!(out, "{:.6}", n).unwrap();
        }
        Value::Str(s) | Value::Raw(s) => out.push_str(s),
        Value::Array(arr) => {
            out.push('[');
            for (i, v) in arr.iter().enumerate() {
                if i > 0 { out.push_str(", "); }
                render_value(v, out);
            }
            out.push(']');
        }
        Value::Object(entries) => {
            out.push('{');
            for (i, (k, v)) in entries.iter().enumerate() {
                if i > 0 { out.push_str(", "); }
                out.push_str(k);
                out.push_str(": ");
                render_value(v, out);
            }
            out.push('}');
        }
    }
}

fn render_math(math: &MathExpr<'_>, out: &mut String) {
    match math {
        MathExpr::Atom(atom) => match atom {
            MathAtom::Number(n) => {
                use std::fmt::Write;
                write!(out, "{:.6}", n).unwrap();
            }
            MathAtom::Ident(s) => out.push_str(s),
            MathAtom::Group(inner) => {
                out.push('(');
                render_math(inner, out);
                out.push(')');
            }
            MathAtom::Array(arr) => {
                out.push('[');
                for (i, m) in arr.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    render_math(m, out);
                }
                out.push(']');
            }
        },
        MathExpr::Binary { left, op, right } => {
            render_math(left, out);
            out.push(' ');
            out.push_str(op.as_str()); // Use enum's as_str() if op is MathOp
            out.push(' ');
            render_math(right, out);
        }
        MathExpr::Unary { op, operand } => {
            out.push_str(match op {
                MathUnaryOp::Negate => "-",
                MathUnaryOp::Plus => "+",
            });
            render_math(operand, out);
        }
    }
}
