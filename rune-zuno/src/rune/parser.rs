/* src/rune/parser.rs */
//!
//! # e8 Notation – RUNE Parser
//!▫~•◦-------------------------‣
//!
//! This module provides parsing functionality for RUNE source code,
//! converting text into `Stmt` structures with proper operator precedence.
//! It uses Pest for lexical analysis and implements expression parsing
//! driven by the grammar’s precedence layering.
//!
//! The parser handles:
//! - **Operator precedence** via grammar layers:
//!   - `mul`   → `*` level
//!   - `add_sub` → `+` / `-`
//!   - `access` / `relation_expr` / `expr` → structural / relation ops
//! - **TOON blocks**: Raw content preservation for later TOON library parsing
//! - **RUNE blocks**: Preferred raw content blocks for executable Rune data
//! - **Root declarations**: Semantic anchors for E8 contexts
//! - **Expression trees**: Recursive binary structures respecting precedence
//!
//! ### Implementation Details
//! - Uses grammar-encoded precedence (`term (op term)*` per layer).
//! - Preserves TOON blocks as raw strings without internal parsing.
//! - Validates all operators against the closed `RuneOp` registry.
//!
//! ### Error Handling
//! Parser errors include:
//! - Invalid operators (not in registry)
//! - Mismatched parentheses
//! - Malformed TOON blocks
//! - Unexpected tokens
//!
//! ### Example
//! ```rust
//! use rune_format::rune::parse_rune;
//!
//! let input = r#"
//! root: continuum
//! data ~RUNE:
//!   users[2]{id,name}:
//!     1,Ada
//!     2,Bob
//! users / 1 := Bob
//! "#;
//!
//! let stmts = parse_rune(input).unwrap();
//! // stmts contains RootDecl, RuneBlock, and ExprStmt
//! ```
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use std::str::FromStr;
use yoshi_derive::AnyError;

use crate::rune::ast::*;
use crate::rune::ops::*;

// Pest grammar reference
#[derive(Parser)]
#[grammar = "rune/grammar.pest"]
pub struct RuneParser;

/// Root error type for parsing RUNE source code.
#[derive(Debug, AnyError)]
pub enum ParseError {
    #[anyerror("Pest parse error: {0}")]
    Pest(Box<pest::error::Error<Rule>>),
    #[anyerror("Invalid operator '{0}' not in registry")]
    InvalidOperator(String),
    #[anyerror("Expected identifier, found: {0}")]
    ExpectedIdent(String),
    #[anyerror("Expected number, found: {0}")]
    ExpectedNumber(String),
    #[anyerror("Parse tree error: {0}")]
    ParseTree(String),
}

/// Parse RUNE source code into a list of statements.
pub fn parse(input: &str) -> Result<Vec<Stmt>, ParseError> {
    let pairs = RuneParser::parse(Rule::file, input).map_err(|e| ParseError::Pest(Box::new(e)))?;
    let mut stmts = Vec::new();

    for pair in pairs {
        if pair.as_rule() == Rule::file {
            for inner_pair in pair.into_inner() {
                match inner_pair.as_rule() {
                    Rule::WHITESPACE | Rule::COMMENT => {} // skip
                    Rule::stmt => {
                        if let Some(stmt_pair) = inner_pair.into_inner().next() {
                            stmts.push(parse_stmt(stmt_pair)?);
                        }
                    }
                    Rule::root_decl | Rule::toon_block | Rule::stmt_expr => {
                        stmts.push(parse_stmt(inner_pair)?);
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(stmts)
}

/// Parse and return typed statements using a shallow type inference pass.
pub fn parse_typed(input: &str) -> Result<Vec<crate::rune::ast::StmtTyped>, ParseError> {
    let stmts = parse(input)?;
    let mut typed: Vec<crate::rune::ast::StmtTyped> = Vec::new();
    for stmt in stmts {
        match stmt {
            crate::rune::ast::Stmt::RootDecl(id) => {
                typed.push(crate::rune::ast::StmtTyped::root(id.to_string()))
            }
            crate::rune::ast::Stmt::ToonBlock { name, content } => typed.push(
                crate::rune::ast::StmtTyped::toon_block(name.to_string(), content),
            ),
            crate::rune::ast::Stmt::KernelDecl { name, archetype } => {
                typed.push(crate::rune::ast::StmtTyped::KernelDecl {
                    name: name.clone(),
                    archetype: archetype.clone(),
                });
            }
            crate::rune::ast::Stmt::RuneBlock { name, content } => typed.push(
                crate::rune::ast::StmtTyped::rune_block(name.to_string(), content),
            ),
            crate::rune::ast::Stmt::Expr(expr) => {
                let te = crate::rune::ast::TypedExpr::infer(&expr);
                typed.push(crate::rune::ast::StmtTyped::expr(te));
            }
        }
    }
    Ok(typed)
}

/// Parse a statement pair into a Stmt.
fn parse_stmt(pair: Pair<Rule>) -> Result<Stmt, ParseError> {
    let rule = pair.as_rule();
    match rule {
        Rule::root_decl => parse_root_decl(pair),
        Rule::toon_block => parse_toon_block(pair),
        Rule::rune_block => parse_rune_block(pair),
        Rule::kernel_decl => parse_kernel_decl(pair),
        Rule::stmt_expr => {
            let expr_pair = pair.into_inner().next().unwrap();
            Ok(Stmt::expr(parse_expr(expr_pair)?))
        }
        _ => Err(ParseError::ParseTree(format!(
            "Unexpected statement rule: {:?}",
            rule
        ))),
    }
}

/// Parse root declaration: `root: name` or `root: e8::continuum`
fn parse_root_decl(pair: Pair<Rule>) -> Result<Stmt, ParseError> {
    let inner = pair.into_inner();

    let mut segments = Vec::new();

    // Collect all identifier segments (with :: separators)
    for pair in inner {
        if pair.as_rule() == Rule::ident {
            segments.push(pair.as_str());
        }
    }

    if segments.is_empty() {
        return Err(ParseError::ParseTree(
            "root declaration missing identifier".to_string(),
        ));
    }

    // Join segments with :: to create the full name
    let name = segments.join("::");
    Ok(Stmt::root(&name))
}

/// Parse TOON block: `name ~TOON:\n  content\n  content`
fn parse_toon_block(pair: Pair<Rule>) -> Result<Stmt, ParseError> {
    let mut inner = pair.into_inner();
    let ident_pair = inner.next().unwrap();
    let name = ident_pair.as_str();

    // The next pair is toon_content (atomic capture of all lines)
    let content_pair = inner.next().unwrap();
    let content = content_pair.as_str();

    // Split into lines and dedent (remove common leading whitespace)
    let lines: Vec<&str> = content.lines().collect();

    if lines.is_empty() {
        return Ok(Stmt::toon_block(name, String::new()));
    }

    // Find minimum indentation (excluding empty lines)
    let min_indent = lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);

    // Remove the common indentation from all lines
    let dedented: Vec<String> = lines
        .iter()
        .map(|line| {
            if line.trim().is_empty() {
                String::new()
            } else {
                line[min_indent..].to_string()
            }
        })
        .collect();

    let final_content = dedented.join("\n");

    Ok(Stmt::toon_block(name, final_content))
}

/// Parse RUNE block: `name ~RUNE:\n  content\n  content`
fn parse_rune_block(pair: Pair<Rule>) -> Result<Stmt, ParseError> {
    let mut inner = pair.into_inner();
    let ident_pair = inner.next().unwrap();
    let name = ident_pair.as_str();

    let content_pair = inner.next().unwrap();
    let content = content_pair.as_str();

    let lines: Vec<&str> = content.lines().collect();

    if lines.is_empty() {
        return Ok(Stmt::rune_block(name, String::new()));
    }

    let min_indent = lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);

    let dedented: Vec<String> = lines
        .iter()
        .map(|line| {
            if line.trim().is_empty() {
                "".to_string()
            } else {
                line.chars().skip(min_indent).collect()
            }
        })
        .collect();

    Ok(Stmt::rune_block(name, dedented.join("\n")))
}

/// Parse kernel parameter: ident : (number | ident | string)
fn parse_kernel_param(pair: Pair<Rule>) -> Result<(Ident, Literal), ParseError> {
    let mut inner = pair.into_inner();
    let name_pair = inner.next().unwrap();
    let name = Ident::new(name_pair.as_str());

    inner.next(); // :
    inner.next(); // WHITESPACE*

    let value_pair = inner.next().unwrap();
    let value = match value_pair.as_rule() {
        Rule::number => {
            let num = value_pair
                .as_str()
                .parse()
                .map_err(|_| ParseError::ExpectedNumber(value_pair.as_str().to_string()))?;
            Literal::Number(num)
        }
        Rule::string => {
            let raw = value_pair.as_str();
            let content = &raw[1..raw.len() - 1];
            let unescaped = content
                .replace("\\\"", "\"")
                .replace("\\\\", "\\")
                .replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t");
            Literal::String(unescaped)
        }
        Rule::ident => Literal::String(value_pair.as_str().to_string()),
        _ => {
            return Err(ParseError::ParseTree(format!(
                "Unexpected value type in kernel param: {:?}",
                value_pair.as_rule()
            )));
        }
    };

    Ok((name, value))
}

/// Parse kernel archetype: CUDA:Archetype:ident(params...)
fn parse_kernel_archetype(pair: Pair<Rule>) -> Result<KernelArchetype, ParseError> {
    let mut inner = pair.into_inner();
    inner.next(); // "CUDA:Archetype:"
    let name_pair = inner.next().unwrap();
    let name = Ident::new(name_pair.as_str());
    inner.next(); // "("

    let mut params = Vec::new();
    while let Some(pair) = inner.next() {
        if pair.as_rule() == Rule::kernel_param {
            params.push(parse_kernel_param(pair)?);
            // Next should be , or )
            let next = inner.next();
            if let Some(sep) = next {
                if sep.as_str() == "," {
                    // Skip WHITESPACE*
                    let ws = inner.next();
                    if let Some(ws_pair) = ws {
                        if ws_pair.as_rule() != Rule::WHITESPACE {
                            return Err(ParseError::ParseTree(
                                "Expected whitespace after comma".to_string(),
                            ));
                        }
                    }
                } else if sep.as_str() == ")" {
                    break;
                } else {
                    return Err(ParseError::ParseTree(format!(
                        "Expected , or ), got {}",
                        sep.as_str()
                    )));
                }
            }
        } else if pair.as_str() == ")" {
            break;
        } else {
            return Err(ParseError::ParseTree(format!(
                "Unexpected in kernel archetype: {:?}",
                pair.as_rule()
            )));
        }
    }

    Ok(KernelArchetype { name, params })
}

/// Parse kernel declaration: semantic_ident := kernel_archetype
fn parse_kernel_decl(pair: Pair<Rule>) -> Result<Stmt, ParseError> {
    let mut inner = pair.into_inner();
    let semantic_ident_pair = inner.next().unwrap();
    let semantic_ident = {
        let mut si_inner = semantic_ident_pair.into_inner();
        let prefix_pair = si_inner.next().unwrap();
        let name_pair = si_inner.next().unwrap();
        let prefix = prefix_pair.as_str().chars().next().unwrap();
        let name = Ident::new(name_pair.as_str());
        SemanticIdent::new(prefix, name)
    };

    inner.next(); // :=
    inner.next(); // WHITESPACE*
    let kernel_archetype_pair = inner.next().unwrap();
    let archetype = parse_kernel_archetype(kernel_archetype_pair)?;

    Ok(Stmt::KernelDecl {
        name: semantic_ident,
        archetype,
    })
}

/// Parse expression using grammar-driven precedence.
///
/// We rely on the Pest grammar to encode precedence via nested rules:
/// - `flow_expr` wraps `struct_expr` (lower precedence)
/// - `struct_expr` wraps `access` (higher precedence)
/// - `access` wraps `term`
///
/// Each non-terminal is parsed as:
///   sub_expr (op sub_expr)*
fn parse_expr(pair: Pair<Rule>) -> Result<Expr, ParseError> {
    match pair.as_rule() {
        // Expression layers for structural operators
        Rule::relation_expr | Rule::flow_expr | Rule::struct_expr | Rule::access => {
            let mut inner = pair.into_inner();

            // First element is always a sub-expression or term.
            let first = inner
                .next()
                .ok_or_else(|| ParseError::ParseTree("Empty expression".to_string()))?;

            let mut left = match first.as_rule() {
                Rule::relation_expr | Rule::flow_expr | Rule::struct_expr | Rule::access => {
                    parse_expr(first)?
                }
                Rule::term => parse_term(first)?,
                _ => parse_term(first)?,
            };

            // Then we expect zero or more (op, rhs) pairs.
            while let Some(op_pair) = inner.next() {
                // Determine what kind of pair this is
                let (op, right) = match op_pair.as_rule() {
                    // If it's one of the named operator rules, parse it
                    Rule::relation_op | Rule::flow_op | Rule::struct_op | Rule::path_op => {
                        let op = parse_operator(op_pair)?;
                        let rhs_pair = inner.next().ok_or_else(|| {
                            ParseError::ParseTree("Missing right operand".to_string())
                        })?;
                        let right = match rhs_pair.as_rule() {
                            Rule::relation_expr
                            | Rule::flow_expr
                            | Rule::struct_expr
                            | Rule::access => parse_expr(rhs_pair)?,
                            Rule::term => parse_term(rhs_pair)?,
                            _ => parse_term(rhs_pair)?,
                        };
                        (op, right)
                    }
                    // If it's another expression layer, something is wrong
                    Rule::access | Rule::struct_expr | Rule::flow_expr | Rule::relation_expr => {
                        return Err(ParseError::ParseTree(format!(
                            "Unexpected expression node where operator expected: {:?}",
                            op_pair.as_rule()
                        )));
                    }
                    _ => {
                        // Fallback: treat as operator by text
                        let op = parse_operator(op_pair)?;
                        let rhs_pair = inner.next().ok_or_else(|| {
                            ParseError::ParseTree("Missing right operand".to_string())
                        })?;
                        let right = match rhs_pair.as_rule() {
                            Rule::relation_expr
                            | Rule::flow_expr
                            | Rule::struct_expr
                            | Rule::access => parse_expr(rhs_pair)?,
                            Rule::term => parse_term(rhs_pair)?,
                            _ => parse_term(rhs_pair)?,
                        };
                        (op, right)
                    }
                };

                left = Expr::binary(left, op, right);
            }

            Ok(left)
        }
        // Direct term -> literal / ident / grouped expr
        Rule::term => parse_term(pair),
        _ => Err(ParseError::ParseTree(format!(
            "Unexpected expression rule: {:?}",
            pair.as_rule()
        ))),
    }
}

/// Parse a term: identifier, number, string, array, grouped expression, or math block.
fn parse_term(pair: Pair<Rule>) -> Result<Expr, ParseError> {
    match pair.as_rule() {
        Rule::term => {
            // Term is a composite rule, get its inner content
            let inner = pair
                .into_inner()
                .next()
                .ok_or_else(|| ParseError::ParseTree("Empty term".to_string()))?;
            parse_term(inner) // Recursively parse the inner rule
        }
        Rule::array_literal => {
            // Parse array literal: [expr, expr, ...]
            let inner = pair.into_inner();
            let mut elements = Vec::new();

            for expr_pair in inner {
                elements.push(parse_expr(expr_pair)?);
            }

            Ok(Expr::Term(Term::Literal(Literal::Array(elements))))
        }
        Rule::object_literal => {
            // Parse object literal: {key: value, ...}
            let inner = pair.into_inner();
            let mut entries = Vec::new();

            for entry_pair in inner {
                if let Rule::object_entry = entry_pair.as_rule() {
                    let mut entry_inner = entry_pair.into_inner();

                    // Parse key (ident or string)
                    let key_pair = entry_inner.next().ok_or_else(|| {
                        ParseError::ParseTree("Missing key in object entry".to_string())
                    })?;
                    let key = match key_pair.as_rule() {
                        Rule::ident => key_pair.as_str().to_string(),
                        Rule::string => {
                            let raw = key_pair.as_str();
                            // Remove surrounding quotes from string keys
                            if raw.len() >= 2 && raw.starts_with('"') && raw.ends_with('"') {
                                raw[1..raw.len() - 1].to_string()
                            } else {
                                raw.to_string()
                            }
                        }
                        _ => {
                            return Err(ParseError::ParseTree(format!(
                                "Unexpected object key rule: {:?}",
                                key_pair.as_rule()
                            )));
                        }
                    };

                    // Parse value (expression)
                    let val_pair = entry_inner.next().ok_or_else(|| {
                        ParseError::ParseTree("Missing value in object entry".to_string())
                    })?;
                    let val_expr = parse_expr(val_pair)?;

                    entries.push((key, val_expr));
                }
            }

            Ok(Expr::Term(Term::Literal(Literal::Object(entries))))
        }
        Rule::semantic_ident => {
            // Parse semantic identifier: prefix:name
            let mut inner = pair.into_inner();
            let prefix_pair = inner.next().unwrap();
            let name_pair = inner.next().unwrap();

            // Extract prefix character (first char before the colon)
            let prefix_str = prefix_pair.as_str();
            let prefix = prefix_str.chars().next().unwrap();

            let name = name_pair.as_str();
            Ok(Expr::Term(Term::semantic_ident(prefix, name)))
        }
        Rule::ident => Ok(Expr::ident(pair.as_str())),
        Rule::fn_call => {
            // Parse function call: name(arg1, arg2, ...)
            let mut inner = pair.into_inner();
            let name_pair = inner
                .next()
                .ok_or_else(|| ParseError::ParseTree("Missing function name".to_string()))?;
            let name = Ident::new(name_pair.as_str());

            let mut args = Vec::new();
            for expr_pair in inner {
                args.push(parse_expr(expr_pair)?);
            }

            Ok(Expr::Term(Term::FunctionCall { name, args }))
        }
        Rule::number => {
            let num: f64 = pair
                .as_str()
                .parse()
                .map_err(|_| ParseError::ExpectedNumber(pair.as_str().to_string()))?;
            Ok(Expr::literal(num))
        }
        Rule::boolean_literal => {
            // Parse boolean literal: B:t (true) or B:f (false)
            let text = pair.as_str();
            match text {
                "B:t" => Ok(Expr::Term(Term::Literal(Literal::bool(true)))),
                "B:f" => Ok(Expr::Term(Term::Literal(Literal::bool(false)))),
                _ => Err(ParseError::ParseTree(format!(
                    "Invalid boolean literal: {}",
                    text
                ))),
            }
        }
        Rule::string => {
            // Parse string, handling escape sequences
            let raw = pair.as_str();
            // Remove surrounding quotes
            let content = &raw[1..raw.len() - 1];
            // Unescape common sequences
            let unescaped = content
                .replace("\\\"", "\"")
                .replace("\\\\", "\\")
                .replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t");
            Ok(Expr::Term(Term::Literal(Literal::String(unescaped))))
        }
        Rule::relation_expr | Rule::flow_expr | Rule::struct_expr | Rule::access => {
            // These are expression nodes that can appear as terms (e.g., in parentheses)
            parse_expr(pair)
        }
        Rule::math_block => {
            // Math blocks are at the term level, parse content as math expression
            let math_expr = parse_math_expr(pair)?;
            Ok(Expr::Term(Term::Math(Box::new(math_expr))))
        }
        _ => Err(ParseError::ParseTree(format!(
            "Unexpected term rule: {:?}",
            pair.as_rule()
        ))),
    }
}

/// Parse math expression from a math block `[...]`.
/// Handles arithmetic operators with proper precedence.
fn parse_math_expr(pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
    // The pair is a math_block, we need the inner math_expr
    let math_expr_pair = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::ParseTree("Empty math block".to_string()))?;

    parse_math_expr_inner(math_expr_pair)
}

/// Internal math expression parser.
fn parse_math_expr_inner(pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
    match pair.as_rule() {
        Rule::math_expr | Rule::math_add | Rule::math_mul | Rule::math_exp => {
            let mut inner = pair.into_inner();

            let first = inner
                .next()
                .ok_or_else(|| ParseError::ParseTree("Empty math expression".to_string()))?;

            let mut left = match first.as_rule() {
                Rule::math_expr | Rule::math_add | Rule::math_mul | Rule::math_exp => {
                    parse_math_expr_inner(first)?
                }
                Rule::math_unary => parse_math_unary(first)?,
                Rule::math_atom => parse_math_atom(first)?,
                _ => parse_math_expr_inner(first)?,
            };

            while let Some(op_pair) = inner.next() {
                let op = parse_math_operator(op_pair)?;

                let rhs_pair = inner.next().ok_or_else(|| {
                    ParseError::ParseTree("Missing right operand in math".to_string())
                })?;

                let right = match rhs_pair.as_rule() {
                    Rule::math_expr | Rule::math_add | Rule::math_mul | Rule::math_exp => {
                        parse_math_expr_inner(rhs_pair)?
                    }
                    Rule::math_unary => parse_math_unary(rhs_pair)?,
                    Rule::math_atom => parse_math_atom(rhs_pair)?,
                    _ => parse_math_atom(rhs_pair)?,
                };

                left = MathExpr::binary(left, op, right);
            }

            Ok(left)
        }
        Rule::math_unary => parse_math_unary(pair),
        Rule::math_atom => parse_math_atom(pair),
        _ => Err(ParseError::ParseTree(format!(
            "Unexpected math expression rule: {:?}",
            pair.as_rule()
        ))),
    }
}

/// Parse a math atom: number, identifier, or grouped expression.
fn parse_math_atom(pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::ParseTree("Empty math atom".to_string()))?;

    match inner.as_rule() {
        Rule::number => {
            let num: f64 = inner
                .as_str()
                .parse()
                .map_err(|_| ParseError::ExpectedNumber(inner.as_str().to_string()))?;
            Ok(MathExpr::atom(MathAtom::Number(num)))
        }
        Rule::ident => Ok(MathExpr::atom(MathAtom::Ident(Ident::new(inner.as_str())))),
        Rule::semantic_ident => {
            // Parse semantic identifier inside math blocks - treat as regular identifier for now
            Ok(MathExpr::atom(MathAtom::Ident(Ident::new(inner.as_str()))))
        }
        Rule::math_array_literal => {
            // Parse array literal inside math blocks
            let elements: Result<Vec<MathExpr>, ParseError> =
                inner.into_inner().map(parse_math_expr_inner).collect();
            Ok(MathExpr::atom(MathAtom::Array(elements?)))
        }
        Rule::math_expr => Ok(MathExpr::atom(MathAtom::Group(Box::new(
            parse_math_expr_inner(inner)?,
        )))),
        _ => Err(ParseError::ParseTree(format!(
            "Unexpected math atom rule: {:?}",
            inner.as_rule()
        ))),
    }
}

/// Parse unary expression: optional prefix operator followed by atom.
fn parse_math_unary(pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
    let mut inner = pair.into_inner();

    let first = inner
        .next()
        .ok_or_else(|| ParseError::ParseTree("Empty unary expression".to_string()))?;

    // Check if first is a unary operator
    match first.as_rule() {
        Rule::math_unary_op => {
            let op = parse_math_unary_operator(first)?;
            let operand_pair = inner.next().ok_or_else(|| {
                ParseError::ParseTree("Missing operand after unary operator".to_string())
            })?;

            let operand = match operand_pair.as_rule() {
                Rule::math_atom => parse_math_atom(operand_pair)?,
                Rule::math_unary => parse_math_unary(operand_pair)?,
                _ => {
                    return Err(ParseError::ParseTree(format!(
                        "Unexpected unary operand rule: {:?}",
                        operand_pair.as_rule()
                    )));
                }
            };

            Ok(MathExpr::unary(op, operand))
        }
        Rule::math_atom => parse_math_atom(first),
        _ => Err(ParseError::ParseTree(format!(
            "Unexpected unary rule: {:?}",
            first.as_rule()
        ))),
    }
}

/// Parse math operator into MathOp.
fn parse_math_operator(pair: Pair<Rule>) -> Result<MathOp, ParseError> {
    match pair.as_str().trim() {
        "+" => Ok(MathOp::Add),
        "-" => Ok(MathOp::Subtract),
        "*" => Ok(MathOp::Multiply),
        "/" => Ok(MathOp::Divide),
        "%" => Ok(MathOp::Modulo),
        "^" => Ok(MathOp::Power),
        "R" => Ok(MathOp::Root),
        op => Err(ParseError::InvalidOperator(format!(
            "Unknown math operator: {}",
            op
        ))),
    }
}

/// Parse unary operator into MathUnaryOp.
fn parse_math_unary_operator(pair: Pair<Rule>) -> Result<MathUnaryOp, ParseError> {
    match pair.as_str().trim() {
        "-" => Ok(MathUnaryOp::Negate),
        "+" => Ok(MathUnaryOp::Plus),
        op => Err(ParseError::InvalidOperator(format!(
            "Unknown unary operator: {}",
            op
        ))),
    }
}

/// Parse operator token into RuneOp.
///
/// We defensively trim whitespace so that rules which
/// include incidental spaces around operators do not
/// accidentally produce `"+"`, `"1 "` or `"b * c"` as a
/// single operator token.
fn parse_operator(pair: Pair<Rule>) -> Result<RuneOp, ParseError> {
    let text = pair.as_str().trim();
    RuneOp::from_str(text).map_err(|_| ParseError::InvalidOperator(text.to_string()))
}
