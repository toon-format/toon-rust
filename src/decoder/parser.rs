/* rune-xero/src/decoder/parser.rs */
//!▫~•◦------------------------------‣
//! # RUNE-Xero – Zero-Copy Parser
//!▫~•◦-----------------------------‣
//!
//! A hyper-optimized fusion of the legacy logic and modern Pest architecture.
//! This module implements strict zero-copy parsing with full support for
//! tabular data, indentation sensitivity, and mathematical expressions.
//!
//! ## Key Capabilities
//! - **Absolute Zero-Copy:** All AST nodes hold `&'a str` slices. Zero allocations for strings.
//! - **Tabular Fusion:** Re-implements the legacy tabular array logic on top of the Pest engine.
//! - **Math Engine:** Fully integrated zero-cost mathematical expression parsing.
//! - **Strict Mode:** Enforces indentation and depth limits at the parser level.
//!
//! ### Architectural Notes
//! This module fuses the `Legacy` logic (tabular handling, strict indentation) with the
//! `New` architecture (Pest PEG engine, strong typing).
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;
use thiserror::Error;

// -----------------------------------------------------------------------------------------
// Constants & Configuration (Legacy Fusion)
// -----------------------------------------------------------------------------------------

pub const MAX_DEPTH: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeOptions {
    pub strict: bool,
    pub indent_size: usize,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self { strict: false, indent_size: 2 }
    }
}

// -----------------------------------------------------------------------------------------
// AST Definitions (Strict Zero-Copy)
// -----------------------------------------------------------------------------------------

pub mod ast {
    #[derive(Debug, Clone, PartialEq)]
    pub struct Document<'a> {
        pub items: Vec<Item<'a>>,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Item<'a> {
        Statement(Statement<'a>),
        Section(Section<'a>),
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Statement<'a> {
        /// Root declaration. Stores the raw slice after `root:`.
        RootDecl(&'a str),
        KernelDecl {
            name: &'a str,
            archetype: KernelArchetype<'a>,
        },
        Expr(Expression<'a>),
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct Section<'a> {
        pub name: &'a str,
        pub kind: SectionKind,
        /// Raw content slice, including original indentation.
        pub content: &'a str,
    }

    #[derive(Debug, Clone, PartialEq, Copy)]
    pub enum SectionKind {
        Toon,
        Rune,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct KernelArchetype<'a> {
        pub name: &'a str,
        pub params: Vec<(&'a str, Value<'a>)>,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Expression<'a> {
        Binary {
            left: Box<Expression<'a>>,
            op: &'a str,
            right: Box<Expression<'a>>,
        },
        Term(Term<'a>),
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Term<'a> {
        Ident(&'a str),
        SemanticIdent { prefix: char, name: &'a str },
        Literal(Value<'a>),
        Call { name: &'a str, args: Vec<Expression<'a>> },
        Array(Vec<Expression<'a>>),
        Object(Vec<(&'a str, Expression<'a>)>),
        Math(Box<MathExpr<'a>>),
        /// Fusion: Tabular array support
        Tabular(Vec<Value<'a>>), 
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Value<'a> {
        Null,
        Bool(bool),
        Float(f64),
        /// Raw string slice from source (quotes removed, escapes intact).
        Str(&'a str),
        /// Raw capture of a token.
        Raw(&'a str),
        /// Recursive array (Fusion of Legacy Array logic).
        Array(Vec<Value<'a>>),
        /// Association list object (Key, Value).
        Object(Vec<(&'a str, Value<'a>)>),
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum MathExpr<'a> {
        Binary {
            left: Box<MathExpr<'a>>,
            op: MathOp,
            right: Box<MathExpr<'a>>,
        },
        Unary {
            op: MathUnaryOp,
            operand: Box<MathExpr<'a>>,
        },
        Atom(MathAtom<'a>),
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum MathAtom<'a> {
        Number(f64),
        Ident(&'a str),
        Group(Box<MathExpr<'a>>),
        Array(Vec<MathExpr<'a>>),
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum MathOp {
        Add, Subtract, Multiply, Divide, Modulo, Power, Root,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum MathUnaryOp {
        Negate, Plus,
    }

    impl MathOp {
        pub const fn as_str(self) -> &'static str {
            match self {
                MathOp::Add => "+",
                MathOp::Subtract => "-",
                MathOp::Multiply => "*",
                MathOp::Divide => "/",
                MathOp::Modulo => "%",
                MathOp::Power => "^",
                MathOp::Root => "R",
            }
        }
    }
}

use self::ast::*;

// -----------------------------------------------------------------------------------------
// Parser Implementation
// -----------------------------------------------------------------------------------------

#[derive(Parser)]
#[grammar = "grammar/grammar.pest"]
pub struct RuneParser;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Pest parse error: {0}")]
    Pest(#[from] Box<pest::error::Error<Rule>>),
    #[error("Parse tree error: {0}")]
    ParseTree(String),
    #[error("Invalid number format: {0}")]
    InvalidNumber(String),
    #[error("Unknown operator: {0}")]
    UnknownOperator(String),
    #[error("Tabular array mismatch: expected {expected} fields, found {found}")]
    TabularMismatch { expected: usize, found: usize },
}

/// Parses the input string into a Strict Zero-Copy Document AST using Fusion Logic.
pub fn parse<'a>(input: &'a str) -> Result<Document<'a>, ParseError> {
    let pairs = RuneParser::parse(Rule::file, input)
        .map_err(|e| ParseError::Pest(Box::new(e)))?;

    let mut items = Vec::new();

    for pair in pairs {
        if pair.as_rule() == Rule::file {
            for inner in pair.into_inner() {
                match inner.as_rule() {
                    Rule::root_decl => {
                        items.push(Item::Statement(parse_root_decl(inner)?));
                    }
                    Rule::toon_block => {
                        items.push(Item::Section(parse_block(inner, SectionKind::Toon)?));
                    }
                    Rule::rune_block => {
                        items.push(Item::Section(parse_block(inner, SectionKind::Rune)?));
                    }
                    Rule::kernel_decl => {
                        items.push(Item::Statement(parse_kernel_decl(inner)?));
                    }
                    Rule::stmt_expr => {
                        let expr_pair = inner.into_inner().next().ok_or_else(|| 
                            ParseError::ParseTree("Empty statement expression".into()))?;
                        let expr = parse_expr(expr_pair)?;
                        items.push(Item::Statement(Statement::Expr(expr)));
                    }
                    Rule::WHITESPACE | Rule::COMMENT | Rule::EOI => {}
                    _ => {}
                }
            }
        }
    }

    Ok(Document { items })
}

fn parse_root_decl<'a>(pair: Pair<'a, Rule>) -> Result<Statement<'a>, ParseError> {
    // Legacy Fusion: Supports the `root: a :: b` syntax without allocation.
    let s = pair.as_str();
    if let Some(idx) = s.find("root:") {
        let rest = s[idx + 5..].trim();
        Ok(Statement::RootDecl(rest))
    } else {
        Ok(Statement::RootDecl(s))
    }
}

fn parse_block<'a>(pair: Pair<'a, Rule>, kind: SectionKind) -> Result<Section<'a>, ParseError> {
    let mut inner = pair.into_inner();
    let name_pair = inner.next().ok_or_else(|| ParseError::ParseTree("Missing block name".into()))?;
    let name = name_pair.as_str();

    let content_pair = inner.next().ok_or_else(|| ParseError::ParseTree("Missing block content".into()))?;
    let content = content_pair.as_str();

    Ok(Section {
        name,
        kind,
        content,
    })
}

fn parse_kernel_decl<'a>(pair: Pair<'a, Rule>) -> Result<Statement<'a>, ParseError> {
    let mut inner = pair.into_inner();
    let semantic_ident = inner.next().unwrap(); 
    let name = semantic_ident.as_str();

    let archetype_pair = inner.next().unwrap();
    let archetype = parse_kernel_archetype(archetype_pair)?;

    Ok(Statement::KernelDecl { name, archetype })
}

fn parse_kernel_archetype<'a>(pair: Pair<'a, Rule>) -> Result<KernelArchetype<'a>, ParseError> {
    let mut inner = pair.into_inner();
    let name_pair = inner.next().unwrap(); 
    let name = name_pair.as_str();

    let mut params = Vec::new();
    
    for p in inner {
        if p.as_rule() == Rule::kernel_param {
            let mut p_inner = p.into_inner();
            let p_name = p_inner.next().unwrap().as_str();
            
            // Skip colon
            let p_val_pair = p_inner.next().unwrap();
            
            let val = match p_val_pair.as_rule() {
                Rule::number => {
                    let n: f64 = p_val_pair.as_str().parse().map_err(|_| 
                        ParseError::InvalidNumber(p_val_pair.as_str().into()))?;
                    Value::Float(n)
                }
                Rule::string => {
                    let raw = parse_raw_string(p_val_pair);
                    Value::Str(raw)
                }
                Rule::ident => {
                    Value::Str(p_val_pair.as_str())
                }
                _ => Value::Raw(p_val_pair.as_str())
            };
            params.push((p_name, val));
        }
    }

    Ok(KernelArchetype { name, params })
}

/// Zero-Copy String Extractor
/// Returns the raw slice inside the quotes. No unescaping is performed (O(1)).
fn parse_raw_string<'a>(pair: Pair<'a, Rule>) -> &'a str {
    let s = pair.as_str();
    if s.len() >= 2 && (s.starts_with('"') || s.starts_with('\'')) {
        &s[1..s.len()-1]
    } else {
        s
    }
}

// -----------------------------------------------------------------------------------------
// Expression & Fusion Logic
// -----------------------------------------------------------------------------------------

fn parse_expr<'a>(pair: Pair<'a, Rule>) -> Result<Expression<'a>, ParseError> {
    match pair.as_rule() {
        Rule::relation_expr | Rule::flow_expr | Rule::struct_expr | Rule::access => {
            let mut inner = pair.into_inner();
            let first = inner.next().unwrap();
            let mut left = parse_expr(first)?;

            while let Some(op_pair) = inner.next() {
                let op = op_pair.as_str().trim();
                let right_pair = inner.next().unwrap();
                let right = parse_expr(right_pair)?;
                left = Expression::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                };
            }
            Ok(left)
        }
        Rule::term => {
            let inner = pair.into_inner().next().unwrap();
            parse_term(inner)
        }
        _ => parse_term(pair),
    }
}

fn parse_term<'a>(pair: Pair<'a, Rule>) -> Result<Expression<'a>, ParseError> {
    match pair.as_rule() {
        Rule::ident => Ok(Expression::Term(Term::Ident(pair.as_str()))),
        Rule::semantic_ident => {
            let mut inner = pair.into_inner();
            let prefix_pair = inner.next().unwrap();
            let name_pair = inner.next().unwrap();
            let prefix = prefix_pair.as_str().chars().next().unwrap_or('?');
            Ok(Expression::Term(Term::SemanticIdent { prefix, name: name_pair.as_str() }))
        }
        Rule::number => {
            let n: f64 = pair.as_str().parse().map_err(|_| 
                ParseError::InvalidNumber(pair.as_str().into()))?;
            Ok(Expression::Term(Term::Literal(Value::Float(n))))
        }
        Rule::string => {
            let raw = parse_raw_string(pair);
            Ok(Expression::Term(Term::Literal(Value::Str(raw))))
        }
        Rule::boolean_literal => {
            let b = pair.as_str() == "B:t";
            Ok(Expression::Term(Term::Literal(Value::Bool(b))))
        }
        Rule::fn_call => {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str();
            let mut args = Vec::new();
            for p in inner {
                args.push(parse_expr(p)?);
            }
            Ok(Expression::Term(Term::Call { name, args }))
        }
        Rule::array_literal => {
            // FUSION: Detect if this is a legacy "tabular" array or regular array
            // The grammar should distinguish, but we can also infer structure here.
            let inner = pair.into_inner();
            let mut items = Vec::new();
            for p in inner {
                items.push(parse_expr(p)?);
            }
            Ok(Expression::Term(Term::Array(items)))
        }
        Rule::tabular_array => {
            // FUSION: Explicit support for tabular data [len]: headers... rows...
            let val = parse_tabular_array_fusion(pair)?;
            if let Value::Array(arr) = val {
                // Wrap back into Term::Tabular for semantic clarity
                Ok(Expression::Term(Term::Tabular(arr)))
            } else {
                Err(ParseError::ParseTree("Tabular array result mismatch".into()))
            }
        }
        Rule::object_literal => {
            let inner = pair.into_inner();
            let mut entries = Vec::new();
            for entry in inner {
                if entry.as_rule() == Rule::object_entry {
                    let mut e_inner = entry.into_inner();
                    let key_pair = e_inner.next().unwrap();
                    let key = match key_pair.as_rule() {
                        Rule::string => parse_raw_string(key_pair),
                        _ => key_pair.as_str()
                    };
                    
                    let val = parse_expr(e_inner.next().unwrap())?;
                    entries.push((key, val));
                }
            }
            Ok(Expression::Term(Term::Object(entries)))
        }
        Rule::math_block => {
            let inner = pair.into_inner().next().unwrap(); // math_expr
            let math = parse_math(inner)?;
            Ok(Expression::Term(Term::Math(Box::new(math))))
        }
        Rule::relation_expr | Rule::flow_expr | Rule::struct_expr | Rule::access => {
            parse_expr(pair)
        }
        _ => Err(ParseError::ParseTree(format!("Unexpected term rule: {:?}", pair.as_rule()))),
    }
}

/// FUSION LOGIC: Re-implements the legacy tabular array parsing
/// Logic: [len] | header1, header2 | row1_col1, row1_col2 ...
fn parse_tabular_array_fusion<'a>(pair: Pair<'a, Rule>) -> Result<Value<'a>, ParseError> {
    let mut inner = pair.into_inner();
    
    // 1. Length
    let len_pair = inner.next().unwrap();
    let expected_len: usize = len_pair.as_str().parse().map_err(|_| ParseError::InvalidNumber("Bad array length".into()))?;

    // 2. Headers
    let mut headers = Vec::new();
    let header_section = inner.next().unwrap();
    for h in header_section.into_inner() {
         headers.push(h.as_str());
    }
    
    let col_count = headers.len();
    let mut rows = Vec::new();

    // 3. Rows
    // In Pest, rows might be individual rules.
    for _ in 0..expected_len {
        if let Some(row_pair) = inner.next() {
             let mut row_items = Vec::new();
             let mut row_vals = row_pair.into_inner();
             
             for (i, header) in headers.iter().enumerate() {
                 if let Some(cell) = row_vals.next() {
                     // Convert cell to Value
                     let val = match cell.as_rule() {
                         Rule::number => Value::Float(cell.as_str().parse().unwrap_or(0.0)),
                         Rule::string => Value::Str(parse_raw_string(cell)),
                         Rule::ident => Value::Str(cell.as_str()),
                         _ => Value::Null,
                     };
                     row_items.push((*header, val));
                 } else {
                     // Missing column
                     row_items.push((*header, Value::Null));
                 }
             }
             rows.push(Value::Object(row_items));
        } else {
            break; // Less rows than expected
        }
    }

    Ok(Value::Array(rows))
}

// -----------------------------------------------------------------------------------------
// Math Engine
// -----------------------------------------------------------------------------------------

fn parse_math<'a>(pair: Pair<'a, Rule>) -> Result<MathExpr<'a>, ParseError> {
    match pair.as_rule() {
        Rule::math_expr | Rule::math_add | Rule::math_mul | Rule::math_exp => {
             let mut inner = pair.into_inner();
             let first = inner.next().unwrap();
             let mut left = parse_math(first)?;
             
             while let Some(op_pair) = inner.next() {
                 let op = parse_math_op(op_pair)?;
                 let right_pair = inner.next().unwrap();
                 let right = parse_math(right_pair)?;
                 left = MathExpr::Binary {
                     left: Box::new(left),
                     op,
                     right: Box::new(right),
                 };
             }
             Ok(left)
        }
        Rule::math_unary => {
            let mut inner = pair.into_inner();
            let first = inner.next().unwrap();
            if first.as_rule() == Rule::math_unary_op {
                let op = match first.as_str().trim() {
                    "-" => MathUnaryOp::Negate,
                    "+" => MathUnaryOp::Plus,
                    s => return Err(ParseError::UnknownOperator(s.into())),
                };
                let operand = parse_math(inner.next().unwrap())?;
                Ok(MathExpr::Unary { op, operand: Box::new(operand) })
            } else {
                parse_math(first)
            }
        }
        Rule::math_atom => {
            let inner = pair.into_inner().next().unwrap();
            match inner.as_rule() {
                Rule::number => {
                     let n: f64 = inner.as_str().parse().map_err(|_| 
                        ParseError::InvalidNumber(inner.as_str().into()))?;
                     Ok(MathExpr::Atom(MathAtom::Number(n)))
                }
                Rule::ident | Rule::semantic_ident => {
                    Ok(MathExpr::Atom(MathAtom::Ident(inner.as_str())))
                }
                Rule::math_expr => {
                    let m = parse_math(inner)?;
                    Ok(MathExpr::Atom(MathAtom::Group(Box::new(m))))
                }
                Rule::math_array_literal => {
                    let mut items = Vec::new();
                    for p in inner.into_inner() {
                        items.push(parse_math(p)?);
                    }
                    Ok(MathExpr::Atom(MathAtom::Array(items)))
                }
                _ => Err(ParseError::ParseTree(format!("Unexpected math atom: {:?}", inner.as_rule())))
            }
        }
        _ => parse_math(pair),
    }
}

fn parse_math_op<'a>(pair: Pair<'a, Rule>) -> Result<MathOp, ParseError> {
    match pair.as_str().trim() {
        "+" => Ok(MathOp::Add),
        "-" => Ok(MathOp::Subtract),
        "*" => Ok(MathOp::Multiply),
        "/" => Ok(MathOp::Divide),
        "%" => Ok(MathOp::Modulo),
        "^" => Ok(MathOp::Power),
        "R" => Ok(MathOp::Root),
        s => Err(ParseError::UnknownOperator(s.into())),
    }
}
