//! RUNE Expression Evaluator
//!
//! Evaluates RUNE expressions with semantic prefixes, array literals, and operators.
//! Supports mathematical operations, semantic type checking, and E8 geometry primitives.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use super::values::{EvalContext, EvalError, Value};
use crate::rune::ops::MathOp;
use crate::rune::{
    Expr, Ident, Literal, MathAtom, MathExpr, MathUnaryOp, RuneOp, SemanticIdent, Stmt, Term,
};
use std::collections::HashMap;

/// RUNE expression evaluator with semantic type support
pub struct Evaluator {
    /// Variable bindings (name -> value)
    variables: HashMap<String, Value>,
    /// Semantic namespace bindings (T:name -> value)
    semantic_vars: HashMap<String, Value>,
}

impl Evaluator {
    /// Create a new evaluator with empty context
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            semantic_vars: HashMap::new(),
        }
    }

    /// Create evaluator with pre-populated context
    pub fn with_context(ctx: EvalContext) -> Self {
        Self {
            variables: ctx.variables,
            semantic_vars: ctx.semantic_vars,
        }
    }

    /// Set a variable value
    pub fn set_var(&mut self, name: impl Into<String>, value: Value) {
        self.variables.insert(name.into(), value);
    }

    /// Set a semantic variable value (e.g., T:Gf8)
    pub fn set_semantic(&mut self, prefix: char, name: impl Into<String>, value: Value) {
        let key = format!("{}:{}", prefix, name.into());
        self.semantic_vars.insert(key, value);
    }

    /// Print SIMD capabilities (diagnostic)
    #[cfg(feature = "simd")]
    pub fn print_simd_info(&self) {
        use super::values::{get_available_f32_256_intrinsics, print_simd_capabilities};
        print_simd_capabilities();
        let intrinsics = get_available_f32_256_intrinsics();
        println!("Available f32x256 intrinsics: {:?}", intrinsics);
    }

    /// Get a variable value
    pub fn get_var(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }

    /// Get a semantic variable value
    pub fn get_semantic(&self, prefix: char, name: &str) -> Option<&Value> {
        let key = format!("{}:{}", prefix, name);
        self.semantic_vars.get(&key)
    }

    /// Evaluate a statement
    pub fn eval_stmt(&mut self, stmt: &Stmt) -> Result<Value, EvalError> {
        match stmt {
            Stmt::RootDecl(root) => {
                // Root declarations don't produce values, but we can store them as context
                Ok(Value::String(root.to_string()))
            }
            Stmt::ToonBlock { name, content } => {
                // TOON blocks are data, not computation - return the raw content
                Ok(Value::String(format!(
                    "TOON block '{}': {} chars",
                    name,
                    content.len()
                )))
            }
            Stmt::Expr(expr) => self.eval_expr(expr),
        }
    }

    /// Evaluate an expression
    pub fn eval_expr(&mut self, expr: &Expr) -> Result<Value, EvalError> {
        match expr {
            Expr::Term(term) => self.eval_term(term),
            Expr::Binary { left, op, right } => {
                let left_val = self.eval_expr(left)?;
                let right_val = self.eval_expr(right)?;
                self.eval_binary_op(&left_val, op, &right_val)
            }
        }
    }

    /// Evaluate a term
    fn eval_term(&self, term: &Term) -> Result<Value, EvalError> {
        match term {
            Term::Literal(lit) => self.eval_literal(lit),
            Term::Ident(ident) => self.eval_ident(ident),
            Term::SemanticIdent(sem) => self.eval_semantic_ident(sem),
            Term::Group(expr) => {
                // Group expressions are used for math blocks [expr]
                // For now, just evaluate the inner expression
                let mut temp_eval = Self {
                    variables: self.variables.clone(),
                    semantic_vars: self.semantic_vars.clone(),
                };
                temp_eval.eval_expr(expr)
            }
            Term::Math(math_expr) => {
                // Math blocks contain MathExpr which needs evaluation
                self.eval_math_expr(math_expr)
            }
        }
    }

    /// Evaluate a math expression
    fn eval_math_expr(&self, math: &MathExpr) -> Result<Value, EvalError> {
        match math {
            MathExpr::Atom(atom) => self.eval_math_atom(atom),
            MathExpr::Binary { left, op, right } => {
                let left_val = self.eval_math_expr(left)?;
                let right_val = self.eval_math_expr(right)?;
                self.eval_math_op(&left_val, op, &right_val)
            }
            MathExpr::Unary { op, operand } => {
                let val = self.eval_math_expr(operand)?;
                self.eval_math_unary_op(op, &val)
            }
        }
    }

    /// Evaluate a math atom
    fn eval_math_atom(&self, atom: &MathAtom) -> Result<Value, EvalError> {
        match atom {
            MathAtom::Number(n) => Ok(Value::Float(*n)),
            MathAtom::Ident(ident) => {
                // Check if it's a semantic identifier (contains ':')
                if ident.0.contains(':') {
                    let parts: Vec<&str> = ident.0.split(':').collect();
                    if parts.len() == 2 && parts[0].len() == 1 {
                        let prefix = parts[0].chars().next().unwrap();
                        self.get_semantic(prefix, parts[1])
                            .cloned()
                            .ok_or_else(|| EvalError::UndefinedVariable(ident.0.clone()))
                    } else {
                        self.eval_ident(ident)
                    }
                } else {
                    self.eval_ident(ident)
                }
            }
            MathAtom::Group(math) => self.eval_math_expr(math),
            MathAtom::Array(elements) => {
                // Evaluate array literal inside math block
                let mut values = Vec::new();
                for elem in elements {
                    values.push(self.eval_math_expr(elem)?);
                }
                Ok(Value::Array(values))
            }
        }
    }

    /// Evaluate a math binary operation
    fn eval_math_op(&self, left: &Value, op: &MathOp, right: &Value) -> Result<Value, EvalError> {
        match op {
            MathOp::Add => left.add(right),
            MathOp::Subtract => left.sub(right),
            MathOp::Multiply => left.mul(right),
            MathOp::Divide => left.div(right),
            MathOp::Power => left.pow(right),
            MathOp::Modulo => left.modulo(right),
            MathOp::Root => Err(EvalError::UnsupportedOperation(
                "Root operator not yet implemented".into(),
            )),
        }
    }

    /// Evaluate a math unary operation
    fn eval_math_unary_op(&self, op: &MathUnaryOp, val: &Value) -> Result<Value, EvalError> {
        match op {
            MathUnaryOp::Negate => val.negate(),
            MathUnaryOp::Plus => Ok(val.clone()),
        }
    }

    /// Evaluate a literal value
    fn eval_literal(&self, lit: &Literal) -> Result<Value, EvalError> {
        match lit {
            Literal::Number(n) => Ok(Value::Float(*n)),
            Literal::String(s) => Ok(Value::String(s.clone())),
            Literal::Array(exprs) => {
                let mut values = Vec::new();
                let mut temp_eval = Self {
                    variables: self.variables.clone(),
                    semantic_vars: self.semantic_vars.clone(),
                };
                for expr in exprs {
                    values.push(temp_eval.eval_expr(expr)?);
                }
                Ok(Value::Array(values))
            }
        }
    }

    /// Evaluate an identifier (variable lookup)
    fn eval_ident(&self, ident: &Ident) -> Result<Value, EvalError> {
        self.variables
            .get(&ident.0)
            .cloned()
            .ok_or_else(|| EvalError::UndefinedVariable(ident.0.clone()))
    }

    /// Evaluate a semantic identifier (T:name, V:velocity, etc.)
    fn eval_semantic_ident(&self, sem: &SemanticIdent) -> Result<Value, EvalError> {
        let key = format!("{}:{}", sem.prefix, sem.name.0);
        self.semantic_vars
            .get(&key)
            .cloned()
            .ok_or_else(|| EvalError::UndefinedVariable(key))
    }

    /// Evaluate a binary operation using RuneOp (structural operations only)
    /// Arithmetic operations are handled by MathOp within math blocks `[]`
    fn eval_binary_op(&self, left: &Value, op: &RuneOp, right: &Value) -> Result<Value, EvalError> {
        use RuneOp::*;

        match op {
            // Comparison operators
            Less => left.lt(right),
            LessEqual => left.le(right),
            Greater => left.gt(right),
            GreaterEqual => left.ge(right),
            Equal => Ok(Value::Bool(left == right)),

            // Structural operators (not for computation) - arithmetic handled by MathOp
            Descendant | Ancestor | Define | FlowRight | FlowLeft | Bind | Namespace | Alias
            | Parallel | Transform | SplitJoin | JoinSplit | AnchorDescend | BranchStabilize
            | RootStabilize | StabilizeRoot | SymmetricSplit | BranchAnchorBranch => {
                Err(EvalError::UnsupportedOperation(format!(
                    "Structural operator {:?} not implemented for computation. Use math blocks `[]` for arithmetic.",
                    op
                )))
            }
        }
    }

    /// Export current context
    pub fn context(&self) -> EvalContext {
        let mut ctx = EvalContext::new();
        for (name, value) in &self.variables {
            ctx.bind(name.clone(), value.clone());
        }
        // Note: semantic_vars would need to be stored differently in EvalContext
        ctx
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rune::parse;

    #[test]
    fn test_eval_literal_number() {
        let mut eval = Evaluator::new();
        let stmts = parse("42").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(42.0));
    }

    #[test]
    fn test_eval_arithmetic() {
        let mut eval = Evaluator::new();

        // Simple addition in math block
        let stmts = parse("[2 + 3]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(5.0));

        // Multiplication in math block
        let stmts = parse("[4 * 5]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(20.0));

        // Complex expression with precedence
        let stmts = parse("[2 + 3 * 4]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(14.0)); // Respects precedence

        // Division in math block
        let stmts = parse("[10 / 2]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(5.0));

        // Power in math block
        let stmts = parse("[2 ^ 3]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(8.0));

        // Modulo in math block
        let stmts = parse("[10 % 3]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(1.0));
    }

    #[test]
    fn test_eval_array_literal() {
        let mut eval = Evaluator::new();
        let stmts = parse("[1, 2, 3]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();

        match result {
            Value::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0], Value::Float(1.0));
                assert_eq!(arr[1], Value::Float(2.0));
                assert_eq!(arr[2], Value::Float(3.0));
            }
            _ => panic!("Expected array value"),
        }
    }

    #[test]
    fn test_eval_array_operations() {
        let mut eval = Evaluator::new();

        // Array addition (element-wise) in math block
        let stmts = parse("[[1, 2, 3] + [4, 5, 6]]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();

        match result {
            Value::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0], Value::Float(5.0));
                assert_eq!(arr[1], Value::Float(7.0));
                assert_eq!(arr[2], Value::Float(9.0));
            }
            _ => panic!("Expected array value"),
        }
    }

    #[test]
    fn test_eval_semantic_prefix() {
        let mut eval = Evaluator::new();

        // Set semantic variable
        eval.set_semantic('T', "Gf8", Value::Float(2.5));

        // Evaluate semantic expression in math block
        let stmts = parse("[T:Gf8 * 3]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(7.5));
    }

    #[test]
    fn test_eval_variables() {
        let mut eval = Evaluator::new();

        // Set variable
        eval.set_var("x", Value::Float(10.0));

        // Use in expression within math block
        let stmts = parse("[x + 5]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(15.0));
    }

    #[test]
    fn test_eval_nested_math() {
        let mut eval = Evaluator::new();

        // Math block with nested operations
        let stmts = parse("[[3, 3, 3] * [2, 2, 2]]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();

        match result {
            Value::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0], Value::Float(6.0));
                assert_eq!(arr[1], Value::Float(6.0));
                assert_eq!(arr[2], Value::Float(6.0));
            }
            _ => panic!("Expected array value"),
        }
    }

    #[test]
    fn test_eval_comparison() {
        let mut eval = Evaluator::new();

        // Comparisons work with RuneOp outside math blocks
        let stmts = parse("5 > 3").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(true));

        let stmts = parse("2 = 2").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_eval_unary_minus() {
        let mut eval = Evaluator::new();

        // Unary minus in math block
        let stmts = parse("[-5]").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Float(-5.0));
    }

    #[test]
    fn test_eval_comparison_operators() {
        let mut eval = Evaluator::new();

        // Test less than or equal
        let stmts = parse("3 <= 5").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(true));

        let stmts = parse("5 <= 5").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(true));

        let stmts = parse("7 <= 5").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(false));

        // Test greater than or equal
        let stmts = parse("5 >= 3").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(true));

        let stmts = parse("5 >= 5").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(true));

        let stmts = parse("3 >= 5").unwrap();
        let result = eval.eval_stmt(&stmts[0]).unwrap();
        assert_eq!(result, Value::Bool(false));
    }
}
