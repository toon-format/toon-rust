/* src/actuator.rs */
//! Bridges geometric intent (XUID-tagged strategies) to executable code edits.
//!
//! This module acts as the "motor cortex" for Geoshi: it looks up a
//! registered strategy by XUID and executes the corresponding tooling
//! (e.g., micro-refactorings) against a mutable code buffer.

use crate::GsaResult;
use std::collections::HashMap;
use xuid::Xuid;

/// Trait representing an executable code-change tool.
///
/// Implementations can be as simple as a regex-based fixer or as
/// sophisticated as an AST-driven refactoring. The actuator only
/// requires a name for logging and an `apply` method that returns the
/// modified code buffer.
pub trait ActuationTool: Send + Sync {
    /// Human-friendly tool name for telemetry/logging.
    fn name(&self) -> &str;

    /// Whether this tool considers itself relevant for the current context.
    fn can_handle(&self, _content: &str, _errors: &[String]) -> bool {
        true
    }

    /// Apply the tool to the provided source, returning a new buffer.
    fn apply(&self, content: &str, errors: &[String]) -> GsaResult<String>;
}

/// Registry plus executor that maps strategy XUIDs to concrete tools.
pub struct GeometricActuator {
    tools: HashMap<Xuid, Box<dyn ActuationTool>>,
}

impl Default for GeometricActuator {
    fn default() -> Self {
        Self::new()
    }
}

impl GeometricActuator {
    /// Create a new actuator with an empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool under a stable strategy identifier.
    pub fn register_tool(&mut self, id: &Xuid, tool: Box<dyn ActuationTool>) {
        self.tools.insert(id.clone(), tool);
    }

    /// Execute a learned/selected strategy against the target buffer.
    ///
    /// Returns `Ok(true)` if the buffer was modified, `Ok(false)` if
    /// no change was made (unknown strategy, inapplicable tool, or
    /// applied tool produced identical content).
    pub fn execute_strategy(
        &self,
        strategy_id: &Xuid,
        target_file_content: &mut String,
        errors: &[String],
    ) -> GsaResult<bool> {
        if let Some(tool) = self.tools.get(strategy_id) {
            if !tool.can_handle(target_file_content, errors) {
                return Ok(false);
            }

            let new_content = tool.apply(target_file_content, errors)?;
            if *target_file_content != new_content {
                *target_file_content = new_content;
                return Ok(true);
            }
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct UppercaseTool;
    impl ActuationTool for UppercaseTool {
        fn name(&self) -> &str {
            "UppercaseTool"
        }
        fn apply(&self, content: &str, _errors: &[String]) -> GsaResult<String> {
            Ok(content.to_uppercase())
        }
    }

    #[test]
    fn actuator_executes_registered_tool() {
        let mut actuator = GeometricActuator::new();
        let strategy_id = xuid::from_path("/strategy/test/uppercase", xuid::XuidType::Experience);
        actuator.register_tool(&strategy_id, Box::new(UppercaseTool));

        let mut content = "hello".to_string();
        let changed = actuator
            .execute_strategy(&strategy_id, &mut content, &[])
            .unwrap();

        assert!(changed);
        assert_eq!(content, "HELLO");
    }
}
