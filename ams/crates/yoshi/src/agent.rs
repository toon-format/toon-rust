/* yoshi/src/agent.rs */
//! Geoshi-driven refactor loop for Yoshi host processes.
//! Parses compiler errors, asks Xage for an action, and routes refactor
//! intent through the geometric actuator into concrete source edits.

#![cfg(feature = "autofix")]

use crate::error::YoError;
use geoshi::actuator::GeometricActuator;
use geoshi::xage::{CognitiveAction, Environment, Xage};
use rand::rng;
use std::fs;
use yoshi_std::corrector::actuator_bridge::actuator_with_std_tools;
use yoshi_std::YoshiError;

/// Run a refactor cycle driven by Geoshi/Xage against the first actionable file in the error list.
///
/// Returns `Ok(true)` if a change was applied, `Ok(false)` otherwise.
pub fn apply_refactor_from_errors(errors: &[String]) -> Result<bool, YoError> {
    if errors.is_empty() {
        return Ok(false);
    }

    let mut xage = Xage::new("yoshi-refactor".to_string(), vec![0.0, 0.0]);
    let env = Environment::new(2, 3);
    let mut rng = rng();

    let action = xage
        .cognitive_cycle_with_errors(&env, &mut rng, errors)
        .map_err(|e| YoError::from(YoshiError::foreign(e)))?;

    match action {
        CognitiveAction::Refactor {
            target_file,
            strategy_xuid,
        } => {
            let mut content = match fs::read_to_string(&target_file) {
                Ok(c) => c,
                Err(_) => return Ok(false),
            };

            let actuator: GeometricActuator = actuator_with_std_tools();
            let changed = xage
                .apply_refactor_action(
                    &CognitiveAction::Refactor {
                        target_file: target_file.clone(),
                        strategy_xuid,
                    },
                    &actuator,
                    &mut content,
                    errors,
                )
                .map_err(|e| YoError::from(YoshiError::foreign(e)))?;

            if changed {
                fs::write(&target_file, content).map_err(YoshiError::foreign)?;
            }

            Ok(changed)
        }
        _ => Ok(false),
    }
}
