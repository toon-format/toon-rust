//! High-level cognitive geometry for adaptive agents with memory, learning, and goal-directed control.
//!
//! # Geoshi – XAGE Module
//!▫~•◦------------------------------------------------‣
//!
//! This module integrates the XAGE (Extended Adaptive Geometric Entity) cognitive stack into the
//! GeoSynth Agent (Geoshi) runtime, providing memory, learning, perception, and goal management that
//! work together to generate context-aware actions and recovery behaviors.
//!
//! ### Key Capabilities
//! - **Goal-directed cognition:** Maintains and prioritizes layered goals, updating them from curiosity,
//!   prediction error, and episodic recall.
//! - **Memory-driven navigation:** Blends episodic, semantic, and working memory to bias exploration and
//!   obstacle-aware motion.
//! - **Adaptive learning:** Uses learned patterns to optimize movement and communication, feeding back
//!   into perception for better prediction fidelity.
//!
//! ### Architectural Notes
//! This module is designed to work with other Geoshi primitives such as `GoalSystem`, `LearningSystem`,
//! and `PerceptionSystem`. Result structures adhere to the `GsaResult<T>` alias and integrate with the
//! surrounding GeoSynth serialization pipeline. The agent behavior composes with higher-level planners
//! through the `CognitiveAction` interface.
//!
//! ### Example
//! ```rust
//! use geoshi::xage::{Environment, Goal, GoalType, Xage};
//! use geoshi::GsaResult;
//! use rand::rng;
//!
//! fn main() -> GsaResult<()> {
//!     let mut agent = Xage::new("agent-0".to_string(), vec![0.0, 0.0]);
//!     let environment = Environment::new(2, 3);
//!
//!     let goal = Goal {
//!         id: "navigate-home".into(),
//!         goal_type: GoalType::Navigate,
//!         priority: 0.8,
//!         target_position: Some(vec![1.0, 1.0]),
//!         deadline: None,
//!     };
//!
//!     let mut rng = rng();
//!     let _ = agent.cognitive_cycle(&environment, &mut rng)?;
//!     let _ = agent.take_action(&goal, &environment)?;
//!     Ok(())
//! }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::{GsaResult, actuator::GeometricActuator};
use rand::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use xuid::{Xuid, XuidType};

/// Extended geometric agent with cognitive capabilities
pub struct Xage {
    pub id: String,
    pub position: Vec<f64>,
    pub cognitive_state: CognitiveState,
    pub memory_system: MemorySystem,
    pub goal_system: GoalSystem,
    pub learning_system: LearningSystem,
    pub perception_system: PerceptionSystem,
}

impl Xage {
    /// Create a new XAGE agent
    pub fn new(id: String, position: Vec<f64>) -> Self {
        Self {
            id,
            position,
            cognitive_state: CognitiveState::default(),
            memory_system: MemorySystem::new(),
            goal_system: GoalSystem::new(),
            learning_system: LearningSystem::new(),
            perception_system: PerceptionSystem::new(),
        }
    }

    /// Process a single cognitive cycle
    pub fn cognitive_cycle(
        &mut self,
        environment: &Environment,
        rng: &mut impl Rng,
    ) -> GsaResult<CognitiveAction> {
        self.cognitive_cycle_with_errors(environment, rng, &[])
    }

    /// Process a single cognitive cycle with error context (used for refactor actions)
    pub fn cognitive_cycle_with_errors(
        &mut self,
        environment: &Environment,
        rng: &mut impl Rng,
        errors: &[String],
    ) -> GsaResult<CognitiveAction> {
        // Update perception
        self.perception_system
            .update_perception(environment, &self.position);

        // Update cognitive state
        self.cognitive_state
            .update(&self.perception_system, &self.memory_system);

        // Process goals
        self.goal_system
            .update_goals(&self.cognitive_state, &self.memory_system);

        // Learn from experience
        self.learning_system
            .learn(&self.cognitive_state, &self.perception_system);

        // Store experiences in memory
        self.memory_system
            .store_experience(&self.cognitive_state, &self.perception_system);

        // Generate action based on goals and cognition
        self.generate_action(environment, rng, errors)
    }

    /// Generate goal-directed action
    fn generate_action(
        &self,
        environment: &Environment,
        rng: &mut impl Rng,
        errors: &[String],
    ) -> GsaResult<CognitiveAction> {
        let active_goals = self.goal_system.get_active_goals();

        if active_goals.is_empty() {
            // Exploration behavior
            return Ok(CognitiveAction::Explore(
                self.generate_exploration_target(rng),
            ));
        }

        // Select primary goal
        let primary_goal = &active_goals[0];

        match primary_goal.goal_type {
            GoalType::Navigate => Ok(self.generate_navigation_action(primary_goal, environment)),
            GoalType::Learn => Ok(self.generate_learning_action(primary_goal)),
            GoalType::Optimize => {
                Ok(self.generate_optimization_action(primary_goal, environment, errors))
            }
            GoalType::Communicate => Ok(self.generate_communication_action(primary_goal)),
        }
    }

    fn generate_navigation_action(
        &self,
        goal: &Goal,
        environment: &Environment,
    ) -> CognitiveAction {
        if let Some(target) = &goal.target_position {
            let direct_direction = self.calculate_direction_to(target);
            let obstacle_penalty = environment.calculate_obstacle_penalty(&self.position, target);

            // Learn-informed correction blended with direct goal vector
            let learned_direction = self.learning_system.predict_optimal_direction(
                &self.position,
                &self.cognitive_state,
                environment,
            );
            let blended_direction = self.blend_direction(
                &learned_direction,
                &direct_direction,
                (1.0 - obstacle_penalty) * self.cognitive_state.confidence_level,
            );

            let base_speed = 0.1_f64 * goal.priority.clamp(0.0, 1.0); // Use goal priority to scale speed
            let adjusted_speed = (base_speed * (1.0_f64 - obstacle_penalty)).max(0.01_f64);
            CognitiveAction::Move(blended_direction, adjusted_speed)
        } else {
            CognitiveAction::Wait
        }
    }

    fn generate_learning_action(&self, goal: &Goal) -> CognitiveAction {
        // Generate learning action based on goal priority and type
        match goal.goal_type {
            GoalType::Learn => {
                // Higher priority goals trigger more intensive observation
                if goal.priority > 0.7 {
                    CognitiveAction::Observe
                } else {
                    // Lower priority learning can involve exploration
                    CognitiveAction::Explore(vec![0.0, 0.0]) // Explore around current position
                }
            }
            _ => CognitiveAction::Observe, // Default to observation for other goal types
        }
    }

    fn generate_optimization_action(
        &self,
        goal: &Goal,
        environment: &Environment,
        errors: &[String],
    ) -> CognitiveAction {
        // If cognitive tension is high, emit a refactor intent targeting the primary source file.
        if self.cognitive_state.prediction_error > 0.8 {
            let target_file = self
                .select_target_from_errors(errors)
                .unwrap_or_else(|| "src/main.rs".to_string());
            return CognitiveAction::Refactor {
                target_file,
                strategy_xuid: xuid::from_path("/strategy/std/LifetimeFixer", XuidType::Experience),
            };
        }

        // Use cognitive biases to find optimal paths
        let learned_direction = self.learning_system.predict_optimal_direction(
            &self.position,
            &self.cognitive_state,
            environment,
        );

        let goal_direction = goal
            .target_position
            .as_ref()
            .map(|target| self.calculate_direction_to(target));

        let confidence_weight = self.cognitive_state.confidence_level.clamp(0.0, 1.0);
        let blended_direction = if let Some(goal_direction) = goal_direction {
            self.blend_direction(&learned_direction, &goal_direction, confidence_weight)
        } else {
            self.normalize_vector(learned_direction)
        };

        let penalty = goal.target_position.as_ref().map_or_else(
            || {
                let fallback_target = vec![
                    self.position.first().copied().unwrap_or_default()
                        + blended_direction.first().copied().unwrap_or_default(),
                    self.position.get(1).copied().unwrap_or_default()
                        + blended_direction.get(1).copied().unwrap_or_default(),
                ];
                environment.calculate_obstacle_penalty(&self.position, &fallback_target)
            },
            |target| environment.calculate_obstacle_penalty(&self.position, target),
        );

        let base_speed = 0.1 + goal.priority.clamp(0.0, 1.0) * 0.15;
        let adjusted_speed = (base_speed * (1.0 - penalty)).max(0.02);

        CognitiveAction::Move(blended_direction, adjusted_speed)
    }

    fn generate_communication_action(&self, goal: &Goal) -> CognitiveAction {
        CognitiveAction::Signal(goal.priority)
    }

    fn generate_exploration_target(&self, rng: &mut impl Rng) -> Vec<f64> {
        // Generate exploration targets based on memory and uncertainty
        let uncertainty_map = self.memory_system.get_uncertainty_map();
        let unexplored_regions = self.find_unexplored_regions(&uncertainty_map);

        if let Some(region) = unexplored_regions.choose(rng) {
            region.clone()
        } else {
            // Fallback to random exploration
            vec![rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)]
        }
    }

    /// Crude heuristic to pick a target file from compiler error messages.
    fn select_target_from_errors(&self, errors: &[String]) -> Option<String> {
        for line in errors {
            // Rust errors often include " --> path:line:col"
            if let Some(idx) = line.find("-->") {
                let candidate = line[idx + 3..].trim();
                if let Some(path) = Self::extract_path_token(candidate) {
                    return Some(path);
                }
            } else if let Some(path) = Self::extract_path_token(line) {
                return Some(path);
            }
        }
        None
    }

    fn extract_path_token(candidate: &str) -> Option<String> {
        for token in candidate.split_whitespace() {
            if let Some(pos) = token.find(".rs") {
                let end = pos + 3;
                let slice = &token[..end.min(token.len())];
                // Strip trailing punctuation like ':' or ','
                let cleaned = slice.trim_end_matches(&[':', ',', ')', '"'][..]);
                return Some(cleaned.to_string());
            }
        }
        None
    }

    fn find_unexplored_regions(&self, uncertainty_map: &HashMap<String, f64>) -> Vec<Vec<f64>> {
        // Simplified: return positions with high uncertainty
        uncertainty_map
            .iter()
            .filter(|(_, uncertainty)| **uncertainty > 0.7)
            .map(|(pos_key, _)| self.key_to_position(pos_key))
            .collect()
    }

    fn key_to_position(&self, key: &str) -> Vec<f64> {
        key.split(',')
            .filter_map(|s| s.parse::<f64>().ok())
            .collect()
    }

    fn calculate_direction_to(&self, target: &[f64]) -> Vec<f64> {
        let direction: Vec<f64> = self
            .position
            .iter()
            .zip(target)
            .map(|(current, target)| target - current)
            .collect();

        self.normalize_vector(direction)
    }

    fn blend_direction(&self, learned: &[f64], goal: &[f64], confidence_weight: f64) -> Vec<f64> {
        let learned_weight = confidence_weight.clamp(0.0, 1.0);
        let goal_weight = 1.0 - learned_weight;

        let combined: Vec<f64> = learned
            .iter()
            .zip(goal.iter())
            .map(|(l, g)| l * learned_weight + g * goal_weight)
            .collect();

        self.normalize_vector(combined)
    }

    fn normalize_vector(&self, mut direction: Vec<f64>) -> Vec<f64> {
        let magnitude = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
        if magnitude > 0.0 {
            direction.iter_mut().for_each(|x| *x /= magnitude);
        }
        direction
    }

    /// Take cognitive action based on a specific goal using ML-driven recovery
    pub fn take_action(
        &self,
        goal: &Goal,
        environment: &Environment,
    ) -> GsaResult<CognitiveAction> {
        // Geoshi acts as the ML engine for recovery, using cognitive systems
        // to intelligently decide recovery actions based on goal and environment

        match goal.goal_type {
            GoalType::Navigate => {
                // Use goal target position for navigation recovery
                if let Some(target) = &goal.target_position {
                    let direct_direction = self.calculate_direction_to(target);
                    let penalty = environment.calculate_obstacle_penalty(&self.position, target);

                    // Cognitive bias: adjust direction based on learned patterns and confidence
                    let adjusted_direction = self.learning_system.predict_optimal_direction(
                        &self.position,
                        &self.cognitive_state,
                        environment,
                    );

                    let blended_direction = self.blend_direction(
                        &adjusted_direction,
                        &direct_direction,
                        (1.0 - penalty) * self.cognitive_state.confidence_level,
                    );

                    let speed = ((1.0 - penalty) * goal.priority * 0.2).max(0.01);
                    Ok(CognitiveAction::Move(blended_direction, speed))
                } else {
                    // Cognitive exploration: use memory-based exploration
                    let uncertainty_map = self.memory_system.get_uncertainty_map();
                    let unexplored_regions = self.find_unexplored_regions(&uncertainty_map);

                    if let Some(region) = unexplored_regions.into_iter().next() {
                        Ok(CognitiveAction::Explore(region))
                    } else {
                        Ok(CognitiveAction::Explore(vec![0.0, 0.0]))
                    }
                }
            }

            GoalType::Learn => {
                // ML-driven learning action: observe environment with high priority
                if goal.priority > 0.7 {
                    Ok(CognitiveAction::Observe)
                } else if self.cognitive_state.curiosity_level > 0.6 {
                    Ok(CognitiveAction::Explore(vec![0.0, 0.0]))
                } else {
                    Ok(CognitiveAction::Observe)
                }
            }

            GoalType::Optimize => {
                // Geometric synthesis recovery: optimize path using cognitive systems
                let current_position = &self.position;
                let optimal_direction = self.learning_system.predict_optimal_direction(
                    current_position,
                    &self.cognitive_state,
                    environment,
                );

                // Cognitive decision: speed based on confidence level
                let speed = self.cognitive_state.confidence_level * goal.priority * 0.15;
                Ok(CognitiveAction::Move(optimal_direction, speed))
            }

            GoalType::Communicate => {
                // Use cognitive state to determine communication priority
                let signal_strength =
                    (self.cognitive_state.emotional_state as u8 as f64 / 5.0) * goal.priority;
                Ok(CognitiveAction::Signal(signal_strength))
            }
        }
    }

    /// Route a refactor action through the geometric actuator to mutate source content.
    pub fn apply_refactor_action(
        &self,
        action: &CognitiveAction,
        actuator: &GeometricActuator,
        target_file_content: &mut String,
        errors: &[String],
    ) -> GsaResult<bool> {
        match action {
            CognitiveAction::Refactor { strategy_xuid, .. } => actuator
                .execute_strategy(strategy_xuid, target_file_content, errors),
            _ => Ok(false),
        }
    }
}

/// Cognitive state representation
#[derive(Debug, Clone)]
pub struct CognitiveState {
    pub attention_focus: Vec<f64>,
    pub emotional_state: EmotionalState,
    pub confidence_level: f64,
    pub curiosity_level: f64,
    pub prediction_error: f64,
}

impl Default for CognitiveState {
    fn default() -> Self {
        Self {
            attention_focus: vec![0.0, 0.0],
            emotional_state: EmotionalState::Neutral,
            confidence_level: 0.5,
            curiosity_level: 0.7,
            prediction_error: 0.0,
        }
    }
}

impl CognitiveState {
    /// Update cognitive state based on perception and memory
    pub fn update(&mut self, perception: &PerceptionSystem, memory: &MemorySystem) {
        // Update attention based on salient features
        self.attention_focus = perception.get_salient_position();

        // Update emotional state based on prediction error
        let predicted_state = memory.predict_state(perception);
        self.prediction_error = perception.calculate_prediction_error(&predicted_state);
        self.emotional_state = self.emotion_from_prediction_error(self.prediction_error);

        // Update confidence based on prediction accuracy
        if self.prediction_error < 0.1 {
            self.confidence_level = (self.confidence_level + 0.05).min(1.0);
        } else {
            self.confidence_level = (self.confidence_level - 0.05).max(0.0);
        }

        // Update curiosity based on novelty
        let novelty = perception.calculate_novelty(memory);
        self.curiosity_level = (novelty * 0.3 + self.curiosity_level * 0.7).clamp(0.0, 1.0);
    }

    fn emotion_from_prediction_error(&self, error: f64) -> EmotionalState {
        match error {
            e if e < 0.1 => EmotionalState::Satisfied,
            e if e < 0.3 => EmotionalState::Neutral,
            e if e < 0.6 => EmotionalState::Surprised,
            _ => EmotionalState::Confused,
        }
    }
}

/// Emotional states for cognitive processing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmotionalState {
    Satisfied,
    Neutral,
    Surprised,
    Confused,
    Curious,
}

/// Memory system with multiple memory types
pub struct MemorySystem {
    pub episodic_memory: VecDeque<EpisodicMemory>,
    pub semantic_memory: HashMap<String, SemanticConcept>,
    pub working_memory: WorkingMemory,
    pub max_episodic_memories: usize,
}

impl Default for MemorySystem {
    fn default() -> Self {
        Self::new()
    }
}

impl MemorySystem {
    /// Create a new memory system
    pub fn new() -> Self {
        Self {
            episodic_memory: VecDeque::new(),
            semantic_memory: HashMap::new(),
            working_memory: WorkingMemory::new(),
            max_episodic_memories: 1000,
        }
    }

    /// Store an experience in episodic memory
    pub fn store_experience(&mut self, state: &CognitiveState, perception: &PerceptionSystem) {
        let experience = EpisodicMemory {
            timestamp: std::time::SystemTime::now(),
            position: perception.current_position.clone(),
            cognitive_state: state.clone(),
            sensory_input: perception.sensory_buffer.clone(),
            action_taken: None, // Would be set by action system
            outcome: None,      // Would be set by consequence evaluation
        };

        // Update semantic memory first
        self.update_semantic_memory(&experience);

        self.episodic_memory.push_back(experience);

        // Maintain memory limit
        while self.episodic_memory.len() > self.max_episodic_memories {
            self.episodic_memory.pop_front();
        }
    }

    /// Predict state based on current perception
    pub fn predict_state(&self, perception: &PerceptionSystem) -> CognitiveState {
        // Simplified prediction based on similar past experiences
        let similar_experiences = self.find_similar_experiences(perception);

        if similar_experiences.is_empty() {
            return CognitiveState::default();
        }

        // Average predictions from similar experiences
        let mut predicted_state = CognitiveState::default();
        let weight_sum: f64 = similar_experiences.iter().map(|(_, weight)| weight).sum();

        for (experience, weight) in similar_experiences {
            let w = weight / weight_sum;
            predicted_state.confidence_level += experience.cognitive_state.confidence_level * w;
            predicted_state.curiosity_level += experience.cognitive_state.curiosity_level * w;
        }

        predicted_state
    }

    /// Get uncertainty map for exploration
    pub fn get_uncertainty_map(&self) -> HashMap<String, f64> {
        let mut uncertainty = HashMap::new();

        // Calculate uncertainty based on memory coverage
        for experience in &self.episodic_memory {
            let key = self.position_to_key(&experience.position);
            let count = uncertainty.entry(key).or_insert(0.0);
            *count += 1.0;
        }

        // Invert counts to get uncertainty (less visited = more uncertain)
        let max_visits = uncertainty.values().copied().fold(0.0, f64::max);
        for value in uncertainty.values_mut() {
            *value = 1.0 - (*value / max_visits).min(1.0);
        }

        uncertainty
    }

    fn find_similar_experiences(
        &self,
        perception: &PerceptionSystem,
    ) -> Vec<(EpisodicMemory, f64)> {
        self.episodic_memory
            .iter()
            .map(|exp| {
                let similarity =
                    self.calculate_similarity(&exp.sensory_input, &perception.sensory_buffer);
                (exp.clone(), similarity)
            })
            .filter(|(_, sim)| *sim > 0.5)
            .collect()
    }

    fn calculate_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }

        let diff_sum: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        let similarity = 1.0 / (1.0 + diff_sum.sqrt());
        similarity.clamp(0.0, 1.0)
    }

    fn update_semantic_memory(&mut self, experience: &EpisodicMemory) {
        // Simplified semantic learning
        let concept_key = format!(
            "position_{:.1}_{:.1}",
            experience.position[0], experience.position[1]
        );

        let concept = self
            .semantic_memory
            .entry(concept_key.clone())
            .or_insert_with(|| SemanticConcept {
                name: concept_key,
                features: HashMap::new(),
                activation: 0.0,
            });

        concept.activation += 0.1;

        // Add features based on sensory input
        for (i, &value) in experience.sensory_input.iter().enumerate() {
            let feature_key = format!("feature_{i}");
            let feature_value = concept.features.entry(feature_key).or_insert(0.0);
            *feature_value = *feature_value * 0.9 + value * 0.1; // Exponential moving average
        }
    }

    fn position_to_key(&self, position: &[f64]) -> String {
        position
            .iter()
            .map(|&x| format!("{x:.1}"))
            .collect::<Vec<_>>()
            .join("_")
    }
}

/// Episodic memory entry
#[derive(Debug, Clone)]
pub struct EpisodicMemory {
    pub timestamp: std::time::SystemTime,
    pub position: Vec<f64>,
    pub cognitive_state: CognitiveState,
    pub sensory_input: Vec<f64>,
    pub action_taken: Option<CognitiveAction>,
    pub outcome: Option<f64>,
}

/// Semantic concept representation
#[derive(Debug, Clone)]
pub struct SemanticConcept {
    pub name: String,
    pub features: HashMap<String, f64>,
    pub activation: f64,
}

/// Working memory for active processing
#[derive(Debug, Clone)]
pub struct WorkingMemory {
    pub active_items: Vec<MemoryItem>,
    pub capacity: usize,
}

impl Default for WorkingMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkingMemory {
    pub fn new() -> Self {
        Self {
            active_items: Vec::new(),
            capacity: 7, // Magical number 7 ± 2
        }
    }

    pub fn add_item(&mut self, item: MemoryItem) {
        self.active_items.push(item);

        // Maintain capacity
        if self.active_items.len() > self.capacity {
            self.active_items.remove(0); // Remove oldest
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryItem {
    pub content: String,
    pub activation: f64,
    pub timestamp: std::time::SystemTime,
}

/// Goal system for goal-directed behavior
pub struct GoalSystem {
    pub active_goals: Vec<Goal>,
    pub goal_stack: Vec<Goal>,
    pub max_goals: usize,
}

impl Default for GoalSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl GoalSystem {
    pub fn new() -> Self {
        Self {
            active_goals: Vec::new(),
            goal_stack: Vec::new(),
            max_goals: 5,
        }
    }

    pub fn update_goals(&mut self, cognitive_state: &CognitiveState, memory: &MemorySystem) {
        // Add exploration goals based on curiosity
        if cognitive_state.curiosity_level > 0.8 {
            let exploration_goal = Goal {
                id: format!("explore_{}", self.active_goals.len()),
                goal_type: GoalType::Navigate,
                priority: cognitive_state.curiosity_level,
                target_position: None, // Will be filled by exploration logic
                deadline: None,
            };
            self.add_goal(exploration_goal);
        }

        // Add learning goals based on prediction error
        if cognitive_state.prediction_error > 0.5 {
            let learning_goal = Goal {
                id: format!("learn_{}", self.active_goals.len()),
                goal_type: GoalType::Learn,
                priority: cognitive_state.prediction_error,
                target_position: None,
                deadline: None,
            };
            self.add_goal(learning_goal);
        }

        // Add memory-based goals based on episodic memory novelty
        let novelty = self.calculate_memory_novelty(memory);
        if novelty > 0.8 {
            let memory_goal = Goal {
                id: format!("memory_explore_{}", self.active_goals.len()),
                goal_type: GoalType::Optimize,
                priority: novelty,
                target_position: None,
                deadline: None,
            };
            self.add_goal(memory_goal);
        }

        // Maintain goal limit
        while self.active_goals.len() > self.max_goals {
            self.active_goals
                .sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
            self.active_goals.pop();
        }
    }

    pub fn add_goal(&mut self, goal: Goal) {
        self.active_goals.push(goal);
        self.active_goals
            .sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
    }

    pub fn get_active_goals(&self) -> &[Goal] {
        &self.active_goals
    }

    /// Calculate novelty based on memory diversity
    fn calculate_memory_novelty(&self, memory: &MemorySystem) -> f64 {
        if memory.episodic_memory.is_empty() {
            return 1.0; // Completely novel
        }

        // Calculate diversity of positions in memory
        let mut positions: HashSet<String> = HashSet::new();
        for experience in &memory.episodic_memory {
            let key = format!(
                "{:.1}_{:.1}",
                experience.position[0], experience.position[1]
            );
            positions.insert(key);
        }

        // Novelty based on position diversity relative to memory size
        let diversity_ratio = positions.len() as f64 / memory.episodic_memory.len() as f64;
        diversity_ratio.clamp(0.0, 1.0)
    }
}

/// Goal representation
#[derive(Debug, Clone)]
pub struct Goal {
    pub id: String,
    pub goal_type: GoalType,
    pub priority: f64,
    pub target_position: Option<Vec<f64>>,
    pub deadline: Option<std::time::SystemTime>,
}

/// Types of goals
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GoalType {
    Navigate,
    Learn,
    Optimize,
    Communicate,
}

/// Learning system for adaptation and improvement
pub struct LearningSystem {
    pub learned_patterns: HashMap<String, LearnedPattern>,
    pub reinforcement_history: Vec<ReinforcementEvent>,
}

impl Default for LearningSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningSystem {
    pub fn new() -> Self {
        Self {
            learned_patterns: HashMap::new(),
            reinforcement_history: Vec::new(),
        }
    }

    pub fn learn(&mut self, cognitive_state: &CognitiveState, perception: &PerceptionSystem) {
        // Learn patterns from perception, scaled by confidence
        let pattern_key = self.extract_pattern_key(perception);
        let pattern = self
            .learned_patterns
            .entry(pattern_key.clone())
            .or_insert_with(|| LearnedPattern {
                key: pattern_key,
                strength: 0.0,
                predictions: HashMap::new(),
            });

        pattern.strength += 0.1 * cognitive_state.confidence_level;
    }

    pub fn predict_optimal_direction(
        &self,
        position: &[f64],
        cognitive_state: &CognitiveState,
        environment: &Environment,
    ) -> Vec<f64> {
        // Use learned patterns first
        let pattern_key = self.position_to_pattern_key(position);

        if let Some(pattern) = self.learned_patterns.get(&pattern_key) {
            // Use pattern predictions to guide movement
            let mut best_direction = vec![0.0, 0.0];
            let mut best_score = 0.0;

            for (direction_key, prediction) in &pattern.predictions {
                let direction = self.direction_key_to_vector(direction_key);
                let score = prediction * pattern.strength;

                if score > best_score {
                    best_score = score;
                    best_direction = direction;
                }
            }

            if best_score > 0.0 {
                return best_direction;
            }
        }

        // Evaluate candidate directions based on environment obstacles
        let candidate_directions = vec![
            vec![1.0, 0.0],       // east
            vec![-1.0, 0.0],      // west
            vec![0.0, 1.0],       // north
            vec![0.0, -1.0],      // south
            vec![0.707, 0.707],   // northeast
            vec![-0.707, 0.707],  // northwest
            vec![0.707, -0.707],  // southeast
            vec![-0.707, -0.707], // southwest
        ];

        let mut best_direction = vec![0.0, 0.0];
        let mut min_penalty = f64::INFINITY;

        for direction in candidate_directions {
            let candidate_target = vec![
                position[0] + direction[0] * 0.1,
                position[1] + direction[1] * 0.1,
            ];
            let penalty = environment.calculate_obstacle_penalty(position, &candidate_target);
            if penalty < min_penalty {
                min_penalty = penalty;
                best_direction = direction;
            }
        }

        if min_penalty >= 0.8 {
            // High penalty, fall back to curiosity-biased random
            let angle = cognitive_state.curiosity_level * std::f64::consts::PI * 2.0;
            vec![angle.cos(), angle.sin()]
        } else {
            best_direction
        }
    }

    fn extract_pattern_key(&self, perception: &PerceptionSystem) -> String {
        format!(
            "pattern_{:.1}_{:.1}_{:.1}",
            perception.sensory_buffer.first().unwrap_or(&0.0),
            perception.sensory_buffer.get(1).unwrap_or(&0.0),
            perception.sensory_buffer.get(2).unwrap_or(&0.0)
        )
    }

    fn position_to_pattern_key(&self, position: &[f64]) -> String {
        format!(
            "pos_{:.1}_{:.1}",
            position.first().unwrap_or(&0.0),
            position.get(1).unwrap_or(&0.0)
        )
    }

    fn direction_key_to_vector(&self, key: &str) -> Vec<f64> {
        match key {
            "north" => vec![0.0, 1.0],
            "south" => vec![0.0, -1.0],
            "east" => vec![1.0, 0.0],
            "west" => vec![-1.0, 0.0],
            _ => vec![0.0, 0.0],
        }
    }
}

/// Learned pattern representation
#[derive(Debug, Clone)]
pub struct LearnedPattern {
    pub key: String,
    pub strength: f64,
    pub predictions: HashMap<String, f64>,
}

/// Reinforcement learning event
#[derive(Debug, Clone)]
pub struct ReinforcementEvent {
    pub timestamp: std::time::SystemTime,
    pub state: Vec<f64>,
    pub action: CognitiveAction,
    pub reward: f64,
}

/// Perception system for sensory processing
pub struct PerceptionSystem {
    pub sensory_buffer: Vec<f64>,
    pub current_position: Vec<f64>,
    pub attention_map: HashMap<String, f64>,
}

impl Default for PerceptionSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl PerceptionSystem {
    pub fn new() -> Self {
        Self {
            sensory_buffer: Vec::new(),
            current_position: vec![0.0, 0.0],
            attention_map: HashMap::new(),
        }
    }

    pub fn update_perception(&mut self, environment: &Environment, position: &[f64]) {
        self.current_position = position.to_vec();

        // Sample environment at current position
        self.sensory_buffer = environment.sample_sensors(position);

        // Update attention map based on salient features
        self.update_attention_map();
    }

    pub fn get_salient_position(&self) -> Vec<f64> {
        // Return position with highest attention
        self.attention_map
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(pos_key, _)| self.key_to_position(pos_key))
            .unwrap_or_else(|| vec![0.0, 0.0])
    }

    pub fn calculate_prediction_error(&self, predicted_state: &CognitiveState) -> f64 {
        // Simplified error calculation
        (self.sensory_buffer.iter().sum::<f64>() - predicted_state.confidence_level).abs()
    }

    pub fn calculate_novelty(&self, memory: &MemorySystem) -> f64 {
        let similar_experiences = memory
            .episodic_memory
            .iter()
            .filter(|exp| self.calculate_similarity(&exp.sensory_input) > 0.8)
            .count();

        1.0 / (1.0 + similar_experiences as f64)
    }

    fn update_attention_map(&mut self) {
        for &value in self.sensory_buffer.iter() {
            let pos_key = format!(
                "{:.1}_{:.1}",
                self.current_position[0], self.current_position[1]
            );
            let attention = self.attention_map.entry(pos_key).or_insert(0.0);
            *attention += value * 0.1;
        }
    }

    fn calculate_similarity(&self, other: &[f64]) -> f64 {
        if self.sensory_buffer.len() != other.len() {
            return 0.0;
        }

        let diff_sum: f64 = self
            .sensory_buffer
            .iter()
            .zip(other)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        1.0 / (1.0 + diff_sum)
    }

    fn key_to_position(&self, key: &str) -> Vec<f64> {
        key.split('_')
            .filter_map(|s| s.parse::<f64>().ok())
            .collect()
    }
}

/// Environment for XAGE agents
#[derive(Debug, Clone)]
pub struct Environment {
    pub dimensions: usize,
    pub features: Vec<EnvironmentalFeature>,
    pub sensory_channels: usize,
}

impl Environment {
    pub fn new(dimensions: usize, sensory_channels: usize) -> Self {
        Self {
            dimensions,
            features: Vec::new(),
            sensory_channels,
        }
    }

    pub fn sample_sensors(&self, position: &[f64]) -> Vec<f64> {
        (0..self.sensory_channels)
            .map(|i| self.sample_channel(position, i))
            .collect()
    }

    /// Calculate obstacle penalty for path from start to target
    pub fn calculate_obstacle_penalty(&self, start: &[f64], target: &[f64]) -> f64 {
        // Simplified obstacle detection based on environmental features
        let mut penalty = 0.0;

        for feature in &self.features {
            let distance_to_start = self.euclidean_distance(start, &feature.position);
            let distance_to_target = self.euclidean_distance(target, &feature.position);

            // If feature is close to the path, add penalty based on intensity
            let path_distance = (distance_to_start + distance_to_target) * 0.5;
            if path_distance < 0.5 {
                // Close to path
                penalty += feature.intensity * (1.0 - path_distance / 0.5);
            }
        }

        penalty.min(0.9) // Cap at 90% penalty
    }

    fn sample_channel(&self, position: &[f64], channel: usize) -> f64 {
        // Simplified sensory sampling
        let base_value = position.iter().sum::<f64>() * 0.1;
        let noise = (channel as f64 * 0.1).sin() * 0.1;
        (base_value + noise).clamp(0.0, 1.0)
    }

    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct EnvironmentalFeature {
    pub position: Vec<f64>,
    pub feature_type: String,
    pub intensity: f64,
}

/// Cognitive actions that agents can take
#[derive(Debug, Clone)]
pub enum CognitiveAction {
    Move(Vec<f64>, f64), // direction, distance
    Observe,
    Wait,
    Explore(Vec<f64>),   // target position
    Signal(f64),         // signal strength
    Communicate(String), // message
    Refactor {
        target_file: String,
        strategy_xuid: Xuid,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xage_creation() {
        let xage = Xage::new("test_agent".to_string(), vec![0.0, 0.0]);

        assert_eq!(xage.id, "test_agent");
        assert_eq!(xage.position, vec![0.0, 0.0]);
        assert!(xage.memory_system.episodic_memory.is_empty());
    }

    #[test]
    fn test_cognitive_cycle() {
        let mut xage = Xage::new("test_agent".to_string(), vec![0.0, 0.0]);
        let environment = Environment::new(2, 3);
        let mut rng = rand::rng();

        let action = xage.cognitive_cycle(&environment, &mut rng).unwrap();

        // Should generate some action
        match action {
            CognitiveAction::Move(_, _)
            | CognitiveAction::Observe
            | CognitiveAction::Wait
            | CognitiveAction::Explore(_)
            | CognitiveAction::Signal(_)
            | CognitiveAction::Communicate(_)
            | CognitiveAction::Refactor { .. } => {} // Valid actions
        }
    }

    #[test]
    fn test_memory_system() {
        let mut memory = MemorySystem::new();
        let cognitive_state = CognitiveState::default();
        let mut perception = PerceptionSystem::new();
        perception.sensory_buffer = vec![0.5, 0.7, 0.3];

        memory.store_experience(&cognitive_state, &perception);

        assert_eq!(memory.episodic_memory.len(), 1);

        let uncertainty = memory.get_uncertainty_map();
        assert!(!uncertainty.is_empty());
    }

    #[test]
    fn test_goal_system() {
        let mut goal_system = GoalSystem::new();
        let cognitive_state = CognitiveState {
            curiosity_level: 0.9,
            prediction_error: 0.7,
            ..Default::default()
        };
        let memory = MemorySystem::new();

        goal_system.update_goals(&cognitive_state, &memory);

        // Should have added goals based on high curiosity and prediction error
        assert!(!goal_system.active_goals.is_empty());
    }

    #[test]
    fn test_learning_system() {
        let mut learning = LearningSystem::new();
        let cognitive_state = CognitiveState::default();
        let mut perception = PerceptionSystem::new();
        perception.sensory_buffer = vec![0.5, 0.7, 0.3];

        learning.learn(&cognitive_state, &perception);

        assert!(!learning.learned_patterns.is_empty());
    }

    #[test]
    fn test_perception_system() {
        let mut perception = PerceptionSystem::new();
        let environment = Environment::new(2, 3);
        let position = vec![0.5, 0.5];

        perception.update_perception(&environment, &position);

        assert_eq!(perception.current_position, position);
        assert_eq!(perception.sensory_buffer.len(), 3);
        assert!(!perception.attention_map.is_empty());
    }

    #[test]
    fn test_environment_sampling() {
        let environment = Environment::new(2, 4);
        let position = vec![0.5, 0.5];

        let sensors = environment.sample_sensors(&position);

        assert_eq!(sensors.len(), 4);
        for &value in &sensors {
            assert!((0.0..=1.0).contains(&value));
        }
    }
}
