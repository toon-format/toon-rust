// src/progen_reactor.rs
//! Procedural generation reactor seeded by sensory/concept/plan inputs and evaluated via
//! UECC-guided critics and fractal traces (see docs/progen.md for more context).
/*!
//! This module introduces:
//! - `ProgenContext`/`ProgenReactor` for turning semantic seeds into generative programs.
//! - `FractalSimtTrace`-aware execution so critics can rewind or propose counterfactuals.
//! - Default critics that reward semantic tension and guard against stale transitions.
!*/
use crate::bitcodec::lossless::gf8_from_lossless_code;
use crate::bitcodec::lossless::gf8_to_lossless_code_closest;
use crate::{
    Gf8, Gf8LosslessCode,
    e8f::E8F,
    fractal_simt::FractalSimtConfig,
    fractal_simt::FractalSimtTrace,
    generative::{GenerativeSynthesizer, ProgramInstr, STATE_TRANSITIONS},
};

/// Semantic context for a procedural branch.
#[derive(Clone, Debug)]
pub struct ProgenContext {
    /// Centroid vector derived from the current cognitive cursor.
    pub centroid: Gf8,
    /// Sensory/perceptual roots (flattened from injector outputs).
    pub sensory_roots: Vec<u8>,
    /// Plan/goal targeting hints (mostly root indices supplied by goals).
    pub plan_targets: Vec<u8>,
    /// Unique timeline identifier (for debugging / tracing).
    pub timeline_id: u64,
}

/// Counterfactual proposal from a critic.
#[derive(Clone, Debug)]
pub struct ProgenCounterfactual {
    pub program_index: usize,
    pub instr: ProgramInstr,
    pub description: &'static str,
}

/// Internal representation of a generative seed.
#[derive(Clone, Debug)]
struct ProgenSeed {
    centroid: Gf8,
    program: Vec<ProgramInstr>,
    context_id: u64,
    goal_targets: Vec<u8>,
}

impl ProgenSeed {
    fn mutate(&self, cf: &ProgenCounterfactual) -> Self {
        let mut program = self.program.clone();
        if cf.program_index < program.len() {
            program[cf.program_index] = cf.instr;
        }
        Self {
            centroid: self.centroid,
            program,
            context_id: self.context_id,
            goal_targets: self.goal_targets.clone(),
        }
    }
}

/// Candidate branch produced by the reactor.
#[derive(Clone, Debug)]
pub struct ProgenBranch {
    pub path: Vec<E8F>,
    pub trace: FractalSimtTrace,
    pub score: f32,
    pub context_id: u64,
}

pub trait ProgenCritic: Send + Sync {
    fn name(&self) -> &'static str;
    fn evaluate(&self, trace: &FractalSimtTrace, path: &[E8F]) -> f32;
    fn propose_counterfactual(
        &self,
        trace: &FractalSimtTrace,
        path: &[E8F],
    ) -> Option<ProgenCounterfactual>;
}

impl ProgenCritic for () {
    fn name(&self) -> &'static str {
        "noop"
    }

    fn evaluate(&self, _trace: &FractalSimtTrace, _path: &[E8F]) -> f32 {
        0.0
    }

    fn propose_counterfactual(
        &self,
        _trace: &FractalSimtTrace,
        _path: &[E8F],
    ) -> Option<ProgenCounterfactual> {
        None
    }
}

/// Default critic that rewards semantic tension and nudges stuck branches.
pub struct SemanticTensionCritic;

impl ProgenCritic for SemanticTensionCritic {
    fn name(&self) -> &'static str {
        "semantic-tension"
    }

    fn evaluate(&self, trace: &FractalSimtTrace, _path: &[E8F]) -> f32 {
        trace
            .entries()
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                entry
                    .root
                    .filter(|root| (*root as usize) < STATE_TRANSITIONS.len())
                    .map(|root| {
                        let transition = &STATE_TRANSITIONS[root as usize];
                        0.01 + (transition.value_a.len() + transition.value_b.len()) as f32 * 0.0003
                            + (i as f32 * 0.0001)
                    })
                    .unwrap_or(0.0)
            })
            .sum()
    }

    fn propose_counterfactual(
        &self,
        trace: &FractalSimtTrace,
        _path: &[E8F],
    ) -> Option<ProgenCounterfactual> {
        if trace.entries().len() < 2 {
            return None;
        }

        let last = trace.entries().last()?;
        let prev = trace.entries().get(trace.entries().len() - 2)?;

        if last.root == prev.root {
            let next_instr =
                ((last.root.unwrap_or(0) as usize + 1) % STATE_TRANSITIONS.len()) as ProgramInstr;
            Some(ProgenCounterfactual {
                program_index: last.index,
                instr: next_instr,
                description: "semantic pivot",
            })
        } else {
            None
        }
    }
}

/// Procedural reactor that spins up sectors of the generative synthesizer + critics.
pub struct ProgenReactor {
    cfg: FractalSimtConfig,
    delta_bank: Vec<Gf8>,
    critics: Vec<Box<dyn ProgenCritic>>,
    program_length: usize,
    max_counterfactuals: usize,
}

impl Default for ProgenReactor {
    fn default() -> Self {
        let delta_bank = STATE_TRANSITIONS
            .iter()
            .map(|entry| gf8_from_lossless_code(Gf8LosslessCode(entry.index)))
            .collect::<Vec<_>>();

        Self {
            cfg: FractalSimtConfig::default(),
            delta_bank,
            critics: vec![Box::new(SemanticTensionCritic)],
            program_length: 32,
            max_counterfactuals: 2,
        }
    }
}

impl ProgenReactor {
    pub fn new(cfg: FractalSimtConfig, program_length: usize, max_counterfactuals: usize) -> Self {
        let delta_bank = STATE_TRANSITIONS
            .iter()
            .map(|entry| gf8_from_lossless_code(Gf8LosslessCode(entry.index)))
            .collect::<Vec<_>>();

        Self {
            cfg,
            delta_bank,
            critics: vec![Box::new(SemanticTensionCritic)],
            program_length: program_length.max(1),
            max_counterfactuals,
        }
    }

    pub fn add_critic(&mut self, critic: Box<dyn ProgenCritic>) {
        self.critics.push(critic);
    }

    pub fn run(&self, contexts: &[ProgenContext]) -> Vec<ProgenBranch> {
        contexts
            .iter()
            .flat_map(|context| self.spawn_from_context(context))
            .collect()
    }

    pub fn best_branch(&self, contexts: &[ProgenContext]) -> Option<ProgenBranch> {
        self.run(contexts).into_iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn spawn_from_context(&self, context: &ProgenContext) -> Vec<ProgenBranch> {
        let mut branches = Vec::new();
        let mut seed = self.build_seed(context);
        if let Some(branch) = self.run_branch(&seed) {
            branches.push(branch.clone());
            for _ in 0..self.max_counterfactuals {
                if let Some(cf) = self.critics.iter().find_map(|critic| {
                    critic.propose_counterfactual(
                        &branches.last().unwrap().trace,
                        &branches.last().unwrap().path,
                    )
                }) {
                    seed = seed.mutate(&cf);
                    if let Some(next) = self.run_branch(&seed) {
                        branches.push(next);
                        continue;
                    }
                }
                break;
            }
        }
        branches
    }

    fn run_branch(&self, seed: &ProgenSeed) -> Option<ProgenBranch> {
        if seed.program.is_empty() {
            return None;
        }

        let synth = GenerativeSynthesizer::new(
            seed.centroid,
            self.delta_bank.clone(),
            seed.program.clone(),
        );
        let mut dst = vec![Gf8::default(); seed.program.len()];
        let trace = synth.fill_reconstructed_with_trace(&mut dst, &self.cfg);

        let path = dst
            .into_iter()
            .map(|gf| {
                let code = gf8_to_lossless_code_closest(&gf);
                E8F::from(code)
            })
            .collect::<Vec<_>>();

        let score = self.score_branch(&trace, &path, seed);
        Some(ProgenBranch {
            path,
            trace,
            score,
            context_id: seed.context_id,
        })
    }

    fn score_branch(&self, trace: &FractalSimtTrace, path: &[E8F], seed: &ProgenSeed) -> f32 {
        let critic_score: f32 = self
            .critics
            .iter()
            .map(|critic| critic.evaluate(trace, path))
            .sum();
        let goal_bonus = if seed.goal_targets.is_empty() {
            0.0
        } else {
            let trace_roots: Vec<u8> = trace
                .entries()
                .iter()
                .filter_map(|entry| entry.root)
                .collect();
            if seed
                .goal_targets
                .iter()
                .any(|goal| trace_roots.contains(goal))
            {
                1.0
            } else {
                0.0
            }
        };
        critic_score + goal_bonus + (path.len() as f32 * 0.005)
    }

    fn build_seed(&self, context: &ProgenContext) -> ProgenSeed {
        let sensory_len = context.sensory_roots.len();
        let plan_len = context.plan_targets.len();
        let mut program = Vec::with_capacity(self.program_length);

        for i in 0..self.program_length {
            if self.cfg.warp_size != 0 && i % self.cfg.warp_size == 0 {
                program.push(0xFF);
                continue;
            }

            let instr = if plan_len > 0 {
                context.plan_targets[i % plan_len]
            } else if sensory_len > 0 {
                context.sensory_roots[i % sensory_len]
            } else {
                (i % STATE_TRANSITIONS.len()) as u8
            };
            program.push(instr % STATE_TRANSITIONS.len() as u8);
        }

        ProgenSeed {
            centroid: context.centroid,
            program,
            context_id: context.timeline_id,
            goal_targets: context.plan_targets.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reactor_returns_best_branch() {
        let context = ProgenContext {
            centroid: gf8_from_lossless_code(Gf8LosslessCode(0)),
            sensory_roots: vec![1, 2, 3],
            plan_targets: vec![4, 5],
            timeline_id: 42,
        };
        let reactor = ProgenReactor {
            program_length: 8,
            ..ProgenReactor::default()
        };
        let branches = reactor.run(&[context.clone()]);
        assert!(!branches.is_empty());
        assert!(branches.iter().any(|branch| branch.score >= 0.0));
        let best = reactor.best_branch(&[context]);
        assert!(best.is_some());
    }
}
