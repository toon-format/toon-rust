// src/generative.rs
//! A compact generative synthesizer for reconstructing clusters of `Gf8` vectors
//! using a centroid, a small LUT of delta vectors and a compact program string.
//!
//! This is a minimal demonstration of the "synthesizer" pattern you described.
//! It trades storage for runtime reconstruction using small SIMD-backed kernels.

use crate::bitcodec::lossless::gf8_to_lossless_code_closest;
use crate::fractal_simt::{
    FractalSimtConfig, FractalSimtTrace, fractal_simt_for_each_indexed, fractal_simt_trace,
};
use crate::{Gf8, gf8_add_simd};

/// Compact program instruction mapping
/// 255 (0xFF) means use centroid, otherwise use LUT entry by index
pub type ProgramInstr = u8;

/// Captures semantic neighbor metadata for the 56 key transitions.
#[derive(Clone, Debug)]
pub struct StateTransition {
    pub id: &'static str,
    pub index: u8,
    pub value_a: &'static str,
    pub value_b: &'static str,
    pub description: &'static str,
}

/// The 56 concept pair transitions described in `e8_cognitive_architecture_mapping.md`
/// and mirrored in `docs/UECC.pdf`.
pub const STATE_TRANSITIONS: [StateTransition; 56] = [
    StateTransition {
        id: "A",
        index: 0,
        value_a: "Self-Agency",
        value_b: "Communion",
        description: "From sovereign choice into shared belonging.",
    },
    StateTransition {
        id: "B",
        index: 1,
        value_a: "Structure",
        value_b: "Flux",
        description: "Holding order while letting flow in.",
    },
    StateTransition {
        id: "C",
        index: 2,
        value_a: "Foresight",
        value_b: "Memory",
        description: "Balancing looking ahead with remembering lessons.",
    },
    StateTransition {
        id: "D",
        index: 3,
        value_a: "Clarity",
        value_b: "Awe",
        description: "Precise focus meets reverent openness.",
    },
    StateTransition {
        id: "E",
        index: 4,
        value_a: "Virtue",
        value_b: "Temptation",
        description: "Maintaining higher ideals when the pull is strong.",
    },
    StateTransition {
        id: "F",
        index: 5,
        value_a: "Momentum",
        value_b: "Stillness",
        description: "Propelling forward yet honoring quiet rest.",
    },
    StateTransition {
        id: "G",
        index: 6,
        value_a: "Stewardship",
        value_b: "Openness",
        description: "Holding responsibility while inviting novel ideas.",
    },
    StateTransition {
        id: "H",
        index: 7,
        value_a: "Mastery",
        value_b: "Risk",
        description: "Confidence tempered by daring leaps.",
    },
    StateTransition {
        id: "I",
        index: 8,
        value_a: "Certainty",
        value_b: "Doubt",
        description: "Grounded belief that still honors questions.",
    },
    StateTransition {
        id: "J",
        index: 9,
        value_a: "Presence",
        value_b: "Absence",
        description: "Being fully here while acknowledging empty space.",
    },
    StateTransition {
        id: "K",
        index: 10,
        value_a: "Creation",
        value_b: "Destruction",
        description: "Birth and fade are two sides of transformation.",
    },
    StateTransition {
        id: "L",
        index: 11,
        value_a: "Unity",
        value_b: "Diversity",
        description: "Togetherness that still celebrates difference.",
    },
    StateTransition {
        id: "M",
        index: 12,
        value_a: "Potential",
        value_b: "Actualization",
        description: "From latent possibility into realized form.",
    },
    StateTransition {
        id: "N",
        index: 13,
        value_a: "Consciousness",
        value_b: "Unconsciousness",
        description: "Awareness that leans into dream logic.",
    },
    StateTransition {
        id: "O",
        index: 14,
        value_a: "Permanence",
        value_b: "Transience",
        description: "Anchoring in the eternal while riding the ephemeral.",
    },
    StateTransition {
        id: "P",
        index: 15,
        value_a: "Order",
        value_b: "Chaos",
        description: "Structure that is resilient to entropy.",
    },
    StateTransition {
        id: "Q",
        index: 16,
        value_a: "Beginning",
        value_b: "Ending",
        description: "The eternal cycle of birth and cessation.",
    },
    StateTransition {
        id: "R",
        index: 17,
        value_a: "Internal",
        value_b: "External",
        description: "The boundary between inner experience and outer reality.",
    },
    StateTransition {
        id: "S",
        index: 18,
        value_a: "Material",
        value_b: "Spiritual",
        description: "The balance of matter and meaning.",
    },
    StateTransition {
        id: "T",
        index: 19,
        value_a: "Effort",
        value_b: "Grace",
        description: "Striving through will versus receiving flow.",
    },
    StateTransition {
        id: "U",
        index: 20,
        value_a: "Simplicity",
        value_b: "Complexity",
        description: "Fundamental clarity alongside intricate systems.",
    },
    StateTransition {
        id: "V",
        index: 21,
        value_a: "Objective",
        value_b: "Subjective",
        description: "Universal truth meets personal perception.",
    },
    StateTransition {
        id: "W",
        index: 22,
        value_a: "Harmony",
        value_b: "Conflict",
        description: "Agreement versus struggle in relationships.",
    },
    StateTransition {
        id: "X",
        index: 23,
        value_a: "Freedom",
        value_b: "Constraint",
        description: "Boundless possibility tempered by limits.",
    },
    StateTransition {
        id: "Y",
        index: 24,
        value_a: "Past",
        value_b: "Future",
        description: "Memory and anticipation in dialogue.",
    },
    StateTransition {
        id: "Z",
        index: 25,
        value_a: "Physical",
        value_b: "Mental",
        description: "Body and mind as interwoven aspects.",
    },
    StateTransition {
        id: "AA",
        index: 26,
        value_a: "Logic",
        value_b: "Intuition",
        description: "Reason and instinct in tandem.",
    },
    StateTransition {
        id: "BB",
        index: 27,
        value_a: "Giving",
        value_b: "Receiving",
        description: "The flow of energy and resources.",
    },
    StateTransition {
        id: "CC",
        index: 28,
        value_a: "Public",
        value_b: "Private",
        description: "Shared life versus individual sanctuary.",
    },
    StateTransition {
        id: "DD",
        index: 29,
        value_a: "Wakefulness",
        value_b: "Sleep",
        description: "Conscious action meets restorative rest.",
    },
    StateTransition {
        id: "EE",
        index: 30,
        value_a: "Strength",
        value_b: "Vulnerability",
        description: "Resilience with openness to harm.",
    },
    StateTransition {
        id: "FF",
        index: 31,
        value_a: "Question",
        value_b: "Answer",
        description: "Seeking knowledge and finding resolution.",
    },
    StateTransition {
        id: "GG",
        index: 32,
        value_a: "Action",
        value_b: "Inaction",
        description: "Doing versus allowing.",
    },
    StateTransition {
        id: "HH",
        index: 33,
        value_a: "Hope",
        value_b: "Despair",
        description: "Expectation extremes about the future.",
    },
    StateTransition {
        id: "II",
        index: 34,
        value_a: "Presence",
        value_b: "Absence",
        description: "Being here versus not being here.",
    },
    StateTransition {
        id: "JJ",
        index: 35,
        value_a: "Success",
        value_b: "Failure",
        description: "Goal outcomes and their twin.",
    },
    StateTransition {
        id: "KK",
        index: 36,
        value_a: "Value",
        value_b: "Meaningless",
        description: "Perceived worth versus lack of significance.",
    },
    StateTransition {
        id: "LL",
        index: 37,
        value_a: "Attachment",
        value_b: "Detachment",
        description: "Emotional connection versus objective distance.",
    },
    StateTransition {
        id: "MM",
        index: 38,
        value_a: "Pleasure",
        value_b: "Pain",
        description: "Spectrum of sensation and feeling.",
    },
    StateTransition {
        id: "NN",
        index: 39,
        value_a: "Known",
        value_b: "Unknown",
        description: "Existing knowledge versus mystery.",
    },
    StateTransition {
        id: "OO",
        index: 40,
        value_a: "Giving Up",
        value_b: "Persistence",
        description: "Knowing when to quit and when to persevere.",
    },
    StateTransition {
        id: "PP",
        index: 41,
        value_a: "Growth",
        value_b: "Stagnation",
        description: "Positive change versus stillness.",
    },
    StateTransition {
        id: "QQ",
        index: 42,
        value_a: "Truth",
        value_b: "Illusion",
        description: "Reality versus deception.",
    },
    StateTransition {
        id: "RR",
        index: 43,
        value_a: "Sound",
        value_b: "Silence",
        description: "Contrast in auditory experience.",
    },
    StateTransition {
        id: "SS",
        index: 44,
        value_a: "Life",
        value_b: "Death",
        description: "Boundaries of biological existence.",
    },
    StateTransition {
        id: "TT",
        index: 45,
        value_a: "Joy",
        value_b: "Sorrow",
        description: "Emotional well-being extremes.",
    },
    StateTransition {
        id: "UU",
        index: 46,
        value_a: "Belief",
        value_b: "Evidence",
        description: "Faith versus empirical proof.",
    },
    StateTransition {
        id: "VV",
        index: 47,
        value_a: "Memory",
        value_b: "Forgetting",
        description: "Retaining the past or letting go.",
    },
    StateTransition {
        id: "WW",
        index: 48,
        value_a: "Reality",
        value_b: "Imagination",
        description: "Actual experience versus conceived worlds.",
    },
    StateTransition {
        id: "XX",
        index: 49,
        value_a: "Public",
        value_b: "Private",
        description: "Shared visibility versus personal space.",
    },
    StateTransition {
        id: "YY",
        index: 50,
        value_a: "Work",
        value_b: "Play",
        description: "Purposeful labor versus joyful activity.",
    },
    StateTransition {
        id: "ZZ",
        index: 51,
        value_a: "Friend",
        value_b: "Enemy",
        description: "Alliance versus opposition.",
    },
    StateTransition {
        id: "AAA",
        index: 52,
        value_a: "Desire",
        value_b: "Aversion",
        description: "Attraction and repulsion.",
    },
    StateTransition {
        id: "BBB",
        index: 53,
        value_a: "Change",
        value_b: "Tradition",
        description: "Innovation versus established practice.",
    },
    StateTransition {
        id: "CCC",
        index: 54,
        value_a: "Youth",
        value_b: "Age",
        description: "Progression through life stages.",
    },
    StateTransition {
        id: "DDD",
        index: 55,
        value_a: "Curiosity",
        value_b: "Complacency",
        description: "Exploring the unknown versus resting in the familiar.",
    },
];

pub fn transition_for(root_index: u8) -> &'static StateTransition {
    &STATE_TRANSITIONS[(root_index as usize) % STATE_TRANSITIONS.len()]
}

/// A tiny generative synthesizer.
///
/// - `centroid`: base Gf8
/// - `deltas`: small LUT of delta vectors (should be small, e.g., 16)
/// - `program`: for each output index, an instruction referencing centroid or a delta
#[derive(Clone)]
pub struct GenerativeSynthesizer {
    pub centroid: Gf8,
    pub deltas: Vec<Gf8>,
    pub program: Vec<ProgramInstr>,
}

impl GenerativeSynthesizer {
    pub fn new(centroid: Gf8, deltas: Vec<Gf8>, program: Vec<ProgramInstr>) -> Self {
        Self {
            centroid,
            deltas,
            program,
        }
    }

    /// Try to construct a new `GenerativeSynthesizer`, validating the program
    /// references all indices into the `deltas` LUT. Returns an `Err` message
    /// on invalid input rather than panicking, allowing callers to handle errors
    /// gracefully.
    pub fn try_new(
        centroid: Gf8,
        deltas: Vec<Gf8>,
        program: Vec<ProgramInstr>,
    ) -> Result<Self, String> {
        // Validate: instruction must be 0xFF (centroid) or a valid index into deltas
        for &instr in program.iter() {
            if instr != 0xFF && (instr as usize) >= deltas.len() {
                return Err(format!(
                    "GenerativeSynthesizer::try_new: program contains out-of-range LUT index {} (deltas.len={})",
                    instr,
                    deltas.len()
                ));
            }
        }
        Ok(Self {
            centroid,
            deltas,
            program,
        })
    }

    /// Reconstruct a single vector by program index.
    pub fn reconstruct(&self, idx: usize) -> Gf8 {
        let instr = self.program[idx];
        if instr == 0xFF {
            self.centroid
        } else {
            // Safe because the program should be generated to only contain valid LUT indexes
            let delta = &self.deltas[instr as usize];
            gf8_add_simd(&self.centroid, delta)
        }
    }

    /// Fill dst with reconstructed vectors for the provided `program` indices.
    /// Uses `fractal_simt_for_each` to demonstrate a scheduler visiting ordering
    /// and reconstructing the values in a cache-friendly order.
    pub fn fill_reconstructed(&self, dst: &mut [Gf8], cfg: &FractalSimtConfig) {
        assert_eq!(
            dst.len(),
            self.program.len(),
            "dst and program length must match"
        );

        // We can't guarantee that a particular `FractalSimtConfig` will visit every
        // element exactly once (and some small sizes / configs can miss indices), so we
        // track visited indices and then fill any unvisited entries in a final linear
        // pass. This preserves the cache-friendly traversal while ensuring correctness.
        let mut visited = vec![false; dst.len()];
        fractal_simt_for_each_indexed(dst, cfg, |_lane, idx, elem| {
            let v = self.reconstruct(idx);
            *elem = v;
            visited[idx] = true;
        });

        // Ensure every index has been reconstructed; fill any missed entries linearly.
        for i in 0..dst.len() {
            if !visited[i] {
                dst[i] = self.reconstruct(i);
            }
        }
    }

    /// Reconstruct directly into a provided `Gf8` mut reference for the given index.
    pub fn reconstruct_into(&self, idx: usize, out: &mut Gf8) {
        *out = self.reconstruct(idx);
    }

    /// Reconstruct into `dst`, returning a trace of the sequence.
    pub fn fill_reconstructed_with_trace(
        &self,
        dst: &mut [Gf8],
        cfg: &FractalSimtConfig,
    ) -> FractalSimtTrace {
        assert_eq!(
            dst.len(),
            self.program.len(),
            "dst and program length must match"
        );
        let mut visited = vec![false; dst.len()];

        let trace = fractal_simt_trace(dst, cfg, |_, idx, elem, trace| {
            let value = self.reconstruct(idx);
            let code = gf8_to_lossless_code_closest(&value);
            if let Some(entry) = trace.entries_mut().last_mut() {
                entry.root = Some(code.0);
                entry.index = idx;
            }
            *elem = value;
            visited[idx] = true;
        });

        for i in 0..dst.len() {
            if !visited[i] {
                dst[i] = self.reconstruct(i);
            }
        }

        trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gf8;

    #[test]
    fn reconstruct_single_matches_expected() {
        let centroid = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let delta = Gf8::new([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let deltas = vec![delta];
        let program = vec![0xFFu8, 0u8];
        let synth = GenerativeSynthesizer::new(centroid, deltas, program);

        let r0 = synth.reconstruct(0);
        assert_eq!(r0, centroid);

        let r1 = synth.reconstruct(1);
        let expected = gf8_add_simd(&centroid, &delta);
        assert_eq!(r1, expected);
    }

    #[test]
    fn fill_reconstructed_matches_linear_reconstruct() {
        let centroid = Gf8::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let delta = Gf8::new([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let deltas = vec![delta];
        let program = vec![0xFFu8, 0u8, 0u8, 0xFFu8];
        let synth = GenerativeSynthesizer::new(centroid, deltas.clone(), program.clone());

        // Linear baseline reconstruction
        let mut baseline = Vec::with_capacity(program.len());
        for i in 0..program.len() {
            baseline.push(synth.reconstruct(i));
        }

        // Use fractal SIMD scheduler to fill into dst
        let cfg = FractalSimtConfig::default();
        let mut dst = vec![Gf8::default(); program.len()];
        synth.fill_reconstructed(&mut dst, &cfg);

        // Compare results
        assert_eq!(dst.len(), baseline.len());
        for i in 0..dst.len() {
            assert_eq!(dst[i], baseline[i], "mismatch at {}", i);
        }
    }

    #[test]
    fn try_new_rejects_invalid_program_index() {
        let centroid = Gf8::default();
        let deltas = Vec::new();
        // Program references LUT index 0 but deltas is empty - should error
        let program = vec![0u8];
        let res = GenerativeSynthesizer::try_new(centroid, deltas, program);
        assert!(
            res.is_err(),
            "Expected try_new to return Err for invalid program indices"
        );
    }

    // Note: `fractal_simt_for_each_indexed` is exercised by the fractal scheduler
    // unit tests located in `fractal_simt.rs`. We avoid repeating these here because
    // parameter-specific behavior is already covered (and some small configs don't
    // guarantee uniform index coverage, which is outside the scope of generative's
    // correctness requirements).

    // Note: We intentionally avoid strict coverage guarantees for the fractal walker
    // since some parameter combinations won't visit every index. The helper is still
    // useful for cache-friendly ordering, and `fill_reconstructed` compensates by
    // filling any missed indices in a final linear pass.
}
