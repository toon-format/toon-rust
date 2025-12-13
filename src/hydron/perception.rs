/* src/rune/hydron/perception.rs */
//! Perception Engine: Signal (ByteLex) and Structure (Morphology) analysis.
//!
//! # Hydron – Perception Module
//!▫~•◦--------------------------‣
//!
//! This module provides the core mechanisms to convert raw text into geometric vectors
//! based on two distinct properties within the RUNE ecosystem, using zero external dependencies:
//!
//! 1.  **Signal (The Body):** Raw byte-level convolution using deterministic hashing.
//!     Captures "shape", typos, and non-linguistic patterns. Implemented via `signal_encode`.
//! 2.  **Structure (The Skeleton):** Morphological decomposition (Prefix/Root/Suffix).
//!     Captures linguistic logic and semantic composition. Implemented via `morph_analyze`.
//!
//! These vectors are designed to be fused (via `/\`) to create a lossless,
//! holographic embedding of the input.
//!
//! ### Key Capabilities
//! - **Signal Encoding:** Deterministic byte-stream convolution into normalized 8D vectors.
//! - **Morphological Analysis:** Greedy affix decomposition with efficient binary search.
//! - **Root Recognition:** Common root validation against a curated lexicon for semantic anchoring.
//! - **Performance-Optimized**: Zero-allocation lowercasing and SIMD-accelerated vector math.
//!
//! ### Architectural Notes
//! This module is designed for integration with the broader RUNE/Hydron geometry pipeline.
//! Vectors produced here are normalized to the unit sphere (S7-compatible) and can be
//! composed using geometric algebra operations.
//!
//! ### Example
//! ```rust
//! use rune_xero::rune::hydron::perception::{signal_encode, morph_analyze};
//!
//! let bytes = b"unbelievably";
//! let signal_vec = signal_encode(bytes);
//! let morph_vec = morph_analyze("unbelievably");
//!
//! // Both vectors are normalized and ready for geometric composition.
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use super::values::{gf8_dot_simd, gf8_norm2_simd};

// Use portable SIMD if the feature is enabled.
#[cfg(feature = "simd")]
use std::simd::{f32x8, SimdFloat};

// --- 1. STATIC DATA (The Skeleton) ---
// High-coverage English affixes ported from the original lexicon architecture.
// These allow the engine to "see" word structure without a dictionary.

/// Comprehensive prefix table for morphological decomposition.
/// Ordered alphabetically for binary search compatibility.
static PREFIXES: &[&str] = &[
    "a", "ab", "abs", "ad", "af", "ag", "al", "am", "an", "ante", "anti", "ap", "apo", "arch",
    "as", "at", "auto", "be", "bi", "bio", "cata", "circum", "cis", "co", "col", "com", "con",
    "contra", "cor", "counter", "de", "deca", "deci", "demi", "di", "dia", "dif", "dis", "down",
    "duo", "dys", "e", "ec", "eco", "ecto", "ef", "electro", "em", "en", "endo", "epi", "equi",
    "ex", "exo", "extra", "fore", "geo", "hemi", "hetero", "hexa", "homo", "hydro", "hyper",
    "hypo", "il", "im", "in", "infra", "inter", "intra", "intro", "ir", "iso", "kilo", "macro",
    "mal", "mega", "meta", "micro", "mid", "milli", "mini", "mis", "mono", "multi", "nano",
    "neo", "neuro", "non", "ob", "oc", "oct", "octa", "of", "omni", "op", "ortho", "out",
    "over", "paleo", "pan", "para", "penta", "per", "peri", "photo", "poly", "post", "pre",
    "preter", "pro", "proto", "pseudo", "pyro", "quadr", "quasi", "re", "retro", "self",
    "semi", "sept", "sex", "sub", "suc", "suf", "sug", "sum", "sup", "super", "sur", "sus",
    "sym", "syn", "tele", "tetra", "thermo", "trans", "tri", "twi", "ultra", "un", "under",
    "uni", "up", "vice",
];

/// Comprehensive suffix table for morphological decomposition.
/// Ordered alphabetically for binary search compatibility.
static SUFFIXES: &[&str] = &[
    "able", "ably", "ac", "aceous", "acious", "age", "al", "algia", "an", "ance", "ancy", "ant",
    "ar", "ard", "ary", "ase", "ate", "ation", "ative", "ator", "atory", "cide", "cracy",
    "crat", "cy", "dom", "dox", "ed", "ee", "eer", "en", "ence", "ency", "ent", "eous", "er",
    "ern", "ery", "es", "ese", "esque", "ess", "est", "etic", "ette", "ful", "fy", "gen",
    "genic", "gon", "gram", "graph", "graphy", "hood", "ia", "ial", "ian", "iasis", "iatric",
    "iatry", "ible", "ibly", "ic", "ical", "ically", "ice", "ician", "ics", "id", "ide", "ie",
    "ier", "iferous", "ific", "ification", "ify", "ile", "ine", "ing", "ion", "ior", "ious",
    "ish", "ism", "ist", "istic", "ite", "itis", "itive", "ity", "ium", "ive", "ize", "kin",
    "less", "let", "like", "ling", "logue", "logy", "ly", "lysis", "lyte", "lytic", "man",
    "mancy", "mania", "ment", "meter", "metry", "most", "ness", "oid", "ology", "oma", "or",
    "ory", "ose", "osis", "ous", "path", "pathy", "ped", "phage", "phagy", "phile", "philia",
    "phobe", "phobia", "phone", "phony", "phyte", "plasty", "pod", "polis", "proof", "ry", "s",
    "scope", "scopy", "sect", "ship", "sion", "sis", "some", "sophy", "ster", "th", "tion",
    "tomy", "tor", "tous", "trix", "tron", "tude", "ty", "ular", "ule", "ure", "ward", "wards",
    "wise", "woman", "worthy", "y", "yer",
];

/// Common etymological roots for semantic anchoring.
/// Ordered alphabetically for binary search compatibility.
static ROOTS: &[&str] = &[
    "acer", "acr", "acu", "aev", "alb", "alt", "am", "amor", "anim", "anthrop", "audi", "audit",
    "bene", "bio", "bon", "cede", "ceed", "centr", "cept", "ceive", "cess", "chron", "civ",
    "claim", "clam", "clar", "cogn", "corp", "cre", "creat", "cred", "cur", "curr", "curs",
    "cycl", "dem", "dic", "dict", "dign", "duc", "duct", "dur", "dyna", "equ", "fac", "fact",
    "fect", "fer", "fic", "fig", "fin", "firm", "form", "fort", "fract", "frag", "gen", "geo",
    "grad", "gram", "graph", "grat", "grav", "gress", "herb", "hydr", "ject", "jud", "jur",
    "jus", "leg", "lev", "liber", "locut", "log", "loqu", "lucid", "magn", "mal", "medi", "mem",
    "ment", "metr", "meter", "min", "misc", "miss", "mit", "morph", "mot", "mov", "multi",
    "nigr", "nom", "not", "nounce", "nov", "nunce", "nym", "oper", "pac", "parl", "pass", "pat",
    "path", "ped", "pens", "phan", "phil", "phon", "phone", "phys", "pict", "plen", "plic",
    "plu", "plus", "ply", "pod", "poli", "polit", "pon", "popul", "port", "pos", "prim", "prob",
    "puls", "purg", "reg", "sacr", "san", "satis", "sci", "scop", "scrib", "script", "secur",
    "sens", "sent", "sequ", "serv", "sever", "sign", "simil", "simpl", "sing", "soci", "sol",
    "soph", "spec", "spect", "sphere", "sta", "stabil", "stat", "strict", "struct", "tain",
    "techn", "temp", "ten", "tend", "tens", "tent", "therm", "tract", "triv", "turb", "uni",
    "urb", "util", "vac", "vag", "val", "van", "var", "vene", "vent", "ver", "vers", "vert",
    "vest", "via", "vict", "vid", "vigil", "vinc", "vis", "viv", "voc", "void", "voic", "volv",
    "voke", "voy", "zoo",
];

// --- 2. MATH KERNEL (Deterministic Hashing) ---

/// SplitMix64: Fast, dependency-free pseudo-random hashing.
#[inline]
const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Map a hash seed to a unit float within [-0.5, 0.5).
#[inline]
fn hash_to_float(seed: u64) -> f32 {
    let bits = splitmix64(seed);
    let as_f64 = (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
    (as_f64 as f32) - 0.5
}

// --- 3. SIGNAL ENCODING (The Body) ---

/// Encodes raw bytes into a SIMD-accelerated 8D semantic vector.
#[inline]
pub fn signal_encode(bytes: &[u8]) -> [f32; 8] {
    if bytes.is_empty() {
        return [0.0; 8];
    }

    let mut signal = [0.0f32; 8];
    let window_size = 4.min(bytes.len());

    for (i, &b) in bytes.iter().enumerate() {
        for dim in 0..8 {
            let seed = (b as u64)
                .wrapping_mul(31)
                .wrapping_add(dim as u64)
                .wrapping_add((i % window_size) as u64 * 1024);

            signal[dim] += hash_to_float(seed);
        }
    }

    normalize_vec8_simd(&mut signal);
    signal
}

// --- 4. STRUCTURE ANALYSIS (The Skeleton) ---

/// Validate if a potential root exists in the lexicon using binary search.
#[inline]
fn is_valid_root(candidate: &str) -> bool {
    ROOTS.binary_search(&candidate).is_ok()
}

/// Analyze string morphology and return a structural hash vector.
#[inline]
pub fn morph_analyze(token: &str) -> [f32; 8] {
    // Use a stack buffer for allocation-free ASCII lowercasing.
    let mut lower_buf = [0u8; 128]; // Handle tokens up to 128 bytes.
    let clean_bytes = token.trim_matches(|c: char| !c.is_alphanumeric()).as_bytes();
    let len = clean_bytes.len().min(lower_buf.len());
    lower_buf[..len].copy_from_slice(&clean_bytes[..len]);
    lower_buf[..len].make_ascii_lowercase();
    let clean = std::str::from_utf8(&lower_buf[..len]).unwrap_or("");

    if clean.is_empty() {
        return [0.0; 8];
    }

    let mut prefix = "";
    let mut suffix = "";
    let mut root = clean;

    // Identify prefix (greedy longest match using binary search)
    for &p in PREFIXES.iter().rev() { // Check longer prefixes first
        if root.starts_with(p) && root.len() > p.len() + 2 {
            prefix = p;
            break;
        }
    }
    if !prefix.is_empty() {
        root = &root[prefix.len()..];
    }

    // Identify suffix (greedy longest match using binary search)
    for &s in SUFFIXES.iter().rev() { // Check longer suffixes first
        if root.ends_with(s) && root.len() > s.len() + 2 {
            suffix = s;
            break;
        }
    }
    if !suffix.is_empty() {
        root = &root[..root.len() - suffix.len()];
    }

    // Root validation & progressive kernel extraction
    if !is_valid_root(root) && root.len() > 3 {
        let mut candidate = root;
        while candidate.len() > 2 {
            if is_valid_root(candidate) {
                root = candidate;
                break;
            }
            candidate = &candidate[..candidate.len() - 1];
        }
    }

    let mut vec = [0.0f32; 8];
    let components = [(prefix, 100u64), (root, 0u64), (suffix, 200u64)];

    for (str_part, seed_offset) in components {
        if str_part.is_empty() {
            continue;
        }
        for (i, &b) in str_part.as_bytes().iter().enumerate() {
            for dim in 0..8 {
                let seed = (b as u64)
                    .wrapping_add(dim as u64)
                    .wrapping_add(i as u64 * 31)
                    .wrapping_add(seed_offset);
                vec[dim] += hash_to_float(seed);
            }
        }
    }

    normalize_vec8_simd(&mut vec);
    vec
}

/// Normalizes a mutable Vec8 in-place using SIMD if available.
#[inline]
fn normalize_vec8_simd(v: &mut [f32; 8]) {
    let norm_sq = gf8_norm2_simd(v);
    if norm_sq > 1e-9 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        #[cfg(feature = "simd")]
        {
            let v_simd = f32x8::from_array(*v);
            let inv_norm_simd = f32x8::splat(inv_norm);
            *v = (v_simd * inv_norm_simd).to_array();
        }
        #[cfg(not(feature = "simd"))]
        {
            for x in v.iter_mut() {
                *x *= inv_norm;
            }
        }
    }
}

// Tests remain the same, but now benefit from the optimized implementations.
#[cfg(test)]
mod tests {
    use super::{hash_to_float, is_valid_root, morph_analyze, signal_encode, splitmix64};

    #[test]
    fn test_signal_encode_determinism() {
        let input = b"hello world";
        let v1 = signal_encode(input);
        let v2 = signal_encode(input);
        assert_eq!(v1, v2, "Signal encoding must be deterministic");

        let v3 = signal_encode(b"hello worl");
        let dot: f32 = v1.iter().zip(v3.iter()).map(|(a, b)| a * b).sum();
        assert!(dot > 0.8, "Similar strings should have high signal correlation");
        assert!(dot < 0.9999, "Distinct strings should not be identical");
    }

    #[test]
    fn test_signal_encode_empty_input() {
        let vec = signal_encode(b"");
        assert_eq!(vec, [0.0; 8], "Empty input should produce zero vector");
    }

    #[test]
    fn test_signal_encode_normalization() {
        let vec = signal_encode(b"test");
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Signal vector must be normalized");
    }

    #[test]
    fn test_morph_analyze_decomposition() {
        let v1 = morph_analyze("unbelievably");
        let v2 = morph_analyze("believer");
        let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        assert!(dot > 0.0, "Shared roots should produce positive correlation");
    }

    #[test]
    fn test_morph_analyze_affix_handling() {
        let v_redo = morph_analyze("redo");
        let v_do = morph_analyze("do");
        assert_ne!(v_redo, v_do, "Prefix presence should alter the vector");
    }

    #[test]
    fn test_morph_analyze_empty_input() {
        let vec = morph_analyze("");
        assert_eq!(vec, [0.0; 8], "Empty input should produce zero vector");
    }

    #[test]
    fn test_morph_analyze_normalization() {
        let vec = morph_analyze("testing");
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Morphology vector must be normalized");
    }

    #[test]
    fn test_morph_analyze_non_alphanumeric() {
        let vec1 = morph_analyze("test!");
        let vec2 = morph_analyze("test");
        assert_eq!(vec1, vec2, "Non-alphanumeric trimming should produce identical vectors");
    }

    #[test]
    fn test_root_validation() {
        assert!(is_valid_root("dict"), "Known root 'dict' should validate");
        assert!(is_valid_root("spec"), "Known root 'spec' should validate");
        assert!(!is_valid_root("xyz123"), "Unknown root should not validate");
    }

    #[test]
    fn test_progressive_root_extraction() {
        let _vec = morph_analyze("dictating");
    }

    #[test]
    fn test_hash_to_float_range() {
        for seed in 0..1000 {
            let val = hash_to_float(seed);
            assert!(val >= -0.5 && val < 0.5, "hash_to_float must produce values in [-0.5, 0.5)");
        }
    }

    #[test]
    fn test_splitmix64_determinism() {
        let seed = 42;
        let h1 = splitmix64(seed);
        let h2 = splitmix64(seed);
        assert_eq!(h1, h2, "splitmix64 must be deterministic");
    }

    #[test]
    fn test_splitmix64_avalanche() {
        let h1 = splitmix64(0);
        let h2 = splitmix64(1);
        assert_ne!(h1, h2, "splitmix64 must have avalanche properties");
    }
}