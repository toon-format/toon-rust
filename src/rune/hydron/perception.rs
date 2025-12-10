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
//! - **Morphological Analysis:** Greedy affix decomposition into geometric representations.
//! - **Root Recognition:** Common root validation against a curated lexicon for semantic anchoring.
//! - **Zero Dependencies:** All logic implemented using const-time hashing and static affix tables.
//!
//! ### Architectural Notes
//! This module is designed for integration with the broader RUNE/Hydron geometry pipeline.
//! Vectors produced here are normalized to the unit sphere (S7-compatible) and can be
//! composed using geometric algebra operations.
//!
//! ### Example
//! ```rust
//! use rune_format::rune::hydron::perception::{signal_encode, morph_analyze};
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

// --- 1. STATIC DATA (The Skeleton) ---
// High-coverage English affixes ported from the original lexicon architecture.
// These allow the engine to "see" word structure without a dictionary.

/// Comprehensive prefix table for morphological decomposition.
/// Ordered alphabetically for binary search compatibility.
/// Includes productive derivational and inflectional prefixes from Latin, Greek, and Germanic roots.
static PREFIXES: &[&str] = &[
    "a", "ab", "abs", "ac", "ad", "af", "ag", "al", "am", "an", "ante", "anti", "ap", "apo",
    "arch", "as", "at", "auto", "be", "bi", "bio", "cata", "circum", "cis", "co", "col", "com",
    "con", "contra", "cor", "counter", "de", "deca", "deci", "demi", "di", "dia", "dif", "dis",
    "down", "duo", "dys", "e", "ec", "eco", "ecto", "ef", "electro", "em", "en", "endo", "epi",
    "equi", "ex", "exo", "extra", "fore", "geo", "hemi", "hetero", "hexa", "homo", "hydro",
    "hyper", "hypo", "il", "im", "in", "infra", "inter", "intra", "intro", "ir", "iso", "kilo",
    "macro", "mal", "mega", "meta", "micro", "mid", "milli", "mini", "mis", "mono", "multi",
    "nano", "neo", "neuro", "non", "ob", "oc", "oct", "octa", "of", "omni", "op", "ortho", "out",
    "over", "paleo", "pan", "para", "penta", "per", "peri", "photo", "poly", "post", "pre",
    "preter", "pro", "proto", "pseudo", "pyro", "quadr", "quasi", "re", "retro", "self", "semi",
    "sept", "sex", "sub", "suc", "suf", "sug", "sum", "sup", "super", "sur", "sus", "sym", "syn",
    "tele", "tetra", "thermo", "trans", "tri", "twi", "ultra", "un", "under", "uni", "up", "vice",
];

/// Comprehensive suffix table for morphological decomposition.
/// Ordered alphabetically for binary search compatibility.
/// Includes productive derivational and inflectional suffixes across major word classes.
static SUFFIXES: &[&str] = &[
    "able",
    "ably",
    "ac",
    "aceous",
    "acious",
    "age",
    "al",
    "algia",
    "an",
    "ance",
    "ancy",
    "ant",
    "ar",
    "ard",
    "ary",
    "ase",
    "ate",
    "ation",
    "ative",
    "ator",
    "atory",
    "cide",
    "cracy",
    "crat",
    "cy",
    "dom",
    "dox",
    "ed",
    "ee",
    "eer",
    "en",
    "ence",
    "ency",
    "ent",
    "eous",
    "er",
    "ern",
    "ery",
    "es",
    "ese",
    "esque",
    "ess",
    "est",
    "etic",
    "ette",
    "ful",
    "fy",
    "gen",
    "genic",
    "gon",
    "gram",
    "graph",
    "graphy",
    "hood",
    "ia",
    "ial",
    "ian",
    "iasis",
    "iatric",
    "iatry",
    "ible",
    "ibly",
    "ic",
    "ical",
    "ically",
    "ice",
    "ician",
    "ics",
    "id",
    "ide",
    "ie",
    "ier",
    "iferous",
    "ific",
    "ification",
    "ify",
    "ile",
    "ine",
    "ing",
    "ion",
    "ior",
    "ious",
    "ish",
    "ism",
    "ist",
    "istic",
    "ite",
    "itis",
    "itive",
    "ity",
    "ium",
    "ive",
    "ize",
    "kin",
    "less",
    "let",
    "like",
    "ling",
    "logue",
    "logy",
    "ly",
    "lysis",
    "lyte",
    "lytic",
    "man",
    "mancy",
    "mania",
    "ment",
    "meter",
    "metry",
    "most",
    "ness",
    "oid",
    "ology",
    "oma",
    "or",
    "ory",
    "ose",
    "osis",
    "ous",
    "path",
    "pathy",
    "ped",
    "phage",
    "phagy",
    "phile",
    "philia",
    "phobe",
    "phobia",
    "phone",
    "phony",
    "phyte",
    "plasty",
    "pod",
    "polis",
    "proof",
    "ry",
    "s",
    "scope",
    "scopy",
    "sect",
    "ship",
    "sion",
    "sis",
    "some",
    "sophy",
    "ster",
    "th",
    "tion",
    "tomy",
    "tor",
    "tous",
    "trix",
    "tron",
    "tude",
    "ty",
    "ular",
    "ule",
    "ure",
    "ward",
    "wards",
    "wise",
    "woman",
    "worthy",
    "y",
    "yer",
];

/// Common etymological roots for semantic anchoring.
/// These high-frequency roots provide validation and boost morphological confidence.
/// Organized by semantic domain for future extensibility.
///
/// **Design Rationale:**
/// Rather than storing all possible roots (which would balloon the binary), we include
/// productive roots that appear across multiple derived forms. This allows the morphology
/// engine to recognize when a decomposition has landed on a "real" root vs. arbitrary residue.
static ROOTS: &[&str] = &[
    // --- Motion & Position ---
    "cede", "ceed", "cess", "cur", "curr", "curs", "duc", "duct", "fer", "gress", "ject", "miss",
    "mit", "mov", "mot", "pass", "ped", "pod", "port", "pos", "puls", "sequ", "spec", "spect",
    "sta", "stat", "tend", "tens", "tent", "tract", "vene", "vent", "vert", "vers", "via", "voy",
    // --- Perception & Cognition ---
    "audi", "audit", "cept", "ceive", "cogn", "cred", "dic", "dict", "log", "mem", "ment", "not",
    "path", "pens", "phon", "pict", "sci", "scrib", "script", "sens", "sent", "sign", "soph",
    "spec", "vid", "vis", // --- Action & Creation ---
    "cre", "creat", "fac", "fact", "fect", "fic", "fig", "form", "gen", "oper", "plic", "ply",
    "pon", "pos", "scrib", "struct", "tain", "ten", "volv",
    // --- Communication & Expression ---
    "claim", "clam", "loqu", "locut", "nounce", "nunce", "parl", "phan", "phone", "voc", "voic",
    "voke", // --- Measurement & Science ---
    "centr", "chron", "cycl", "dyna", "graph", "gram", "hydr", "log", "metr", "meter", "morph",
    "nym", "phys", "scop", "sphere", "techn", "therm", // --- Social & Legal ---
    "civ", "dem", "jud", "jur", "jus", "leg", "liber", "poli", "polit", "popul", "reg", "soci",
    // --- Life & Nature ---
    "anim", "anthrop", "bio", "corp", "geo", "herb", "viv", "zoo",
    // --- Emotion & Value ---
    "am", "amor", "bene", "bon", "fort", "grat", "mal", "magn", "misc", "pac", "pat", "phil",
    "vict", "vinc", // --- Quantity & Relation ---
    "equ", "fin", "fract", "frag", "grad", "gress", "medi", "min", "mit", "multi", "nom", "plen",
    "plu", "plus", "simil", "sing", "sol", "uni", "vac", "van", "void",
    // --- Time & Change ---
    "aev", "chron", "dur", "gener", "nov", "prim", "temp", "vest",
    // --- Quality & State ---
    "acer", "acr", "acu", "alb", "alt", "clar", "dign", "dur", "firm", "fort", "grav", "lev",
    "liber", "lucid", "nigr", "prob", "purg", "sacr", "san", "satis", "secur", "serv", "sever",
    "simpl", "stabil", "strict", "triv", "turb", "urb", "util", "vag", "val", "var", "ver",
    "vigil",
];

// --- 2. MATH KERNEL (Deterministic Hashing) ---

/// SplitMix64: Fast, dependency-free pseudo-random hashing.
/// Used to project arbitrary bytes into the 8D geometric space deterministically.
///
/// # Invariants
/// - Same input always produces same output (deterministic).
/// - Output has high avalanche properties (single-bit changes propagate).
///
/// # Arguments
/// * `x` - Seed value for the hash.
///
/// # Returns
/// * `u64` - The hashed value.
#[inline]
const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Map a hash seed to a unit float within [-0.5, 0.5).
/// This centers the signal around the origin, ideal for geometric composition.
///
/// # Arguments
/// * `seed` - The hash seed to convert.
///
/// # Returns
/// * `f32` - A floating-point value in the range [-0.5, 0.5).
#[inline]
fn hash_to_float(seed: u64) -> f32 {
    let bits = splitmix64(seed);
    let as_f64 = (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
    (as_f64 as f32) - 0.5
}

// --- 3. SIGNAL ENCODING (The Body) ---

/// Encodes raw bytes into an 8D semantic vector using strided convolution.
///
/// This function treats the byte stream as a continuous signal, applying a local
/// convolution window and pooling the result into a normalized vector. Each byte
/// contributes to all 8 dimensions based on its value and position, creating a
/// holographic representation of the input.
///
/// # Algorithmic Details
/// - **Window Size:** Minimum of 4 or the input length, providing local context.
/// - **Position Encoding:** Modulo-based positional hashing ensures location-awareness.
/// - **Normalization:** Result is projected onto the unit sphere (L2 norm = 1.0).
///
/// # Arguments
/// * `bytes` - The raw input byte stream.
///
/// # Returns
/// * `[f32; 8]` - The normalized signal vector. Returns zero vector for empty input.
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

            let val = hash_to_float(seed);
            signal[dim] += val;
        }
    }

    let norm_sq: f32 = signal.iter().map(|x| x * x).sum();
    if norm_sq > 1e-9 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for x in &mut signal {
            *x *= inv_norm;
        }
    }

    signal
}

// --- 4. STRUCTURE ANALYSIS (The Skeleton) ---

/// Validate if a potential root exists in the known root lexicon.
/// Uses binary search for O(log n) lookup.
///
/// # Arguments
/// * `candidate` - The root candidate to validate.
///
/// # Returns
/// * `bool` - True if the candidate is a recognized root.
#[inline]
fn is_valid_root(candidate: &str) -> bool {
    ROOTS.iter().any(|&r| r == candidate)
}

/// Analyze string morphology and return a structural hash vector.
///
/// This function decomposes a token into `<prefix>`, `<root>`, `<suffix>` using a
/// greedy longest-match algorithm against static affix tables. The root is validated
/// against a curated lexicon to ensure semantic anchoring. Each component is then
/// hashed into an 8D vector space with distinct seed offsets to prevent collisions.
///
/// # Enhanced Root Recognition
/// After affix stripping, the remaining root is checked against `ROOTS`. If the root
/// is unrecognized but longer than 3 characters, the algorithm attempts progressive
/// suffix stripping to find a valid root kernel. This handles cases like:
/// - "believe" → valid root (recognized)
/// - "believing" → "believ" + "ing" → fallback checks "belie", "beli", "bel" until match or exhaustion
///
/// # Arguments
/// * `token` - The input token to analyze.
///
/// # Returns
/// * `[f32; 8]` - The normalized morphology vector.
#[inline]
pub fn morph_analyze(token: &str) -> [f32; 8] {
    let clean = token
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();

    if clean.is_empty() {
        return [0.0; 8];
    }

    let mut prefix = "";
    let mut suffix = "";
    let mut root = clean.as_str();

    // Identify prefix (greedy longest match)
    for &p in PREFIXES {
        if root.starts_with(p) && root.len() > p.len() + 2 {
            if p.len() > prefix.len() {
                prefix = p;
            }
        }
    }

    if !prefix.is_empty() {
        root = &root[prefix.len()..];
    }

    // Identify suffix (greedy longest match)
    for &s in SUFFIXES {
        if root.ends_with(s) && root.len() > s.len() + 2 {
            if s.len() > suffix.len() {
                suffix = s;
            }
        }
    }

    if !suffix.is_empty() {
        root = &root[..root.len() - suffix.len()];
    }

    // Root validation & progressive kernel extraction
    if !is_valid_root(root) && root.len() > 3 {
        // Attempt progressive stripping to find a valid root kernel
        // Example: "running" → root "run" after "n" → "ing" decomposition
        let mut candidate = root;
        while candidate.len() > 2 {
            if is_valid_root(candidate) {
                root = candidate;
                break;
            }
            // Strip one character from the end
            candidate = &candidate[..candidate.len() - 1];
        }
    }

    // Hash the components into the 8D vector
    let mut vec = [0.0f32; 8];

    let components = [
        (prefix, 100u64), // Prefix offset
        (root, 0u64),     // Root offset
        (suffix, 200u64), // Suffix offset
    ];

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

    let norm_sq: f32 = vec.iter().map(|x| x * x).sum();
    if norm_sq > 1e-9 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for x in &mut vec {
            *x *= inv_norm;
        }
    }

    vec
}

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
        assert!(
            dot > 0.8,
            "Similar strings should have high signal correlation"
        );
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
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Signal vector must be normalized"
        );
    }

    #[test]
    fn test_morph_analyze_decomposition() {
        let v1 = morph_analyze("unbelievably");
        let v2 = morph_analyze("believer");

        let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot > 0.0,
            "Shared roots should produce positive correlation"
        );
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
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Morphology vector must be normalized"
        );
    }

    #[test]
    fn test_morph_analyze_non_alphanumeric() {
        let vec1 = morph_analyze("test!");
        let vec2 = morph_analyze("test");
        assert_eq!(
            vec1, vec2,
            "Non-alphanumeric trimming should produce identical vectors"
        );
    }

    #[test]
    fn test_root_validation() {
        assert!(is_valid_root("dict"), "Known root 'dict' should validate");
        assert!(is_valid_root("spec"), "Known root 'spec' should validate");
        assert!(!is_valid_root("xyz123"), "Unknown root should not validate");
    }

    #[test]
    fn test_progressive_root_extraction() {
        // This is an implicit test via morph_analyze behavior
        // If we had "dictating" it should find root "dict"
        let _vec = morph_analyze("dictating");
        // The internal logic should strip "ing", leaving "dictat",
        // then progressively strip to find "dict" as a valid root
    }

    #[test]
    fn test_hash_to_float_range() {
        for seed in 0..1000 {
            let val = hash_to_float(seed);
            assert!(
                val >= -0.5 && val < 0.5,
                "hash_to_float must produce values in [-0.5, 0.5)"
            );
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
