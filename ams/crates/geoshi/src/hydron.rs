//! Hydrogen mathematics and morphological cognition
//
//! # HYDRON MODULE
//!▫~•◦------------------------------------------------‣
//
//! Advanced hydrogen mathematics implementation for morphological cognition,
//! resonance modeling, and linguistic analysis through quantum-inspired
//! geometric transformations and morphological decomposition.
//!
//! ### Key Capabilities
//! - **Hydrogen Mathematics:** Quantum-inspired mathematical structures.
//! - **Morphological Analysis:** Linguistic decomposition into roots, prefixes, suffixes.
//! - **Signal Encoding:** Geometric encoding of linguistic tokens into E8 space.
//! - **Resonance Modeling:** Quantum harmonic analysis for cognitive patterns.
//! - **Linguistic Cognition:** Natural language processing through geometric algebra.
//!
//! ### Technical Features
//! - **Morpheme Database:** Comprehensive prefix, suffix, and root inventories.
//! - **Geometric Hashing:** 8D E8 lattice embedding of linguistic features.
//! - **Perceptual Encoding:** Signal-to-geometric transformation algorithms.
//! - **Quantum Resonance:** Frequency domain analysis in cognitive manifolds.
//!
//! ### Usage Patterns
//! ```rust
//! use geoshi::hydron::{morph_analyze, signal_encode};
//!
//! let morphological_features = morph_analyze("geometric");
//! let signal_embedding = signal_encode(b"cognitive pattern");
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use lazy_static::lazy_static;
use ndarray::Array1;
use std::collections::HashSet;

lazy_static! {
    static ref PREFIXES: HashSet<&'static str> = [
        "a", "ab", "abs", "ac", "ad", "af", "ag", "al", "am", "an", "ante", "anti", "ap", "apo",
        "arch", "as", "at", "auto", "be", "bi", "bio", "cata", "circum", "cis", "co", "col", "com",
        "con", "contra", "cor", "counter", "de", "deca", "deci", "demi", "di", "dia", "dif", "dis",
        "down", "duo", "dys", "e", "ec", "eco", "ecto", "ef", "electro", "em", "en", "endo", "epi",
        "equi", "ex", "exo", "extra", "fore", "geo", "hemi", "hetero", "hexa", "homo", "hydro",
        "hyper", "hypo", "il", "im", "in", "infra", "inter", "intra", "intro", "ir", "iso", "kilo",
        "macro", "mal", "mega", "meta", "micro", "mid", "milli", "mini", "mis", "mono", "multi",
        "nano", "neo", "neuro", "non", "ob", "oc", "oct", "octa", "of", "omni", "op", "ortho",
        "out", "over", "paleo", "pan", "para", "penta", "per", "peri", "photo", "poly", "post",
        "pre", "preter", "pro", "proto", "pseudo", "pyro", "quadr", "quasi", "re", "retro", "self",
        "semi", "sept", "sex", "sub", "suc", "suf", "sug", "sum", "sup", "super", "sur", "sus",
        "sym", "syn", "tele", "tetra", "thermo", "trans", "tri", "twi", "ultra", "un", "under",
        "uni", "up", "vice"
    ]
    .into_iter()
    .collect();
    static ref SUFFIXES: HashSet<&'static str> = [
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
        "yer"
    ]
    .into_iter()
    .collect();
    static ref ROOTS: HashSet<&'static str> = [
        "cede", "ceed", "cess", "cur", "curr", "curs", "duc", "duct", "fer", "gress", "ject",
        "miss", "mit", "mov", "mot", "pass", "ped", "pod", "port", "pos", "puls", "sequ", "spec",
        "spect", "sta", "stat", "tend", "tens", "tent", "tract", "vene", "vent", "vert", "vers",
        "via", "voy", "audi", "audit", "cept", "ceive", "cogn", "cred", "dic", "dict", "log",
        "mem", "ment", "not", "path", "pens", "phon", "pict", "sci", "scrib", "script", "sens",
        "sent", "sign", "soph", "spec", "vid", "vis", "cre", "creat", "fac", "fact", "fect", "fic",
        "fig", "form", "gen", "oper", "plic", "ply", "pon", "pos", "scrib", "struct", "tain",
        "ten", "volv", "claim", "clam", "loqu", "locut", "nounce", "nunce", "parl", "phan",
        "phone", "voc", "voic", "voke", "centr", "chron", "cycl", "dyna", "graph", "gram", "hydr",
        "log", "metr", "meter", "morph", "nym", "phys", "scop", "sphere", "techn", "therm", "civ",
        "dem", "jud", "jur", "jus", "leg", "liber", "poli", "polit", "popul", "reg", "soci",
        "anim", "anthrop", "bio", "corp", "geo", "herb", "viv", "zoo", "am", "amor", "bene", "bon",
        "fort", "grat", "mal", "magn", "misc", "pac", "pat", "phil", "vict", "vinc", "equ", "fin",
        "fract", "frag", "grad", "gress", "medi", "min", "mit", "multi", "nom", "plen", "plu",
        "plus", "simil", "sing", "sol", "uni", "vac", "van", "void", "aev", "chron", "dur",
        "gener", "nov", "prim", "temp", "vest", "acer", "acr", "acu", "alb", "alt", "clar", "dign",
        "dur", "firm", "fort", "grav", "lev", "liber", "lucid", "nigr", "prob", "purg", "san",
        "satis", "secur", "serv", "sever", "simpl", "stabil", "strict", "triv", "turb", "urb",
        "util", "vag", "val", "var", "ver", "vigil"
    ]
    .into_iter()
    .collect();
}

pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

pub fn hash_to_float(seed: u64) -> f64 {
    let bits = splitmix64(seed);
    let as_f64 = (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
    as_f64 - 0.5
}

pub fn signal_encode(byte_seq: &[u8]) -> Array1<f64> {
    if byte_seq.is_empty() {
        return Array1::zeros(8);
    }
    let mut signal: Array1<f64> = Array1::zeros(8);
    let window = std::cmp::min(4, byte_seq.len());
    for (i, &b) in byte_seq.iter().enumerate() {
        for dim in 0..8 {
            let seed = (b as u64 * 31) + dim as u64 + (i % window) as u64 * 1024;
            signal[dim] += hash_to_float(seed);
        }
    }
    let norm = signal.dot(&signal).sqrt();
    if norm > 1e-9 {
        signal /= norm;
    }
    signal
}

pub fn is_valid_root(candidate: &str) -> bool {
    ROOTS.contains(candidate)
}

pub fn morph_analyze(token: &str) -> Array1<f64> {
    let clean: String = token
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase();

    if clean.is_empty() {
        return Array1::zeros(8);
    }

    let mut prefix = "";
    let mut suffix = "";
    let mut root = clean.as_str();

    for &p in PREFIXES.iter() {
        if root.starts_with(p) && root.len() > p.len() + 2 && p.len() > prefix.len() {
            prefix = p;
        }
    }
    if !prefix.is_empty() {
        root = &root[prefix.len()..];
    }

    for &s in SUFFIXES.iter() {
        if root.ends_with(s) && root.len() > s.len() + 2 && s.len() > suffix.len() {
            suffix = s;
        }
    }
    if !suffix.is_empty() {
        root = &root[..root.len() - suffix.len()];
    }

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

    let mut vec: Array1<f64> = Array1::zeros(8);
    for (part, offset) in [(prefix, 100), (root, 0), (suffix, 200)] {
        if part.is_empty() {
            continue;
        }
        for (i, ch) in part.as_bytes().iter().enumerate() {
            for dim in 0..8 {
                let seed = *ch as u64 + dim as u64 + i as u64 * 31 + offset as u64;
                vec[dim] += hash_to_float(seed);
            }
        }
    }
    let norm = vec.dot(&vec).sqrt();
    if norm > 1e-9 {
        vec /= norm;
    }
    vec
}
