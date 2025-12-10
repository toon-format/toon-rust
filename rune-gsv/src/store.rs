use crate::addressing::WeylSemanticAddress;
use crate::slot::SGLRuneSlot;
use crate::persistence;
use anyhow::Result;
use dirs;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use once_cell::sync::Lazy;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct QuantizedContinuum {
    pub storage_path: String,
    pub vault_name: String,
    pub slots: Arc<RwLock<HashMap<String, SGLRuneSlot>>>,
    /// normalized_intent -> key
    pub intent_index: Arc<RwLock<HashMap<String, String>>>,
    /// head buckets: head (0..239) -> keys
    pub head_buckets: Arc<RwLock<HashMap<u8, HashSet<String>>>>,
}

impl QuantizedContinuum {
    /// default constructor using `$HOME/.rasv/gsv.rune` vault
    pub fn default() -> Self {
        Self::with_name("gsv")
    }

    /// create a continuation that uses a named vault in the user's home .rasv directory
    pub fn with_name(vault_name: impl Into<String>) -> Self {
        let vn = vault_name.into();
        let mut path = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        path.push(".rasv");
        let fname = format!("{}.rune", vn);
        path.push(fname);
        Self::new(path.to_string_lossy().to_string(), vn)
    }

    /// create using explicit storage path and vault name
    pub fn new(storage_path: impl Into<String>, vault_name: impl Into<String>) -> Self {
        let path = storage_path.into();
        let vn = vault_name.into();
        let slots = Arc::new(RwLock::new(HashMap::new()));
        let intent_index = Arc::new(RwLock::new(HashMap::new()));
        let head_buckets = Arc::new(RwLock::new(HashMap::new()));
        let mut qc = Self { storage_path: path, vault_name: vn, slots, intent_index, head_buckets };
        // hydrate cache from disk at startup
        if let Ok(vec) = persistence::read_rune(&qc.storage_path) {
            let mut map = qc.slots.write().unwrap();
            let mut iidx = qc.intent_index.write().unwrap();
            let mut buckets = qc.head_buckets.write().unwrap();
            for s in vec {
                let key = s.address.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(",");
                if !s.intent.is_empty() { iidx.insert(normalize_intent(&s.intent), key.clone()); }
                for h in &s.address { buckets.entry(*h).or_default().insert(key.clone()); }
                map.insert(key, s);
            }
        }
        qc
    }

    fn now_ts() -> f64 {
        let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        dur.as_secs_f64()
    }

    pub fn storage_path_for_name(vault_name: &str) -> String {
        let mut path = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        path.push(".rasv");
        let fname = format!("{}.rune", vault_name);
        path.push(fname);
        path.to_string_lossy().to_string()
    }

    /// Store a slot under a given intent; compute address if missing.
    pub fn store(&mut self, intent: &str, mut bundle: SGLRuneSlot) -> Result<String> {
        if bundle.created_at == 0.0 { bundle.created_at = Self::now_ts(); }
        // compute address from intent if missing/empty
        if bundle.address.len() != 8 {
            let wa = WeylSemanticAddress::from_text_intent(intent)?;
            bundle.address = wa.heads.to_vec();
        }
        let key = slugify_addrs(&bundle.address);
        self.slots.write().unwrap().insert(key.clone(), bundle.clone());
        // update indices
        if !bundle.intent.is_empty() {
            self.intent_index.write().unwrap().insert(normalize_intent(&bundle.intent), key.clone());
        }
        for h in &bundle.address { self.head_buckets.write().unwrap().entry(*h).or_default().insert(key.clone()); }
        // persist to disk
        let all: Vec<SGLRuneSlot> = self.slots.read().unwrap().values().cloned().collect();
        persistence::write_rune_atomic(&self.storage_path, &all)?;
        Ok(key)
    }

    /// Retrieve a specific slot by intent - tries a normalized intent map first
    pub fn retrieve(&self, intent: &str) -> Option<SGLRuneSlot> {
        let normalized = normalize_intent(intent);
        if let Some(key) = self.intent_index.read().unwrap().get(&normalized) { return self.slots.read().unwrap().get(key).cloned(); }
        let addr = WeylSemanticAddress::from_text_intent(intent).ok()?;
        let key = slugify_addrs(&addr.heads.to_vec());
        self.slots.read().unwrap().get(&key).cloned()
    }

    /// Query similar intents using head buckets for candidate selection
    pub fn query_similar_intents(&self, intent: &str, k: usize) -> Vec<SGLRuneSlot> {
        let addr = WeylSemanticAddress::from_text_intent(intent).unwrap();
        let mut candidates: HashSet<String> = HashSet::new();
        {
            let buckets = self.head_buckets.read().unwrap();
            for h in &addr.heads {
                if let Some(keys) = buckets.get(h) {
                    for key in keys { candidates.insert(key.clone()); }
                }
            }
        }
        let mut hints: Vec<(usize, SGLRuneSlot)> = Vec::new();
        for key in candidates.iter() {
            if let Some(s) = self.slots.read().unwrap().get(key) {
                let sa_heads: [u8; 8] = s.address.clone().try_into().unwrap_or([0u8;8]);
                let sa = WeylSemanticAddress { heads: sa_heads, digest: [0u8; 32], context: None };
                let score = addr.matches(&sa);
                if score > 0 { hints.push((score, s.clone())); }
            }
        }
        hints.sort_by(|a,b| b.0.cmp(&a.0));
        hints.into_iter().take(k).map(|(_, s)| s).collect()
    }
}

static DEFAULT_STORE: Lazy<RwLock<QuantizedContinuum>> = Lazy::new(|| RwLock::new(QuantizedContinuum::default()));

/// Return a reference to the global default continuum store (lazily initialized)
pub fn default_store() -> &'static RwLock<QuantizedContinuum> {
    &DEFAULT_STORE
}

fn slugify_addrs(a: &Vec<u8>) -> String { a.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(",") }

fn normalize_intent(s: &str) -> String { s.trim().to_lowercase() }
