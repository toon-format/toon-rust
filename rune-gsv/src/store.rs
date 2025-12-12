use crate::addressing::{AddressingKind, WeylSemanticAddress};
use crate::builtins::*;
use crate::persistence::{AppendLog, MappedStore};
use crate::query::QueryResult;
use crate::slot::SGLRuneSlot;
use dirs;
use once_cell::sync::Lazy;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use yoshi::error::Result;

#[derive(Debug)]
pub struct QuantizedContinuum {
    pub storage_path: PathBuf,
    pub vault_name: String,
    pub intent_index: HashMap<String, u64>,       // normalized_intent -> offset
    pub head_buckets: HashMap<u8, Vec<u64>>,      // head byte -> offsets
    pub log: AppendLog,
    pub mmap: MappedStore,
}

impl QuantizedContinuum {
    pub fn default() -> Self {
        Self::with_name("gsv")
    }

    pub fn with_name(vault_name: impl Into<String>) -> Self {
        let vn = vault_name.into();
        let mut path = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        path.push(".rasv");
        path.push(format!("{}.rune", vn));
        Self::new(path, vn).expect("init QuantizedContinuum")
    }

    pub fn new(storage_path: impl Into<PathBuf>, vault_name: impl Into<String>) -> Result<Self> {
        let storage_path = storage_path.into();
        let log = AppendLog::open(&storage_path.parent().unwrap_or(&storage_path))?;
        let mmap = MappedStore::open(&storage_path.parent().unwrap_or(&storage_path))?;
        let mut intent_index = HashMap::new();
        let mut head_buckets: HashMap<u8, Vec<u64>> = HashMap::new();

        for (offset, slot) in mmap.iter() {
            intent_index.insert(normalize_intent(&slot.intent), offset);
            for h in &slot.address {
                head_buckets.entry(*h).or_default().push(offset);
            }
        }

        Ok(Self {
            storage_path,
            vault_name: vault_name.into(),
            intent_index,
            head_buckets,
            log,
            mmap,
        })
    }

    fn now_ts() -> f64 {
        let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        dur.as_secs_f64()
    }

    pub fn store(&mut self, intent: &str, mut bundle: SGLRuneSlot) -> Result<String> {
        if bundle.created_at == 0.0 {
            bundle.created_at = Self::now_ts();
        }
        if bundle.address.len() != 8 {
            let wa = WeylSemanticAddress::from_text_intent(intent)?;
            bundle.address = wa.heads.to_vec();
        }
        let offset = self.log.append(&bundle)?;
        self.intent_index
            .insert(normalize_intent(&bundle.intent), offset);
        for h in &bundle.address {
            self.head_buckets.entry(*h).or_default().push(offset);
        }
        self.mmap = MappedStore::open(&self.storage_path.parent().unwrap_or(&self.storage_path))?;
        Ok(format!("{}", offset))
    }

    pub fn retrieve(&self, intent: &str) -> Option<SGLRuneSlot> {
        let normalized = normalize_intent(intent);
        if let Some(off) = self.intent_index.get(&normalized) {
            return self.mmap.get_at(*off).ok();
        }
        let addr = WeylSemanticAddress::from_text_intent(intent).ok()?;
        for off in self.head_buckets.get(&addr.heads[0]).cloned().unwrap_or_default() {
            if let Ok(slot) = self.mmap.get_at(off) {
                if slot.address == addr.heads {
                    return Some(slot);
                }
            }
        }
        None
    }

    pub fn query_similar_intents(&self, intent: &str, k: usize) -> Vec<SGLRuneSlot> {
        let addr = match WeylSemanticAddress::from_text_intent(intent) {
            Ok(a) => a,
            Err(_) => return vec![],
        };
        let mut candidates: HashSet<u64> = HashSet::new();
        for h in &addr.heads {
            if let Some(offsets) = self.head_buckets.get(h) {
                for off in offsets {
                    candidates.insert(*off);
                }
            }
        }
        let mut hints: Vec<(usize, SGLRuneSlot)> = Vec::new();
        for off in candidates {
            if let Ok(slot) = self.mmap.get_at(off) {
                let sa_heads: [u8; 8] = slot.address.clone().try_into().unwrap_or([0u8; 8]);
                let sa = WeylSemanticAddress {
                    heads: sa_heads,
                    digest: [0u8; 32],
                    context: None,
                };
                let score = addr.matches(&sa);
                if score > 0 {
                    hints.push((score, slot));
                }
            }
        }
        hints.sort_by(|a, b| b.0.cmp(&a.0));
        hints.into_iter().take(k).map(|(_, s)| s).collect()
    }
}

static DEFAULT_STORE: Lazy<RwLock<QuantizedContinuum>> =
    Lazy::new(|| RwLock::new(QuantizedContinuum::default()));

pub fn default_store() -> &'static RwLock<QuantizedContinuum> {
    &DEFAULT_STORE
}

fn normalize_intent(s: &str) -> String {
    s.trim().to_lowercase()
}
