# GeoSynthetic Vault (GSV) – Architecture Spec

Goal: Deterministic, ANN-free semantic vault on E8 roots (rune-hex) with stable hashing, typed payloads, and RUNE/TUI integration.

## Principles

- Deterministic everywhere: intent → hash → E8 address; pure functions, no RNG.
- ANN-free: exact matching / head overlap; optional metric distance, but no approximate indexes.
- Strong typing: serde-friendly structs; no dynamic JSON maps except at well-bounded payload edges.
- Bounded IO: fixed storage path, predictable sizes, streaming-friendly.
- Replayable: same inputs yield identical addresses, slots, and query order.

## Directory Layout (planned)

```
rune-gsv/
  Cargo.toml              # crate metadata (to add)
  src/
    lib.rs                # re-exports and feature gates
    errors.rs             # GsvError, type aliases
    addressing.rs         # intent → hash → E8 heads/tails (WeylSemanticAddress)
    slot.rs               # Slot structs: SemanticGraph, ExecutionTrace, Ranking, SlotMeta
    store.rs              # in-memory store + save/load orchestration
    persistence.rs        # RUNE/JSON read/write using rune-format encode/decode
    query.rs              # exact/similarity queries over addresses and rankings
    builtins.rs           # hydron evaluator built-ins (ASV.Store/Get/Query)
    integration.rs        # helpers to translate to/from rune-hex types (DomR, roots)
    prelude.rs            # convenient exports
  docs/
    ARCHITECTURE.md       # this spec
  examples/
    simple_store.rs       # demo: store/query a bundle
```

## Core Data Types

- `WeylSemanticAddress { heads: [u8; 8], digest: [u8; 32], context: Option<String> }`
  - Derived via SHA-256 of normalized intent text; heads = first 8 bytes mod 240.
  - Optional tails later for multi-head buckets.
- `SemanticGraph { nodes: Vec<Node>, frames: Vec<Frame> }`
  - `Node { id, kind, label, types, meta }` (meta is `HashMap<String, Value>` bounded in size)
  - `Frame { id, concept_id, roles: HashMap<String, Vec<String>>, scope }`
- `ExecutionTrace { id, effect_set_id, steps: Vec<Step> }` (Step is typed map of primitives)
- `RankingData { query_id, text, candidate_ids, elo_scores: HashMap<String, f32> }`
- `Slot { address: [u8; 8], intent: String, created_at: f64, semantic_graph, execution, ranking }`
- `VaultStats { total_intents, collisions, last_save }`

## Addressing

- Input: intent string → lowercased UTF-8 bytes → SHA-256 → hex digest stored.
- Heads: first 8 bytes `byte % 240` (aligns with rune-hex 240-root base lattice).
- Deterministic; no random salts. Collision handling: merge/overwrite policy in store.

## Storage & Persistence

- Primary persistence: on-disk RUNE/JSON via existing `rune-format` encode/decode; file path configurable (default `out/gsv.rune`).
- In-memory cache: `HashMap<String, Slot>` keyed by `heads` as comma-joined string, hydrated at startup from disk and flushed back on save.
- Save/Load: atomic write (temp file + rename) to avoid corruption; tolerate empty/missing file.

## Querying

- Exact recall: recompute address from intent; lookup by key.
- Similarity: count matching heads (0–8); optional tie-break by aggregated ELO score.
- No ANN; optional later: Hamming distance search over 8-byte head vector.

## Built-ins (hydron evaluator)

- `ASV.Store ~ [intent, bundle]` → returns address key.
- `ASV.Get ~ [intent]` → returns slot map.
- `ASV.Query ~ [intent, k]` → returns top-k `ContinuumHint` maps `{intent, address_heads, match_score, elo_score}`.
- Deterministic, pure functions; no hidden state beyond store contents.

## Integration with rune-hex / hydron

- Use `rune_hex::hex::DomR` and root tables when available to enrich slots (optional: project heads to nearest roots).
- Keep `no_std` potential by gating FS/serde under features; default uses std.
- Hydron evaluator: register built-ins behind the `hydron` feature in `rune-format`.

## Determinism & Safety

- Hashing: SHA-256 only; FNV/LCG reserved for future geocog worm compatibility, but not used here.
- Ordering: stable sort for query results (match_score desc, elo_score desc, then intent lexicographically).
- Bounds: cap slot counts and payload sizes (e.g., max nodes/frames/elo entries) to prevent blowup.
- Threading: initial version single-threaded; store protected by `RwLock` if shared.

## Missing Modules / Work Items

- New crate scaffolding `rune-gsv` (Cargo.toml, src/lib.rs, modules above).
- Addressing + Slot + Store + Persistence implementations.
- Hydron built-ins wiring + TUI REPL commands to call them.
- Tests: unit tests for addressing determinism, save/load round-trip, store/query behavior.
- Examples: simple end-to-end store/query demo.
- Bounds/config: configurable limits and storage path.

## Non-Goals (v1)

- ANN/vector search; no approximate indices.
- Distributed/clustered storage.
- Async I/O (use blocking std FS first; can add async feature later).
- ByteLex/embedding pipelines (can be layered later as a mapper crate).

## Suggested First Milestones

1) Scaffold crate + addressing + slot types + store + persistence + unit tests.
2) Expose built-ins; hook TUI commands (`asv store|get|query`).
3) Optional: rune-hex enrichment (DomR projection) and head-distance query.
