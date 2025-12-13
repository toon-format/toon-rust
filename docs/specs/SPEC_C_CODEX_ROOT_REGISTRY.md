# Xypher Codex — Codex Root Registry (CRR) v1.0
**Subtitle:** Canonical Deterministic Mapping of 240 E8 Roots to Semantic Tiers
**Status:** Frozen / Canonical
**Date:** 2025-12-13

---

## 0. Preface

This specification is the **spine** of the Xypher Codex. It establishes the authoritative, immutable mapping between the 240 E8 root vectors and their role in the semantic hierarchy.

**Invariant:**
- **Total Roots:** 240 (Exact)
- **Hierarchy:** 8 Taproots → 16 Laterals → 24 Tertiaries → 192 Crosses.
- **Assignment:** Deterministic, based on E8 geometry and fixed indices.

---

## 1. Domain Structure

The Codex is partitioned into **8 Domains** (D0–D7), each containing exactly **30 roots**.

**Domain Index:** `d ∈ [0..7]`
**Root Index Range:** `root_id ∈ [d*30 .. d*30 + 29]`

Within each Domain (30 slots), reserved positions are fixed:

| Pos (0-29) | Tier       | Count | Notes |
|:----------:|:----------:|:-----:|:------|
| **14**     | Taproot    | 1     | The domain center (Tier 0) |
| **4, 24**  | Lateral    | 2     | Polarity/Mode switches (Tier 1) |
| **8, 18, 22** | Tertiary | 3   | Composed primitives (Tier 2) |
| *Others*   | Cross      | 24    | High-mix fusion states (Tier 3) |

---

## 2. E8 Geometric Basis (Canonical)

The 240 E8 roots are generated via the standard realization (Type I + Type II), sorted deterministically.

### 2.1 Canonical Generation Order
1.  **Type I Roots (112):** `±e_i ± e_j` ($i < j$).
    *   Sorted lexicographically by $(i, j, s_i, s_j)$.
2.  **Type II Roots (128):** $\frac{1}{2}(\pm 1, \dots, \pm 1)$ (even parity of minus signs).
    *   Sorted lexicographically by sign bitmask (0..255).

### 2.2 Canonical Simple Roots (The 8 Taproots)
We fix the simple root basis $\{\alpha_1 \dots \alpha_8\}$:
- $\alpha_1 = e_1 - e_2$
- $\alpha_2 = e_2 - e_3$
- $\dots$
- $\alpha_7 = e_7 - e_8$
- $\alpha_8 = \frac{1}{2}(e_1 + e_2 + e_3 + e_4 - e_5 - e_6 - e_7 - e_8)$

---

## 3. Assignment Algorithm (Deterministic)

### 3.1 Taproot Assignment (Tier 0)
For each domain $d \in [0..7]$:
- `root_id = d * 30 + 14`
- Vector: $\alpha_{d+1}$ (using 1-based index for simple roots)

### 3.2 Lateral Assignment (Tier 1)
For each domain $d$:
- Pos 4: `Lateral A` (+). Vector: Nearest distinct root to $+\alpha_{d+1}$ (excluding Taproot).
- Pos 24: `Lateral B` (-). Vector: Nearest distinct root to $-\alpha_{d+1}$.

### 3.3 Tertiary Assignment (Tier 2)
For each domain $d$, 3 slots (Pos 8, 18, 22) are assigned derived vectors based on Dynkin neighbors of $\alpha_{d+1}$.
- Vectors are `snap_to_E8(taproot + neighbor)`.
- Collisions with existing assigned roots are resolved by picking the next nearest distinct root.

### 3.4 Cross Assignment (Tier 3)
For each domain $d$, the remaining 24 slots are filled with the remaining unassigned E8 roots.
- **Scoring:** Roots are scored by proximity (dot product) to the domain's Taproot vector.
- **Selection:** The top 24 unassigned roots closest to Taproot $d$ are assigned to domain $d$.
- **Sorting:** Assigned to the 24 Cross slots (in index order) based on score descending.

---

## 4. Root Registry Table Schema

The implementation must provide an O(1) lookup table or function:

```rust
pub struct CodexRoot {
    pub id: u16,           // 0..239
    pub vector: [f32; 8],  // E8 vector
    pub tier: Tier,        // Taproot, Lateral, Tertiary, Cross
    pub domain: u8,        // 0..7
    pub family_mask: u32,  // Bitmask of allowed Operator Families
}
```

---

## 5. Validation Golden Hash

To ensure determinism, the generated registry must match a specific hash (SHA-256 of the sorted root vectors).
*(Hash to be established upon first canonical generation implementation)*.
