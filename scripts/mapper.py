#!/usr/bin/env python3
"""
E8 Root System Generator - Xypher Codex v2.2
============================================
Programmatically generates the complete 240-root E8 lattice with 100% accuracy.

© 2025 ArcMoon Studios • SPDX-License-Identifier MIT OR Apache-2.0 • Author: Lord Xyn
"""

import json
import csv
import struct
from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict
import numpy as np

# ========================================================================================
# CONSTANTS & CONFIGURATION
# ========================================================================================

EPSILON = 1e-6
EXPECTED_NORM_SQ = 2.0
EXPECTED_DEGREE = 56
TOTAL_ROOTS = 240

# Domain definitions (raw and normalized)
DOMAINS = {
    "Ethics": {
        "raw": [0.0, 0.5, 0.7, 0.8, 1.0, 0.0, 0.6, -0.2],
        "normalized": [0.0, 0.3133, 0.4386, 0.5013, 0.6266, 0.0, 0.3760, -0.1253]
    },
    "Psychology": {
        "raw": [0.7, 0.3, 0.6, 0.9, 0.0, 0.4, 0.3, 0.5],
        "normalized": [0.4700, 0.2014, 0.4028, 0.6042, 0.0, 0.2685, 0.2014, 0.3357]
    },
    "Technology": {
        "raw": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        "normalized": [0.4500, 0.4000, 0.3500, 0.3000, 0.2500, 0.2000, 0.1500, 0.1000]
    },
    "Relationships": {
        "raw": [-0.8, -0.4, 0.3, 0.5, 0.4, -0.2, 0.5, -0.6],
        "normalized": [-0.5714, -0.2857, 0.2143, 0.3571, 0.2857, -0.1429, 0.3571, -0.4286]
    },
    "Economics": {
        "raw": [0.6, 0.9, 0.7, 0.6, 0.2, 0.8, 0.3, 0.5],
        "normalized": [0.3409, 0.5114, 0.3977, 0.3409, 0.1136, 0.4545, 0.1705, 0.2841]
    },
    "Creativity": {
        "raw": [0.4, -0.7, 0.5, 0.3, 0.0, 0.6, 0.4, 0.8],
        "normalized": [0.2727, -0.4773, 0.3409, 0.2045, 0.0, 0.4091, 0.2727, 0.5455]
    },
    "Spirituality": {
        "raw": [-0.5, -0.6, 0.0, -0.8, 0.7, -0.7, 0.6, -0.4],
        "normalized": [-0.3012, -0.3614, 0.0, -0.4819, 0.4217, -0.4217, 0.3614, -0.2410]
    },
    "Physical": {
        "raw": [0.0, 0.7, -0.4, 0.4, 0.0, 0.8, 0.2, 0.6],
        "normalized": [0.0, 0.5150, -0.2943, 0.2943, 0.0, 0.5885, 0.1471, 0.4414]
    },
    "Existential": {
        "raw": [0.8, -0.3, 0.6, 0.0, 0.5, 0.0, 0.0, -0.4],
        "normalized": [0.6504, -0.2439, 0.4878, 0.0, 0.4065, 0.0, 0.0, -0.3252]
    },
    "Education": {
        "raw": [0.4, 0.6, 0.8, 0.9, 0.5, 0.5, 0.7, 0.6],
        "normalized": [0.2222, 0.3333, 0.4444, 0.5000, 0.2778, 0.2778, 0.3889, 0.3333]
    },
    "Health": {
        "raw": [0.3, 0.5, 0.5, 0.6, 0.3, 0.4, 0.6, 0.4],
        "normalized": [0.2308, 0.3846, 0.3846, 0.4615, 0.2308, 0.3077, 0.4615, 0.3077]
    },
    "Nature": {
        "raw": [0.2, 0.4, 0.6, 0.8, 0.6, 0.4, 0.2, 0.0],
        "normalized": [0.1250, 0.2500, 0.3750, 0.5000, 0.3750, 0.2500, 0.1250, 0.0]
    },
    "Politics": {
        "raw": [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6],
        "normalized": [0.5714, 0.4286, 0.2857, 0.1429, 0.0, -0.1429, -0.2857, -0.4286]
    },
    "Law": {
        "raw": [0.5, 0.7, 0.9, 0.3, 0.1, 0.0, -0.2, -0.4],
        "normalized": [0.3571, 0.5, 0.6429, 0.2143, 0.0714, 0.0, -0.1429, -0.2857]
    },
    "Science": {
        "raw": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        "normalized": [0.4500, 0.4000, 0.3500, 0.3000, 0.2500, 0.2000, 0.1500, 0.1000]
    },
    "Mathematics": {
        "raw": [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
        "normalized": [0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536]
    },
    "Art": {
        "raw": [0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7],
        "normalized": [0.4472, 0.3191, 0.1910, 0.0638, -0.0638, -0.1910, -0.3191, -0.4472]
    },
    "Culture": {
        "raw": [0.4, 0.6, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2],
        "normalized": [0.2673, 0.4009, 0.5345, 0.4009, 0.2673, 0.1336, 0.0, -0.1336]
    },
    "History": {
        "raw": [0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.3, 0.1],
        "normalized": [0.1826, 0.3043, 0.4261, 0.5478, 0.4261, 0.3043, 0.1826, 0.0609]
    },
    "Language": {
        "raw": [0.8, 0.4, 0.0, -0.4, -0.8, -0.4, 0.0, 0.4],
        "normalized": [0.5345, 0.2673, 0.0, -0.2673, -0.5345, -0.2673, 0.0, 0.2673]
    },
    "Religion": {
        "raw": [-0.6, -0.4, 0.0, 0.4, 0.6, 0.4, 0.0, -0.4],
        "normalized": [-0.4264, -0.2843, 0.0, 0.2843, 0.4264, 0.2843, 0.0, -0.2843]
    },
    "Media": {
        "raw": [0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9],
        "normalized": [0.2673, 0.1604, 0.0535, -0.0535, -0.1604, -0.2673, -0.3742, -0.4810]
    },
    "Sports": {
        "raw": [0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5],
        "normalized": [0.5345, 0.4155, 0.2966, 0.1776, 0.0590, -0.0590, -0.1776, -0.2966]
    },
    "Business": {
        "raw": [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6],
        "normalized": [0.5714, 0.4286, 0.2857, 0.1429, 0.0, -0.1429, -0.2857, -0.4286]
    },
    "Philosophy": {
        "raw": [0.2, 0.0, -0.2, -0.4, -0.6, -0.4, -0.2, 0.0],
        "normalized": [0.1826, 0.0, -0.1826, -0.3651, -0.5478, -0.3651, -0.1826, 0.0]
    },
    "Engineering": {
        "raw": [0.6, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4],
        "normalized": [0.4264, 0.5686, 0.4264, 0.2843, 0.1421, 0.0, -0.1421, -0.2843]
    },
    "Architecture": {
        "raw": [0.3, 0.5, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3],
        "normalized": [0.2673, 0.4455, 0.6236, 0.4455, 0.2673, 0.0891, -0.0891, -0.2673]
    },
    "Agriculture": {
        "raw": [0.4, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6],
        "normalized": [0.4009, 0.6014, 0.4009, 0.2005, 0.0, -0.2005, -0.4009, -0.6014]
    },
    "Transportation": {
        "raw": [0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7],
        "normalized": [0.4472, 0.3191, 0.1910, 0.0638, -0.0638, -0.1910, -0.3191, -0.4472]
    },
    "Energy": {
        "raw": [0.8, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4],
        "normalized": [0.4804, 0.4804, 0.3603, 0.2402, 0.1201, 0.0, -0.1201, -0.2402]
    },
    "Space": {
        "raw": [0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5],
        "normalized": [0.5345, 0.4155, 0.2966, 0.1776, 0.0590, -0.0590, -0.1776, -0.2966]
    },
    "Military": {
        "raw": [0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8],
        "normalized": [0.4264, 0.2843, 0.1421, 0.0, -0.1421, -0.2843, -0.4264, -0.5686]
    },
    "Fashion": {
        "raw": [0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9],
        "normalized": [0.2673, 0.1604, 0.0535, -0.0535, -0.1604, -0.2673, -0.3742, -0.4810]
    },
    "Music": {
        "raw": [0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0],
        "normalized": [0.2673, 0.1336, 0.0, -0.1336, -0.2673, -0.4009, -0.5345, -0.6682]
    },
    "Film": {
        "raw": [0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8],
        "normalized": [0.4264, 0.2843, 0.1421, 0.0, -0.1421, -0.2843, -0.4264, -0.5686]
    },
    "Gaming": {
        "raw": [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6],
        "normalized": [0.5714, 0.4286, 0.2857, 0.1429, 0.0, -0.1429, -0.2857, -0.4286]
    },
    "Finance": {
        "raw": [0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5],
        "normalized": [0.5345, 0.4155, 0.2966, 0.1776, 0.0590, -0.0590, -0.1776, -0.2966]
    },
    "Food": {
        "raw": [0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.3, 0.1],
        "normalized": [0.1826, 0.3043, 0.4261, 0.5478, 0.4261, 0.3043, 0.1826, 0.0609]
    }
}

# ========================================================================================
# DATA STRUCTURES
# ========================================================================================

@dataclass
class Root:
    """E8 root vector with metadata."""
    index: int
    type: str  # "TypeI" or "TypeII"
    coords: List[float]  # 8D vector
    
    def norm_sq(self) -> float:
        """Squared L2 norm."""
        return sum(x * x for x in self.coords)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.coords, dtype=np.float32)


@dataclass
class E8System:
    """Complete E8 root system with validation."""
    roots: List[Root]
    adjacency: Dict[int, List[int]]  # index -> [neighbor_indices]
    adjacency_mode: str = 'positive'
    
    def validate(self, adj_mode: str = 'positive') -> Dict[str, any]:
        """Run full validation suite."""
        errors = []
        
        # Check count
        if len(self.roots) != TOTAL_ROOTS:
            errors.append(f"Expected {TOTAL_ROOTS} roots, got {len(self.roots)}")
        
        # Check each root
        type_i_count = 0
        type_ii_count = 0
        
        for root in self.roots:
            # Norm check
            norm_sq = root.norm_sq()
            if abs(norm_sq - EXPECTED_NORM_SQ) > EPSILON:
                errors.append(f"Root {root.index} has invalid norm²: {norm_sq}")
            
            # Type counting
            if root.type == "TypeI":
                type_i_count += 1
            elif root.type == "TypeII":
                type_ii_count += 1
        
        if type_i_count != 112:
            errors.append(f"Expected 112 Type I roots, got {type_i_count}")
        if type_ii_count != 128:
            errors.append(f"Expected 128 Type II roots, got {type_ii_count}")
        
        # Check adjacency degrees
        # Determine expected degree based on adjacency mode
        expected_degree_by_mode = {
            'positive': 56,
            'absolute': 112
        }
        if adj_mode not in expected_degree_by_mode:
            errors.append(f"Unknown adjacency validation mode: {adj_mode}")
        else:
            expected_degree = expected_degree_by_mode[adj_mode]
            for idx, neighbors in self.adjacency.items():
                if len(neighbors) != expected_degree:
                    errors.append(f"Root {idx} has {len(neighbors)} neighbors, expected {expected_degree}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "type_i_count": type_i_count,
            "type_ii_count": type_ii_count,
            "total_roots": len(self.roots)
        }


# ========================================================================================
# CORE GENERATION ALGORITHMS
# ========================================================================================

def generate_type_i_roots() -> List[Root]:
    """
    Generate all 112 Type I roots: ±eᵢ ± eⱼ where i < j.
    
    Returns:
        List of 112 Type I Root objects in canonical order.
    """
    roots = []
    index = 0
    
    for i in range(8):
        for j in range(i + 1, 8):
            for sign_i in [1.0, -1.0]:
                for sign_j in [1.0, -1.0]:
                    coords = [0.0] * 8
                    coords[i] = sign_i
                    coords[j] = sign_j
                    
                    roots.append(Root(
                        index=index,
                        type="TypeI",
                        coords=coords
                    ))
                    index += 1
    
    assert len(roots) == 112, f"Type I generation failed: got {len(roots)} roots"
    return roots


def generate_type_ii_roots() -> List[Root]:
    """
    Generate all 128 Type II roots: ½(±e₁ ± e₂ ± ... ± e₈) with even # of minus signs.
    
    Returns:
        List of 128 Type II Root objects in canonical order.
    """
    roots = []
    index = 112  # Continue indexing from Type I
    
    for bits in range(256):
        # Check even parity (even number of 1-bits means even number of minus signs)
        if bin(bits).count('1') % 2 == 0:
            coords = []
            for i in range(8):
                if (bits >> i) & 1:
                    coords.append(-0.5)
                else:
                    coords.append(0.5)
            
            roots.append(Root(
                index=index,
                type="TypeII",
                coords=coords
            ))
            index += 1
    
    assert len(roots) == 128, f"Type II generation failed: got {len(roots)} roots"
    return roots


def generate_all_roots() -> List[Root]:
    """Generate complete 240-root E8 system."""
    type_i = generate_type_i_roots()
    type_ii = generate_type_ii_roots()
    return type_i + type_ii


# ========================================================================================
# ADJACENCY & GEOMETRY
# ========================================================================================

def inner_product(a: List[float], b: List[float]) -> float:
    """Euclidean inner product in ℝ⁸."""
    return sum(x * y for x, y in zip(a, b))


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance in ℝ⁸."""
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def build_adjacency_graph(roots: List[Root], mode: str = 'positive') -> Dict[int, List[int]]:
    """
    Build adjacency graph via kissing condition: ⟨α, β⟩ = 1.
    Only count neighbors with positive inner product (1), which yields degree 56.
    
    Returns:
        Dict mapping root index to list of neighbor indices.
    """
    adjacency = {i: [] for i in range(len(roots))}
    
    for i in range(len(roots)):
        for j in range(i + 1, len(roots)):
            ip = inner_product(roots[i].coords, roots[j].coords)

            if mode == 'positive':
                # Positive-only adjacency: ⟨α,β⟩ = +1
                if abs(ip - 1.0) < EPSILON:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
            elif mode == 'absolute':
                # Absolute adjacency: |⟨α,β⟩| = 1 (both +1 and -1)
                if abs(abs(ip) - 1.0) < EPSILON:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
            else:
                raise ValueError(f"Unknown adjacency mode: {mode}")
    
    return adjacency


# ========================================================================================
# DOMAIN UTILITIES
# ========================================================================================

def normalize_vector(v: List[float]) -> List[float]:
    """Normalize vector to unit length."""
    norm = sum(x * x for x in v) ** 0.5
    if norm < EPSILON:
        raise ValueError("Cannot normalize near-zero vector")
    return [x / norm for x in v]


def normalize_domains() -> None:
    """Ensure all canonical domains have exactly unit-length normalized vectors.

    This recalculates normalized vectors from the authoritative `raw` values to
    avoid relying on precomputed fields that may suffer minor rounding drift.
    """
    for name, data in DOMAINS.items():
        raw = data.get('raw')
        if raw is None:
            raise ValueError(f"Domain {name} has no 'raw' vector to normalize")
        data['normalized'] = normalize_vector(raw)


def compute_affinity(root: List[float], domain: List[float]) -> float:
    """
    Compute affinity score between root and domain vector.
    
    Args:
        root: 8D root vector
        domain: 8D unit-norm domain vector
    
    Returns:
        Affinity score ∈ [-1, 1]
    """
    # Compute affinity (cosine) using higher-precision math where possible.
    ip = inner_product(root, domain)
    root_norm = sum(x * x for x in root) ** 0.5
    domain_norm = sum(x * x for x in domain) ** 0.5
    if root_norm < EPSILON or domain_norm < EPSILON:
        raise ValueError("Cannot compute affinity with near-zero norm")
    affinity = ip / (root_norm * domain_norm)

    # If the affinity slightly exceeds the numerical limits because of
    # floating-point rounding (e.g., 1.0000000002), treat it as 1.0 within
    # EPSILON; otherwise, raise to indicate an unexpected error in data.
    if affinity > 1.0 + EPSILON or affinity < -1.0 - EPSILON:
        raise ValueError(f"Affinity out of bounds: {affinity}")
    if affinity > 1.0:
        affinity = 1.0
    if affinity < -1.0:
        affinity = -1.0
    return affinity


def project_to_domains(point: List[float]) -> Dict[str, float]:
    """Project an 8D point onto all canonical domains."""
    projections = {}
    for name, data in DOMAINS.items():
        domain_vec = data["normalized"]
        projections[name] = compute_affinity(point, domain_vec)
    return projections


# ========================================================================================
# EXPORT FUNCTIONS
# ========================================================================================

def export_to_json(system: E8System, filename: str):
    """Export E8 system to JSON."""
    data = {
        "metadata": {
            "version": "2.2.0",
            "total_roots": len(system.roots),
            "type_i_count": sum(1 for r in system.roots if r.type == "TypeI"),
            "type_ii_count": sum(1 for r in system.roots if r.type == "TypeII")
        ,
            "adjacency_mode": getattr(system, 'adjacency_mode', 'positive')
        },
        "roots": [asdict(r) for r in system.roots],
        "adjacency": {str(k): v for k, v in system.adjacency.items()},
        "domains": DOMAINS
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Exported to {filename}")


def export_to_csv(system: E8System, filename: str):
    """Export roots to CSV table."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Index', 'Type', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8'])
        
        # Rows
        for root in system.roots:
            writer.writerow([root.index, root.type] + root.coords)
    
    print(f"✓ Exported to {filename}")


def export_to_binary(system: E8System, filename: str):
    """Export roots to compact binary format."""
    with open(filename, 'wb') as f:
        # Magic header
        f.write(b'E8RT')
        
        # Version
        f.write(struct.pack('H', 2))  # v2
        
        # Root count
        f.write(struct.pack('H', len(system.roots)))
        
        # Roots (240 × 8 × 4 bytes = 7680 bytes)
        for root in system.roots:
            for coord in root.coords:
                f.write(struct.pack('f', coord))
    
    print(f"✓ Exported to {filename}")


def export_adjacency_to_csv(system: E8System, filename: str):
    """Export adjacency graph to CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Root_Index', 'Neighbor_Indices', 'Degree'])
        
        for idx in sorted(system.adjacency.keys()):
            neighbors = system.adjacency[idx]
            writer.writerow([idx, ' '.join(map(str, neighbors)), len(neighbors)])
    
    print(f"✓ Exported adjacency to {filename}")


# ========================================================================================
# ANALYSIS TOOLS
# ========================================================================================

def find_nearest_root(point: List[float], roots: List[Root]) -> Tuple[int, float]:
    """
    Find nearest root to a given point.
    
    Returns:
        (index, distance)
    """
    min_dist = float('inf')
    min_idx = -1
    
    for root in roots:
        dist = euclidean_distance(point, root.coords)
        if dist < min_dist:
            min_dist = dist
            min_idx = root.index
    
    return min_idx, min_dist


def compute_barycenter(root_weights: Dict[int, float], roots: List[Root]) -> List[float]:
    """
    Compute weighted barycenter of roots.
    
    Args:
        root_weights: {root_index: weight}
        roots: List of all roots
    
    Returns:
        8D barycentric coordinate
    """
    centroid = [0.0] * 8
    total_weight = sum(root_weights.values())
    
    for idx, weight in root_weights.items():
        for i in range(8):
            centroid[i] += (weight / total_weight) * roots[idx].coords[i]
    
    return centroid


# ========================================================================================
# MAIN EXECUTION
# ========================================================================================

def main():
    """Generate, validate, and export E8 root system."""
    print("=" * 80)
    print("E8 Root System Generator - Xypher Codex v2.2")
    print("=" * 80)
    print()
    
    # Generate roots
    print("Generating 240 E8 roots...")
    roots = generate_all_roots()
    print(f"✓ Generated {len(roots)} roots")
    print(f"  - Type I:  {sum(1 for r in roots if r.type == 'TypeI')} roots")
    print(f"  - Type II: {sum(1 for r in roots if r.type == 'TypeII')} roots")
    print()
    
    # Parse CLI options
    import argparse
    parser = argparse.ArgumentParser(description='E8 Root System Generator')
    parser.add_argument('--adjacency-mode', choices=['positive', 'absolute'], default='positive', help='Adjacency mode: positive (⟨α,β⟩=+1) or absolute (|⟨α,β⟩|=1)')
    parser.add_argument('--export-prefix', default='', help='Optional prefix for exported files')
    args = parser.parse_args()

    # Build adjacency
    print("Building adjacency graph (mode: {})...".format(args.adjacency_mode))
    adjacency = build_adjacency_graph(roots, mode=args.adjacency_mode)
    print("✓ Adjacency graph constructed")
    print()

    # Re-normalize canonical domains from raw input to ensure perfect math
    # and to avoid relying on precomputed 'normalized' vectors that can drift.
    normalize_domains()
    
    # Create system
    system = E8System(roots=roots, adjacency=adjacency, adjacency_mode=args.adjacency_mode)
    
    # Validate
    print("Validating system...")
    validation = system.validate(adj_mode=args.adjacency_mode)
    if validation["valid"]:
        print("✓ All validation checks passed")
    else:
        print("✗ Validation FAILED:")
        for error in validation["errors"]:
            print(f"  - {error}")
        return
    print()
    
    # Export
    print("Exporting data...")
    prefix = args.export_prefix
    export_to_json(system, prefix + "e8_roots.json")
    export_to_csv(system, prefix + "e8_roots.csv")
    export_to_binary(system, prefix + "e8_roots.bin")
    export_adjacency_to_csv(system, prefix + "e8_adjacency.csv")
    print()
    
    # Example usage
    print("Example: Projecting a point...")
    test_point = [0.7, 0.3, 0.6, 0.9, 0.0, 0.4, 0.3, 0.5]  # Psychology-like
    projections = project_to_domains(test_point)
    
    print("Domain affinities:")
    for domain, affinity in sorted(projections.items(), key=lambda x: -x[1]):
        print(f"  {domain:15s}: {affinity:+.4f}")
    print()
    
    nearest_idx, dist = find_nearest_root(test_point, roots)
    print(f"Nearest root: Index {nearest_idx} (distance: {dist:.4f})")
    print()
    
    print("=" * 80)
    print("Generation complete. Files created:")
    print("  - e8_roots.json (full data)")
    print("  - e8_roots.csv (root table)")
    print("  - e8_roots.bin (binary format)")
    print("  - e8_adjacency.csv (graph structure)")
    print("=" * 80)


if __name__ == "__main__":
    main()