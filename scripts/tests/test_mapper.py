import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from mapper import generate_all_roots, build_adjacency_graph, E8System, compute_affinity

class MapperTests(unittest.TestCase):
    def test_positive_adjacency_degree(self):
        roots = generate_all_roots()
        adj = build_adjacency_graph(roots, mode='positive')
        # Expect each root to have 56 neighbors in positive mode
        for i, neigh in adj.items():
            self.assertEqual(len(neigh), 56, f"Root {i} has {len(neigh)} neighbors, expected 56")

    def test_absolute_adjacency_degree(self):
        roots = generate_all_roots()
        adj = build_adjacency_graph(roots, mode='absolute')
        # Expect each root to have 112 neighbors in absolute mode
        for i, neigh in adj.items():
            self.assertEqual(len(neigh), 112, f"Root {i} has {len(neigh)} neighbors, expected 112")

    def test_affinity_clamping_and_normalization(self):
        roots = generate_all_roots()
        # Use the test point from the script to verify affinities range within [-1,1]
        test_point = [0.7, 0.3, 0.6, 0.9, 0.0, 0.4, 0.3, 0.5]
        # Ensure compute_affinity returns values within [-1,1]
        # We'll normalize a domain vector (unit-norm) from the DOMAINS values in the module
        from mapper import DOMAINS, normalize_vector
        d = DOMAINS['Psychology']['normalized']
        norm = normalize_vector(d)
        affinity = compute_affinity(test_point, norm)
        self.assertGreaterEqual(affinity, -1.0)
        self.assertLessEqual(affinity, 1.0)

if __name__ == '__main__':
    unittest.main()
