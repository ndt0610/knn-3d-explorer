import importlib.util
import unittest
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    "knn", Path(__file__).resolve().parents[1] / "knn_3d_manual_app.py"
)
knn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(knn)


class KnnTests(unittest.TestCase):
    def test_majority_of_nearest_neighbors(self):
        label, votes, ordered = knn.classify_knn(
            np.array([[0, 0, 0], [1, 0, 0]]),
            np.array([[4, 0, 0], [5, 0, 0]]),
            np.array([0, 0, 0]),
            k=3,
        )
        self.assertEqual(label, 1)
        self.assertEqual(votes, {1: 2, 2: 1})
        self.assertEqual([point["dist"] for point in ordered], [0, 1, 4, 5])

    def test_tie_uses_closest_point(self):
        label, _, _ = knn.classify_knn([[2, 0, 0]], [[1, 0, 0]], np.array([0, 0, 0]), 2)
        self.assertEqual(label, 2)

    def test_rejects_invalid_k(self):
        with self.assertRaises(ValueError):
            knn.classify_knn([[0, 0, 0]], [[1, 0, 0]], np.array([0, 0, 0]), 3)


if __name__ == "__main__":
    unittest.main()
