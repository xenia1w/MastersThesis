import math
import unittest

import torch

from src.features.incremental_embeddings import (
    running_centroids,
    select_k,
    stability_point,
)


class TestIncrementalEmbeddings(unittest.TestCase):
    def test_running_centroids(self) -> None:
        emb1 = torch.tensor([1.0, 0.0])
        emb2 = torch.tensor([0.0, 1.0])
        centroids = running_centroids([emb1, emb2])
        self.assertEqual(len(centroids), 2)
        self.assertTrue(torch.allclose(centroids[0], emb1))
        expected = torch.tensor([math.sqrt(0.5), math.sqrt(0.5)])
        self.assertTrue(torch.allclose(centroids[1], expected, atol=1e-4))

    def test_select_k(self) -> None:
        embs = [torch.ones(2) for _ in range(3)]
        centroids = running_centroids(embs)
        selected = select_k(centroids, [1, 3, 5])
        self.assertEqual(set(selected.keys()), {1, 3})

    def test_stability_point(self) -> None:
        cosines = [0.8, 0.9, 0.951, 0.952, 0.9525]
        stable_k = stability_point(cosines)
        self.assertEqual(stable_k, 3)


if __name__ == "__main__":
    unittest.main()
