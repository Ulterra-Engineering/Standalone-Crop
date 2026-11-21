import unittest

from full_cutter_detector import select_center_cluster


class FullCutterDetectorSelectionTests(unittest.TestCase):
    def test_overlapping_multiscale_candidates_are_fused(self):
        selected = select_center_cluster(
            [
                {"candidate": "S640C1", "confidence": 0.8, "box": (40, 40, 60, 60)},
                {"candidate": "S1024C1", "confidence": 0.6, "box": (41, 41, 61, 61)},
            ],
            (100, 100),
        )
        self.assertEqual(selected["member_count"], 2)
        self.assertEqual(selected["members"], ("S640C1", "S1024C1"))
        self.assertAlmostEqual(selected["confidence"], 0.7)

    def test_center_cluster_beats_more_confident_edge_cluster(self):
        selected = select_center_cluster(
            [
                {"candidate": "edge", "confidence": 0.99, "box": (0, 0, 20, 20)},
                {"candidate": "center", "confidence": 0.30, "box": (40, 40, 60, 60)},
            ],
            (100, 100),
        )
        self.assertEqual(selected["members"], ("center",))

    def test_confidence_breaks_equal_center_distance_tie(self):
        selected = select_center_cluster(
            [
                {"candidate": "low", "confidence": 0.4, "box": (20, 40, 40, 60)},
                {"candidate": "high", "confidence": 0.8, "box": (60, 40, 80, 60)},
            ],
            (100, 100),
        )
        self.assertEqual(selected["members"], ("high",))

    def test_empty_candidates_return_none(self):
        self.assertIsNone(select_center_cluster([], (100, 100)))


if __name__ == "__main__":
    unittest.main()
