from __future__ import annotations

import unittest

from coaching.evaluate_visual_evidence import evaluate_visual_evidence


class EvaluateVisualEvidenceTests(unittest.TestCase):
    def test_evaluates_boolean_visual_labels(self) -> None:
        labels = [
            {
                "run_id": "run-1",
                "center_event_index": 4,
                "labels": {
                    "target_visible": True,
                    "crosshair_near_target": False,
                    "player_exposed": True,
                },
            }
        ]
        predictions = [
            {
                "run_id": "run-1",
                "center_event_index": 4,
                "coaching": {
                    "visual_evidence": {
                        "target_visible": True,
                        "crosshair_near_target": True,
                        "player_exposed": True,
                    }
                },
            }
        ]

        metrics = evaluate_visual_evidence(labels, predictions)

        self.assertEqual(metrics["matched_prediction_rows"], 1)
        self.assertEqual(metrics["total_compared_labels"], 3)
        self.assertAlmostEqual(metrics["micro_accuracy"], 2 / 3)
        self.assertEqual(metrics["per_label"]["target_visible"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
