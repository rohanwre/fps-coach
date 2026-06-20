"""Evaluate VLM evidence against manually labeled event windows."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


LABEL_FIELDS = ("target_visible", "crosshair_near_target", "player_exposed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate visual coaching evidence labels.")
    parser.add_argument("--labels", required=True, help="Labeled event-window JSONL path.")
    parser.add_argument("--predictions", required=True, help="Visual coaching JSONL path.")
    parser.add_argument("--output", required=True, help="Metrics JSON output path.")
    return parser.parse_args()


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return row.get("run_id"), row.get("center_event_index")


def evaluate_visual_evidence(
    labeled_windows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    predictions = {row_key(row): row for row in prediction_rows}
    per_label: dict[str, dict[str, float | int]] = {}
    total_correct = 0
    total_compared = 0

    for field in LABEL_FIELDS:
        true_positive = false_positive = false_negative = correct = compared = 0
        for window in labeled_windows:
            expected = window.get("labels", {}).get(field)
            prediction = predictions.get(row_key(window), {})
            actual = prediction.get("coaching", {}).get("visual_evidence", {}).get(field)
            if not isinstance(expected, bool) or not isinstance(actual, bool):
                continue
            compared += 1
            correct += int(actual == expected)
            true_positive += int(actual and expected)
            false_positive += int(actual and not expected)
            false_negative += int(not actual and expected)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        per_label[field] = {
            "compared": compared,
            "accuracy": correct / compared if compared else 0.0,
            "precision": precision,
            "recall": recall,
        }
        total_correct += correct
        total_compared += compared

    return {
        "matched_prediction_rows": sum(row_key(window) in predictions for window in labeled_windows),
        "labeled_window_count": len(labeled_windows),
        "total_compared_labels": total_compared,
        "micro_accuracy": total_correct / total_compared if total_compared else 0.0,
        "per_label": per_label,
    }


def main() -> None:
    args = parse_args()
    metrics = evaluate_visual_evidence(
        load_jsonl(pathlib.Path(args.labels)),
        load_jsonl(pathlib.Path(args.predictions)),
    )
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote visual evidence metrics to {output_path}")


if __name__ == "__main__":
    main()
