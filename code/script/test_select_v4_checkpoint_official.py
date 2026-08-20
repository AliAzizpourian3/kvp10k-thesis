"""Safe CPU-only tests for V4 official checkpoint selection helpers."""

import tempfile
import unittest
from pathlib import Path

import torch

from kvp10k_official_eval import evaluate_table
from select_v4_checkpoint_official import (
    CheckpointCandidate,
    discover_unique_checkpoints,
    finalize_entity_metrics,
    new_entity_counts,
    select_best_validation_result,
    update_entity_counts,
)


class EntityMetricTests(unittest.TestCase):
    def test_key_value_confusion_is_penalized(self):
        labels = torch.tensor([[1, 2]])
        predictions = torch.tensor([[1, 1]])
        attention = torch.ones_like(labels)
        counts = new_entity_counts()
        update_entity_counts(counts, predictions, labels, attention)
        metrics = finalize_entity_metrics(counts)

        self.assertEqual(metrics["legacy_buggy"]["f1"], 1.0)
        self.assertEqual(metrics["corrected_micro"]["precision"], 0.5)
        self.assertEqual(metrics["corrected_micro"]["recall"], 0.5)
        self.assertEqual(metrics["corrected_micro"]["f1"], 0.5)
        self.assertEqual(metrics["per_class"]["key"]["fp"], 1)
        self.assertEqual(metrics["per_class"]["value"]["fn"], 1)

    def test_padding_is_ignored(self):
        labels = torch.tensor([[1, 2, 2]])
        predictions = torch.tensor([[1, 2, 1]])
        attention = torch.tensor([[1, 1, 0]])
        counts = new_entity_counts()
        update_entity_counts(counts, predictions, labels, attention)
        metrics = finalize_entity_metrics(counts)
        self.assertEqual(metrics["corrected_micro"]["f1"], 1.0)


class CheckpointDiscoveryTests(unittest.TestCase):
    def test_identical_best_model_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory, payload in (
                ("checkpoint-1", b"one"),
                ("checkpoint-2", b"two"),
                ("best_model", b"two"),
            ):
                target = root / directory / "pytorch_model.bin"
                target.parent.mkdir()
                target.write_bytes(payload)

            candidates = discover_unique_checkpoints(root)
            self.assertEqual(len(candidates), 2)
            second = next(item for item in candidates if item.canonical_path.parent.name == "checkpoint-2")
            self.assertEqual(
                {path.parent.name for path in second.aliases},
                {"checkpoint-2", "best_model"},
            )


class SelectionTests(unittest.TestCase):
    @staticmethod
    def result(name: str, f1: float) -> dict:
        candidate = CheckpointCandidate(
            candidate_id=name,
            canonical_path=Path(f"/tmp/{name}/pytorch_model.bin"),
            sha256=name,
            aliases=(),
        )
        return {
            "checkpoint": candidate.to_dict(),
            "canonical_path": str(candidate.canonical_path),
            "metrics": {
                "official_pair_metrics": {
                    "regular": {"text_location": {"f1": f1}}
                }
            },
        }

    def test_highest_regular_text_location_f1_wins(self):
        low = self.result("checkpoint-1", 0.2)
        high = self.result("checkpoint-2", 0.3)
        self.assertIs(select_best_validation_result([low, high]), high)

    def test_numeric_checkpoint_order_breaks_exact_tie(self):
        later = self.result("checkpoint-10", 0.3)
        earlier = self.result("checkpoint-2", 0.3)
        self.assertIs(select_best_validation_result([later, earlier]), earlier)

    def test_official_macro_evaluator_is_used(self):
        box = [0, 0, 10, 10]
        kvp = {
            "type": "kvp",
            "key": {"text": "key", "bbox": box},
            "value": {"text": "value", "bbox": box},
        }
        table = evaluate_table(
            [{"document_id": "doc", "predictions": [kvp], "ground_truths": [kvp]}]
        )
        self.assertEqual(table["regular"]["text_location"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
