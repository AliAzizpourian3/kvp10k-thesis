"""Focused CPU-safe tests for the corrected V5 trainer."""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train_stage4b_v5 import (
    finalize_entity_metrics,
    load_training_checkpoint,
    new_entity_counts,
    numerical_checkpoints,
    optimizer_steps_per_epoch,
    resolve_resume_checkpoint,
    save_training_checkpoint,
    total_optimizer_steps,
    update_entity_counts,
)


def run_configuration() -> dict:
    return {
        "model": "tiny-test",
        "data_dir": "/tmp/data",
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "num_epochs": 30,
        "early_stopping_patience": 10,
        "linker_loss_weight": 5.0,
        "seed": 42,
        "validation_fraction": 0.1,
        "include_images": False,
        "score_threshold": 0.5,
        "warmup_optimizer_steps": 500,
        "optimizer_steps_per_epoch": 3,
        "total_optimizer_steps": 90,
    }


class SchedulerStepTests(unittest.TestCase):
    def test_steps_are_optimizer_updates_not_batches(self):
        self.assertEqual(optimizer_steps_per_epoch(17, 8), 3)
        self.assertEqual(total_optimizer_steps(17, 8, 30), 90)

    def test_exact_accumulation_boundary(self):
        self.assertEqual(optimizer_steps_per_epoch(16, 8), 2)


class EntityMetricTests(unittest.TestCase):
    def test_key_value_confusion_is_false_positive_and_false_negative(self):
        labels = torch.tensor([[1, 2]])
        predictions = torch.tensor([[1, 1]])
        attention = torch.ones_like(labels)
        counts = new_entity_counts()
        update_entity_counts(counts, predictions, labels, attention)
        metrics = finalize_entity_metrics(counts)

        self.assertEqual(metrics["micro"]["precision"], 0.5)
        self.assertEqual(metrics["micro"]["recall"], 0.5)
        self.assertEqual(metrics["micro"]["f1"], 0.5)
        self.assertEqual(metrics["per_class"]["key"]["fp"], 1)
        self.assertEqual(metrics["per_class"]["value"]["fn"], 1)
        self.assertIn("f1", metrics["macro"])


class NumericalCheckpointTests(unittest.TestCase):
    def test_newest_checkpoint_uses_integer_order_and_requires_full_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for number in (2, 9, 10):
                checkpoint = root / f"checkpoint-{number}"
                checkpoint.mkdir()
                (checkpoint / "pytorch_model.bin").touch()
                (checkpoint / "training_state.pt").touch()
            incomplete = root / "checkpoint-20"
            incomplete.mkdir()
            (incomplete / "pytorch_model.bin").touch()

            self.assertEqual(
                [path.name for path in numerical_checkpoints(root)],
                ["checkpoint-2", "checkpoint-9", "checkpoint-10"],
            )
            self.assertEqual(
                resolve_resume_checkpoint(root, "auto").name, "checkpoint-10"
            )


class ResumeRoundTripTests(unittest.TestCase):
    def test_full_state_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = run_configuration()
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)

            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: 1.0 - step / 100
            )
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            loss = model(torch.ones(1, 3)).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            history = {"epochs": [{"epoch": 1, "selection_score": 0.25}]}
            checkpoint = root / "checkpoint-1"
            save_training_checkpoint(
                checkpoint,
                model,
                optimizer,
                scheduler,
                scaler,
                completed_epoch=1,
                global_optimizer_step=1,
                best_validation_score=0.25,
                early_stopping_counter=0,
                training_history=history,
                run_configuration=configuration,
            )
            expected_python = random.random()
            expected_numpy = float(np.random.rand())
            expected_torch = float(torch.rand(1).item())

            restored_model = torch.nn.Linear(3, 2)
            restored_optimizer = torch.optim.AdamW(
                restored_model.parameters(), lr=2e-5
            )
            restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
                restored_optimizer, lr_lambda=lambda step: 1.0 - step / 100
            )
            restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
            state = load_training_checkpoint(
                checkpoint,
                restored_model,
                restored_optimizer,
                restored_scheduler,
                restored_scaler,
                torch.device("cpu"),
                configuration,
            )

            self.assertEqual(state["completed_epoch"], 1)
            self.assertEqual(state["global_optimizer_step"], 1)
            self.assertEqual(state["best_validation_score"], 0.25)
            self.assertEqual(state["training_history"], history)
            self.assertEqual(scheduler.state_dict(), restored_scheduler.state_dict())
            self.assertEqual(scaler.state_dict(), restored_scaler.state_dict())
            self.assertEqual(len(restored_optimizer.state), len(optimizer.state))
            for expected, actual in zip(model.parameters(), restored_model.parameters()):
                self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(float(np.random.rand()), expected_numpy)
            self.assertEqual(float(torch.rand(1).item()), expected_torch)


if __name__ == "__main__":
    unittest.main()
