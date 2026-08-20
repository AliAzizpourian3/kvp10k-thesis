"""Corrected V5 training for LayoutLMv3 plus the span-level biaffine linker.

V5 intentionally keeps the V4 architecture, prepared data, seed-42 split, and
blank visual input.  It corrects optimizer-step scheduling, class-aware entity
metrics, checkpoint state/resume, and checkpoint selection.  Validation uses
the official regular text+location macro pair F1 at link threshold 0.5; the
test set is evaluated exactly once, after training, using the selected model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from evaluate_stage4b import _span_to_bbox, _span_to_text
from kvp10k_official_eval import evaluate_table
from layoutlm_model_v2 import LayoutLMv3KVPModelV2
from stage4_kvp_dataset import create_stage4_dataloaders


LOGGER = logging.getLogger(__name__)
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
ENTITY_CLASSES = {1: "key", 2: "value"}
SCORE_THRESHOLD = 0.5
NED_THRESHOLD = 0.2
IOU_THRESHOLD = 0.3
WARMUP_OPTIMIZER_STEPS = 500
STATE_FILENAME = "training_state.pt"
WEIGHTS_FILENAME = "pytorch_model.bin"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def optimizer_steps_per_epoch(number_of_batches: int, accumulation_steps: int) -> int:
    if number_of_batches < 0:
        raise ValueError("number_of_batches must be non-negative")
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    return math.ceil(number_of_batches / accumulation_steps)


def total_optimizer_steps(
    number_of_batches: int, accumulation_steps: int, epochs: int
) -> int:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    return optimizer_steps_per_epoch(number_of_batches, accumulation_steps) * epochs


def checkpoint_number(path: Path) -> int:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Not a numerical checkpoint directory: {path}")
    return int(match.group(1))


def numerical_checkpoints(output_dir: Path, require_full_state: bool = True) -> list[Path]:
    checkpoints = []
    if not output_dir.is_dir():
        return checkpoints
    for path in output_dir.iterdir():
        if not path.is_dir() or CHECKPOINT_PATTERN.fullmatch(path.name) is None:
            continue
        if require_full_state and not (
            (path / WEIGHTS_FILENAME).is_file() and (path / STATE_FILENAME).is_file()
        ):
            continue
        checkpoints.append(path)
    return sorted(checkpoints, key=checkpoint_number)


def resolve_resume_checkpoint(output_dir: Path, specification: str) -> Path:
    if specification == "auto":
        checkpoints = numerical_checkpoints(output_dir, require_full_state=True)
        if not checkpoints:
            raise FileNotFoundError(
                f"No complete numerical checkpoints found under {output_dir}"
            )
        return checkpoints[-1]
    path = Path(specification).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint directory not found: {path}")
    checkpoint_number(path)
    for filename in (WEIGHTS_FILENAME, STATE_FILENAME):
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Resume checkpoint lacks {filename}: {path}")
    if path.parent.resolve() != output_dir.resolve():
        raise ValueError(
            f"Resume checkpoint {path} is not inside output directory {output_dir}"
        )
    return path


def new_entity_counts() -> dict:
    return {
        name: {"tp": 0, "fp": 0, "fn": 0}
        for name in ENTITY_CLASSES.values()
    }


def update_entity_counts(
    counts: dict,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    active = attention_mask == 1
    for class_id, name in ENTITY_CLASSES.items():
        predicted_class = (predictions == class_id) & active
        true_class = (labels == class_id) & active
        counts[name]["tp"] += int((predicted_class & true_class).sum().item())
        counts[name]["fp"] += int((predicted_class & ~true_class).sum().item())
        counts[name]["fn"] += int((~predicted_class & true_class).sum().item())


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def finalize_entity_metrics(counts: dict) -> dict:
    per_class = {name: prf(**value) for name, value in counts.items()}
    micro_counts = {
        key: sum(value[key] for value in counts.values())
        for key in ("tp", "fp", "fn")
    }
    macro = {
        metric: sum(value[metric] for value in per_class.values()) / len(per_class)
        for metric in ("precision", "recall", "f1")
    }
    return {
        "primary": "micro one-vs-rest over KEY and VALUE",
        "micro": prf(**micro_counts),
        "macro": macro,
        "per_class": per_class,
    }


def capture_random_states() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_states(states: dict) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"].cpu())
    if torch.cuda.is_available() and states.get("cuda"):
        torch.cuda.set_rng_state_all([state.cpu() for state in states["cuda"]])


def save_training_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    completed_epoch: int,
    global_optimizer_step: int,
    best_validation_score: float,
    early_stopping_counter: int,
    training_history: dict,
    run_configuration: dict,
) -> dict:
    """Atomically save evaluator weights and complete resumable trainer state."""
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    weights_path = checkpoint_dir / WEIGHTS_FILENAME
    atomic_torch_save(model.state_dict(), weights_path)
    state = {
        "format_version": 1,
        "model_weights_file": WEIGHTS_FILENAME,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "gradient_scaler_state": scaler.state_dict(),
        "completed_epoch": completed_epoch,
        "global_optimizer_step": global_optimizer_step,
        "best_validation_score": best_validation_score,
        "early_stopping_counter": early_stopping_counter,
        "training_history": training_history,
        "random_states": capture_random_states(),
        "run_configuration": run_configuration,
        "saved_at": utc_now(),
    }
    atomic_torch_save(state, checkpoint_dir / STATE_FILENAME)
    digest = sha256_file(weights_path)
    metadata = {
        "checkpoint_path": str(checkpoint_dir.resolve()),
        "weights_path": str(weights_path.resolve()),
        "weights_sha256": digest,
        "completed_epoch": completed_epoch,
        "global_optimizer_step": global_optimizer_step,
        "state_file": STATE_FILENAME,
    }
    write_json(checkpoint_dir / "checkpoint_metadata.json", metadata)
    return metadata


def assert_resume_configuration(saved: dict, current: dict) -> None:
    keys = (
        "model",
        "data_dir",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "num_epochs",
        "early_stopping_patience",
        "linker_loss_weight",
        "seed",
        "validation_fraction",
        "include_images",
        "score_threshold",
        "warmup_optimizer_steps",
        "optimizer_steps_per_epoch",
        "total_optimizer_steps",
    )
    mismatches = {
        key: {"saved": saved.get(key), "current": current.get(key)}
        for key in keys
        if saved.get(key) != current.get(key)
    }
    if mismatches:
        raise ValueError(f"Resume configuration mismatch: {mismatches}")


def load_training_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    run_configuration: dict,
) -> dict:
    state = torch.load(
        checkpoint_dir / STATE_FILENAME, map_location=device, weights_only=False
    )
    assert_resume_configuration(state["run_configuration"], run_configuration)
    model_state = torch.load(
        checkpoint_dir / WEIGHTS_FILENAME, map_location=device, weights_only=False
    )
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    scaler.load_state_dict(state["gradient_scaler_state"])
    restore_random_states(state["random_states"])
    return state


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_value(arguments: list[str], default: str = "unknown") -> str:
    try:
        result = subprocess.run(
            ["git", *arguments], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def environment_record(device: torch.device) -> dict:
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    cuda = {}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        cuda = {
            "device_name": torch.cuda.get_device_name(device),
            "total_memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(device)),
        }
    return {
        "recorded_at": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "cuda": cuda,
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_PARTITION",
                "SLURMD_NODENAME",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "offline_environment": {
            key: os.environ.get(key)
            for key in (
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_HOME",
                "HUGGINGFACE_HUB_CACHE",
                "PYTORCH_CUDA_ALLOC_CONF",
            )
        },
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_status_porcelain": git_value(["status", "--porcelain"], default=""),
    }


def transfer_warm_start(model: torch.nn.Module, checkpoint: Path) -> dict:
    checkpoint = checkpoint.resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model_state = model.state_dict()
    transferred = []
    skipped = []
    for original_key, value in state.items():
        key = original_key.removeprefix("module.")
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            transferred.append(key)
        else:
            skipped.append(original_key)
    model.load_state_dict(model_state, strict=True)

    groups = {}
    for group, prefix in (
        ("encoder", "encoder."),
        ("entity_classifier", "entity_classifier."),
        ("linker", "linker."),
    ):
        keys = [key for key in transferred if key.startswith(prefix)]
        groups[group] = {
            "transferred_keys": len(keys),
            "transferred_parameters": sum(model_state[key].numel() for key in keys),
        }
    if any(groups[name]["transferred_keys"] == 0 for name in groups):
        raise RuntimeError(f"Warm start failed to transfer every required group: {groups}")
    return {
        "source": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "compatible_keys_loaded": len(transferred),
        "compatible_parameters_loaded": sum(
            model_state[key].numel() for key in transferred
        ),
        "skipped_keys": len(skipped),
        "groups": groups,
    }


def base_dataset_and_index(dataset: Any, index: int) -> tuple[Any, int]:
    while isinstance(dataset, Subset):
        index = dataset.indices[index]
        dataset = dataset.dataset
    return dataset, index


def raw_sample_context(dataset: Any, index: int) -> tuple[dict, list[str], list[list[int]], list[int | None], Path]:
    base, base_index = base_dataset_and_index(dataset, index)
    json_path = base.json_files[base_index]
    with json_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    words, word_bboxes = base._parse_lmdx_text(data.get("lmdx_text", ""), data)
    normalized = base._normalize_bboxes(
        word_bboxes, data.get("image_width", 1), data.get("image_height", 1)
    )
    blank_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
    encoded = base.processor(
        images=blank_image,
        text=words,
        boxes=normalized,
        return_tensors="pt",
        padding="max_length",
        max_length=base.max_seq_length,
        truncation=True,
    )
    return data, words, word_bboxes, encoded.word_ids(), json_path


def decode_regular_predictions(
    outputs: dict, dataset: Any, sample_indices: list[int]
) -> list[tuple[dict, list[dict], Path]]:
    decoded = []
    for offset, sample_index in enumerate(sample_indices):
        data, words, word_bboxes, word_ids, json_path = raw_sample_context(
            dataset, sample_index
        )
        predictions = []
        scores = outputs["link_scores"][offset]
        keys = outputs["key_indices"][offset]
        values = outputs["value_indices"][offset]
        if scores is not None and len(keys) and len(values):
            best_positions = torch.argmax(scores, dim=1)
            best_scores = torch.sigmoid(
                scores[
                    torch.arange(len(keys), device=scores.device), best_positions
                ]
            )
            for key_position, key_span in enumerate(keys):
                confidence = float(best_scores[key_position].item())
                if confidence < SCORE_THRESHOLD:
                    continue
                value_span = values[int(best_positions[key_position].item())]
                key_text = _span_to_text(
                    key_span[0], key_span[1], word_ids, words
                )
                value_text = _span_to_text(
                    value_span[0], value_span[1], word_ids, words
                )
                if not key_text or not value_text:
                    continue
                predictions.append(
                    {
                        "type": "kvp",
                        "key": {
                            "text": key_text,
                            "bbox": _span_to_bbox(
                                key_span[0], key_span[1], word_ids, word_bboxes
                            ),
                        },
                        "value": {
                            "text": value_text,
                            "bbox": _span_to_bbox(
                                value_span[0], value_span[1], word_ids, word_bboxes
                            ),
                        },
                        "link_confidence": confidence,
                    }
                )
        predictions.sort(key=lambda item: item["link_confidence"], reverse=True)
        decoded.append((data, predictions, json_path))
    return decoded


def compact_official_table(table: dict) -> dict:
    return {
        category: {
            mode: {
                key: value
                for key, value in metrics.items()
                if key != "per_document"
            }
            for mode, metrics in modes.items()
        }
        for category, modes in table.items()
    }


def verify_blank_visual_input(dataloaders: dict) -> dict:
    datasets = {}
    for name, loader in dataloaders.items():
        base, _ = base_dataset_and_index(loader.dataset, 0)
        if base.include_images or base.image_base_dir is not None:
            raise RuntimeError(f"{name} dataset is configured to load real images")
        datasets[name] = {
            "include_images": base.include_images,
            "image_base_dir": base.image_base_dir,
        }
    sample = dataloaders["train"].dataset[0]["pixel_values"]
    return {
        "policy": "generated solid-white 224x224 placeholder; no image file opened",
        "datasets": datasets,
        "sample_tensor": {
            "shape": list(sample.shape),
            "dtype": str(sample.dtype),
            "minimum": float(sample.min().item()),
            "maximum": float(sample.max().item()),
            "mean": float(sample.mean().item()),
            "nonzero_values": int(torch.count_nonzero(sample).item()),
            "total_values": sample.numel(),
        },
    }


class Stage4bV5Trainer:
    def __init__(
        self,
        model: LayoutLMv3KVPModelV2,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
        output_dir: Path,
        configuration: dict,
        device: torch.device,
        max_train_batches: int | None = None,
        max_validation_documents: int | None = None,
        max_test_documents: int | None = None,
        skip_final_test: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.configuration = configuration
        self.device = device
        self.max_train_batches = max_train_batches
        self.max_validation_documents = max_validation_documents
        self.max_test_documents = max_test_documents
        self.skip_final_test = skip_final_test
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=configuration["learning_rate"],
            weight_decay=0.01,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=configuration["warmup_optimizer_steps"],
            num_training_steps=configuration["total_optimizer_steps"],
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.completed_epoch = 0
        self.global_optimizer_step = 0
        self.best_validation_score = float("-inf")
        self.early_stopping_counter = 0
        self.training_history = {
            "model_version": "v5_corrected_official_selection",
            "selection_metric": "official regular text+location macro pair F1",
            "score_threshold": SCORE_THRESHOLD,
            "epochs": [],
        }

    def resume(self, checkpoint_dir: Path) -> None:
        state = load_training_checkpoint(
            checkpoint_dir,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.device,
            self.configuration,
        )
        self.completed_epoch = int(state["completed_epoch"])
        self.global_optimizer_step = int(state["global_optimizer_step"])
        self.best_validation_score = float(state["best_validation_score"])
        self.early_stopping_counter = int(state["early_stopping_counter"])
        self.training_history = state["training_history"]
        LOGGER.info(
            "Resumed complete state from %s (epoch=%d, optimizer_step=%d)",
            checkpoint_dir,
            self.completed_epoch,
            self.global_optimizer_step,
        )

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.global_optimizer_step += 1

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_entity_loss = 0.0
        total_link_loss = 0.0
        batches = 0
        link_batches = 0
        limit = self.max_train_batches or len(self.train_loader)
        progress = tqdm(self.train_loader, total=min(len(self.train_loader), limit), desc=f"Train {epoch}", leave=False)
        for batch_index, batch in enumerate(progress):
            if batch_index >= limit:
                break
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            bbox = batch["bbox"].to(self.device)
            pixel_values = batch["pixel_values"].to(self.device)
            entity_labels = batch["entity_labels"].to(self.device)
            link_labels = batch["link_labels"].to(self.device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                entity_labels=entity_labels,
                link_labels=link_labels,
            )
            entity_loss = outputs["entity_loss"]
            link_loss = outputs["link_loss"]
            loss = entity_loss
            if link_loss is not None:
                loss = loss + self.configuration["linker_loss_weight"] * link_loss
                total_link_loss += float(link_loss.item())
                link_batches += 1
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_index}"
                )
            self.scaler.scale(
                loss / self.configuration["gradient_accumulation_steps"]
            ).backward()
            batches += 1
            total_loss += float(loss.item())
            total_entity_loss += float(entity_loss.item())
            if batches % self.configuration["gradient_accumulation_steps"] == 0:
                self._optimizer_step()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        if batches and batches % self.configuration["gradient_accumulation_steps"]:
            self._optimizer_step()
        if not batches:
            raise RuntimeError("Training loader produced no batches")
        return {
            "loss": total_loss / batches,
            "entity_loss": total_entity_loss / batches,
            "link_loss": total_link_loss / max(link_batches, 1),
            "batches": batches,
            "batches_with_link_loss": link_batches,
            "optimizer_steps": optimizer_steps_per_epoch(
                batches, self.configuration["gradient_accumulation_steps"]
            ),
        }

    def evaluate(
        self,
        loader: DataLoader,
        split: str,
        predictions_dir: Path,
        maximum_documents: int | None,
    ) -> dict:
        self.model.eval()
        entity_counts = new_entity_counts()
        documents = []
        total_entity_loss = 0.0
        total_link_loss = 0.0
        loss_batches = 0
        link_loss_batches = 0
        sample_index = 0
        predictions_dir.mkdir(parents=True, exist_ok=False)
        target_documents = min(len(loader.dataset), maximum_documents or len(loader.dataset))
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Evaluate {split}", leave=False):
                if sample_index >= target_documents:
                    break
                current_size = min(batch["input_ids"].size(0), target_documents - sample_index)
                input_ids = batch["input_ids"][:current_size].to(self.device)
                attention_mask = batch["attention_mask"][:current_size].to(self.device)
                bbox = batch["bbox"][:current_size].to(self.device)
                pixel_values = batch["pixel_values"][:current_size].to(self.device)
                labels = batch["entity_labels"][:current_size].to(self.device)
                link_labels = batch["link_labels"][:current_size].to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    entity_labels=None,
                    link_labels=link_labels,
                )
                active = attention_mask.reshape(-1) == 1
                entity_loss = F.cross_entropy(
                    outputs["entity_logits"].reshape(-1, 3)[active],
                    labels.reshape(-1)[active],
                )
                total_entity_loss += float(entity_loss.item())
                loss_batches += 1
                if outputs["link_loss"] is not None:
                    total_link_loss += float(outputs["link_loss"].item())
                    link_loss_batches += 1
                predicted = torch.argmax(outputs["entity_logits"], dim=-1)
                update_entity_counts(
                    entity_counts,
                    predicted.cpu(),
                    labels.cpu(),
                    attention_mask.cpu(),
                )
                indices = list(range(sample_index, sample_index + current_size))
                for data, predictions, json_path in decode_regular_predictions(
                    outputs, loader.dataset, indices
                ):
                    ground_truths = data.get("gt_kvps", {}).get("kvps_list", [])
                    documents.append(
                        {
                            "document_id": data.get("hash_name", json_path.stem),
                            "predictions": predictions,
                            "ground_truths": ground_truths,
                        }
                    )
                    write_json(predictions_dir / json_path.name, {"kvps_list": predictions})
                sample_index += current_size
        if sample_index != target_documents:
            raise RuntimeError(
                f"{split} processed {sample_index}, expected {target_documents}"
            )
        official = compact_official_table(
            evaluate_table(
                documents,
                ned_threshold=NED_THRESHOLD,
                iou_threshold=IOU_THRESHOLD,
            )
        )
        return {
            "split": split,
            "documents": sample_index,
            "score_threshold": SCORE_THRESHOLD,
            "official_thresholds": {
                "ned_strict_less_than": NED_THRESHOLD,
                "iou_inclusive_greater_equal": IOU_THRESHOLD,
            },
            "loss": {
                "entity": total_entity_loss / max(loss_batches, 1),
                "link": total_link_loss / max(link_loss_batches, 1),
            },
            "entity_metrics": finalize_entity_metrics(entity_counts),
            "official_pair_metrics": official,
        }

    def _record_best(self, checkpoint_metadata: dict, validation_score: float) -> None:
        best_dir = self.output_dir / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_metadata["weights_path"], best_dir / WEIGHTS_FILENAME)
        write_json(
            best_dir / "selection.json",
            {
                "selected_at": utc_now(),
                "selection_metric": "official regular text+location macro pair F1",
                "score_threshold": SCORE_THRESHOLD,
                "validation_score": validation_score,
                **checkpoint_metadata,
            },
        )

    def train(self) -> dict:
        stopped_early = False
        for epoch in range(self.completed_epoch + 1, self.configuration["num_epochs"] + 1):
            LOGGER.info("=== V5 epoch %d/%d ===", epoch, self.configuration["num_epochs"])
            train_metrics = self.train_epoch(epoch)
            validation_staging = self.output_dir / f".validation-{epoch}.tmp"
            if validation_staging.exists():
                shutil.rmtree(validation_staging)
            validation_metrics = self.evaluate(
                self.validation_loader,
                "validation",
                validation_staging,
                self.max_validation_documents,
            )
            score = validation_metrics["official_pair_metrics"]["regular"][
                "text_location"
            ]["f1"]
            improved = score > self.best_validation_score
            if improved:
                self.best_validation_score = score
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            self.completed_epoch = epoch
            epoch_record = {
                "epoch": epoch,
                "completed_at": utc_now(),
                "global_optimizer_step": self.global_optimizer_step,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": validation_metrics,
                "selection_score": score,
                "is_best": improved,
                "early_stopping_counter": self.early_stopping_counter,
            }
            self.training_history["epochs"].append(epoch_record)
            checkpoint_dir = self.output_dir / f"checkpoint-{epoch}"
            checkpoint_metadata = save_training_checkpoint(
                checkpoint_dir,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.completed_epoch,
                self.global_optimizer_step,
                self.best_validation_score,
                self.early_stopping_counter,
                self.training_history,
                self.configuration,
            )
            predictions_dir = checkpoint_dir / "validation_predictions"
            os.replace(validation_staging, predictions_dir)
            validation_result = {
                "checkpoint": checkpoint_metadata,
                "selection_metric": {
                    "category": "regular",
                    "mode": "text_location",
                    "score_threshold": SCORE_THRESHOLD,
                    "value": score,
                },
                "metrics": validation_metrics,
            }
            write_json(checkpoint_dir / "validation_result.json", validation_result)
            epoch_record["checkpoint"] = checkpoint_metadata
            write_json(self.output_dir / "training_history.json", self.training_history)
            if improved:
                self._record_best(checkpoint_metadata, score)
            LOGGER.info(
                "Epoch %d official validation regular text+location F1=%.10f%s",
                epoch,
                score,
                " (new best)" if improved else "",
            )
            if self.early_stopping_counter >= self.configuration["early_stopping_patience"]:
                stopped_early = True
                LOGGER.info("Early stopping after epoch %d", epoch)
                break

        completion = {
            "completed_at": utc_now(),
            "completed_epoch": self.completed_epoch,
            "global_optimizer_step": self.global_optimizer_step,
            "best_validation_score": self.best_validation_score,
            "stopped_early": stopped_early,
            "maximum_epochs": self.configuration["num_epochs"],
        }
        write_json(self.output_dir / "training_completion.json", completion)
        if self.skip_final_test:
            completion["test_evaluation"] = "skipped by explicit smoke-test option"
            write_json(self.output_dir / "training_completion.json", completion)
            return completion

        selection_path = self.output_dir / "best_model" / "selection.json"
        with selection_path.open(encoding="utf-8") as handle:
            selection = json.load(handle)
        selected_checkpoint = Path(selection["checkpoint_path"])
        selected_weights = selected_checkpoint / WEIGHTS_FILENAME
        digest = sha256_file(selected_weights)
        if digest != selection["weights_sha256"]:
            raise RuntimeError("Selected checkpoint SHA-256 changed before test evaluation")
        state = torch.load(selected_weights, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state, strict=True)
        test_metrics = self.evaluate(
            self.test_loader,
            "test",
            self.output_dir / "final_test_predictions",
            self.max_test_documents,
        )
        test_result = {
            "evaluated_at": utc_now(),
            "selected_by_validation_only": True,
            "checkpoint_path": str(selected_checkpoint),
            "checkpoint_sha256": digest,
            "validation_selection": selection,
            "metrics": test_metrics,
        }
        write_json(self.output_dir / "test_evaluation_official.json", test_result)
        completion["test_evaluation"] = {
            "result_path": str(
                (self.output_dir / "test_evaluation_official.json").resolve()
            ),
            "checkpoint_path": str(selected_checkpoint),
            "checkpoint_sha256": digest,
        }
        write_json(self.output_dir / "training_completion.json", completion)
        return completion


def build_configuration(args: argparse.Namespace, train_batches: int) -> dict:
    effective_batches = args.max_train_batches or train_batches
    steps_per_epoch = optimizer_steps_per_epoch(
        effective_batches, args.gradient_accumulation_steps
    )
    return {
        "model": "microsoft/layoutlmv3-base + existing span-level biaffine linker",
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "linker_loss_weight": args.linker_loss_weight,
        "seed": args.seed,
        "validation_fraction": args.val_fraction,
        "include_images": False,
        "visual_input": "generated solid-white placeholder",
        "score_threshold": SCORE_THRESHOLD,
        "selection_metric": "official regular text+location macro pair F1",
        "ned_threshold": NED_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "warmup_optimizer_steps": WARMUP_OPTIMIZER_STEPS,
        "number_of_training_batches": effective_batches,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": steps_per_epoch * args.num_epochs,
        "max_train_batches": args.max_train_batches,
        "max_validation_documents": args.max_validation_documents,
        "max_test_documents": args.max_test_documents,
        "skip_final_test": args.skip_final_test,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data/prepared"))
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/outputs/stage4b_v5")
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--linker_loss_weight", type=float, default=5.0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warm_start",
        type=Path,
        default=Path("data/outputs/stage4b_canary_B/best_model/pytorch_model.bin"),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        nargs="?",
        const="auto",
        default=None,
        help="Explicit checkpoint directory, or 'auto' for newest numeric checkpoint",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_validation_documents", type=int, default=None)
    parser.add_argument("--max_test_documents", type=int, default=None)
    parser.add_argument("--skip_final_test", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("V5 batch_size is fixed at 1")
    if args.seed != 42:
        raise ValueError("V5 seed is fixed at 42")
    if args.val_fraction != 0.1:
        raise ValueError("V5 validation fraction is fixed at 0.1")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Prepared data directory not found: {args.data_dir}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    validate_args(args)
    args.output_dir = args.output_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.warm_start = args.warm_start.resolve()
    resume_checkpoint = None
    if args.resume_from_checkpoint is None:
        if args.output_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing V5 output: {args.output_dir}"
            )
        args.output_dir.mkdir(parents=True)
    else:
        resume_checkpoint = resolve_resume_checkpoint(
            args.output_dir, args.resume_from_checkpoint
        )

    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    set_all_seeds(args.seed)
    dataloaders = create_stage4_dataloaders(
        data_dir=str(args.data_dir),
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=0,
        include_images=False,
        image_base_dir=None,
    )
    visual_record = verify_blank_visual_input(dataloaders)
    configuration = build_configuration(args, len(dataloaders["train"]))
    data_counts = {
        "prepared_train_source": len(dataloaders["train"].dataset)
        + len(dataloaders["val"].dataset),
        "train": len(dataloaders["train"].dataset),
        "validation": len(dataloaders["val"].dataset),
        "test": len(dataloaders["test"].dataset),
        "train_batches": len(dataloaders["train"]),
        "validation_batches": len(dataloaders["val"]),
        "test_batches": len(dataloaders["test"]),
    }
    model = LayoutLMv3KVPModelV2(use_linker=True)
    warm_start_provenance = None
    if resume_checkpoint is None:
        if not args.warm_start.is_file():
            raise FileNotFoundError(f"Warm-start checkpoint not found: {args.warm_start}")
        warm_start_provenance = transfer_warm_start(model, args.warm_start)
        LOGGER.info("Warm start provenance: %s", warm_start_provenance)
        configuration["warm_start_provenance"] = warm_start_provenance
    else:
        saved_configuration_path = args.output_dir / "run_configuration.json"
        if not saved_configuration_path.is_file():
            raise FileNotFoundError(
                f"Resume output lacks run configuration: {saved_configuration_path}"
            )
        with saved_configuration_path.open(encoding="utf-8") as handle:
            saved_configuration = json.load(handle)
        assert_resume_configuration(saved_configuration, configuration)
        configuration = saved_configuration

    manifest_path = args.output_dir / "run_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest.setdefault("resume_events", []).append(
            {
                "at": utc_now(),
                "command": [sys.executable, *sys.argv],
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "checkpoint": str(resume_checkpoint),
                "environment": environment_record(device),
            }
        )
    else:
        manifest = {
            "status": "running",
            "started_at": utc_now(),
            "command": [sys.executable, *sys.argv],
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "configuration": configuration,
            "data_counts": data_counts,
            "visual_input": visual_record,
            "warm_start": warm_start_provenance,
            "environment": environment_record(device),
            "resume_events": [],
        }
    write_json(manifest_path, manifest)
    write_json(args.output_dir / "run_configuration.json", configuration)

    trainer = Stage4bV5Trainer(
        model=model,
        train_loader=dataloaders["train"],
        validation_loader=dataloaders["val"],
        test_loader=dataloaders["test"],
        output_dir=args.output_dir,
        configuration=configuration,
        device=device,
        max_train_batches=args.max_train_batches,
        max_validation_documents=args.max_validation_documents,
        max_test_documents=args.max_test_documents,
        skip_final_test=args.skip_final_test,
    )
    if resume_checkpoint is not None:
        trainer.resume(resume_checkpoint)
    completion = trainer.train()
    manifest["status"] = "completed"
    manifest["completed_at"] = utc_now()
    manifest["completion"] = completion
    write_json(manifest_path, manifest)
    LOGGER.info("V5 run complete: %s", completion)


if __name__ == "__main__":
    main()
