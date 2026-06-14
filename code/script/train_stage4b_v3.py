"""
Stage 4b V3 Training: Linker-Only with Predicted Entities.

Key change from V2:
  V2 uses teacher forcing — the linker sees GT entity labels during training
  but predicted entities during inference. This mismatch causes 65% of test
  documents to have zero key/value spans, killing link recall.

  V3 freezes the encoder + entity classifier (already trained in V2) and
  trains ONLY the linker head, using the classifier's own predictions to
  derive key/value spans. The linker learns to work with realistic, imperfect
  entity predictions from the start.

Architecture:
  1. LayoutLMv3 encoder        — FROZEN (loaded from V2+TF checkpoint)
  2. Entity classifier          — FROZEN (loaded from V2+TF checkpoint)
  3. SpanBiaffineLinker          — TRAINABLE (re-initialized or loaded)

The model forward is called with entity_labels=None (so the linker uses
argmax(entity_logits) for spans) but link_labels is still provided for loss.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import json
from pathlib import Path
import logging
from datetime import datetime
import config
from layoutlm_model_v2 import create_model, LayoutLMv3KVPModelV2
from stage4_kvp_dataset import LayoutLMv3PreparedDataset, create_stage4_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class Stage4bV3Trainer:
    """Trainer for Stage 4b V3: Linker-only training with predicted entities."""

    def __init__(
        self,
        model: LayoutLMv3KVPModelV2,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        output_dir: str,
        learning_rate: float = 1e-4,
        num_epochs: int = 30,
        early_stopping_patience: int = 10,
        linker_loss_weight: float = 1.0,
        gradient_accumulation_steps: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.linker_loss_weight = linker_loss_weight
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device

        # Freeze encoder + entity classifier
        frozen_count = 0
        trainable_count = 0
        for name, param in model.named_parameters():
            if name.startswith("linker."):
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1

        logger.info(f"Frozen parameters: {frozen_count} tensors")
        logger.info(f"Trainable parameters (linker only): {trainable_count} tensors")

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"Total trainable params: {total_trainable:,}")

        self.optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.01
        )

        self.total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=200,
            num_training_steps=self.total_steps
        )

        self.training_history = {
            "epochs": [],
            "train_losses": [],
            "val_link_losses": [],
            "val_link_f1s": [],
            "linker_loss_weight": linker_loss_weight,
            "model_version": "v3_linker_only"
        }
        self.best_val_metric = 0.0
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        # But encoder and entity_classifier stay in eval mode (frozen)
        self.model.encoder.eval()
        self.model.entity_classifier.eval()

        total_loss = 0.0
        num_batches = 0
        batches_with_links = 0
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch_idx, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            bbox           = batch["bbox"].to(self.device)
            pixel_values   = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)
            link_labels    = batch.get("link_labels")
            if link_labels is not None:
                link_labels = link_labels.to(self.device)

            # V3: entity_labels=None → linker uses predicted entities
            # link_labels provided → model computes link loss
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                entity_labels=None,       # ← predicted entities
                link_labels=link_labels    # ← GT links for loss
            )

            link_loss = outputs.get('link_loss')

            if link_loss is not None:
                loss = self.linker_loss_weight * link_loss

                if torch.isnan(loss):
                    logger.error(f"NaN loss at step {batch_idx}!")
                    continue

                loss_scaled = loss / self.gradient_accumulation_steps
                loss_scaled.backward()
                batches_with_links += 1

                total_loss += loss.item()
            else:
                # No key/value spans found → no linker loss
                # This is expected for some documents
                pass

            num_batches += 1

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            l_l = link_loss.item() if link_loss is not None else 0.0
            pbar.set_postfix({
                "link_loss": f"{l_l:.4f}",
                "w_links": f"{batches_with_links}/{num_batches}"
            })

        # Flush remaining gradients
        if num_batches % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

        avg_loss = total_loss / max(batches_with_links, 1)
        logger.info(f"Batches with link signal: {batches_with_links}/{num_batches}")
        return avg_loss

    def validate(self):
        """Validate using link loss + a simple link accuracy metric.

        Since we're only training the linker, we track:
        1. Average link loss on val set
        2. Link accuracy: fraction of key spans whose argmax value is correct
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        total_correct = 0
        total_keys_with_gt = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bbox           = batch["bbox"].to(self.device)
                pixel_values   = batch.get("pixel_values")
                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device)
                link_labels    = batch.get("link_labels")
                if link_labels is not None:
                    link_labels = link_labels.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    entity_labels=None,
                    link_labels=link_labels
                )

                link_loss = outputs.get('link_loss')
                if link_loss is not None:
                    total_loss += link_loss.item()
                    num_batches += 1

                # Compute argmax link accuracy
                link_scores = outputs.get('link_scores')
                key_spans = outputs.get('key_indices')
                val_spans = outputs.get('value_indices')

                if link_scores is not None:
                    for b in range(input_ids.size(0)):
                        if link_scores[b] is None:
                            continue
                        scores = link_scores[b]  # [nk, nv]
                        k_sp = key_spans[b]
                        v_sp = val_spans[b]
                        if len(k_sp) == 0 or len(v_sp) == 0:
                            continue

                        # Get span-level GT
                        from layoutlm_model_v2 import collapse_link_labels_to_spans
                        span_gt = collapse_link_labels_to_spans(
                            link_labels[b], k_sp, v_sp
                        )

                        # For each key, check if argmax value matches GT
                        best_val = torch.argmax(scores, dim=1)
                        for ki in range(len(k_sp)):
                            gt_row = span_gt[ki]
                            if gt_row.sum() > 0:  # this key has a GT link
                                total_keys_with_gt += 1
                                if gt_row[best_val[ki]] > 0.5:
                                    total_correct += 1

        avg_loss = total_loss / max(num_batches, 1)
        link_acc = total_correct / max(total_keys_with_gt, 1)
        logger.info(f"Link accuracy (argmax): {total_correct}/{total_keys_with_gt} = {link_acc:.4f}")

        return avg_loss, link_acc

    def train(self):
        logger.info(f"Starting Stage 4b V3 (linker-only) training")
        logger.info(f"Output directory: {self.output_dir}")

        for epoch in range(self.num_epochs):
            logger.info(f"\n=== Epoch {epoch+1}/{self.num_epochs} ===")

            train_loss = self.train_epoch()
            logger.info(f"Train link loss: {train_loss:.4f}")

            val_loss, val_link_acc = self.validate()
            logger.info(f"Val link loss: {val_loss:.4f}, Val link acc: {val_link_acc:.4f}")

            self.training_history["epochs"].append(epoch + 1)
            self.training_history["train_losses"].append(train_loss)
            self.training_history["val_link_losses"].append(val_loss)
            self.training_history["val_link_f1s"].append(val_link_acc)

            ckpt_dir = self.output_dir / f"checkpoint-{epoch+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), ckpt_dir / "pytorch_model.bin")

            # Early stopping on link accuracy (not entity F1)
            if val_link_acc > self.best_val_metric:
                self.best_val_metric = val_link_acc
                self.patience_counter = 0
                best_dir = self.output_dir / "best_model"
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), best_dir / "pytorch_model.bin")
                logger.info(f"New best link acc: {self.best_val_metric:.4f} (saved)")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        history_file = self.output_dir / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {history_file}")

        return self.training_history


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stage 4b V3: Linker-Only Training")
    parser.add_argument("--data_dir",                    type=str,   default="../../data/prepared")
    parser.add_argument("--output_dir",                  type=str,   default="../../data/outputs/stage4b_v3")
    parser.add_argument("--checkpoint",                  type=str,   required=True,
                        help="Path to V2+TF best_model/pytorch_model.bin")
    parser.add_argument("--reinit_linker",               action="store_true",
                        help="Re-initialize linker weights instead of loading from checkpoint")
    parser.add_argument("--batch_size",                  type=int,   default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=4)
    parser.add_argument("--learning_rate",               type=float, default=1e-4)
    parser.add_argument("--num_epochs",                  type=int,   default=30)
    parser.add_argument("--early_stopping_patience",     type=int,   default=10)
    parser.add_argument("--val_fraction",                type=float, default=0.1)
    parser.add_argument("--linker_loss_weight",          type=float, default=5.0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create model
    model = LayoutLMv3KVPModelV2(use_linker=True)

    # Load full checkpoint (encoder + entity_classifier + linker)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded checkpoint: {ckpt_path}")
    if missing:
        logger.warning(f"Missing keys: {missing[:5]}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}")

    # Optionally re-initialize linker weights
    if args.reinit_linker:
        logger.info("Re-initializing linker weights from scratch")
        model.linker._init_weights()
    else:
        logger.info("Keeping linker weights from checkpoint (fine-tuning)")

    # Data
    dataloaders = create_stage4_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        include_images=False
    )
    train_loader = dataloaders['train']
    val_loader   = dataloaders['val']
    test_loader  = dataloaders['test']
    logger.info(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} | Test: {len(test_loader)}")

    trainer = Stage4bV3Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        linker_loss_weight=args.linker_loss_weight,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device=device
    )

    trainer.train()
    logger.info(f"Training complete! Best link accuracy: {trainer.best_val_metric:.4f}")


if __name__ == "__main__":
    main()
