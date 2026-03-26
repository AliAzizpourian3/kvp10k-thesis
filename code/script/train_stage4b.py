"""
Stage 4b Training: LayoutLMv3 KVP Extraction with Biaffine Linker.

Implements:
- LayoutLMv3 encoder + entity classifier + Biaffine linker
- Key-value linking with configurable loss weight
- Mixed precision (bf16) training for memory efficiency
  NOTE: bf16 is used instead of fp16 because the biaffine pairwise scorer
  produces large intermediate values that overflow fp16 (max ~65504), causing
  NaN loss. bf16 shares fp32's dynamic range and is natively supported on A100.
- Training from prepared JSON data (data/prepared/train/*.json)
- Validation with early stopping (patience=3 on F1)
- Checkpoint saving to data/outputs/stage4b_{linker_loss_weight}/
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Processor, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import os
import json
from pathlib import Path
import logging
from datetime import datetime
import config
from layoutlm_model import create_model, LayoutLMv3KVPModel
from stage4_kvp_dataset import LayoutLMv3PreparedDataset, create_stage4_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# bf16 does not need a GradScaler (no overflow risk), but we keep the
# scaler instantiation as a no-op (enabled=False) so the code path is
# identical and easy to switch back if needed.
_USE_BF16 = True   # A100-safe; set False to revert to fp32


class Stage4bTrainer:
    """Trainer for Stage 4b: LayoutLMv3 with Biaffine Linker."""

    def __init__(
        self,
        model: LayoutLMv3KVPModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        output_dir: str,
        learning_rate: float = 5e-5,
        num_epochs: int = 10,
        early_stopping_patience: int = 3,
        linker_loss_weight: float = 1.0,
        gradient_accumulation_steps: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.linker_loss_weight = linker_loss_weight
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * self.total_steps),
            num_training_steps=self.total_steps
        )

        self.training_history = {
            "epochs": [],
            "train_losses": [],
            "val_losses": [],
            "val_f1s": [],
            "linker_loss_weight": linker_loss_weight
        }
        self.best_val_f1 = 0.0
        self.patience_counter = 0

        # bf16: GradScaler is not needed (bf16 has fp32 dynamic range, no overflow).
        # We disable it explicitly so the optimizer step path is clean.
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.autocast_dtype = torch.bfloat16 if _USE_BF16 else torch.float32
        logger.info(f"Mixed precision: {'bf16' if _USE_BF16 else 'fp32 (no autocast)'}")

    def restore_history(self, checkpoint_dir: Path):
        """Restore training history and best_val_f1 from a previous run."""
        history_file = self.output_dir / "training_history.json"
        if history_file.exists():
            with open(history_file) as f:
                saved = json.load(f)
            self.training_history = saved
            if saved.get('val_f1s'):
                self.best_val_f1 = max(saved['val_f1s'])
            self.patience_counter = 0
            logger.info(f"\u2713 Restored training history: best_val_f1={self.best_val_f1:.4f}")
        else:
            logger.warning(f"No training_history.json found in {self.output_dir}, history not restored")

    def _optimizer_step(self):
        """
        Single helper that always performs the optimizer step correctly.
        With bf16 (scaler disabled) this reduces to: clip -> step -> zero_grad -> scheduler.
        The scaler calls are no-ops when enabled=False, keeping the code path uniform.
        """
        self.scaler.unscale_(self.optimizer)          # no-op when disabled
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)              # calls optimizer.step() internally
        self.scaler.update()                          # no-op when disabled
        self.optimizer.zero_grad()
        self.scheduler.step()

    def train_epoch(self):
        """Train one epoch with bf16 mixed precision."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch_idx, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            bbox           = batch["bbox"].to(self.device)
            pixel_values   = batch.get("pixel_values", None)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)
            entity_labels  = batch.get("entity_labels", None)
            if entity_labels is not None:
                entity_labels = entity_labels.to(self.device)
            link_labels    = batch.get("link_labels", None)
            if link_labels is not None:
                link_labels = link_labels.to(self.device)

            with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    entity_labels=entity_labels,
                    link_labels=link_labels
                )

                entity_loss = outputs.get('entity_loss')
                link_loss   = outputs.get('link_loss')

                if entity_loss is not None and link_loss is not None:
                    loss = entity_loss + self.linker_loss_weight * link_loss
                elif entity_loss is not None:
                    loss = entity_loss
                else:
                    loss = outputs['loss']

                loss = loss / self.gradient_accumulation_steps

            # bf16 backward (scaler.scale is a no-op when disabled)
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self._optimizer_step()

            total_loss  += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}"})

        # Flush last partial accumulation batch
        if num_batches % self.gradient_accumulation_steps != 0:
            self._optimizer_step()

        return total_loss / max(num_batches, 1)

    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        val_tp = val_fp = val_fn = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bbox           = batch["bbox"].to(self.device)
                pixel_values   = batch.get("pixel_values", None)
                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device)
                entity_labels  = batch.get("entity_labels", None)
                if entity_labels is not None:
                    entity_labels = entity_labels.to(self.device)
                link_labels    = batch.get("link_labels", None)
                if link_labels is not None:
                    link_labels = link_labels.to(self.device)

                with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        bbox=bbox,
                        pixel_values=pixel_values,
                        entity_labels=entity_labels,
                        link_labels=link_labels
                    )

                    entity_loss = outputs.get('entity_loss')
                    link_loss   = outputs.get('link_loss')

                    if entity_loss is not None and link_loss is not None:
                        loss = entity_loss + self.linker_loss_weight * link_loss
                    elif entity_loss is not None:
                        loss = entity_loss
                    else:
                        loss = outputs['loss']

                total_loss  += loss.item()
                num_batches += 1

                entity_logits     = outputs["entity_logits"]
                pred_labels       = torch.argmax(entity_logits, dim=-1)
                gt_labels         = entity_labels
                mask              = attention_mask

                key_val_mask      = ((gt_labels == 1) | (gt_labels == 2)) & (mask == 1)
                pred_key_val_mask = ((pred_labels == 1) | (pred_labels == 2)) & (mask == 1)

                val_tp += ((pred_labels == gt_labels) & key_val_mask).sum().item()
                val_fp += (pred_key_val_mask & ~key_val_mask).sum().item()
                val_fn += (~pred_key_val_mask & key_val_mask).sum().item()

        precision = val_tp / (val_tp + val_fp + 1e-8)
        recall    = val_tp / (val_tp + val_fn + 1e-8)
        val_f1    = 2 * precision * recall / (precision + recall + 1e-8)

        return total_loss / max(num_batches, 1), val_f1

    def train(self):
        """Full training loop with early stopping."""
        logger.info(f"Starting Stage 4b training | linker_loss_weight={self.linker_loss_weight}")
        logger.info(f"Output directory: {self.output_dir}")

        for epoch in range(self.num_epochs):
            logger.info(f"\n=== Epoch {epoch+1}/{self.num_epochs} ===")

            train_loss = self.train_epoch()
            logger.info(f"Train loss: {train_loss:.4f}")

            val_loss, val_f1 = self.validate()
            logger.info(f"Val loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

            self.training_history["epochs"].append(epoch + 1)
            self.training_history["train_losses"].append(train_loss)
            self.training_history["val_losses"].append(val_loss)
            self.training_history["val_f1s"].append(val_f1)

            ckpt_dir = self.output_dir / f"checkpoint-{epoch+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), ckpt_dir / "pytorch_model.bin")
            logger.info(f"Checkpoint saved to {ckpt_dir}")

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                best_dir = self.output_dir / "best_model"
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), best_dir / "pytorch_model.bin")
                logger.info(f"New best F1: {self.best_val_f1:.4f} (saved)")
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

    parser = argparse.ArgumentParser(description="Stage 4b Training: LayoutLMv3 with Linker")
    parser.add_argument("--data_dir",                    type=str,   default="../../data/prepared")
    parser.add_argument("--output_dir",                  type=str,   default="../../data/outputs/stage4b")
    parser.add_argument("--batch_size",                  type=int,   default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=16)
    parser.add_argument("--learning_rate",               type=float, default=5e-5)
    parser.add_argument("--num_epochs",                  type=int,   default=10)
    parser.add_argument("--early_stopping_patience",     type=int,   default=3)
    parser.add_argument("--val_fraction",                type=float, default=0.1)
    parser.add_argument("--linker_loss_weight",          type=float, default=1.0,
                        help="Linker loss weight \u03bb")
    parser.add_argument("--include_images",              action="store_true")
    parser.add_argument("--resume_from_checkpoint",      action="store_true")
    parser.add_argument("--pretrained_encoder",          type=str,   default=None,
                        help="Path to Stage 4a checkpoint (pytorch_model.bin)")

    args = parser.parse_args()

    # ── Checkpoint resumption ────────────────────────────────────────────────
    latest_ckpt = None
    if args.resume_from_checkpoint:
        output_path = Path(args.output_dir)
        if output_path.exists():
            checkpoints = sorted(
                [d for d in output_path.iterdir()
                 if d.is_dir() and d.name.startswith("checkpoint-")]
            )
            if checkpoints:
                latest_ckpt = checkpoints[-1]
                logger.info(f"Found checkpoint: {latest_ckpt}")

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Dataloaders ──────────────────────────────────────────────────────────
    dataloaders = create_stage4_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        include_images=args.include_images
    )
    train_loader = dataloaders['train']
    val_loader   = dataloaders['val']
    test_loader  = dataloaders['test']
    logger.info(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches | Test: {len(test_loader)} batches")

    # ── Model (CPU first to avoid memory spike on load) ────────────────────────
    model = LayoutLMv3KVPModel(use_linker=True)
    logger.info("LayoutLMv3KVPModel (with linker) created on CPU")

    # ── Transfer Stage 4a weights ────────────────────────────────────────────
    # Both stages share identical key names (encoder.*, entity_classifier.*).
    # linker.* keys are Stage 4b-only and stay randomly initialised.
    if args.pretrained_encoder and not latest_ckpt:
        pretrained_path = Path(args.pretrained_encoder)
        if pretrained_path.exists():
            state_dict  = torch.load(pretrained_path, map_location="cpu")
            model_state = model.state_dict()
            transferred, skipped = [], []

            for key, value in state_dict.items():
                if key in model_state:
                    model_state[key] = value
                    transferred.append(key)
                else:
                    skipped.append(key)

            model.load_state_dict(model_state, strict=False)

            enc_keys = [k for k in transferred if k.startswith('encoder.')]
            cls_keys = [k for k in transferred if k.startswith('entity_classifier.')]
            logger.info(f"\u2713 Transferred {len(transferred)} keys on CPU (skipped {len(skipped)})")
            logger.info(f"  Encoder keys: {len(enc_keys)}, Classifier keys: {len(cls_keys)}")

            if len(enc_keys) == 0:
                sample = list(state_dict.keys())[:10]
                logger.error(f"ERROR: 0 encoder keys transferred! Checkpoint samples: {sample}")
                raise RuntimeError("Pretrained encoder transfer failed \u2014 0 encoder keys matched.")
        else:
            logger.warning(f"Pretrained encoder path not found: {pretrained_path}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Stage4bTrainer(
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

    # ── Resume from checkpoint ───────────────────────────────────────────────
    if latest_ckpt:
        model_path = latest_ckpt / "pytorch_model.bin"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            logger.info(f"\u2713 Loaded weights from {latest_ckpt}")
        trainer.restore_history(latest_ckpt)

    # ── Train ────────────────────────────────────────────────────────────────
    trainer.train()
    logger.info(f"Training complete! Best validation F1: {trainer.best_val_f1:.4f}")


if __name__ == "__main__":
    main()
