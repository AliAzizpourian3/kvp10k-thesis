"""
Stage 4a Training: LayoutLMv3 KVP Extraction (No Linker).

Implements:
- LayoutLMv3 encoder + entity classifier only
- Training from prepared JSON data (data/prepared/train/*.json)
- Validation with early stopping (patience=3 on F1)
- Checkpoint saving to data/outputs/stage4a/
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


class Stage4aTrainer:
    """Trainer for Stage 4a: LayoutLMv3 without linker."""
    
    def __init__(
        self,
        model: LayoutLMv3KVPModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate=5e-5,
        num_epochs=10,
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        early_stopping_patience=3,
        output_dir="data/outputs/stage4a",
        device=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.output_dir = output_dir
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Setup learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'best_epoch': 0,
            'best_val_f1': 0.0,
            'epochs_no_improve': 0
        }
        
        logger.info(f"✓ Initialized Stage 4a Trainer")
        logger.info(f"  Model: LayoutLMv3 (no linker)")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  Early stopping patience: {early_stopping_patience}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Output dir: {output_dir}")
    
    def train_epoch(self):
        """Single training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                bbox=batch['bbox'],
                pixel_values=batch.get('pixel_values'),
                entity_labels=batch['entity_labels'],
                link_labels=None  # No linking for Stage 4a
            )
            
            loss = outputs['loss']
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            # Only step optimizer every N batches
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps  # Undo scaling for logging
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self):
        """Validation pass with F1 computation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # For F1: track TP, FP, FN across all val batches
        val_tp = 0
        val_fp = 0
        val_fn = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch_idx, batch in enumerate(pbar):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    bbox=batch['bbox'],
                    pixel_values=batch.get('pixel_values'),
                    entity_labels=batch['entity_labels'],
                    link_labels=None
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                num_batches += 1
                
                # Compute token-level F1 (simplified: count correct entity predictions)
                pred_labels = torch.argmax(outputs['entity_logits'], dim=-1)
                gt_labels = batch['entity_labels']
                mask = batch['attention_mask']
                
                # Count matches for non-padding, non-Other tokens (1=Key, 2=Value)
                key_val_mask = ((gt_labels == 1) | (gt_labels == 2)) & (mask == 1)
                pred_key_val_mask = ((pred_labels == 1) | (pred_labels == 2)) & (mask == 1)
                
                tp = ((pred_labels == gt_labels) & key_val_mask).sum().item()
                fp = (pred_key_val_mask & ~key_val_mask).sum().item()
                fn = (~pred_key_val_mask & key_val_mask).sum().item()
                
                val_tp += tp
                val_fp += fp
                val_fn += fn
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        
        # Compute F1
        precision = val_tp / (val_tp + val_fp + 1e-8)
        recall = val_tp / (val_tp + val_fn + 1e-8)
        val_f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return avg_loss, val_f1
    
    def train(self):
        """Full training loop with early stopping."""
        logger.info("Starting training...")
        logger.info("=" * 80)
        
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"\n--- Epoch {epoch}/{self.num_epochs} ---")
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_f1 = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            logger.info(f"Val loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            
            # Early stopping check
            if val_f1 > self.history['best_val_f1']:
                self.history['best_val_f1'] = val_f1
                self.history['best_epoch'] = epoch
                self.history['epochs_no_improve'] = 0
                
                # Save checkpoint
                checkpoint_dir = Path(self.output_dir) / f"checkpoint-{epoch}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")
                with open(checkpoint_dir / "config.json", "w") as f:
                    json.dump({
                        'epoch': epoch,
                        'val_f1': val_f1,
                        'val_loss': val_loss,
                        'train_loss': train_loss
                    }, f, indent=2)
                
                logger.info(f"✓ Best F1 so far! Saved checkpoint to {checkpoint_dir}")
            else:
                self.history['epochs_no_improve'] += 1
                logger.info(f"No improvement for {self.history['epochs_no_improve']} epoch(s)")
                
                if self.history['epochs_no_improve'] >= self.early_stopping_patience:
                    logger.info(f"\nEarly stopping (patience={self.early_stopping_patience} reached)")
                    break
        
        # Save final training history
        history_file = Path(self.output_dir) / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"\n✓ Training complete! Best val F1: {self.history['best_val_f1']:.4f} at epoch {self.history['best_epoch']}")
        logger.info(f"History saved to {history_file}")
        
        return self.history


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Stage 4a: LayoutLMv3 without linker")
    parser.add_argument(
        "--data_dir",
        default="data/prepared",
        help="Path to prepared train/val/test data"
    )
    parser.add_argument(
        "--output_dir",
        default="data/outputs/stage4a",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (will be multiplied by gradient_accumulation_steps)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch_size * accumulation)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (LayoutLMv3 paper default)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience (stop if no improvement for N epochs)"
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation"
    )
    parser.add_argument(
        "--include_images",
        action="store_true",
        help="Include image features (slower, more memory)"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume training from latest checkpoint (auto-detects)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("STAGE 4a: LayoutLMv3 KVP Extraction (No Linker)")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Num epochs: {args.num_epochs}")
    logger.info(f"Early stopping patience: {args.early_stopping_patience}")
    logger.info(f"Val fraction: {args.val_fraction}")
    logger.info(f"Include images: {args.include_images}")
    
    # Check for checkpoint resumption
    latest_ckpt = None
    if args.resume_from_checkpoint:
        output_path = Path(args.output_dir)
        if output_path.exists():
            checkpoints = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
            if checkpoints:
                latest_ckpt = checkpoints[-1]
                logger.info(f"Found checkpoint: {latest_ckpt}")
    
    # Create data loaders
    logger.info("\nLoading data...")
    dataloaders = create_stage4_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        include_images=args.include_images
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("\nCreating model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(
        model_name="microsoft/layoutlmv3-base",
        freeze_base=False,
        use_linker=False,  # Stage 4a: no linker
        device=device
    )
    
    # Load checkpoint if resuming
    if latest_ckpt:
        logger.info(f"\nResuming training from {latest_ckpt}...")
        model_path = latest_ckpt / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            logger.info(f"✓ Loaded model weights from checkpoint")
    
    # Create trainer
    trainer = Stage4aTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
        device=device
    )
    
    # Train
    history = trainer.train()
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Stage 4a training complete!")
    logger.info(f"Best validation F1: {history['best_val_f1']:.4f}")
    logger.info(f"Best model saved in checkpoint-{history['best_epoch']}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
