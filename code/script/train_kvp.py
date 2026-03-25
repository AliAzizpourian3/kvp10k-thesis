"""
Training script for LayoutLMv3 KVP extraction model.

Implements:
- Training loop with entity classification loss
- Validation and checkpointing
- Progress tracking
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
import config
from layoutlm_model import create_model, LayoutLMv3KVPModel
from kvp_dataset import KVP10kDataset, create_dataloaders


class KVPTrainer:
    """
    Trainer for LayoutLMv3 KVP extraction.
    """
    
    def __init__(
        self,
        model: LayoutLMv3KVPModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate=5e-5,
        num_epochs=3,
        warmup_ratio=0.1,
        output_dir="outputs/stage4_model",
        device=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.device = device if device else config.DEVICE
        
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
            'val_f1': []
        }
        
        print(f"✓ Initialized KVP Trainer")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Total steps: {total_steps}")
        print(f"  Output dir: {output_dir}")
        print(f"  Device: {self.device}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch in pbar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            bbox = batch['bbox'].to(self.device)
            entity_labels = batch['entity_labels'].to(self.device)
            
            pixel_values = None
            if 'pixel_values' in batch:
                pixel_values = batch['pixel_values'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                entity_labels=entity_labels
            )
            
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # For entity classification metrics
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                bbox = batch['bbox'].to(self.device)
                entity_labels = batch['entity_labels'].to(self.device)
                
                pixel_values = None
                if 'pixel_values' in batch:
                    pixel_values = batch['pixel_values'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    entity_labels=entity_labels
                )
                
                loss = outputs['loss']
                entity_logits = outputs['entity_logits']
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions for metrics
                preds = torch.argmax(entity_logits, dim=-1)
                
                # Only consider non-padding tokens
                mask = attention_mask.bool()
                
                all_preds.extend(preds[mask].cpu().numpy())
                all_labels.extend(entity_labels[mask].cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Compute entity classification F1
        from sklearn.metrics import f1_score, classification_report
        
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        
        print(f"\nValidation Metrics:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (micro): {f1_micro:.4f}")
        
        # Detailed classification report
        print("\nEntity Classification Report:")
        print(classification_report(
            all_labels,
            all_preds,
            target_names=['Other', 'Key', 'Value'],
            zero_division=0
        ))
        
        return avg_loss, f1_macro
    
    def train(self):
        """Run complete training loop."""
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        
        best_val_f1 = 0.0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 80)
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Training loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_f1 = self.validate()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            
            # Save checkpoint if best
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_checkpoint(epoch, val_f1, is_best=True)
                print(f"✓ Saved best model (F1: {val_f1:.4f})")
            
            # Save regular checkpoint
            self.save_checkpoint(epoch, val_f1, is_best=False)
        
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
        print(f"Best validation F1: {best_val_f1:.4f}")
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, epoch, val_f1, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_f1': val_f1,
            'history': self.history
        }
        
        if is_best:
            path = os.path.join(self.output_dir, 'best_model.pt')
        else:
            path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.output_dir, 'training_history.json')
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"✓ Saved training history to {history_path}")
    
    @staticmethod
    def load_checkpoint(model, checkpoint_path, device=None):
        """Load model from checkpoint."""
        if device is None:
            device = config.DEVICE
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']+1}")
        print(f"  Val F1: {checkpoint['val_f1']:.4f}")
        
        return model, checkpoint


def train_kvp_model(
    train_dataset,
    val_dataset,
    batch_size=4,
    learning_rate=2e-5,
    num_epochs=3,
    freeze_base=True,
    use_linker=True,  # NEW: Enable/disable linking module
    output_dir="outputs/stage4_model"
):
    """
    Complete training pipeline for KVP extraction.
    
    Args:
        train_dataset: HuggingFace dataset for training
        val_dataset: HuggingFace dataset for validation
        batch_size: Training batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        freeze_base: Whether to freeze LayoutLMv3 base
        use_linker: Whether to use linking module (Stage 4b) or not (Stage 4a)
        output_dir: Where to save checkpoints
    
    Returns:
        Trained model and training history
    """
    # Create processor
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base",
        apply_ocr=False  # We have our own annotations
    )
    
    # Create PyTorch datasets
    train_ds = KVP10kDataset(
        train_dataset,
        processor,
        max_seq_length=512,
        include_images=False  # Set to True if GPU available for image features
    )
    
    val_ds = KVP10kDataset(
        val_dataset,
        processor,
        max_seq_length=512,
        include_images=False
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_ds,
        val_ds,
        batch_size=batch_size,
        num_workers=0  # Windows compatibility
    )
    
    # Create model with or without linker
    model = create_model(
        model_name="microsoft/layoutlmv3-base",
        freeze_base=freeze_base,
        use_linker=use_linker  # Pass through to model creation
    )
    
    # Create trainer
    trainer = KVPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        output_dir=output_dir
    )
    
    # Train
    history = trainer.train()
    
    return model, history, trainer


if __name__ == "__main__":
    print("KVP Trainer module ready")
