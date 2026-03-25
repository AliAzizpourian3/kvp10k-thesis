"""
IBM Mistral-7B LoRA Baseline for KVP10k

Faithful reproduction of IBM's training pipeline using their exact
hyper-parameters from config/base.yaml + config/kvp.yaml.

IBM configuration:
  Model:    mistralai/Mistral-7B-Instruct-v0.2, 4-bit QLoRA (NF4)
  LoRA:     r=4, alpha=4, dropout=0.05, bias=none
            targets (IBM):  ["out_proj", "up_proj", "down_proj", "Wqkv"]
            targets (HF):   ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj"]
  Training: lr=5e-4 AdamW, 8 epochs, batch=1, grad_accum=4, max_len=8192, seed=0
  Prompt:   LMDX format (<Document>…</Document><Task>…</Task>### Response:)
  Target:   sorted list-of-lists with quantized bboxes
  Inference: greedy decoding (temperature=0)

Differences from IBM:
  1. HuggingFace Trainer (not PyTorch Lightning)
  2. PyMuPDF text extraction (not Tesseract OCR)
  3. LoRA target_modules use HF names (see above)
  4. 4-bit QLoRA quantization (IBM uses bf16) — needed for A100 40GB
  5. paged_adamw_8bit optimizer + gradient checkpointing for memory

Prerequisites:
  Run prepare_data.py first to create data/prepared/{train,test}/*.json

Usage:
  python mistral_baseline.py train   --data_dir ../../data/prepared --output_dir ./out
  python mistral_baseline.py predict --checkpoint ./out/checkpoint  --data_dir ../../data/prepared --output_dir ./predictions
"""

import os
import sys
import ast
import json
import glob
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# IBM Configuration (exact values from config/base.yaml + config/kvp.yaml)
# ============================================================================
IBM_CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "max_length": 8192,
    "lora_r": 4,
    "lora_alpha": 4,
    "lora_dropout": 0.05,
    # IBM: ["out_proj","up_proj","down_proj","Wqkv"]  →  HF equivalents:
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    "lr": 5e-4,
    "epochs": 8,
    "batch_size": 1,
    "grad_accum": 4,   # effective batch = 4
    "seed": 0,
    "bf16": True,
    "max_new_tokens": 2048,
}


# ============================================================================
# Dataset  — loads pre-computed LMDX prompts from prepare_data.py output
# ============================================================================
class KVP10kMistralDataset(TorchDataset):
    """
    Reads per-sample JSONs (from prepare_data.py) and tokenises them.
    Labels mask the prompt tokens with -100 so the loss is computed only
    on the model's response (matching IBM's training loop).
    """

    def __init__(self, data_dir: str, tokenizer, max_length: int, split: str = "train"):
        self.tokenizer = tokenizer
        self.max_length = max_length

        data_path = Path(data_dir) / split
        if not data_path.exists():
            data_path = Path(data_dir)          # caller may already point at split dir

        json_files = sorted(glob.glob(str(data_path / "*.json")))
        logger.info(f"Scanning {len(json_files)} files in {data_path} …")

        self.samples: List[Dict[str, str]] = []
        skipped = 0
        for fpath in json_files:
            try:
                with open(fpath) as fh:
                    data = json.load(fh)
                prompt = data.get("full_prompt", "")
                target = data.get("target_text", "[]")
                if not prompt or target == "[]":
                    skipped += 1
                    continue
                self.samples.append({
                    "prompt": prompt,
                    "target": target,
                    "hash_name": data.get("hash_name", ""),
                })
            except Exception:
                skipped += 1

        logger.info(f"Loaded {len(self.samples)} samples  (skipped {skipped})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        full_text = s["prompt"] + s["target"] + self.tokenizer.eos_token

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)

        # ---- mask prompt tokens so loss is only on the response ----
        prompt_enc = self.tokenizer(
            s["prompt"], truncation=True, max_length=self.max_length, add_special_tokens=False,
        )
        prompt_len = len(prompt_enc["input_ids"])

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attn_mask == 0] = -100

        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


# ============================================================================
# Model + LoRA setup
# ============================================================================
def _build_model_and_tokenizer(cfg: Dict = None):
    """Load Mistral-7B in 4-bit (QLoRA) and attach LoRA adapters."""
    cfg = cfg or IBM_CONFIG
    name = cfg["model_name"]
    logger.info(f"Loading {name} in 4-bit QLoRA mode …")

    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # 4-bit quantization config (QLoRA: NF4 + double quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg["bf16"] else torch.float16,
    )

    model_kw: Dict[str, Any] = {
        "quantization_config": bnb_config,
        "device_map": "auto",
    }

    # Use flash_attention_2 if available, else sdpa (PyTorch 2.x default)
    try:
        import flash_attn  # noqa: F401
        model_kw["attn_implementation"] = "flash_attention_2"
        logger.info("Using flash_attention_2")
    except ImportError:
        model_kw["attn_implementation"] = "sdpa"
        logger.info("flash-attn not installed, using SDPA attention")

    model = AutoModelForCausalLM.from_pretrained(name, **model_kw)
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["lora_targets"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tok


# ============================================================================
# Training
# ============================================================================
def train_mistral(data_dir: str, output_dir: str, cfg: Dict = None, resume_from_checkpoint: str = None):
    """Fine-tune Mistral-7B with LoRA on prepared KVP10k data."""
    cfg = cfg or IBM_CONFIG
    model, tok = _build_model_and_tokenizer(cfg)

    train_ds = KVP10kMistralDataset(data_dir, tok, cfg["max_length"], split="train")
    eval_ds = None
    if (Path(data_dir) / "test").exists():
        eval_ds = KVP10kMistralDataset(data_dir, tok, cfg["max_length"], split="test")
        if len(eval_ds) == 0:
            eval_ds = None

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        bf16=cfg["bf16"],
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds else "no",
        save_total_limit=2,
        seed=cfg["seed"],
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)
    if resume_from_checkpoint:
        logger.info(f"Resuming training from {resume_from_checkpoint}")
    else:
        logger.info("Starting training …")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    ckpt_dir = os.path.join(output_dir, "checkpoint")
    trainer.save_model(ckpt_dir)
    tok.save_pretrained(ckpt_dir)
    logger.info(f"Saved checkpoint → {ckpt_dir}")
    return trainer


# ============================================================================
# Inference
# ============================================================================
def predict_mistral(checkpoint_dir: str, data_dir: str, output_dir: str, cfg: Dict = None):
    """Generate predictions using a trained checkpoint."""
    cfg = cfg or IBM_CONFIG
    logger.info(f"Loading checkpoint from {checkpoint_dir}")

    tok = AutoTokenizer.from_pretrained(checkpoint_dir)
    tok.pad_token = tok.eos_token

    # QLoRA checkpoint: load adapter with 4-bit base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg["bf16"] else torch.float16,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_dir,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    data_path = Path(data_dir)
    if (data_path / "test").exists():
        data_path = data_path / "test"

    json_files = sorted(glob.glob(str(data_path / "*.json")))
    logger.info(f"Predicting {len(json_files)} samples …")
    os.makedirs(output_dir, exist_ok=True)

    for i, fpath in enumerate(json_files):
        if i % 50 == 0:
            logger.info(f"  {i}/{len(json_files)}")

        with open(fpath) as fh:
            data = json.load(fh)

        prompt = data.get("full_prompt", "")
        if not prompt:
            continue

        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=cfg["max_length"])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=cfg["max_new_tokens"],
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

        response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        kvps = _parse_response(response)

        pred_file = Path(output_dir) / f"{data.get('hash_name', f'sample_{i}')}.json"
        with open(pred_file, "w") as fh:
            json.dump({"kvps_list": kvps}, fh, indent=2)

    logger.info(f"Wrote {len(json_files)} prediction files → {output_dir}")


def _parse_entity(s: str) -> Tuple[str, Optional[List[int]]]:
    """Parse 'text left|top|right|bottom' → (text, [l,t,r,b]) or (text, None)."""
    parts = s.rsplit(" ", 1)
    if len(parts) == 2:
        try:
            bbox = [int(x) for x in parts[1].split("|")]
            if len(bbox) == 4:
                return parts[0], bbox
        except ValueError:
            pass
    return s, None


def _parse_response(text: str) -> List[Dict]:
    """Parse model output (Python list-of-lists) into IBM KVP format."""
    text = text.strip()
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []

    if not isinstance(parsed, list):
        return []

    kvps = []
    for item in parsed:
        if not isinstance(item, list) or not item:
            continue
        k_text, k_bbox = _parse_entity(str(item[0]))
        if len(item) >= 2:
            v_text, v_bbox = _parse_entity(str(item[1]))
            kvp: Dict[str, Any] = {
                "type": "kvp",
                "key":   {"text": k_text},
                "value": {"text": v_text},
            }
            if k_bbox:
                kvp["key"]["bbox"] = k_bbox
            if v_bbox:
                kvp["value"]["bbox"] = v_bbox
        else:
            kvp = {"type": "unvalued", "key": {"text": k_text}}
            if k_bbox:
                kvp["key"]["bbox"] = k_bbox
        kvps.append(kvp)
    return kvps


# ============================================================================
# Convenience wrappers (called from main.py / Slurm scripts)
# ============================================================================
def train_mistral_baseline(data_dir: str, output_dir: str):
    """Entry-point used by main.py."""
    return train_mistral(data_dir=data_dir, output_dir=output_dir)


def predict_mistral_baseline(data_dir: str, checkpoint_dir: str, output_dir: str):
    """Entry-point used by main.py."""
    return predict_mistral(
        checkpoint_dir=checkpoint_dir,
        data_dir=data_dir,
        output_dir=output_dir,
    )


# ============================================================================
# CLI
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="Mistral-7B LoRA – KVP10k")
    sub = p.add_subparsers(dest="cmd")

    tp = sub.add_parser("train")
    tp.add_argument("--data_dir",   required=True, help="Root of prepared data (with train/ and test/ sub-dirs)")
    tp.add_argument("--output_dir", required=True, help="Where to store checkpoints and logs")
    tp.add_argument("--resume_from_checkpoint", default=None, help="Path to checkpoint dir to resume from")

    pp = sub.add_parser("predict")
    pp.add_argument("--checkpoint", required=True, help="Path to saved checkpoint dir")
    pp.add_argument("--data_dir",   required=True, help="Root of prepared data (test/ sub-dir)")
    pp.add_argument("--output_dir", required=True, help="Where to write prediction JSONs")

    args = p.parse_args()
    if args.cmd == "train":
        train_mistral(data_dir=args.data_dir, output_dir=args.output_dir, resume_from_checkpoint=args.resume_from_checkpoint)
    elif args.cmd == "predict":
        predict_mistral(checkpoint_dir=args.checkpoint, data_dir=args.data_dir, output_dir=args.output_dir)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
