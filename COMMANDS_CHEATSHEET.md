# Commands Cheat Sheet

This file collects the most useful project commands in one place.

Use these from the project root unless noted otherwise.

## 1. Most Important Right Now

### Check the Slurm queue

```bash
squeue -u "$USER"
```

### Check A100 partition health

```bash
sinfo -p a100 && echo && sinfo -R | head -20
```

### Follow the current Mistral job log

```bash
grep -E "[0-9]+/9976|epoch|eval_loss" "$WORK/kvp10k_thesis/logs/kvp_stage3_mistral-1550434.out" | tail -30
```

and 

```bash
grep -oE "[0-9]+/9976" "$WORK/kvp10k_thesis/logs/kvp_stage3_mistral-1550434.out" | tail -1
```

```bash
tail -f "$WORK/kvp10k_thesis/logs/kvp_stage3_mistral-1546224.out"
```

### Check whether training outputs are appearing

```bash
find "$WORK/kvp10k_thesis/data/outputs/stage3_mistral" -maxdepth 2 -type d -name 'checkpoint*' | sort
```

### Check whether predictions were written

```bash
ls "$WORK/kvp10k_thesis/data/outputs/stage3_mistral/predictions/" | wc -l
```

### Read the final evaluation file

```bash
cat "$WORK/kvp10k_thesis/data/outputs/stage3_mistral/evaluation.json"
```

## 2. Re-run Stage 3 Mistral

### Submit the Stage 3 Mistral job

```bash
cd "$WORK/kvp10k_thesis"
sbatch logs/stage3_mistral.sbatch
```

### Watch the job queue after submission

```bash
squeue -u "$USER"
```

## 3. Data Preparation Commands

### Count prepared train pages

```bash
ls "$WORK/kvp10k_thesis/data/prepared/train/" | wc -l
```

### Count prepared test pages

```bash
ls "$WORK/kvp10k_thesis/data/prepared/test/" | wc -l
```

### Check the end of the train-preparation log

```bash
tail -30 "$WORK/kvp10k_thesis/logs/prepare_train.out"
```

### Run preparation manually on the login node

```bash
cd "$WORK/kvp10k_thesis/code/script"
nohup bash "$WORK/kvp10k_thesis/logs/stage3_prepare_data.sbatch" > "$WORK/kvp10k_thesis/logs/prepare_data.out" 2>&1 &
```

## 4. Logs And Failure Diagnosis

### Read the previous failed Mistral log

```bash
tail -50 "$WORK/kvp10k_thesis/logs/kvp_stage3_mistral-1545825.out"
```

### List relevant logs

```bash
ls "$WORK/kvp10k_thesis/logs/"*prepare* "$WORK/kvp10k_thesis/logs/"*mistral* "$WORK/kvp10k_thesis/logs/"*submit*
```

## 5. Environment Checks

### Check that bitsandbytes is installed

```bash
"$WORK/kvp10k_thesis/env/kvp10k_env/bin/python" -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

### Validate the Mistral training script syntax

```bash
"$WORK/kvp10k_thesis/env/kvp10k_env/bin/python" -c "import ast; ast.parse(open('$WORK/kvp10k_thesis/code/script/mistral_baseline.py').read()); print('Syntax OK')"
```

### Check GPU visibility inside a job environment

```bash
"$WORK/kvp10k_thesis/env/kvp10k_env/bin/python" -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## 6. Thesis And Reporting

### Compile the thesis

```bash
cd "$WORK/kvp10k_thesis/LaTeX_Thesis"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

### Open the supervisor report file

File:
[LaTeX_Thesis/SUPERVISOR_PROGRESS_REPORTS.md](/home/woody/iwi5/iwi5413h/kvp10k_thesis/LaTeX_Thesis/SUPERVISOR_PROGRESS_REPORTS.md)

### Open the presentation brief

File:
[LaTeX_Thesis/PRESENTATION_1_BRIEF.md](/home/woody/iwi5/iwi5413h/kvp10k_thesis/LaTeX_Thesis/PRESENTATION_1_BRIEF.md)

## 7. Useful Paths

- Project root: `/home/woody/iwi5/iwi5413h/kvp10k_thesis`
- Prepared data: `/home/woody/iwi5/iwi5413h/kvp10k_thesis/data/prepared`
- Mistral outputs: `/home/woody/iwi5/iwi5413h/kvp10k_thesis/data/outputs/stage3_mistral`
- Logs: `/home/woody/iwi5/iwi5413h/kvp10k_thesis/logs`
- Thesis: `/home/woody/iwi5/iwi5413h/kvp10k_thesis/LaTeX_Thesis`

## 8. Current Job IDs (Stage 4 - Tensor Reshape Fix)

**Stage 4a (Entity Classification):**
- Job ID: `1557881` (kvp_stage4a_layout) — PENDING ✅

**Stage 4b (Entity + Relation, λ Sweep):**
- Lambda 0.5: `1557882` (kvp_stage4b_l05) — PENDING ✅
- Lambda 1.0: `1557883` (kvp_stage4b_l10) — PENDING ✅
- Lambda 2.0: `1557884` (kvp_stage4b_l20) — PENDING ✅

**Status**: 4 PENDING on A100

### NEW FIX: Tensor Contiguity (Commit f417fd5):

**Bug**: RuntimeError: `view size is not compatible with input tensor's size and stride`
- After truncating entity_logits to text_seq_len, the tensor becomes non-contiguous in memory
- `.view()` requires contiguous tensors and fails
- `.reshape()` handles non-contiguous tensors automatically

**Fix Applied**:
- Replaced all `.view()` calls with `.reshape()` in loss computation
- Line 361: `entity_logits.reshape(-1, self.num_labels)[active_loss]`
- Line 362: `entity_labels.reshape(-1)[active_loss]`
- Also line 360: `active_loss = attention_mask.reshape(-1) == 1`

### All 4 Fixes Now Applied:

1. ✅ **layoutlm_model.py (1ce23fd)** — Variable shadowing fix
2. ✅ **train_stage4a.py (6bdde45)** — Checkpoint history restoration  
3. ✅ **train_stage4b.py (de38918)** — Checkpoint history restoration
4. ✅ **layoutlm_model.py (f417fd5)** — Tensor reshape for contiguity ← **NEW**

### LayoutLMv3 Patch Token Truncation Fix (ROOT CAUSE):

**Root Cause**: LayoutLMv3 encoder appends ~197 visual patch tokens from ViT patch embeddings to the sequence:
- `input_ids` shape: [batch=4, seq=512]
- `sequence_output` from encoder: [batch=4, seq=709, hidden] ← 197 visual tokens!
- `attention_mask`: [batch=4, seq=512] ← only covers text tokens
- This caused: `entity_logits` [2836, 3] vs `attention_mask` [2048] shape mismatch

**Fix Applied in layoutlm_model.py**:
1. ✅ After entity_logits = self.entity_classifier(sequence_output):
   - Truncate entity_logits to text-only: `entity_logits[:, :text_seq_len, :]`
   - Truncate sequence_output to text-only: `sequence_output[:, :text_seq_len, :]`
2. ✅ Pass truncated sequence_output to linker (same issue would occur in BiaffineLinker)

**Result**: All tensors now aligned:
- entity_logits: [batch, 512, 3]
- attention_mask: [batch, 512]
- entity_labels: [batch, 512]
- bbox: [batch, 512, 4]
- link_labels: [batch, 512, 512]

This is the standard LayoutLMv3 fine-tuning approach from the official code.

**All files compiled successfully** ✅

**Monitor Stage 4a:**
```bash
tail -f logs/kvp_stage4a_layout-1557881.out
```