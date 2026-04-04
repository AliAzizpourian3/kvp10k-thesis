# Commands Cheat Sheet (Only What You Need Now)

Use these from the project root:

```bash
cd "$WORK/kvp10k_thesis"
```

This just moves you into the project folder first.

## 1. Quick Status

### Check queue

```bash
squeue -u "$USER"
```

This shows if your jobs are waiting (`PD`), running (`R`), or finished/failed (gone from queue).

### Check A100 health

```bash
sinfo -p a100 && echo && sinfo -R | head -20
```

This checks if A100 nodes are available and whether any nodes are reported as problematic.

## 2. Stage 4b Monitoring

### List latest Stage 4b logs

```bash
ls -lt logs/kvp_stage4b_canary*.err | head -10
```

Shows your newest canary/experiment log files.

### Check for errors in a log

```bash
grep -E "ERROR|NaN|RuntimeError|OutOfMemory" logs/kvp_stage4b_canary_B-<JOB_ID>.err | tail -20
```

### Check training progress (epoch/loss/F1)

```bash
grep -E "Epoch|Train loss|Val loss|Val F1|best|Early" logs/kvp_stage4b_canary_B-<JOB_ID>.err
```

### Follow a log live (replace job id)

```bash
tail -f logs/kvp_stage4b_canary_B-<JOB_ID>.err
```

## 3. Submit / Re-run Jobs

### Run a new canary (main experiment, pos_weight fix)

```bash
sbatch slurm/submit_stage4b_canary_B.sh
```

### Run the lambda sweep (after canary confirms linker works)

```bash
sbatch logs/stage4b_lambda05.sbatch
sbatch logs/stage4b_lambda10.sbatch
sbatch logs/stage4b_lambda20.sbatch
```

### Evaluate a trained checkpoint

```bash
sbatch slurm/submit_eval.sh data/outputs/stage4b_canary_B
# or for lambda runs:
sbatch slurm/submit_eval.sh data/outputs/stage4b_lambda10
```

Results saved to `eval_results.json` inside the checkpoint dir.

### Confirm queued

```bash
squeue -u "$USER"
```

## 4. Fast Preflight Checks

### Confirm Stage 4a checkpoint exists (required for transfer)

```bash
ls -l data/outputs/stage4a/best_model/
```

### Confirm train script compiles

```bash
env/kvp10k_env/bin/python -m py_compile code/script/train_stage4b.py && echo OK
```

## 5. Useful Paths

- Project root: /home/woody/iwi5/iwi5413h/kvp10k_thesis
- Outputs: data/outputs/  (stage4b_canary_B, stage4b_lambda05/10/20)
- Slurm scripts: slurm/
- Lambda sbatch files: logs/stage4b_lambda*.sbatch

Tip — quick restart after a long break:

```bash
git pull origin master
squeue -u "$USER"
```