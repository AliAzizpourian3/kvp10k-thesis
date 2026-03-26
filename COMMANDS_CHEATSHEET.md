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
ls -1 logs/kvp_stage4b_l*.out | tail -20
```

This lists your newest Stage 4b output log files.

### Check latest errors/warnings in all Stage 4b logs

```bash
grep -E "ERROR|Traceback|RuntimeError|OutOfMemory|NaN" logs/kvp_stage4b_l*.out | tail -50
```

This quickly shows only bad signs (crashes, OOM, NaN) from all Stage 4b logs.

### Check training progress (epoch/loss/F1)

```bash
grep -E "Epoch|Train loss|Val loss|Val F1|best" logs/kvp_stage4b_l*.out | tail -80
```

This shows whether training is actually progressing (epochs, loss, validation F1).

### Follow one log live (replace job id)

```bash
tail -f logs/kvp_stage4b_l05-<JOB_ID>.out
```

Use this when a job starts running and you want live updates.

## 3. Clean + Resubmit Stage 4b (All 3)

### Cancel current Stage 4b jobs (replace IDs)

```bash
scancel <JOB1> <JOB2> <JOB3>
```

Stops old jobs so they do not waste GPU time.

### Pull latest fix

```bash
git pull origin master
```

Downloads the newest code fixes from GitHub.

### Clean old Stage 4b outputs

```bash
rm -rf data/outputs/stage4b_lambda*/
```

Deletes old failed checkpoints so the next run starts clean.

### Submit all three lambdas

```bash
sbatch logs/stage4b_lambda05.sbatch
sbatch logs/stage4b_lambda10.sbatch
sbatch logs/stage4b_lambda20.sbatch
```

Starts the 3 Stage 4b experiments (lambda 0.5, 1.0, 2.0).

### Confirm queued

```bash
squeue -u "$USER"
```

Run this right after `sbatch` to confirm all 3 jobs were accepted.

## 4. Fast Preflight Checks

### Confirm Stage 4a checkpoint exists (required for transfer)

```bash
ls -l data/outputs/stage4a/checkpoint-9/model.pt
```

This must exist, otherwise Stage 4b cannot load pretrained Stage 4a weights.

### Confirm train script compiles

```bash
"$WORK/kvp10k_thesis/env/kvp10k_env/bin/python" -m py_compile code/script/train_stage4b.py
```

Quick syntax check before submitting jobs.

## 5. Useful Paths

- Project root: /home/woody/iwi5/iwi5413h/kvp10k_thesis
- Stage 4b outputs: /home/woody/iwi5/iwi5413h/kvp10k_thesis/data/outputs
- Logs: /home/woody/iwi5/iwi5413h/kvp10k_thesis/logs

Tip: if you are in a hurry, use only these 4 commands in order:

```bash
git pull origin master
rm -rf data/outputs/stage4b_lambda*/
sbatch logs/stage4b_lambda05.sbatch && sbatch logs/stage4b_lambda10.sbatch && sbatch logs/stage4b_lambda20.sbatch
squeue -u "$USER"
```