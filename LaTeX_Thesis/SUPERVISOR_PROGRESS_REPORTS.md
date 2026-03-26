# Supervisor Progress Reports

This file is a running log for the regular progress updates sent to the supervisor.

Each reporting period should be updated in one place with:
- work completed
- current status
- results available so far
- blockers and risks
- next steps

## Report 01

Date: 2026-03-08

### Scope of this period

- thesis setup and problem framing
- implementation of the staged KVP extraction pipeline
- completion of earlier non-neural stages
- reconstruction of the Stage 3 Mistral baseline workflow
- preparation of the first learned-baseline training run on TinyGPU

### What was completed

- defined the overall thesis direction around key-value pair extraction from business documents using KVP10k
- organized the implementation into a staged pipeline:
	- Stage 0: evaluation protocol
	- Stage 1: dataset ingestion and preprocessing support
	- Stage 2: layout clustering and analysis
	- Stage 3: baseline implementation and evaluation
	- Stage 4: main model stage planned for later work
- completed the earlier Stage 2 layout-clustering pipeline
- completed the earlier Stage 3 heuristic baseline pipeline
- verified that the older Stage 3 ground-truth path was not sufficient for a trustworthy Mistral baseline
- created `prepare_data.py` to convert raw KVP10k PDF sources into page-level JSON training samples
- rewrote `mistral_baseline.py` so it trains on prepared LMDX prompt/target data instead of incorrect raw-row assumptions
- added `evaluate_mistral.py` for entity-level evaluation against the prepared test ground truth
- adapted the workflow to TinyGPU by splitting login-node data preparation from offline GPU training
- pre-downloaded the Mistral model into the Hugging Face cache for offline compute-node execution
- completed test preparation: 581 / 1051 pages prepared successfully (55.3%), 532 with non-empty KVPs
- completed train preparation: 5389 / 9656 pages prepared successfully (55.8%), 404 with zero KVPs
- submitted the first corrected A100 training run, diagnosed the memory failure, and changed the training setup to QLoRA for the resubmitted run

### Current technical status

- Stage 0 and Stage 1 are in place as supporting pipeline components
- Stage 2 is complete
- Stage 3 heuristic baselines are complete
- Stage 3 Mistral preparation is complete
- the first corrected A100 training run failed with CUDA out-of-memory on A100 40GB
- the training setup was changed to QLoRA (4-bit NF4), paged 8-bit AdamW, and gradient checkpointing
- the corrected second A100 training run has been submitted and is waiting in queue
- Stage 4 has not yet started because Stage 3 needs to be measured cleanly first

### Results available so far

- Stage 2 clustering has been run successfully and is available as completed earlier work
- heuristic baseline outputs have been generated earlier and establish a non-neural reference point
- test preparation completed with 581 prepared pages out of 1051 unique pages (55.3\%)
- among prepared test pages, 532 contain non-empty KVP supervision
- train preparation completed with 5389 prepared pages out of 9656 unique pages (55.8\%)
- among prepared train pages, 404 contain zero KVPs after preparation

### Current findings

- the old Stage 3 Mistral path was not technically trustworthy and would not have been a valid reproduction
- the preparation pipeline is necessary because the raw KVP10k rows are not directly usable for this baseline
- TinyGPU compute nodes have no internet access, so preparation and training must run in separate environments
- a large portion of source PDFs are broken or inaccessible, which explains much of the preparation loss rate
- Mistral-7B requires memory adaptation on A100 40GB, even with LoRA
- the main challenge in this project is not only model choice, but building a technically valid end-to-end pipeline from raw documents to fair evaluation

### Risks and blockers

- final Stage 3 learned-baseline metrics are still pending until the corrected GPU run completes
- if memory remains too high even with QLoRA, sequence length or other settings may need to be reduced
- broken or inaccessible source PDFs limit the number of usable pages that can be prepared
- Stage 4 should not be started before Stage 3 is measured and documented properly

### Next steps

1. wait for the corrected A100 Mistral job to start and finish
2. collect prediction and evaluation outputs
3. update thesis chapters and supervisor report with measured Stage 3 results
4. decide whether additional Stage 3 tuning is needed after the first complete learned-baseline run
5. continue with later thesis stages and comparative analysis

## Report Template

Copy this section for the next reporting period.

### Report XX

Date: YYYY-MM-DD

#### Scope of this period

- 

#### What was completed

- 

#### Current technical status

- 

#### Results available so far

- 

#### Risks and blockers

- 

#### Next steps

1. 