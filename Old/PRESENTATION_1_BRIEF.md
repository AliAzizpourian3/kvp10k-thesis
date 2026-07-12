# First Thesis Presentation Brief

Date: 2026-03-08

## What This First Presentation Is Usually For

For a first thesis presentation, the team usually does not expect final results.

They usually want to understand:
- what problem the thesis is solving
- why the problem matters
- what exact goal you are pursuing
- what data and methods you plan to use
- what the work plan looks like
- what risks or technical difficulties already appeared

So this talk is usually more about direction, scope, and method than about final numbers.

## What You Should Definitely Cover

### 1. Thesis Goal

State the thesis in one simple sentence:

The goal of this thesis is to build and evaluate a pipeline for key-value pair extraction from document images, using KVP10k and comparing heuristic baselines with a Mistral-based learned baseline.

Short version for speaking:

I am working on extracting structured key-value information from real business documents and comparing classical baselines with a large-language-model-based approach.

### 2. Problem Motivation

Explain why this matters:
- business documents such as invoices, forms, and receipts contain important structured information
- in practice, this information is embedded in noisy layouts, not in clean tables
- extracting key-value pairs automatically is useful for document understanding and automation
- the task is hard because document layout, OCR quality, and annotation structure all matter

### 3. Dataset and Task

Mention clearly:
- dataset: IBM KVP10k
- task: extract key-value pairs from document pages
- output: structured pairs such as field name and field value, optionally with locations
- challenge: real documents are messy, annotations are not directly ready for model training

### 4. Planned Technical Path

Present the pipeline in a clean sequence:

1. prepare the raw dataset into usable page-level training examples
2. run heuristic baselines as non-neural reference points
3. run the Mistral baseline as the main learned approach
4. evaluate predictions against ground truth with text and layout-aware matching
5. compare strengths, weaknesses, and failure cases

### 5. Current Progress

This is the current honest status:
- earlier setup and CPU heuristic baselines are done
- Stage 3 Mistral pipeline was audited and rebuilt because the earlier version was not trustworthy
- test preparation is complete
- train preparation is complete
- the first A100 Mistral run was submitted successfully
- that first GPU run failed with CUDA out-of-memory, so the training configuration now needs to be adjusted

This is actually good presentation material because it shows real engineering progress and a real technical constraint, not confusion.

### 6. Risks and Challenges

Mention a few concrete ones:
- TinyGPU compute nodes do not have internet access
- PDF downloading therefore had to be separated from GPU training
- many KVP10k source PDFs are broken or inaccessible
- the raw dataset has repeated annotator copies per page
- Mistral training currently exceeds A100 40 GB memory with the present configuration

This makes the work look realistic and technically grounded.

## Should You Explain Data Preparation In Detail?

Yes, but not all of it in the main flow.

For the first presentation, data preparation should be mentioned because it is not a side detail here. It is part of the actual thesis problem. The Mistral baseline could not be trusted until the preparation path was fixed.

But you should present it at two levels.

### In the main presentation

Keep it short and conceptual:
- the raw KVP10k rows were not directly usable for the intended Mistral baseline
- a dedicated preparation step was built to download PDFs, extract text boxes, fuse OCR with annotations, and create model-ready prompt/target pairs
- this was necessary to make Stage 3 technically valid

That is enough for the main narrative.

### Only if they ask for more detail

Then mention the important technical details:
- PDF validation was needed because many URLs return broken files or HTML pages
- KVP10k contains about five annotator copies per page, so grouping by `hash_name` was required
- the richest annotation set among copies was selected for each unique page
- page numbering had to be converted from dataset indexing to PyMuPDF indexing
- outputs are saved as one prepared JSON per page with prompt text, target text, and ground truth KVPs

So: mention preparation in the presentation, but do not drown the audience in implementation detail unless they ask.

## A Good Simple Structure For Your Slides

You can structure the first presentation like this:

1. Topic and thesis goal
2. Why key-value extraction matters
3. Dataset and task definition
4. Overall pipeline and methodology
5. Baselines and planned comparison
6. Current progress
7. Challenges and risks
8. Next steps

That is a strong first-presentation structure.

## What To Say About Current Status

You can say it in a compact way like this:

The classical baselines are already in place. During the neural baseline stage, I found that the original Mistral path was not technically reliable, so I rebuilt the preparation, training, and evaluation flow to match the intended benchmark more closely. The dataset preparation is now complete, and the next step is to adjust the training setup so the Mistral run fits within the available GPU memory.

## What Not To Overemphasize In The First Talk

Avoid spending too much time on:
- exact code file names
- low-level helper functions
- every preprocessing edge case
- detailed implementation bugs unless they directly changed the thesis method

The first talk should stay at the level of research goal, pipeline logic, current progress, and main technical obstacles.

## Questions The Team May Ask

These are likely:
- What exactly is your research question?
- Why did you choose KVP10k?
- Why compare heuristics with Mistral?
- How do you evaluate correctness?
- What makes the task difficult?
- What is currently the main bottleneck?
- What will count as a successful thesis outcome?

You should be ready to answer those directly.

## Q&A Reserve Details

This section is not for the main presentation. It is a reserve for follow-up questions if the discussion becomes technical.

### If they ask why data preparation was necessary

Short answer:

The raw dataset rows were not directly usable for the intended Mistral baseline. I needed a page-level preparation step that turns raw PDF sources and annotations into model-ready prompt and target pairs.

Extra detail if needed:
- the older baseline assumed fields were already available in the right training format
- in practice, the PDFs had to be downloaded first
- text and bounding boxes had to be extracted from the actual document page
- annotations then had to be fused with extracted words to build usable key-value supervision

### If they ask what exactly the preparation step does

Short answer:

It downloads the source PDF, extracts page text and bounding boxes, aligns that text with annotations, builds key-value pairs, and writes one prepared JSON file per page for training and evaluation.

Extra detail if needed:
- PDF validity is checked because some dataset URLs do not return real PDFs
- dataset page numbers are converted to the indexing used by PyMuPDF
- KVP supervision is built from annotation linking information
- the output contains prompt text, target text, and ground-truth KVPs

### If they ask why PyMuPDF was used instead of IBM's OCR path

Short answer:

IBM's original path uses OCR tooling, but on this cluster PyMuPDF was the more practical choice because it avoids an additional system OCR dependency and works directly on PDFs with text layers.

Extra detail if needed:
- Tesseract was not available as a ready cluster dependency in the environment I was using
- PyMuPDF is simpler operationally for this setting
- the tradeoff is that scanned PDFs without a text layer cannot provide words through this method

### If they ask about repeated annotations in KVP10k

Short answer:

KVP10k contains multiple annotator copies for the same page, so I had to group rows by page identity and avoid treating them as separate PDF downloads.

Extra detail if needed:
- grouping is done by `hash_name`
- the same page may appear about five times with different annotations
- the richest annotation set was selected for each unique page

### If they ask how evaluation is done

Short answer:

Predictions are compared against prepared ground truth at the entity level, using both text-based matching and text-plus-location matching.

Extra detail if needed:
- text similarity is measured with normalized edit distance
- layout agreement is checked through bounding-box overlap
- this allows both content-only and content-plus-layout evaluation views

### If they ask why the old Mistral path was not trustworthy

Short answer:

It was not aligned with the actual dataset structure or the intended IBM-style baseline setup, so running it would not have produced a meaningful reproduction.

Extra detail if needed:
- it expected fields that were not really present in the raw data in usable form
- it used the wrong prompt and response assumptions
- it did not correctly reflect the required prepared-data workflow

### If they ask about the cluster-specific engineering issue

Short answer:

The cluster separates internet-accessible preparation from offline GPU training, so the workflow had to be split across login-node preparation and offline compute-node execution.

Extra detail if needed:
- compute nodes do not have internet access
- PDFs therefore had to be downloaded on the login node
- model weights also had to be cached before GPU jobs started
- offline environment settings were necessary during training

### If they ask why the first Mistral run failed

Short answer:

The first corrected training run exceeded the memory limit of the available A100 40 GB GPU. It has been fixed and resubmitted.

Extra detail if needed:
- Mistral-7B in bf16 uses ~14GB for weights plus ~28GB for AdamW optimizer states, exceeding 40GB
- the fix was switching to QLoRA: 4-bit quantized base model, 8-bit paged optimizer, and gradient checkpointing
- QLoRA is a standard technique that preserves model quality while fitting within smaller GPU memory
- the corrected second run has been submitted and is in the queue

### If they ask what success would look like for the thesis

Short answer:

Success means delivering a technically correct end-to-end pipeline, evaluating both heuristic and learned baselines fairly, and identifying where each approach works or fails on KVP extraction.

Extra detail if needed:
- the thesis does not depend only on beating every baseline numerically
- a valid contribution also includes robust pipeline design, fair comparison, and analysis of limitations

## A Safe Research Framing

If you want a clear academic framing, use something close to this:

The thesis investigates how well different approaches can extract key-value pairs from document pages, starting from heuristic baselines and extending to a large-language-model-based baseline, with attention to the full pipeline from raw documents to evaluated predictions.

## Immediate Next Step For Presentation Prep

Before making slides, prepare these four items first:

1. a one-sentence thesis goal
2. a one-slide pipeline diagram
3. a one-slide current status summary
4. a one-slide risks and next steps summary

If needed, this note can later be turned into a short slide outline.

## Why This Thesis Makes Sense

If someone challenges the topic with something like, "aren't there already many models for this?", the right answer is yes, models exist, but the problem is still not trivial or solved in practice.

What makes this thesis valid is that document key-value extraction is not just a matter of picking a model and pressing run. The result depends on the full pipeline:
- how the raw documents are accessed
- how text and layout information are extracted
- how annotations are aligned with document content
- how training examples are constructed
- how predictions are evaluated fairly
- how the method behaves under real infrastructure constraints

That means the thesis is not only about using Mistral. It is about building and evaluating a technically correct end-to-end system for structured information extraction from documents.

### Strong Short Answer

Existing models do not make this thesis unnecessary, because the real challenge is the full document understanding pipeline, not only the choice of model. My thesis studies how to build and evaluate that pipeline on a realistic benchmark, and compares simpler heuristic methods with a stronger language-model-based baseline.

### Stronger Academic Answer

The contribution of this thesis is not necessarily a brand-new model architecture. The contribution is a technically sound and reproducible study of key-value extraction on KVP10k, including data preparation, baseline comparison, practical model adaptation, evaluation design, and analysis of limitations and failure cases.

### Why This Is Good Enough For A Master's Thesis

This is a good master's thesis because it is:
- well motivated by a real document-understanding problem
- technically nontrivial
- grounded in a real benchmark
- broad enough to include engineering and evaluation work
- focused enough to be completed within a thesis timeline
- capable of producing both quantitative results and qualitative analysis

It does not need to be rocket science. A master's thesis is often strong when it shows clear problem framing, technically correct implementation, fair comparison, and honest analysis.

### What You Can Legitimately Be Proud Of

You can say that the work already includes more than simply running an existing model:
- identifying that an earlier baseline path was not technically trustworthy
- rebuilding the preparation, training, and evaluation path so the experiment is valid
- handling messy real benchmark issues such as broken PDFs and repeated annotator copies
- comparing heuristic and learned approaches instead of assuming a larger model is automatically better
- working under realistic cluster and memory constraints

### Simple Spoken Version

I did not choose this topic just to run an existing model. The interesting part is building a valid pipeline for extracting key-value information from real documents and evaluating it properly. There are models for related tasks, but in practice the problem still depends heavily on preprocessing, layout handling, supervision construction, and fair evaluation. That is why this is still a meaningful thesis topic.