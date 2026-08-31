# Final Thesis Revision Plan After Supervisor Feedback

Date: 2026-08-29

## 1. Purpose and scope

This plan governs the revision of the thesis after the supervisor review of
`LaTeX_Thesis_260826_V2_commented.pdf` and the related email.

The review contains 81 PDF annotations:

| Part | Number of comments |
|---|---:|
| Abstract | 1 |
| Chapter 1 | 6 |
| Chapter 2 | 8 |
| Chapter 3 | 2 |
| Chapter 4 | 2 |
| Chapter 5 | 1 |
| Chapter 6 | 58 |
| Chapter 7 | 1 |
| Chapter 8 | 2 |
| **Total** | **81** |

Chapter 6 received a detailed review. The other chapters received a selective
review. The marked passages are examples of recurring issues. The email
therefore requires a complete citation, definition, and context check across
the thesis.

This is a writing and presentation revision. It is not a new implementation
investigation.

The work must obey these scope limits:

- Do not reopen the coordinate-normalisation investigation unless a supervisor
  comment directly requires it.
- Do not perform a broad code audit.
- Do not start a new training run or evaluation run without separate approval.
  The user separately approved one evaluation-only run of the frozen final V4
  checkpoint 15. This approval does not permit retraining, checkpoint
  reselection, threshold tuning, or repeated test access.
- Use existing artifacts only when a supervisor comment requires an evidence
  check.
- Do not change a numerical value without verified source evidence.
- Do not change code, data, scripts, results, figures, tables, or bibliography
  entries unless an approved thesis correction requires that specific change.
- Protect all unrelated tracked and untracked repository files.

### 1.1 Current revision status

Last updated: 2026-08-30

| Part | Status | Next required action |
|---|---|---|
| Chapter 1 | Approved edits and final result/roadmap synchronisation applied | Run the final citation and layout check |
| Chapter 2 | Approved supervisor-comment and chapter-wide edits applied | Run the final integration and layout check |
| Chapter 3 | Approved supervisor-comment and chapter-wide edits applied | Run the final integration and layout check |
| Chapter 4 | Approved edits and final checkpoint provenance synchronised | Inspect Figure 4.1 and run the final citation/layout check |
| Chapter 5 | Approved package applied using the final verified implementations | Inspect the moved prompt figure during the final compilation |
| Chapter 6 | All 58 comments, final results, and terminology cleanups applied | Inspect revised tables during the final compilation |
| Chapter 7 | Approved substantive discussion revision applied | Run the final citation, reference, and layout check |
| Chapter 8 | Approved substantive conclusion revision applied | Run the final citation, reference, and layout check |
| Abstracts and front matter | Final review completed; abstract comment closed | Compile and inspect the PDF |

The highlighted abstract comment now has a final status:

| Supervisor issue | Status |
|---|---|
| Add sufficient task, dataset, and system context before reporting the abstract results | `DONE` |

Final supervisor-comment accounting:

| Status | Count |
|---|---:|
| `DONE` | 69 |
| `ADDRESSED ELSEWHERE` | 11 |
| `NO CHANGE` | 1 |
| **Total** | **81** |

Chapter 1 closed all six highlighted PDF comments:

| PDF page | Supervisor issue | Status |
|---|---|---|
| 17 | Explain the extraction task | `DONE` |
| 17 | Expand KVP and explain KVP10k | `DONE` |
| 17 | Explain the Mistral baseline | `DONE` |
| 17 | Explain what the relation connects | `DONE` |
| 17 | Explain segment and entity linking | `DONE` |
| 18 | Clarify ownership of the classifier and linker | `DONE` |

The approved Chapter 1 revision also:

- defined Regular, Unkeyed, and Unvalued items;
- defined annotation geometry;
- revised RQ2 for the direct same-subset comparison;
- revised RQ3 to refer to development diagnostics;
- kept RQ1 and RQ4 unchanged;
- used the exact entity labels `KEY`, `VALUE`, and `O`;
- defined `O` as the letter O and as the label for tokens outside both entity
  classes;
- defined V4 as the final LayoutLMv3-based span-level discriminative model;
- distinguished the published KVP10k Mistral-7B baseline from the reconstructed
  Mistral-7B reference system;
- used “bounding-box filtering” instead of the less clear “entity-span
  retention” in the contribution summary;
- distinguished prior mechanisms from thesis-specific implementation and
  integration;
- preserved all existing experimental results and numerical values;
- passed `git diff --check`;
- did not compile the thesis.

Chapter 1 integration status:

- The roadmap now follows the final descriptive Chapter 6 structure.
- The Chapter 1 headline values use the final clean Mistral and checkpoint-15
  V4 artifacts.
- Chapter 8 now answers the final RQ2 and RQ3 formulations directly.
- Keep `KEY`, `VALUE`, and `O`, the published-versus-reconstructed Mistral
  distinction, and the full first-use V4 definition in the remaining edits.
- Chapter 8 now refers to Chapter 1 for detailed contribution provenance and
  retains only a concise conclusion summary.
- The final matched English/German abstract review is complete.


Chapter 2 closed all eight highlighted PDF comments:

| PDF page | Supervisor issue | Status |
|---|---|---|
| 21 | Remove the prose em dash from the opening | `DONE` |
| 21 | Cite representative earlier document-extraction systems | `DONE` |
| 21 | Support the layout-change limitation with literature | `DONE` |
| 21 | Cite and define LSTM at first use | `DONE` |
| 22 | Support token classification and BIO with literature | `DONE` |
| 22 | Cite FUNSD, CORD, and DocVQA | `DONE` |
| 23 | Define and explain LMDX | `DONE` |
| 26 | Explain the published Mistral-7B baseline | `DONE` |

The approved Chapter 2 revision also:

- distinguishes the published KVP10k Mistral-7B generative baseline from the
  reconstructed Mistral-7B reference system;
- explains the published input ordering, coordinate representation, and
  structured output;
- states that the local system is not an exact reproduction;
- uses `KEY`, `VALUE`, and the letter `O`;
- uses text-only, location-only, and text+location terminology;
- defines V1, V2, and V4 by their roles in the thesis;
- introduces the span-level discriminative design descriptively before version
  labels and defines V4 at its first reader-facing use in Chapter 2;
- identifies V1 and V2 as development variants and V4 as the final span-level
  discriminative model;
- states the scaled dot-product divisor as `sqrt(d_k)`;
- removes unsupported or unnecessary claims;
- moves annotation-row and prepared-subset details out of the literature
  review;
- adds the approved BERT, CNN, LSTM, and BIO glossary entries;
- adds six approved primary-source bibliography entries and corrects the IoU
  entry metadata;
- renames the final section to `Summary and Thesis Objectives` so that its
  title matches its revised content;
- removes the redundant unsupported claim that fixed rules remain useful as
  simple baselines;
- removes all prose em dashes from Chapter 2;
- removes the remaining prose and heading em dashes in Chapters 5 and 6 as a
  targeted global style correction;
- replaces visible Chapter 6 table dash placeholders with `Not scored` where
  no cluster metric was calculated and with `N/A`, explicitly defined as not
  applicable, for generative-system entity metrics;
- leaves source divider comments unchanged because they do not appear in the
  compiled thesis;
- preserves all experimental results and numerical values;
- passes `git diff --check`;
- does not compile the thesis.

Chapter 3 addressed both highlighted PDF comments:

| Supervisor issue | Status |
|---|---|
| Add example images and explain the provided versus used train/validation/test splits | `DONE` |
| Explain/remove the unexplained “Stage 2” terminology | `DONE` |

The approved Chapter 3 revision also:

- distinguishes official annotation rows, official unique-page splits,
  prepared unique pages, and the internal train/validation split;
- explains the preparation exclusions and states that no per-cause exclusion
  counts are available;
- adds a compact dataset-flow table with explicit units;
- defines a prepared page and the Regular, Unkeyed, and Unvalued annotation
  categories;
- distinguishes the separate 4,985-page Mistral training subset from the
  4,851/538 discriminative-model train/validation split;
- states that the 581 prepared official-test pages are excluded from the
  internal train/validation split and are used for test evaluation;
- reuses the existing cluster pages as illustrative examples and states that
  they do not show the linking references;
- motivates the annotation-geometry clustering, defines its inputs and use,
  and states that its clusters are exploratory geometry strata rather than
  verified document genres;
- defines the geometry features and the train-fitted test assignment;
- preserves and explains the weak cluster separation;
- defines PCA and t-SNE and limits the t-SNE plot to descriptive use;
- distinguishes unique pages, annotation rows, coordinate-bearing rows, and
  resolved linking references in the spatial-link audit;
- distinguishes the 755 annotations with non-empty linking values from the
  755 resolved coordinate-bearing pair-distance instances recorded by the
  saved audit output;
- explains that the first-200-row audit is not a random or dataset-wide
  prevalence estimate;
- adds primary references for the public KVP10k repository and the silhouette,
  Davies--Bouldin, and Calinski--Harabasz criteria;
- preserves all experimental results and numerical values;
- passes `git diff --check`;
- does not compile the thesis or run new experiments.

Chapter 3 integration status:

- Reader-facing `Stage 2` wording has been removed from later chapters and
  Appendix A.
- Split-count attribution, category names, and annotation-geometry terminology
  have received the final consistency check.
- Check citations, cross-references, and layout during the final compilation
  and PDF review.


Chapter 4 closed both highlighted PDF comments:

| PDF page | Supervisor issue | Status |
|---|---|---|
| 39 | Replace the vague relation term “scored” with the exact logit, probability, threshold, prediction, loss, and metric sequence | `DONE` |
| 40 | Preserve the architecture overview and identify thesis contributions and the V1/V3 paths | `DONE` |

The approved Chapter 4 revision also:

- distinguishes the pretrained LayoutLMv3-base encoder and the general
  scaled-dot-product principle from the thesis-specific classifier, linker,
  span grouping, training pipeline, evaluator integration, and diagnostics;
- marks component ownership and the V1, V2/V4, and V3 paths in the architecture
  figure and caption;
- uses the exact entity labels `KEY`, `VALUE`, and `O`;
- defines V1 as the token-level linker variant, V2 as the first span-level
  variant, V3 as a linker-only diagnostic, and V4 as the final
  LayoutLMv3-based span-level discriminative model;
- states that V3 reuses the V2 span-level architecture, freezes the encoder and
  classifier, uses predicted spans with ground-truth relation targets, and has
  no headline benchmark result;
- defines the relation-logit function, sigmoid relation probability,
  probability threshold, relation prediction, and later evaluation metrics;
- defines token-pair and span-pair candidate units and all displayed equation
  symbols;
- keeps unweighted entity loss, relation positive-class weighting, and the
  general joint-loss definitions in Methodology;
- removes the premature Mistral--V4 performance comparison from Methodology;
- replaces reader-facing Stage terminology with descriptive method terms where
  appropriate;
- uses primary citations and the existing glossary entries for named prior
  methods, datasets, and acronyms;
- corrects V4 early-stopping patience from 10 to 5 and states that early
  stopping is permitted from epoch 10, as verified in the retained V4
  configuration;
- limits the test-selection statement to V4 and states that its checkpoint
  selection and early stopping use only validation Regular text+location macro
  pair F1;
- preserves all experimental results and benchmark values;
- passes `git diff --check`;
- does not compile the thesis or run experiments.

Chapter 4 integration status:

- Chapter 5 and the factsheet record the numerical safeguards and the verified
  early-stopping setting: patience 5, with stopping permitted from epoch 10.
- Chapter 4 now identifies epoch 10 as the entity-pretraining checkpoint and
  checkpoint 15 as the selected final V4 checkpoint.
- Chapter 4 now records the selected encoder and classifier plus a fresh
  relation linker and excludes the earlier recovery method from the
  checkpoint-15 result.
- The ownership, entity-label, variant-role, and relation terminology is
  aligned across the revised result-bearing text.
- Inspect the updated Figure 4.1 node and path-label layout during the final
  approved compilation and PDF review.


## 1.6 Chapter 5 revision completed

Chapter 5 closed its highlighted PDF comment:

| Supervisor issue | Status |
|---|---|
| Keep useful reproducibility information, but remove the exact checkpoint path, hash, and similar low-level run details from the main manuscript | `DONE` |

The approved Chapter 5 revision:

- renames the chapter to `Implementation and Experimental Setup`;
- distinguishes the published KVP10k baseline from the reconstructed
  Mistral-7B reference system;
- documents the final clean Stage 3 preparation, prompt, target serializer,
  singleton Unvalued handling, parser, response-only supervision, QLoRA
  configuration, fixed eight-epoch training, and separate 581-page inference;
- states that Stage 3 uses 4,985 training pages after 404 empty-target
  exclusions and performs no metric-based checkpoint selection;
- moves the Mistral prompt figure and implementation details from Chapter 6;
- adds the Wolf et al. (2020) citation for the Hugging Face Transformers
  library while preserving the existing BERT citation;
- removes the unsupported approximate 35-to-200 word-count claim;
- documents the final LayoutLMv3 input mapping from prepared 0-to-100
  coordinates to the 0-to-1000 model grid and keeps exported boxes on the
  prepared 0-to-100 grid;
- documents the fixed 4,851/538 training/validation manifests and the absence
  of test access during development and checkpoint selection;
- records entity pretraining followed by joint V4 training, with the selected
  entity encoder and classifier transferred to a fresh relation linker;
- records patience 5, stopping permitted from epoch 10, and final joint
  checkpoint 15 selected only by validation Regular text+location macro F1;
- distinguishes relation logits, sigmoid probabilities, the probability
  threshold, relation predictions, loss values, and benchmark metrics;
- records the verified numerical safeguards, candidate limits, optimiser-step
  accounting, and partial final accumulation rule;
- removes the old Canary B lineage, checkpoint hash, absolute path, allocation
  history, and outdated code tree from the main manuscript;
- preserves the required post-selection test-evaluation sentence;
- does not change any Chapter 6 result value.

Integration status after Chapter 5:

- The single checkpoint-15 test artifact is complete and verified.
- Chapter 6 and its central V4 value source use the final clean Mistral and
  bbox-scale-fixed checkpoint-15 result lineage.
- The controlled global pass synchronised Chapter 1, Chapter 4, Chapters 7--8,
  both abstracts, and the factsheet with this lineage.
- The substantive revisions of Chapters 7 and 8 are complete.
- Compile and inspect the moved prompt figure and the revised Chapter 6 tables
  only during the later approved full-document layout check.

## 1.7 Chapter 6 revision completed

All 58 Chapter 6 PDF comments now have a final status:

| ID | Supervisor issue | Status |
|---|---|---|
| C01 | Refer back to the method explanations | `DONE` |
| C02 | Explain the two relation-candidate granularities | `DONE` |
| C03 | Remove or explain the low-level training terms in the overview | `DONE` |
| C04 | Define KVP category equality in evaluation | `DONE` |
| C05 | Clarify Regular relation counts and units | `DONE` |
| C06 | Move detailed Mistral training setup out of Results | `ADDRESSED ELSEWHERE` |
| C07 | Define the structured Mistral output | `ADDRESSED ELSEWHERE` |
| C08 | Introduce and cite LMDX earlier | `ADDRESSED ELSEWHERE` |
| C09 | Remove the visible prose em dash | `DONE` |
| C10 | Explain why the prepared and published subsets differ | `DONE` |
| C11 | Explain loss on prompt tokens | `ADDRESSED ELSEWHERE` |
| C12 | Explain response-only supervision and its counterpart | `ADDRESSED ELSEWHERE` |
| C13 | Avoid treating response-only supervision as an assumed default | `ADDRESSED ELSEWHERE` |
| C14 | Explain prompt-plus-response versus response-only loss earlier | `ADDRESSED ELSEWHERE` |
| C15 | Explain the two-string prompt instruction | `ADDRESSED ELSEWHERE` |
| C16 | Define target serialisation and prompt-field replacement | `ADDRESSED ELSEWHERE` |
| C17 | Explain use of prepared rather than full official splits | `DONE` |
| C18 | Remove unexplained Stage 2 wording | `DONE` |
| C19 | Rewrite the clustering-assignment sentence | `DONE` |
| C20 | Explain Regular in contrast with Unkeyed and Unvalued | `DONE` |
| C21 | Motivate the cluster analysis before its result | `DONE` |
| C22 | Remove the unsupported old-lineage unmatched-entity rates | `DONE` |
| C23 | Correct the cluster-table alignment and long row text | `DONE` |
| C24 | Add a simple Mistral cluster interpretation | `DONE` |
| C25 | Remove the unexplained Stage hierarchy | `DONE` |
| C26 | Clarify the entity-pretraining checkpoint | `DONE` |
| C27 | Explain or remove the legacy entity metric | `DONE` |
| C28 | Explain entity pretraining in basic terms | `DONE` |
| C29 | Define the relation-candidate count symbols | `DONE` |
| C30 | Move class-weighting detail to Methodology | `ADDRESSED ELSEWHERE` |
| C31 | Remove repeated random-seed statements | `DONE` |
| C32 | Give provenance for the entity-pretraining result | `DONE` |
| C33 | Remove repeated diagnostic disclaimers | `DONE` |
| C34 | Do not present V1 as a benchmark experiment without a valid metric | `DONE` |
| C35 | Simplify the competing Stage, Experiment, Result, and Version schemes | `DONE` |
| C36 | Distinguish relation logits, probabilities, predictions, and metrics | `DONE` |
| C37 | Explain weight initialisation without vague warm-start wording | `DONE` |
| C38 | Replace the generic diagnostic claim with measured evidence | `DONE` |
| C39 | Define a relation logit | `DONE` |
| C40 | Empty supervisor comment on the repeated seed highlight | `NO CHANGE` |
| C41 | Interpret the negative V2 logits | `DONE` |
| C42 | Explain the logit-to-probability transition and takeaway | `DONE` |
| C43 | Define the bounding-box retention calculation with an equation | `DONE` |
| C44 | Define relation-label construction and its units | `DONE` |
| C45 | Remove the formulaic 'A, not B' diagnostic sentence | `DONE` |
| C46 | Remove repeated evaluator-threshold text | `DONE` |
| C47 | Make final V4 the central evaluation and restate its protocol | `DONE` |
| C48 | Add and interpret a direct same-page Mistral--V4 table | `DONE` |
| C49 | Locate the wrong-link evidence in the qualitative table | `DONE` |
| C50 | Preserve the immediate entity-versus-pair takeaway | `DONE` |
| C51 | State the purpose and meaning of special-category coverage | `DONE` |
| C52 | Explain the cluster result in plain language | `DONE` |
| C53 | Interpret the qualitative cases | `DONE` |
| C54 | Remove the checkpoint hash and job-level record | `DONE` |
| C55 | Remove repeated published-result context | `DONE` |
| C56 | Explain that test metrics did not select the checkpoint | `DONE` |
| C57 | State the no-ablation limitation only once | `DONE` |
| C58 | Keep the summary and add clear takeaways | `DONE` |

`ADDRESSED ELSEWHERE` means that Chapters 2, 4, or 5 now contain the requested
definition or implementation detail and Chapter 6 supplies only the needed
cross-reference. C40 remains `NO CHANGE` because its comment field is empty;
the repeated seed text was removed under C31.

The completed Chapter 6 revision:

- uses the six-part descriptive structure approved in Section 9;
- makes the same-581-page Mistral--V4 comparison the main result;
- updates clean Mistral Regular F1 to 0.757/0.694/0.657 and checkpoint-15 V4
  Regular F1 to 0.365/0.491/0.329;
- records validation checkpoint selection at 0.331 and keeps entity-pretraining
  epoch 10 separate from final joint checkpoint 15;
- replaces the unsupported checkpoint-15 test-entity table with verified
  checkpoint-15 validation entity metrics;
- removes old-lineage recovery values and reports direct special-category
  coverage only;
- retains comparative cluster values derived from saved final per-page metrics
  and the frozen training-fitted cluster map, without model inference;
- checks the qualitative examples against saved checkpoint-15 predictions and
  ground truth, removes the invalid Syscode error, and corrects the Zip, Page,
  and ext. descriptions;
- corrects the historical diagnostic claim from missing teacher forcing to the
  verified training/inference span-source mismatch;
- defines each diagnostic unit and gives an immediate interpretation after
  important results;
- removes hashes, job history, repeated evaluator detail, and most Chapter 5
  implementation repetition;
- changes no code, data, model output, or unrelated thesis chapter.

### Completed controlled global result-consistency pass

The controlled pass:

- changes the Chapter 1 same-page headline values to V4/Mistral 0.329/0.657
  and aligns its roadmap with the final Chapter 6 structure;
- identifies epoch 10 only as the entity-pretraining checkpoint and checkpoint
  15 as the selected final V4 checkpoint in Chapter 4;
- replaces the old initialisation account with the verified selected encoder
  and classifier plus a fresh relation linker;
- synchronises Chapters 7--8 with the final Mistral, V4, direct-All, entity,
  and annotation-geometry values;
- removes unsupported checkpoint-15 test-entity, confusion-count, and recovery
  claims and labels retained entity metrics as validation evidence;
- synchronises both abstracts and the factsheet with the final artifacts;
- uses the full published KVP10k Mistral-7B baseline name in Chapter 6 and keeps
  one canonical label for its annotation-geometry table;
- leaves Chapter 3 unchanged because its dataset statistics are from a
  different evidence unit.

Remaining follow-up: inspect table placement, line wrapping, citations,
references, and page layout during the later full compilation.

## 1.8 Chapter 7 revision completed

Chapter 7's only highlighted PDF comment now has a final status:

| PDF page | Supervisor issue | Status |
|---|---|---|
| 73 (printed page 57) | Put immediate interpretations close to the Chapter 6 results where practical | `ADDRESSED ELSEWHERE` |

`ADDRESSED ELSEWHERE` is appropriate because Chapter 6 now gives a short
interpretation after each main result, while Chapter 7 preserves the deeper
synthesis, limitations, literature connections, and implications.

The approved Chapter 7 revision:

- renames the opening section to `Cross-Result Interpretation` and states the
  Chapter 6--7 division explicitly;
- preserves the entity-versus-pair interpretation and identifies entity F1
  as validation evidence with a different evaluation unit from pair F1;
- keeps the central same-page Mistral--V4 and direct-output coverage findings
  without repeating the complete Chapter 6 result tables;
- retains V1 and V2 only as development evidence and avoids a causal claim
  about candidate granularity;
- removes the duplicate annotation-geometry table and the repeated V1/V2
  raw diagnostic subsection;
- replaces `Root-Cause Analysis` with `Interpretation of Development Diagnostics`
  and states that the individual and interaction effects were not isolated;
- deepens the qualitative discussion into candidate-set/span and
  relation-selection error routes;
- links the observed evidence to prior relation-modeling methods and keeps
  causal and cross-benchmark claims cautious;
- distinguishes the official KVP10k split counts from the local prepared
  counts and attributes the released and reconstructed Mistral setups precisely;
- converts future-work proposals into controlled, testable comparisons;
- changes no result value, model artifact, code, or other thesis chapter.

## 1.9 Chapter 8 revision completed

Both highlighted Chapter 8 comments now have a final status:

| PDF page | Supervisor issue | Status |
|---|---|---|
| 79 (printed page 63) | Present reiterated contributions as connected prose and keep the detailed explanation in the Introduction | `DONE` |
| 82 (printed page 66) | Add future work and remove the non-RQ subheadings | `DONE` |

The approved Chapter 8 revision:

- replaces the enumerated contribution list with a concise, connected opening
  and refers to Chapter 1 for the detailed ownership distinction;
- retains `Research Questions Revisited` as the only section-level heading and
  removes the separate summary, practical-implications, and closing headings;
- answers RQ1--RQ4 directly with the final result lineage and canonical V4 and
  reconstructed Mistral-7B reference-system terminology;
- states that the V2/V4 paths are designed to reduce the candidate space rather
  than claiming a controlled measured reduction;
- keeps entity F1 explicitly labelled as validation evidence and distinguishes
  its unit from complete-pair F1;
- retains the lack of direct Unkeyed and Unvalued coverage without repeating
  the secondary All F1 value;
- removes low-level checkpoint history, illustrative candidate counts, the
  published-baseline comparison, and other details already established in
  earlier chapters;
- adds concise future work that keeps the final coordinate mapping and data
  splits fixed while testing model-design choices;
- ends with the central scientific finding about span construction and relation
  selection;
- changes no result value, experimental artifact, code, or other thesis chapter.

## 2. Material to preserve

The supervisor gave positive feedback that must guide the revision.

- The thesis already reads well and has a good overall structure.
- The Chapter 4 architecture overview is useful.
- The annotation-geometry analysis is a good idea.
- Detailed reproducibility information is useful, although some details belong
  in an appendix or repository record.
- The V4 entity-versus-linking interpretation is a good model for other result
  interpretations.
- The Chapter 6 summary is useful.
- Chapter 7 contains valuable interpretation and synthesis.
- The research-question review in Chapter 8 is useful.
- The main conclusion is generally good.

The revision must improve access to this material. It must not remove useful
analysis only because a shorter takeaway is added elsewhere.

## 3. Required feedback records

Create one issue ledger for all 81 annotations. Each entry must contain:

- PDF page and annotation number;
- highlighted text;
- full supervisor comment;
- source file and location;
- issue type;
- local or recurring status;
- proposed action;
- required citation or evidence;
- cross-chapter effect;
- final status.

Use one of these final statuses for every annotation:

- `DONE`
- `ADDRESSED ELSEWHERE`
- `NO CHANGE + reason`

Create a second ledger for positive comments. Mark the related material as
protected.

Do not infer a correction from an empty or unclear annotation. Use the local
context and related comments. If the intended change remains unclear, record
`NO CHANGE + reason` or request a decision.

## 4. Evidence policy

Evidence checks must be limited to claims and values affected by supervisor
comments. Use existing thesis sources and saved artifacts.

For each affected result, record:

- system or variant;
- split;
- number and unit;
- preprocessing coverage;
- metric;
- averaging method;
- checkpoint-selection method;
- source artifact;
- thesis locations that report the result.

Apply these rules:

- Do not mix row counts, page counts, unique hashes, and coordinate-bearing
  units.
- Do not mix pooled diagnostic measures with page-macro benchmark metrics.
- Do not imply that published and reconstructed results use the same pages.
- Compare reconstructed Mistral and V4 directly only when they use the same 581
  prepared test pages and the same evaluator.
- Do not invent a missing metric.
- Do not retain a known-invalid metric as a performance result.
- If no valid artifact exists, describe the item as development history or a
  diagnostic.
- If a rationale is not documented, describe the setting as heuristic. Do not
  construct a rationale after the fact.
- If an evidence check would require a new run, stop and request approval.

## 5. Reader-facing terminology

Use descriptive titles as the main navigation system. Keep experiment or
version identifiers only as secondary labels when they help trace the work.

| Current term | Reader-facing treatment |
|---|---|
| Stage 2 | Use “annotation-geometry clustering analysis.” |
| Stage 3 | Use “reconstructed Mistral-7B reference system.” |
| Stage 4a | Use “entity-classification initialization” or “entity-classification checkpoint.” |
| Stage 4b | Use the applicable model or linker description. |
| V1 | Keep as the token-level linker variant. |
| V2 | Keep as the first span-level linker variant. |
| V3 | Define once from verified records. Do not give it a headline result without valid evidence. |
| V4 | Define as “the final LayoutLMv3-based span-level discriminative model.” Use `V4` after the definition. |
| Experiment 1–8 | Do not use as the primary section hierarchy. Retain only where useful for traceability. |
| Result 1–8 | Replace with descriptive result headings. |
| D1–D3 | Use descriptive diagnostic headings. Keep identifiers only as optional secondary labels. |
| Linker score | Use the exact term: logit, probability, prediction, loss, precision, recall, or F1. |
| Regular link | Use “Regular key–value pair” or “candidate relation,” as applicable. |

Internal code names, script names, checkpoint names, and directory names remain
unchanged.

Define the technical sequence precisely:

1. The linker computes a relation logit.
2. A sigmoid converts the logit to a probability.
3. The decoder applies a threshold.
4. The evaluator computes precision, recall, and F1.

Define these benchmark categories at first use:

- Regular key–value pair;
- unkeyed value;
- unvalued key;
- All aggregate.

Use “Regular text+location” in metric labels. Do not use the vague label
“combined” when “text+location” is the defined benchmark term.

### 5.1 Locked consistency decisions from Chapter 1

#### Entity labels

- The exact model labels are `KEY`, `VALUE`, and `O`.
- `O` is the uppercase letter O. It is not the number zero.
- `O` denotes a token that belongs to neither the `KEY` class nor the `VALUE`
  class.
- Do not use `OTHER` as the literal class label.
- Ordinary prose can use the word “other” only when it does not name the model
  label.
- In LaTeX, use `\texttt{KEY}`, `\texttt{VALUE}`, and `\texttt{O}` when naming
  the exact labels.

#### Mistral system names

- Use “published KVP10k Mistral-7B generative baseline” for the external system
  reported in the KVP10k paper.
- Use “reconstructed Mistral-7B reference system” for the implementation in
  this thesis.
- State that the reconstructed reference system is based on the published
  approach.
- Do not describe the reconstructed system as an exact reproduction.
- Use “published result” and “internal same-subset result” to distinguish the
  two evaluation settings.

#### V4 definition

- At first important use, write “the final LayoutLMv3-based span-level
  discriminative model (V4)” or an equivalent full definition.
- Use `V4` alone after the full definition.
- Do not use “final corrected span-level configuration” as the first
  definition. The development corrections belong in the later method and
  results discussion.

#### Ownership wording

Use this distinction consistently:

- The pretrained LayoutLMv3 encoder and the general scaled-dot-product scoring
  principle are prior work.
- The task-specific classifier and linker modules, span-grouping path, training
  pipeline, integration with the released evaluator, and diagnostic procedures
  were implemented for this thesis.
- KVP10k, its released evaluator, Mistral-7B, and the published KVP10k baseline
  design are prior resources or methods.
- “Integration with the released evaluator” is thesis work. The released
  evaluator itself is not claimed as thesis work.

#### Diagnostic terminology

- In high-level summaries, use “candidate construction, bounding-box filtering,
  and relation supervision.”
- Do not use “entity-span retention” in the Introduction.
- Detailed diagnostic sections can use more specific terms after they define
  the measured unit and procedure.

#### Verified numerical facts and final result lineage

- The KVP10k paper reports 10,707 pages. Grouping the released annotation
  rows by `hash_name` gives 9,656 unique training pages and 1,051 unique test
  pages.
- The annotation-geometry analysis uses 9,124 unique coordinate-bearing
  training pages.
- The controlled internal comparison uses the same prepared 581-page test
  subset for V4 and the reconstructed Mistral-7B reference system. Their exact
  document-ID sets are equal.
- The final clean reconstructed-Mistral artifact reports Regular F1 values of
  0.757268 for text, 0.694388 for location, and 0.657228 for text+location.
- The bbox-scale-fixed checkpoint-15 V4 artifact reports Regular F1 values of
  0.365022 for text, 0.490903 for location, and 0.329114 for text+location.
  Its Regular text+location precision and recall are 0.347247 and 0.312781.
- Checkpoint 15 was selected by validation Regular text+location F1 0.331464
  before one evaluation on the 581-page test subset. Entity pretraining uses
  its separate epoch-10 checkpoint.
- The final V4 direct decoder emits Regular pairs only. Its All
  text+location F1 is 0.214139. No checkpoint-15 recovery result exists.
- The earlier Mistral value 0.661788 and the earlier V4 value 0.345 belong to
  older result lineages. The old V4 recovery, test-entity, and qualitative
  values must not be mixed with checkpoint-15 results.
- Chapter 6, the other result-bearing chapters, both abstracts,
  `chapters/v4_result_values.tex`, and the factsheet now use the final
  compatible artifacts.

## 6. Global thesis checks

### 6.1 Citation check

- Support externally based claims with literature.
- Cite named methods, models, and datasets at their first substantive use.
- Prefer the primary paper.
- Cite a source at the sentence that it supports.
- For an unsupported general claim, add a suitable source, narrow the claim, or
  remove it.
- Support original findings with an internal result reference, not an external
  citation.
- Search the existing bibliography before adding a new entry.

### 6.2 Definition and context check

At the first important use of a concept, explain:

1. what it is;
2. its input and output;
3. why it is relevant;
4. how the thesis uses it.

At later uses, give a short reminder and a cross-reference. Do not repeat the
full explanation.

### 6.3 Acronym check

Use the existing `glossaries-extra` setup. Do not add another acronym package.
Check the first use of KVP, OCR, LSTM, LMDX, NED, IoU, QLoRA, PCA, and related
terms.

### 6.4 Contribution check

Distinguish:

- the existing KVP10k dataset and evaluator;
- the published Mistral approach;
- pretrained LayoutLMv3;
- adapted implementation components;
- components implemented for this thesis;
- thesis-specific modifications;
- evaluations and analyses performed in this thesis.

Use the same ownership wording in Chapters 1, 4, and 8.

### 6.5 Targeted style check

Search for:

- prose em dashes;
- formulaic “X is A, not B” constructions;
- vague uses of “score,” “problem,” “better,” and “this”;
- repeated seeds, thresholds, caveats, and metric definitions;
- unsupported strong claims;
- missing introductions to tables and figures;
- missing immediate interpretations;
- long hashes, paths, and job identifiers in the main text.

Do not perform a broad British-English conversion. Make only relevant and
approved style changes.

## 7. Chapter revision order

Prepare and approve exact chapter packages in this order:

1. Chapter 1 -- approved edits and final result/roadmap integration applied
2. Chapter 2 -- approved comment and chapter-wide edits applied
3. Chapter 3 -- approved comment and chapter-wide edits applied
4. Chapter 4 -- approved edits and final checkpoint provenance synchronised
5. Chapter 5 -- approved package applied; final test artifact confirmed
6. Definition and method completeness gate -- completed for the final result lineage
7. Chapter 6 -- approved supervisor-comment and final result revision applied
8. Controlled global result-consistency pass -- completed
9. Chapter 7 -- approved substantive revision applied
10. Chapter 8 -- approved substantive revision applied
11. Final abstract and front-matter review -- completed
12. Final compilation and PDF inspection

The abstracts come last because they depend on the final terminology and
results presentation.

## 8. Chapter-by-chapter objectives

### Chapter 1: Introduction

- [x] Explain the task input and output in plain language.
- [x] Define KVP and KVP10k at first use.
- [x] Explain Regular, Unkeyed, and Unvalued outputs at the correct level.
- [x] Explain the reconstructed generative Mistral reference.
- [x] Explain relation linking and the token-to-span distinction.
- [x] Cite or narrow broad motivation claims.
- [x] Separate prior work from thesis work.
- [x] State what was reconstructed, adapted, implemented, evaluated, and
  analysed.
- [x] Update the roadmap for the final Chapter 6 structure.
- [x] Synchronise the final Mistral and V4 headline values.
- [ ] Run the final citation and layout check.

### Chapter 2: Literature Review

- [x] Remove the repetitive opening.
- [x] Add primary references for common practices and named methods or datasets.
- [x] Expand acronyms at first use.
- [x] Explain and cite LMDX before Chapter 6.
- [x] Explain the published Mistral baseline and its structured output.
- [x] Distinguish the published approach from the reconstruction.
- [x] Remove reader-facing internal stage terminology.
- [x] Use `KEY`, `VALUE`, and `O` consistently.
- [x] Use text-only, location-only, and text+location terminology.
- [x] Describe scaled dot-product scoring with the `sqrt(d_k)` divisor.
- [x] Remove prose em dashes from Chapter 2.
- [x] Move local annotation-row and prepared-subset details to the dataset and
  evaluation chapters.
- [ ] Run the final cross-chapter terminology, citation, and numerical check.

### Chapter 3: Dataset and Analysis

- [x] Explain official splits, prepared subsets, and exclusions with correct
  units.
- [x] Add one compact dataset-flow table if it materially improves clarity.
- [x] Use a clear annotation example if an existing figure can provide it.
- [x] Define Regular, Unkeyed, and Unvalued annotations.
- [x] Remove Stage 2 terminology from Chapter 3.
- [x] Motivate annotation-geometry clustering.
- [x] State that the clusters describe geometry and are not verified document
  genres.
- [x] Preserve the weak-separation limitation.
- [x] Add primary references for named clustering criteria.
- [x] Explain the units and limits of the spatial-link audit.
- [ ] Run the final cross-chapter terminology, citation, numerical, and layout
  check.

### Chapter 4: Methodology

- [x] Preserve the architecture overview.
- [x] Mark pretrained, adapted, and thesis-specific components.
- [x] Explain V1 and the V2/V4 span-level path.
- [x] Include V3 only after its exact role is verified.
- [x] Define relation logits, probabilities, thresholds, and candidate units.
- [x] Define all equation symbols.
- [x] Place class weighting and general loss definitions here.
- [x] Synchronise the final checkpoint provenance and direct-output scope.
- [ ] Run the final citation and layout check.

### Chapter 5: Implementation and Experimental Setup

- [x] Move Mistral prompt, serialization, supervision, and training details from
  Chapter 6.
- [x] Define the Mistral and V4 output representations precisely.
- [x] Explain fixed-epoch Stage 3 training and validation-based V4 checkpoint
  selection.
- [x] State the random seeds in their respective model paths.
- [x] Use the final bbox-scale-fixed Stage 4 lineage and checkpoint 15.
- [x] Record numerical safeguards and candidate limits.
- [x] Keep scientifically relevant reproducibility details.
- [x] Move hashes, long paths, job identifiers, and low-level run records to the
  repository records.
- [x] Confirm the single checkpoint-15 test artifact and freeze its result
  lineage.
- [ ] Run the final cross-chapter terminology, citation, numerical, and layout
  check.

### Chapter 6: Experiments and Results

- [x] Apply the descriptive six-section structure in Section 9.
- [x] Address all 58 PDF comments and recurring issues.
- [x] Use final clean Mistral and checkpoint-15 V4 values.
- [x] Make the same-581-page comparison the central result.
- [x] Keep V1--V3 as development and diagnostic evidence.
- [x] Verify entity, cluster, recovery, and qualitative provenance.
- [x] Add an immediate interpretation after each important result.
- [x] Remove repeated Chapter 5 implementation detail and low-level run data.
- [x] Run the controlled cross-chapter result and terminology pass.
- [ ] Inspect revised tables and page layout during the later compilation.

### Chapter 7: Discussion

- [x] Synchronise final result values and remove unsupported final-lineage
  claims.
- [x] Preserve the deeper interpretations that the supervisor found useful.
- [x] Keep cross-result synthesis, limitations, literature links, causal
  uncertainty, implications, and future directions.
- [x] Remove only genuine duplication.
- [x] Permit a brief repeat of a central result when it supports the
  discussion.
- [ ] Run the final citation, reference, and layout check.

### Chapter 8: Conclusion

- [x] Synchronise final result values and checkpoint provenance.
- [x] Preserve the research-question review.
- [x] Put detailed contribution ownership in Chapter 1.
- [x] Present the contribution summary in connected prose.
- [x] Remove low-level development detail.
- [x] Add an explicit future-work paragraph.
- [x] End with the central thesis finding.
- [ ] Run the final citation, reference, and layout check.

### Abstracts and front matter

- [x] Synchronise both abstracts with the final result lineage.
- [x] Match the German abstract to the final English meaning for the changed
  evidence statements.
- [x] Review both abstracts after Chapters 7--8 are stable.
- [x] Check the AI-use declaration, glossary, bibliography, and front-matter
  order.

## 9. Implemented Chapter 6 structure

### 6.1 Evaluation Scope and Evidence

Defines Regular, Unkeyed, Unvalued, text, location, and text+location matching;
separates final test, validation-development, and diagnostic evidence; states
the same-581-page scope; and refers to Chapters 3--5 for full details.

### 6.2 Reconstructed Mistral-7B Reference Results

Reports final clean category-level results and keeps the published KVP10k
result as separate external context. Prompt, QLoRA, serialisation, supervision,
and parser details remain in Chapter 5.

### 6.3 Discriminative-Model Development

Uses descriptive subsections for entity-classification initialisation,
token-level candidate construction (V1), span-level candidate construction
(V2), and the predicted-span diagnostic (V3). It reports only evidence with
valid provenance.

### 6.4 Diagnostic Pipeline Analysis

Defines relation logits and probabilities, gives the bounding-box removal-rate
equation, and traces relation labels across word, token-pair, and span-pair
units. It describes the verified training/inference span-source mismatch and
does not claim missing teacher forcing.

### 6.5 Final V4 Evaluation

Makes checkpoint 15 the centre of the chapter. Validation Regular
text+location macro F1 0.331 selected it before test evaluation. The final
Regular test F1 values are 0.365 for text, 0.491 for location, and 0.329 for
text+location.

The section contains:

- one prominent same-581-page Mistral--V4 table;
- checkpoint-15 validation entity metrics, clearly labelled by split and unit;
- direct category coverage, with no unsupported recovery result;
- one comparative geometry table from saved final metrics and frozen cluster
  assignments;
- qualitative examples checked against checkpoint-15 saved predictions;
- only the training-record facts that affect interpretation.

### 6.6 Main Findings

Uses a compact evidence-scope-finding table and concise takeaways. It states
that reconstructed Mistral is stronger, complete pair extraction remains
harder than entity classification, direct V4 output has no Unkeyed or Unvalued
coverage, and geometry patterns are descriptive because cluster separation is
weak.

### Provenance decisions

- Final clean Mistral and checkpoint-15 V4 use exactly the same 581 document
  IDs.
- The old checkpoint-10 V4 result, old V4 recovery table, and old test-entity
  metrics are excluded from the final lineage.
- V4 cluster values are deterministic aggregates of saved final per-page
  metrics under the frozen cluster map; no model inference was repeated.
- Six qualitative cases were checked against saved checkpoint-15 predictions
  and ground truth. The old Syscode error was removed because checkpoint 15
  predicts it correctly.
- The final V4 decoder emits Regular pairs only. No checkpoint-15 recovery
  artifact exists.

## 10. Chapter 6 and Chapter 7 boundary

Use this division:

- Chapter 6 gives an immediate takeaway of one to three sentences after each
  result.
- Chapter 7 gives deeper explanation, limitations, literature connections, and
  implications.

Move or adapt only the immediate meaning of:

- the direct Mistral–V4 result;
- the entity-versus-pair result;
- the direct-output coverage result;
- the comparative cluster result;
- the qualitative examples.

Do not delete useful Chapter 7 material only because Chapter 6 gains a shorter
version.

## 11. Case-handling rules

| Case | Required action |
|---|---|
| Missing definition | Improve the first definition and add a short later reminder or cross-reference. |
| Missing citation | Cite the primary source, narrow the claim, or remove the unsupported claim. |
| Explanation exists later | Add a short local takeaway and keep the deeper later explanation. |
| Requested rationale | Check existing records. If no rationale is recorded, call the choice heuristic. |
| Requested metric | Verify split, unit, predictions, evaluator, and aggregation before reporting it. |
| Invalid or unavailable metric | Do not report it as a performance result. Explain the limitation. |
| Structural problem | Move the full explanation to its canonical chapter and leave a short cross-reference. |
| Missing interpretation | Add observation, comparison, cautious meaning, and limitation. |
| Terminology conflict | Correct reader-facing text globally. Do not rename artifacts. |
| Positive comment | Protect the material and use it as a local style model. |
| Ambiguous or empty comment | Do not guess. Record the reason for no change or request a decision. |
| Factual conflict | Stop the text edit and verify the unit, split, metric, and source. |
| Possible new run | Inspect existing artifacts first. Request approval before any new run. |
| Excessive detail | Keep the scientific decision and move low-level records to an appendix or repository. |
| Style issue | Perform a targeted search only. Do not start an unrelated language conversion. |
| Figure or table issue | Correct the source only when the visual materially improves understanding. Inspect the compiled PDF later. |

## 12. Exact chapter correction packages

Before editing each chapter, prepare a package with:

| Field | Required content |
|---|---|
| Location | File, section, and exact passage |
| Supervisor concern | Exact issue and whether it is recurring |
| Current text | Exact current source text |
| Proposed action | Add, replace, move, remove, cite, or retain |
| Proposed text | Exact replacement or insertion |
| Evidence | Internal source or primary citation |
| Related changes | Cross-reference, glossary, table, or duplicate text |
| Numerical change | Must normally be “No” |
| Decision | Any point that requires user approval |

The user approves each package before the chapter source changes.

After each approved chapter edit:

1. inspect the exact diff;
2. run `git diff --check`;
3. list all changed files;
4. confirm that no unrelated file changed;
5. verify changed claims and numbers;
6. check citations and cross-references;
7. update the issue and strengths ledgers.

## 13. Conservative use of tables and figures

Add a visual element only when it materially improves understanding.

The two planned additions with clear value are:

1. one compact dataset-flow table in Chapter 3, if the current text cannot show
   the split and exclusion flow clearly;
2. one direct reconstructed-Mistral versus V4 table in Chapter 6.

Use two or three explanatory sentences instead of a new table or figure when
prose is sufficient.

## 14. Final checks and definition of completion

The revision is complete only when:

- all 81 annotations have a final status;
- all global email concerns have a thesis-wide check;
- positive material remains intact;
- all affected values agree with verified evidence;
- all direct comparisons use compatible data and metrics;
- named methods and datasets have suitable first-use citations;
- important terms and acronyms are defined at first use;
- each result table has an introduction and an immediate interpretation;
- Chapters 6 and 7 have distinct purposes;
- Chapters 1, 6, 7, 8, and both abstracts agree;
- no code or experimental artifact changes without explicit approval;
- `git diff --check` succeeds;
- LaTeX has no undefined citations, undefined references, duplicate labels, or
  unresolved layout errors;
- the final PDF passes a page-by-page visual check;
- the final Git diff contains only approved thesis files.

## 15. Immediate next action

Review the final targeted content and citation diff. Then perform the approved
full compilation and page-by-page PDF inspection, including the identified
table, figure, and appendix-listing layout checks.
