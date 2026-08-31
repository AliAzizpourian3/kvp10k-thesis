# Whole-Thesis Supervisor-Compliance and Writing Audit

## Executive summary

The compiled thesis is now **substantively close to final**. The major result-lineage problems, chapter-structure problems, unsupported recovery claims, old entity metrics, duplicated discussion material, and most terminology problems have been removed. The current PDF tells one coherent scientific story: V1 is token-level linking, V2 is the first span-level development path, V3 is a diagnostic span-level configuration without a headline benchmark result, and V4 is the final LayoutLMv3-based span-level discriminative system; the reconstructed Mistral-7B reference system obtains Regular text+location F1 **0.657**, V4 obtains **0.329**, final-checkpoint validation entity micro/macro F1 are **0.813/0.795**, checkpoint **15** is the validation-selected final V4 checkpoint, and entity-pretraining **epoch 10** is a separate earlier checkpoint. fileciteturn0file0

The supervisor revision plan records **81 annotations**: one abstract comment, 19 across Chapters 1–5, 58 in Chapter 6, one in Chapter 7, and two in Chapter 8. The Chapter 6 ledger records 47 `DONE`, 10 `ADDRESSED ELSEWHERE`, and one `NO CHANGE` because the annotation itself was empty. The substantive content of the current PDF supports the conclusion that **all 81 highlighted annotations have been dealt with**. However, the plan still lacks a formal final status entry for the one abstract annotation. Once that bookkeeping item is entered, the overall ledger would read **69 `DONE`, 11 `ADDRESSED ELSEWHERE`, and 1 `NO CHANGE`**. fileciteturn0file3

I do **not** recommend another broad rewrite. I found a small set of remaining issues:

| Audit area | Verdict | Main remaining issue |
|---|---|---|
| Supervisor annotations | **Substantively complete** | Abstract annotation should be formally marked `DONE` in the ledger |
| Final results/provenance | **Clean** | No stale internal `0.345` or `0.662` remains in the compiled PDF |
| Chapter boundaries | **Clean** | Methods/Results/Discussion/Conclusion separation is now strong |
| Causal wording | **Mostly clean** | One Chapter 6 entity-vs-pair sentence should be softened |
| Citations | **Small but important residual issue** | KVP10k paper is presently made to appear to support the 9,656/1,051 split counts; it does not |
| Evaluator provenance | **One citation missing** | Chapter 2's `IoU >= 0.3` code claim needs the repository citation |
| Legacy terminology | **Two residues** | `Stage 4a` in Chapter 4 and `Stage 2` in Appendix A |
| Front matter | **Clean** | No change recommended |
| Abbreviations | **Clean** | Previous unused entries are gone; t-SNE is corrected |
| Layout | **Needs final pass** | Table 6.9, Figure 3.1 and Appendix Listing A.3 are the main visual issues |
| Repository | **High-priority synchronisation risk** | Connected GitHub `master` still contains an old thesis/result lineage |

The most important new discovery is **citation attribution**, not an experimental problem. The KVP10k paper reports **10,707 pages** and the published Regular baseline values, including text+location F1 **0.611**. It does **not** state the unique train/test counts **9,656/1,051**. Those counts are obtained from the released train/test annotation data and should therefore be described as release-derived counts, rather than presented as figures sourced directly from the paper. citeturn4view0turn5view0

The second important discovery is operational: the **connected GitHub default branch is substantially behind the compiled thesis**. Its Chapter 8 still contains obsolete `0.345`, `0.662`, Stage 4a wording, and the obsolete recovery result `0.229 → 0.261`; its Chapter 6 likewise retains the old experiment hierarchy and old reconstructed-Mistral result lineage. This does not invalidate the current PDF, but the repository must be synchronised with the approved working tree before the project is considered reproducible. fileciteturn6file0 fileciteturn8file0

My recommendation is therefore **one very small content/citation correction pass**, followed by compilation and the visual pass. Do not reopen experiments, do not alter result values, and do not undertake a broad stylistic rewrite.

## Supervisor feedback and result provenance

### Annotation-by-annotation compliance

The revision plan's chapter records and the current PDF agree on the substantive disposition of the supervisor's comments. The supervisor also made it clear that highlighted citation/context problems were examples rather than an exhaustive list; this whole-thesis audit therefore treats the global citation and definition pass as part of supervisor compliance rather than limiting the check to the 81 highlighted locations. fileciteturn0file3

| Part | Comments | Current audit | Remaining action |
|---|---:|---|---|
| Abstract | 1 | **Addressed substantively.** It now defines the task, KVP10k, reconstructed Mistral and V4 before reporting results. | Add explicit `DONE` status to the revision ledger. |
| Chapter 1 | 6 | **All addressed.** Task, KVP terminology, model context, relation problem, thesis scope and ownership are clear. | None. |
| Chapter 2 | 8 | **All highlighted comments addressed.** Relevant methods/datasets are introduced and cited. | Two *global citation-audit* fixes remain; see below. |
| Chapter 3 | 2 | **Both addressed.** Dataset preparation and annotation-geometry analysis are explained. | Remove one remaining `Stage 2` reference in the appendix, not in Chapter 3 itself. |
| Chapter 4 | 2 | **Both addressed.** Architecture, candidate paths and ownership are now clear. | Delete an unnecessary historical `Stage 4a` parenthesis. |
| Chapter 5 | 1 | **Addressed.** Low-level run/provenance clutter was removed and implementation material is placed appropriately. | None. |
| Chapter 6 | 58 | **All explicitly processed.** 47 `DONE`, 10 `ADDRESSED ELSEWHERE`, C40 `NO CHANGE` because it was empty. | One cautious-wording refinement and one visible typo; visual table cleanup later. |
| Chapter 7 | 1 | **Addressed elsewhere as intended.** Chapter 6 now interprets results immediately; Chapter 7 performs synthesis. | One terminology consistency edit. |
| Chapter 8 | 2 | **Both addressed.** Contributions are prose, non-RQ headings were removed, future work was added. | None. |

The current abstract is particularly successful relative to the supervisor's comment. It no longer assumes the reader understands KVP10k, Mistral or V4; it describes KVP extraction as identifying text, boxes and relations, identifies Regular versus incomplete entries, explains both systems, states the shared 581-page test subset, and then presents the main results. It contains no checkpoint-15, epoch-10, scheduler or “evaluated once” provenance detail. The German version carries the same scientific meaning. fileciteturn0file0

### Result and checkpoint lineage

A whole-PDF text scan found **no occurrences of the stale internal values `0.345`, `0.662`, `0.661788`, `0.229` or `0.261`**. The number `0.611` still appears, correctly, as the **published KVP10k Regular text+location baseline**, and the KVP10k paper's Table 1 confirms that published value. fileciteturn0file0 citeturn5view0

The final result lineage in the compiled thesis is internally coherent:

| Evidence | Current value/provenance | Audit |
|---|---|---|
| Reconstructed Mistral Regular text F1 | **0.757** | Correct |
| Reconstructed Mistral Regular location F1 | **0.694** | Correct |
| Reconstructed Mistral Regular text+location F1 | **0.657** | Correct |
| Final V4 Regular text F1 | **0.365** | Correct |
| Final V4 Regular location F1 | **0.491** | Correct |
| Final V4 Regular text+location F1 | **0.329** | Correct |
| V4 T+L precision / recall | **0.347 / 0.313** rounded | Correct |
| V4 validation entity micro / macro F1 | **0.813 / 0.795** | Correctly labelled validation evidence |
| Entity-pretraining selection | **epoch 10**, validation KEY/VALUE micro F1 **0.829** | Correct |
| Final V4 selection | **checkpoint 15**, validation Regular T+L F1 **0.331** | Correct |
| Test-use timing | Test evaluated after validation selection | Correct |
| Direct V4 special-category coverage | Regular only; All T+L **0.214** | Correct |
| Checkpoint-15 recovery | **No synchronised recovery artefact; no recovery score claimed** | Correct |

These figures appear consistently in the abstract, Chapters 6–8 and the concluding scientific narrative. fileciteturn0file0

The checkpoint distinction is also clear in the current Results chapter: Section 6.3.1 identifies **epoch 10** as the selected entity-pretraining checkpoint; Section 6.5 states that validation Regular text+location F1 selected **checkpoint 15** before test evaluation; Section 6.5.1 explicitly reminds the reader that the transferred entity-pretraining checkpoint was selected separately at epoch 10. This removes the earlier risk of treating the two numbers as referring to the same training phase. fileciteturn0file0

### V1–V4 naming

The names **V1, V2, V3 and V4 should remain**. Their current use is coherent rather than stale:

**V1** is the token-level candidate formulation; **V2** is the first span-level development formulation; **V3** is an intermediate diagnostic configuration without a separate headline result; **V4** is the final LayoutLMv3-based span-level discriminative system. The thesis no longer relies on “Experiment 1–8” or “Stage 3/4a” as its principal reader-facing structure. fileciteturn0file0

This is also conceptually defensible against the underlying literature. LayoutLMv3 is correctly presented as a multimodal document pre-training architecture using unified text/image masking and word-patch alignment, while the thesis-specific classifier/linker/span path is distinguished from that pretrained component. citeturn7academia13 The Mistral reference is correctly treated as a reconstruction built around the Mistral-7B family rather than as a model invented by the thesis. citeturn8academia0 QLoRA and LMDX are likewise cited to their primary publications rather than being presented as thesis contributions. citeturn7academia12turn7search0

## Chapter architecture and writing quality

### Methods, implementation and results are now separated properly

The Methods/Results boundary is substantially improved. Chapter 4 owns architecture, equations, losses, token/span candidates and the training design. Chapter 5 owns concrete preparation, serialisation, QLoRA settings, inference, checkpoint management and the evaluator implementation. Chapter 6 references those chapters rather than re-teaching them before every result. fileciteturn0file0

This directly resolves one of the most persistent Chapter 6 supervisor concerns: the chapter no longer reads as a sequence of eight vaguely comparable “experiments”. It now has a logical evidence progression from evaluation scope, through the reconstructed Mistral reference, discriminative development, diagnostics and finally V4 evaluation. fileciteturn0file3

Chapter 6 also now contains **immediate interpretation**, as requested by the supervisor. Examples include the Mistral comparison caveat immediately after the reference result, interpretation of the V2 logit distribution after reporting it, interpretation of bounding-box retention immediately after the diagnostic counts, interpretation of the V4 same-page comparison after Table 6.4, entity-versus-pair interpretation after Table 6.5, output-coverage interpretation after Table 6.6, and the geometry-strata interpretation after Table 6.7. fileciteturn0file0

Chapter 7 no longer duplicates those tables and raw diagnostics. Its opening explicitly tells the reader that Chapter 6 provides immediate interpretations and that the Discussion relates the results across evidence types. The current Section 7.1 then synthesises entity-versus-pair extraction, candidate granularity, system-level comparison and geometry strata; Section 7.2 synthesises development diagnostics; later sections cover qualitative errors, literature context, limitations and future directions. That is a good Discussion role. fileciteturn0file0

Chapter 8 now does what the supervisor asked: **Research Questions Revisited** is the only section-level structure, RQ1–RQ4 are answered directly, future work appears as ordinary concluding prose rather than another hierarchy of headings, and the final paragraph ends with the scientific conclusion that plausible entities are not sufficient unless the system also forms the right spans and relations. fileciteturn0file0

### Repetition

The remaining repetition is **mostly purposeful rather than problematic**. `0.657/0.329` appears in the abstract, central Results comparison, Discussion synthesis and Conclusion because it is the principal system comparison. Likewise, `0.813/0.331` is repeated because it supports the thesis's principal internal diagnostic conclusion. The former raw diagnostic chains, class-level entity values, duplicated cluster table, published-baseline recap and recovery discussion have been removed from the later chapters. fileciteturn0file0

I therefore do **not** recommend another “remove repetition” sweep. Doing so now risks deleting the useful reinforcement of the main results.

### Causal language

Most causal language is now appropriately cautious. Chapter 7 repeatedly says that the development paths do not isolate candidate granularity as a causal factor, the geometry comparison does not identify a causal mechanism, V4's combined corrections cannot quantify the contribution of one change, and qualitative cases are illustrative rather than frequency estimates. Chapter 8 likewise states that no isolated causal effect is assigned to any one pipeline change. fileciteturn0file0

There is, however, **one sentence in Chapter 6 that is stronger than the otherwise careful treatment**. In §6.5.3 the current text effectively says:

> “The difference shows that complete pair extraction remains harder than token classification. Together with the verified cases … it identifies the entity-to-relation decoding path as a major internal limitation.”

Because **0.813 token micro F1 and 0.331 page-macro pair F1 use different units**, they cannot by themselves quantify how much harder one task is. Chapter 7 and Chapter 8 already phrase the same interpretation more carefully. I recommend making Chapter 6 consistent:

```tex
Although the metrics are not directly comparable, the lower complete-pair
result is consistent with additional errors arising between entity
classification and complete-pair extraction. Together with the verified cases
in Table~\ref{tab:v4-qualitative-errors}, this supports the interpretation that
the entity-to-relation decoding path is a major internal limitation. It does
not isolate the linker from upstream label and span errors.
```

This keeps the immediate takeaway that the supervisor liked while removing an unnecessary implication of a controlled causal decomposition. fileciteturn0file0

### Voice and general prose

There is **no systemic passive-voice problem**. Most prose uses neutral scholarly constructions such as “the analysis identifies”, “the system uses”, “the chapter reports”, or “the comparison shows”. There are isolated first-person constructions, for example “We use AdamW” in Chapter 4, but they are not frequent enough to justify a global voice rewrite. fileciteturn0file0

The thesis also now defines the terminology that previously caused readability problems. OCR, LSTM, BIO, LMDX, LoRA/QLoRA, NED, IoU, PCA and t-SNE are introduced in prose before or around their technical use. I found no remaining global definition gap comparable to the ones your supervisor originally highlighted. fileciteturn0file0

Two **legacy labels** remain and should be removed because they are no longer useful to the reader:

In Chapter 4 §4.6.3:

```tex
\textbf{Entity-classification initialisation (historically Stage 4a):}
```

should become:

```tex
\textbf{Entity-classification initialisation:}
```

And Appendix A.2 still says:

> “training-fitted Stage 2 artifact”

and captions Listing A.3 as:

> “frozen Stage 2 scaler and k-means model”.

The revision plan explicitly identified later-chapter/appendix `Stage 2` removal as a deferred global integration task, so this is genuinely unfinished rather than merely stylistic. fileciteturn0file3

Use instead:

```tex
training-fitted annotation-geometry clustering artifact
(including its scaler and k-means model)
```

and:

```tex
\caption{Assignment of test pages with the frozen training-fitted
annotation-geometry scaler and k-means model}
```

The internal code variable `stage2_artifact` should **not** be renamed merely for thesis prose consistency; it is an implementation identifier.

One more terminology edit is advisable in the latest Chapter 7 source, §7.2, around lines 102–105. It currently says:

```tex
V4 changes coordinate handling, span retention, and relation-label
construction together.
```

Use the terminology employed elsewhere:

```tex
V4 changes coordinate handling, bounding-box filtering, and relation-label
construction together.
```

This is a small but worthwhile consistency fix.

## Citations, definitions and front matter

### Primary-source quality is generally strong

The bibliography and in-text citations now rely predominantly on primary papers for the central methods. LayoutLMv3 is supported by its original paper, which describes its unified text/image masking and word-patch-alignment objectives. citeturn7academia13 Mistral is linked to the original Mistral-7B technical paper. citeturn8academia0 QLoRA is supported by the original paper describing a frozen 4-bit quantised base model with LoRA adapters, NF4, double quantisation and paged optimisers. citeturn7academia12 LMDX is cited to its ACL publication, which explicitly addresses document information extraction and grounding/localisation. citeturn7search0

The KVP10k paper is also being used correctly for the **task definition, 10,707-page total and published baseline results**. The paper states that KVP10k contains 10,707 annotated images and targets KVP discovery without predefined keys, and its Table 1 reports the published Regular F1 values `0.659/0.650/0.611`. citeturn2search1turn5view0

### High-priority citation-attribution correction

The paper does **not** provide the unique split counts **9,656 training / 1,051 test**. A search of the paper finds no `9656` or `1051`. Those are counts obtained from the released train/test annotation data after grouping the annotation rows by page identity. citeturn5view1turn5view2

The current thesis makes the paper look like the source for these split counts in at least three places:

**Chapter 2 §2.6**, PDF printed p. 9:

> “KVP10k [Nap+24] … contains 10,707 unique pages, with 9,656 training pages and 1,051 test pages.”

**Chapter 3 §3.1**, PDF printed p. 13:

> “The KVP10k dataset … [Nap+24]. It contains 10,707 unique pages, with 9,656 pages in the official training split and 1,051 pages in the official test split.”

**Chapter 7 §7.5.1**, latest source lines 144–148:

```tex
the official KVP10k training split contains 9{,}656 unique pages, and its test split
contains 1{,}051 unique pages~\cite{kvp10k}.
```

This is the clearest remaining example of the kind of citation problem your supervisor asked you to check globally.

I recommend making Chapter 3 the canonical provenance statement:

```tex
The KVP10k paper reports 10{,}707 pages~\cite{kvp10k}. In the released
annotation data used in this thesis, grouping annotation rows by
\texttt{hash_name} gives 9{,}656 unique pages in the training partition and
1{,}051 in the test partition. The public release provides training and test
partitions but no separate validation partition~\cite{kvp10k_repository}.
```

The official KVP10k repository documents the released annotation structure, including `hash_name`, and the train/test preparation and benchmarking pipeline. citeturn3search0

Then Chapter 2 can simply say:

```tex
KVP10k~\cite{kvp10k} is a benchmark for open-vocabulary key--value pair
extraction from business documents. The paper reports 10{,}707 pages. In the
released annotations used in this thesis, the train and test partitions contain
9{,}656 and 1{,}051 unique pages, respectively, as detailed in
Section~\ref{sec:kvp10k}.
```

And Chapter 7 should become:

```tex
The KVP10k paper reports 10{,}707 pages~\cite{kvp10k}. In the released
annotations used here, Section~\ref{sec:kvp10k} counts 9{,}656 unique training
pages and 1{,}051 unique test pages. The local preparation pipeline retained
5{,}389 training pages and 581 test pages.
```

This produces a clean provenance chain:

**paper → 10,707 total**;  
**released annotation data → 9,656/1,051 unique split counts**;  
**local preparation → 5,389/581**.

That distinction is methodologically stronger than trying to attach all three levels to the KVP10k paper.

### Evaluator-code citation

Chapter 2 §2.5 correctly observes that the paper says location IoU must be **above 0.3**, whereas the released code accepts **exactly 0.3**. The paper indeed says the IoU must exceed 0.3. citeturn4view0 The released `benchmark.py` passes a threshold of `0.3`, and the official `MetricsCalculator` implements the match as:

```python
return self._get_intersection_over_union(box1, box2) >= min_iou_4_same_location
```

so the thesis's scientific observation is correct. fileciteturn11file0 fileciteturn12file0

The only problem is that the Chapter 2 sentence currently lacks the **repository citation**. Add it:

```tex
An inspection of the released evaluator~\cite{kvp10k_repository} shows that it
accepts an IoU value of exactly $0.3$, because its implemented boundary is
inclusive ($\geq 0.3$).
```

Chapter 5 already handles this paper/code distinction more explicitly, so this is a citation completion rather than a methodological change.

### Abstracts and declarations

The final English and German abstracts now match the final result lineage. Both state `0.365/0.491/0.329`, Mistral `0.657`, and validation entity `0.813/0.795`; both explain that entity and pair metrics use different units and avoid claiming that the linker alone explains the gap. Neither contains checkpoint-selection, resume, epoch or recovery details. fileciteturn0file0

The most recent approved front-matter pass changed only the two abstracts, abbreviation list and the stale Chapter 1 roadmap sentence; it explicitly excluded the AI-use declaration, statutory declaration, bibliography order, title information and acknowledgements. fileciteturn0file1

The compiled PDF contains both declarations:

- the statutory German declaration appears before the AI declaration;
- the separate AI-use declaration identifies ChatGPT and Codex and lists writing/style, structural-consistency, passage drafting/revision, programming, debugging and code-review assistance;
- it states that AI outputs were critically checked against sources, code and experimental records and that AI tools were not used as scientific sources. fileciteturn0file0

I recommend **no automatic declaration edit**. The only remaining responsibility is personal rather than editorial: because the declaration says all uses are disclosed completely and truthfully, only you can confirm that its list still describes your actual use. Codex should not alter that self-declaration on its own.

The title, advisor and dates are internally consistent in the PDF. Whether birth details and the exact registered title match the university registration cannot be determined from the repository; that should be checked manually against the registration record. fileciteturn0file0

### Abbreviations

The abbreviation cleanup is successful. The current compiled list no longer contains the unused `NLP`, `LM`, `LLM`, `UMAP`, `HPC`, `SLURM`, `DPI` or `RoI` entries. `F1` and `PDF` remain, and the t-SNE expansion now reads **“t-distributed stochastic neighbor embedding”** rather than the former inconsistent capitalisation. fileciteturn0file0

Core technical abbreviations are also introduced in prose before substantive use: OCR, LSTM, BIO, LMDX, LoRA, QLoRA, NED, IoU, PCA and t-SNE all receive explanations. No further abbreviation pass is necessary.

## Repository and visual-layout audit

### The connected GitHub branch is stale

This is the most important repository-level finding.

The connected repository is `AliAzizpourian3/kvp10k-thesis`, but its default `master` branch does **not** represent the thesis shown in the current compiled PDF. The remote Chapter 8 still contains the old enumerated contribution section, old V4 **0.345**, old reconstructed Mistral **0.662**, a claim that the entity-pretraining checkpoint does not survive, old cluster values, old recovery `0.229 → 0.261`, and the removed `Practical Implications`/`Closing Remarks` structure. fileciteturn6file0

The remote Chapter 6 is likewise from the old result lineage: it begins with `Overview and Chapter Structure`, uses the old `Experiment 1`, `Experiment 2`, etc. organisation, reports reconstructed Mistral Regular text+location **0.662**, and retains older diagnostic material. fileciteturn8file0

This means there are currently two realities:

**Compiled/local thesis:** final corrected lineage, `0.657/0.329`, revised Chapters 6–8. fileciteturn0file0  
**Connected remote `master`:** obsolete lineage and old chapter structure. fileciteturn6file0 fileciteturn8file0

Do **not** use the remote branch as the source of truth for further thesis edits. Codex should work against the current local working tree. After the final PDF is accepted, the current source should be committed and pushed, and a final repository check should confirm that searching the remote tree for `0.345` and `0.662` does not find obsolete thesis claims.

### Figure and table review

The current PDF was visually inspected at the previously flagged pages.

**Figure 4.1, PDF p. 44 / printed p. 26:** the earlier arrow/text-overlap concern appears **resolved in this compile**. The encoder, entity classifier, span grouper, linker and candidate-path labels are legible, and the ownership annotations are visible. I would not alter the figure now unless the next compilation changes its geometry. fileciteturn0file0

**Figure 5.1, PDF p. 59 / printed p. 41:** the prompt is readable at normal page scale; the caption explains `{lmdx_text}` and the singleton Unvalued exception. No immediate layout correction is required. fileciteturn0file0

**Figure 3.1, PDF p. 33 / printed p. 15:** this remains a legitimate visual-pass item. Both document examples are small relative to the page, especially panel (a), so the annotation outlines are difficult to inspect; the page also contains substantial unused horizontal space. The supervisor wanted concrete page examples, so increasing the useful image area would improve the effectiveness of an already-correct figure. Crop the examples more tightly and/or enlarge the two panels; do not change the underlying examples. fileciteturn0file0

**Table 6.8, PDF p. 75 / printed p. 57:** it is scientifically readable. Some cells wrap awkwardly (`Competing nearby value`, `Merged value span`, long saved-output descriptions), but I would treat this as **medium/low visual priority** rather than a content problem. fileciteturn0file0

**Table 6.9, PDF p. 76 / printed p. 58:** this is the most obvious table-formatting problem. Narrow justified columns generate distracting internal word breaks such as `Mis-tral`, `mi-cro`, `diagnos-tic`, `mis-match`, `rela-tion`, `clus-ter`, and similar forms. The table should be reformatted during the visual pass, probably with wider/ragged-right text columns rather than further shortening its content. fileciteturn0file0

The paragraph immediately below Table 6.9 also contains a genuine visible text error:

> “important **candidate- construction** and supervision constraints”

Change it to:

> “important **candidate construction** and supervision constraints”.

That is a source-text correction and should be made before the visual pass. fileciteturn0file0

**Appendix Listing A.3, PDF pp. 88–89 / printed pp. 70–71:** the listing spills only its last three lines onto the next page, leaving printed p. 71 almost entirely blank. This is a strong layout-pass candidate. A small local reduction in listing spacing/font size or surrounding vertical space should allow A.3 to remain on one page. Do not shrink all code listings globally merely to solve this one page. fileciteturn0file0

The abbreviation list uses a second page for only four entries. This is visually sparse but not defective, and I would **not** spend time altering the glossary template merely to save one page unless the final compile makes it particularly distracting. fileciteturn0file0

### Compile-state caveat

The compiled PDF contains no visible `??` or `[?]` unresolved-reference markers in the extracted text, which is encouraging. A definitive LaTeX reference/citation audit still requires the final compiler log after the targeted edits. Therefore undefined references, duplicate labels, overfull boxes and bibliography warnings belong to the **next compilation check**, not to another substantive manuscript rewrite.

## Targeted corrections and remaining workflow

### Necessary source changes

The following is the complete correction package I would give Codex. I would **not** ask it to make anything broader.

| File | Exact location / current snippet | Required change | Rationale | Priority |
|---|---|---|---|---|
| `chapters/02_literature_review.tex` | §2.5: `An inspection of the released evaluator shows that it accepts an IoU value of exactly...` | Add `\cite{kvp10k_repository}` after “released evaluator”. | The `>=0.3` observation comes from released code, not the paper. | **High** |
| `chapters/02_literature_review.tex` | §2.6: `It contains 10{,}707 unique pages, with 9{,}656 training pages and 1{,}051 test pages.` | Separate paper-supported 10,707 total from release-derived 9,656/1,051; cross-reference §3.1. | Prevent KVP10k paper from appearing to support split counts it does not report. | **High** |
| `chapters/03_dataset.tex` | §3.1 opening: `It contains 10{,}707 unique pages, with 9{,}656 pages in the official training split...` | State that paper reports 10,707; state that grouping released annotation rows by `hash_name` yields 9,656/1,051. | Makes the provenance of the canonical split counts explicit. | **High** |
| `chapters/07_discussion.tex` | §7.5.1, latest source lines 144–148: `official KVP10k training split contains 9{,}656...~\cite{kvp10k}` | Cite paper only for 10,707 total and refer to §3.1 for release-derived split counts. | Same citation-attribution issue in Discussion. | **High** |
| `chapters/06_experiments_results.tex` | §6.5.3 entity-versus-pair paragraph | Replace “difference shows… identifies…” with the cautious wording proposed above. | Metrics use different units; preserve interpretation without overclaiming. | **Medium** |
| `chapters/06_experiments_results.tex` | §6.6 after Table 6.9: `candidate- construction` | Replace with `candidate construction`. | Visible typo in final-facing summary. | **Medium** |
| `chapters/04_methodology.tex` | §4.6.3: `Entity-classification initialisation (historically Stage 4a)` | Remove `(historically Stage 4a)`. | Removes unnecessary legacy hierarchy. | **Medium** |
| `chapters/07_discussion.tex` | §7.2, latest source lines 102–103: `coordinate handling, span retention, and relation-label construction` | Use `coordinate handling, bounding-box filtering, and relation-label construction`. | Matches canonical terminology across Chapters 1, 6 and 8. | **Medium** |
| `chapters/appendix_a_code.tex` | Appendix A.2 prose: `training-fitted Stage 2 artifact`; Listing A.3 caption: `frozen Stage 2 scaler...` | Replace reader-facing `Stage 2` with `annotation-geometry clustering` wording. Leave `stage2_artifact` code variable unchanged. | This is explicitly listed as a deferred global terminology task in the revision plan. | **Medium** |
| `THESIS_SUPERVISOR_REVISION_PLAN.md` | Abstract/front-matter section currently has checked result synchronisation but no explicit status for the one abstract annotation | Record the abstract annotation as `DONE` after the above audit. | Makes the 81-comment ledger formally auditable. | **Medium** |

For the split-count patch, I recommend these exact formulations.

**Chapter 3 canonical version:**

```tex
The KVP10k paper reports 10{,}707 pages~\cite{kvp10k}. In the released
annotation data used in this thesis, grouping annotation rows by
\texttt{hash_name} gives 9{,}656 unique pages in the training partition and
1{,}051 in the test partition. The public release provides training and test
partitions but no separate validation partition~\cite{kvp10k_repository}.
```

The paper itself supports the 10,707 total, while the official repository documents the released data structure and annotation identifiers. citeturn2search1turn3search0

**Chapter 2:**

```tex
KVP10k~\cite{kvp10k} is a benchmark for open-vocabulary key--value pair
extraction from business documents. The paper reports 10{,}707 pages. In the
released annotations used in this thesis, the train and test partitions contain
9{,}656 and 1{,}051 unique pages, respectively, as detailed in
Section~\ref{sec:kvp10k}.
```

**Chapter 7:**

```tex
The KVP10k paper reports 10{,}707 pages~\cite{kvp10k}. In the released
annotations used here, Section~\ref{sec:kvp10k} counts 9{,}656 unique training
pages and 1{,}051 unique test pages. The local preparation pipeline retained
5{,}389 training pages and 581 test pages.
```

**Chapter 2 evaluator citation:**

```tex
An inspection of the released evaluator~\cite{kvp10k_repository} shows that it
accepts an IoU value of exactly $0.3$, because its implemented boundary is
inclusive ($\geq 0.3$).
```

The official code confirms both the benchmark threshold and inclusive comparison. fileciteturn11file0 fileciteturn12file0

### Changes that should not be made

No result values should change. The English and German abstracts should remain as compiled. Chapter 8 does not need another substantive rewrite. V1/V2/V3/V4 should **not** be renamed. The AI and statutory declarations should not be rewritten by Codex. Front-matter/bibliography order should not be rearranged merely to satisfy the old PRL document. The contribution language should not be expanded again. No experiment, recovery run or entity-test result should be recreated merely to fill an old historical gap. fileciteturn0file0 fileciteturn0file3

### Visual pass after those edits

The visual pass should concentrate on only four places:

| Location | Action | Priority |
|---|---|---|
| Table 6.9, PDF p. 76 / printed p. 58 | Widen/rebalance text columns; use ragged-right cells to eliminate excessive hyphenation | **High visual** |
| Appendix A.3, PDF pp. 88–89 / printed pp. 70–71 | Keep the three orphaned final code lines on the preceding page with a local listing adjustment | **High visual** |
| Figure 3.1, PDF p. 33 / printed p. 15 | Crop/enlarge page examples so the annotations are inspectable | **Medium visual** |
| Table 6.8, PDF p. 75 / printed p. 57 | Improve wrapping only if convenient after Table 6.9 is fixed | **Low/medium visual** |

Figure 4.1 and Figure 5.1 should simply be rechecked after recompilation; **I do not currently see a reason to redesign them**. fileciteturn0file0

### Recommended remaining sequence

```mermaid
flowchart LR
    A["Whole-thesis audit<br/>complete"] --> B["Targeted content/citation fixes<br/>no result changes"]
    B --> C["Compile full thesis<br/>inspect log"]
    C --> D["Visual/layout pass<br/>Table 6.9, Appendix A.3,<br/>Figure 3.1, Table 6.8"]
    D --> E["Recompile and page-by-page PDF check"]
    E --> F["Final consistency scan<br/>values, citations, refs, labels"]
    F --> G["Synchronise Git repository<br/>commit/push final source"]
    G --> H["Verify remote source matches<br/>submitted PDF"]
```

The repository synchronisation step is important because the connected `master` branch is presently old enough to contradict the submitted scientific results. It should happen **after** the final content/layout changes, not before, so there is one authoritative final commit rather than another sequence of partial commits. fileciteturn6file0 fileciteturn8file0

The final thesis therefore does **not** need another broad supervisor-feedback rewrite. It needs a **small citation/provenance correction package, three terminology/writing cleanups, one typo correction, then compilation and layout work**. Once those are complete and the revision plan records the abstract annotation explicitly, the substantive supervisor-compliance ledger can reasonably be considered closed.