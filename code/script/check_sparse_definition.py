"""
check_sparse_definition.py
Run from: /home/woody/iwi5/iwi5413h/kvp10k_thesis/
Output:  data/outputs/stage2/sparse_definition_check.txt
"""

import pickle, json, os, glob
import numpy as np
from pathlib import Path
from collections import defaultdict

OUT = "data/outputs/stage2/sparse_definition_check.txt"
lines = []

def log(s=""):
    print(s)
    lines.append(s)

# ── 1. Stage 2 pickle ─────────────────────────────────────────────────────────
with open("data/outputs/stage2/cluster_assignments.pkl", "rb") as f:
    s2 = pickle.load(f)

cr   = s2["clustering_result"]
km   = cr["kmeans"]
sc   = cr["scaler"]
feats = np.asarray(s2["layout_features"])   # shape (N, 13)
labs  = np.asarray(s2["cluster_labels"])    # 0 / 1

# Which label is Dense? (higher mean n_boxes = feature col 0)
mean_nbox = [feats[labs == c, 0].mean() for c in [0, 1]]
dense_label  = int(np.argmax(mean_nbox))
sparse_label = 1 - dense_label

log("=" * 60)
log("CLUSTER FEATURE STATISTICS (raw, before scaling)")
log("=" * 60)
feature_names = [
    "n_boxes", "n_kvp_boxes", "kvp_density",
    "vertical_spread", "mean_spacing", "std_spacing",
    "mean_box_width", "mean_box_height",
    "mean_aspect_ratio", "std_aspect_ratio",
    "coverage", "n_columns_est", "page_fill_ratio"
]
for c, name in [(dense_label, "Dense"), (sparse_label, "Sparse")]:
    mask = labs == c
    log(f"\n  {name} cluster  (n={mask.sum()})")
    for i, fn in enumerate(feature_names[:6]):   # first 6 most interpretable
        col = feats[mask, i]
        log(f"    {fn:20s}  mean={col.mean():.2f}  median={np.median(col):.2f}  "
            f"min={col.min():.2f}  max={col.max():.2f}")

# ── 2. Sparse docs in Stage 3 eval ───────────────────────────────────────────
log("\n" + "=" * 60)
log("SPARSE DOCS IN STAGE 3 EVAL — OCR TOKEN COUNT")
log("=" * 60)

with open("data/outputs/stage2/test_cluster_map.json") as f:
    cluster_map = json.load(f)   # hash_name -> "Cluster_0_Dense" / "Cluster_1_Sparse"

eval_path = "data/outputs/stage3_mistral/evaluation_stage0_ned02_iou03.json"
if not os.path.exists(eval_path):
    # fallback to whichever eval JSON exists
    candidates = glob.glob("data/outputs/stage3_mistral/evaluation*.json")
    eval_path  = candidates[0] if candidates else None

eval_hashes = set()
if eval_path:
    with open(eval_path) as f:
        ev = json.load(f)
    eval_hashes = {d["hash_name"] for d in ev.get("per_document", [])}

sparse_hashes = {h for h, c in cluster_map.items()
                 if "Sparse" in c and h in eval_hashes}
dense_hashes  = {h for h, c in cluster_map.items()
                 if "Dense"  in c and h in eval_hashes}

log(f"\n  Sparse docs evaluated: {len(sparse_hashes)}")
log(f"  Dense  docs evaluated: {len(dense_hashes)}")

# OCR word count from lmdx_text in prepared/test GT files
gt_dir = "data/prepared/test"
gt_files = {Path(p).stem: p for p in glob.glob(f"{gt_dir}/*.json")}

def word_count(hash_name):
    p = gt_files.get(hash_name)
    if not p:
        return None
    with open(p) as f:
        d = json.load(f)
    txt = d.get("lmdx_text", "")
    if isinstance(txt, list):
        return len(txt)
    return len(txt.split())

sparse_wc = [word_count(h) for h in sparse_hashes]
dense_wc  = [word_count(h) for h in dense_hashes]
sparse_wc = [x for x in sparse_wc if x is not None]
dense_wc  = [x for x in dense_wc  if x is not None]

if sparse_wc:
    log(f"\n  OCR word count — Sparse:  mean={np.mean(sparse_wc):.0f}  "
        f"median={np.median(sparse_wc):.0f}  min={min(sparse_wc)}  max={max(sparse_wc)}")
if dense_wc:
    log(f"  OCR word count — Dense:   mean={np.mean(dense_wc):.0f}  "
        f"median={np.median(dense_wc):.0f}  min={min(dense_wc)}  max={max(dense_wc)}")

# ── 3. Example hash_names ─────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("EXAMPLE hash_names")
log("=" * 60)

sparse_examples = sorted(sparse_hashes)[:3]
dense_examples  = sorted(dense_hashes)[:3]

log("\n  Sparse cluster examples:")
for h in sparse_examples:
    wc = word_count(h)
    log(f"    {h}   words={wc}")

log("\n  Dense cluster examples:")
for h in dense_examples:
    wc = word_count(h)
    log(f"    {h}   words={wc}")

# ── 4. Sanity: are sparse pages truly low-box or just low-KVP? ───────────────
log("\n" + "=" * 60)
log("SANITY CHECK — do sparse pages have few KVPs, few OCR boxes, or both?")
log("=" * 60)

# n_kvp_boxes for sparse vs dense in eval set
# We only have feature vectors for the full dataset (not filtered to eval).
# Use cluster_stats if available, else recompute from the full feature array.
stats = s2.get("cluster_stats")
if stats is not None:
    log("\n  cluster_stats (all pages in Stage 2 dataset):")
    log(stats.to_string())
else:
    for c, name in [(dense_label, "Dense"), (sparse_label, "Sparse")]:
        mask = labs == c
        log(f"\n  {name}: n_boxes mean={feats[mask,0].mean():.1f}  "
            f"n_kvp_boxes mean={feats[mask,1].mean():.1f}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines))

print(f"\nSaved to {OUT}")