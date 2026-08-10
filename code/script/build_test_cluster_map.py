"""
Build the TRUE Stage 2 layout-cluster assignment for every test-split page.

Each Stage 3 test document is assigned the layout cluster (Dense / Sparse) that
the actual Stage 2 model would give it — by recomputing the identical 13 layout
features (features.extract_layout_features) on the raw KVP10k annotations and
running them through the *saved* Stage 2 StandardScaler + KMeans model.

This replaces the previous density heuristic in analyze_stage3_errors.py, which
invented its own threshold and ignored Stage 2 entirely.

Pages whose annotator copies contain no usable polygons are labelled
Geometry_Unavailable rather than projected from an out-of-distribution zero vector.

Output: data/outputs/stage2/test_cluster_map.json
    { hash_name: {"cluster": "Cluster_<n>"|"Geometry_Unavailable", "n_boxes": int} }

Run from repo root:
  HF_DATASETS_OFFLINE=1 env/kvp10k_env/bin/python code/script/build_test_cluster_map.py
"""

import os
import sys
import json
import pickle
import logging
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from features import extract_layout_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STAGE2_PKL = "data/outputs/stage2/cluster_assignments.pkl"
OUT_JSON   = "data/outputs/stage2/test_cluster_map.json"


def main(split: str = "test"):
    # ── 1. Load saved Stage 2 scaler + kmeans ────────────────────────────────
    with open(STAGE2_PKL, "rb") as f:
        s2 = pickle.load(f)
    scaler = s2["clustering_result"]["scaler"]
    kmeans = s2["clustering_result"]["kmeans"]
    feature_names = s2.get("feature_names", list(config.LAYOUT_FEATURE_NAMES))
    feature_indices = [config.LAYOUT_FEATURE_NAMES.index(name) for name in feature_names]

    # ── 2. Load raw KVP10k split and group annotator copies by hash_name ─────
    from datasets import load_dataset
    logger.info(f"Loading KVP10k '{split}' split (offline cache) …")
    ds = load_dataset(config.DATASET_NAME, split=split, cache_dir=config.KVP_CACHE)

    groups = defaultdict(list)
    for i in range(len(ds)):
        groups[ds[i]["hash_name"]].append(i)
    logger.info(f"{len(ds)} rows -> {len(groups)} unique pages")

    # ── 3. Assign each page its true Stage 2 cluster ─────────────────────────
    # For pages with multiple annotator copies, use the copy with the most
    # coordinate boxes (richest layout) to avoid spuriously labelling a page
    # empty when a sibling copy carries the geometry.
    mapping = {}
    for h, idxs in groups.items():
        best_feat, best_nb = None, -1.0
        for i in idxs:
            feat = extract_layout_features(ds[i])
            if feat[0] > best_nb:
                best_nb, best_feat = float(feat[0]), feat
        if best_nb <= 0:
            mapping[h] = {"cluster": "Geometry_Unavailable", "n_boxes": 0}
            continue
        selected_feat = best_feat[feature_indices]
        x_scaled = scaler.transform(selected_feat.reshape(1, -1))
        label = int(kmeans.predict(x_scaled)[0])
        cluster = f"Cluster_{label}"
        mapping[h] = {"cluster": cluster, "n_boxes": int(round(best_nb))}

    # ── 4. Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(mapping, f, indent=2)

    logger.info(f"Saved {len(mapping)} pages -> {OUT_JSON}")
    for cluster in sorted({item["cluster"] for item in mapping.values()}):
        count = sum(1 for item in mapping.values() if item["cluster"] == cluster)
        logger.info(f"  {cluster}: {count} ({100*count/len(mapping):.1f}%)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build true Stage 2 cluster map for a split")
    p.add_argument("--split", default="test")
    args = p.parse_args()
    main(args.split)
