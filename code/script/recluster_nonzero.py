# Script/recluster_nonzero.py
# Re-clusters non-zero pages only (pages with at least 1 annotation box),
# generates t-SNE and UMAP for both the original (all-pages) and the new
# (non-zero-only) clustering, and saves both sets of outputs.
#
# Run from: /home/woody/iwi5/iwi5413h/kvp10k_thesis/
#   env/kvp10k_env/bin/python code/script/recluster_nonzero.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.utils import resample
import umap

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PKL   = "data/outputs/stage2/cluster_assignments.pkl"
OUT_ORIG_PKL   = "data/outputs/stage2/cluster_assignments_all.pkl"
OUT_NZ_PKL     = "data/outputs/stage2/cluster_assignments_nonzero.pkl"
FIGURES_DIR    = "LaTeX_Thesis/figures/stage2"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs("data/outputs/stage2", exist_ok=True)

# Features with heavy tails → log1p before scaling
SKEWED = ["n_boxes", "total_area", "mean_area", "std_area",
          "mean_width", "mean_height", "density", "mean_spacing"]

# ── 1. Load original clustering ────────────────────────────────────────────────
print("=" * 70)
print("Loading original clustering (all pages) …")
with open(FEATURES_PKL, "rb") as f:
    stage2 = pickle.load(f)

X_all = np.asarray(stage2["layout_features"], dtype=float)
y_all = np.asarray(stage2["cluster_labels"])
try:
    cols = list(stage2["df_features"].columns)
except Exception:
    cols = [f"f{i}" for i in range(X_all.shape[1])]

n_zero = int((np.abs(X_all).sum(axis=1) == 0).sum())
print(f"  Total pages : {len(X_all)}")
print(f"  All-zero    : {n_zero} ({100*n_zero/len(X_all):.1f}%)")
print(f"  Cluster 0 (Dense) : {(y_all==0).sum()}")
print(f"  Cluster 1 (Sparse/empty) : {(y_all==1).sum()}")

# Save the original (unchanged) as a clearly-named copy
import copy
orig_save = copy.copy(stage2)
orig_save["source"] = "all_pages"
with open(OUT_ORIG_PKL, "wb") as f:
    pickle.dump(orig_save, f, protocol=4)
print(f"  Saved: {OUT_ORIG_PKL}")


# ── 2. Re-cluster non-zero pages ───────────────────────────────────────────────
print()
print("=" * 70)
print("Re-clustering non-zero pages only …")

is_zero   = np.abs(X_all).sum(axis=1) == 0
nz_mask   = ~is_zero
X_nz_raw  = X_all[nz_mask]
orig_idx  = np.where(nz_mask)[0]   # original row indices, kept for reference

print(f"  Non-zero pages : {len(X_nz_raw)}")

# log1p on heavy-tailed features
X_t = X_nz_raw.copy()
for feat in SKEWED:
    if feat in cols:
        j = cols.index(feat)
        X_t[:, j] = np.log1p(np.clip(X_t[:, j], 0, None))

scaler   = RobustScaler()
X_nz_sc  = scaler.fit_transform(X_t)

# K-Means k=2 (same k as original) with same hyper-params
kmeans_nz = KMeans(n_clusters=2, random_state=42, n_init=10)
y_nz      = kmeans_nz.fit_predict(X_nz_sc)

# Make sure label 0 = dense (higher n_boxes mean → higher cluster center on col 0)
c0_nb = X_nz_raw[y_nz == 0, cols.index("n_boxes")].mean()
c1_nb = X_nz_raw[y_nz == 1, cols.index("n_boxes")].mean()
if c0_nb < c1_nb:          # flip so 0 = dense
    y_nz = 1 - y_nz

print(f"  Cluster 0 (Dense)  : {(y_nz==0).sum()}")
print(f"  Cluster 1 (Sparse) : {(y_nz==1).sum()}")
print(f"  Dense n_boxes mean : {X_nz_raw[y_nz==0, cols.index('n_boxes')].mean():.1f}")
print(f"  Sparse n_boxes mean: {X_nz_raw[y_nz==1, cols.index('n_boxes')].mean():.1f}")

# Save non-zero re-clustering result
nz_save = {
    "source"        : "nonzero_pages_only",
    "layout_features": X_nz_raw,
    "cluster_labels" : y_nz,
    "original_indices": orig_idx,
    "feature_names"  : cols,
    "kmeans_model"   : kmeans_nz,
    "scaler"         : scaler,
    "optimal_k"      : 2,
}
with open(OUT_NZ_PKL, "wb") as f:
    pickle.dump(nz_save, f, protocol=4)
print(f"  Saved: {OUT_NZ_PKL}")


# ── 3. Shared helpers ──────────────────────────────────────────────────────────
COLORS = {0: "#003366", 1: "#E87722"}

def prep_features(X_raw, label):
    """log1p + RobustScaler + tiny jitter (so duplicate zeros don't halo)."""
    X_t = X_raw.copy()
    for feat in SKEWED:
        if feat in cols:
            j = cols.index(feat)
            X_t[:, j] = np.log1p(np.clip(X_t[:, j], 0, None))
    X_sc = RobustScaler().fit_transform(X_t)
    rng  = np.random.default_rng(42)
    X_sc = X_sc + rng.normal(0.0, 0.01, size=X_sc.shape)
    return X_sc

def stratified_subsample(X_sc, y, n_total=3000, seed=42):
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n0   = min(len(idx0), int(n_total * len(idx0) / len(y)))
    n1   = min(len(idx1), n_total - n0)
    idx  = np.concatenate([
        resample(idx0, n_samples=n0, random_state=seed, replace=False),
        resample(idx1, n_samples=n1, random_state=seed, replace=False),
    ])
    return X_sc[idx], y[idx], n0, n1

def scatter_ax(ax, coords, labels, title, lab0, lab1, draw_order=(1, 0)):
    LBLS = {0: lab0, 1: lab1}
    for c in draw_order:
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=COLORS[c], label=LBLS[c],
                   alpha=0.3, s=5, linewidths=0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Dimension 1", fontsize=9)
    ax.set_ylabel("Dimension 2", fontsize=9)
    ax.legend(markerscale=4, fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)


# ── 4. Build scaled versions ───────────────────────────────────────────────────
print()
print("=" * 70)
print("Preparing scaled features …")
X_all_sc = prep_features(X_all, "all")
X_nz_sc2 = prep_features(X_nz_raw, "nz")    # fresh scaler (no jitter effect difference)


# ── 5. t-SNE (side-by-side: original | non-zero only) ─────────────────────────
print()
print("Running t-SNE …")

# Original (all pages) subsample
X_sub_all, y_sub_all, n0a, n1a = stratified_subsample(X_all_sc, y_all)
print(f"  All-pages subsample : {n0a} Dense + {n1a} Sparse/empty = {len(y_sub_all)}")

# Non-zero subsample (use all if small enough, else 3000)
if len(X_nz_sc2) <= 3000:
    X_sub_nz, y_sub_nz = X_nz_sc2, y_nz
    print(f"  Non-zero subsample  : all {len(y_sub_nz)} (small enough)")
else:
    X_sub_nz, y_sub_nz, n0b, n1b = stratified_subsample(X_nz_sc2, y_nz)
    print(f"  Non-zero subsample  : {n0b} Dense + {n1b} Sparse = {len(y_sub_nz)}")

tsne = TSNE(n_components=2, perplexity=30, max_iter=1500,
            init="pca", random_state=42, n_jobs=-1)

print("  Fitting t-SNE on all-pages subsample …")
X_tsne_all = tsne.fit_transform(X_sub_all)
print("  Fitting t-SNE on non-zero subsample …")
X_tsne_nz  = tsne.fit_transform(X_sub_nz)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
scatter_ax(axes[0], X_tsne_all, y_sub_all,
           f"t-SNE · All Pages (n={len(y_sub_all):,})",
           "Cluster 0 (Dense)", "Cluster 1 (Sparse/empty)")
scatter_ax(axes[1], X_tsne_nz,  y_sub_nz,
           f"t-SNE · Non-zero Pages only (n={len(y_sub_nz):,})",
           "Cluster 0 (Dense)", "Cluster 1 (Sparse)")
plt.tight_layout()
out = f"{FIGURES_DIR}/tsne_clusters_comparison.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# Also save individual non-zero t-SNE
fig, ax = plt.subplots(figsize=(7, 5))
scatter_ax(ax, X_tsne_nz, y_sub_nz,
           f"t-SNE · Non-zero Pages only (n={len(y_sub_nz):,})",
           "Cluster 0 (Dense)", "Cluster 1 (Sparse)")
plt.tight_layout()
out = f"{FIGURES_DIR}/tsne_clusters_nonzero.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")


# ── 6. UMAP (side-by-side) ─────────────────────────────────────────────────────
print()
print("Running UMAP …")

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric="euclidean", random_state=42)

print("  Fitting UMAP on all pages …")
X_umap_all = reducer.fit_transform(X_all_sc)
print("  Fitting UMAP on non-zero pages …")
X_umap_nz  = reducer.fit_transform(X_nz_sc2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
scatter_ax(axes[0], X_umap_all, y_all,
           f"UMAP · All Pages (n={len(y_all):,})",
           "Cluster 0 (Dense)", "Cluster 1 (Sparse/empty)")
scatter_ax(axes[1], X_umap_nz,  y_nz,
           f"UMAP · Non-zero Pages only (n={len(y_nz):,})",
           "Cluster 0 (Dense)", "Cluster 1 (Sparse)")
plt.tight_layout()
out = f"{FIGURES_DIR}/umap_clusters_comparison.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# Also save individual non-zero UMAP
fig, ax = plt.subplots(figsize=(7, 5))
scatter_ax(ax, X_umap_nz, y_nz,
           f"UMAP · Non-zero Pages only (n={len(y_nz):,})",
           "Cluster 0 (Dense)", "Cluster 1 (Sparse)")
plt.tight_layout()
out = f"{FIGURES_DIR}/umap_clusters_nonzero.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")


# ── 7. Summary ─────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("DONE — outputs saved:")
print(f"  Pickles:")
print(f"    {OUT_ORIG_PKL}")
print(f"    {OUT_NZ_PKL}")
print(f"  Figures:")
print(f"    {FIGURES_DIR}/tsne_clusters_comparison.png  (side-by-side)")
print(f"    {FIGURES_DIR}/tsne_clusters_nonzero.png     (non-zero only)")
print(f"    {FIGURES_DIR}/umap_clusters_comparison.png  (side-by-side)")
print(f"    {FIGURES_DIR}/umap_clusters_nonzero.png     (non-zero only)")
print("=" * 70)
