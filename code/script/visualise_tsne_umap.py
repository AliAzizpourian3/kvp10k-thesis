# Script/visualise_tsne_umap.py
# Generates t-SNE and UMAP plots of Stage 2 layout features
# Run from: /home/woody/iwi5/iwi5413h/kvp10k_thesis/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.utils import resample
import umap
import os
import pickle

# ── 1. Load features ─────────────────────────────────────────────────────────
FEATURES_PKL = "data/outputs/stage2/cluster_assignments.pkl"
FIGURES_DIR  = "LaTeX_Thesis/figures/stage2"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Heavy-tailed count / area features → compress with log1p before scaling.
SKEWED = ["n_boxes", "total_area", "mean_area", "std_area",
          "mean_width", "mean_height", "density", "mean_spacing"]

with open(FEATURES_PKL, "rb") as f:
    stage2 = pickle.load(f)

X = np.asarray(stage2["layout_features"], dtype=float)
y = np.asarray(stage2["cluster_labels"])  # 0 = dense, 1 = sparse
try:
    cols = list(stage2["df_features"].columns)
except Exception:
    cols = [f"f{i}" for i in range(X.shape[1])]

print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
print(f"  Cluster 0 (Dense):  {(y==0).sum()} samples")
print(f"  Cluster 1 (Sparse): {(y==1).sum()} samples")
n_zero = int((np.abs(X).sum(axis=1) == 0).sum())
print(f"  All-zero (empty) pages: {n_zero} ({100*n_zero/len(X):.1f}%)")

# log1p on skewed columns to tame outliers
X_t = X.copy()
for name in SKEWED:
    if name in cols:
        j = cols.index(name)
        X_t[:, j] = np.log1p(np.clip(X_t[:, j], 0, None))

X_scaled = RobustScaler().fit_transform(X_t)

# Jitter so the ~16.5k identical empty-page vectors form ONE compact cluster
# instead of being scattered into a random halo by the manifold solvers.
rng = np.random.default_rng(42)
X_scaled = X_scaled + rng.normal(0.0, 0.01, size=X_scaled.shape)

COLORS = {0: "#003366", 1: "#E87722"}
LABELS = {0: "Cluster 0 (Dense)", 1: "Cluster 1 (Sparse / empty)"}

def scatter(ax, coords, y_labels, title, draw_order=(1, 0)):
    # Draw sparse first so dense (smaller cluster) is always visible on top
    for c in draw_order:
        mask = y_labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=COLORS[c], label=LABELS[c],
                   alpha=0.3, s=5, linewidths=0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Dimension 1", fontsize=10)
    ax.set_ylabel("Dimension 2", fontsize=10)
    ax.legend(markerscale=4, fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)

# ── 2. t-SNE (stratified subsample for readability) ──────────────────────────
print("\nPreparing stratified subsample for t-SNE...")
idx0 = np.where(y == 0)[0]
idx1 = np.where(y == 1)[0]
n_total = 3000
n0 = min(len(idx0), int(n_total * len(idx0) / len(y)))
n1 = min(len(idx1), n_total - n0)
idx_sub = np.concatenate([
    resample(idx0, n_samples=n0, random_state=42, replace=False),
    resample(idx1, n_samples=n1, random_state=42, replace=False)
])
X_sub = X_scaled[idx_sub]
y_sub = y[idx_sub]
print(f"  Subsample: {n0} Dense + {n1} Sparse = {len(idx_sub)} total")
print("Running t-SNE (perplexity=30, max_iter=1500)...")
tsne = TSNE(n_components=2, perplexity=30, max_iter=1500,
           init="pca", random_state=42, n_jobs=-1)
X_tsne = tsne.fit_transform(X_sub)
fig, ax = plt.subplots(figsize=(7, 5))
scatter(ax, X_tsne, y_sub, f"t-SNE Projection of Layout Clusters (n={len(idx_sub):,})")
plt.tight_layout()
out = f"{FIGURES_DIR}/tsne_clusters.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ── 3. UMAP (full dataset) ────────────────────────────────────────────────────
print("\nRunning UMAP (n_neighbors=15, min_dist=0.1) on full dataset...")
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric="euclidean", random_state=42)
X_umap = reducer.fit_transform(X_scaled)
fig, ax = plt.subplots(figsize=(7, 5))
scatter(ax, X_umap, y, f"UMAP Projection of Layout Clusters (n={len(y):,})")
plt.tight_layout()
out = f"{FIGURES_DIR}/umap_clusters.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

print("\nDone. Figures saved to LaTeX_Thesis/figures/stage2/")
print("  tsne_clusters.png  — t-SNE on stratified 3k subsample")
print("  umap_clusters.png  — UMAP on full dataset")