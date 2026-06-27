# Script/diagnose_stage2_clusters.py
# Diagnose Stage 2 layout clusters:
# - total all-zero rows
# - all-zero rows per cluster
# - PCA explained variance
# - whether Cluster 1 looks sparse or empty-heavy

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

FEATURES_PKL = "data/outputs/stage2/cluster_assignments.pkl"

# Features that are usually heavy-tailed; log1p helps before scaling
SKEWED = [
    "n_boxes", "total_area", "mean_area", "std_area",
    "mean_width", "mean_height", "density", "mean_spacing"
]

def pct(a, b):
    return 100.0 * a / b if b else 0.0

with open(FEATURES_PKL, "rb") as f:
    stage2 = pickle.load(f)

X = np.asarray(stage2["layout_features"], dtype=float)
y = np.asarray(stage2["cluster_labels"])

# Try to recover feature names and metadata if present
df_features = stage2.get("df_features", None)
if df_features is not None and hasattr(df_features, "columns"):
    cols = list(df_features.columns)
else:
    cols = [f"f{i}" for i in range(X.shape[1])]

print("=" * 80)
print("STAGE 2 DIAGNOSTIC")
print("=" * 80)
print(f"Loaded feature matrix: X.shape = {X.shape}")
print(f"Cluster labels: {np.unique(y)}")
print()

# ------------------------------------------------------------------
# 1) Global all-zero analysis
# ------------------------------------------------------------------
row_abs_sum = np.abs(X).sum(axis=1)
is_zero = row_abs_sum == 0
n_total = len(X)
n_zero = int(is_zero.sum())

print("[1] ALL-ZERO ROWS")
print(f"Total rows:           {n_total}")
print(f"All-zero rows:        {n_zero} ({pct(n_zero, n_total):.1f}%)")
print(f"Non-zero rows:        {n_total - n_zero} ({pct(n_total - n_zero, n_total):.1f}%)")
print()

# ------------------------------------------------------------------
# 2) Zero rows per cluster
# ------------------------------------------------------------------
print("[2] ALL-ZERO ROWS PER CLUSTER")
cluster_stats = []
for c in sorted(np.unique(y)):
    mask = (y == c)
    n_c = int(mask.sum())
    z_c = int(is_zero[mask].sum())
    nz_c = n_c - z_c
    cluster_stats.append((c, n_c, z_c, nz_c))
    print(
        f"Cluster {c}: total={n_c:6d} | zero={z_c:6d} ({pct(z_c, n_c):5.1f}%) "
        f"| non-zero={nz_c:6d} ({pct(nz_c, n_c):5.1f}%)"
    )
print()

# ------------------------------------------------------------------
# 3) Feature-wise summary to see what differentiates clusters
# ------------------------------------------------------------------
print("[3] FEATURE SUMMARY BY CLUSTER")
df = pd.DataFrame(X, columns=cols)
df["cluster"] = y
df["is_zero"] = is_zero.astype(int)

# Show mean per cluster for all features
means = df.groupby("cluster")[cols].mean().T
means.columns = [f"cluster_{c}_mean" for c in means.columns]

# Also compute medians, useful when data is skewed
medians = df.groupby("cluster")[cols].median().T
medians.columns = [f"cluster_{c}_median" for c in medians.columns]

summary = pd.concat([means, medians], axis=1)

with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 160):
    print(summary.round(4))
print()

# ------------------------------------------------------------------
# 4) Non-zero-only cluster check
#    If cluster 1 is still lower-density / lower-box-count among non-zero pages,
#    then it is genuinely sparse, not just empty-heavy.
# ------------------------------------------------------------------
print("[4] NON-ZERO-ONLY CHECK")
nonzero_mask = ~is_zero
X_nz = X[nonzero_mask]
y_nz = y[nonzero_mask]
df_nz = pd.DataFrame(X_nz, columns=cols)
df_nz["cluster"] = y_nz

print(f"Non-zero rows available: {len(df_nz)}")

if len(df_nz) > 0:
    nz_counts = df_nz["cluster"].value_counts().sort_index()
    for c in sorted(np.unique(y)):
        n_c_nz = int(nz_counts.get(c, 0))
        print(f"Cluster {c}: non-zero rows = {n_c_nz}")

    # Print a few likely "layout complexity" features if they exist
    key_feats = [f for f in ["n_boxes", "density", "total_area", "mean_spacing", "vertical_spread", "horizontal_spread"] if f in cols]
    if key_feats:
        print("\nMeans on NON-ZERO rows only:")
        print(df_nz.groupby("cluster")[key_feats].mean().round(4))
        print("\nMedians on NON-ZERO rows only:")
        print(df_nz.groupby("cluster")[key_feats].median().round(4))
print()

# ------------------------------------------------------------------
# 5) PCA diagnostics (NO jitter)
#    Use log1p + RobustScaler, but no noise injection.
# ------------------------------------------------------------------
print("[5] PCA EXPLAINED VARIANCE")

X_t = X.copy()
for feat in SKEWED:
    if feat in cols:
        j = cols.index(feat)
        X_t[:, j] = np.log1p(np.clip(X_t[:, j], 0, None))

X_scaled = RobustScaler().fit_transform(X_t)

pca = PCA(n_components=min(X_scaled.shape[1], 10), random_state=42)
X_pca = pca.fit_transform(X_scaled)

evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)

for i, (v, c) in enumerate(zip(evr, cum), start=1):
    print(f"PC{i:>2}: explained variance = {v:.4f} | cumulative = {c:.4f}")
print()

# Cluster centers in PCA space (first 2 PCs)
print("Cluster means in PCA space (PC1, PC2):")
for c in sorted(np.unique(y)):
    mask = (y == c)
    pc1_mean = X_pca[mask, 0].mean()
    pc2_mean = X_pca[mask, 1].mean()
    print(f"Cluster {c}: PC1 mean = {pc1_mean:.4f}, PC2 mean = {pc2_mean:.4f}")
print()

# ------------------------------------------------------------------
# 6) Simple rule-based interpretation
# ------------------------------------------------------------------
print("[6] RULE-BASED INTERPRETATION")

# empty-heavy heuristic
cluster_zero_pct = {c: pct(z, n) for c, n, z, _ in cluster_stats}
c1_zero_pct = cluster_zero_pct.get(1, None)

if c1_zero_pct is None:
    print("Cluster 1 not found; cannot assess 'sparse vs empty-heavy'.")
else:
    if c1_zero_pct >= 80:
        print(f"Cluster 1 is VERY LIKELY empty-heavy: {c1_zero_pct:.1f}% of its rows are all-zero.")
    elif c1_zero_pct >= 50:
        print(f"Cluster 1 is MIXED but strongly empty-heavy: {c1_zero_pct:.1f}% all-zero.")
    else:
        print(f"Cluster 1 is NOT primarily empty-heavy: only {c1_zero_pct:.1f}% all-zero.")

    # optional non-zero comparison if layout features exist
    if len(df_nz) > 0 and all(f in cols for f in ["n_boxes", "density"]):
        means_nz = df_nz.groupby("cluster")[["n_boxes", "density"]].mean()
        if 0 in means_nz.index and 1 in means_nz.index:
            nb0, nb1 = means_nz.loc[0, "n_boxes"], means_nz.loc[1, "n_boxes"]
            de0, de1 = means_nz.loc[0, "density"], means_nz.loc[1, "density"]

            print(f"Among NON-ZERO rows: cluster_0 mean n_boxes={nb0:.2f}, cluster_1 mean n_boxes={nb1:.2f}")
            print(f"Among NON-ZERO rows: cluster_0 mean density={de0:.4f}, cluster_1 mean density={de1:.4f}")

            if nb1 < nb0 and de1 < de0:
                print("This suggests Cluster 1 still behaves like a genuinely sparser layout class among non-empty pages.")
            else:
                print("This suggests the 'sparse' interpretation may be weak once empty pages are removed.")

print()
print("=" * 80)
print("Done.")
print("=" * 80)