"""Rebuild Stage 2 clustering on unique coordinate-bearing training pages."""

import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from features import extract_layout_features


OUTPUT_DIR = Path("data/outputs/stage2")
FIGURE_DIR = Path("LaTeX_Thesis/figures/stage2")
ARTIFACT_PATH = OUTPUT_DIR / "cluster_assignments.pkl"
PAGE_MAP_PATH = OUTPUT_DIR / "train_cluster_map.json"
SUMMARY_PATH = OUTPUT_DIR / "unique_page_clustering_summary.json"

FEATURE_NAMES_13 = list(config.LAYOUT_FEATURE_NAMES)
FEATURE_NAMES = [name for name in FEATURE_NAMES_13 if name != "density"]
FEATURE_INDICES = [i for i, name in enumerate(FEATURE_NAMES_13) if name != "density"]
RANDOM_SEED = 42
K_VALUES = list(range(2, 11))


def select_unique_coordinate_pages(dataset):
    groups = defaultdict(list)
    for index in range(len(dataset)):
        groups[dataset[index]["hash_name"]].append(index)

    page_hashes = []
    selected_indices = []
    features = []
    unavailable = []
    for hash_name, indices in groups.items():
        best_index = None
        best_features = None
        best_box_count = -1.0
        for index in indices:
            row_features = extract_layout_features(dataset[index])
            if row_features[0] > best_box_count:
                best_index = index
                best_features = row_features
                best_box_count = float(row_features[0])
        if best_box_count <= 0:
            unavailable.append(hash_name)
            continue
        page_hashes.append(hash_name)
        selected_indices.append(best_index)
        features.append(best_features[FEATURE_INDICES])

    return (
        page_hashes,
        selected_indices,
        np.asarray(features, dtype=float),
        unavailable,
        len(groups),
    )


def select_k(features_scaled):
    results = {
        "k_values": [],
        "inertias": [],
        "silhouette_scores": [],
        "davies_bouldin_scores": [],
        "calinski_harabasz_scores": [],
        "all_models": {},
    }
    for k in K_VALUES:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=10,
        )
        labels = model.fit_predict(features_scaled)
        results["k_values"].append(k)
        results["inertias"].append(float(model.inertia_))
        results["silhouette_scores"].append(
            float(silhouette_score(features_scaled, labels))
        )
        results["davies_bouldin_scores"].append(
            float(davies_bouldin_score(features_scaled, labels))
        )
        results["calinski_harabasz_scores"].append(
            float(calinski_harabasz_score(features_scaled, labels))
        )
        results["all_models"][k] = model
        print(
            f"k={k}: silhouette={results['silhouette_scores'][-1]:.4f}, "
            f"DB={results['davies_bouldin_scores'][-1]:.4f}, "
            f"CH={results['calinski_harabasz_scores'][-1]:.2f}"
        )

    best_index = int(np.argmax(results["silhouette_scores"]))
    results["recommended_k"] = results["k_values"][best_index]
    return results


def stable_cluster_order(labels, centers_unscaled):
    n_boxes_index = FEATURE_NAMES.index("n_boxes")
    order = np.argsort(centers_unscaled[:, n_boxes_index])[::-1]
    remap = {int(old): int(new) for new, old in enumerate(order)}
    remapped_labels = np.asarray([remap[int(label)] for label in labels])
    return remapped_labels, order, remap


def plot_optimal_k(results):
    k_values = results["k_values"]
    recommended_k = results["recommended_k"]
    series = [
        ("inertias", "Inertia", "Elbow method"),
        ("silhouette_scores", "Silhouette score", "Silhouette"),
        ("davies_bouldin_scores", "Davies-Bouldin index", "Davies-Bouldin"),
        ("calinski_harabasz_scores", "Calinski-Harabasz score", "Calinski-Harabasz"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, (key, ylabel, title) in zip(axes.flat, series):
        axis.plot(k_values, results[key], marker="o", linewidth=2)
        axis.axvline(recommended_k, color="#c33", linestyle="--")
        axis.set(xlabel="Number of clusters (k)", ylabel=ylabel, title=title)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "optimal_k_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(unique, counts, color="#315f72")
    axis.set(xlabel="Cluster", ylabel="Unique training pages", title="Unique-page cluster distribution")
    axis.set_xticks(unique)
    for label, count in zip(unique, counts):
        axis.text(label, count, f"{count:,}\n({100 * count / len(labels):.1f}%)", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cluster_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pca(pca_coordinates, labels, variance_ratio):
    fig, axis = plt.subplots(figsize=(10, 8))
    scatter = axis.scatter(
        pca_coordinates[:, 0],
        pca_coordinates[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.45,
        s=12,
    )
    axis.set(
        xlabel=f"PC1 ({100 * variance_ratio[0]:.1f}% variance)",
        ylabel=f"PC2 ({100 * variance_ratio[1]:.1f}% variance)",
        title="Unique-page geometry clusters",
    )
    axis.grid(alpha=0.2)
    fig.colorbar(scatter, ax=axis, label="Cluster")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pca_clusters.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tsne(features_scaled, labels):
    rng = np.random.default_rng(RANDOM_SEED)
    if len(labels) > 3000:
        indices = np.sort(rng.choice(len(labels), size=3000, replace=False))
    else:
        indices = np.arange(len(labels))
    coordinates = TSNE(
        n_components=2,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
    ).fit_transform(features_scaled[indices])
    fig, axis = plt.subplots(figsize=(10, 8))
    scatter = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels[indices],
        cmap="tab10",
        alpha=0.5,
        s=12,
    )
    axis.set(title=f"t-SNE projection of {len(indices):,} unique training pages")
    axis.set_xticks([])
    axis.set_yticks([])
    fig.colorbar(scatter, ax=axis, label="Cluster")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "tsne_clusters.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distributions(features_frame, labels):
    frame = features_frame.copy()
    frame["cluster"] = labels
    selected = [
        "n_boxes",
        "total_area",
        "mean_area",
        "mean_aspect_ratio",
        "vertical_spread",
        "mean_spacing",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, feature in zip(axes.flat, selected):
        sns.boxplot(data=frame, x="cluster", y=feature, ax=axis, showfliers=False)
        axis.set_title(feature.replace("_", " "))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def json_ready_k_results(results):
    return {
        key: value
        for key, value in results.items()
        if key != "all_models"
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        config.DATASET_NAME,
        split="train",
        cache_dir=config.KVP_CACHE,
    )
    page_hashes, selected_indices, features, unavailable, total_pages = (
        select_unique_coordinate_pages(dataset)
    )
    print(
        f"Selected {len(page_hashes):,} coordinate-bearing pages from "
        f"{total_pages:,} unique training pages; {len(unavailable):,} unavailable"
    )
    if len(page_hashes) != 9124:
        raise RuntimeError(f"Expected 9,124 coordinate-bearing pages, found {len(page_hashes):,}")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    k_results = select_k(features_scaled)
    optimal_k = k_results["recommended_k"]
    selected_model = k_results["all_models"][optimal_k]
    raw_labels = selected_model.labels_
    centers_unscaled = scaler.inverse_transform(selected_model.cluster_centers_)
    labels, order, remap = stable_cluster_order(raw_labels, centers_unscaled)
    ordered_centers_scaled = selected_model.cluster_centers_[order]
    ordered_model = KMeans(
        n_clusters=optimal_k,
        random_state=RANDOM_SEED,
        n_init=10,
    )
    ordered_model.cluster_centers_ = ordered_centers_scaled
    ordered_model.n_features_in_ = features_scaled.shape[1]
    ordered_model._n_threads = selected_model._n_threads

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca_coordinates = pca.fit_transform(features_scaled)
    features_frame = pd.DataFrame(features, columns=FEATURE_NAMES)
    cluster_stats = features_frame.assign(cluster=labels).groupby("cluster").agg(["mean", "std", "median"])

    plot_optimal_k(k_results)
    plot_cluster_distribution(labels)
    plot_pca(pca_coordinates, labels, pca.explained_variance_ratio_)
    plot_tsne(features_scaled, labels)
    plot_feature_distributions(features_frame, labels)

    clustering_result = {
        "labels": labels,
        "scaler": scaler,
        "kmeans": ordered_model,
        "features_scaled": features_scaled,
    }
    artifact = {
        "schema_version": 2,
        "unit": "unique_coordinate_bearing_training_page",
        "page_hashes": page_hashes,
        "selected_row_indices": selected_indices,
        "unavailable_page_hashes": unavailable,
        "feature_names": FEATURE_NAMES,
        "removed_duplicate_feature": "density",
        "cluster_labels": labels,
        "optimal_k": optimal_k,
        "optimal_k_results": {
            **json_ready_k_results(k_results),
            "scaler": scaler,
            "features_scaled": features_scaled,
        },
        "layout_features": features,
        "df_features": features_frame,
        "clustering_result": clustering_result,
        "pca_result": {
            "pca": pca,
            "transformed": pca_coordinates,
            "variance_ratio": pca.explained_variance_ratio_,
        },
        "cluster_stats": cluster_stats,
        "cluster_ordering": "descending mean n_boxes",
        "original_label_remap": remap,
    }
    with ARTIFACT_PATH.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=4)

    train_map = {
        hash_name: {
            "cluster": f"Cluster_{int(label)}",
            "n_boxes": int(round(features[index, FEATURE_NAMES.index('n_boxes')])),
        }
        for index, (hash_name, label) in enumerate(zip(page_hashes, labels))
    }
    with PAGE_MAP_PATH.open("w", encoding="utf-8") as handle:
        json.dump(train_map, handle, indent=2)

    counts = {str(cluster): int((labels == cluster).sum()) for cluster in np.unique(labels)}
    summary = {
        "raw_training_rows": len(dataset),
        "unique_training_pages": total_pages,
        "coordinate_bearing_training_pages": len(page_hashes),
        "geometry_unavailable_training_pages": len(unavailable),
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "optimal_k": optimal_k,
        "cluster_counts": counts,
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "k_selection": json_ready_k_results(k_results),
        "cluster_centers_unscaled": scaler.inverse_transform(ordered_centers_scaled).tolist(),
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()