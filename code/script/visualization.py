"""
Visualization utilities for KVP10k experiments.
Handles plotting of distributions, clusters, and sample documents.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import config


def plot_cluster_distribution(cluster_labels, save_path=None):
    """
    Plot distribution of samples across clusters.
    
    Args:
        cluster_labels: numpy array of cluster assignments
        save_path: Path to save figure (optional)
    """
    unique, counts = np.unique(cluster_labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(unique, counts, color='steelblue', alpha=0.7)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Distribution of Samples Across Clusters')
    ax.set_xticks(unique)
    
    # Add percentage labels
    total = len(cluster_labels)
    for i, (cluster_id, count) in enumerate(zip(unique, counts)):
        pct = 100 * count / total
        ax.text(cluster_id, count, f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"Saved cluster distribution to {save_path}")
    
    plt.close()
    return fig


def plot_pca_clusters(pca_features, cluster_labels, variance_ratio=None, save_path=None):
    """
    Plot PCA visualization of clusters.
    
    Args:
        pca_features: numpy array of shape (n_samples, 2) with PCA components
        cluster_labels: numpy array of cluster assignments
        variance_ratio: Explained variance ratio for each component
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot with cluster colors
    scatter = ax.scatter(
        pca_features[:, 0],
        pca_features[:, 1],
        c=cluster_labels,
        cmap=config.COLORMAP,
        alpha=0.6,
        s=50
    )
    
    # Add labels
    if variance_ratio is not None:
        ax.set_xlabel(f'PC1 ({100*variance_ratio[0]:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({100*variance_ratio[1]:.1f}% variance)')
    else:
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
    
    ax.set_title('Layout Clustering (PCA Visualization)')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster ID')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"Saved PCA plot to {save_path}")
    
    plt.close()
    return fig


def plot_feature_distributions(features_df, cluster_labels, save_path=None):
    """
    Plot distributions of layout features by cluster.
    
    Args:
        features_df: pandas DataFrame with layout features
        cluster_labels: numpy array of cluster assignments
        save_path: Path to save figure (optional)
    """
    # Add cluster column
    df = features_df.copy()
    df['cluster'] = cluster_labels
    
    # Select key features to plot
    key_features = ['n_boxes', 'mean_area', 'mean_width', 'mean_height', 
                   'density', 'vertical_spread']
    
    n_features = len(key_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i, feat in enumerate(key_features):
        if feat in df.columns:
            # Box plot by cluster
            df.boxplot(column=feat, by='cluster', ax=axes[i])
            axes[i].set_title(feat)
            axes[i].set_xlabel('Cluster ID')
            axes[i].set_ylabel('Value')
            plt.sca(axes[i])
            plt.xticks(rotation=0)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Feature Distributions by Cluster', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"Saved feature distributions to {save_path}")
    
    plt.close()
    return fig


def plot_kv_distance_distribution(distances, save_path=None):
    """
    Plot distribution of key-value linking distances.
    
    Args:
        distances: List of distances
        save_path: Path to save figure (optional)
    """
    if not distances:
        print("No distances to plot")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(distances, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Distance (normalized coordinates)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'KV Link Distance Distribution (n={len(distances)})')
    ax1.axvline(np.mean(distances), color='red', linestyle='--', 
                label=f'Mean: {np.mean(distances):.4f}')
    ax1.axvline(np.median(distances), color='green', linestyle='--',
                label=f'Median: {np.median(distances):.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(distances, vert=True)
    ax2.set_ylabel('Distance (normalized coordinates)')
    ax2.set_title('KV Link Distance Box Plot')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"Saved KV distance plot to {save_path}")
    
    plt.close()
    return fig


def plot_cluster_statistics_table(features_df, cluster_labels, save_path=None):
    """
    Create a table showing cluster statistics.
    
    Args:
        features_df: pandas DataFrame with layout features
        cluster_labels: numpy array of cluster assignments
        
    Args:
        save_path: Path to save figure (optional)

    Returns:
        pandas DataFrame with cluster statistics
    """
    df = features_df.copy()
    df['cluster'] = cluster_labels
    
    # Compute statistics by cluster
    cluster_stats = df.groupby('cluster').agg({
        'n_boxes': ['mean', 'std'],
        'total_area': ['mean', 'std'],
        'mean_area': ['mean', 'std'],
        'density': ['mean', 'std']
    }).round(4)
    
    print("\nCluster Statistics:")
    print(cluster_stats)

    if save_path:
        # Render a simple matplotlib table for headless saving
        fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * len(cluster_stats.index)))
        ax.axis('off')
        ax.set_title('Cluster Statistics (mean ± std)', pad=12)

        table = ax.table(
            cellText=cluster_stats.values,
            rowLabels=[str(i) for i in cluster_stats.index],
            colLabels=[" ".join(col).strip() for col in cluster_stats.columns],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"Saved cluster statistics table to {save_path}")
        plt.close(fig)
    
    return cluster_stats


def display_sample_image(image_path):
    """
    Display a sample document image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image object
    """
    img = Image.open(image_path)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Sample Document\nSize: {img.size}')
    plt.tight_layout()
    plt.close()
    
    return img


def plot_feature_importance(feature_names, importances, save_path=None):
    """
    Plot feature importance (e.g., from clustering or classification).
    
    Args:
        feature_names: List of feature names
        importances: List of importance values
        save_path: Path to save figure (optional)
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(sorted_names)), sorted_importances, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")
    
    plt.close()
    return fig


def plot_optimal_k_analysis(optimal_k_results, save_path=None):
    """
    Plot comprehensive optimal k analysis with all metrics.
    
    Args:
        optimal_k_results: Dictionary from find_optimal_k()
        save_path: Path to save figure (optional)
    """
    k_values = optimal_k_results['k_values']
    inertias = optimal_k_results['inertias']
    silhouette = optimal_k_results['silhouette_scores']
    davies_bouldin = optimal_k_results['davies_bouldin_scores']
    calinski_harabasz = optimal_k_results['calinski_harabasz_scores']
    recommended_k = optimal_k_results['recommended_k']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Elbow plot (Inertia)
    ax = axes[0, 0]
    ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax.axvline(recommended_k, color='red', linestyle='--', alpha=0.7, label=f'Recommended k={recommended_k}')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    ax.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Silhouette Score (higher is better)
    ax = axes[0, 1]
    ax.plot(k_values, silhouette, 'go-', linewidth=2, markersize=8)
    ax.axvline(recommended_k, color='red', linestyle='--', alpha=0.7, label=f'Recommended k={recommended_k}')
    best_idx = np.argmax(silhouette)
    ax.plot(k_values[best_idx], silhouette[best_idx], 'r*', markersize=20, label='Best')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Analysis (Higher is Better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Davies-Bouldin Index (lower is better)
    ax = axes[1, 0]
    ax.plot(k_values, davies_bouldin, 'mo-', linewidth=2, markersize=8)
    ax.axvline(recommended_k, color='red', linestyle='--', alpha=0.7, label=f'Recommended k={recommended_k}')
    best_idx = np.argmin(davies_bouldin)
    ax.plot(k_values[best_idx], davies_bouldin[best_idx], 'r*', markersize=20, label='Best')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Davies-Bouldin Index', fontsize=12)
    ax.set_title('Davies-Bouldin Index (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Calinski-Harabasz Score (higher is better)
    ax = axes[1, 1]
    ax.plot(k_values, calinski_harabasz, 'co-', linewidth=2, markersize=8)
    ax.axvline(recommended_k, color='red', linestyle='--', alpha=0.7, label=f'Recommended k={recommended_k}')
    best_idx = np.argmax(calinski_harabasz)
    ax.plot(k_values[best_idx], calinski_harabasz[best_idx], 'r*', markersize=20, label='Best')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Calinski-Harabasz Score', fontsize=12)
    ax.set_title('Calinski-Harabasz Score (Higher is Better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle(f'Optimal K Selection Analysis (Recommended: k={recommended_k})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"Saved optimal k analysis to {save_path}")
    
    plt.close()
    return fig


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization.py...")
    
    # Generate dummy data for testing
    np.random.seed(42)
    n_samples = 200
    
    # Dummy cluster labels
    cluster_labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.65, 0.25, 0.10])
    
    # Dummy PCA features
    pca_features = np.random.randn(n_samples, 2)
    variance_ratio = np.array([0.60, 0.27])
    
    # Plot cluster distribution
    plot_cluster_distribution(cluster_labels)
    
    # Plot PCA clusters
    plot_pca_clusters(pca_features, cluster_labels, variance_ratio)
    
    print("Visualization tests complete!")
