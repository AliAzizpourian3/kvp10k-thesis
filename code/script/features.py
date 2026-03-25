"""
Feature extraction and layout analysis for KVP10k documents.
Handles layout feature computation and clustering.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tqdm import tqdm
import config


def extract_layout_features(example):
    """
    Extract layout features from a single document page.
    
    Features extracted (13 total):
    - Number of bounding boxes
    - Total area covered
    - Mean/std box area
    - Mean width/height/aspect ratio
    - Mean x/y position
    - Layout density
    - Vertical/horizontal spread
    - Mean inter-box spacing
    
    Args:
        example: Dictionary with 'annotations' field
        
    Returns:
        numpy array of shape (13,) with layout features
    """
    annotations = example.get('annotations', [])
    
    if not annotations:
        # Return zeros for pages with no annotations
        return np.zeros(13)
    
    # Extract bounding boxes from coordinates
    boxes = []
    for ann in annotations:
        coords = ann.get('coordinates', [])
        if coords:
            # coords is list of {x, y} dicts
            xs = [c['x'] for c in coords]
            ys = [c['y'] for c in coords]
            
            if xs and ys:
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                width = x_max - x_min
                height = y_max - y_min
                area = width * height
                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                
                boxes.append({
                    'x_min': x_min, 'x_max': x_max,
                    'y_min': y_min, 'y_max': y_max,
                    'width': width, 'height': height,
                    'area': area, 'cx': cx, 'cy': cy
                })
    
    if not boxes:
        return np.zeros(13)
    
    # Compute features
    n_boxes = len(boxes)
    areas = [b['area'] for b in boxes]
    widths = [b['width'] for b in boxes]
    heights = [b['height'] for b in boxes]
    aspect_ratios = [b['width'] / (b['height'] + 1e-6) for b in boxes]
    xs = [b['cx'] for b in boxes]
    ys = [b['cy'] for b in boxes]
    
    # Basic statistics
    total_area = sum(areas)
    mean_area = np.mean(areas)
    std_area = np.std(areas) if len(areas) > 1 else 0
    mean_width = np.mean(widths)
    mean_height = np.mean(heights)
    mean_aspect_ratio = np.mean(aspect_ratios)
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    
    # Spread and density
    vertical_spread = max(ys) - min(ys) if len(ys) > 1 else 0
    horizontal_spread = max(xs) - min(xs) if len(xs) > 1 else 0
    density = total_area  # Fraction of page covered
    
    # Inter-box spacing (average distance between centers)
    if n_boxes > 1:
        spacings = []
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                dx = boxes[i]['cx'] - boxes[j]['cx']
                dy = boxes[i]['cy'] - boxes[j]['cy']
                dist = np.sqrt(dx**2 + dy**2)
                spacings.append(dist)
        mean_spacing = np.mean(spacings)
    else:
        mean_spacing = 0
    
    # Return feature vector
    features = np.array([
        n_boxes,
        total_area,
        mean_area,
        std_area,
        mean_width,
        mean_height,
        mean_aspect_ratio,
        mean_x,
        mean_y,
        density,
        vertical_spread,
        horizontal_spread,
        mean_spacing
    ])
    
    return features


def extract_features_from_dataset(dataset):
    """
    Extract layout features from entire dataset.
    
    Args:
        dataset: HuggingFace dataset
        
    Returns:
        numpy array of shape (n_samples, 13)
    """
    print(f"Extracting layout features from {len(dataset)} samples...")
    features_list = []
    
    for example in tqdm(dataset):
        features = extract_layout_features(example)
        features_list.append(features)
    
    features_array = np.array(features_list)
    print(f"Extracted features: {features_array.shape}")
    return features_array


def cluster_layouts(features, n_clusters=None, random_state=None):
    """
    Cluster document layouts using K-means.
    
    Args:
        features: numpy array of shape (n_samples, n_features)
        n_clusters: Number of clusters (default from config)
        random_state: Random seed (default from config)
        
    Returns:
        Dictionary with:
        - 'labels': cluster labels
        - 'scaler': fitted StandardScaler
        - 'kmeans': fitted KMeans model
        - 'features_scaled': scaled features
    """
    if n_clusters is None:
        n_clusters = config.NUM_CLUSTERS
    if random_state is None:
        random_state = config.RANDOM_SEED
    
    print(f"Clustering layouts into {n_clusters} groups...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(features_scaled)
    
    print(f"Clustering complete. Cluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} samples ({100*count/len(labels):.1f}%)")
    
    return {
        'labels': labels,
        'scaler': scaler,
        'kmeans': kmeans,
        'features_scaled': features_scaled
    }


def compute_pca(features, n_components=2):
    """
    Compute PCA for visualization.
    
    Args:
        features: numpy array of scaled features
        n_components: Number of PCA components
        
    Returns:
        Dictionary with:
        - 'pca': fitted PCA model
        - 'transformed': PCA-transformed features
        - 'variance_ratio': explained variance ratio
    """
    print(f"Computing PCA with {n_components} components...")
    
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(features)
    
    total_variance = sum(pca.explained_variance_ratio_)
    print(f"Total variance explained: {100*total_variance:.1f}%")
    
    return {
        'pca': pca,
        'transformed': transformed,
        'variance_ratio': pca.explained_variance_ratio_
    }


def compute_kv_distances(example):
    """
    Compute distances between linked key-value pairs.
    
    Args:
        example: Dictionary with 'annotations' field
        
    Returns:
        List of distances (empty if no links found)
    """
    annotations = example.get('annotations', [])
    
    if not annotations:
        return []
    
    # Build annotation ID to coords mapping
    ann_by_id = {}
    for ann in annotations:
        ann_id = ann.get('id')
        coords = ann.get('coordinates', [])
        
        if ann_id and coords:
            xs = [c['x'] for c in coords]
            ys = [c['y'] for c in coords]
            if xs and ys:
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                ann_by_id[ann_id] = (cx, cy)
    
    # Find links and compute distances
    distances = []
    for ann in annotations:
        # Safely get Linking attribute
        attrs = ann.get('attributes', {})
        if attrs:
            linking_attr = attrs.get('Linking')
            if linking_attr and isinstance(linking_attr, dict):
                linking = linking_attr.get('value')
            else:
                linking = None
        else:
            linking = None
        
        if linking:
            source_id = ann.get('id')
            
            if source_id in ann_by_id:
                key_cx, key_cy = ann_by_id[source_id]
                
                # Try exact match first
                if linking in ann_by_id:
                    val_cx, val_cy = ann_by_id[linking]
                    dist = np.sqrt((val_cx - key_cx)**2 + (val_cy - key_cy)**2)
                    distances.append(dist)
                else:
                    # Try prefix matching (for truncated IDs)
                    for aid, (val_cx, val_cy) in ann_by_id.items():
                        if aid.startswith(linking):
                            dist = np.sqrt((val_cx - key_cx)**2 + (val_cy - key_cy)**2)
                            distances.append(dist)
                            break
    
    return distances


def analyze_kv_links(dataset):
    """
    Analyze key-value linking distances across dataset.
    
    Args:
        dataset: HuggingFace dataset
        
    Returns:
        Dictionary with:
        - 'all_distances': list of all distances
        - 'distances_per_example': list of distance lists per example
        - 'stats': summary statistics
    """
    print(f"Analyzing KV links in {len(dataset)} samples...")
    
    all_distances = []
    distances_per_example = []
    
    for example in tqdm(dataset):
        dists = compute_kv_distances(example)
        distances_per_example.append(dists)
        all_distances.extend(dists)
    
    # Compute statistics
    stats = {
        'total_links': len(all_distances),
        'examples_with_links': sum(1 for d in distances_per_example if len(d) > 0),
        'mean_distance': np.mean(all_distances) if all_distances else 0,
        'median_distance': np.median(all_distances) if all_distances else 0,
        'std_distance': np.std(all_distances) if all_distances else 0,
        'min_distance': min(all_distances) if all_distances else 0,
        'max_distance': max(all_distances) if all_distances else 0
    }
    
    print(f"\nKV Link Statistics:")
    print(f"  Total links: {stats['total_links']}")
    print(f"  Examples with links: {stats['examples_with_links']}")
    print(f"  Mean distance: {stats['mean_distance']:.4f}")
    print(f"  Median distance: {stats['median_distance']:.4f}")
    
    return {
        'all_distances': all_distances,
        'distances_per_example': distances_per_example,
        'stats': stats
    }


def find_optimal_k(features, k_range=range(2, 11), random_state=None):
    """
    Find optimal number of clusters using multiple methods.
    
    Methods:
    1. Elbow method (inertia)
    2. Silhouette score (higher is better)
    3. Davies-Bouldin index (lower is better)
    4. Calinski-Harabasz score (higher is better)
    
    Args:
        features: numpy array of shape (n_samples, n_features)
        k_range: range of k values to test
        random_state: Random seed
        
    Returns:
        Dictionary with:
        - 'k_values': tested k values
        - 'inertias': inertia for each k
        - 'silhouette_scores': silhouette score for each k
        - 'davies_bouldin_scores': DB index for each k
        - 'calinski_harabasz_scores': CH score for each k
        - 'recommended_k': recommended k value
        - 'all_models': fitted KMeans models for each k
    """
    if random_state is None:
        random_state = config.RANDOM_SEED
    
    print(f"\n{'='*80}")
    print("OPTIMAL K SELECTION")
    print(f"{'='*80}")
    print(f"Testing k values: {list(k_range)}")
    print(f"Samples: {len(features)}")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    k_values = []
    inertias = []
    silhouette_scores_list = []
    davies_bouldin_scores_list = []
    calinski_harabasz_scores_list = []
    all_models = {}
    
    for k in tqdm(k_range, desc="Testing k values"):
        # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        # Compute metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(features_scaled, labels)
        davies_bouldin = davies_bouldin_score(features_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(features_scaled, labels)
        
        k_values.append(k)
        inertias.append(inertia)
        silhouette_scores_list.append(silhouette)
        davies_bouldin_scores_list.append(davies_bouldin)
        calinski_harabasz_scores_list.append(calinski_harabasz)
        all_models[k] = kmeans
        
        print(f"  k={k}: Inertia={inertia:.2f}, Silhouette={silhouette:.4f}, "
              f"DB={davies_bouldin:.4f}, CH={calinski_harabasz:.2f}")
    
    # Recommend k based on silhouette score (most reliable)
    best_idx = np.argmax(silhouette_scores_list)
    recommended_k = k_values[best_idx]
    
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION: k = {recommended_k}")
    print(f"  (Based on highest silhouette score: {silhouette_scores_list[best_idx]:.4f})")
    print(f"{'='*80}\n")
    
    return {
        'k_values': k_values,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores_list,
        'davies_bouldin_scores': davies_bouldin_scores_list,
        'calinski_harabasz_scores': calinski_harabasz_scores_list,
        'recommended_k': recommended_k,
        'all_models': all_models,
        'scaler': scaler,
        'features_scaled': features_scaled
    }


if __name__ == "__main__":
    # Test feature extraction
    print("Testing features.py...")
    
    # Would need dataset loaded to test
    print("Run via main.py to test with actual dataset")
