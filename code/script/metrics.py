"""
Evaluation metrics for KVP extraction.
Implements NED, IoU, and F1 scoring.
"""

import numpy as np
from typing import List, Tuple, Dict
import config


def normalized_edit_distance(pred: str, target: str) -> float:
    """
    Compute Normalized Edit Distance (NED) between two strings.
    
    NED = Levenshtein distance / max(len(pred), len(target))
    
    Args:
        pred: Predicted string
        target: Ground truth string
        
    Returns:
        NED value in [0, 1], where 0 = perfect match
    """
    # Levenshtein distance (dynamic programming)
    m, n = len(pred), len(target)
    
    if m == 0:
        return 1.0 if n > 0 else 0.0
    if n == 0:
        return 1.0
    
    # Create DP table
    dp = np.zeros((m+1, n+1), dtype=int)
    
    # Initialize first row and column
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            if pred[i-1] == target[j-1]:
                cost = 0
            else:
                cost = 1
            
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    
    # Normalize by max length
    edit_distance = dp[m][n]
    max_len = max(m, n)
    ned = edit_distance / max_len
    
    return ned


def intersection_over_union(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU (Intersection over Union) between two bounding boxes.
    
    Box format: [x_min, y_min, x_max, y_max]
    
    Args:
        box1: First bounding box
        box2: Second bounding box
        
    Returns:
        IoU value in [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Compute intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    iou = intersection_area / union_area
    return iou


def coords_to_box(coordinates: List[Dict]) -> List[float]:
    """
    Convert KVP10k coordinates format to bounding box.
    
    Args:
        coordinates: List of {x, y} dictionaries
        
    Returns:
        Box in [x_min, y_min, x_max, y_max] format
    """
    if not coordinates:
        return [0, 0, 0, 0]
    
    xs = [c['x'] for c in coordinates]
    ys = [c['y'] for c in coordinates]
    
    return [min(xs), min(ys), max(xs), max(ys)]


def match_prediction_to_ground_truth(
    pred_box: List[float],
    pred_text: str,
    gt_boxes: List[List[float]],
    gt_texts: List[str],
    iou_threshold: float = None,
    ned_threshold: float = None
) -> Tuple[bool, int]:
    """
    Match a prediction to ground truth using IoU and NED thresholds.
    
    Args:
        pred_box: Predicted bounding box
        pred_text: Predicted text
        gt_boxes: List of ground truth boxes
        gt_texts: List of ground truth texts
        iou_threshold: IoU threshold for matching (default from config)
        ned_threshold: NED threshold for matching (default from config)
        
    Returns:
        (matched, best_gt_index) where matched is True if a match is found
    """
    if iou_threshold is None:
        iou_threshold = config.IOU_THRESHOLD
    if ned_threshold is None:
        ned_threshold = config.NED_THRESHOLD
    
    best_score = -1
    best_idx = -1
    matched = False
    
    for i, (gt_box, gt_text) in enumerate(zip(gt_boxes, gt_texts)):
        # Compute IoU
        iou = intersection_over_union(pred_box, gt_box)
        
        # Compute NED
        ned = normalized_edit_distance(pred_text, gt_text)
        
        # Check if both criteria are met
        if iou >= iou_threshold and ned <= ned_threshold:
            # Use combined score for best match
            score = iou - ned  # Higher IoU, lower NED = better
            if score > best_score:
                best_score = score
                best_idx = i
                matched = True
    
    return matched, best_idx


def compute_precision_recall_f1(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = None,
    ned_threshold: float = None
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score for KVP extraction.
    
    Args:
        predictions: List of predicted KV pairs
                     Each dict has: {text, box, key/value}
        ground_truths: List of ground truth KV pairs
        iou_threshold: IoU threshold
        ned_threshold: NED threshold
        
    Returns:
        Dictionary with precision, recall, F1
    """
    if iou_threshold is None:
        iou_threshold = config.IOU_THRESHOLD
    if ned_threshold is None:
        ned_threshold = config.NED_THRESHOLD
    
    if not predictions:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if not ground_truths:
        return {'precision': 0.0, 'recall': 0.0 if not predictions else 1.0, 'f1': 0.0}
    
    # Extract boxes and texts
    pred_boxes = [p['box'] for p in predictions]
    pred_texts = [p['text'] for p in predictions]
    gt_boxes = [g['box'] for g in ground_truths]
    gt_texts = [g['text'] for g in ground_truths]
    
    # Track matches
    matched_preds = set()
    matched_gts = set()
    
    for i, (pred_box, pred_text) in enumerate(zip(pred_boxes, pred_texts)):
        matched, gt_idx = match_prediction_to_ground_truth(
            pred_box, pred_text, gt_boxes, gt_texts,
            iou_threshold, ned_threshold
        )
        
        if matched and gt_idx not in matched_gts:
            matched_preds.add(i)
            matched_gts.add(gt_idx)
    
    # Compute metrics
    true_positives = len(matched_preds)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truths) - len(matched_gts)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def evaluate_kvp_extraction(
    all_predictions: List[List[Dict]],
    all_ground_truths: List[List[Dict]],
    iou_threshold: float = None,
    ned_threshold: float = None
) -> Dict[str, float]:
    """
    Evaluate KVP extraction across multiple documents.
    
    Args:
        all_predictions: List of prediction lists (one per document)
        all_ground_truths: List of ground truth lists (one per document)
        iou_threshold: IoU threshold
        ned_threshold: NED threshold
        
    Returns:
        Dictionary with aggregated metrics
    """
    all_tp = 0
    all_fp = 0
    all_fn = 0
    
    for preds, gts in zip(all_predictions, all_ground_truths):
        metrics = compute_precision_recall_f1(preds, gts, iou_threshold, ned_threshold)
        all_tp += metrics['true_positives']
        all_fp += metrics['false_positives']
        all_fn += metrics['false_negatives']
    
    # Aggregate metrics
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_true_positives': all_tp,
        'total_false_positives': all_fp,
        'total_false_negatives': all_fn,
        'total_predictions': all_tp + all_fp,
        'total_ground_truths': all_tp + all_fn
    }


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics.py...")
    
    # Test NED
    print("\nTesting NED:")
    print(f"NED('hello', 'hello') = {normalized_edit_distance('hello', 'hello'):.4f}")  # 0.0
    print(f"NED('hello', 'helo') = {normalized_edit_distance('hello', 'helo'):.4f}")    # ~0.2
    print(f"NED('hello', 'world') = {normalized_edit_distance('hello', 'world'):.4f}")  # 0.8
    
    # Test IoU
    print("\nTesting IoU:")
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    box3 = [0, 0, 10, 10]
    print(f"IoU(box1, box2) = {intersection_over_union(box1, box2):.4f}")  # 0.25
    print(f"IoU(box1, box3) = {intersection_over_union(box1, box3):.4f}")  # 1.0
    
    # Test F1 computation
    print("\nTesting F1 computation:")
    predictions = [
        {'text': 'Invoice', 'box': [10, 10, 50, 30]},
        {'text': 'Total', 'box': [10, 50, 40, 70]}
    ]
    ground_truths = [
        {'text': 'Invoice', 'box': [10, 10, 50, 30]},
        {'text': 'Total:', 'box': [10, 50, 40, 70]},
        {'text': 'Date', 'box': [10, 90, 40, 110]}
    ]
    
    metrics = compute_precision_recall_f1(predictions, ground_truths)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    
    print("\nMetrics tests complete!")
