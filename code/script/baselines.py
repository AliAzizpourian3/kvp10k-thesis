"""
Baseline models for KVP extraction - Stage 3.
Implements simple baselines with IBM-compatible output format.
"""

import numpy as np
from collections import defaultdict
import json
import os
from tqdm import tqdm
import config


class NearestNeighborBaseline:
    """
    Simple rule-based baseline: pair keys with nearest values spatially.
    """
    
    def __init__(self, max_distance=0.3):
        """
        Args:
            max_distance: Maximum normalized distance to consider pairing
        """
        self.max_distance = max_distance
        self.name = "Nearest Neighbor Baseline"
    
    def predict(self, example):
        """
        Predict KV pairs for a single example.
        
        Args:
            example: Dict with 'annotations' field
            
        Returns:
            List of predictions in IBM format
        """
        annotations = example.get('annotations', [])
        
        # Separate keys and values
        keys = []
        values = []
        
        for ann in annotations:
            coords = ann.get('coordinates', [])
            if not coords:
                continue
            
            # Compute centroid and bbox
            xs = [c['x'] for c in coords]
            ys = [c['y'] for c in coords]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            
            ann_data = {
                'id': ann.get('id'),
                'text': ann.get('text', ''),
                'bbox': bbox,
                'centroid': (cx, cy)
            }
            
            # Check if linking exists (this is a key)
            linking = ann.get('attributes', {}).get('Linking', {}).get('value')
            if linking:
                keys.append(ann_data)
            # Otherwise could be a value (simplified heuristic)
            elif ann.get('text', '').strip():  # Has text
                values.append(ann_data)
        
        # Pair keys with nearest values
        predictions = []
        used_values = set()
        
        for key in keys:
            key_cx, key_cy = key['centroid']
            
            # Find nearest value
            min_dist = float('inf')
            nearest_val = None
            
            for val in values:
                if val['id'] in used_values:
                    continue
                
                val_cx, val_cy = val['centroid']
                dist = np.sqrt((val_cx - key_cx)**2 + (val_cy - key_cy)**2)
                
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    nearest_val = val
            
            if nearest_val:
                predictions.append({
                    'type': 'kvp',
                    'key': {
                        'text': key['text'],
                        'bbox': key['bbox']
                    },
                    'value': {
                        'text': nearest_val['text'],
                        'bbox': nearest_val['bbox']
                    }
                })
                used_values.add(nearest_val['id'])
            else:
                # Unvalued key
                predictions.append({
                    'type': 'unvalued',
                    'key': {
                        'text': key['text'],
                        'bbox': key['bbox']
                    }
                })
        
        return predictions
    
    def predict_dataset(self, dataset):
        """
        Predict for entire dataset.
        
        Args:
            dataset: HuggingFace dataset
            
        Returns:
            List of predictions (one per example)
        """
        print(f"Running {self.name} on {len(dataset)} examples...")
        predictions = []
        
        for example in tqdm(dataset):
            preds = self.predict(example)
            predictions.append(preds)
        
        return predictions


def save_predictions_ibm_format(predictions_list, dataset, output_folder):
    """
    Save predictions in IBM benchmark format.
    
    IBM format per file:
    {
        "kvps_list": [
            {"type": "kvp", "key": {"text": "...", "bbox": [x1,y1,x2,y2]}, "value": {...}},
            {"type": "unvalued", "key": {...}},
            {"type": "unkeyed", "key": {...}, "value": {...}}
        ]
    }
    
    Args:
        predictions_list: List of predictions (one per example)
        dataset: Original dataset (for image IDs)
        output_folder: Path to save JSON files
    """
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Saving {len(predictions_list)} predictions to {output_folder}...")
    
    for idx, (preds, example) in enumerate(zip(predictions_list, dataset)):
        # Get image ID from example
        image_id = example.get('id', f'image_{idx}')
        
        # Convert to IBM format
        ibm_format = {"kvps_list": preds}
        
        # Save to file
        filename = f"{image_id}.json"
        filepath = os.path.join(output_folder, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(ibm_format, f, indent=2)
    
    print(f"✓ Saved {len(predictions_list)} prediction files")


def create_ground_truth_ibm_format(dataset, output_folder):
    """
    Convert HuggingFace dataset annotations to IBM benchmark format.
    
    Args:
        dataset: HuggingFace dataset with annotations
        output_folder: Path to save GT JSON files
    """
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Converting {len(dataset)} examples to IBM GT format...")
    
    for idx, example in enumerate(tqdm(dataset)):
        image_id = example.get('id', f'image_{idx}')
        annotations = example.get('annotations', [])
        
        # Build annotation lookup
        ann_by_id = {}
        for ann in annotations:
            ann_id = ann.get('id')
            coords = ann.get('coordinates', [])
            text = ann.get('text', '')
            
            if coords:
                xs = [c['x'] for c in coords]
                ys = [c['y'] for c in coords]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                
                ann_by_id[ann_id] = {
                    'text': text,
                    'bbox': bbox,
                    'linking': ann.get('attributes', {}).get('Linking', {}).get('value')
                }
        
        # Extract KV pairs based on linking
        kvps_list = []
        used_ids = set()
        
        for ann_id, ann_data in ann_by_id.items():
            linking = ann_data.get('linking')
            
            if linking and ann_id not in used_ids:
                # Find linked value
                value_id = linking
                value_data = None
                
                # Try exact match
                if value_id in ann_by_id:
                    value_data = ann_by_id[value_id]
                else:
                    # Try prefix match (for truncated IDs)
                    for vid, vdata in ann_by_id.items():
                        if vid and vid.startswith(value_id):  # Check vid is not None
                            value_data = vdata
                            value_id = vid
                            break
                
                if value_data:
                    # Regular KVP
                    kvps_list.append({
                        'type': 'kvp',
                        'key': {
                            'text': ann_data['text'],
                            'bbox': ann_data['bbox']
                        },
                        'value': {
                            'text': value_data['text'],
                            'bbox': value_data['bbox']
                        }
                    })
                    used_ids.add(ann_id)
                    used_ids.add(value_id)
                else:
                    # Unvalued (key without value)
                    kvps_list.append({
                        'type': 'unvalued',
                        'key': {
                            'text': ann_data['text'],
                            'bbox': ann_data['bbox']
                        }
                    })
                    used_ids.add(ann_id)
        
        # Save to file
        filename = f"{image_id}.json"
        filepath = os.path.join(output_folder, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({'kvps_list': kvps_list}, f, indent=2)
    
    print(f"✓ Saved {len(dataset)} GT files")


class ImprovedHeuristicBaseline:
    """
    Enhanced rule-based baseline with multiple heuristics:
    - Direction bias (values typically right/below keys)
    - Alignment checks (horizontal/vertical alignment)
    - OCR pattern matching (key/value text patterns)
    - Column structure detection
    - Multi-criteria scoring
    """
    
    def __init__(self, max_distance=0.4):
        """
        Args:
            max_distance: Maximum normalized distance to consider pairing
        """
        self.max_distance = max_distance
        self.name = "Improved Heuristic Baseline"
    
    def _compute_direction_score(self, key_centroid, val_centroid):
        """
        Score based on typical key-value spatial relationships.
        Values are typically to the right or below keys.
        
        Returns: score in [0, 1], higher is better
        """
        key_cx, key_cy = key_centroid
        val_cx, val_cy = val_centroid
        
        dx = val_cx - key_cx
        dy = val_cy - key_cy
        
        score = 0.0
        
        # Prefer right (positive dx)
        if dx > 0:
            score += 0.5
        
        # Prefer below or same level (small dy, or positive)
        if abs(dy) < 0.05:  # Horizontally aligned
            score += 0.3
        elif dy > 0:  # Below
            score += 0.2
        
        return score
    
    def _compute_alignment_score(self, key_bbox, val_bbox):
        """
        Score based on horizontal/vertical alignment.
        
        Returns: score in [0, 1], higher is better
        """
        k_x1, k_y1, k_x2, k_y2 = key_bbox
        v_x1, v_y1, v_x2, v_y2 = val_bbox
        
        score = 0.0
        
        # Horizontal alignment (y-coordinates overlap)
        y_overlap = min(k_y2, v_y2) - max(k_y1, v_y1)
        k_height = k_y2 - k_y1
        v_height = v_y2 - v_y1
        
        if y_overlap > 0:
            alignment_ratio = y_overlap / min(k_height, v_height)
            score += alignment_ratio * 0.4
        
        # Vertical alignment (x-coordinates overlap)
        x_overlap = min(k_x2, v_x2) - max(k_x1, v_x1)
        k_width = k_x2 - k_x1
        v_width = v_x2 - v_x1
        
        if x_overlap > 0:
            alignment_ratio = x_overlap / min(k_width, v_width)
            score += alignment_ratio * 0.3
        
        return min(score, 1.0)
    
    def _compute_pattern_score(self, key_text, val_text):
        """
        Score based on OCR pattern matching.
        Keys often end with ':', values are often numeric or alphanumeric.
        
        Returns: score in [0, 1], higher is better
        """
        score = 0.0
        
        key_text = key_text.strip()
        val_text = val_text.strip()
        
        # Key patterns
        if key_text.endswith(':'):
            score += 0.3
        
        # Value patterns
        if val_text:
            # Numeric values
            if any(c.isdigit() for c in val_text):
                score += 0.2
            
            # Date patterns (contains /)
            if '/' in val_text:
                score += 0.15
            
            # Currency patterns ($, €, etc.)
            if any(c in val_text for c in '$€£¥'):
                score += 0.15
        
        return min(score, 1.0)
    
    def _detect_columns(self, annotations):
        """
        Detect column structure by clustering x-coordinates.
        
        Returns: dict mapping column_id to list of annotation indices
        """
        if not annotations:
            return {}
        
        # Extract x-coordinates of centroids
        centroids = []
        for ann in annotations:
            coords = ann.get('coordinates', [])
            if coords:
                xs = [c['x'] for c in coords]
                cx = sum(xs) / len(xs)
                centroids.append(cx)
            else:
                centroids.append(0)
        
        # Simple column detection: cluster by x-coordinate
        # For simplicity, use fixed threshold
        columns = defaultdict(list)
        sorted_indices = sorted(range(len(centroids)), key=lambda i: centroids[i])
        
        col_id = 0
        last_cx = -1
        
        for idx in sorted_indices:
            cx = centroids[idx]
            
            # New column if gap > 0.15
            if last_cx >= 0 and (cx - last_cx) > 0.15:
                col_id += 1
            
            columns[col_id].append(idx)
            last_cx = cx
        
        return columns
    
    def _compute_column_score(self, key_idx, val_idx, columns):
        """
        Score based on column structure.
        Keys and values in same or adjacent columns score higher.
        
        Returns: score in [0, 1], higher is better
        """
        # Find columns for key and value
        key_col = None
        val_col = None
        
        for col_id, indices in columns.items():
            if key_idx in indices:
                key_col = col_id
            if val_idx in indices:
                val_col = col_id
        
        if key_col is None or val_col is None:
            return 0.5  # Neutral
        
        # Same column or adjacent
        col_diff = abs(val_col - key_col)
        
        if col_diff == 0:
            return 0.3  # Same column (moderate - could be stacked)
        elif col_diff == 1:
            return 1.0  # Adjacent columns (ideal)
        else:
            return 0.1  # Far apart
    
    def predict(self, example):
        """
        Predict KV pairs using multiple heuristics.
        
        Args:
            example: Dict with 'annotations' field
            
        Returns:
            List of predictions in IBM format
        """
        annotations = example.get('annotations', [])
        
        # Separate keys and values
        keys = []
        values = []
        all_anns = []  # For column detection
        
        for idx, ann in enumerate(annotations):
            coords = ann.get('coordinates', [])
            if not coords:
                continue
            
            # Compute centroid and bbox
            xs = [c['x'] for c in coords]
            ys = [c['y'] for c in coords]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            
            ann_data = {
                'idx': idx,
                'id': ann.get('id'),
                'text': ann.get('text', ''),
                'bbox': bbox,
                'centroid': (cx, cy)
            }
            
            all_anns.append(ann_data)
            
            # Check if linking exists (this is a key)
            linking = ann.get('attributes', {}).get('Linking', {}).get('value')
            if linking:
                keys.append(ann_data)
            elif ann.get('text', '').strip():  # Has text
                values.append(ann_data)
        
        # Detect column structure
        columns = self._detect_columns(annotations)
        
        # Pair keys with values using multi-criteria scoring
        predictions = []
        used_values = set()
        
        for key in keys:
            key_cx, key_cy = key['centroid']
            key_bbox = key['bbox']
            key_text = key['text']
            key_idx = key['idx']
            
            # Score all candidate values
            candidates = []
            
            for val in values:
                if val['id'] in used_values:
                    continue
                
                val_cx, val_cy = val['centroid']
                val_bbox = val['bbox']
                val_text = val['text']
                val_idx = val['idx']
                
                # Compute distance
                dist = np.sqrt((val_cx - key_cx)**2 + (val_cy - key_cy)**2)
                
                if dist > self.max_distance:
                    continue
                
                # Multi-criteria scoring
                distance_score = 1.0 - (dist / self.max_distance)  # [0, 1]
                direction_score = self._compute_direction_score((key_cx, key_cy), (val_cx, val_cy))
                alignment_score = self._compute_alignment_score(key_bbox, val_bbox)
                pattern_score = self._compute_pattern_score(key_text, val_text)
                column_score = self._compute_column_score(key_idx, val_idx, columns)
                
                # Weighted combination
                total_score = (
                    0.25 * distance_score +
                    0.25 * direction_score +
                    0.25 * alignment_score +
                    0.15 * pattern_score +
                    0.10 * column_score
                )
                
                candidates.append({
                    'value': val,
                    'score': total_score,
                    'distance': dist
                })
            
            # Select best candidate
            if candidates:
                best = max(candidates, key=lambda x: x['score'])
                predictions.append({
                    'type': 'kvp',
                    'key': {
                        'text': key['text'],
                        'bbox': key['bbox']
                    },
                    'value': {
                        'text': best['value']['text'],
                        'bbox': best['value']['bbox']
                    }
                })
                used_values.add(best['value']['id'])
            else:
                # Unvalued key
                predictions.append({
                    'type': 'unvalued',
                    'key': {
                        'text': key['text'],
                        'bbox': key['bbox']
                    }
                })
        
        return predictions
    
    def predict_dataset(self, dataset):
        """
        Predict for entire dataset.
        
        Args:
            dataset: HuggingFace dataset
            
        Returns:
            List of predictions (one per example)
        """
        print(f"Running {self.name} on {len(dataset)} examples...")
        predictions = []
        
        for example in tqdm(dataset):
            preds = self.predict(example)
            predictions.append(preds)
        
        return predictions


if __name__ == "__main__":
    print("Baselines module ready for Stage 3")
