"""
Stage 3 Results Analysis
Analyzes predictions vs ground truth and displays benchmark scores.
"""

import json
import os
from glob import glob
import pandas as pd
from collections import Counter


def analyze_stage3_results(results_dir="outputs/stage3_test", baseline_name="nearest_neighbor_baseline"):
    """
    Analyze predictions vs ground truth for a baseline.
    
    Args:
        results_dir: Directory containing ground_truth/ and predictions/
        baseline_name: Name of baseline folder
    """
    pred_dir = os.path.join(results_dir, "predictions", baseline_name)
    gt_dir = os.path.join(results_dir, "ground_truth")
    
    if not os.path.exists(pred_dir):
        print(f"❌ Prediction directory not found: {pred_dir}")
        return
    
    if not os.path.exists(gt_dir):
        print(f"❌ Ground truth directory not found: {gt_dir}")
        return
    
    # Load all files
    pred_files = sorted(glob(os.path.join(pred_dir, "*.json")))
    gt_files = sorted(glob(os.path.join(gt_dir, "*.json")))
    
    print("="*80)
    print(f"STAGE 3 RESULTS ANALYSIS: {baseline_name}")
    print("="*80)
    print(f"\n📁 Prediction files: {len(pred_files)}")
    print(f"📁 Ground truth files: {len(gt_files)}")
    
    # Statistics
    stats = []
    all_pred_types = []
    all_gt_types = []
    
    for pred_file, gt_file in zip(pred_files, gt_files):
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred = json.load(f)
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt = json.load(f)
        
        image_id = os.path.basename(pred_file).replace('.json', '')
        
        pred_types = [kv['type'] for kv in pred['kvps_list']]
        gt_types = [kv['type'] for kv in gt['kvps_list']]
        
        all_pred_types.extend(pred_types)
        all_gt_types.extend(gt_types)
        
        stats.append({
            'image_id': image_id,
            'pred_kvps': len(pred['kvps_list']),
            'gt_kvps': len(gt['kvps_list']),
            'pred_types': pred_types,
            'gt_types': gt_types
        })
    
    # Summary statistics
    df = pd.DataFrame(stats)
    
    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    print(f"Total predicted KVPs:     {df['pred_kvps'].sum()}")
    print(f"Total ground truth KVPs:  {df['gt_kvps'].sum()}")
    print(f"Avg predicted per doc:    {df['pred_kvps'].mean():.2f} (±{df['pred_kvps'].std():.2f})")
    print(f"Avg GT per doc:           {df['gt_kvps'].mean():.2f} (±{df['gt_kvps'].std():.2f})")
    
    # Type distribution
    print("\n" + "="*80)
    print("📋 TYPE DISTRIBUTION")
    print("="*80)
    
    pred_type_counts = Counter(all_pred_types)
    gt_type_counts = Counter(all_gt_types)
    
    print("\nPredictions:")
    for kv_type, count in pred_type_counts.most_common():
        pct = 100 * count / len(all_pred_types)
        print(f"  {kv_type:12s}: {count:4d} ({pct:5.1f}%)")
    
    print("\nGround Truth:")
    for kv_type, count in gt_type_counts.most_common():
        pct = 100 * count / len(all_gt_types)
        print(f"  {kv_type:12s}: {count:4d} ({pct:5.1f}%)")
    
    # Sample examples
    print("\n" + "="*80)
    print("📄 SAMPLE EXAMPLES (First 3)")
    print("="*80)
    
    for i in range(min(3, len(stats))):
        print(f"\n[{i+1}] Image: {stats[i]['image_id']}")
        print(f"    Predicted:    {stats[i]['pred_kvps']} KVPs  {stats[i]['pred_types']}")
        print(f"    Ground Truth: {stats[i]['gt_kvps']} KVPs  {stats[i]['gt_types']}")
    
    print("\n" + "="*80)
    print("✓ Analysis complete")
    print("="*80)
    
    return df


if __name__ == "__main__":
    # Analyze latest test results
    analyze_stage3_results()
