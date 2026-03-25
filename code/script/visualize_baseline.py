"""
Visualization tools for Stage 3 baseline results.
"""

import json
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import numpy as np


def visualize_baseline_results(results_dir="outputs/stage3_test", baseline_name="nearest_neighbor_baseline", save_dir=None):
    """
    Create visualizations for baseline evaluation results.
    
    Args:
        results_dir: Directory containing ground_truth/ and predictions/
        baseline_name: Name of baseline folder
        save_dir: Directory to save plots (default: results_dir/visualizations)
    """
    pred_dir = os.path.join(results_dir, "predictions", baseline_name)
    gt_dir = os.path.join(results_dir, "ground_truth")
    
    if save_dir is None:
        save_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load all files
    pred_files = sorted(glob(os.path.join(pred_dir, "*.json")))
    gt_files = sorted(glob(os.path.join(gt_dir, "*.json")))
    
    print(f"Creating visualizations for {len(pred_files)} files...")
    
    # Collect data
    data = []
    all_pred_types = []
    all_gt_types = []
    
    for pred_file, gt_file in zip(pred_files, gt_files):
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred = json.load(f)
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt = json.load(f)
        
        image_id = os.path.basename(pred_file).replace('.json', '')
        
        pred_kvps = pred['kvps_list']
        gt_kvps = gt['kvps_list']
        
        pred_types = [kv['type'] for kv in pred_kvps]
        gt_types = [kv['type'] for kv in gt_kvps]
        
        all_pred_types.extend(pred_types)
        all_gt_types.extend(gt_types)
        
        data.append({
            'image_id': image_id,
            'pred_count': len(pred_kvps),
            'gt_count': len(gt_kvps),
            'pred_types': pred_types,
            'gt_types': gt_types
        })
    
    df = pd.DataFrame(data)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Predicted vs GT counts (scatter plot)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['gt_count'], df['pred_count'], alpha=0.6, s=100)
    
    # Add diagonal line (perfect prediction)
    max_count = max(df['gt_count'].max(), df['pred_count'].max())
    plt.plot([0, max_count], [0, max_count], 'r--', alpha=0.5, label='Perfect prediction')
    
    plt.xlabel('Ground Truth KVP Count', fontsize=12)
    plt.ylabel('Predicted KVP Count', fontsize=12)
    plt.title(f'Baseline: Predicted vs Ground Truth KVP Counts\n{baseline_name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, 'pred_vs_gt_counts.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # 2. Type distribution comparison (bar chart)
    pred_type_counts = Counter(all_pred_types)
    gt_type_counts = Counter(all_gt_types)
    
    # Get all unique types
    all_types = sorted(set(list(pred_type_counts.keys()) + list(gt_type_counts.keys())))
    
    pred_counts = [pred_type_counts.get(t, 0) for t in all_types]
    gt_counts = [gt_type_counts.get(t, 0) for t in all_types]
    
    x = np.arange(len(all_types))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, gt_counts, width, label='Ground Truth', alpha=0.8)
    plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
    
    plt.xlabel('KVP Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Type Distribution: Predicted vs Ground Truth\n{baseline_name}', fontsize=14)
    plt.xticks(x, all_types, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    save_path = os.path.join(save_dir, 'type_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # 3. Count difference distribution (histogram)
    df['count_diff'] = df['pred_count'] - df['gt_count']
    
    plt.figure(figsize=(10, 6))
    plt.hist(df['count_diff'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Perfect match')
    
    plt.xlabel('Prediction Error (Pred - GT)', fontsize=12)
    plt.ylabel('Number of Documents', fontsize=12)
    plt.title(f'Distribution of Prediction Errors\n{baseline_name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    save_path = os.path.join(save_dir, 'error_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # 4. Summary statistics table (as image)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Documents', len(df)],
        ['Total GT KVPs', df['gt_count'].sum()],
        ['Total Predicted KVPs', df['pred_count'].sum()],
        ['Avg GT per doc', f"{df['gt_count'].mean():.2f} ± {df['gt_count'].std():.2f}"],
        ['Avg Pred per doc', f"{df['pred_count'].mean():.2f} ± {df['pred_count'].std():.2f}"],
        ['Mean Error', f"{df['count_diff'].mean():.2f}"],
        ['Median Error', f"{df['count_diff'].median():.2f}"],
        ['',  ''],
        ['Type Distribution (GT)', ''],
    ]
    
    for kv_type, count in gt_type_counts.most_common():
        pct = 100 * count / len(all_gt_types) if all_gt_types else 0
        summary_data.append([f'  {kv_type}', f'{count} ({pct:.1f}%)'])
    
    summary_data.append(['', ''])
    summary_data.append(['Type Distribution (Pred)', ''])
    
    for kv_type, count in pred_type_counts.most_common():
        pct = 100 * count / len(all_pred_types) if all_pred_types else 0
        summary_data.append([f'  {kv_type}', f'{count} ({pct:.1f}%)'])
    
    table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title(f'Baseline Evaluation Summary\n{baseline_name}', fontsize=14, pad=20)
    
    save_path = os.path.join(save_dir, 'summary_table.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    print(f"\n✓ All visualizations saved to: {save_dir}")
    
    return df


if __name__ == "__main__":
    # Visualize latest test results
    visualize_baseline_results()
