"""
Stage 3 Error Analysis: Per-cluster breakdown and error taxonomy.

Analyzes Stage 3 Mistral baseline predictions to answer:
1. Do errors differ by layout type (dense vs sparse)?
2. What kinds of errors dominate (wrong text, wrong location, hallucinated, missed)?

Output: per-cluster error tables for thesis Chapter 7.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Layout density classification
# ============================================================================

def classify_layout_density(image_width: int, image_height: int, num_annotations: int) -> str:
    """
    Classify document layout as sparse or dense based on:
    - Image dimensions (quantized to Stage 2 clusters)
    - Annotation count as proxy for annotation density
    
    Stage 2 clustering (from Chapter 4 analysis):
      Cluster 0 (dense):  ~35% of dataset, mean n_boxes ≈ 25.36
      Cluster 1 (sparse): ~65% of dataset, mean n_boxes ≈ 0.062
    
    Simple heuristic: use num_annotations normalized by page area.
    """
    page_area = image_width * image_height
    density = num_annotations / max(1, page_area / 1000000)  # annotations per megapixel
    
    # Cluster 1 (sparse) if density < median (threshold ~0.5); Cluster 0 (dense) otherwise
    if density < 0.5:
        return "Cluster_1_Sparse"
    else:
        return "Cluster_0_Dense"


# ============================================================================
# Error categorization
# ============================================================================

def categorize_errors(pred_kvps: List[Dict], gt_kvps: List[Dict]) -> Dict[str, int]:
    """
    Categorize predicted vs GT KVPs into error types:
    - Correct: matched in evaluation
    - Hallucinated: predicted but no GT match
    - Missed: GT exists but not predicted
    - Wrong_text: text mismatch but location match
    - Wrong_location: text match but location mismatch
    
    For now, use simple heuristic: compare extracted text sets.
    """
    pred_texts = set()
    gt_texts = set()
    
    for kvp in pred_kvps:
        if kvp.get("type") == "kvp":
            text = f"{kvp.get('key',{}).get('text','')}|||{kvp.get('value',{}).get('text','')}"
        else:
            text = kvp.get('key', {}).get('text', '')
        pred_texts.add(text)
    
    for kvp in gt_kvps:
        if kvp.get("type") == "kvp":
            text = f"{kvp.get('key',{}).get('text','')}|||{kvp.get('value',{}).get('text','')}"
        else:
            text = kvp.get('key', {}).get('text', '')
        gt_texts.add(text)
    
    correct = len(pred_texts & gt_texts)
    hallucinated = len(pred_texts - gt_texts)
    missed = len(gt_texts - pred_texts)
    
    return {
        "correct": correct,
        "hallucinated": hallucinated,
        "missed": missed,
        "pred_total": len(pred_kvps),
        "gt_total": len(gt_kvps),
    }


# ============================================================================
# Main analysis
# ============================================================================

def analyze_stage3_errors(
    eval_json: str,
    pred_dir: str,
    gt_dir: str,
    output_dir: str,
):
    """
    Load Stage 3 evaluation results + predictions/GT, categorize errors by cluster.
    """
    eval_json = Path(eval_json)
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation summary
    with open(eval_json) as f:
        eval_data = json.load(f)
    
    per_doc = eval_data.get("per_document", [])
    logger.info(f"Loaded {len(per_doc)} document evaluations")
    
    # Categorize by cluster
    cluster_stats = defaultdict(lambda: {
        "correct": 0,
        "hallucinated": 0,
        "missed": 0,
        "pred_total": 0,
        "gt_total": 0,
        "docs": 0,
    })
    
    error_details = []
    
    for doc_row in per_doc:
        hash_name = doc_row.get("hash_name", "")
        
        # Load prediction and GT
        pred_file = pred_dir / f"{hash_name}.json"
        gt_file = gt_dir / f"{hash_name}.json"
        
        if not pred_file.exists() or not gt_file.exists():
            continue
        
        with open(pred_file) as f:
            pred_data = json.load(f)
        with open(gt_file) as f:
            gt_data = json.load(f)
        
        pred_kvps = pred_data.get("kvps_list", [])
        gt_kvps = gt_data.get("gt_kvps", {}).get("kvps_list", [])
        
        # Extract layout info from GT file to classify density
        image_width = gt_data.get("image_width", 2550)
        image_height = gt_data.get("image_height", 3300)
        cluster = classify_layout_density(image_width, image_height, len(gt_kvps))
        
        # Categorize errors
        errors = categorize_errors(pred_kvps, gt_kvps)
        
        # Accumulate statistics
        stats = cluster_stats[cluster]
        stats["correct"] += errors["correct"]
        stats["hallucinated"] += errors["hallucinated"]
        stats["missed"] += errors["missed"]
        stats["pred_total"] += errors["pred_total"]
        stats["gt_total"] += errors["gt_total"]
        stats["docs"] += 1
        
        error_details.append({
            "hash_name": hash_name,
            "cluster": cluster,
            **errors,
            "n_text_only_tp": doc_row.get("text_only_tp", 0),
            "n_text_bbox_tp": doc_row.get("text_bbox_tp", 0),
        })
    
    # Save detailed results
    detail_file = output_dir / "error_details.json"
    with open(detail_file, "w") as f:
        json.dump(error_details, f, indent=2)
    logger.info(f"Saved {len(error_details)} document details → {detail_file}")
    
    # Generate summary tables
    summary_file = output_dir / "cluster_error_summary.json"
    summary = {}
    for cluster, stats in cluster_stats.items():
        if stats["docs"] == 0:
            continue
        
        correct_rate = stats["correct"] / max(1, stats["pred_total"])
        hallucinate_rate_of_preds = stats["hallucinated"] / max(1, stats["pred_total"])
        hallucinate_rate_of_gt = stats["hallucinated"] / max(1, stats["gt_total"])
        missed_rate = stats["missed"] / max(1, stats["gt_total"])
        
        summary[cluster] = {
            "n_docs": stats["docs"],
            "n_pred_kvps": stats["pred_total"],
            "n_gt_kvps": stats["gt_total"],
            "correct_kvps": stats["correct"],
            "hallucinated_kvps": stats["hallucinated"],
            "missed_kvps": stats["missed"],
            "correct_rate": round(correct_rate, 4),
            "hallucinate_rate_of_preds": round(hallucinate_rate_of_preds, 4),
            "hallucinate_rate_of_gt": round(hallucinate_rate_of_gt, 4),
            "missed_rate": round(missed_rate, 4),
        }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved cluster summary → {summary_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("STAGE 3 ERROR ANALYSIS BY LAYOUT CLUSTER")
    print("="*80)
    for cluster in sorted(summary.keys()):
        s = summary[cluster]
        print(f"\n{cluster}:")
        print(f"  Documents: {s['n_docs']}")
        print(f"  Predicted KVPs: {s['n_pred_kvps']}")
        print(f"  GT KVPs: {s['n_gt_kvps']}")
        print(f"  Correct: {s['correct_kvps']} ({s['correct_rate']:.1%})")
        print(f"  Hallucinated: {s['hallucinated_kvps']} ({s['hallucinate_rate_of_preds']:.1%} of predictions, {s['hallucinate_rate_of_gt']:.1%} of GT)")
        print(f"  Missed: {s['missed_kvps']} ({s['missed_rate']:.1%} of GT)")
    print("="*80)
    
    return error_details


# ============================================================================
# Additional Analysis 1: Worst Offender Documents
# ============================================================================

def analyze_worst_offenders(error_details, top_n=15):
    """
    Find documents with highest hallucination counts.
    Reports top_n worst offenders and checks if they share patterns (cluster bias, etc.).
    
    Fields used from error_details:
      - hash_name: document identifier
      - gt_total: number of ground truth KVPs
      - pred_total: number of predicted KVPs
      - hallucinated: number of hallucinated (unmatched predicted) KVPs
      - cluster: layout cluster classification
    """
    offenders = []
    for doc in error_details:
        hash_name = doc.get("hash_name", "unknown")
        gt_total = doc.get("gt_total", 0)
        pred_total = doc.get("pred_total", 0)
        hallucinated = doc.get("hallucinated", 0)
        cluster = doc.get("cluster", "unknown")
        
        if pred_total > 0:
            halluc_rate = hallucinated / pred_total
        else:
            halluc_rate = 0.0
        
        offenders.append({
            "hash_name": hash_name,
            "gt_total": gt_total,
            "pred_total": pred_total,
            "hallucinated": hallucinated,
            "halluc_rate": round(halluc_rate, 4),
            "cluster": cluster,
            "zero_gt_nonzero_pred": (gt_total == 0 and pred_total > 0)
        })
    
    # Sort by absolute hallucination count descending
    offenders_sorted = sorted(offenders, key=lambda x: x["hallucinated"], reverse=True)
    top_offenders = offenders_sorted[:top_n]
    
    # Summary stats
    zero_gt_docs = [o for o in offenders if o["zero_gt_nonzero_pred"]]
    cluster_counts = {}
    for o in top_offenders:
        c = o["cluster"]
        cluster_counts[c] = cluster_counts.get(c, 0) + 1
    
    result = {
        "top_offenders": top_offenders,
        "total_zero_gt_nonzero_pred_docs": len(zero_gt_docs),
        "total_hallucinated_from_zero_gt_docs": sum(o["hallucinated"] for o in zero_gt_docs),
        "top_offenders_cluster_distribution": cluster_counts
    }
    
    print("\n" + "="*80)
    print("WORST OFFENDER DOCUMENTS (Highest Hallucination Counts)")
    print("="*80)
    print(f"Documents with 0 GT but predictions made: {len(zero_gt_docs)}")
    print(f"Total hallucinated KVPs from zero-GT docs: {result['total_hallucinated_from_zero_gt_docs']}")
    print(f"\nTop {top_n} documents by hallucination count:")
    for o in top_offenders:
        print(f"  {o['hash_name']}: {o['hallucinated']} hallucinated / "
              f"{o['pred_total']} predicted / {o['gt_total']} GT "
              f"(rate={o['halluc_rate']}) [{o['cluster']}]")
    print(f"\nCluster distribution in top offenders: {cluster_counts}")
    print("="*80)
    
    return result


# ============================================================================
# Additional Analysis 2: Key vs Value Extraction Asymmetry
# ============================================================================

def analyze_key_value_asymmetry(error_details):
    """
    Check whether keys or values are harder to extract correctly.
    
    NOTE: Current error_details.json does **not** contain per-field NED metrics
    (matched_pairs with key_ned/value_ned). This analysis is a placeholder
    that returns a warning. To enable this analysis, the error categorization
    would need to track per-field text accuracy separately.
    """
    
    print("\n" + "="*80)
    print("KEY vs VALUE EXTRACTION ASYMMETRY")
    print("="*80)
    print("⚠️  WARNING: Per-field NED metrics (key_ned, value_ned) not yet tracked.")
    print("To enable this analysis, modify categorize_errors() to decompose KVP pairs")
    print("and compute Levenshtein distance separately for keys and values.")
    print("="*80)
    
    result = {
        "status": "not_yet_implemented",
        "reason": "matched_pairs with key_ned/value_ned fields not in current error_details",
        "note": "This would require refactoring error categorization to track per-component metrics"
    }
    
    return result


if __name__ == "__main__":
    import argparse
    
    p = argparse.ArgumentParser(description="Analyze Stage 3 errors by layout cluster")
    p.add_argument(
        "--eval_json",
        default="data/outputs/stage3_mistral/evaluation_stage0_ned02_iou03.json",
        help="Stage 3 evaluation JSON",
    )
    p.add_argument(
        "--pred_dir",
        default="data/outputs/stage3_mistral/predictions",
        help="Prediction directory",
    )
    p.add_argument(
        "--gt_dir",
        default="data/prepared/test",
        help="Ground truth directory",
    )
    p.add_argument(
        "--output_dir",
        default="data/outputs/stage3_mistral/error_analysis",
        help="Output directory for error analysis",
    )
    args = p.parse_args()
    
    # Run main analysis and capture error details
    error_details = analyze_stage3_errors(args.eval_json, args.pred_dir, args.gt_dir, args.output_dir)
    
    # ── NEW: Additional Stage 3 Insights ──────────────────────────────────
    worst_offenders = analyze_worst_offenders(error_details, top_n=15)
    kv_asymmetry = analyze_key_value_asymmetry(error_details)
    
    # Save combined additional insights
    additional_insights = {
        "worst_offenders_summary": {
            "total_zero_gt_nonzero_pred_docs": worst_offenders["total_zero_gt_nonzero_pred_docs"],
            "total_hallucinated_from_zero_gt_docs": worst_offenders["total_hallucinated_from_zero_gt_docs"],
            "top_offenders_cluster_distribution": worst_offenders["top_offenders_cluster_distribution"],
            "top_15_offenders": worst_offenders["top_offenders"]
        },
        "key_value_asymmetry": kv_asymmetry
    }
    
    output_path = os.path.join(args.output_dir, "additional_insights.json")
    with open(output_path, "w") as f:
        json.dump(additional_insights, f, indent=2)
    logger.info(f"Additional insights saved to {output_path}")
    print(f"\n✓ Additional insights saved to {output_path}")
