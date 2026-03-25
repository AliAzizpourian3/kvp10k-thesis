"""
Utility functions for KVP10k experiments.
General helpers that don't fit in specific modules.
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
import config


def save_results(results, experiment_name, output_dir=None):
    """
    Save experiment results to disk.
    
    Args:
        results: Dictionary of results to save
        experiment_name: Name for the experiment
        output_dir: Output directory (default from config)
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(output_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(exist_ok=True)
    
    # Save results as pickle
    pickle_path = exp_dir / "results.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {pickle_path}")
    
    # Save metadata as JSON (serializable parts)
    metadata = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'config': {
            'dataset': config.DATASET_NAME,
            'subset_n': config.SUBSET_N,
            'device': config.DEVICE,
            'model': config.MODEL_NAME,
            'batch_size': config.BATCH_SIZE,
            'num_clusters': config.NUM_CLUSTERS,
            'ned_threshold': config.NED_THRESHOLD,
            'iou_threshold': config.IOU_THRESHOLD
        }
    }
    
    json_path = exp_dir / "metadata.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {json_path}")
    
    return exp_dir


def load_results(experiment_path):
    """
    Load saved experiment results.
    
    Args:
        experiment_path: Path to experiment directory
        
    Returns:
        Dictionary of results
    """
    exp_dir = Path(experiment_path)
    
    # Load pickle
    pickle_path = exp_dir / "results.pkl"
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
    
    # Load metadata
    json_path = exp_dir / "metadata.json"
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded experiment: {metadata['experiment_name']}")
    print(f"Timestamp: {metadata['timestamp']}")
    
    return results, metadata


def create_experiment_log(experiment_name, description, output_dir=None):
    """
    Create a log file for an experiment.
    
    Args:
        experiment_name: Name of experiment
        description: Description of what the experiment does
        output_dir: Output directory (default from config)
        
    Returns:
        Path to log file
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(output_dir) / f"{experiment_name}_{timestamp}.log"
    
    with open(log_path, 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Description: {description}\n")
        f.write("="*80 + "\n\n")
    
    print(f"Created log file: {log_path}")
    return log_path


def append_to_log(log_path, message):
    """
    Append message to log file.
    
    Args:
        log_path: Path to log file
        message: Message to append
    """
    with open(log_path, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")


def print_stage_header(stage_num, stage_name):
    """
    Print a formatted stage header.
    
    Args:
        stage_num: Stage number
        stage_name: Stage name
    """
    print("\n" + "="*80)
    print(f"STAGE {stage_num}: {stage_name}")
    print("="*80 + "\n")


def get_sample_by_index(dataset, index):
    """
    Get a sample from dataset by index.
    
    Args:
        dataset: HuggingFace dataset
        index: Index of sample
        
    Returns:
        Sample dictionary
    """
    return dataset[index]


def count_annotations_by_label(example):
    """
    Count annotations by label type.
    
    Args:
        example: Example from dataset
        
    Returns:
        Dictionary of label counts
    """
    annotations = example.get('annotations', [])
    
    label_counts = {}
    for ann in annotations:
        label = ann.get('label', {}).get('value')
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    return label_counts


def get_linking_pairs(example):
    """
    Extract all linking pairs (key-value) from an example.
    
    Args:
        example: Example from dataset
        
    Returns:
        List of (source_id, target_id) tuples
    """
    annotations = example.get('annotations', [])
    
    pairs = []
    for ann in annotations:
        source_id = ann.get('id')
        linking = ann.get('attributes', {}).get('Linking', {}).get('value')
        
        if source_id and linking:
            pairs.append((source_id, linking))
    
    return pairs


def format_time(seconds):
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def ensure_dir(path):
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test utilities
    print("Testing utils.py...")
    
    # Test time formatting
    print(f"10 seconds: {format_time(10)}")
    print(f"90 seconds: {format_time(90)}")
    print(f"7200 seconds: {format_time(7200)}")
    
    # Test stage header
    print_stage_header(0, "Test Stage")
    
    print("Utils tests complete!")
