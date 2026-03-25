"""
Configuration file for KVP10k extraction experiments.
All parameters controlled here for easy experimentation.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
# Default to repo-relative paths so this runs on Linux clusters without editing.
_SCRIPT_DIR = Path(__file__).resolve().parent

# Allow overriding via env var for cluster-specific layouts.
_WORKSPACE_ROOT = Path(os.environ.get("KVP10K_WORKSPACE_ROOT", str(_SCRIPT_DIR)))

WORKSPACE_ROOT = str(_WORKSPACE_ROOT)
SCRIPT_DIR = str(_SCRIPT_DIR)
OUTPUT_DIR = os.environ.get("KVP10K_OUTPUT_DIR", str(_SCRIPT_DIR / "outputs"))

# HuggingFace datasets cache for KVP10k
KVP_CACHE = os.environ.get("KVP10K_HF_CACHE", str(_SCRIPT_DIR / "hf_cache"))

# ============================================================================
# DATASET PARAMETERS
# ============================================================================
DATASET_NAME = "ibm/KVP10k"
SUBSET_N = None  # Full dataset (48,280 train / 5,255 test)
                # Set to a number (e.g., 3000) for development subset
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

# ============================================================================
# HARDWARE & COMPUTE
# ============================================================================
DEVICE = "cpu"  # Will change to "cuda" when GPU available
BATCH_SIZE = 2  # Small for CPU, will increase for GPU

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
# Development model (small, fast, CPU-friendly)
MODEL_NAME_DEV = "google/flan-t5-small"

# Production model (larger, GPU-required)
MODEL_NAME_GPU = "mistralai/Mistral-7B-Instruct-v0.2"

# Active model (switch based on DEVICE)
MODEL_NAME = MODEL_NAME_DEV if DEVICE == "cpu" else MODEL_NAME_GPU

# ============================================================================
# LAYOUT ANALYSIS PARAMETERS
# ============================================================================
NUM_CLUSTERS = 3  # Number of layout clusters (from Stage 2 analysis)
RANDOM_SEED = 42  # For reproducibility

# Layout features extracted per page (13 features)
LAYOUT_FEATURE_NAMES = [
    "n_boxes", "total_area", "mean_area", "std_area",
    "mean_width", "mean_height", "mean_aspect_ratio",
    "mean_x", "mean_y", "density", "vertical_spread",
    "horizontal_spread", "mean_spacing"
]

# ============================================================================
# EVALUATION PROTOCOL (Stage 0)
# ============================================================================
EVALUATION_PROTOCOL = {
    "text_matching": {
        "metric": "NED",  # Normalized Edit Distance
        "threshold": 0.2,
        "description": "NED < 0.2 means predicted text is 'close enough'"
    },
    "location_matching": {
        "metric": "IoU",  # Intersection over Union
        "threshold": 0.3,
        "description": "IoU > 0.3 means bounding boxes overlap sufficiently"
    },
    "overall_metric": {
        "metric": "F1",
        "components": ["precision", "recall"],
        "description": "F1 combines precision and recall for overall performance"
    }
}

# Text matching threshold
NED_THRESHOLD = 0.2

# Location matching threshold
IOU_THRESHOLD = 0.3

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
FIGURE_SIZE = (12, 8)
DPI = 300
COLORMAP = "viridis"

# ============================================================================
# STAGE 3-8 PARAMETERS (To be populated)
# ============================================================================
# Stage 3: Baseline definitions
BASELINES = {
    "paper_baseline": "TBD",  # From the paper
    "layoutlmv3_simple": "TBD",
    "layoutlmv3_linker": "TBD"
}

# Stage 4: LayoutLMv3 + Linker parameters
LAYOUTLMV3_CONFIG = {
    "model_name": "microsoft/layoutlmv3-base",
    "max_seq_length": 512,
    "learning_rate": 5e-5,
    "num_epochs": 3
}

# Stage 6: Robustness experiments
ROBUSTNESS_TESTS = {
    "rotation": [0, 5, 10, 15],  # degrees
    "occlusion": [0.0, 0.1, 0.2, 0.3],  # fraction of document
    "dropout": [0.0, 0.1, 0.2, 0.3]  # fraction of annotations
}

# ============================================================================
# DEBUG & LOGGING
# ============================================================================
VERBOSE = True
LOG_LEVEL = "INFO"

def print_config():
    """Print current configuration."""
    print("=" * 80)
    print("KVP10k EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Subset size: {SUBSET_N}")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Num clusters: {NUM_CLUSTERS}")
    print(f"NED threshold: {NED_THRESHOLD}")
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print("=" * 80)

if __name__ == "__main__":
    print_config()
