#!/usr/bin/env python
"""
Create a two-panel qualitative figure for Stage 2 annotation-geometry clusters.

Combines one representative Cluster 0 page and one Cluster 1 page side-by-side.
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "output" / "cluster_samples"
FIGURES_DIR = REPO_ROOT / "LaTeX_Thesis" / "figures" / "stage2"

# Preferred sample pages (already exported)
CLUSTER_0_SAMPLE = OUTPUT_DIR / "Cluster_0" / "08_7e159acbb59982fef482b344ce20923f1747602c71618071cb59e00c3611a667.jpg"
CLUSTER_1_SAMPLE = OUTPUT_DIR / "Cluster_1" / "19_7407f44e6736c449d792ab68b90c531dd6a8df5712627137f06b5bb1473a1979.jpg"

OUTPUT_FIGURE = FIGURES_DIR / "stage2_cluster_examples.png"


def main():
    if not CLUSTER_0_SAMPLE.exists():
        raise FileNotFoundError(f"Cluster 0 sample not found: {CLUSTER_0_SAMPLE}")
    if not CLUSTER_1_SAMPLE.exists():
        raise FileNotFoundError(f"Cluster 1 sample not found: {CLUSTER_1_SAMPLE}")

    # Load images
    img0 = Image.open(CLUSTER_0_SAMPLE).convert("RGB")
    img1 = Image.open(CLUSTER_1_SAMPLE).convert("RGB")

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img0)
    axes[0].set_title("(a) Cluster 0: Higher box count, broad spread", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(img1)
    axes[1].set_title("(b) Cluster 1: Lower box count, compact spread", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    fig.tight_layout()

    # Save figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
