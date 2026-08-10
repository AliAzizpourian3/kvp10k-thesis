#!/usr/bin/env python
"""Create the Stage 2 qualitative figure with KVP10k polygon overlays.

Run from the repository root:
  HF_DATASETS_OFFLINE=1 env/kvp10k_env/bin/python create_stage2_qualitative_figure.py
"""

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = REPO_ROOT / "code" / "script"
sys.path.insert(0, str(SCRIPT_DIR))
import config

FIGURES_DIR = REPO_ROOT / "LaTeX_Thesis" / "figures" / "stage2"
OUTPUT_FIGURE = FIGURES_DIR / "stage2_cluster_examples.png"
SAMPLES = {
    "Cluster 0: Higher box count, broad spread": "7e159acbb59982fef482b344ce20923f1747602c",
    "Cluster 1: Lower box count, compact spread": "7407f44e6736c449d792ab68b90c531dd6a8df",
}


def image_from_value(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and value.get("bytes") is not None:
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, dict) and value.get("path"):
        return Image.open(value["path"])).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)}")


def polygon_lists(value):
    """Return every coordinate polygon found in nested KVP10k annotation data."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"polygon", "polygons", "points", "vertices"}:
                found.extend(polygon_lists(item))
            else:
                found.extend(polygon_lists(item))
    elif isinstance(value, (list, tuple)):
        if len(value) >= 6 and all(isinstance(x, (int, float)) for x in value):
            found.append(list(value))
        elif len(value) >= 3 and all(isinstance(x, (list, tuple)) and len(x) >= 2 and isinstance(x[0], (int, float)) and isinstance(x[1], (int, float)) for x in value):
            found.append([coordinate for point in value for coordinate in point[:2]])
        else:
            for item in value:
                found.extend(polygon_lists(item))
    return found


def usable_polygons(row):
    candidates = polygon_lists(row.get("annotations", row))
    unique = []
    seen = set()
    for polygon in candidates:
        if len(polygon) < 6 or len(polygon) % 2:
            continue
        key = tuple(float(x) for x in polygon)
        if key not in seen:
            seen.add(key)
            unique.append(polygon)
    return unique


def select_richest_copy(dataset, hash_prefix):
    matches = [row for row in dataset if row["hash_name"].startswith(hash_prefix)]
    if not matches:
        raise KeyError(f"No test row begins with {hash_prefix}")
    return max(matches, key=lambda row: len(usable_polygons(row)))


def scale_polygon(polygon, width, height):
    xs = polygon[0::2]
    ys = polygon[1::2]
    # KVP10k stores either pixel coordinates or 0--1000 normalized coordinates.
    if max(abs(x) for x in xs) <= 1000 and max(abs(y) for y in ys) <= 1000:
        return [(x * width / 1000, y * height / 1000) for x, y in zip(xs, ys)]
    return list(zip(xs, ys))


def draw_polygons(image, polygons):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        points = scale_polygon(polygon, *canvas.size)
        if len(points) >= 3:
            draw.line(points + [points[0]], fill=(220, 35, 35), width=max(2, round(min(canvas.size) / 500)))
    return canvas


def main():
    dataset = load_dataset(config.DATASET_NAME, split="test", cache_dir=config.KVP_CACHE)
    panels = []
    for title, hash_prefix in SAMPLES.items():
        row = select_richest_copy(dataset, hash_prefix)
        polygons = usable_polygons(row)
        if not polygons:
            raise RuntimeError(f"No usable polygons found for {row['hash_name']}")
        panels.append((title, row["hash_name"], draw_polygons(image_from_value(row["image"]), polygons), len(polygons)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, (title, hash_name, image, count) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(f"{title}\n{count} annotation polygons", fontsize=10, fontweight="bold")
        axis.axis("off")
        print(f"{title}: {hash_name} ({count} polygons)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
