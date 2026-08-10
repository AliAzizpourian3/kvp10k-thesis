#!/usr/bin/env python
"""Create the Stage 2 qualitative figure with KVP10k annotation overlays.

Run from the repository root:
  HF_DATASETS_OFFLINE=1 env/kvp10k_env/bin/python create_stage2_qualitative_figure.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = REPO_ROOT / "code" / "script"
sys.path.insert(0, str(SCRIPT_DIR))
import config
from features import extract_layout_features

FIGURES_DIR = REPO_ROOT / "LaTeX_Thesis" / "figures" / "stage2"
OUTPUT_FIGURE = FIGURES_DIR / "stage2_cluster_examples.png"
SAMPLES = {
    "Cluster 0: Higher box count, broad spread": {
        "hash_prefix": "7e159acbb59982fef482b344ce20923f1747602c",
        "image": REPO_ROOT / "output" / "cluster_samples" / "Cluster_0" / "08_7e159acbb59982fef482b344ce20923f1747602c71618071cb59e00c3611a667.jpg",
    },
    "Cluster 1: Lower box count, compact spread": {
        "hash_prefix": "7407f44e6736c449d792ab68b90c531dd6a8df",
        "image": REPO_ROOT / "output" / "cluster_samples" / "Cluster_1" / "19_7407f44e6736c449d792ab68b90c531dd6a8df5712627137f06b5bb1473a1979.jpg",
    },
}


def select_richest_copy(dataset, hash_prefix):
    matches = [row for row in dataset if row["hash_name"].startswith(hash_prefix)]
    if not matches:
        raise KeyError(f"No test row begins with {hash_prefix}")
    return max(matches, key=lambda row: float(extract_layout_features(row)[0]))


def _numbers(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_points(value):
    """Convert common KVP10k coordinate encodings to a list of (x, y) points."""
    if isinstance(value, dict):
        if "x" in value and "y" in value and _numbers(value["x"]) and _numbers(value["y"]):
            return [(float(value["x"]), float(value["y"]))]
        return []
    if not isinstance(value, (list, tuple)):
        return []
    if len(value) >= 3 and all(isinstance(point, dict) for point in value):
        points = [_to_points(point) for point in value]
        flattened = [point for group in points for point in group]
        return flattened if len(flattened) >= 3 else []
    if len(value) >= 3 and all(isinstance(point, (list, tuple)) and len(point) >= 2 and _numbers(point[0]) and _numbers(point[1]) for point in value):
        return [(float(point[0]), float(point[1])) for point in value]
    if len(value) >= 6 and len(value) % 2 == 0 and all(_numbers(item) for item in value):
        return [(float(value[index]), float(value[index + 1])) for index in range(0, len(value), 2)]
    return []


def extract_polygons(value):
    """Recursively find coordinate lists in an annotation record."""
    polygons = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = key.lower()
            if key_lower in {"polygon", "polygons", "coordinates", "points", "vertices"}:
                points = _to_points(item)
                if len(points) >= 3:
                    polygons.append(points)
                else:
                    polygons.extend(extract_polygons(item))
            else:
                polygons.extend(extract_polygons(item))
    elif isinstance(value, (list, tuple)):
        points = _to_points(value)
        if len(points) >= 3:
            polygons.append(points)
        else:
            for item in value:
                polygons.extend(extract_polygons(item))
    return polygons


def usable_polygons(row):
    polygons = extract_polygons(row.get("annotations", []))
    unique = []
    seen = set()
    for polygon in polygons:
        key = tuple((round(x, 8), round(y, 8)) for x, y in polygon)
        if key not in seen:
            seen.add(key)
            unique.append(polygon)
    return unique


def image_points(points, width, height):
    max_x = max(abs(x) for x, _ in points)
    max_y = max(abs(y) for _, y in points)
    if max_x <= 1.01 and max_y <= 1.01:
        return [(x * width, y * height) for x, y in points]
    if max_x <= 1000 and max_y <= 1000:
        return [(x * width / 1000, y * height / 1000) for x, y in points]
    return points


def draw_polygons(image, polygons):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(2, round(min(width, height) / 500))
    for polygon in polygons:
        points = image_points(polygon, width, height)
        draw.line(points + [points[0]], fill=(220, 35, 35), width=line_width)
    return canvas


def main():
    dataset = load_dataset(config.DATASET_NAME, split="test", cache_dir=config.KVP_CACHE)
    panels = []
    for title, sample in SAMPLES.items():
        image_path = sample["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing exported sample image: {image_path}")
        row = select_richest_copy(dataset, sample["hash_prefix"])
        polygons = usable_polygons(row)
        if not polygons:
            raise RuntimeError(
                f"No coordinate polygons found for {row['hash_name']}. "
                f"Annotation keys: {list(row.get('annotations', [{}])[0].keys()) if row.get('annotations') else []}"
            )
        image = Image.open(image_path).convert("RGB")
        panels.append((title, row["hash_name"], draw_polygons(image, polygons), len(polygons)))

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
