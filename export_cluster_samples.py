"""
Export reproducible raw-image samples from corrected Stage 2 test clusters.
"""

import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = REPO_ROOT / "code" / "script"
sys.path.insert(0, str(SCRIPT_DIR))
import config  # uses DATASET_NAME and KVP_CACHE from your repo

MAP_PATH = REPO_ROOT / "data" / "outputs" / "stage2" / "test_cluster_map.json"
OUTPUT_DIR = REPO_ROOT / "output" / "cluster_samples"
SPLIT = "test"
N_PER_GROUP = 50
SEED = 42
GROUPS = ("Cluster_0", "Cluster_1", "Geometry_Unavailable")


def save_image(value, destination: Path) -> None:
    if isinstance(value, Image.Image):
        image = value
    elif isinstance(value, dict) and value.get("bytes") is not None:
        image = Image.open(io.BytesIO(value["bytes"]))
    elif isinstance(value, dict) and value.get("path"):
        image = Image.open(value["path"])
    else:
        raise TypeError(f"Unsupported image value: {type(value)}")

    image.convert("RGB").save(destination, quality=95)


def download_pdf(url: str) -> bytes:
    response = requests.get(url, timeout=60, allow_redirects=True)
    response.raise_for_status()
    if len(response.content) < 200 or response.content[:5] != b"%PDF-":
        raise ValueError(f"Downloaded content from {url} is not a PDF")
    return response.content


def render_pdf_page_to_image(pdf_bytes: bytes, page_number: int, destination: Path) -> None:
    import fitz

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_number < 0 or page_number >= len(document):
            raise ValueError(f"Page {page_number + 1} is out of range for this PDF")
        page = document[page_number]
        zoom = 150 / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        image.convert("RGB").save(destination, quality=95)
    finally:
        document.close()


def main() -> None:
    if not MAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MAP_PATH}. First run code/script/build_test_cluster_map.py."
        )

    with MAP_PATH.open(encoding="utf-8") as handle:
        cluster_map = json.load(handle)

    grouped_hashes = defaultdict(list)
    for hash_name, record in cluster_map.items():
        group = record["cluster"]
        if group in GROUPS:
            grouped_hashes[group].append(hash_name)

    selected = {}
    for group in GROUPS:
        candidates = sorted(grouped_hashes[group])
        rng = random.Random(f"{SEED}:{group}")
        selected[group] = rng.sample(
            candidates,
            k=min(N_PER_GROUP, len(candidates)),
        )

    dataset = load_dataset(
        config.DATASET_NAME,
        split=SPLIT,
        cache_dir=config.KVP_CACHE,
    )

    index_by_hash = {}
    for index, row in enumerate(dataset):
        index_by_hash.setdefault(row["hash_name"], index)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for group in GROUPS:
        group_dir = OUTPUT_DIR / group
        group_dir.mkdir(exist_ok=True)
        for number, hash_name in enumerate(selected[group], start=1):
            if hash_name not in index_by_hash:
                raise KeyError(f"{hash_name} is absent from raw {SPLIT} data")
            row = dataset[index_by_hash[hash_name]]
            image_path = group_dir / f"{number:02d}_{hash_name}.jpg"
            if row.get("image") is not None:
                save_image(row["image"], image_path)
            else:
                image_url = row.get("image_url")
                if not image_url:
                    raise KeyError(f"{hash_name} has neither an embedded image nor an image_url")
                page_number = max(0, int(row.get("page_number", 1)) - 1)
                try:
                    pdf_bytes = download_pdf(image_url)
                    render_pdf_page_to_image(pdf_bytes, page_number, image_path)
                    exported = True
                except Exception as exc:
                    exported = False
                    image_path = None
                    manifest.append(
                        {
                            "group": group,
                            "sample_number": number,
                            "hash_name": hash_name,
                            "n_boxes": cluster_map[hash_name].get("n_boxes"),
                            "image_file": None,
                            "error": str(exc),
                        }
                    )
                    continue
            if image_path is not None:
                manifest.append(
                    {
                        "group": group,
                        "sample_number": number,
                        "hash_name": hash_name,
                        "n_boxes": cluster_map[hash_name].get("n_boxes"),
                        "image_file": str(image_path.relative_to(REPO_ROOT)),
                    }
                )

    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": SPLIT,
                "cluster_map": str(MAP_PATH.relative_to(REPO_ROOT)),
                "random_seed": SEED,
                "requested_per_group": N_PER_GROUP,
                "groups": list(GROUPS),
                "samples": manifest,
            },
            handle,
            indent=2,
        )

    print(f"Saved {len(manifest)} images to {OUTPUT_DIR}")
    for group in GROUPS:
        print(f"  {group}: {len(selected[group])} images")


if __name__ == "__main__":
    main()