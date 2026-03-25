"""
Data preparation for KVP10k Mistral-7B baseline.

Replicates IBM's pipeline (download_dataset.py → dataset.py):
  1. Download PDF from image_url (HuggingFace dataset)
  2. Render page_number at 300 DPI
  3. Extract word-level text + bounding boxes (PyMuPDF native text layer)
  4. Fuse extracted words with annotation coordinates → ground truth KVPs
  5. Create LMDX-format prompt and target for Mistral training

NOTE: IBM uses Tesseract OCR; we use PyMuPDF native text extraction.
  - Advantage: No system dependency (tesseract-ocr not installed on cluster)
  - Caveat:  Scanned PDFs without text layers produce no words (handled gracefully)

Output per sample:  data/prepared/{split}/{hash_name}.json
  Keys: hash_name, image_width, image_height, num_words, num_kvps,
        lmdx_text, target_text, full_prompt, gt_kvps

Usage:
  python prepare_data.py --split test  --workers 8
  python prepare_data.py --split train --workers 16
  python prepare_data.py --split test  --workers 4 --limit 50   # quick test
"""

import os
import sys
import json
import logging
import argparse
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (matching IBM)
# ---------------------------------------------------------------------------
QUANTIZED_W = 100
QUANTIZED_H = 100
WORD_MATCH_THRESHOLD = 0.6  # from IBM's OcrAnnotationFusion
PDF_DPI = 300
DOWNLOAD_TIMEOUT = 60  # seconds per PDF

LMDX_PROMPT_TEMPLATE = """\
<Document>
{lmdx_text}
</Document>
<Task>
From the document, extract the text keys and values.
Please provide the response in the form of a Python list of lists.
Each inner list should contain exactly two strings: the first string is the key and the second string is the value.
Each key and value should be followed by its bounding box in the format: left|top|right|bottom.
</Task>
### Response:
"""

# ---------------------------------------------------------------------------
# PDF download & text extraction
# ---------------------------------------------------------------------------

def download_pdf(url: str) -> Optional[bytes]:
    """Download a PDF, return bytes or None."""
    try:
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        # Validate it's actually a PDF (not an HTML error page)
        if len(resp.content) < 200 or resp.content[:5] != b"%PDF-":
            return None
        return resp.content
    except Exception:
        return None


def extract_words_pymupdf(
    pdf_bytes: bytes, page_number: int
) -> Tuple[Optional[List[Dict]], Optional[Tuple[int, int]]]:
    """
    Extract word-level text and bboxes via PyMuPDF's native text layer.

    Returns (words, (width_px, height_px)) or (None, None).
    word format: {'text': str, 'bbox': [left, top, right, bottom]}  (pixels at PDF_DPI)
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF (fitz) is required: pip install PyMuPDF")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if page_number < 0 or page_number >= len(doc):
            doc.close()
            return None, None

        page = doc[page_number]
        zoom = PDF_DPI / 72.0
        width_px = int(page.rect.width * zoom)
        height_px = int(page.rect.height * zoom)

        # (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        raw = page.get_text("words")
        doc.close()

        if not raw:
            return None, (width_px, height_px)

        words = []
        for w in raw:
            text = w[4].strip()
            if not text:
                continue
            words.append({
                "text": text,
                "bbox": [
                    int(w[0] * zoom),
                    int(w[1] * zoom),
                    int(w[2] * zoom),
                    int(w[3] * zoom),
                ],
            })
        return words if words else None, (width_px, height_px)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# OCR-annotation fusion  (simplified IBM OcrAnnotationFusion)
# ---------------------------------------------------------------------------

def _pct_inside(word_bbox: List[int], ann_rect: Dict) -> float:
    """Fraction of word bbox area that lies inside ann_rect."""
    xo = max(0, min(word_bbox[2], ann_rect["right"]) - max(word_bbox[0], ann_rect["left"]))
    yo = max(0, min(word_bbox[3], ann_rect["bottom"]) - max(word_bbox[1], ann_rect["top"]))
    area = (word_bbox[2] - word_bbox[0]) * (word_bbox[3] - word_bbox[1])
    return (xo * yo) / area if area > 0 else 0.0


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def fuse_ocr_with_annotations(
    words: List[Dict],
    annotations: List[Dict],
    img_w: int,
    img_h: int,
) -> Dict:
    """
    Match OCR words to annotation bboxes, create ground-truth KVPs.
    Returns {'kvps_list': [...]}.
    """
    wr = QUANTIZED_W / img_w
    hr = QUANTIZED_H / img_h

    entities: Dict[str, Dict] = {}

    for ann in annotations:
        ann_id = ann.get("_id", "")
        if not ann_id:
            continue

        label = (ann.get("label") or "").strip()

        # Rectangle from 'coordinates' (list of {x,y}) or 'points' (dict of {label,x,y})
        coords = ann.get("coordinates")
        points = ann.get("points")
        xs, ys = [], []

        if coords and isinstance(coords, list):
            for c in coords:
                if isinstance(c, dict) and "x" in c and "y" in c:
                    xs.append(c["x"] * img_w)
                    ys.append(c["y"] * img_h)
        elif points and isinstance(points, dict):
            for p in points.values():
                if isinstance(p, dict) and "x" in p and "y" in p:
                    xs.append(p["x"] * img_w)
                    ys.append(p["y"] * img_h)

        if len(xs) < 2 or len(ys) < 2:
            continue

        rect = {"left": min(xs), "top": min(ys), "right": max(xs), "bottom": max(ys)}

        # Match OCR words inside this annotation bbox
        matched = [w for w in words if _pct_inside(w["bbox"], rect) >= WORD_MATCH_THRESHOLD]
        matched.sort(key=lambda w: (w["bbox"][1], w["bbox"][0]))
        text = " ".join(w["text"] for w in matched)

        # Quantized bbox & centre
        q_bbox = [
            _clamp(round(rect["left"]   * wr), 0, QUANTIZED_W),
            _clamp(round(rect["top"]    * hr), 0, QUANTIZED_H),
            _clamp(round(rect["right"]  * wr), 0, QUANTIZED_W),
            _clamp(round(rect["bottom"] * hr), 0, QUANTIZED_H),
        ]
        cx = round(((rect["right"] + rect["left"]) / 2) * wr)
        cy = round(((rect["bottom"] + rect["top"]) / 2) * hr)

        # Linking
        attrs = ann.get("attributes") or {}
        linking = None
        if isinstance(attrs, dict):
            lf = attrs.get("Linking")
            if isinstance(lf, dict):
                v = lf.get("value")
                if v and v != "NA":
                    linking = v

        entities[ann_id] = {
            "id": ann_id,
            "label": label,
            "text": text,
            "q_bbox": q_bbox,
            "center_x": cx,
            "center_y": cy,
            "linking": linking,
        }

    # ---- build KVP list ----
    kvps: List[Dict] = []
    used: set = set()

    # Regular KVPs  (value annotation → links to key annotation)
    for eid, ent in entities.items():
        link = ent.get("linking")
        if not link or eid in used:
            continue
        key_ent = entities.get(link)
        if key_ent is None:
            for k, v in entities.items():
                if k.startswith(link) or link.startswith(k):
                    key_ent = v
                    break
        if key_ent:
            kvps.append({
                "type": "kvp",
                "key":   {"text": key_ent["text"], "bbox": key_ent["q_bbox"]},
                "value": {"text": ent["text"],     "bbox": ent["q_bbox"]},
                "center_x": key_ent["center_x"],
                "center_y": key_ent["center_y"],
            })
            used.add(eid)
            used.add(key_ent["id"])

    # Floating types → unkeyed KVP
    for eid, ent in entities.items():
        if eid in used:
            continue
        if "floating" in ent["label"].lower():
            key_label = ent["label"].lower().replace("floating", "").strip()
            kvps.append({
                "type": "unkeyed",
                "key":   {"text": key_label},
                "value": {"text": ent["text"], "bbox": ent["q_bbox"]},
                "center_x": ent["center_x"],
                "center_y": ent["center_y"],
            })
            used.add(eid)

    # Unvalued keys
    for eid, ent in entities.items():
        if eid in used:
            continue
        if ent["label"].lower() == "unvalued_key":
            kvps.append({
                "type": "unvalued",
                "key": {"text": ent["text"], "bbox": ent["q_bbox"]},
                "center_x": ent["center_x"],
                "center_y": ent["center_y"],
            })
            used.add(eid)

    return {"kvps_list": kvps}


# ---------------------------------------------------------------------------
# LMDX prompt creation
# ---------------------------------------------------------------------------

def create_lmdx_text(words: List[Dict], img_w: int, img_h: int) -> str:
    """
    Group OCR words into lines, format as LMDX:
      sentence_text left|top|right|bottom   (quantised to 100x100)
    """
    if not words:
        return ""

    wr = QUANTIZED_W / img_w
    hr = QUANTIZED_H / img_h

    sorted_words = sorted(words, key=lambda w: (w["bbox"][1], w["bbox"][0]))

    lines: List[List[Dict]] = [[sorted_words[0]]]
    for w in sorted_words[1:]:
        prev = lines[-1][-1]
        prev_cy = (prev["bbox"][1] + prev["bbox"][3]) / 2.0
        curr_cy = (w["bbox"][1] + w["bbox"][3]) / 2.0
        avg_h = ((prev["bbox"][3] - prev["bbox"][1]) + (w["bbox"][3] - w["bbox"][1])) / 2.0
        if avg_h > 0 and abs(curr_cy - prev_cy) < avg_h * 0.6:
            lines[-1].append(w)
        else:
            lines.append([w])

    parts = []
    for line in lines:
        line.sort(key=lambda w: w["bbox"][0])
        text  = " ".join(w["text"] for w in line)
        left  = min(w["bbox"][0] for w in line)
        top   = min(w["bbox"][1] for w in line)
        right = max(w["bbox"][2] for w in line)
        bot   = max(w["bbox"][3] for w in line)
        ql = _clamp(round(left  * wr), 0, QUANTIZED_W)
        qt = _clamp(round(top   * hr), 0, QUANTIZED_H)
        qr = _clamp(round(right * wr), 0, QUANTIZED_W)
        qb = _clamp(round(bot   * hr), 0, QUANTIZED_H)
        parts.append(f"{text} {ql}|{qt}|{qr}|{qb}")

    return "\n".join(parts)


def create_target_text(gt: Dict) -> str:
    """Format GT KVPs as IBM target string (sorted by centre_y, centre_x)."""
    kvps = gt.get("kvps_list", [])
    if not kvps:
        return "[]"

    kvps_sorted = sorted(kvps, key=lambda k: (k.get("center_y", 0), k.get("center_x", 0)))

    def _fmt(ent: Dict) -> str:
        t = ent.get("text", "")
        if "bbox" in ent:
            b = ent["bbox"]
            return f"{t} {b[0]}|{b[1]}|{b[2]}|{b[3]}"
        return t

    result = []
    for kvp in kvps_sorted:
        if kvp.get("type") == "unvalued":
            result.append([_fmt(kvp["key"])])
        else:
            result.append([_fmt(kvp["key"]), _fmt(kvp.get("value", {}))])

    return str(result)


# ---------------------------------------------------------------------------
# Process one sample
# ---------------------------------------------------------------------------

def process_sample(sample: Dict, idx: int) -> Optional[Dict]:
    """Download PDF, extract text, fuse annotations, build LMDX prompt."""
    hash_name   = sample.get("hash_name", f"sample_{idx}")
    url         = sample.get("image_url", "")
    page_num    = sample.get("page_number", 1)  # 1-indexed in KVP10k
    page_idx    = max(0, page_num - 1)            # 0-indexed for PyMuPDF
    annotations = sample.get("annotations", [])

    if not url:
        return None

    pdf_bytes = download_pdf(url)
    if pdf_bytes is None:
        return None

    words, dims = extract_words_pymupdf(pdf_bytes, page_idx)
    if words is None or dims is None or not words:
        return None

    img_w, img_h = dims
    gt   = fuse_ocr_with_annotations(words, annotations, img_w, img_h)
    lmdx = create_lmdx_text(words, img_w, img_h)
    tgt  = create_target_text(gt)
    prompt = LMDX_PROMPT_TEMPLATE.format(lmdx_text=lmdx)

    return {
        "hash_name":    hash_name,
        "image_width":  img_w,
        "image_height": img_h,
        "num_words":    len(words),
        "num_kvps":     len(gt.get("kvps_list", [])),
        "lmdx_text":    lmdx,
        "target_text":  tgt,
        "full_prompt":  prompt,
        "gt_kvps":      gt,
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_split(
    split: str,
    output_dir: str,
    workers: int = 4,
    limit: Optional[int] = None,
):
    """Process an entire HuggingFace split and save per-sample JSONs.

    The KVP10k dataset contains ~5 annotator copies per page (same hash_name,
    same page, different annotations).  We group by hash_name, download the
    PDF only once, and use the annotation set that yields the most KVPs.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    from datasets import load_dataset

    logger.info(f"Loading KVP10k '{split}' split …")
    ds = load_dataset(config.DATASET_NAME, split=split, cache_dir=config.KVP_CACHE)

    # --- group samples by hash_name ---
    from collections import defaultdict
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for i in range(len(ds)):
        s = ds[i]
        groups[s["hash_name"]].append(s)

    all_hashes = list(groups.keys())
    n = min(limit, len(all_hashes)) if limit else len(all_hashes)
    all_hashes = all_hashes[:n]
    logger.info(
        f"{len(ds)} rows → {len(groups)} unique pages.  Processing {n} pages  (workers={workers})"
    )

    out_path = Path(output_dir) / split
    out_path.mkdir(parents=True, exist_ok=True)

    success = failed = no_kvps = 0

    def _process_group(hash_name: str) -> Optional[Dict]:
        """Download PDF once, try all annotator sets, keep best."""
        copies = groups[hash_name]
        url = copies[0].get("image_url", "")
        page_num = copies[0].get("page_number", 1)
        page_idx = max(0, page_num - 1)

        if not url:
            return None

        pdf_bytes = download_pdf(url)
        if pdf_bytes is None:
            return None

        words, dims = extract_words_pymupdf(pdf_bytes, page_idx)
        if words is None or dims is None or not words:
            return None

        img_w, img_h = dims

        # Try each annotator's annotations and keep the richest GT
        best_result = None
        best_kvps = -1
        for copy in copies:
            annotations = copy.get("annotations", [])
            if not annotations:
                continue
            gt = fuse_ocr_with_annotations(words, annotations, img_w, img_h)
            n_kvps = len(gt.get("kvps_list", []))
            if n_kvps > best_kvps:
                best_kvps = n_kvps
                best_result = gt

        if best_result is None:
            best_result = {"kvps_list": []}

        lmdx = create_lmdx_text(words, img_w, img_h)
        tgt = create_target_text(best_result)
        prompt = LMDX_PROMPT_TEMPLATE.format(lmdx_text=lmdx)

        return {
            "hash_name": hash_name,
            "image_width": img_w,
            "image_height": img_h,
            "num_words": len(words),
            "num_kvps": len(best_result.get("kvps_list", [])),
            "lmdx_text": lmdx,
            "target_text": tgt,
            "full_prompt": prompt,
            "gt_kvps": best_result,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_process_group, h): h for h in all_hashes}
        with tqdm(total=n, desc=split) as pbar:
            for fut in as_completed(futs):
                try:
                    result = fut.result()
                except Exception:
                    result = None

                if result is None:
                    failed += 1
                else:
                    fp = out_path / f"{result['hash_name']}.json"
                    with open(fp, "w") as fh:
                        json.dump(result, fh)
                    success += 1
                    if result["num_kvps"] == 0:
                        no_kvps += 1
                pbar.update(1)

    logger.info(
        f"\n{'='*60}\n"
        f"  Split:     {split}\n"
        f"  Pages:     {n}\n"
        f"  Success:   {success}  ({100*success/max(n,1):.1f}%)\n"
        f"  Failed:    {failed}  ({100*failed/max(n,1):.1f}%)\n"
        f"  No KVPs:   {no_kvps}\n"
        f"  Output:    {out_path}\n"
        f"{'='*60}"
    )
    return {"total": n, "success": success, "failed": failed}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare KVP10k data for Mistral training")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument(
        "--output_dir", default=None,
        help="Root output directory (default: ../../data/prepared relative to this script)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Process only N samples (for testing)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent.parent.parent / "data" / "prepared")

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for s in splits:
        process_split(s, args.output_dir, args.workers, args.limit)


if __name__ == "__main__":
    main()
