"""Quick verification of V4 link label fixes — full pipeline."""
import json, torch, sys
sys.path.insert(0, "code/script")
from stage4_kvp_dataset import LayoutLMv3PreparedDataset
from layoutlm_model_v2 import group_contiguous_spans, collapse_link_labels_to_spans

dataset = LayoutLMv3PreparedDataset("data/prepared", "test")

total_gt = 0
total_word_pos = 0
total_span_pos = 0
total_tok_pos = 0
n = min(20, len(dataset))

for docidx in range(n):
    jf = dataset.json_files[docidx]
    with open(jf) as f:
        data = json.load(f)
    lmdx = data.get("lmdx_text", "")
    words, bboxes = dataset._parse_lmdx_text(lmdx, data)
    kvps = data.get("gt_kvps", {}).get("kvps_list", [])
    entity_labels, word_link = dataset._generate_labels(
        words, kvps, bboxes, data.get("image_width", 1), data.get("image_height", 1)
    )
    word_link_pos = int(word_link.sum().item())
    raw_kvps = sum(1 for k in kvps if k.get("type") == "kvp")

    # Get token-level item
    item = dataset[docidx]
    tok_link = item["link_labels"]
    tok_pos = int(tok_link.sum().item())
    
    # Span-level collapse
    key_spans = group_contiguous_spans(item["entity_labels"], item["attention_mask"], item["bbox"], 1)
    val_spans = group_contiguous_spans(item["entity_labels"], item["attention_mask"], item["bbox"], 2)
    nk, nv = len(key_spans), len(val_spans)
    if nk > 0 and nv > 0:
        span_gt = collapse_link_labels_to_spans(tok_link, key_spans, val_spans)
        span_pos = int(span_gt.sum().item())
    else:
        span_pos = 0

    total_gt += raw_kvps
    total_word_pos += word_link_pos
    total_tok_pos += tok_pos
    total_span_pos += span_pos
    print(f"Doc {docidx:2d}: words={len(words):3d}  GT={raw_kvps:2d}  "
          f"wrd_link={word_link_pos:2d}  tok_link={tok_pos:4d}  "
          f"key_spans={nk:2d}  val_spans={nv:2d}  span_link={span_pos:2d}")

print(f"\nTotals over {n} docs:")
print(f"  GT KVPs:          {total_gt}")
print(f"  Word-level links: {total_word_pos} ({total_word_pos/total_gt*100:.0f}%)")
print(f"  Token-level links:{total_tok_pos}")
print(f"  Span-level links: {total_span_pos} ({total_span_pos/max(1,total_gt)*100:.0f}%)")
