# Technical Deep Dive: lmdx_text Parsing & LayoutLMv3 Integration

## lmdx_text Format Specification

The `lmdx_text` field in prepared data contains layout-aware text with per-word coordinate information.

### Format Structure

```
word1 x1|y1|x2|y2
word2 x1|y1|x2|y2
...
wordN x1|y1|x2|y2
```

### Coordinate System

- **Space**: Pixel coordinates (absolute position on page)
- **Origin**: Top-left (0,0)
- **Range**: [0, image_width] × [0, image_height]
- **Type**: Integer values
- **Bounding Box**: (x1, y1) = top-left, (x2, y2) = bottom-right

### Example (Real Data)

```
Francis Marion University 33|6|69|8
Purchasing Office 43|9|60|10
PO Box 100547 44|11|58|12
Florence, SC 29502-0547 39|13|63|14
```

On a 2550 × 3300 page:
- "Francis Marion University" spans pixels (33-69, 6-8) → width=36px, height=2px
- "Purchasing Office" spans pixels (43-60, 9-10) → width=17px, height=1px
- etc.

## Parsing Pipeline

### Step 1: Extract Words and Pixel Coordinates

```python
def _parse_lmdx_text(self, lmdx_text: str, data: Dict):
    words = []
    bboxes = []
    
    for line in lmdx_text.split('\n'):
        # Parse: "word x1|y1|x2|y2"
        parts = line.rsplit(' ', 1)  # Split on last space
        word = parts[0]              # "Francis Marion University"
        coords = parts[1]            # "33|6|69|8"
        
        # Extract coordinates
        x1, y1, x2, y2 = map(int, coords.split('|'))
        
        words.append(word)
        bboxes.append([x1, y1, x2, y2])
    
    return words, bboxes
```

**Result**:
```python
words = ['Francis', 'Marion', 'University', ...]
# Actually: Should keep word as phrase! Bug fix needed
# Reality: words = ['Francis Marion University', ...]
bboxes = [[33, 6, 69, 8], [43, 9, 60, 10], ...]
```

### Step 2: Normalize to LayoutLMv3 Scale

LayoutLMv3 expects bounding boxes in normalized [0, 1000] range:

```python
def _normalize_bboxes(self, bboxes, image_width, image_height):
    normalized = []
    for x1, y1, x2, y2 in bboxes:
        x1_norm = int((x1 / image_width) * 1000)
        y1_norm = int((y1 / image_height) * 1000)
        x2_norm = int((x2 / image_width) * 1000)
        y2_norm = int((y2 / image_height) * 1000)
        normalized.append([x1_norm, y1_norm, x2_norm, y2_norm])
    return normalized
```

**Conversion Example** (page 2550 × 3300):
- Input: [33, 6, 69, 8]
- Normalized:
  - x1_norm = (33 / 2550) × 1000 = 12
  - y1_norm = (6 / 3300) × 1000 = 1
  - x2_norm = (69 / 2550) × 1000 = 27
  - y2_norm = (8 / 3300) × 1000 = 2
- Output: [12, 1, 27, 2]

### Step 3: Tokenization

LayoutLMv3Processor tokenizes words into subword tokens:

```python
encoded = processor(
    text=words,              # List of word strings
    boxes=bboxes_normalized, # [seq_len, 4] normalized bboxes
    return_tensors="pt",
    padding="max_length",
    max_length=512,
    truncation=True
)
```

**Output**:
```python
encoded['input_ids']     # [1, 512] - CLS + tokens + SEP + PAD
encoded['attention_mask'] # [1, 512] - 1 for real tokens, 0 for padding
encoded['bbox']          # [1, 512, 4] - bbox per token (propagated from words)
encoded.word_ids()       # [512] - maps token_idx to word_idx
```

**Token-to-Word Mapping Example**:
```
Word level:   "Francis"     "Marion"      "University"
Tokens:      ["F","ran"]    ["Mar","ion"] ["Uni","ver","sity"]
word_ids:    [  0,    0  ]  [ 1,    1  ] [ 2,     2,     2   ]
```

## Entity Label Generation

### Algorithm

1. Parse ground truth KVPs from `gt_kvps`
2. For each KVP, find word spans:
   - Find words containing key text
   - Find words containing value text
3. Mark found words with labels: 1=Key, 2=Value

### Example

**Ground Truth**:
```python
{
    "type": "kvp",
    "key": {"text": "Posting Date:", "bbox": [39, 21, 52, 23]},
    "value": {"text": "11/02/2018", "bbox": [53, 21, 63, 23]},
    "center_x": 45,
    "center_y": 22
}
```

**Word-Level Matching**:
```python
words = ["Posting", "Date:", "11/02/2018", ...]
#         word_0    word_1     word_2

entity_labels = [1, 1, 2, 0, 0, ...]
#                 ^^^^  ^
#               Key-words  Value word
```

**Token-Level Alignment** (via word_ids):
```python
Tokens:        [CLS] [Post] [ing] [Date] [:] [11] [/] [02] [/] [2018] [SEP] [PAD]...
word_ids:      [None] [  0  ] [ 0 ] [  1  ] [1]  [ 2 ] [2] [ 2 ] [2]  [2]  [None]...
entity_labels: [  0 ] [  1  ] [ 1 ] [  1  ] [1]  [ 2 ] [2] [ 2 ] [2]  [2]  [  0 ]...
```

## Link Label Generation

### Algorithm

Link labels encode key-value pair relationships in a [seq_len × seq_len] adjacency matrix:

```python
link_labels = torch.zeros((seq_len, seq_len), dtype=torch.float32)

for kvp in kvps:
    key_indices = find_word_span(words, kvp['key']['text'])
    val_indices = find_word_span(words, kvp['value']['text'])
    
    # Connect each key word to each value word
    for key_idx in key_indices:
        for val_idx in val_indices:
            link_labels[key_idx, val_idx] = 1.0
```

### Interpretation

```python
link_labels[i, j] = 1.0  # Word i (key) is linked to word j (value)
link_labels[i, j] = 0.0  # No relationship
```

### Example (with 3 KVPs)

```
KVP 1: key="Posting Date:"  value="11/02/2018"
KVP 2: key="Company:"       value="Francis Marion University"
KVP 3: key="Address:"       value="PO Box 100547"

Words: ["Posting", "Date:", "11/02/2018", "Company:", "Francis", "Marion", "University", ...]
Index:    0        1         2             3           4         5         6          ...

Link labels[0, 2] = 1  # "Posting" → "11/02/2018"
Link labels[1, 2] = 1  # "Date:" → "11/02/2018"
Link labels[3, 4] = 1  # "Company:" → "Francis"
Link labels[3, 5] = 1  # "Company:" → "Marion"
Link labels[3, 6] = 1  # "Company:" → "University"
...
```

## Data Flow Visualization

```
┌─────────────────────────────┐
│ Raw JSON: data/prepared/... │
│ - lmdx_text                 │
│ - image_width/height        │
│ - gt_kvps[]                 │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Parse lmdx_text             │
│ words[], bboxes[]           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Normalize Bboxes            │
│ [0, image_w] → [0, 1000]    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Generate Labels             │
│ entity_labels[]             │
│ link_labels[][]             │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ LayoutLMv3Processor         │
│ Tokenize + Align            │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Output Tensors              │
│ input_ids[512]              │
│ attention_mask[512]         │
│ bbox[512, 4]                │
│ entity_labels[512]          │
│ link_labels[512, 512]       │
│ pixel_values[3, 224, 224]   │
└─────────────────────────────┘
```

## Handling Edge Cases

### 1. Multi-Token Words

Some words tokenize into multiple subword tokens:
```python
# Input: ["University"]
# After tokenization: ["Uni", "ver", "sity"]

# word_ids: [0, 0, 0] - all map back to word_0
# Alignment propagates entity label to all tokens
```

### 2. Special Tokens

```python
# Tokens: [CLS] ... [SEP] [PAD] ...
# word_ids: [None] ... [None] [None] ...
# entity_labels: [0] (default "Other" for special tokens)
```

### 3. Truncation

If sequence exceeds max_length (512):
```python
# Input words: 700 (too long)
# After truncation: 510 words + [CLS] + [SEP]
# Truncated link_labels will be [512, 512] padded with zeros
```

### 4. Missing Ground Truth

If no KVPs or parsing fails:
```python
entity_labels = [0, 0, 0, ...]  # All "Other"
link_labels = zeros([512, 512]) # No relationships
# But training continues (graceful degradation)
```

## LayoutLMv3 Model Interface

```python
class LayoutLMv3KVPModel(nn.Module):
    def forward(
        self,
        input_ids,           # [batch, seq_len]
        attention_mask,      # [batch, seq_len]
        bbox,                # [batch, seq_len, 4]
        pixel_values=None,   # [batch, 3, 224, 224]
        entity_labels=None,  # [batch, seq_len] - for entity classification
        link_labels=None,    # [batch, seq_len, seq_len] - for relation extraction
        use_linker=False     # Whether to include relation extractor
    ):
        # Returns:
        # - entity_logits: [batch, seq_len, 3] for O/K/V classification
        # - link_logits: [batch, seq_len, seq_len] for relationships (if use_linker=True)
```

## Performance Considerations

1. **Memory**: link_labels adds quadratic memory (512×512=262K floats per sample)
2. **Speed**: Link label computation is negligible compared to tokenization
3. **Batching**: Custom collator ensures proper tensor alignment
4. **Caching**: HuggingFace cache directory for model weights (~/hf_cache/)

## References

- **LayoutLMv3 Paper**: [LayoutLMv3: Pre-training for Document Analysis with Unified Text and Image Masking](https://arxiv.org/abs/2212.08290)
- **KVP10k Dataset**: [KVP10k: A Large-Scale Document Key-Value Pair Extraction Dataset](https://arxiv.org/abs/2003.06965)
- **Generated Format**: lmdx_text from `prepare_data.py`
