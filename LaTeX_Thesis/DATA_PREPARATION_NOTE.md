# Data Preparation Note

Date: 2026-03-08

## Why Data Preparation Was Needed

The raw KVP10k dataset was not directly ready for training the Stage 3 Mistral baseline.

The raw rows contain:
- annotations
- a PDF URL
- a page number
- a page identifier (`hash_name`)

But they do not directly give a final model-ready training sample in the form needed by the Mistral pipeline.

So the data had to be prepared into page-level JSON files containing:
- extracted document text
- bounding boxes
- reconstructed key-value supervision
- model prompt text
- model target text

## Important Counts

The raw dataset contains repeated annotator copies for the same page.

Real counts observed during preparation:
- test split: 5255 raw rows became 1051 unique pages
- train split: 48280 raw rows became 9656 unique pages

Final preparation results:
- test split: 581 prepared pages kept out of 1051 unique pages (55.3\%)
- among those prepared test pages, 532 have non-empty KVP supervision
- train split: 5389 prepared pages kept out of 9656 unique pages (55.8\%)
- among those prepared train pages, 404 have zero KVPs after preparation

The main loss source is broken, inaccessible, or unusable source PDFs, not random filtering.

## Short Definitions

### What annotation means

An annotation is a human-made label on the document.

Example:
- this rectangle is the key `Orders due:`
- this rectangle is the value `November 5, 2021`
- this value is linked to that key

So annotation is not the text itself. It is the dataset label describing what a region on the page means.

### What duplicate annotator copies mean

The same page can appear multiple times in the raw dataset because multiple annotators labeled that same page separately.

That is why the number of raw rows is much larger than the number of unique pages.

This duplication is at the page level, not at the level of one single field.

### Why grouping by `hash_name` does not lose fields

Grouping by `hash_name` means we keep one final version of the whole page, not one final key or one final value.

So if a page contains fields such as:
- name
- family name
- bill number
- address

we do not keep only one of those fields.

Instead, we choose one annotation version for that page and keep the full set of fields from that selected version.

The preparation script chooses the richest usable annotation set among the duplicate copies of the same page.

## What the Preparation Script Does

The preparation logic is implemented in `prepare_data.py`.

### 1. Load the raw KVP10k split

The script reads the train or test split from Hugging Face.

### 2. Group rows by `hash_name`

This removes duplicate annotator copies at the page level.

Without this step:
- the same PDF page would be downloaded several times
- the same page would appear as multiple training examples
- slightly different annotation copies of the same page would be mixed together incorrectly

### 3. Download the original PDF

The dataset row points to a PDF URL. The script downloads the real source PDF for that page.

### 4. Validate the PDF

The script checks whether the downloaded content is actually a PDF.

This is necessary because some URLs return:
- broken files
- permission-denied pages
- HTML error pages instead of a PDF

### 5. Open the correct page

The dataset page number is 1-indexed, but PyMuPDF uses 0-indexing.

So the script converts the page number before reading the page.

### 6. Extract words and boxes from the PDF

The script uses PyMuPDF to extract:
- the words on the page
- the location of each word as a bounding box

### 7. Discard unusable pages

If the page has no usable text, it is discarded.

This usually happens when:
- the PDF is broken
- the PDF page has no text layer

### What a text layer means

A text layer is the hidden machine-readable text inside a PDF.

Simple intuition:
- if you can select text in a PDF, it usually has a text layer
- if the PDF is only a scanned image, it may have no text layer

If there is no text layer, PyMuPDF cannot extract usable words directly.

### 8. Fuse extracted words with annotation boxes

This means the script matches the text found on the page with the labeled annotation rectangles.

Why this is needed:
- annotations tell us what region is important
- extracted words tell us what text is actually there
- the model needs both together

So the script checks which words fall sufficiently inside each annotation rectangle and assigns those words to that annotation.

This is normal preprocessing work for document understanding pipelines.

## Did IBM provide this step?

Yes, in principle.

The IBM KVP10k repository states that they provide:
- code for downloading and preparing the dataset
- OCR processing
- annotation and OCR fusion
- training and benchmarking code

In particular, the repository documents a `download_dataset.py` script that:
- downloads the matching PDF
- extracts the correct page
- runs OCR
- creates ground-truth JSON files

So the preparation idea is part of their published workflow.

However, this does not remove the need to reproduce and adapt it locally.

What still had to be done in this thesis:
- adapt the workflow to the actual Hugging Face raw-row format in use here
- handle repeated annotator copies explicitly
- make the pipeline work on TinyGPU
- replace IBM's OCR dependency path with PyMuPDF text extraction for this environment
- produce prepared outputs in the format needed by the corrected Mistral training code

So this was not unnecessary extra work. It was a required reproduction and adaptation step.

### 9. Reconstruct key-value pairs using annotation linking

The annotations contain linking information that connects a value region to its corresponding key region.

The script uses this linking to rebuild:
- normal key-value pairs
- unvalued keys
- unkeyed values

This is how the raw labeled rectangles become structured KVP supervision.

## Did IBM provide this linking logic?

Yes, this is also part of the official data-preparation idea.

But again, the thesis still needed to reproduce and adapt that logic into a working local preparation pipeline.

### 10. Normalize coordinates

Different pages have different image sizes.

The script converts bounding boxes into a common $100 \times 100$ coordinate system so the data is consistent across pages.

### 11. Build the LMDX prompt text

The script groups words into line-like text segments and writes them in a format like:

```text
Orders due: 40|13|57|16
November 5, 2021 35|16|62|19
1 lb. shelled $15.00 40|25|71|27
```

Each line contains:
- the text
- the normalized location `left|top|right|bottom`

This is useful because the model sees both the content and rough layout of the document.

### Real Example

From one prepared page:
- `Orders due: 40|13|57|16`
- `November 5, 2021 35|16|62|19`

The target then contains a KVP like:

```text
['Orders due: 40|13|57|15', 'November 5, 2021 35|17|62|19']
```

So the model is trained to transform document text with layout hints into structured key-value output.

### 12. Save one prepared JSON per unique page

Each final prepared file contains:
- `hash_name`
- image size
- number of words
- number of KVPs
- `lmdx_text`
- `target_text`
- `full_prompt`
- `gt_kvps`

These JSON files are the actual train/test inputs for the corrected Stage 3 Mistral pipeline.

## Why This Approach Is Defensible

This preparation path is reasonable because it:
- starts from the real source PDF
- checks that the source file is valid
- extracts actual document text and actual locations
- uses annotation linking instead of guessing key-value relations
- handles duplicate annotator copies explicitly
- rejects unusable pages instead of silently training on corrupted input
- converts everything into a reproducible page-level format for training and evaluation

So the correct claim is not that the data is magically perfect.

The correct claim is:

The data is now technically defensible, reproducible, and much more trustworthy than using the raw dataset rows directly for the Stage 3 Mistral baseline.

## Short Supervisor-Friendly Version

The raw KVP10k rows were not directly model-ready for the Mistral baseline. They still had to be converted from PDF references and repeated annotator copies into unique page-level training examples with extracted text, bounding boxes, reconstructed key-value supervision, and prompt/target formatting. The preparation pipeline therefore downloads the real PDF, validates it, extracts page text, aligns text with annotations, reconstructs the key-value pairs using annotation linking, and writes one final prepared JSON per unique page. This is why the preparation step is necessary and why it is part of the substantive technical work of the thesis.