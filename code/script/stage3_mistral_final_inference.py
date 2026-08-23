"""Final, inference-only Stage 3 Mistral prediction path.

This module is separate from ``mistral_baseline.py`` so that a queued training
job can continue to use its submitted implementation without changes. It
loads a completed adapter checkpoint, preserves every decoded generation, and
supports offline reparsing without another model run.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Optional


LOGGER = logging.getLogger(__name__)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_LENGTH = 8192
MAX_NEW_TOKENS = 2048
EXPECTED_TEST_DOCUMENTS = 581
CANONICAL_UNKEYED_CATEGORIES = frozenset(
    {
        "name",
        "date",
        "address",
        "amount",
        "phone",
        "email",
        "website",
        "year",
        "document type",
        "document title",
        "text",
    }
)


def _normalize_label(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _parse_entity(value: str) -> tuple[str, Optional[list[int]]]:
    """Parse ``text left|top|right|bottom`` without rejecting plain text."""
    value = str(value)
    parts = value.rsplit(" ", 1)
    if len(parts) == 2:
        try:
            bbox = [int(item) for item in parts[1].split("|")]
        except ValueError:
            bbox = []
        if len(bbox) == 4:
            return parts[0], bbox
    return value, None


def _decode_response(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decode a response and return predictions with parse-status metadata."""
    stripped = text.strip()
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return [], {
            "parsing_succeeded": False,
            "items_seen": 0,
            "malformed_entry_count": 0,
        }

    if not isinstance(parsed, list):
        return [], {
            "parsing_succeeded": False,
            "items_seen": 0,
            "malformed_entry_count": 0,
        }

    kvps: list[dict[str, Any]] = []
    malformed_entries = 0
    for item in parsed:
        if not isinstance(item, list) or not item:
            malformed_entries += 1
            continue

        key_text, key_bbox = _parse_entity(str(item[0]))
        if len(item) == 1:
            kvp: dict[str, Any] = {
                "type": "unvalued",
                "key": {"text": key_text},
            }
            if key_bbox is not None:
                kvp["key"]["bbox"] = key_bbox
            kvps.append(kvp)
            continue

        value_text, value_bbox = _parse_entity(str(item[1]))
        normalized_key = _normalize_label(key_text)
        normalized_value = _normalize_label(value_text)

        if normalized_value == "not presented":
            kvp = {"type": "unvalued", "key": {"text": key_text}}
            if key_bbox is not None:
                kvp["key"]["bbox"] = key_bbox
            kvps.append(kvp)
            continue

        implicit_category = ""
        if normalized_key.startswith("implicit "):
            implicit_category = _normalize_label(normalized_key.removeprefix("implicit "))

        if (
            key_bbox is None
            and normalized_key in CANONICAL_UNKEYED_CATEGORIES
        ) or implicit_category in CANONICAL_UNKEYED_CATEGORIES:
            category = implicit_category or normalized_key
            kvp = {
                "type": "unkeyed",
                "key": {"text": category},
                "value": {"text": value_text},
            }
            if value_bbox is not None:
                kvp["value"]["bbox"] = value_bbox
            kvps.append(kvp)
            continue

        kvp = {
            "type": "kvp",
            "key": {"text": key_text},
            "value": {"text": value_text},
        }
        if key_bbox is not None:
            kvp["key"]["bbox"] = key_bbox
        if value_bbox is not None:
            kvp["value"]["bbox"] = value_bbox
        kvps.append(kvp)

    return kvps, {
        "parsing_succeeded": True,
        "items_seen": len(parsed),
        "malformed_entry_count": malformed_entries,
    }


def parse_response(text: str) -> list[dict[str, Any]]:
    """Return type-aware KVP predictions from a list-of-lists response."""
    kvps, _ = _decode_response(text)
    return kvps


def _parse_response(text: str) -> list[dict[str, Any]]:
    """Backward-compatible alias for parser tests and saved workflows."""
    return parse_response(text)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary_path, path)


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _valid_prediction(path: Path) -> bool:
    data = _load_json(path)
    return data is not None and isinstance(data.get("kvps_list"), list)


def _saved_raw_response(path: Path, hash_name: str) -> Optional[str]:
    data = _load_json(path)
    if data is None or data.get("hash_name") != hash_name:
        return None
    response = data.get("raw_response")
    return response if isinstance(response, str) else None


def _complete_raw_record(path: Path, hash_name: str) -> bool:
    data = _load_json(path)
    return bool(
        data is not None
        and data.get("hash_name") == hash_name
        and data.get("complete") is True
        and isinstance(data.get("raw_response"), str)
        and isinstance(data.get("kvps_list"), list)
        and isinstance(data.get("parsing_succeeded"), bool)
        and isinstance(data.get("parsed_entry_count"), int)
    )


def _prepared_documents(data_dir: Path) -> dict[str, Path]:
    test_dir = data_dir / "test" if (data_dir / "test").is_dir() else data_dir
    documents: dict[str, Path] = {}
    for path in sorted(test_dir.glob("*.json")):
        data = _load_json(path)
        if data is None:
            raise ValueError(f"Invalid prepared JSON: {path}")
        hash_name = data.get("hash_name")
        if not isinstance(hash_name, str) or not hash_name:
            raise ValueError(f"Missing hash_name: {path}")
        if path.stem != hash_name:
            raise ValueError(f"Filename/hash mismatch: {path.name} != {hash_name}.json")
        if hash_name in documents:
            raise ValueError(f"Duplicate prepared hash: {hash_name}")
        documents[hash_name] = path
    return documents


def find_longest_prompt(data_dir: Path, tokenizer_name: str, output: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    documents = _prepared_documents(data_dir)
    if len(documents) != EXPECTED_TEST_DOCUMENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_DOCUMENTS} test documents, found {len(documents)}"
        )

    longest: Optional[dict[str, Any]] = None
    for index, (hash_name, path) in enumerate(documents.items(), start=1):
        data = _load_json(path)
        assert data is not None
        prompt = data.get("full_prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Missing full_prompt: {path}")
        token_count = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        candidate = {
            "hash_name": hash_name,
            "prepared_file": str(path.resolve()),
            "untruncated_prompt_tokens": token_count,
            "inference_input_tokens": min(token_count, MAX_LENGTH),
        }
        if longest is None or token_count > longest["untruncated_prompt_tokens"]:
            longest = candidate
        if index % 100 == 0:
            LOGGER.info("Tokenized %d/%d test prompts", index, len(documents))

    assert longest is not None
    payload = {
        "tokenizer": tokenizer_name,
        "max_length": MAX_LENGTH,
        "documents_considered": len(documents),
        "longest": longest,
    }
    _atomic_write_json(output, payload)
    return payload


def _load_model(checkpoint: Path):
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint,
        quantization_config=quantization,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def _write_parsed_outputs(
    *,
    hash_name: str,
    raw_response: str,
    raw_path: Path,
    prediction_path: Path,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    kvps, parse_status = _decode_response(raw_response)
    _atomic_write_json(prediction_path, {"kvps_list": kvps})
    record: dict[str, Any] = {
        "hash_name": hash_name,
        "raw_response": raw_response,
        "kvps_list": kvps,
        **parse_status,
        "parsed_entry_count": len(kvps),
        "complete": True,
    }
    if metadata:
        record["inference_metadata"] = metadata
    _atomic_write_json(raw_path, record)


def run_inference(
    checkpoint: Path,
    data_dir: Path,
    output_root: Path,
    selected_hashes: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    import torch

    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    documents = _prepared_documents(data_dir)
    if selected_hashes is None:
        if len(documents) != EXPECTED_TEST_DOCUMENTS:
            raise RuntimeError(
                f"Expected {EXPECTED_TEST_DOCUMENTS} test documents, found {len(documents)}"
            )
        selected = documents
    else:
        requested = list(selected_hashes)
        missing = sorted(set(requested) - set(documents))
        if missing:
            raise KeyError(f"Unknown prepared hashes: {missing}")
        selected = {hash_name: documents[hash_name] for hash_name in requested}

    raw_dir = output_root / "raw_responses"
    prediction_dir = output_root / "predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    pending_generation: list[tuple[str, Path]] = []
    reparsed = 0
    skipped = 0
    for hash_name, prepared_path in selected.items():
        raw_path = raw_dir / f"{hash_name}.json"
        prediction_path = prediction_dir / f"{hash_name}.json"
        if _complete_raw_record(raw_path, hash_name) and _valid_prediction(prediction_path):
            skipped += 1
            continue
        saved_response = _saved_raw_response(raw_path, hash_name)
        if saved_response is not None:
            _write_parsed_outputs(
                hash_name=hash_name,
                raw_response=saved_response,
                raw_path=raw_path,
                prediction_path=prediction_path,
                metadata={"source": "saved_raw_response"},
            )
            reparsed += 1
            continue
        pending_generation.append((hash_name, prepared_path))

    generated = 0
    peak_memory_bytes = 0
    if pending_generation:
        model, tokenizer = _load_model(checkpoint)
        for index, (hash_name, prepared_path) in enumerate(pending_generation, start=1):
            data = _load_json(prepared_path)
            assert data is not None
            prompt = data.get("full_prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"Missing full_prompt: {prepared_path}")

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            raw_response = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            current_peak = torch.cuda.max_memory_allocated()
            peak_memory_bytes = max(peak_memory_bytes, current_peak)

            raw_path = raw_dir / f"{hash_name}.json"
            prediction_path = prediction_dir / f"{hash_name}.json"
            _atomic_write_json(
                raw_path,
                {
                    "hash_name": hash_name,
                    "raw_response": raw_response,
                    "complete": False,
                },
            )
            _write_parsed_outputs(
                hash_name=hash_name,
                raw_response=raw_response,
                raw_path=raw_path,
                prediction_path=prediction_path,
                metadata={
                    "source": "model_generation",
                    "checkpoint": str(checkpoint),
                    "input_tokens": int(inputs["input_ids"].shape[1]),
                    "generated_tokens": int(
                        output.shape[1] - inputs["input_ids"].shape[1]
                    ),
                    "max_length": MAX_LENGTH,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "do_sample": False,
                    "peak_cuda_memory_bytes": int(current_peak),
                },
            )
            generated += 1
            LOGGER.info("Completed %d/%d: %s", index, len(pending_generation), hash_name)

    expected_names = {f"{hash_name}.json" for hash_name in selected}
    prediction_names = {path.name for path in prediction_dir.glob("*.json")}
    raw_names = {path.name for path in raw_dir.glob("*.json")}
    if prediction_names != expected_names:
        raise RuntimeError(
            "Prediction filename mismatch: "
            f"missing={sorted(expected_names - prediction_names)}, "
            f"extra={sorted(prediction_names - expected_names)}"
        )
    if raw_names != expected_names:
        raise RuntimeError(
            "Raw-response filename mismatch: "
            f"missing={sorted(expected_names - raw_names)}, "
            f"extra={sorted(raw_names - expected_names)}"
        )
    for hash_name in selected:
        if not _valid_prediction(prediction_dir / f"{hash_name}.json"):
            raise RuntimeError(f"Invalid prediction record: {hash_name}")
        if not _complete_raw_record(raw_dir / f"{hash_name}.json", hash_name):
            raise RuntimeError(f"Incomplete raw-response record: {hash_name}")

    summary = {
        "checkpoint": str(checkpoint),
        "documents_considered": len(selected),
        "prediction_files": len(prediction_names),
        "raw_response_files": len(raw_names),
        "generated": generated,
        "reparsed_from_saved_raw": reparsed,
        "skipped_complete": skipped,
        "peak_cuda_memory_bytes": int(peak_memory_bytes),
        "max_length": MAX_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "complete": True,
    }
    _atomic_write_json(output_root / "inference_summary.json", summary)
    return summary


def _hash_from_selection(path: Path) -> str:
    data = _load_json(path)
    if data is None:
        raise ValueError(f"Invalid selection file: {path}")
    longest = data.get("longest")
    if not isinstance(longest, dict) or not isinstance(longest.get("hash_name"), str):
        raise ValueError(f"Missing longest.hash_name in {path}")
    return longest["hash_name"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    longest_parser = subparsers.add_parser("find-longest")
    longest_parser.add_argument("--data-dir", required=True, type=Path)
    longest_parser.add_argument("--tokenizer", default=MODEL_NAME)
    longest_parser.add_argument("--output", required=True, type=Path)

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--checkpoint", required=True, type=Path)
    smoke_parser.add_argument("--data-dir", required=True, type=Path)
    smoke_parser.add_argument("--output-root", required=True, type=Path)
    smoke_parser.add_argument("--selection-file", required=True, type=Path)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--checkpoint", required=True, type=Path)
    predict_parser.add_argument("--data-dir", required=True, type=Path)
    predict_parser.add_argument("--output-root", required=True, type=Path)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if args.command == "find-longest":
        print(json.dumps(find_longest_prompt(args.data_dir, args.tokenizer, args.output), indent=2))
    elif args.command == "smoke":
        selected_hash = _hash_from_selection(args.selection_file)
        print(
            json.dumps(
                run_inference(
                    args.checkpoint,
                    args.data_dir,
                    args.output_root,
                    selected_hashes=[selected_hash],
                ),
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                run_inference(args.checkpoint, args.data_dir, args.output_root),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
