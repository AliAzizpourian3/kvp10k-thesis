"""Create a small, explicitly curated V5 qualitative-example set.

The GPU evaluation writes a larger deterministic candidate pool.  This script
selects readable records from that pool without rerunning inference or changing
any benchmark threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_stage4b_v5 import utc_now, write_json


def _find(records: list[dict[str, Any]], document_id: str, word_or_key: str) -> dict[str, Any]:
    for record in records:
        if record.get("document_id") != document_id:
            continue
        if record.get("word") == word_or_key:
            return record
        prediction = record.get("prediction") or {}
        ground_truth = record.get("ground_truth") or {}
        predicted_key = prediction.get("key", {}).get("text")
        ground_truth_key = ground_truth.get("key", {}).get("text")
        if word_or_key in {predicted_key, ground_truth_key}:
            return record
    raise KeyError(f"Candidate not found: {document_id} / {word_or_key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--entity-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    with args.candidates.open(encoding="utf-8") as handle:
        candidates = json.load(handle)
    with args.entity_candidates.open(encoding="utf-8") as handle:
        entity = json.load(handle)

    pair_candidates = candidates["pair_analysis"]["candidates"]
    correct_specs = [
        ("efc7468875a54c1becf94cc01fa01ee732adc095a438b6044d7e901ba71b6344", "Instructor:"),
        ("5c9befa2336e89d621efde9dfb845f40cf3537f8ba7411354e50c7102a02b0ad", "Details:"),
        ("035deab0593f06dda63a4966fa30531f2c96f7505eec76e135a8a0592ed5d760", "Special Note:"),
        ("1d752623731b1446f58b3ec87f8271b390b3240a0c8410a4a0d6ba1633268982", "Address:"),
    ]
    wrong_specs = [
        ("1d38adadf89e9ba2c03f3f6cc89cfac24dc9cff59de825d8ea9969e96dd98e21", "Total Networks"),
        ("0e3008db1915a75e6badd0d786759540521c72be2640d3a196fd25619f39ba61", "Syscode"),
        ("07de02d344336182543e7e7592f614d7acce0412ddf034a477195574c98a8aeb", "Zip"),
        ("04d279efe18c2f4c385d8ceaee48a544a9e4f16099ff184b047f09e4578e5643", "Fax:"),
    ]
    missed_specs = [
        ("9aa0013771206f9b35873bc3c22a8844d31a12f07c1c0eb603a5ddbe0c7518ca", "Page"),
        ("5a58456c08ee67739cbfe2e13fe45d740dd86afe4c9bfeb92197ad93caaab7b1", "EDI:"),
        ("00dbb5450efe3a6cb80ebb0a4f09dafb14606c7731bffaf3f439cc5067929673", "ext."),
        ("1ab8b1c4f0bbb463686c528269faa1ef5dab2b09d2cb98d4dd6837c21405f940", "Spots"),
    ]

    entity_candidates = entity["top_examples"]
    entity_specs = {
        "key_predicted_as_value": [
            ("712c9843c931963bb575073fc138b22f4899f5445b3f39d918fadabbfee388fb", "COMPANY")
        ],
        "value_predicted_as_key": [
            ("27df4f83b95e73cc77a2030e35321a71345229fdbc428ad3c3561e6e4122c7cc", "card")
        ],
        "missed_entity": [
            ("60986c99f78d1f41459dd07c49dc726caa251aae5092bf46d66250d3f9272208", "YYYY-MM-DD")
        ],
        "spurious_entity": [
            ("b33f2b9454c5e808dcd6236936a16621b85b2e5e6397ae82d5b2ca7ebae2ba11", "This")
        ],
    }

    write_json(
        args.output,
        {
            "metadata": {
                "created_at": utc_now(),
                "source_candidates": str(args.candidates),
                "source_entity_candidates": str(args.entity_candidates),
                "selection": "manual readability review of deterministic candidate pools",
                "matching_mode": "official text_location",
                "ned_threshold": 0.2,
                "iou_threshold": 0.3,
                "caution": (
                    "Entity examples are token-level disagreements with KVP10k annotations; "
                    "context can make individual labels semantically ambiguous."
                ),
            },
            "correct_pairs": [_find(pair_candidates["correct_pairs"], *spec) for spec in correct_specs],
            "wrong_links": [_find(pair_candidates["wrong_links"], *spec) for spec in wrong_specs],
            "missed_pairs": [_find(pair_candidates["missed_pairs"], *spec) for spec in missed_specs],
            "entity_errors": {
                kind: [_find(entity_candidates[kind], *spec) for spec in specs]
                for kind, specs in entity_specs.items()
            },
        },
    )


if __name__ == "__main__":
    main()
