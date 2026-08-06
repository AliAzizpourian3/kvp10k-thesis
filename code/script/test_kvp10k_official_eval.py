"""Regression tests for IBM KVP10k benchmark-compatible evaluation."""

from kvp10k_official_eval import evaluate_documents, evaluate_table, kvps_match


FULL_BOX = [0, 0, 10, 10]
IOU_BOUNDARY_BOX = [0, 0, 10, 3]


def _unvalued(text: str, bbox=FULL_BOX) -> dict:
    return {"type": "unvalued", "key": {"text": text, "bbox": bbox}}


def _regular(key: str, value: str) -> dict:
    return {
        "type": "kvp",
        "key": {"text": key, "bbox": FULL_BOX},
        "value": {"text": value, "bbox": FULL_BOX},
    }


def test_released_boundary_semantics() -> None:
    assert not kvps_match(_unvalued("abcdf"), _unvalued("abcde"), "text_only")
    assert kvps_match(
        _unvalued("same", IOU_BOUNDARY_BOX),
        _unvalued("same", FULL_BOX),
        "location_only",
    )


def test_macro_aggregation_differs_from_pooled_counts() -> None:
    documents = [
        {
            "document_id": "perfect",
            "predictions": [_unvalued("same")],
            "ground_truths": [_unvalued("same")],
        },
        {
            "document_id": "wrong",
            "predictions": [_unvalued("wrong")] * 3,
            "ground_truths": [_unvalued("target")],
        },
    ]
    result = evaluate_documents(documents, "text_only")
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert (result["tp"], result["fp"], result["fn"]) == (1, 3, 1)


def test_empty_gt_documents_are_excluded_like_official_code() -> None:
    documents = [
        {"document_id": "empty", "predictions": [_unvalued("fp")], "ground_truths": []},
        {
            "document_id": "scored",
            "predictions": [_unvalued("same")],
            "ground_truths": [_unvalued("same")],
        },
    ]
    result = evaluate_documents(documents, "text_only")
    assert result["f1"] == 1.0
    assert result["documents_scored"] == 1
    assert result["documents_excluded_empty_gt"] == 1


def test_types_and_category_filtering() -> None:
    documents = [{
        "document_id": "types",
        "predictions": [_regular("key", "value")],
        "ground_truths": [_regular("key", "value"), _unvalued("key")],
    }]
    table = evaluate_table(documents)
    assert table["regular"]["text_only"]["f1"] == 1.0
    assert table["unvalued"]["text_only"]["f1"] == 0.0
    assert table["all"]["text_only"]["precision"] == 1.0
    assert table["all"]["text_only"]["recall"] == 0.5


if __name__ == "__main__":
    test_released_boundary_semantics()
    test_macro_aggregation_differs_from_pooled_counts()
    test_empty_gt_documents_are_excluded_like_official_code()
    test_types_and_category_filtering()
    print("All official KVP10k evaluator tests passed.")