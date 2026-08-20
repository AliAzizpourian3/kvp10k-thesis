import unittest

from analyze_v5_qualitative import match_indices, select_distinct_documents
from export_v5_recovery_analysis import entity_error_kind


def _pair(key: str, value: str) -> dict:
    box = [0, 0, 10, 10]
    return {
        "type": "kvp",
        "key": {"text": key, "bbox": box},
        "value": {"text": value, "bbox": box},
    }


class V5FinalAnalysisTests(unittest.TestCase):
    def test_key_value_confusions_are_explicit(self):
        self.assertEqual(entity_error_kind(1, 2), "key_predicted_as_value")
        self.assertEqual(entity_error_kind(2, 1), "value_predicted_as_key")
        self.assertEqual(entity_error_kind(1, 1), None)

    def test_official_matching_is_one_to_one(self):
        predictions = [_pair("Date", "2026"), _pair("Date", "2026")]
        matches, used = match_indices(predictions, [_pair("Date", "2026")])
        self.assertEqual(matches, {0: 0})
        self.assertEqual(used, {0})

    def test_example_selection_uses_distinct_documents(self):
        records = [
            {"document_id": "a", "value": 1},
            {"document_id": "a", "value": 2},
            {"document_id": "b", "value": 3},
        ]
        self.assertEqual(
            select_distinct_documents(records, 2),
            [records[0], records[2]],
        )


if __name__ == "__main__":
    unittest.main()
