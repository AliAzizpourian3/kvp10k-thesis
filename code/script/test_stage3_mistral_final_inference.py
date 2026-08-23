"""CPU-only tests for the final Stage 3 response parser."""

import unittest

from mistral_baseline import _parse_response as _baseline_parse_response
from stage3_mistral_final_inference import _decode_response, _parse_response


class Stage3FinalParserTests(unittest.TestCase):
    """Verify special-category parsing and baseline parser parity."""

    def test_regular_pair(self):
        result = _parse_response("[['Invoice 1|2|3|4', '42 5|6|7|8']]")
        self.assertEqual(
            result,
            [{
                "type": "kvp",
                "key": {"text": "Invoice", "bbox": [1, 2, 3, 4]},
                "value": {"text": "42", "bbox": [5, 6, 7, 8]},
            }],
        )

    def test_current_unkeyed_name(self):
        result = _parse_response("[['Name', 'Ada Lovelace 5|6|7|8']]")
        self.assertEqual(result[0]["type"], "unkeyed")
        self.assertEqual(result[0]["key"], {"text": "name"})
        self.assertEqual(result[0]["value"]["bbox"], [5, 6, 7, 8])

    def test_current_unkeyed_address(self):
        result = _parse_response("[['  ADDRESS  ', 'Main Street 5|6|7|8']]")
        self.assertEqual(result[0]["type"], "unkeyed")
        self.assertEqual(result[0]["key"], {"text": "address"})

    def test_trained_unkeyed_amount(self):
        result = _parse_response("[['amount', '$42.00 5|6|7|8']]")
        self.assertEqual(result[0]["type"], "unkeyed")
        self.assertEqual(result[0]["key"], {"text": "amount"})

    def test_current_unvalued(self):
        result = _parse_response("[['Signature 1|2|3|4']]")
        self.assertEqual(
            result,
            [{
                "type": "unvalued",
                "key": {"text": "Signature", "bbox": [1, 2, 3, 4]},
            }],
        )

    def test_ibm_implicit_name(self):
        result = _parse_response("[['implicit name', 'Ada 5|6|7|8']]")
        self.assertEqual(result[0]["type"], "unkeyed")
        self.assertEqual(result[0]["key"], {"text": "name"})

    def test_ibm_not_presented(self):
        result = _parse_response("[['Signature 1|2|3|4', 'not presented']]")
        self.assertEqual(
            result,
            [{
                "type": "unvalued",
                "key": {"text": "Signature", "bbox": [1, 2, 3, 4]},
            }],
        )

    def test_malformed_bbox_stays_regular(self):
        result = _parse_response("[['Invoice 1|2|3', '42 5|6|7|8']]")
        self.assertEqual(result[0]["type"], "kvp")
        self.assertEqual(result[0]["key"], {"text": "Invoice 1|2|3"})

    def test_malformed_outer_response(self):
        result, status = _decode_response("not a Python list")
        self.assertEqual(result, [])
        self.assertFalse(status["parsing_succeeded"])

    def test_non_category_missing_key_bbox_stays_regular(self):
        result = _parse_response("[['invoice number', '42 5|6|7|8']]")
        self.assertEqual(result[0]["type"], "kvp")
        self.assertEqual(result[0]["key"], {"text": "invoice number"})

    def test_baseline_uses_final_parser(self):
        response = (
            "[['Name', 'Ada 5|6|7|8'], "
            "['Signature 1|2|3|4', 'not presented']]"
        )
        self.assertEqual(_baseline_parse_response(response), _parse_response(response))


if __name__ == "__main__":
    unittest.main()
