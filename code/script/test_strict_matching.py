"""Boundary tests for the paper-literal pooled diagnostic matcher.

Values exactly at a threshold (NED == 0.2, IoU == 0.3) MUST be rejected. Before the
fix these were accepted (inclusive <=/>=), which was more lenient than the paper
wording. IBM's released benchmark differs by using inclusive IoU; that behavior is
tested separately in test_kvp10k_official_eval.py. No GPU or dataset required.
"""
import sys

sys.path.insert(0, "code/script")

from evaluate_mistral import _ned, _iou, match_entities
from evaluate_stage4b import _text_overlap, _bbox_ok
from metrics import match_prediction_to_ground_truth
import analyze_v4_diagnostics as diag


# --- Fixtures with EXACT boundary values ------------------------------------
# NED("abcde", "abcdf") = 1 / max(5, 5) = 0.2 exactly.
NED_AT = ("abcde", "abcdf")          # NED == 0.2
NED_BELOW = ("abcde", "abcde")       # NED == 0.0  (< 0.2)
# Box B fully inside box A: inter = area(B) = 30, union = area(A) = 100 -> IoU = 0.3.
IOU_AT = ([0, 0, 10, 10], [0, 0, 10, 3])    # IoU == 0.3
IOU_ABOVE = ([0, 0, 10, 10], [0, 0, 10, 4])  # IoU == 0.4  (> 0.3)


def _check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    assert condition, label


def test_ned_fixtures():
    _check("NED('abcde','abcdf') == 0.2", abs(_ned(*NED_AT) - 0.2) < 1e-12)
    _check("IoU boundary box == 0.3", abs(_iou(*IOU_AT) - 0.3) < 1e-12)
    _check("IoU above box == 0.4", abs(_iou(*IOU_ABOVE) - 0.4) < 1e-12)


def test_evaluate_stage4b_text_overlap():
    _check("_text_overlap rejects NED == 0.2", _text_overlap(*NED_AT, ned_thresh=0.2) is False)
    _check("_text_overlap accepts NED < 0.2", _text_overlap(*NED_BELOW, ned_thresh=0.2) is True)


def test_evaluate_stage4b_bbox_ok():
    _check("_bbox_ok rejects IoU == 0.3", _bbox_ok(*IOU_AT, iou_thresh=0.3) is False)
    _check("_bbox_ok accepts IoU > 0.3", _bbox_ok(*IOU_ABOVE, iou_thresh=0.3) is True)
    # Missing/malformed boxes keep the text-only fallback (returns True).
    _check("_bbox_ok fallback when box missing", _bbox_ok(None, [0, 0, 10, 3], iou_thresh=0.3) is True)


def test_evaluate_mistral_match_entities():
    # A single pred/GT pair at the NED boundary must NOT be counted as a match.
    preds = [(NED_AT[0], None, "key")]
    gts = [(NED_AT[1], None, "key")]
    res = match_entities(preds, gts, ned_thresh=0.2, use_bbox=False)
    _check("match_entities rejects NED == 0.2 (tp==0)", res["tp"] == 0)
    # Below the boundary it matches.
    preds_ok = [(NED_BELOW[0], None, "key")]
    gts_ok = [(NED_BELOW[1], None, "key")]
    res_ok = match_entities(preds_ok, gts_ok, ned_thresh=0.2, use_bbox=False)
    _check("match_entities accepts NED < 0.2 (tp==1)", res_ok["tp"] == 1)
    # IoU boundary with a text match must also be rejected.
    preds_iou = [(NED_BELOW[0], IOU_AT[0], "key")]
    gts_iou = [(NED_BELOW[1], IOU_AT[1], "key")]
    res_iou = match_entities(preds_iou, gts_iou, ned_thresh=0.2, iou_thresh=0.3, use_bbox=True)
    _check("match_entities rejects IoU == 0.3 (tp==0)", res_iou["tp"] == 0)


def test_metrics_find_best_match():
    matched, _ = match_prediction_to_ground_truth(
        IOU_AT[0], NED_AT[0], [IOU_AT[1]], [NED_AT[1]],
        iou_threshold=0.3, ned_threshold=0.2,
    )
    _check("match_prediction rejects NED==0.2 & IoU==0.3", matched is False)
    matched_ok, _ = match_prediction_to_ground_truth(
        IOU_ABOVE[0], NED_BELOW[0], [IOU_ABOVE[1]], [NED_BELOW[1]],
        iou_threshold=0.3, ned_threshold=0.2,
    )
    _check("match_prediction accepts NED<0.2 & IoU>0.3", matched_ok is True)


def test_diagnostic_passes():
    official = {"name": "official", "ned": 0.2, "iou": 0.3, "use_bbox": True}
    # prediction = (key, value, confidence, key_box, value_box)
    # ground_truth = (key, value, key_box, value_box)
    pred_at = (NED_AT[0], NED_BELOW[0], 0.9, IOU_AT[0], IOU_ABOVE[0])
    gt_at = (NED_AT[1], NED_BELOW[1], IOU_AT[1], IOU_ABOVE[1])
    m_at = diag._pair_metrics(pred_at, gt_at)
    _check("_passes rejects key NED == 0.2", diag._passes(m_at, pred_at, gt_at, official) is False)

    pred_iou = (NED_BELOW[0], NED_BELOW[0], 0.9, IOU_AT[0], IOU_ABOVE[0])
    gt_iou = (NED_BELOW[1], NED_BELOW[1], IOU_AT[1], IOU_ABOVE[1])
    m_iou = diag._pair_metrics(pred_iou, gt_iou)
    _check("_passes rejects key IoU == 0.3", diag._passes(m_iou, pred_iou, gt_iou, official) is False)

    pred_ok = (NED_BELOW[0], NED_BELOW[0], 0.9, IOU_ABOVE[0], IOU_ABOVE[0])
    gt_ok = (NED_BELOW[1], NED_BELOW[1], IOU_ABOVE[1], IOU_ABOVE[1])
    m_ok = diag._pair_metrics(pred_ok, gt_ok)
    _check("_passes accepts NED<0.2 & IoU>0.3", diag._passes(m_ok, pred_ok, gt_ok, official) is True)


if __name__ == "__main__":
    tests = [
        test_ned_fixtures,
        test_evaluate_stage4b_text_overlap,
        test_evaluate_stage4b_bbox_ok,
        test_evaluate_mistral_match_entities,
        test_metrics_find_best_match,
        test_diagnostic_passes,
    ]
    for test in tests:
        test()
    print(f"\nAll {len(tests)} strict-matching boundary tests passed.")
