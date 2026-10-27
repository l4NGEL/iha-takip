import numpy as np
import pytest

from iha_takip.metrics import (
    average_precision,
    best_f1_threshold,
    iou_xywh,
    match_frame,
    pr_at_threshold,
    size_bin,
)


def test_iou_identical_and_disjoint():
    assert iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou_xywh((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


def test_iou_half_overlap():
    # 10x10 ve 5 kaydirilmis: kesisim 50, birlesim 150
    assert iou_xywh((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_match_frame_only_best_scoring_match_is_tp():
    dets = np.array([[0, 0, 10, 10, 0.4], [1, 0, 10, 10, 0.9], [50, 50, 5, 5, 0.8]])
    tp = match_frame(dets, (0, 0, 10, 10))
    assert tp.tolist() == [False, True, False]


def test_match_frame_absent_target_all_fp():
    dets = np.array([[0, 0, 10, 10, 0.9]])
    assert match_frame(dets, None).tolist() == [False]


def test_ap_perfect_and_zero():
    assert average_precision(np.array([0.9, 0.8]), np.array([True, True]), 2) == pytest.approx(1.0)
    assert average_precision(np.array([0.9]), np.array([False]), 1) == pytest.approx(0.0)


def test_ap_half_recall():
    # Tek TP en ustte, 2 hedef: recall 0.5'te precision 1 -> AP 0.5
    ap = average_precision(np.array([0.9, 0.5]), np.array([True, False]), 2)
    assert ap == pytest.approx(0.5)


def test_ap_interpolates_precision_envelope():
    # FP, TP, TP: P/R noktalari (0,0) (1/2,1/2) (2/3,1) -> zarf 2/3 tum araliga
    ap = average_precision(np.array([0.9, 0.8, 0.7]), np.array([False, True, True]), 2)
    assert ap == pytest.approx(2 / 3)


def test_pr_at_threshold():
    scores = np.array([0.9, 0.6, 0.3])
    tp = np.array([True, False, True])
    p, r = pr_at_threshold(scores, tp, 4, 0.5)
    assert (p, r) == (pytest.approx(0.5), pytest.approx(0.25))


def test_best_f1_threshold_picks_group_end():
    scores = np.array([0.9, 0.8, 0.8, 0.1])
    tp = np.array([True, True, False, False])
    thr, f1 = best_f1_threshold(scores, tp, 2)
    # 0.9 esiginde P=1 R=0.5 F1=0.667; 0.8 esiginde P=2/3 R=1 F1=0.8
    assert thr == pytest.approx(0.8)
    assert f1 == pytest.approx(0.8)


def test_size_bin():
    assert size_bin((0, 0, 8, 8)) == "<16"
    assert size_bin((0, 0, 16, 16)) == "16-32"
    assert size_bin((0, 0, 40, 30)) == ">=32"
