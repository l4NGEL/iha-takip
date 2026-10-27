import pytest

from iha_takip.antiuav import parse_label
from iha_takip.tracking_metrics import bootstrap_diff, score_sequence, summarize


def frames():
    return parse_label({
        "exist": [1, 1, 0, 0, 1],
        "gt_rect": [[0, 0, 10, 10], [0, 0, 10, 10], [], [], []],  # son kare gecersiz
        "TS": [1, 0, 0, 0, 0],
    })


def test_state_accuracy_rules():
    preds = {0: (0, 0, 10, 10), 2: (5, 5, 3, 3)}  # kare 1 kacirildi, kare 2 yanlis alarm, kare 3 dogru bos
    s = score_sequence("x", frames(), preds)
    assert s.n_frames == 4  # gecersiz kare disarida
    assert s.per_frame.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert s.sa == pytest.approx(0.5)


def test_summary_fields():
    s = score_sequence("x", frames(), {0: (0, 0, 10, 10), 1: (5, 0, 10, 10)})
    r = summarize([s])
    assert r["mean_iou_on_present"] == pytest.approx((1 + 50 / 150) / 2, abs=1e-4)
    assert r["success_iou50_on_present"] == pytest.approx(0.5)
    assert r["absent_correct_rate"] == pytest.approx(1.0)
    assert r["output_rate_on_present"] == pytest.approx(1.0)
    assert r["attributes"]["TS"]["n_frames"] == 1


def test_bootstrap_diff_sign():
    a = [score_sequence(f"s{i}", frames(), {}) for i in range(5)]
    b = [score_sequence(f"s{i}", frames(), {0: (0, 0, 10, 10), 1: (0, 0, 10, 10)}) for i in range(5)]
    d = bootstrap_diff(a, b, n=200)
    assert d["diff"] == pytest.approx(0.5)
    assert d["b_better_in"] == 5
