import pytest

from iha_takip.antiuav import parse_label, to_yolo_line


def test_parse_label_states():
    data = {
        "exist": [1, 0, 1, 1],
        "gt_rect": [[10, 20, 8, 6], [], [], [5, 5, 0, 4]],
        "TS": [1, 0, 0, 0],
    }
    frames = parse_label(data)
    assert [f.state for f in frames] == ["present", "absent", "invalid", "invalid"]
    assert frames[0].box == (10.0, 20.0, 8.0, 6.0)
    assert frames[0].attrs == {"TS": 1}
    assert frames[1].filename == "000002.jpg"


def test_parse_label_short_rect_list():
    frames = parse_label({"exist": [0, 1], "gt_rect": [[]]})
    assert [f.state for f in frames] == ["absent", "invalid"]


def test_to_yolo_line_center_and_size():
    line = to_yolo_line((310, 246, 20, 20), 640, 512)
    cls, cx, cy, w, h = line.split()
    assert cls == "0"
    assert float(cx) == pytest.approx(0.5, abs=1e-6)
    assert float(cy) == pytest.approx(0.5, abs=1e-6)
    assert float(w) == pytest.approx(20 / 640, abs=1e-6)
    assert float(h) == pytest.approx(20 / 512, abs=1e-6)


def test_to_yolo_line_clips_to_image():
    _, cx, _, w, _ = to_yolo_line((-4, 0, 10, 10), 640, 512).split()
    assert float(w) == pytest.approx(6 / 640, abs=1e-6)
    assert float(cx) == pytest.approx(3 / 640, abs=1e-6)


def test_to_yolo_line_outside_raises():
    with pytest.raises(ValueError):
        to_yolo_line((700, 10, 5, 5), 640, 512)
