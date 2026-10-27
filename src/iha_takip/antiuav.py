"""Anti-UAV410 etiket okuyucu.

Her dizi klasorunde 000001.jpg ... ve IR_label.json bulunur. JSON alanlari:
  exist   : kare basina 0/1 (hedef gorunur mu)
  gt_rect : kare basina [x, y, w, h] (sol ust kose, piksel); hedef yoksa [] olabilir
  TS, FM, OC, DBC, ... : kare basina 0/1 oznitelik bayraklari
Veride kucuk tutarsizliklar var (exist=1 ama kutu bos ya da sifir boyutlu).
Bunlar 'gecersiz' sayilir ve egitimde de, degerlendirmede de dislanir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ATTRS = ("LR", "VE", "OC", "FM", "SV", "IC", "DBC", "TS")


@dataclass
class Frame:
    index: int  # 0 tabanli; dosya adi f"{index + 1:06d}.jpg"
    state: str  # "present" | "absent" | "invalid"
    box: tuple[float, float, float, float] | None  # x, y, w, h
    attrs: dict[str, int] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return f"{self.index + 1:06d}.jpg"


def _valid_box(r) -> bool:
    return isinstance(r, (list, tuple)) and len(r) == 4 and r[2] > 0 and r[3] > 0


def parse_label(data: dict) -> list[Frame]:
    exist = data["exist"]
    rects = data["gt_rect"]
    frames = []
    for i, e in enumerate(exist):
        r = rects[i] if i < len(rects) else []
        attrs = {a: int(data[a][i]) for a in ATTRS if a in data and i < len(data[a])}
        if e and _valid_box(r):
            frames.append(Frame(i, "present", tuple(float(v) for v in r), attrs))
        elif not e:
            frames.append(Frame(i, "absent", None, attrs))
        else:
            frames.append(Frame(i, "invalid", None, attrs))
    return frames


def load_sequence(seq_dir: Path) -> list[Frame]:
    with open(Path(seq_dir) / "IR_label.json", encoding="utf-8") as f:
        return parse_label(json.load(f))


def to_yolo_line(box, img_w: int, img_h: int, cls: int = 0) -> str:
    """[x, y, w, h] piksel -> 'cls cx cy w h' normalize; goruntu disina tasan kisim kirpilir."""
    x, y, w, h = box
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(img_w), x + w), min(float(img_h), y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"kutu goruntu disinda: {box}")
    cx, cy = (x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h
    return f"{cls} {cx:.6f} {cy:.6f} {(x1 - x0) / img_w:.6f} {(y1 - y0) / img_h:.6f}"


def list_sequences(root: Path, split: str) -> list[str]:
    return sorted(p.name for p in (Path(root) / split).iterdir() if (p / "IR_label.json").exists())
