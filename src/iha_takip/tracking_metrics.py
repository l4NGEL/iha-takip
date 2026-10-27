"""Anti-UAV410 durum dogrulugu (state accuracy, SA).

Kare basina:
  hedef gorunur (exist=1) : tahmin kutusu ile IoU (tahmin yoksa 0)
  hedef yok      (exist=0) : tahmin de yoksa 1, varsa 0
SA bir dizinin kare ortalamasidir; rapor edilen deger dizi ortalamalarinin ortalamasidir
(Anti-UAV410 makalesi ve resmi degerlendirme kodu gibi). Gecersiz etiketli kareler
(exist=1 ama kutu bos) her iki tarafta da sayilmaz.

Not: Anti-UAV410'daki SOT izleyicileri ilk karede gercek kutuyla baslatilir. Bu projedeki hat
tespit tabanlidir ve hicbir gercek kutu almaz; sayilar o tablolarla dogrudan kiyaslanamaz.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from iha_takip.antiuav import Frame
from iha_takip.metrics import iou_xywh


@dataclass
class SeqScore:
    name: str
    sa: float
    n_frames: int
    per_frame: np.ndarray  # kare basina skor (gecersizler haric)
    attrs: dict[str, np.ndarray] = field(default_factory=dict)  # oznitelik maskeleri
    present: np.ndarray | None = None  # hedef gorunur maskesi
    pred_present: np.ndarray | None = None


def score_sequence(name: str, frames: list[Frame], preds: dict[int, tuple[float, float, float, float]]) -> SeqScore:
    """preds: kare no -> (x, y, w, h); karede tahmin yoksa anahtar yok."""
    vals, present, pred_present = [], [], []
    attr_masks: dict[str, list[bool]] = {}
    for fr in frames:
        if fr.state == "invalid":
            continue
        p = preds.get(fr.index)
        if fr.state == "present":
            vals.append(iou_xywh(p, fr.box) if p is not None else 0.0)
        else:
            vals.append(1.0 if p is None else 0.0)
        present.append(fr.state == "present")
        pred_present.append(p is not None)
        for a, v in fr.attrs.items():
            attr_masks.setdefault(a, []).append(bool(v))
    arr = np.array(vals, dtype=float)
    return SeqScore(
        name=name,
        sa=float(arr.mean()) if len(arr) else float("nan"),
        n_frames=len(arr),
        per_frame=arr,
        attrs={k: np.array(v) for k, v in attr_masks.items()},
        present=np.array(present),
        pred_present=np.array(pred_present),
    )


def summarize(scores: list[SeqScore], attrs=("TS", "FM", "DBC", "SV", "OC")) -> dict:
    sa = np.array([s.sa for s in scores])
    allv = np.concatenate([s.per_frame for s in scores])
    pres = np.concatenate([s.present for s in scores])
    pred = np.concatenate([s.pred_present for s in scores])
    out = {
        "n_sequences": len(scores),
        "n_frames": int(len(allv)),
        "SA_mean_of_sequences": round(float(sa.mean()), 4),
        "SA_frame_level": round(float(allv.mean()), 4),
        "success_iou50_on_present": round(float((allv[pres] >= 0.5).mean()), 4),
        "mean_iou_on_present": round(float(allv[pres].mean()), 4),
        "absent_frames": int((~pres).sum()),
        "absent_correct_rate": round(float(allv[~pres].mean()), 4) if (~pres).any() else None,
        "output_rate_on_present": round(float(pred[pres].mean()), 4),
        "attributes": {},
    }
    for a in attrs:
        m = np.concatenate([s.attrs.get(a, np.zeros(s.n_frames, bool)) for s in scores]) & pres
        if m.any():
            out["attributes"][a] = {"n_frames": int(m.sum()), "mean_iou": round(float(allv[m].mean()), 4)}
    return out


def bootstrap_diff(a: list[SeqScore], b: list[SeqScore], n: int = 1000, seed: int = 0) -> dict:
    """Ayni dizilerde b - a SA farki icin dizi duzeyinde bootstrap %95 araligi."""
    da = {s.name: s.sa for s in a}
    db = {s.name: s.sa for s in b}
    names = sorted(set(da) & set(db))
    d = np.array([db[k] - da[k] for k in names])
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_sequences": len(names), "b_better_in": int((d > 0).sum()), "b_worse_in": int((d < 0).sum())}
