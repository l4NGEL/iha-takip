"""Tek hedefli kareler icin tespit metrikleri.

Anti-UAV'de her karede en fazla bir IHA vardir. Bir tespit, kutusunun gercek
kutuyla IoU'su esigi gectiginde dogru (TP) sayilir; ayni karedeki ikinci
eslesen tespit yanlis (FP) sayilir (COCO/VOC ile ayni kural).
"""

from __future__ import annotations

import numpy as np


def iou_xywh(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_frame(dets: np.ndarray, gt, iou_thr: float = 0.5) -> np.ndarray:
    """dets: (N,5) [x, y, w, h, skor], skora gore azalan sirada olmasi gerekmez.
    gt: [x, y, w, h] ya da None. Doner: (N,) bool, her tespit TP mi."""
    tp = np.zeros(len(dets), dtype=bool)
    if gt is None or len(dets) == 0:
        return tp
    order = np.argsort(-dets[:, 4])
    for i in order:
        if iou_xywh(dets[i, :4], gt) >= iou_thr:
            tp[i] = True
            break  # tek hedef: en yuksek skorlu eslesen TP, gerisi FP
    return tp


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """Tum noktalarda enterpolasyonlu AP (VOC 2010+ / COCO tek esik)."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp_c = np.cumsum(tp[order])
    fp_c = np.cumsum(~tp[order])
    recall = tp_c / n_gt
    precision = tp_c / np.maximum(tp_c + fp_c, 1)
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def pr_at_threshold(scores: np.ndarray, tp: np.ndarray, n_gt: int, thr: float) -> tuple[float, float]:
    keep = scores >= thr
    n_tp = int(tp[keep].sum())
    n_det = int(keep.sum())
    precision = n_tp / n_det if n_det else 1.0
    recall = n_tp / n_gt if n_gt else float("nan")
    return precision, recall


def best_f1_threshold(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> tuple[float, float]:
    """F1'i en buyuk yapan skor esigi (val'de secilir, test'e aynen tasinir)."""
    order = np.argsort(-scores, kind="stable")
    s, t = scores[order], tp[order]
    tp_c, fp_c = np.cumsum(t), np.cumsum(~t)
    prec = tp_c / (tp_c + fp_c)
    rec = tp_c / n_gt
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    # Esit skorlu tespitler ayni esikte birlikte girer; yalniz grup sonlari aday.
    last = np.r_[s[1:] != s[:-1], True]
    i = int(np.argmax(np.where(last, f1, -1)))
    return float(s[i]), float(f1[i])


def size_bin(box) -> str:
    s = (box[2] * box[3]) ** 0.5
    return "<16" if s < 16 else ("16-32" if s < 32 else ">=32")
