"""C++/TensorRT hattinin Python referansiyla esdegerligi.

  prepare : test kumesinden kucuk bir alt kume (her diziden ilk N kare) hard link ile kurar.
  pre     : C++ CUDA on islem ciktisi (.bin) vs Ultralytics LetterBox + numpy.
  dets    : TensorRT tespitleri (iha_track --dump-dets CSV) vs ONNX Runtime FP32 (CPU).

  python scripts/check_parity.py prepare --out data/parity --n-seq 5 --n-frames 40
  python scripts/check_parity.py pre --image data/parity/<seq>/000001.jpg --bin results/parity/pre.bin
  python scripts/check_parity.py dets --root data/parity --trt-dets results/parity/dets_fp16 --onnx checkpoints/m.onnx
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iha_takip.antiuav import list_sequences  # noqa: E402
from iha_takip.metrics import iou_xywh  # noqa: E402


def letterbox_ref(img: np.ndarray, size: int = 640) -> np.ndarray:
    from ultralytics.data.augment import LetterBox

    lb = LetterBox(new_shape=(size, size), auto=False, scale_fill=False, scaleup=True, center=True)
    out = lb(image=img)
    return out[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0  # BGR->RGB, CHW


def cmd_prepare(a) -> None:
    src = a.root / "test"
    seqs = list_sequences(a.root, "test")
    step = max(1, len(seqs) // a.n_seq)
    chosen = seqs[::step][: a.n_seq]
    for s in chosen:
        d = a.out / s
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, a.n_frames + 1):
            dst = d / f"{i:06d}.jpg"
            if not dst.exists():
                os.link(src / s / f"{i:06d}.jpg", dst)
    print("diziler:", chosen)


def cmd_pre(a) -> None:
    img = cv2.imread(str(a.image), cv2.IMREAD_COLOR)
    ref = letterbox_ref(img)
    got = np.fromfile(a.bin, dtype=np.float32).reshape(ref.shape)
    diff = np.abs(ref - got)
    r = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
         "max_abs_in_uint8_levels": float(diff.max() * 255)}
    print(json.dumps(r, indent=2))
    if a.json:
        a.json.write_text(json.dumps(r, indent=2))


def read_dets(path: Path) -> dict[int, list[tuple]]:
    out = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["frame"])].append(tuple(float(row[k]) for k in ("x", "y", "w", "h", "score")))
    return out


def cmd_dets(a) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    ious, sdiff, agree, n = [], [], 0, 0
    for seq in sorted(p.name for p in a.root.iterdir() if p.is_dir()):
        trt = read_dets(a.trt_dets / f"{seq}.csv")
        frames = sorted((a.root / seq).glob("*.jpg"))
        for fi, fp in enumerate(frames):
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            x = letterbox_ref(img)[None]
            out = sess.run(None, {name: x})[0][0]
            r = min(640 / w, 640 / h)
            left = round((640 - round(w * r)) / 2 - 0.1)
            top = round((640 - round(h * r)) / 2 - 0.1)
            keep = out[:, 4] >= 0.01
            o = out[keep]
            ref = [((b[0] - left) / r, (b[1] - top) / r, (b[2] - b[0]) / r, (b[3] - b[1]) / r, b[4]) for b in o]
            t = trt.get(fi, [])
            n += 1
            rt = max(ref, key=lambda d: d[4]) if ref else None
            tt = max(t, key=lambda d: d[4]) if t else None
            if rt is not None and tt is not None:
                ious.append(iou_xywh(rt[:4], tt[:4]))
                sdiff.append(abs(rt[4] - tt[4]))
            # Ayni esik kararini veriyorlar mi (val'de secilen esik)
            agree += (rt is not None and rt[4] >= a.thr) == (tt is not None and tt[4] >= a.thr)
    ious, sdiff = np.array(ious), np.array(sdiff)
    r = {
        "frames": n,
        "top1_iou_mean": round(float(ious.mean()), 4),
        "top1_iou_min": round(float(ious.min()), 4),
        "top1_iou_ge_0.9": round(float((ious >= 0.9).mean()), 4),
        "score_absdiff_mean": round(float(sdiff.mean()), 4),
        "score_absdiff_max": round(float(sdiff.max()), 4),
        f"same_decision_at_thr_{a.thr}": round(agree / n, 4),
    }
    print(json.dumps(r, indent=2))
    if a.json:
        a.json.write_text(json.dumps(r, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", type=Path, default=Path("data/antiuav410"))
    p.add_argument("--out", type=Path, default=Path("data/parity"))
    p.add_argument("--n-seq", type=int, default=5)
    p.add_argument("--n-frames", type=int, default=40)
    q = sub.add_parser("pre")
    q.add_argument("--image", type=Path, required=True)
    q.add_argument("--bin", type=Path, required=True)
    q.add_argument("--json", type=Path, default=None)
    d = sub.add_parser("dets")
    d.add_argument("--root", type=Path, default=Path("data/parity"))
    d.add_argument("--trt-dets", type=Path, required=True)
    d.add_argument("--onnx", type=Path, required=True)
    d.add_argument("--thr", type=float, default=0.415)
    d.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    {"prepare": cmd_prepare, "pre": cmd_pre, "dets": cmd_dets}[a.cmd](a)


if __name__ == "__main__":
    main()
