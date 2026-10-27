"""Detektoru Anti-UAV410 dizilerinin TUM karelerinde (ya da her N karede) degerlendirir.

Iki adim:
  predict : kare basina tespitleri (conf >= 0.01, en fazla 10) .npz'ye yazar.
            Bu dosya takip asamasinda da girdi olarak kullanilir.
  score   : AP50, secilen esikte precision/recall, boyut ve oznitelik kirilimi,
            hedefsiz karelerde yanlis alarm.

Esik kurali (onceden sabit): val'de F1'i en buyuk yapan skor esigi secilir ve
test'e degistirilmeden tasinir.

Ornek:
  python scripts/eval_detector.py predict --weights runs/yolo26s/weights/best.pt --split val --stride 5 --out results/det/s_val.npz
  python scripts/eval_detector.py score --pred results/det/s_val.npz --split val --json results/det/s_val.json
  python scripts/eval_detector.py score --pred results/det/s_test.npz --split test --thr-from results/det/s_val.json --json results/det/s_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iha_takip.antiuav import list_sequences, load_sequence  # noqa: E402
from iha_takip.metrics import (  # noqa: E402
    average_precision,
    best_f1_threshold,
    match_frame,
    pr_at_threshold,
    size_bin,
)

ATTR_REPORT = ("TS", "FM", "DBC", "SV", "OC")


def predict(args) -> None:
    from ultralytics import YOLO

    model = YOLO(args.weights)
    seqs = list_sequences(args.root, args.split)
    all_keys, paths = [], []
    for si, seq in enumerate(seqs):
        for f in load_sequence(args.root / args.split / seq):
            if f.index % args.stride == 0:
                all_keys.append((si, f.index))
                paths.append(str(args.root / args.split / seq / f.filename))
    # Gorseller is parcaciklariyla okunur, model numpy yigini alir. Ultralytics'e
    # yol listesi vermek her dizide kaynak kurulumu (~11 kare/sn) ya da tek listede
    # tum dosyalari acip Windows tanitici sinirina takilmaya yol aciyor.
    pool = ThreadPoolExecutor(8)
    chunks = [paths[i:i + args.batch] for i in range(0, len(paths), args.batch)]

    def results():
        nxt = pool.submit(lambda c: list(pool.map(cv2.imread, c)), chunks[0]) if chunks else None
        for ci in range(len(chunks)):
            imgs = nxt.result()
            if ci + 1 < len(chunks):
                nxt = pool.submit(lambda c: list(pool.map(cv2.imread, c)), chunks[ci + 1])
            yield from model.predict(imgs, imgsz=args.imgsz, conf=0.01, max_det=10, half=True, verbose=False)

    keys, boxes, offsets = [], [], [0]
    t0 = time.time()
    for n, (key, r) in enumerate(zip(all_keys, results()), 1):
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        d = np.c_[xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2], conf].astype(np.float32)
        keys.append(key)
        boxes.append(d)
        offsets.append(offsets[-1] + len(d))
        if n % 5000 == 0 or n == len(paths):
            print(f"{n}/{len(paths)} kare, {n / (time.time() - t0):.1f} kare/sn", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        seqs=np.array(seqs),
        keys=np.array(keys, dtype=np.int32).reshape(-1, 2),
        dets=np.concatenate(boxes) if boxes else np.zeros((0, 5), np.float32),
        offsets=np.array(offsets, dtype=np.int64),
        weights=str(args.weights),
        stride=args.stride,
    )
    print("yazildi:", args.out)


def load_pred(path: Path):
    # NpzFile'da her z["x"] erisimi diziyi yeniden acar; donguden once bir kez yukle.
    with np.load(path) as z:
        seqs, keys, dets, offsets = list(z["seqs"]), z["keys"], z["dets"], z["offsets"]
    return {
        (seqs[si], int(fi)): dets[offsets[k]:offsets[k + 1]]
        for k, (si, fi) in enumerate(keys)
    }


def score(args) -> None:
    pred = load_pred(args.pred)
    # Her tespit icin: skor, TP mi, ve bagli oldugu karenin gruplari
    scores, tps, groups = [], [], []
    n_gt = defaultdict(int)
    absent_frames, absent_top = 0, []
    by_seq = defaultdict(list)
    for (seq, fi) in pred:
        by_seq[seq].append(fi)
    for seq, idxs in by_seq.items():
        frames = {f.index: f for f in load_sequence(args.root / args.split / seq)}
        for fi in idxs:
            fr = frames[fi]
            if fr.state == "invalid":
                continue
            d = pred[(seq, fi)]
            gt = fr.box if fr.state == "present" else None
            tp = match_frame(d, gt, args.iou)
            g = {"all"}
            if gt is not None:
                g.add("size" + size_bin(gt))
                g.update(a for a in ATTR_REPORT if fr.attrs.get(a))
                for k in g:
                    n_gt[k] += 1
            else:
                absent_frames += 1
                absent_top.append(float(d[:, 4].max()) if len(d) else 0.0)
            for j in range(len(d)):
                scores.append(float(d[j, 4]))
                tps.append(bool(tp[j]))
                # FP'ler tum gruplarda sayilir (kare grubu); boylece grup AP'si
                # "bu tip karelerde" AP anlamina gelir.
                groups.append(g)
    scores, tps = np.array(scores), np.array(tps)

    if args.thr_from:
        thr = json.loads(Path(args.thr_from).read_text())["threshold"]
        thr_src = str(args.thr_from)
    else:
        thr, _ = best_f1_threshold(scores, tps, n_gt["all"])
        thr_src = "bu kumede en iyi F1"

    report = {"split": args.split, "pred": str(args.pred), "iou": args.iou,
              "threshold": thr, "threshold_source": thr_src, "groups": {}}
    for k in ["all", "size<16", "size16-32", "size>=32", *ATTR_REPORT]:
        if not n_gt[k]:
            continue
        m = np.array([k in g for g in groups])
        p, r = pr_at_threshold(scores[m], tps[m], n_gt[k], thr)
        report["groups"][k] = {
            "n_frames": n_gt[k],
            "AP50": round(average_precision(scores[m], tps[m], n_gt[k]), 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
        }
    absent_top = np.array(absent_top)
    report["absent"] = {
        "n_frames": absent_frames,
        "false_alarm_rate": round(float((absent_top >= thr).mean()), 4) if absent_frames else None,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--pred", type=Path, required=True)
    s.add_argument("--split", required=True, choices=["val", "test"])
    s.add_argument("--iou", type=float, default=0.5)
    s.add_argument("--thr-from", type=Path, default=None)
    s.add_argument("--json", type=Path, default=None)
    for x in (p, s):
        x.add_argument("--root", type=Path, default=Path("data/antiuav410"))
    args = ap.parse_args()
    {"predict": predict, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
