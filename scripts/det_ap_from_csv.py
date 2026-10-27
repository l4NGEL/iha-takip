"""C++ tespit dokumlerinden (dizi basina CSV) AP50; FP16 / INT8 motorlarini kiyaslamak icin.

  python scripts/det_ap_from_csv.py --dets results/trk/dets_val_fp16 results/trk/dets_val_int8 --split val
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iha_takip.antiuav import list_sequences, load_sequence  # noqa: E402
from iha_takip.metrics import average_precision, match_frame, pr_at_threshold, size_bin  # noqa: E402


def ap_for(dets_dir: Path, root: Path, split: str, thr: float) -> dict:
    groups = {"all": [[], [], 0], "size<16": [[], [], 0]}
    for seq in list_sequences(root, split):
        per = defaultdict(list)
        with open(dets_dir / f"{seq}.csv", newline="") as f:
            for r in csv.DictReader(f):
                per[int(r["frame"])].append([float(r[k]) for k in ("x", "y", "w", "h", "score")])
        for fr in load_sequence(root / split / seq):
            if fr.state == "invalid":
                continue
            d = np.array(per.get(fr.index, []), dtype=np.float32).reshape(-1, 5)
            gt = fr.box if fr.state == "present" else None
            tp = match_frame(d, gt)
            keys = ["all"] + (["size<16"] if gt is not None and size_bin(gt) == "<16" else [])
            for k in keys:
                groups[k][0].extend(d[:, 4].tolist())
                groups[k][1].extend(tp.tolist())
                groups[k][2] += gt is not None
    out = {}
    for k, (s, t, n) in groups.items():
        s, t = np.array(s), np.array(t, dtype=bool)
        p, r = pr_at_threshold(s, t, n, thr)
        out[k] = {"AP50": round(average_precision(s, t, n), 4), "precision": round(p, 4), "recall": round(r, 4), "n": n}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", type=Path, nargs="+", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--root", type=Path, default=Path("data/antiuav410"))
    ap.add_argument("--thr", type=float, default=0.415)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    rep = {str(d): ap_for(d, a.root, a.split, a.thr) for d in a.dets}
    text = json.dumps(rep, indent=2)
    print(text)
    if a.json:
        a.json.write_text(text)


if __name__ == "__main__":
    main()
