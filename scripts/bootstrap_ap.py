"""Iki tahmin dosyasinin AP50 farki icin dizi duzeyinde bootstrap guven araligi.

Ayni dizideki kareler bagimsiz degildir; bu yuzden yeniden ornekleme birimi
karedir degil dizidir.

  python scripts/bootstrap_ap.py --a results/det/s_val.npz --b results/det/s-p2_val.npz --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_detector import load_pred  # noqa: E402
from iha_takip.antiuav import load_sequence  # noqa: E402
from iha_takip.metrics import average_precision, match_frame, size_bin  # noqa: E402


def per_seq(pred_path: Path, root: Path, split: str, group: str):
    """Dizi -> (skorlar, tp, n_gt) ; group 'all' ya da 'size<16'."""
    pred = load_pred(pred_path)
    frames_by_seq = defaultdict(list)
    for seq, fi in pred:
        frames_by_seq[seq].append(fi)
    out = {}
    for seq, idxs in frames_by_seq.items():
        frames = {f.index: f for f in load_sequence(root / split / seq)}
        s, t, n = [], [], 0
        for fi in idxs:
            fr = frames[fi]
            if fr.state == "invalid":
                continue
            gt = fr.box if fr.state == "present" else None
            if group != "all" and (gt is None or "size" + size_bin(gt) != group):
                continue
            d = pred[(seq, fi)]
            tp = match_frame(d, gt)
            n += gt is not None
            s.extend(d[:, 4].tolist())
            t.extend(tp.tolist())
        out[seq] = (np.array(s), np.array(t, dtype=bool), n)
    return out


def ap_of(parts, seqs):
    s = np.concatenate([parts[q][0] for q in seqs])
    t = np.concatenate([parts[q][1] for q in seqs])
    return average_precision(s, t, sum(parts[q][2] for q in seqs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--root", type=Path, default=Path("data/antiuav410"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    report = {"a": str(args.a), "b": str(args.b), "split": args.split, "n_boot": args.n}
    for group in ("all", "size<16"):
        A = per_seq(args.a, args.root, args.split, group)
        B = per_seq(args.b, args.root, args.split, group)
        seqs = sorted(set(A) & set(B))
        base = ap_of(B, seqs) - ap_of(A, seqs)
        diffs = []
        for _ in range(args.n):
            smp = list(rng.choice(seqs, size=len(seqs), replace=True))
            diffs.append(ap_of(B, smp) - ap_of(A, smp))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        report[group] = {"AP_a": round(ap_of(A, seqs), 4), "AP_b": round(ap_of(B, seqs), 4),
                         "diff_b_minus_a": round(base, 4), "ci95": [round(lo, 4), round(hi, 4)],
                         "n_sequences": len(seqs)}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
