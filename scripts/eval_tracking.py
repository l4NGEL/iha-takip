"""C++ takip ciktilarini (kare basina CSV) Anti-UAV SA ile puanlar.

  python scripts/eval_tracking.py --pred results/trk/val_bytetrack --split val --json results/trk/val_bytetrack.json
  python scripts/eval_tracking.py --pred results/trk/test_top1 --pred-b results/trk/test_bytetrack --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iha_takip.antiuav import list_sequences, load_sequence  # noqa: E402
from iha_takip.tracking_metrics import bootstrap_diff, score_sequence, summarize  # noqa: E402


def read_track_csv(path: Path) -> dict[int, tuple[float, float, float, float]]:
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["valid"] == "1":
                out[int(row["frame"])] = (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
    return out


def score_dir(pred_dir: Path, root: Path, split: str):
    scores = []
    for seq in list_sequences(root, split):
        path = pred_dir / f"{seq}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        scores.append(score_sequence(seq, load_sequence(root / split / seq), read_track_csv(path)))
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--pred-b", type=Path, default=None, help="kiyas: b - a farki icin bootstrap")
    ap.add_argument("--split", required=True)
    ap.add_argument("--root", type=Path, default=Path("data/antiuav410"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    a = score_dir(args.pred, args.root, args.split)
    report = {"pred": str(args.pred), "split": args.split, **summarize(a)}
    if args.pred_b:
        b = score_dir(args.pred_b, args.root, args.split)
        report["b"] = {"pred": str(args.pred_b), **summarize(b)}
        report["b_minus_a"] = bootstrap_diff(a, b)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
