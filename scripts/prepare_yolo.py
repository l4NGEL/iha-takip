"""Anti-UAV410 -> Ultralytics YOLO formati.

Resmi bolum korunur: train (200 dizi) egitim, val (90 dizi) model secimi,
test (120 dizi) yalniz son degerlendirme icin; test burada donusturulmez.

Ardisik kareler neredeyse ayni oldugu icin train/val her `--stride` karede bir
orneklenir. Hedefin olmadigi kareler (exist=0) az oldugu (%2) ve yanlis alarmi
ogrenmek icin gerekli oldugu icin egitimde stride'dan bagimsiz hepsi alinir;
val dogal dagilimda kalir (ayni stride).
Gorseller kopyalanmaz, NTFS/ext4 hard link ile baglanir.

Kullanim: python scripts/prepare_yolo.py --root data/antiuav410 --out data/yolo
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iha_takip.antiuav import list_sequences, load_sequence, to_yolo_line  # noqa: E402


def link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert(root: Path, out: Path, split: str, stride: int, all_absent: bool) -> dict:
    img_dir, lbl_dir = out / split / "images", out / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    stats = {"present": 0, "absent": 0, "skipped_invalid": 0, "sequences": 0}
    seqs = list_sequences(root, split)
    for seq in seqs:
        seq_dir = root / split / seq
        frames = load_sequence(seq_dir)
        first = cv2.imread(str(seq_dir / frames[0].filename), cv2.IMREAD_GRAYSCALE)
        h, w = first.shape[:2]
        for fr in frames:
            if fr.state == "invalid":
                stats["skipped_invalid"] += 1
                continue
            if fr.index % stride and (fr.state == "present" or not all_absent):
                continue
            src = seq_dir / fr.filename
            if not src.exists():
                continue
            try:
                text = to_yolo_line(fr.box, w, h) + "\n" if fr.state == "present" else ""
            except ValueError:
                stats["skipped_invalid"] += 1
                continue
            stem = f"{seq}_{fr.index + 1:06d}"
            link(src, img_dir / f"{stem}.jpg")
            (lbl_dir / f"{stem}.txt").write_text(text)
            stats[fr.state] += 1
        stats["sequences"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/antiuav410"))
    ap.add_argument("--out", type=Path, default=Path("data/yolo"))
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--val-stride", type=int, default=20)
    args = ap.parse_args()

    summary = {
        "train": convert(args.root, args.out, "train", args.stride, all_absent=True),
        "val": convert(args.root, args.out, "val", args.val_stride, all_absent=False),
        "stride": args.stride,
        "val_stride": args.val_stride,
    }
    (args.out / "dataset.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: train/images\nval: val/images\nnames:\n  0: uav\n"
    )
    print(json.dumps(summary, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path("results/prepare_yolo.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
