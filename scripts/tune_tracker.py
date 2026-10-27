"""Takip ayarlarini val dizilerinde secer (test'e dokunmaz).

Onceden sabit kural: kucuk izgarada val SA'si (dizi ortalamasi) en yuksek yapilandirma.
Izgara: high_thr x min_hits x coast; new_thr = high_thr + 0.1, low_thr = 0.1.
Her yapilandirma C++ iha_track_offline ile (TensorRT tespit dokumu uzerinde) kosulur.

  python scripts/tune_tracker.py --dets results/trk/dets_val_fp16 --split val --out results/trk/tune_val.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_tracking import score_dir  # noqa: E402
from iha_takip.tracking_metrics import summarize  # noqa: E402

GRID = {"high_thr": [0.3, 0.415, 0.5], "min_hits": [1, 2, 3], "coast": [0, 5, 15, 30]}


def run_offline(dets: Path, split: str, out: Path, flags: list[str]) -> None:
    cmd = (f"cpp/build-trt/iha_track_offline --dets-dir {dets.as_posix()} --root /data/antiuav410/{split} "
           f"--out-dir {out.as_posix()} " + " ".join(flags))
    # Kabuk sarmalayicisi yerine dogrudan docker: Windows'ta "bash" WSL'ye cozulebiliyor.
    r = subprocess.run(["docker", "run", "--rm", "-v", f"{Path.cwd()}:/work", "-v", "iha-data:/data", "-w", "/work",
                        "iha-takip:trt", "bash", "-c", cmd], capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(r.stderr[-2000:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", type=Path, default=Path("data/antiuav410"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tmp", type=Path, default=Path("results/trk/_tune_tmp"))
    ap.add_argument("--grid", type=str, default=None, help='JSON, ornek {"high_thr":[0.3],"min_hits":[1],"coast":[5]}')
    ap.add_argument("--extra", type=str, default="", help="tum yapilandirmalara eklenen bayraklar")
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()
    grid = json.loads(args.grid) if args.grid else GRID

    rows = []
    configs = [] if args.no_baseline else [("top1", ["--mode top1 --top1-thr 0.415"])]
    for h, m, c in itertools.product(grid["high_thr"], grid["min_hits"], grid["coast"]):
        configs.append((f"bt_h{h}_m{m}_c{c}" + (f" [{args.extra}]" if args.extra else ""),
                        [f"--high-thr {h} --new-thr {h + 0.1:.3f} --low-thr 0.1 --min-hits {m} --coast {c} {args.extra}"]))
    for i, (name, flags) in enumerate(configs, 1):
        shutil.rmtree(args.tmp, ignore_errors=True)
        run_offline(args.dets, args.split, args.tmp, flags)
        s = summarize(score_dir(args.tmp, args.root, args.split))
        rows.append({"name": name, "flags": " ".join(flags), "SA": s["SA_mean_of_sequences"],
                     "SA_frame": s["SA_frame_level"], "mean_iou_present": s["mean_iou_on_present"],
                     "absent_correct": s["absent_correct_rate"], "output_rate": s["output_rate_on_present"]})
        print(f"[{i}/{len(configs)}] {name}: SA {s['SA_mean_of_sequences']:.4f}", flush=True)
    shutil.rmtree(args.tmp, ignore_errors=True)
    tracked = [r for r in rows if r["name"] != "top1"]
    best = max(tracked, key=lambda r: r["SA"])
    report = {"split": args.split, "rule": "en yuksek val SA (dizi ortalamasi)", "selected": best,
              "baseline_top1": None if args.no_baseline else rows[0], "all": sorted(rows, key=lambda r: -r["SA"])}
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("secilen:", best)


if __name__ == "__main__":
    main()
