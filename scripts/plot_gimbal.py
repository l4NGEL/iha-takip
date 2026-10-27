"""gimbal_sim izinden (seed 0) hedef/gimbal azimutu ve isaretleme hatasi grafigi."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="results/gimbal/trace_6ms.csv")
    ap.add_argument("--out", default="docs/gimbal_sim.png")
    a = ap.parse_args()
    d = defaultdict(lambda: defaultdict(list))
    with open(a.trace, newline="") as f:
        for r in csv.DictReader(f):
            for k in ("t", "target_az", "gimbal_az", "err_deg"):
                d[r["ctrl"]][k].append(float(r[k]))
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    t = d["raw"]["t"]
    ax[0].plot(t, d["raw"]["target_az"], "k-", lw=1.2, label="hedef")
    names = {"raw": "ham tespit + PID", "track": "takip + PID", "lead": "takip + gecikme telafisi + ileri besleme"}
    for c, col in (("raw", "tab:red"), ("track", "tab:orange"), ("lead", "tab:blue")):
        ax[0].plot(d[c]["t"], d[c]["gimbal_az"], color=col, lw=0.9, label=names[c])
        ax[1].plot(d[c]["t"], d[c]["err_deg"], color=col, lw=0.8, label=names[c])
    ax[0].set_ylabel("azimut (derece)")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[1].set_ylabel("isaretleme hatasi (derece)")
    ax[1].set_xlabel("zaman (s)")
    ax[1].set_ylim(0, 5)
    fig.suptitle("Kapali dongu gimbal benzetimi: 30 fps, 6 ms hat gecikmesi, %20 tespit kaybi, 1 px gurultu", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print("yazildi:", a.out)


if __name__ == "__main__":
    main()
