"""YOLO26 detektor egitimi (Anti-UAV410 train, model secimi val).

Iki aday onceden sabit:
  s    : yolo26s.yaml     (P3-P5 baslik, stride 8-32)
  s-p2 : yolo26-p2.yaml   (ek P2 baslik, stride 4; <16 px hedefler icin)
Ikisi de COCO onegitimli yolo26s.pt agirliklariyla baslar (eslesen katmanlar aktarilir).

Kullanim: python scripts/train_yolo.py --variant s --epochs 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO, settings

VARIANTS = {
    "s": "yolo26s.yaml",
    "s-p2": "yolo26s-p2.yaml",  # olcek harfi dosya adindan okunur -> yolo26-p2.yaml, scale=s
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--data", default="data/yolo/dataset.yaml")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0, help="hiz olcumu icin veri orani")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    # Ultralytics MLflow geri cagrisi yerel dosya deposunda hata veriyor; kapali.
    settings.update({"mlflow": False, "wandb": False, "tensorboard": False})

    model = YOLO(VARIANTS[args.variant]).load("yolo26s.pt")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        fraction=args.fraction,
        project=str(Path("runs").resolve()),
        name=args.name or f"yolo26{args.variant}",
        exist_ok=True,
        seed=0,
        deterministic=False,
        cos_lr=True,
        close_mosaic=3,
        # Termal goruntu tek kanalli; renk artirmalari anlamsiz, yalniz parlaklik oynasin.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        fliplr=0.5,
        plots=True,
    )


if __name__ == "__main__":
    main()
