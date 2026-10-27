#!/usr/bin/env bash
# Iki aday sirayla: once standart baslik, sonra P2. Ayni ayarlar.
# P2 baslik 16'lik batch'te 8 GB'a sigmiyor; batch 8. Ultralytics nbs=64 ile
# gradyan biriktirdigi icin etkin batch ikisinde de 64.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
for spec in "s 16" "s-p2 8"; do
  set -- $spec; v=$1; bs=$2
  .venv/Scripts/python scripts/train_yolo.py --variant "$v" --batch "$bs" --epochs 12 > "results/train_$v.log" 2>&1
  echo "$v EXIT $?"
done
