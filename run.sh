#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"

# ICGNN baseline across the 12 dataset/noise settings.
python train.py --dataset cs --ptb_rate 0.2 --noise uniform --label_rate 0.01 --K 50 --local_conflict_weight 0.9 --warmup_epochs 25 --scale1 1.0 --p_u 0.9 --pseudo_type mix --temp 1.0 --alpha 1.0 --c 1.0 --runs 5
python train.py --dataset cs --ptb_rate 0.2 --noise pair --label_rate 0.01 --K 50 --local_conflict_weight 0.9 --warmup_epochs 25 --scale1 1.0 --p_u 0.9 --pseudo_type mix --temp 1.0 --alpha 1.0 --c 1.0 --runs 5
python train.py --dataset citeseer --ptb_rate 0.2 --noise uniform --label_rate 0.05 --K 50 --local_conflict_weight 0.8 --warmup_epochs 15 --scale1 1.0 --pseudo_type mix --temp 0.1 --alpha 1.0 --c 1.0 --runs 5
python train.py --dataset citeseer --ptb_rate 0.2 --noise pair --label_rate 0.05 --K 50 --local_conflict_weight 0.8 --warmup_epochs 15 --scale1 1.0 --pseudo_type mix --temp 0.1 --alpha 1.0 --c 1.0 --runs 5
# only use neighbor smoothing
python train.py --dataset photo --ptb_rate 0.2 --noise uniform --label_rate 0.01 --K 100 --local_conflict_weight 0.8 --warmup_epochs 15 --temp 1.0 --alpha 1.0 --c 1.0 --runs 5
python train.py --dataset photo --ptb_rate 0.2 --noise pair --label_rate 0.01 --K 100 --local_conflict_weight 0.8 --warmup_epochs 15 --temp 1.0 --alpha 1.0 --c 1.0 --runs 5
python train.py --dataset pubmed --ptb_rate 0.2 --noise uniform --label_rate 0.01 --K 75 --local_conflict_weight 0.8 --warmup_epochs 30 --scale1 1.0 --temp 0.5 --alpha 1.0 --c 1.0 --runs 5
python train.py --dataset pubmed --ptb_rate 0.2 --noise pair --label_rate 0.01 --K 75 --local_conflict_weight 0.8 --warmup_epochs 30 --scale1 1.0 --temp 0.5 --alpha 1.0 --c 1.0 --runs 5
python train.py --dataset dblp --ptb_rate 0.2 --noise uniform --label_rate 0.01 --K 50 --local_conflict_weight 0.9 --warmup_epochs 30 --scale1 1.0 --temp 0.01 --alpha 1.0 --c 1.0 --runs 5
python train.py --dataset dblp --ptb_rate 0.2 --noise pair --label_rate 0.01 --K 50 --local_conflict_weight 0.9 --warmup_epochs 30 --scale1 1.0 --temp 0.01 --alpha 1.0 --c 1.0 --runs 5
python train.py --dataset cora --ptb_rate 0.2 --noise uniform --label_rate 0.05 --K 50 --local_conflict_weight 0.9 --warmup_epochs 30 --scale1 1.0 --temp 0.01 --alpha 1.0 --c 1.0 --runs 5
python train.py --dataset cora --ptb_rate 0.2 --noise pair --label_rate 0.05 --K 50 --local_conflict_weight 0.9 --warmup_epochs 30 --scale1 1.0 --temp 0.01 --alpha 1.0 --c 1.0 --runs 5
