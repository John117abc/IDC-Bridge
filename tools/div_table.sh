#!/bin/bash
# 汇总 A / B / B+hinge 的多样性 + oracle + minFDE
set -e
cd /gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python

show() {  # $1=名字 $2=ckpt $3=pdms npz（可空）
  echo "---------- $1 ----------"
  if [ -n "$3" ] && [ -f "$3" ]; then
    $PY tools/eval_div.py --ckpt "$2" --tag "$1" --pdms-npz "$3" 2>/dev/null | tail -7
    $PY tools/ab_stats.py --a /workspace/data/ab60k_A/eval.npz --b "$3" 2>/dev/null \
      | grep -E "^oracle|全模态 PDMS|逐场景全模态|PDMS>0|PDMS>60|minFDE p50" || true
  else
    $PY tools/eval_div.py --ckpt "$2" --tag "$1" 2>/dev/null | tail -7
  fi
}

show "A  (60k, WTA+div)"  /workspace/data/ab60k_A/latest.pt   /workspace/data/ab60k_A/eval.npz
show "B  (60k, +IAWR)"    /workspace/data/ab60k_B/latest.pt   /workspace/data/ab60k_B/eval.npz
show "bh05 (IAWR+hinge .05)" /workspace/data/bh/bh05/latest.pt /workspace/data/bh/bh05/eval.npz
show "bh10 (IAWR+hinge .10)" /workspace/data/bh/bh10/latest.pt /workspace/data/bh/bh10/eval.npz
