#!/bin/bash
# 把续训日志的 epoch 编号 +4，与中断前的日志拼成完整 8 epoch 曲线
set -e
cd /workspace/data/logs
sed -e 's/^ep   1 /ep   5 /' -e 's/^ep   2 /ep   6 /' \
    -e 's/^ep   3 /ep   7 /' -e 's/^ep   4 /ep   8 /' \
    ab60k_B_s1_resume.log > b_s1_r.log
cat ab60k_B_s1.log b_s1_r.log > ab60k_B_s1_full.log
echo "epoch 行数: $(grep -c '^ep' ab60k_B_s1_full.log)"
tail -3 ab60k_B_s1_full.log
cd /gpudrive/baselines/IDC-Bridge
/gpudrive/.venv/bin/python tools/plot_ab_60k.py \
    --a-eval /workspace/data/ab60k_A_s1/eval.npz \
    --b-eval /workspace/data/ab60k_B_s1/eval.npz \
    --a-log /workspace/data/logs/ab60k_A_s1.log \
    --b-log /workspace/data/logs/ab60k_B_s1_full.log \
    --out /workspace/data/ab60k/ab_curves_s1.png
