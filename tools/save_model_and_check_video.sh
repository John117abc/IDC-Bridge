#!/bin/bash
set -u
D=/mnt/2T_HDD/plant2_vector
SAVE=$D/ckpt_best_100
mkdir -p $SAVE

# 1) 保存模型（100 分那次用的 ckpt）
cp -f $D/ckpt_occpatch_final/best.pt $SAVE/plant2_occpatch_100_20261126.pt
sha256sum $SAVE/plant2_occpatch_100_20261126.pt | tee $SAVE/SHA256.txt

# 2) 附带证据
cp -f $D/eval_occpatch_24759.txt $SAVE/ 2>/dev/null
cp -f $D/eval_occpatch_24206.txt $SAVE/ 2>/dev/null
cp -f $D/ckpt_occpatch2/best.pt $SAVE/plant2_occpatch_241_20261126.pt 2>/dev/null

cat > $SAVE/README.md <<'EOF'
# 100 分模型（plan 模式）

- 文件：plant2_occpatch_100_20261126.pt
- 来源：ckpt_occpatch2/best.pt（occ-aux 逐 patch 监督，stage2）
- val minFDE 0.241 / top1FDE 0.522 / acc 57.7%
- 闭环（plan2 + LONG_REF）：
  - 24759：score 100.0，完成 100%，0 碰撞，均速 6.31 m/s
  - 24206：score 60.0，完成 100%，1 碰撞，均速 4.26 m/s
- 加载方式：agent 读 ckpt 内 use_occ=True / occ_pos=False 自动匹配
- 复现：VEC_REF_FROM=plan VEC_REF_MODE=plan2 VEC_IDC_LONG_REF=1
EOF
echo "--- 保存目录 ---"
ls -la --time-style=+%H:%M $SAVE

# 3) 看录制产物（GIF 用）
echo "=== video 目录 ==="
ls $D/video 2>/dev/null | head -3
echo "=== 最近 run 目录 ==="
RD=$(ls -dt $D/closedloop/fast_* | head -1)
echo "$RD"
ls -la "$RD" | head -6
echo "=== 是否有图片/相机帧 ==="
find "$RD" -maxdepth 1 -type d | head -5
find "$RD" -maxdepth 2 -name "*.png" 2>/dev/null | head -3
