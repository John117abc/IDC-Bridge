#!/bin/bash
set -u
A=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
GN="jiangchengxuan"
GE="jiangchengxuan@users.noreply.github.com"
cd "$A" || exit 1

echo "=== 1) 放权重到 assets/ ==="
mkdir -p assets
cp -f /mnt/2T_HDD/plant2_vector/ckpt_conf/best.pt assets/plant2_ckpt_conf_best.pt
ls -la assets/plant2_ckpt_conf_best.pt
sha256sum assets/plant2_ckpt_conf_best.pt

echo "=== 2) 检查是否被 .gitignore 忽略 ==="
git check-ignore -v assets/plant2_ckpt_conf_best.pt || echo "未被忽略（可直接 add）"
echo "--- 现有 .gitignore 里和 assets/ckpt 相关的行 ---"
grep -nE 'assets|ckpt|\.pt$' .gitignore 2>/dev/null | head -5

echo "=== 3) 提交 ==="
git add assets/plant2_ckpt_conf_best.pt tools/fetch_artifacts.sh
git status --short | head -5
git -c user.name="$GN" -c user.email="$GE" commit -q -m "chore: 提供闭环规划器权重 (14MB) 与产物获取脚本

- assets/plant2_ckpt_conf_best.pt: ep40 + 置信度头，开环 minFDE 0.375m / top-1 FDE 0.883m
  sha256 c53dd6278f701dd3b873e2fff4ca9ebf954b4f3c9280b8e102e5bce19aa56611
- tools/fetch_artifacts.sh: 一键取权重(仓库内或 \$CKPT_URL) / 导出 CARLA 地图 /
  构建向量数据集 / 训练 / 校验，保证从数据完全可复现"
git log --oneline -1

echo "=== 4) push ==="
git push 2>&1 | tail -4

echo "=== 5) 校验脚本 ==="
bash tools/fetch_artifacts.sh check 2>&1 | tail -8
