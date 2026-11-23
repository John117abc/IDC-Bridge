#!/bin/bash
set -u
A=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
GN="jiangchengxuan"
GE="jiangchengxuan@users.noreply.github.com"
cd "$A" || exit 1

echo "=== 1) .gitignore 加例外 ==="
grep -q '^!assets/\*.pt' .gitignore 2>/dev/null || printf '\n# 允许提交的权重（规划器产物）\n!assets/*.pt\n' >> .gitignore
tail -4 .gitignore

echo "=== 2) 强制加入权重 + 其它本次文件 ==="
git add -f assets/plant2_ckpt_conf_best.pt
git add .gitignore tools/fetch_artifacts.sh tools/git_assets.sh \
        src/metrics/openloop_pdms_vec.py src/metrics/openloop_pdms_torch.py \
        src/env/candidate_sampler.py 2>/dev/null
git status --short | head -10

echo "=== 3) 提交 ==="
git -c user.name="$GN" -c user.email="$GE" commit -q -m "chore: 权重入库 + .gitignore 例外 + 补齐 PDMS/candidate_sampler

- assets/plant2_ckpt_conf_best.pt (14MB, sha256 c53dd627...6611): ep40 + 置信度头
- .gitignore: 放行 assets/*.pt（其余 *.pt 仍忽略）
- src/metrics/openloop_pdms_{vec,torch}.py: 向量化 PDMS（CPU/GPU，与 score_trajectory 位级一致）
- src/env/candidate_sampler.py: install_external（外部参考注入）
- tools/fetch_artifacts.sh: 取权重/建地图/建数据/训练/校验"
git log --oneline -1

echo "=== 4) push ==="
git push 2>&1 | tail -4

echo "=== 5) 确认权重已在仓库里 ==="
git ls-files -s assets/plant2_ckpt_conf_best.pt
git cat-file -s $(git rev-parse HEAD:assets/plant2_ckpt_conf_best.pt) 2>/dev/null

echo "=== 6) 测试 fetch 脚本从仓库取权重 ==="
bash tools/fetch_artifacts.sh ckpt 2>&1 | tail -4
