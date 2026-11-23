#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
C=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

echo "=== checkpoint 内容（直接打印）==="
docker exec -i idc_hist /gpudrive/.venv/bin/python -c "
import torch
ck = torch.load('$C', map_location='cpu')
print('TOP', list(ck.keys()))
for k, v in ck.items():
    if hasattr(v, 'keys'):
        print(k, '->', len(v), 'tensors; first:', list(v.keys())[:3])
        for kk in list(v.keys())[:3]:
            try: print('   ', kk, tuple(v[kk].shape))
            except Exception: pass
" 2>&1 | head -18

echo
echo "=== 谁在组装 ref_error（搜 dphi / DIM_REF_ERROR）==="
grep -rIln --include=*.py -e "dphi" -e "DIM_REF_ERROR" "$R/src" | head -6
grep -rIn --include=*.py -e "def .*state" -e "ref_error" "$R/src/env/env_utils.py" "$R/src/env/candidate_sampler.py" 2>/dev/null | head -10
