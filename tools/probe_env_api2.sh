#!/bin/bash
# 看 env 提供哪些观测张量 + 形状
cd /gpudrive || exit 1
echo "=== env_torch 里的 *_tensor 方法 ==="
grep -n "def .*tensor" gpudrive/env/env_torch.py | head -30
echo "=== sim 侧（gpudrive_sim / sim helper）==="
grep -rn "def .*_tensor" gpudrive/env/*.py gpudrive/env/**/*.py 2>/dev/null | head -30
echo "=== 现有 builder 用了哪些 ==="
cd /gpudrive/baselines/IDC-Bridge || exit 1
grep -n "tensor()" src/env/idc_state_builder.py | head -20
