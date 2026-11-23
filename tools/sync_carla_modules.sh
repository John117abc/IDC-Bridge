#!/bin/bash
set -u
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
P=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python

echo "=== 1) 打补丁：src 路径 ==="
$PY $R/tools/patch_idc_src_path.py 2>&1 | tail -2

echo "=== 2) 复制 4 个模块到 carla_garage ==="
for m in plant_agent_idc idc_carla frames idc_state_carla plant_agent_vec; do
  cp -f $R/tools/$m.py $P/carla_garage/$m.py 2>/dev/null && echo "  copied $m.py"
done
ls -la $P/carla_garage/*.py | head -6

echo "=== 3) 语法检查 ==="
$PY - <<'EOF'
import ast, glob
for p in glob.glob('/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/carla_garage/*.py'):
    try:
        ast.parse(open(p).read())
    except SyntaxError as e:
        print('FAIL', p, e)
print('AST 检查完成')
EOF
