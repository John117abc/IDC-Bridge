#!/bin/bash
set -u
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
S=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2

echo "=== 关键补丁是否在文件里 ==="
for f in $R/tools/plant_agent_idc.py $S/carla_garage/plant_agent_idc.py; do
  echo -n "$(basename $(dirname $f))/$(basename $f): "
  echo -n "LON_REF=$(grep -c 'VEC_IDC_LONG_REF' $f) "
  echo -n "PLAN2=$(grep -c 'VEC_REF_MODE' $f) "
  echo "SMOOTH=$(grep -c 'VEC_VTGT_DMAX' $f)"
done
for f in $R/tools/carla_pid_exec.py $S/carla_garage/carla_pid_exec.py; do
  echo -n "$(basename $(dirname $f))/$(basename $f): "
  echo "LON_RATIO=$(grep -c 'VEC_LON_RATIO' $f)"
done
echo
echo "=== IDC-Bridge 分支/远程 ==="
cd $R && git branch --show-current && git remote -v | head -2
echo "=== plant2 分支/远程 ==="
cd $S && git branch --show-current && git remote -v | head -2
echo
echo "=== carla_pid_exec 在 IDC-Bridge 的 git 状态 ==="
cd $R && git status --porcelain -- tools/carla_pid_exec.py; echo "(空=无改动或未跟踪)"
git ls-files --error-unmatch tools/carla_pid_exec.py >/dev/null 2>&1 && echo "tracked" || echo "untracked"
