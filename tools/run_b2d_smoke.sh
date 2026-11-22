#!/bin/bash
set -u
# Bench2Drive 闭环 smoke：可指定 RUN_NAME，每次独立目录、不 resume
ROOT=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
WORK_DIR=$ROOT/Bench2Drive
CARLA_ROOT=/home/jiangchengxuan/carla-0.9.15-linux
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
BASE=/mnt/2T_HDD/plant2_vector/closedloop
RUN=${RUN_NAME:-run_$(date +%H%M%S)}
OUT=$BASE/$RUN
mkdir -p "$OUT"

pkill -f "carla-port=2000" 2>/dev/null
sleep 3

export CARLA_ROOT
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla:${WORK_DIR}/leaderboard:${WORK_DIR}/scenario_runner:${ROOT}
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export RECORD_PATH=

ROUTES=${WORK_DIR}/leaderboard/data/two_routes.xml
if [ "${ROUTES_FILE:-}" != "" ]; then ROUTES=${ROUTES_FILE}; fi
AGENT=${ROOT}/carla_garage/plant_agent_vec.py
CFG=${ROOT}/carla_garage
CKPT=${OUT}/run.json

echo "=== 运行 $RUN $(date) ==="
echo "ROUTES=$ROUTES  ROUTE_ONLY=${VEC_ROUTE_ONLY:-0}"
cd "$WORK_DIR" || exit 1
VEC_CKPT=/mnt/2T_HDD/plant2_vector/ckpt_conf/best.pt \
  VEC_RECORD=${VEC_RECORD:-0} VEC_ROUTE_ONLY=${VEC_ROUTE_ONLY:-0} \
  CUDA_VISIBLE_DEVICES=0 timeout 1500 "$PY" \
  "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
  --routes="${ROUTES}" --repetitions=1 --track=${CHALLENGE_TRACK_CODENAME} \
  --checkpoint="${CKPT}" --agent="${AGENT}" --agent-config="${CFG}" \
  --debug=0 --record="${RECORD_PATH}" --resume=False --port=2000 \
  --traffic-manager-port=8000 --gpu-rank=0 > "$OUT/run.log" 2>&1
echo "exit=$?"
grep -E 'VecAgent|Traceback|Error' "$OUT/run.log" | tail -4
echo "$OUT"
