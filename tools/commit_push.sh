#!/bin/bash
set -u
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
S=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2

# ---------------- IDC-Bridge ----------------
cd $R || exit 1
FILES="tools/plant_agent_idc.py tools/carla_pid_exec.py tools/idc_state_carla.py \
tools/train_planner.py tools/train_plant2_occ.py \
tools/eval_ckpt.sh tools/eval_panel_plan2.sh tools/p3_auto_pipeline.sh tools/p3_train.sh \
tools/save_model_and_check_video.sh \
tools/patch_occ_aux.py tools/patch_occ_perpatch.py tools/patch_aux_posweight.py \
tools/patch_long_ref.py tools/patch_long_smooth.py tools/patch_long_smooth2.py"

for f in $FILES; do
  [ -e "$f" ] && git add "$f" 2>/dev/null
done

git -c user.name="jiangchengxuan" -c user.email="jiangchengxuan@users.noreply.github.com" \
    commit -m "P0: plan2 executable reference + longitudinal fixes; P3: occ-aux supervision

P0 (plan -> execution):
- idc_state_carla.build_path_ext: extrapolate plan to >=28m, PCHIP over arc length,
  analytic heading/curvature, speed = min(plan, curvature) with floor
- plant_agent_idc: VEC_REF_MODE=plan2 branch (default legacy), VEC_IDC_LONG_REF
  (v_tgt = v_ref + 0.15*acc instead of integrating acc from 0), v_tgt low-pass
  + rate limit
- carla_pid_exec: replace hard full-brake (v/v_tgt>1.032 -> brake 1.0) with
  proportional braking; KP/KD/KI/ratio configurable (KD 1.58->0.25)

P3 (occupancy supervision):
- train_planner.Planner(occ_aux): per-token 4x4 patch reconstruction head
- train_plant2_occ: --occ-aux --aux-pos-weight (BCE pos_weight for sparsity),
  --occ-pos support
- p3_auto_pipeline.sh: probe-driven auto continuation with model checkpointing

Results (24759, plan2+LONG_REF): completed 100%, 0 collisions, score 100.0;
full-brake ratio 10.5% -> 0%, mean speed 3.79 -> 5.27 m/s" 2>&1 | tail -3

git push origin "$(git branch --show-current)" 2>&1 | tail -3
echo "--- IDC-Bridge 最近提交 ---"
git log --oneline -2

# ---------------- plant2 ----------------
echo
echo "===== plant2 ====="
cd $S || exit 1
for f in carla_garage/plant_agent_idc.py carla_garage/carla_pid_exec.py \
         carla_garage/idc_state_carla.py carla_garage/plant_agent_vec.py; do
  [ -e "$f" ] && git add "$f"
done
git -c user.name="jiangchengxuan" -c user.email="jiangchengxuan@users.noreply.github.com" \
    commit -m "carla_garage: sync plan2 reference, longitudinal PID smoothing, occ-aux agent hooks" \
    2>&1 | tail -3
git push origin "$(git branch --show-current)" 2>&1 | tail -3
echo "--- plant2 最近提交 ---"
git log --oneline -2
