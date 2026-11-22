#!/bin/bash
D=/mnt/2T_HDD/plant2_vector/video
L=/mnt/2T_HDD/plant2_vector
echo "等待录制结束..."
for i in $(seq 1 40); do
  n=$(ls "$D" 2>/dev/null | wc -l)
  if [ "$n" -gt 60 ] && ! pgrep -f leaderboard_evaluator > /dev/null; then break; fi
  sleep 15
done
n=$(ls "$D" 2>/dev/null | wc -l)
echo "帧数: $n"
grep -E 'status|VecAgent' "$L/rec_run.log" 2>/dev/null | tail -3
/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python - <<'EOF'
import glob, os
from PIL import Image
fs = sorted(glob.glob('/mnt/2T_HDD/plant2_vector/video/frame_*.png'))
print('frames', len(fs))
if not fs:
    raise SystemExit
step = max(1, len(fs) // 80)
sel = fs[::step][:80]
imgs = [Image.open(f).convert('RGB').resize((480, 270)) for f in sel]
imgs[0].save('/mnt/2T_HDD/plant2_vector/run.gif', save_all=True,
             append_images=imgs[1:], duration=120, loop=0)
print('gif saved', len(imgs), 'frames')
EOF
ls -la "$L/run.gif" 2>/dev/null
