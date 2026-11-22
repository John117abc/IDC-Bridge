#!/bin/bash
# 用法：bash show_results.sh <run_dir>
D=${1:-}
[ -z "$D" ] && { echo "need run dir"; exit 1; }
echo "=== 目录 $D ==="
/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python - "$D" <<'EOF'
import json, sys, os
d = sys.argv[1]
fp = os.path.join(d, 'run.json')
if not os.path.exists(fp):
    print('no run.json'); raise SystemExit
z = json.load(open(fp))
c = z.get('_checkpoint', {})
for r in c.get('records', []):
    print('route', r.get('route_id'), '|', r.get('status'),
          '| score', (r.get('scores') or {}).get('score_composed'),
          '| meta', {k: round(v, 1) for k, v in (r.get('meta') or {}).items()
                     if isinstance(v, (int, float))})
g = c.get('global_record', {})
print('GLOBAL status:', g.get('status'))
inf = g.get('infractions') or {}
print('infractions:', {k: round(v, 2) for k, v in inf.items() if v})
EOF
echo "=== agent 关键行 ==="
grep -E '\[VecAgent\] (route|稠密化|dbg|ckpt)' "$D/run.log" 2>/dev/null | head -5
