#!/bin/bash
D=/mnt/2T_HDD/plant2_vector/closedloop
echo "=== checkpoint 摘要 ==="
/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python - <<'EOF'
import json
d = json.load(open('/mnt/2T_HDD/plant2_vector/closedloop/smoke.json'))
c = d.get('_checkpoint', {})
recs = c.get('records', [])
print('records:', len(recs))
for r in recs[:2]:
    print({k: r.get(k) for k in ('route_id', 'status', 'scores', 'meta') if k in r})
g = c.get('global_record', {})
print('global:', {k: g.get(k) for k in ('route_id', 'status', 'infractions')})
print('scores global:', g.get('scores'))
EOF
echo "=== agent 日志（最后 10 条）==="
grep -E '\[VecAgent\]|Traceback|Error|exception' "$D/smoke.log" | tail -10
echo "=== 结果目录 ==="
ls "$D/save" 2>/dev/null | head -3
