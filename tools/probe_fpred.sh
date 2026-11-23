#!/bin/bash
set -u
docker exec idc_hist bash -lc "grep -n 'def f_pred_batch' -A 60 /gpudrive/baselines/IDC-Bridge/src/agents/idc_agent.py | head -70"
