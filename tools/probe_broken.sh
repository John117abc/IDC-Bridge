#!/bin/bash
set -u
docker exec idc_hist bash -lc "sed -n '212,246p' /gpudrive/baselines/IDC-Bridge/src/env/planner_runtime.py"
