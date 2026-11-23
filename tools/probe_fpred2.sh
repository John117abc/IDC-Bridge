#!/bin/bash
set -u
docker exec idc_hist bash -lc "sed -n '499,545p' /gpudrive/baselines/IDC-Bridge/src/agents/idc_agent.py"
