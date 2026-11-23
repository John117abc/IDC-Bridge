#!/bin/bash
set -u
docker exec idc_hist bash -lc "n=\$(grep -n 'def _encode' /gpudrive/baselines/IDC-Bridge/src/agents/idc_agent.py | cut -d: -f1); sed -n \"\${n},\$((n+70))p\" /gpudrive/baselines/IDC-Bridge/src/agents/idc_agent.py"
