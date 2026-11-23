#!/bin/bash
set -u
docker exec idc_hist bash -lc "sed -n '28,62p' /gpudrive/baselines/IDC-Bridge/src/models/continuous_actor_critic.py"
