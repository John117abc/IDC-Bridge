#!/bin/bash
set -u
docker exec idc_hist bash -lc "grep -n 'optimizer' /gpudrive/baselines/IDC-Bridge/src/utils/checkpoint.py | head -14"
