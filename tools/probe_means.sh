#!/bin/bash
set -u
docker exec idc_hist bash -lc "sed -n '180,225p' /gpudrive/gpudrive/env/env_torch.py"
