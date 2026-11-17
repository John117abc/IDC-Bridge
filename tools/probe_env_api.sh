#!/usr/bin/env bash
F=/gpudrive/gpudrive/env/env.py
grep -n "self_observation" "$F" | head -10
echo "--- get methods ---"
grep -n "def get_" "$F" | head -24
