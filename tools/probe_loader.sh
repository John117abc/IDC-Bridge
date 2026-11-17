#!/usr/bin/env bash
F=/gpudrive/gpudrive/env/dataset.py
grep -n "class SceneDataLoader" -A10 "$F"
echo "--- attrs ---"
grep -n "self\." "$F" | grep -Ei "scen|file|len|ids" | head -14
