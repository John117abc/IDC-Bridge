#!/usr/bin/env bash
P=/gpudrive/.venv/bin/python
for v in 2-18-0 2-17-0 2-16-1 2-16-0 2-15-0; do
    echo "=== try waymo-open-dataset-tf-$v ==="
    $P -m pip install --no-deps --quiet "waymo-open-dataset-tf-$v" 2>&1 | tail -2
    if $P -c "import waymo_open_dataset" 2>/dev/null; then
        echo "INSTALLED tf-$v"
        break
    fi
done
echo "--- 验证 proto ---"
$P -c "from waymo_open_dataset.protos import scenario_pb2; print('scenario_pb2 ok')" 2>&1 | tail -2
