#!/bin/bash
echo "=== /data/data_json 下的划分 ==="
ls /data/data_json 2>/dev/null
echo "=== 各划分场景数（tfrecord-*.json）==="
for d in /data/data_json/*/; do
  n=$(find "$d" -maxdepth 1 -name 'tfrecord-*.json' 2>/dev/null | wc -l)
  echo "$d  ->  $n"
done
echo "=== 训练集文件数 + 总量 ==="
find /data/data_json/training -maxdepth 1 -name 'tfrecord-*.json' | wc -l
du -sh /data/data_json/training 2>/dev/null
echo "=== 文件名规律 ==="
find /data/data_json/training -maxdepth 1 -name 'tfrecord-*.json' | head -3
find /data/data_json/training -maxdepth 1 -name 'tfrecord-*.json' | tail -2
