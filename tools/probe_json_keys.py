"""检查场景 JSON 里到底有哪些字段（特别是有没有历史帧）。"""
import glob
import json

files = sorted(glob.glob('/data/data_json/training/tfrecord-*.json'))
print('files:', len(files))
d = json.load(open(files[0]))
print('top keys:', sorted(d.keys()))
print('metadata:', d.get('metadata'))
for k, v in d.items():
    if isinstance(v, list):
        print(f'  {k}: list len={len(v)}')
o = d['objects'][0]
print('object keys:', sorted(o.keys()))
for k, v in o.items():
    if isinstance(v, list):
        print(f'  objects[0].{k}: len={len(v)}')
    else:
        print(f'  objects[0].{k}: {v}')
# 是否存在任何"历史"字段
hist_keys = [k for k in d.keys() if 'hist' in k.lower() or 'past' in k.lower()]
print('history-like keys:', hist_keys)
