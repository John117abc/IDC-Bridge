import glob
import json

files = sorted(glob.glob('/data/data_json/testing/tfrecord-*.json'))
print('testing files:', len(files))
for f in files[:6]:
    d = json.load(open(f))
    o = d['objects'][0]
    print(f.split('/')[-1], 'pos_len=', len(o.get('position') or []),
          'n_obj=', len(d.get('objects') or []))
# 找找有没有别的全长度导出
import os
for cand in ('/data/data_json/testing_full', '/data/data_json/validation',
             '/data/data_json/test'):
    print(cand, 'exists' if os.path.isdir(cand) else '-')
