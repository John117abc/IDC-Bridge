import glob
import json

f = sorted(glob.glob('/data/data_json/testing/tfrecord-*.json'))[0]
d = json.load(open(f))
print('file', f.split('/')[-1])
print('top keys:', list(d.keys())[:12])
print('has metadata:', 'metadata' in d, d.get('metadata'))
objs = d.get('objects') or []
print('n objects', len(objs))
if objs:
    print('obj keys:', list(objs[0].keys()))
    print('pos len', len(objs[0].get('position') or []))
    print('has heading', 'heading' in objs[0])
    print('has valid', 'valid' in objs[0])
print('roads', len(d.get('roads') or []))
