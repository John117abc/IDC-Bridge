"""探针3：site-packages 里有没有现成的 *_pb2（含 scenario/map）？pip 可用吗？"""
import glob
import subprocess

SP = '/gpudrive/.venv/lib/python3.11/site-packages'
pb2 = glob.glob(SP + '/**/*_pb2.py', recursive=True)
print('pb2 总数:', len(pb2))
hits = [p for p in pb2 if any(k in p.lower() for k in
                              ('scenario', 'waymo', 'map', 'label', 'vector'))]
print('相关 pb2:', hits[:10])

import waymax
print('waymax:', waymax.__file__)
wm_files = [p for p in glob.glob(SP + '/waymax/**/*.py', recursive=True)
            if 'dataset' in p][:8]
print('waymax dataset 文件:', wm_files)

for cmd in (['python', '-m', 'pip', '--version'],
            ['/gpudrive/.venv/bin/python', '-m', 'pip', '--version']):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        print(cmd, '->', (r.stdout or r.stderr or '').strip()[:120])
    except Exception as e:                       # noqa: BLE001
        print(cmd, '失败:', type(e).__name__)
