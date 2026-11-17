"""通用启动器：python tools/run_script.py <script.sh> [log]  (避开 shell 引号问题)"""
import os
import subprocess
import sys

if len(sys.argv) < 2:
    raise SystemExit('usage: run_script.py <script.sh> [log]')
src = sys.argv[1]
if not src.startswith('/'):
    src = '/gpudrive/baselines/IDC-Bridge/' + src
log = sys.argv[2] if len(sys.argv) > 2 else '/workspace/data/logs/outer.log'
dst = '/tmp/_run_' + os.path.basename(src)
data = open(src, 'rb').read().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
open(dst, 'wb').write(data)
os.chmod(dst, 0o755)
os.makedirs(os.path.dirname(log), exist_ok=True)
with open(log, 'ab') as f:
    p = subprocess.Popen(['bash', dst], stdout=f, stderr=f,
                         start_new_session=True)
print('launched', dst, 'pid', p.pid, 'log', log)
