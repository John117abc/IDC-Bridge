"""用 Python 启动夜间流水线（避开 shell 引号/CRLF 问题）。"""
import os
import subprocess

SRC = '/gpudrive/baselines/IDC-Bridge/tools/run_night.sh'
DST = '/tmp/night.sh'
LOG = '/workspace/data/logs/night_outer.log'

data = open(SRC, 'rb').read().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
open(DST, 'wb').write(data)
os.chmod(DST, 0o755)
os.makedirs(os.path.dirname(LOG), exist_ok=True)
with open(LOG, 'ab') as f:
    p = subprocess.Popen(['bash', DST], stdout=f, stderr=f,
                         start_new_session=True)
print('launched pid', p.pid)
