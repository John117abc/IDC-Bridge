"""探针4：容器能不能自举 pip / 能不能联网（决定历史帧怎么补）"""
import glob
import subprocess

PY = '/gpudrive/.venv/bin/python'

# 1) ensurepip 能否自举
try:
    r = subprocess.run([PY, '-m', 'ensurepip', '--version'],
                       capture_output=True, text=True, timeout=30)
    print('ensurepip:', r.returncode, (r.stdout or r.stderr or '').strip()[:120])
except Exception as e:                           # noqa: BLE001
    print('ensurepip 失败:', type(e).__name__, e)

# 2) 网络
try:
    import urllib.request
    with urllib.request.urlopen('https://pypi.org/simple/', timeout=8) as f:
        print('网络: ok, status', f.status)
except Exception as e:                           # noqa: BLE001
    print('网络: 不可用 ->', type(e).__name__, str(e)[:100])

# 3) GPUDrive 仓库里有没有 json 导出脚本（可能带 history 参数）
hits = []
for pat in ('/gpudrive/scripts/**/*.py', '/gpudrive/python/**/*.py',
            '/gpudrive/baselines/**/export*.py'):
    hits += glob.glob(pat, recursive=True)
print('可能的导出脚本:', [h for h in hits if 'json' in h.lower() or 'export' in h.lower()][:8])
