"""探针2：能不能自己生成 scenario_pb2？（repo 里有没有 proto / protoc / 网络）"""
import glob
import os
import subprocess

print('--- 仓库里的 proto ---')
protos = []
for pat in ('/gpudrive/**/scenario.proto', '/gpudrive/**/*.proto'):
    protos += glob.glob(pat, recursive=True)
print('found:', protos[:8], 'total', len(protos))

print('--- 已安装的 protobuf 工具 ---')
for mod in ('grpc_tools', 'google.protobuf'):
    try:
        __import__(mod)
        print(f'  {mod}: ok')
    except Exception as e:                       # noqa: BLE001
        print(f'  {mod}: {type(e).__name__}')

print('--- 是否已缓存 waymo 包 ---')
for root in ('/root/.cache/pip', '/gpudrive/.venv/lib'):
    hits = glob.glob(root + '/**/*waymo*', recursive=True)[:5]
    if hits:
        print(' ', root, hits)

print('--- pip 能否联网（5 秒超时）---')
try:
    r = subprocess.run(['/gpudrive/.venv/bin/pip', 'download', '--no-deps',
                        '-d', '/tmp/dl', 'waymo-open-dataset-tf-2-11-0'],
                       capture_output=True, text=True, timeout=20)
    print('returncode', r.returncode)
    print((r.stdout or '')[-400:])
    print((r.stderr or '')[-400:])
except Exception as e:                           # noqa: BLE001
    print('pip 失败:', type(e).__name__, e)
