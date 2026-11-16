#!/usr/bin/env bash
echo "--- 容器内网关 ---"
ip route 2>/dev/null | head -3
echo "--- 试探可用代理 ---"
/gpudrive/.venv/bin/python - <<'PY'
import os, socket, urllib.request
cands = []
for host in ("host.docker.internal", "172.17.0.1", "172.18.0.1"):
    for port in (7897, 7890, 1080, 8080):
        cands.append((host, port))
# 去重、先测端口可达性
ok = []
for h, p in cands:
    try:
        s = socket.create_connection((h, p), timeout=1.0)
        s.close()
        ok.append((h, p))
    except Exception:
        pass
print("端口可达:", ok)
for h, p in ok:
    proxy = f"http://{h}:{p}"
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    try:
        with opener.open("https://pypi.org/simple/", timeout=6) as f:
            print(proxy, "-> OK", f.status)
            break
    except Exception as e:
        print(proxy, "->", type(e).__name__, str(e)[:60])
PY
