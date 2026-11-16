"""检查容器内 torch 是否用 GPU，以及前向速度。"""
import os
import subprocess
import time

print('CUDA_VISIBLE_DEVICES =', os.environ.get('CUDA_VISIBLE_DEVICES'))
try:
    out = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True,
                         timeout=20)
    print('nvidia-smi:', (out.stdout or out.stderr).strip()[:300])
except Exception as e:                           # noqa: BLE001
    print('nvidia-smi 失败:', type(e).__name__)

import torch                                  # noqa: E402

print('torch', torch.__version__, '| cuda avail', torch.cuda.is_available(),
      '| count', torch.cuda.device_count())
if torch.cuda.is_available():
    x = torch.randn(128, 192, device='cuda')
    m = torch.nn.Linear(192, 160).to('cuda')
    for _ in range(5):
        m(x)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(100):
        m(x)
    torch.cuda.synchronize()
    print('GPU 100 次前向:', round(time.time() - t, 4), 's')
else:
    print('!!! torch 看不到 CUDA !!!')
