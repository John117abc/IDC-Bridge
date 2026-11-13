"""在线逐维归一化。

设计要点
--------
1. **先验收缩**：累计统计量里额外计入 ``prior_n`` 个方差为 ``prior_var``、
   均值为 0 的虚拟样本。这样冷启动（n=0）时接近恒等映射，随样本增多平滑过渡，
   避免"前 500 步不归一化、第 501 步突然切换"的跳变。
2. **只记逐维统计**：作用在任何 ``[..., D]`` 张量上，不关心前置维度。
3. **可持久化**：``state_dict`` / ``load_state_dict`` 进出 checkpoint。
4. 输出裁到 ``±clip``，防止个别维度在统计量还没收敛时炸掉。
"""
import torch


class RunningNorm:
    def __init__(self, dim: int, eps: float = 1e-3, clip: float = 10.0,
                 prior_n: float = 200.0, prior_var: float = 1.0):
        self.dim = int(dim)
        self.eps = float(eps)
        self.clip = float(clip)
        self.prior_n = float(prior_n)
        self.prior_var = float(prior_var)
        self._sum = torch.zeros(self.dim, dtype=torch.float64)
        self._sumsq = torch.zeros(self.dim, dtype=torch.float64)
        self._n = 0.0

    @property
    def count(self) -> float:
        return self._n

    @torch.no_grad()
    def update(self, x):
        """x: [N, D] 或 [..., D]，按最后一维统计。"""
        if x is None:
            return
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        flat = x.detach().reshape(-1, self.dim).double().cpu()
        if flat.shape[0] == 0:
            return
        self._sum += flat.sum(dim=0)
        self._sumsq += (flat * flat).sum(dim=0)
        self._n += float(flat.shape[0])

    def stats(self):
        """返回 (mean, std)，形状 [D]，float32。"""
        n = self._n + self.prior_n
        mean = self._sum / n
        # E[x^2] - E[x]^2，其中先验样本的 sum_x=0, sum_x2=prior_n*prior_var
        ex2 = (self._sumsq + self.prior_var * self.prior_n) / n
        var = torch.clamp(ex2 - mean * mean, min=self.eps * self.eps)
        return mean.float(), var.sqrt().float()

    @torch.no_grad()
    def __call__(self, x):
        mean, std = self.stats()
        mean = mean.to(x.device)
        std = std.to(x.device)
        return torch.clamp((x - mean) / std, -self.clip, self.clip)

    # ---------- 持久化 ----------
    def state_dict(self):
        return {'sum': self._sum.clone(), 'sumsq': self._sumsq.clone(),
                'n': self._n, 'dim': self.dim}

    def load_state_dict(self, st):
        if not st:
            return
        if int(st.get('dim', self.dim)) != self.dim:
            return  # 维度不匹配（换了表示层）-> 保持冷启动，别污染
        self._sum = st['sum'].double().cpu().clone()
        self._sumsq = st['sumsq'].double().cpu().clone()
        self._n = float(st['n'])
