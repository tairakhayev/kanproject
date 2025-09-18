# kanlib/models/kan.py
import torch
import torch.nn as nn
from kan import KAN  # убедись, что pykan импортируется


class KANRegressor(nn.Module):
    """
    KAN для прогноза временных рядов.
    Вход:  x  — [B, C, L] (нормализовано per-channel)
    Выход: y  — [B, H]  (uni)  или [B, C, H] (multi)

    Фишки:
    - Learnable pooling: depthwise Conv1d(kernel=stride=pool) вместо avg-pool.
    - Residual head: KAN(f) + Linear(f) для покрытия линейной части.
    """
    def __init__(
        self,
        C: int = 9,
        L: int = 500,
        H: int = 125,
        pool: int = 5,
        hidden=(64, 32),
        target_mode: str = "uni",   # "uni" | "multi"
        grid: int = 3,
        k: int = 3,
        kan_kwargs: dict | None = None,
    ):
        super().__init__()
        self.C, self.L, self.H = C, L, H
        self.pool = int(pool)
        self.target_mode = target_mode

        # learnable pooling (depthwise) или Identity
        if self.pool > 1:
            self.down = nn.Conv1d(C, C, kernel_size=self.pool, stride=self.pool,
                                  groups=C, bias=False)
            Lp = L // self.pool
        else:
            self.down = nn.Identity()
            Lp = L

        D_in = C * Lp
        out_dim = H if target_mode == "uni" else C * H

        # KAN trunk
        hidden = list(hidden) if isinstance(hidden, (list, tuple)) else [int(hidden)]
        layers = [D_in] + hidden + [out_dim]
        kan_kwargs = {} if kan_kwargs is None else dict(kan_kwargs)
        try:
            self.kan = KAN(layers, grid=grid, k=k, **kan_kwargs)
        except TypeError:
            # совместимость со старыми версиями pykan
            self.kan = KAN(layers)

        # Residual linear head
        self.lin = nn.Linear(D_in, out_dim)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

        self.flatten = nn.Flatten()

    def forward_features(self, x):   # (B,C,L) -> (B,D)
        # верни тензор перед финальной регрессией
        return self.feat(x)          # имя блока подставь по своему коду

    def forward(self, x):
        z = self.forward_features(x)
        return self.head(z)