import torch
import torch.nn as nn

class LSTMLite(nn.Module):
    def __init__(
        self,
        C: int = 9,
        L: int = 500,
        H: int = 125,
        hidden: int = 64,
        target_mode: str = "uni",          # "uni" | "multi"
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.target_mode = target_mode
        self.H = H
        self.C = C
        self.num_dirs = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=C,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        out_dim = H if target_mode == "uni" else C * H
        self.head = nn.Linear(hidden * self.num_dirs, out_dim)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        x = x.transpose(1, 2).contiguous()       # -> [B, L, C]
        out, _ = self.lstm(x)                    # -> [B, L, hidden * num_dirs]
        h_last = out[:, -1, :]                   # last time step
        y = self.head(h_last)                    # -> [B, out_dim]
        if self.target_mode == "multi":
            y = y.view(y.size(0), self.C, self.H)
        return y
