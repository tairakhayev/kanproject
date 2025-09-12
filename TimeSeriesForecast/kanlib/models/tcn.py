
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=0, dilation=dilation)
    def forward(self, x):                # x: [B,C,L]
        x = nn.functional.pad(x, (self.pad, 0))  # left-pad for causality
        return self.conv(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = (nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())

    def forward(self, x):                # [B,C,L]
        out = self.net(x)
        res = self.downsample(x)
        return out + res

class TCNLite(nn.Module):
    def __init__(self, C=9, L=500, H=125, hidden=64, n_blocks=3,
                 kernel_size=3, dropout=0.1, target_mode="uni"):
        super().__init__()
        self.target_mode = target_mode
        self.C, self.L, self.H = C, L, H

        layers = []
        in_ch = C
        for b in range(n_blocks):
            dil = 2 ** b
            layers.append(TemporalBlock(in_ch, hidden, kernel_size, dil, dropout))
            in_ch = hidden
        self.tcn = nn.Sequential(*layers)

        out_dim = H if target_mode == "uni" else C * H
        self.head = nn.Linear(hidden, out_dim)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):                # x: [B,C,L]
        h = self.tcn(x)                  # [B,hidden,L]
        h_last = h[:, :, -1]             # last time step (causal)
        y = self.head(h_last)            # [B,out_dim]
        if self.target_mode == "multi":
            y = y.view(y.size(0), self.C, self.H)
        return y
