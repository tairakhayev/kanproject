import torch, numpy as np, os, json
from torch.utils.data import Dataset, DataLoader

class ForecastDataset(Dataset):
    def __init__(self, X_list, Y_list, mean, std, target_mode="uni", target_ch_idx=0, norm_y=True):
        X = np.stack(X_list)            # [N, C, L]
        self.mean = mean.astype(np.float32); self.std = (std + 1e-6).astype(np.float32)
        self.X = (X - self.mean[:, None]) / self.std[:, None]
        Y = np.stack(Y_list)            # [N, H] or [N, C, H]
        if norm_y and target_mode == "uni":
            Y = (Y - self.mean[target_ch_idx]) / self.std[target_ch_idx]
        elif norm_y and target_mode == "multi":
            Y = (Y - self.mean[:, None]) / self.std[:, None]
        self.Y = Y.astype(np.float32)
        self.target_mode = target_mode

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

def compute_channel_stats(X_list):
    X = np.stack(X_list)  # [N, C, L]
    C = X.shape[1]
    mean = X.transpose(0,2,1).reshape(-1, C).mean(axis=0)
    std  = X.transpose(0,2,1).reshape(-1, C).std(axis=0)
    return mean.astype(np.float32), std.astype(np.float32)

def train_loop(model, loaders, loss_fn, optimizer, device, epochs=100, patience=15, save_path=None, metric_fn=None):
    best = {"val": np.inf, "path": None}
    wait = 0
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
        # val
        model.eval()
        with torch.no_grad():
            vals = []
            for xb, yb in loaders["val"]:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vals.append(loss_fn(pred, yb).item())
            val_loss = float(np.mean(vals))
        if val_loss < best["val"]:
            best["val"] = val_loss; wait = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                best["path"] = save_path
        else:
            wait += 1
            if wait >= patience: break
    return best
