import yaml, json, os, numpy as np, torch
from train import ForecastDataset, compute_channel_stats, train_loop
from models.lstm import LSTMLite
from models.kan import KANRegressor
from metrics import mae, rmse, smape

def load_config(p): 
    with open(p) as f: return yaml.safe_load(f)

def select_model(cfg, C, L, H):
    name = cfg["model"]["name"]
    if name == "lstm":
        return LSTMLite(C=C, L=L, H=H, **cfg["model"].get("params", {}))
    if name == "kan":
        return KANRegressor(C=C, L=L, H=H, **cfg["model"].get("params", {}))
    raise ValueError(name)

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    np.random.seed(cfg["random_seed"]); torch.manual_seed(cfg["random_seed"])
    # Здесь ты подключишь свой загрузчик файлов + препроцесс + windowing + split
    # Ниже — только показ, как считать статы и обучить
    # X_train, Y_train, X_val, Y_val, X_test, Y_test, C, L, H должны быть подготовлены заранее
    ...
