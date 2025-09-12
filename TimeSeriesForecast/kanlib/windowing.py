import numpy as np

def make_forecast_pairs(signal: np.ndarray, L: int, H: int, stride: int,
                        target_mode: str = "uni", target_ch: int = 0):
    pairs = []
    t = 0
    T = signal.shape[1]
    while t + L + H <= T:
        X = signal[:, t : t + L]
        if target_mode == "uni":
            Y = signal[target_ch, t + L : t + L + H]
        else:
            Y = signal[:, t + L : t + L + H]
        pairs.append((X.astype(np.float32), Y.astype(np.float32)))
        t += stride
    return pairs
