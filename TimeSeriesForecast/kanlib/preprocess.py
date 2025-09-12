# code/preprocess.py
import mne
import numpy as np

def bandpass_notch(arr: np.ndarray, fs: float, band=(1.0, 45.0), notch=(50.0, 100.0)):
    # MNE предпочитает float64 на этапе фильтрации
    x = np.ascontiguousarray(arr, dtype=np.float64)

    # на всякий случай вычистим NaN/Inf
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, copy=False)

    # полосовой
    x = mne.filter.filter_data(x, fs, band[0], band[1], verbose=False)

    # notch(и)
    if notch:
        freqs = list(notch) if isinstance(notch, (list, tuple)) else [notch]
        x = mne.filter.notch_filter(x, fs, freqs=freqs, verbose=False)

    # возвращаем в float32 для экономии памяти
    return x.astype(np.float32, copy=False)
