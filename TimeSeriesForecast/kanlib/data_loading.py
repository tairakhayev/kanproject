# code/data_loading.py
import mne
import numpy as np

def read_edf(path: str, pick_channels=None, fs_target: float | None = None):
    # читаем с preload=True, чтобы могли безопасно фильтровать/ресэмплировать
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

    # выбираем нужные каналы сразу
    if pick_channels is not None:
        raw.pick(pick_channels, verbose=False)

    # average reference
    raw.set_eeg_reference('average', projection=False, verbose=False)

    # при необходимости — корректный ресэмплинг через Raw.resample
    fs = float(raw.info['sfreq'])
    if fs_target is not None and abs(fs - fs_target) > 1e-6:
        raw.resample(fs_target, npad="auto", verbose=False)
        fs = float(raw.info['sfreq'])

    # возвращаем массив в [C, T] и fs
    data = raw.get_data().astype(np.float32, copy=False)
    return data, fs
