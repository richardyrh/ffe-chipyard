import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from .config import SimConfig


class Adc:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
    
    def sample(self, signal: np.ndarray, sampling_point: float = 0):
        indices_per_symbol = int(self.cfg.sim_frequency_hz // self.cfg.symbol_frequency_hz)
        sampling_indices = np.arange(0, signal.shape[0], indices_per_symbol, dtype=np.int32)
        sampling_indices[:] += int(sampling_point * indices_per_symbol)
        sampling_indices = sampling_indices[sampling_indices < signal.shape[0]]
        samples = signal[sampling_indices]

        return {
            "indices": sampling_indices,
            "samples": samples,
        }
