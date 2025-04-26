import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from .config import SimConfig


class SimTransmitter:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
    
    def generate_signal(
        self,
        symbols: np.ndarray,
        ignore_delay: bool = True
    ):
        n_symbols = symbols.shape[0]

        print(f"signal duration: {(n_symbols / self.cfg.symbol_frequency_hz) * 1e9:.2f} ns")

        sim_time_s = np.arange(0, n_symbols / self.cfg.symbol_frequency_hz, 1 / self.cfg.sim_frequency_hz)
        
        # convert the symbols to fill real-time clock
        pam5_signal = np.repeat(symbols, self.cfg.sim_frequency_hz // self.cfg.symbol_frequency_hz) / 2

        # low pass filter
        window_size = 11
        cutoff_frequency = 10e6
        low_pass_filter = signal.firwin(window_size, cutoff_frequency, window="hamming", fs=self.cfg.sim_frequency_hz)
        filtered_signal = signal.convolve(pam5_signal, low_pass_filter)
        # remove delay
        if ignore_delay:
            filtered_signal = filtered_signal[window_size//2:-window_size//2+1]
        else:
            filtered_signal = filtered_signal[:-window_size+1]

        return {
            "time_s": sim_time_s,
            "raw_signal": pam5_signal,
            "smooth_signal": filtered_signal,
        }
