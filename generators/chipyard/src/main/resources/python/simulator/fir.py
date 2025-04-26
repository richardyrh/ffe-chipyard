import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from .config import SimConfig
from .ethernet_channels import EthernetChannelProfile

class Fir:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
    
    @staticmethod
    def calculate_taps(
        channel_profile: EthernetChannelProfile,
        cable_length: float = 100,
        num_taps: int = 8,
        fir_type: int = 4
    ) -> np.ndarray:
        freq_mhz = channel_profile.freq_mhz
        attenuation_db = -channel_profile.attenuation_db

        # attenuation_db /= (100 / cable_length)

        plt.plot(freq_mhz, attenuation_db)
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Attenuation (dB)")
        plt.savefig("channel_profile.png")








    
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
