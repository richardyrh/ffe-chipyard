from dataclasses import MISSING

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from .config import SimConfig
from .ethernet_channels import EthernetChannel

class Fir:
    def __init__(self,
        cfg: SimConfig,
        num_taps: int | None = None,
        taps: np.ndarray | None = None,
        fir_type: int = 4
    ):
        self.cfg = cfg
        self.fir_type = fir_type
        self.taps: np.ndarray | None = taps
        if self.taps is not None:
            self.num_taps = self.taps.shape[0]
        else:
            self.num_taps = num_taps

    def calculate_taps(self, channel: EthernetChannel) -> np.ndarray:
        # nyquist frequency is twice the sampling frequency
        nyquist_freq_hz = self.cfg.symbol_frequency_hz

        # only consider frequencies up to nyquist frequency
        valid_indices = channel.freq_mhz <= nyquist_freq_hz / 1e6
        freq_mhz = channel.freq_mhz[valid_indices]
        attenuation_db = -channel.attenuation_db[valid_indices]
        channel_gain = 10 ** (attenuation_db / 20)
        
        # convert MHz to Hz
        freq_hz = freq_mhz * 1e6
        

        desired_filter_attenuation_db = -attenuation_db
        desired_filter_attenuation_db -= np.max(desired_filter_attenuation_db)

        print("desired_filter_attenuation_db:", desired_filter_attenuation_db)


        # calculate gain from attenuation
        desired_filter_gain = 10 ** (desired_filter_attenuation_db / 20)

        # normalize gain to 0dB at 0Hz
        desired_filter_gain -= desired_filter_gain[0]

        print("desired_filter_gain:", desired_filter_gain)


        # Ensure frequencies are properly normalized (0 to 1)
        normalized_freqs = freq_hz / nyquist_freq_hz

        print("normalized_freqs:", normalized_freqs)
        
        # Design the FIR filter
        fir_taps = signal.firwin2(
            self.num_taps,
            normalized_freqs,
            desired_filter_gain,
            fs=2,
            antisymmetric=(self.fir_type % 2 == 0)
        )

        self.taps = np.array(fir_taps)

        print("fir_taps:", fir_taps)

        # visualize frequency response
        fir_response = signal.freqz(fir_taps, worN=freq_mhz.shape[0], whole=True, fs=nyquist_freq_hz)

        response_freq_hz = fir_response[0]
        response_gain = np.abs(fir_response[1])

        response_gain_db = 20 * np.log10(response_gain)

        print("response_freq_hz:", response_freq_hz)
        print("response_gain:", response_gain)

        compensated_channel_db = attenuation_db + response_gain_db
        compensated_channel_gain = channel_gain * response_gain

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        ax1.plot(freq_mhz, attenuation_db, label="channel")
        ax1.plot(freq_mhz, desired_filter_attenuation_db, label="desired")
        ax1.plot(response_freq_hz / 1e6, response_gain_db, label="fir")
        ax1.plot(response_freq_hz / 1e6, compensated_channel_db, label="compensated")

        ax2.plot(freq_mhz, channel_gain, label="channel")
        ax2.plot(freq_mhz, desired_filter_gain, label="desired")
        ax2.plot(response_freq_hz / 1e6, response_gain, label="fir")
        ax2.plot(response_freq_hz / 1e6, compensated_channel_gain, label="compensated")


        ax1.set_ylabel("Attenuation (dB)")
        ax2.set_ylabel("Gain")
        
        ax1.set_ylim(-40, 0.1)
        ax2.set_ylim(0, 1.1)
        
        ax1.grid(True)
        ax2.grid(True)
        
        ax1.legend()
        ax2.legend()
        
        plt.tight_layout()
        fig.savefig("characteristics.png")

        plt.close(fig)


    def bode_plot(self, channel: EthernetChannel):
        nyquist_freq_hz = self.cfg.symbol_frequency_hz
        
        freq_mhz = channel.freq_mhz
        attenuation_db = -channel.attenuation_db
        
        # Calculate frequency response
        w, h = signal.freqz(self.taps, worN=8000, whole=True, fs=nyquist_freq_hz)
        magnitude = 20 * np.log10(np.abs(h))
        phase = np.unwrap(np.angle(h)) * 180 / np.pi

        # Create Bode plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Magnitude plot
        ax1.semilogx(w/1e6, magnitude, "b-", label="Filter Response")
        ax1.semilogx(freq_mhz, attenuation_db, "r--", label="Channel Response")
        ax1.set_title("Bode Plot")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True, which="both", linestyle="--")
        ax1.legend()

        # Phase plot
        ax2.semilogx(w/1e6, phase, "b-", label="Filter Phase")
        ax2.set_xlabel("Frequency (MHz)")
        ax2.set_ylabel("Phase (degrees)")
        ax2.grid(True, which="both", linestyle="--")
        ax2.legend()

        ax1.axvline(x=self.cfg.symbol_frequency_hz/1e6, color="k", linestyle="--")
        ax2.axvline(x=self.cfg.symbol_frequency_hz/1e6, color="k", linestyle="--")
        
        plt.tight_layout()
        plt.savefig("bode_plot.png")
        plt.close(fig)

    
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

    def filter(self, signal: np.ndarray):
        return np.convolve(signal, self.taps, mode="full")