from dataclasses import MISSING

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt



class EthernetChannelProfile:
    freq_mhz: np.ndarray = MISSING
    attenuation_db: np.ndarray = MISSING

class GenericEthernetCable(EthernetChannelProfile):
    freq_mhz = np.array([0, 1, 4, 8, 10, 16, 20, 25, 31.25, 62.5, 100])
    attenuation_db = np.array([0, 2.0, 3.8, 5.3, 6.0, 7.6, 8.5, 9.5, 10.7, 15.4, 19.8])


class EthernetChannel:
    def __init__(self, characteristic: EthernetChannelProfile, cable_length: float = 100):
        self.freq_mhz = characteristic.freq_mhz
        self.attenuation_db = characteristic.attenuation_db * (cable_length / 100)
        self.cable_length = cable_length

    def attenuate(self, signal: np.ndarray, simulation_frequency: float) -> np.ndarray:
        # attenuate the signal at each frequency
        # Perform FFT to transform the signal to the frequency domain
        signal_freq = np.fft.fft(signal)

        # Create a frequency axis
        freq_axis = np.fft.fftfreq(len(signal), d=1/simulation_frequency)

        # Interpolate the attenuation values to match the frequency axis
        attenuation_interp = np.interp(np.abs(freq_axis), self.freq_mhz * 1e6, self.attenuation_db)

        # Convert attenuation from dB to linear scale
        attenuation_linear = 10 ** (-attenuation_interp / 20)

        # Apply the attenuation to the frequency domain signal
        signal_freq_attenuated = signal_freq * attenuation_linear

        # Transform the attenuated signal back to the time domain
        signal_attenuated = np.fft.ifft(signal_freq_attenuated).real

        return signal_attenuated
