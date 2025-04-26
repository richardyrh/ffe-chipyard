from dataclasses import MISSING

import numpy as np

try:
    from .config import SimConfig
except:
    from config import SimConfig


class EthernetChannelProfile:
    freq_mhz: np.ndarray = MISSING
    attenuation_db: np.ndarray = MISSING  # dB / 100 m


# https://www.molex.com/pdm_docs/sd/796081000_sd.pdf
class MolexCable(EthernetChannelProfile):
    freq_mhz        = np.array([  0,     1,     4,     8,    10,    16,    20,    25, 31.25,  62.5,   100,   125,   200,   250, ])
    attenuation_db  = np.array([1.0,   2.0,   3.8,   5.3,   6.0,   7.6,   8.5,   9.5,  10.7,  15.4,  19.8,  22.4,  29.0,  32.8, ])


# https://www.mouser.com/pdfdocs/AmphenolPCDCat6Cable.PDF?srsltid=AfmBOopG1adWqhWubfl4_4myw3_lfCQffIrgziSVXLf1fljtON-CFfcO
class AmphenolCable(EthernetChannelProfile):
    freq_mhz        = np.array([     1,     4,     8,    10,    16,    20,    25, 31.25,  62.5,   100,   200,   250, ])
    attenuation_db  = np.array([   3.1,   5.8,   8.0,   9.0,  11.4,  12.8,  14.1,  16.1,  23.2,  29.9,  43.7,  49.7, ])


# https://its.dlink.co.in/assets/patt_1529307287.pdf
class DLinkCable(EthernetChannelProfile):
    freq_mhz        = np.array([     1,     4,     8,    10,    16,    20,    25, 31.25,  62.5,   100,   200,   250,   500,   600, ])
    attenuation_db  = np.array([   2.0,   3.8,   5.3,   6.0,   7.6,   8.5,   9.5,  10.7,  15.4,  19.8,  29.0,  32.8,  45.3,    51, ])


# https://www.farnell.com/datasheets/35178.pdf
class AlcatelCable(EthernetChannelProfile):
    freq_mhz        = np.array([     1,     4,    10,    16,    20,    25, 31.25,  62.5,   100,   155,   200,   250,   300,   350, ])
    attenuation_db  = np.array([   1.9,   3.8,   5.9,   7.5,   8.4,  10.6,  15.1,  19.4,  24.5,  28.0,  31.7,  35.0,  38.1,  41.0, ])


# https://www.com-cables.com/media/catalog/product/cache/1/image/9df78eab33525d08d6e5fb8d27136e95/c/a/cat6-uutp-24-awg-cca-105076.pdf
class ComCables(EthernetChannelProfile):
    freq_mhz        = np.array([     1,     4,     8,    10,    16,    20,    25, 31.25,  62.5,   100,   155,   200,   250,   300,   350,   400,   450,   500,   550, ])
    attenuation_db  = np.array([   2.0,   3.8,   5.3,   5.9,   7.4,   8.3,   9.3,  10.4,  14.9,  19.0,  23.9,  27.4,  30.8,  34.0,  37.0,  39.7,  42.1,  44.9,  47.3, ])


# https://leviton.com/content/dam/leviton/network-solutions/product_documents/product_specification/cat6-uutp-24-awg-cca-105076.pdf
class LevitonCable(EthernetChannelProfile):
    freq_mhz        = np.array([     1,     4,    10,    20,   100,   200,   250, ])
    attenuation_db  = np.array([   2.0,   3.8,   6.0,   8.5,  19.9,  29.0,  32.8, ])


class EthernetChannel:
    def __init__(self, cfg: SimConfig, characteristic: EthernetChannelProfile, cable_length: float = 100):
        self.cfg = cfg
        self.freq_mhz = characteristic.freq_mhz
        self.attenuation_db = characteristic.attenuation_db * (cable_length / 100)
        self.cable_length = cable_length

    def attenuate(self, signal: np.ndarray) -> np.ndarray:
        # Convert frequency response to time-domain FIR filter
        # Create frequency response with proper phase
        freq_axis = np.linspace(0, self.cfg.sim_frequency_hz/2, len(signal)//2 + 1)
        attenuation_interp = np.interp(freq_axis, self.freq_mhz * 1e6, self.attenuation_db)
        
        # Convert to linear scale and add phase
        attenuation_linear = 10 ** (-attenuation_interp / 20)
        freq_response = attenuation_linear * np.exp(-1j * np.pi * freq_axis / (self.cfg.sim_frequency_hz/2))
        
        # Create full frequency response (negative frequencies)
        full_freq_response = np.concatenate([freq_response, np.conj(freq_response[-2:0:-1])])
        
        # Convert to time domain to get FIR filter coefficients
        fir_coeffs = np.fft.ifft(full_freq_response).real
        
        # Apply causal FIR filter
        # Pad signal with zeros to handle edge effects
        padded_signal = np.pad(signal, (len(fir_coeffs)-1, 0), mode='constant')
        signal_attenuated = np.convolve(padded_signal, fir_coeffs, mode='valid')
        
        return signal_attenuated


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))  # Create a larger figure
    plt.plot(AmphenolCable.freq_mhz, -AmphenolCable.attenuation_db, '-o', label="Amphenol")
    plt.plot(DLinkCable.freq_mhz, -DLinkCable.attenuation_db, '-o', label="D-Link")
    plt.plot(AlcatelCable.freq_mhz, -AlcatelCable.attenuation_db, '-o', label="Alcatel")
    plt.plot(ComCables.freq_mhz, -ComCables.attenuation_db, '-o', label="Com Cables")
    plt.plot(LevitonCable.freq_mhz, -LevitonCable.attenuation_db, '-o', label="Leviton")
    plt.plot(MolexCable.freq_mhz, -MolexCable.attenuation_db, '-o', label="Molex")

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Frequency Response (dB / 100 m)")

    plt.legend()
    plt.tight_layout()  # Adjust spacing
    plt.savefig("ethernet_cable_channel_characteristics.png", dpi=300)
    plt.show()

