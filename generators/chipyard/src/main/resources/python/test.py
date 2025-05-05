import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz

# Parameters
fs = 125e6  # Sampling frequency: 125 MHz
nyquist = fs / 2

# Design a low-pass FIR filter
numtaps = 101  # Number of filter taps
cutoff = 62.5e6  # Cutoff frequency: 62.5 MHz
normalized_cutoff = cutoff / nyquist

# Generate filter coefficients
fir_coefficients = firwin(numtaps, normalized_cutoff, window='hamming')

# Frequency response
w, h = freqz(fir_coefficients, worN=8000)
frequencies = w * nyquist / np.pi

# Plot frequency response
plt.figure(figsize=(10, 6))
plt.plot(frequencies, 20 * np.log10(np.abs(h)))
plt.title('FIR Filter Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.show()

# Simulate a signal: 10 MHz sine wave with noise
t = np.arange(0, 1e-6, 1/fs)  # 1 microsecond of data
signal = np.sin(2 * np.pi * 10e6 * t) + 0.5 * np.random.randn(len(t))

# Apply the FIR filter
filtered_signal = lfilter(fir_coefficients, 1.0, signal)

# Plot original and filtered signals
plt.figure(figsize=(10, 6))
plt.plot(t * 1e6, signal, label='Original Signal')
plt.plot(t * 1e6, filtered_signal, label='Filtered Signal', linewidth=2)
plt.title('Original and Filtered Signals')
plt.xlabel('Time (µs)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
