# Final correct FIR filter implementation
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

cable_length = 10
num_taps = 8
fir_type = 4

# Original frequency and attenuation data
# freq_MHz = np.array([1, 4, 8, 10, 16, 20, 25, 31.25, 62.5, 100, 200, 250, 500])
# attenuation_dB = 49.7 - np.array([3.1, 5.8, 8.0, 9.0, 11.4, 12.8, 14.1, 16.1, 23.2, 29.9, 43.7, 49.7, 49.7])
# freq_MHz = np.array([0, 1, 4, 8, 10, 16, 20, 25, 31.25, 62.5, 100, 200, 250])
# attenuation_dB = 49.7 - np.array([0, 3.1, 5.8, 8.0, 9.0, 11.4, 12.8, 14.1, 16.1, 23.2, 29.9, 43.7, 49.7])
# freq_MHz = np.array([0, 1, 4, 8, 10, 16, 20, 25, 31.25, 62.5, 100, 125])
# attenuation_dB = np.array([0, 3.1, 5.8, 8.0, 9.0, 11.4, 12.8, 14.1, 16.1, 23.2, 29.9, 36.8])
# freq_MHz = np.array([0, 1, 4, 8, 10, 16, 20, 25, 31.25, 62.5])
# attenuation_dB = np.array([0, 3.1, 5.8, 8.0, 9.0, 11.4, 12.8, 14.1, 16.1, 23.2])
freq_MHz = np.array([0, 1, 4, 8, 10, 16, 20, 25, 31.25, 62.5])
attenuation_dB = np.array([0, 2.0, 3.8, 5.3, 6.0, 7.6, 8.5, 9.5, 10.7, 15.4])
# attenuation_dB[-1] += 10
attenuation_dB = np.max(attenuation_dB) - attenuation_dB + 0
attenuation_dB /= (100 / cable_length)

# Convert MHz to Hz
freq_Hz = freq_MHz * 1e6

# Nyquist frequency (fs/2 = highest freq)
nyquist_freq = freq_Hz[-1]

gain = 10 ** (-attenuation_dB / 20)

if fir_type == 2:
  gain[-1] = 0
elif fir_type == 3:
  gain[-1] = 0
  gain[0] = 0
elif fir_type == 4:
  gain[0] = 0


print("gain", gain)

# fir_taps = signal.firwin2(num_taps, freq_Hz / np.max(freq_Hz), gain, fs=2.0)
fir_taps = signal.firwin2(num_taps, freq_Hz / np.max(freq_Hz), gain, fs=2.0, antisymmetric=(fir_type % 2 == 0))

# Frequency response visualization
freq_response, response = signal.freqz(fir_taps, whole=True) # 8000)
# freq_response, response = signal.freqz(fir_taps, worN=8000)
plt.figure(figsize=(12, 5))
ax = plt.subplot(121)
ax.plot(((freq_response / np.pi) * nyquist_freq / 1e6)[:8000], (20 * np.log10(np.abs(response)))[:8000])
plt.title('FIR Filter Frequency Response')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
# plt.savefig("filter.png")
# plt.show()

print(fir_taps)

taps_quantized = np.clip(np.round(fir_taps * 128), -128, 127).astype(np.int8)
print("taps_quantized\n", taps_quantized)


freq_MHz = np.array([1, 4, 8, 10, 16, 20, 25, 31.25, 62.5, 100, 200, 250])
attenuation_dB = np.array([2.0, 3.8, 5.3, 6.0, 7.6, 8.5, 9.5, 10.7, 15.4, 19.8, 29.0, 32.8])
attenuation_dB /= (100 / cable_length)

# Sampling parameters
# fs = 1e10  # 1 GHz sampling frequency
fs = nyquist_freq * 2
t = np.arange(0, 2e-4, 1/fs)  # simulate for 1 µs

sim_freqs = np.linspace(1, 127, 101)
sim_atten = np.interp(x = sim_freqs, xp = freq_MHz, fp = attenuation_dB)

# Generate original signals with respective attenuations
signals = np.array([np.sin(2 * np.pi * f * 1e6 * t) * 10 ** (-att/20)
                    for f, att in zip(sim_freqs, sim_atten)])

# Apply FIR filter
filtered_signals = np.array([signal.lfilter(fir_taps, 1.0, sig) for sig in signals])

# plt.figure(figsize=(10, 6))
# ax.plot(t, signals[-2], label='Original signal')
# ax.plot(t, signals[-1], label='Original -1 signal')
# plt.plot(t, filtered_signals[-2], label='Filtered signal')
# plt.grid(True)
# plt.legend()
# plt.show()

# Measure amplitudes after filtering (steady-state)
orig_amplitudes = np.array([np.max(np.abs(sig)) for sig in signals])
filtered_amplitudes = np.array([np.max(np.abs(sig[300:])) for sig in filtered_signals])
# for i in range(len(filtered_signals)):
#     sig = filtered_signals[i]
#     filtered_amplitudes[i] = np.max(np.abs(sig[100:]))
# Convert amplitudes to dB
orig_amplitudes_dB = 20 * np.log10(orig_amplitudes)
filtered_amplitudes_dB = 20 * np.log10(filtered_amplitudes)

# Plot original vs. filtered attenuations
# plt.figure(figsize=(10, 6))
ax = plt.subplot(122)
ax.plot(sim_freqs, orig_amplitudes_dB,  label='Original Attenuation (dB)')
ax.plot(sim_freqs, filtered_amplitudes_dB,  label='Filtered Attenuation (dB)')
plt.title('Original vs. Filtered Signal Attenuation')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.legend()
plt.savefig("fir.png")
# plt.show()

