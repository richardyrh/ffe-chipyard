import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from fir_filter import FirFilterGenerator


np.random.seed(420)


simulation_frequency = 125e6 # Hz
sampling_frequency = 125e6 # Hz

# Define target SNR in dB
# target_snr_db = 100
# lowpass_cutoff = 0.

target_snr_db = 20
lowpass_cutoff = 125e6 / 4  # Hz


n_symbols = 50
duration = n_symbols * (1 / sampling_frequency)

print(f"Duration: {duration} s")

# create symbols to send, in range {-2, -1, 0, 1, 2}
pam5_symbols = np.random.randint(-2, 3, n_symbols, dtype=np.int8).astype(np.float32) / 2
print(f"Symbols: {pam5_symbols}")

# convert the symbols to fill real-time clock
pam5_signal_base = np.repeat(pam5_symbols, simulation_frequency // sampling_frequency).astype(np.float32)
# pass through DAC (i.e. scale by 0x3F and convert to single ended)
# pam5_signal = (pam5_signal_base + 2) * 0x3F
pam5_signal = pam5_signal_base

# low pass filter the signal
# pam5_signal = signal.lfilter([1], [1, -lowpass_cutoff], pam5_signal)

# apply convolution smoothing
window_size = int(sampling_frequency // lowpass_cutoff)
pam5_signal = pam5_signal # + np.convolve(pam5_signal, np.ones(window_size) / window_size, mode="valid")

# # Calculate signal power
# signal_power = np.var(pam5_signal)
# # Calculate required noise power based on target SNR
# noise_power = signal_power / (10 ** (target_snr_db / 10))

# # Add noise to the signal
# noise = np.random.randn(len(pam5_signal)) * np.sqrt(noise_power)
# pam5_signal_noisy = pam5_signal + noise

# # Verify actual SNR
# actual_signal_power = np.var(pam5_signal)
# actual_noise_power = np.var(noise)
# actual_snr_db = 10 * np.log10(actual_signal_power / actual_noise_power)
# print(f"Target SNR: {target_snr_db:.2f} dB")
# print(f"Actual SNR: {actual_snr_db:.2f} dB")
cable_length = 50

freq_MHz = np.array([0, 1, 4, 8, 10, 16, 20, 25, 31.25, 62.5, 100])
attenuation_dB = np.array([0, 2.0, 3.8, 5.3, 6.0, 7.6, 8.5, 9.5, 10.7, 15.4, 19.8])
attenuation_dB *= (cable_length / 100)

# attenuate the signal at each frequency
# Perform FFT to transform the signal to the frequency domain
pam5_signal_freq = np.fft.fft(pam5_signal)

# Create a frequency axis
freq_axis = np.fft.fftfreq(len(pam5_signal), d=1/simulation_frequency)

# Interpolate the attenuation values to match the frequency axis
attenuation_interp = np.interp(np.abs(freq_axis), freq_MHz * 1e6, attenuation_dB)

# Convert attenuation from dB to linear scale
attenuation_linear = 10 ** (-attenuation_interp / 20)

# Apply the attenuation to the frequency domain signal
pam5_signal_freq_attenuated = pam5_signal_freq * attenuation_linear

# Transform the attenuated signal back to the time domain
pam5_signal_noisy = np.fft.ifft(pam5_signal_freq_attenuated).real


# create time axis
time_axis = np.arange(len(pam5_signal)) / simulation_frequency
# convert to ns
time_axis = time_axis / 8 * 1e9

# plot the signal
plt.plot(time_axis, pam5_signal)
plt.plot(time_axis, pam5_signal_noisy)

# create symbol boundaries
symbol_boundaries = np.arange(0, len(pam5_signal), simulation_frequency // sampling_frequency)

# draw vertical lines at the symbol boundaries
for i in range(n_symbols):
    plt.axvline(x=symbol_boundaries[i], color="k", linestyle="--", alpha=0.5)

# draw triangle to represent sampling point
# samping_offset = 0.25 * (1 / sampling_frequency) * 1e9
# samping_offset_s = e-9

# sampled_points = []

# print(f"Samping offset: {samping_offset_s * 1e9} ns")
# for i in range(len(symbol_boundaries)):
#     index = int(symbol_boundaries[i] + samping_offset_s * 1e9)
#     if index < len(pam5_signal):
#         plt.plot([symbol_boundaries[i] + samping_offset_s * 1e9], [pam5_signal_noisy[index]], 'ro', alpha=0.5)
#         sampled_points.append(pam5_signal_noisy[index])

plt.xlabel("Time (ns)")
plt.ylabel("Amplitude (LSB)")
plt.title("PAM5 Signal and Noisy Signal")
plt.legend(["Original", "Noisy"])

plt.savefig("test_pam5_signal.png")

plt.clf()
plt.close()



gen = FirFilterGenerator()

taps = gen.calculate_taps(cable_length=cable_length)


result = gen.simulate(pam5_signal_noisy)

plt.clf()
plt.close()

# upsample to 1ghz then plot
upsampled_points = np.repeat(pam5_signal, 8)
upsampled_noisy_points = np.repeat(pam5_signal_noisy, 8)
# upsample results with smoothing
result_upsampled = np.repeat(result, 8)
upsampled_result = np.convolve(result_upsampled, np.ones(8) / 8, mode="valid")

plt.plot(np.hstack([np.zeros(20), upsampled_points]))
plt.plot(np.hstack([np.zeros(20), upsampled_noisy_points]))
plt.plot(upsampled_result)
plt.legend(["Sampled", "Sampled Noisy", "Filtered"])
plt.savefig("test_pam5_signal_filtered.png")

