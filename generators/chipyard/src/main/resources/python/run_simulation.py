import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from fir_filter import FirFilterGenerator
from pam5_symbol_generator import Pam5SymbolGenerator
from ethernet_channel import GenericEthernetCable, EthernetChannel


np.random.seed(420)


simulation_frequency = 125e6 # Hz
sampling_frequency = 125e6 # Hz

n_symbols = 100
duration = n_symbols * (1 / sampling_frequency)

print(f"Duration: {duration} s")

# === create symbols to send, in range {-2, -1, 0, 1, 2} ===

pam5_symbols = Pam5SymbolGenerator.random(n_symbols)
# pam5_symbols = Pam5SymbolGenerator.alternate_01(n_symbols)
# pam5_symbols = Pam5SymbolGenerator.alternate_02(n_symbols)
# pam5_symbols = Pam5SymbolGenerator.alternate_012(n_symbols)
# pam5_symbols = Pam5SymbolGenerator.sine_wave(n_symbols)
# pam5_symbols = Pam5SymbolGenerator.sawtooth_wave(n_symbols)

print(f"Symbols: {pam5_symbols}")


# convert the symbols to fill real-time clock
pam5_signal = np.repeat(pam5_symbols, simulation_frequency // sampling_frequency) / 2


# Model channel attenuation
cable_length = 50

channel = EthernetChannel(GenericEthernetCable, cable_length)

pam5_signal_noisy = channel.attenuate(pam5_signal, simulation_frequency)


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


plt.xlabel("Time (ns)")
plt.ylabel("Amplitude (LSB)")
plt.title("PAM5 Signal and Attenuated Signal")
plt.legend(["Original", "Attenuated"])

plt.savefig("test_pam5_signal.png")

plt.clf()
plt.close()



gen = FirFilterGenerator()
taps = gen.calculate_taps(channel_profile=GenericEthernetCable, cable_length=cable_length)

result = gen.simulate(pam5_signal_noisy)

# show the attenuation before and after the filter
input_power = np.max(np.abs(pam5_signal_noisy))
output_power = np.max(np.abs(result))

print(f"Input Power: {input_power:.2f}")
print(f"Output Power: {output_power:.2f}")

attenuation = output_power / input_power
attenuation_lsb = attenuation * 0xFF

print(f"Attenuation: {10 * np.log10(attenuation):.2f} dB, {attenuation_lsb:.2f} LSB")

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

