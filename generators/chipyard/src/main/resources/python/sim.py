import numpy as np
import matplotlib.pyplot as plt

from simulator.config import SimConfig
from simulator.adc import Adc
from simulator.ethernet_channels import EthernetChannel, MolexCable
from simulator.pam5_symbol_generator import Pam5SymbolGenerator
from simulator.transmitter import SimTransmitter
from simulator.fir import Fir


np.random.seed(42)


cfg = SimConfig()

n_symbols = 10
ignore_delay = False
sampling_offset = 0.7

thresholds = np.array([-0.4, -0.2, 0.2, 0.4])

# === create data to send, in range {-2, -1, 0, 1, 2} ===

# data = Pam5SymbolGenerator.random(n_symbols)
# data = Pam5SymbolGenerator.dc(n_symbols, 1)
# data = Pam5SymbolGenerator.alternate_01(n_symbols)
# data = Pam5SymbolGenerator.alternate_02(n_symbols)
data = Pam5SymbolGenerator.alternate_012(n_symbols)
# data = Pam5SymbolGenerator.sawtooth_wave(n_symbols)
# data = Pam5SymbolGenerator.sine_wave(n_symbols)


print("data:", data)



tx = SimTransmitter(cfg)
channel = EthernetChannel(cfg, MolexCable, cable_length=100)
adc = Adc(cfg)

# fir = Fir(cfg)
fir = Fir(
    cfg,
#     taps=np.array([ 0.0026,  0.0166,  0.0532,  0.5011, -0.5011, -0.0532, -0.0166, -0.0026])
    # taps=np.array([  0.616,  0.,     0.,     0.535, -0.535, -0.,     0.,    -0.616])
    # taps=np.array([-1.,     1.,     0.778, -1.,    -0.111,  0.556, -1.,     0.333])
    # taps=np.array([ 0.2, -0.5,  0.7, -0.1,  0.1, -0.7,  0.5, -0.2]),
    taps=np.array([  0.04, -0.1,   0.22,  0.26, -0.26, -0.22,  0.1,  -0.04])
)

signals = tx.generate_signal(symbols=data, ignore_delay=ignore_delay)
signals["attenuated_signal"] = channel.attenuate(signals["smooth_signal"])

# rescale attenuated signal to [-1, 1]
scaling_factor = 4
signals["attenuated_signal_rescaled"] = signals["attenuated_signal"] * scaling_factor


sampling_point = 0.5 + sampling_offset
samples = adc.sample(signals["attenuated_signal_rescaled"], sampling_point=sampling_point)


# taps = fir.calculate_taps(channel=channel)
# fir.bode_plot(channel=channel)

samples["filtered"] = fir.filter(samples["samples"])

if ignore_delay:
    delay = 3
    samples["filtered"] = samples["filtered"][delay:]

samples["filtered"] = samples["filtered"][:samples["samples"].shape[0]]

reconstructed_data = np.zeros_like(samples["filtered"])
for i in range(samples["filtered"].shape[0]):
    if samples["filtered"][i] < thresholds[0]:
        reconstructed_data[i] = -2
    elif samples["filtered"][i] < thresholds[1]:
        reconstructed_data[i] = -1
    elif samples["filtered"][i] < thresholds[2]:
        reconstructed_data[i] = 0
    elif samples["filtered"][i] < thresholds[3]:
        reconstructed_data[i] = 1
    else:
        reconstructed_data[i] = 2

calc_delay = 2

valid_data = data[10:-10]
valid_reconstructed_data = reconstructed_data[10+calc_delay:][:valid_data.shape[0]]

loss = np.sum(np.abs(valid_data != valid_reconstructed_data)) / len(valid_data)

print("data:", valid_data)
print("reconstructed_data:", valid_reconstructed_data)
print("loss:", loss)


total_gain = 1

samples["filtered"] *= total_gain

samples["filtered"] = np.concatenate([samples["filtered"][3:], np.zeros(3)])

samples["filtered_time"] = np.repeat(samples["filtered"], cfg.sim_frequency_hz // cfg.symbol_frequency_hz)
samples["filtered_time"] = np.concatenate([np.zeros(samples["indices"][0] // 2), samples["filtered_time"], np.zeros(samples["indices"][0] // 2)])[:len(signals["time_s"])]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(signals["time_s"] * 1e9, signals["raw_signal"], label="original")
ax.plot(signals["time_s"] * 1e9, signals["smooth_signal"], label="smoothed", alpha=0.25)
ax.plot(signals["time_s"] * 1e9, signals["attenuated_signal"], label="attenuated", alpha=0.25)
ax.plot(signals["time_s"] * 1e9, signals["attenuated_signal_rescaled"], label="attenuated rescaled", alpha=0.25)

ax.scatter(signals["time_s"][samples["indices"]] * 1e9, samples["samples"], label="samples", s=10)
ax.scatter(signals["time_s"][samples["indices"]] * 1e9, samples["filtered"], label="filtered samples", s=10)
ax.scatter(signals["time_s"][samples["indices"]] * 1e9, reconstructed_data, label="reconstructed", s=10)

ax.plot(signals["time_s"] * 1e9, samples["filtered_time"], label="filtered samples")

for threshold in thresholds:
    ax.axhline(threshold, color="gray", linestyle="--")

ax.set_xlabel("Time (ns)")
ax.set_ylabel("Amplitude")

ax.set_ylim(-1.1, 1.1)


ax.legend()

fig.savefig("simulation.png")


