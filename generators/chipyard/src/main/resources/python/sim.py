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

# === create data to send, in range {-2, -1, 0, 1, 2} ===

# data = Pam5SymbolGenerator.random(n_symbols)
# data = Pam5SymbolGenerator.alternate_01(n_symbols)
# data = Pam5SymbolGenerator.alternate_02(n_symbols)
# data = Pam5SymbolGenerator.alternate_012(n_symbols)
# data = Pam5SymbolGenerator.sine_wave(n_symbols)
data = Pam5SymbolGenerator.sawtooth_wave(n_symbols)




tx = SimTransmitter(cfg)
adc = Adc(cfg)

signals = tx.generate_signal(symbols=data, ignore_delay=False)

channel = EthernetChannel(cfg, MolexCable, cable_length=100)

signals["attenuated_signal"] = channel.attenuate(signals["smooth_signal"])

samples = adc.sample(signals["attenuated_signal"], sampling_point=0.66)



fir = Fir(cfg)

taps = fir.calculate_taps(channel=channel)

fir.bode_plot(channel=channel)

# breakpoint()
exit()




plt.plot(signals["time_s"] * 1e9, signals["raw_signal"], label="original")
plt.plot(signals["time_s"] * 1e9, signals["smooth_signal"], label="smoothed")
plt.plot(signals["time_s"] * 1e9, signals["attenuated_signal"], label="attenuated")
plt.scatter(signals["time_s"][samples["indices"]] * 1e9, samples["samples"], label="samples", s=10)
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")

plt.ylim(-1.1, 1.1)


plt.legend()

plt.savefig("simulation.png")













