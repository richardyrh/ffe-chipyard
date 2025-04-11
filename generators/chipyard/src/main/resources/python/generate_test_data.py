# Final correct FIR filter implementation
import numpy as np

from fir_filter import FirFilterGenerator


# define quantizationscaling factor
s = 100

gen = FirFilterGenerator()


taps = gen.calculate_taps()

taps[:] = 0
taps[0] = 0.5
taps[1] = 0.25
taps[2] = 0.125

q_taps = gen.quantize(taps, s)

print("taps", taps)
print("q_taps", q_taps)

with open("../memory/taps.hex", "w") as f:
    for tap in q_taps:
        f.write(f"{tap:02x}\n")

example_input = np.zeros((10,), dtype=np.float32)
example_input[0] = 1
example_input[1] = 1
example_input[2] = 1

q_example_input = gen.quantize(example_input, s)

with open("../memory/ffe_in.hex", "w") as f:
    for i in q_example_input:
        f.write(f"{i:02x} {i:02x} {i:02x} {i:02x}\n")


gen.simulate(example_input)

q_expected = gen.simulate_q(example_input, s)

with open("../memory/ffe_out.hex", "w") as f:
    for i in q_expected:
        f.write(f"{i:02x} {i:02x} {i:02x} {i:02x}\n")

