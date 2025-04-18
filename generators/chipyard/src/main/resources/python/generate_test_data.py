"""
generate_test_data.py

Usage:

```bash
# cd into chipyard root directory
source ./env.sh
python ./generators/chipyard/src/main/resources/python/generate_test_data.py
```

The script will generate the following files:

- taps.hex: the tap (weight) values for the FIR filter
- ffe_in.hex: the input values for the FFE
- ffe_out.hex: the expected output values for the FFE

"""

import os

# Final correct FIR filter implementation
import numpy as np

from fir_filter import FirFilterGenerator


output_directory = "./generators/chipyard/src/main/resources/memory/"

# define quantizationscaling factor
s = 128

gen = FirFilterGenerator()


taps = gen.calculate_taps()

taps[:] = 0
taps[0] = 0.5
taps[1] = 0.25
taps[2] = 0.125

q_taps = gen.quantize(taps, s)

print("taps", taps)
print("q_taps", q_taps)

with open(os.path.join(output_directory, "taps.hex"), "w") as f:
    for tap in q_taps:
        f.write(f"{tap:02x}\n")

example_input = np.zeros((10,), dtype=np.float32)
example_input[0] = 1
# example_input[1] = 1
# example_input[2] = 1

q_example_input = gen.quantize(example_input, s)

with open(os.path.join(output_directory, "ffe_in.hex"), "w") as f:
    f.write(f"00 00 00 00\n")
    for i in q_example_input:
        f.write(f"{i:02x} {i:02x} {i:02x} {i:02x}\n")


gen.simulate(example_input)

q_expected = gen.simulate_q(example_input, s)

with open(os.path.join(output_directory, "ffe_out.hex"), "w") as f:
    f.write(f"00 00 00 00\n")
    for i in q_expected:
        f.write(f"{i:02x} {i:02x} {i:02x} {i:02x}\n")

