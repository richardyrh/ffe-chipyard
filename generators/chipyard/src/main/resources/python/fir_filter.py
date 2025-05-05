# Final correct FIR filter implementation
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from simulator.ethernet_channels import EthernetChannelProfile


np.set_printoptions(precision=4)


class FirFilterGenerator:
    """
    Generates a FIR filter based on the cable length and number of taps.
    """
    def __init__(self, cable_length: float = 10, num_taps: int = 8, fir_type: int = 4):
        """
        Initializes the FIR filter generator.

        Args:
            cable_length: The length of the cable in meters.
            num_taps: The number of taps in the FIR filter.
            fir_type: The type of FIR filter to generate.
        """
        self.cable_length = cable_length
        self.num_taps = num_taps
        self.fir_type = fir_type
        self.taps = None
    
    @staticmethod
    def quantize(data: np.ndarray, scale: float = 1e3, max_value: int = 127, min_value: int = -128):
        """
        Quantizes the data to the specified scale and range.

        Args:
            data: The data to quantize.
            scale: The scaling factor.
            max_value: The maximum value of the quantized data, this should correspond to the bitwidth of the data type.
            min_value: The minimum value of the quantized data, this should correspond to the bitwidth of the data type.
        """
        q = (data * scale).astype(np.int32)
        q = np.clip(q, min_value, max_value)
        return q
    
    @staticmethod
    def dequantize(data: np.ndarray, scale: float = 1e3):
        """
        Dequantizes the data to the specified scale.

        Args:
            data: The data to dequantize.
            scale: The scaling factor.
        """
        return data.astype(np.float32) / scale
    
    def calculate_taps(self, channel_profile: EthernetChannelProfile, cable_length: float = 100) -> np.ndarray:
        """
        Calculates the taps for the FIR filter.
        """
        self.cable_length = cable_length
        # Original frequency and attenuation data
        freq_MHz = np.array([0, 1, 4, 8, 10, 16, 20, 25, 31.25, 62.5])
        attenuation_dB = np.array([0, 2.0, 3.8, 5.3, 6.0, 7.6, 8.5, 9.5, 10.7, 15.4])
        # attenuation_dB[-1] += 10
        attenuation_dB = np.max(attenuation_dB) - attenuation_dB + 0
        attenuation_dB /= (100 / self.cable_length)

        # Convert MHz to Hz
        freq_Hz = freq_MHz * 1e6

        # Nyquist frequency (fs/2 = highest freq)
        nyquist_freq = freq_Hz[-1]

        self.nyquist_freq = nyquist_freq

        gain = 10 ** (-attenuation_dB / 20)

        if self.fir_type == 2:
            gain[-1] = 0
        elif self.fir_type == 3:
            gain[-1] = 0
        elif self.fir_type == 4:
            gain[0] = 0


        print("gain", gain)

        # fir_taps = signal.firwin2(num_taps, freq_Hz / np.max(freq_Hz), gain, fs=2.0)
        fir_taps = signal.firwin2(self.num_taps, freq_Hz / np.max(freq_Hz), gain, fs=2.0, antisymmetric=(self.fir_type % 2 == 0))

        # Frequency response visualization
        freq_response, response = signal.freqz(fir_taps, whole=True)
        plt.figure(figsize=(12, 5))
        ax = plt.subplot(121)
        ax.plot(((freq_response / np.pi) * nyquist_freq / 1e6)[:8000], (20 * np.log10(np.abs(response)))[:8000])
        plt.title(f"FIR Filter Frequency Response (Type {self.fir_type})")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True)
        plt.savefig(f"filter_{self.fir_type}.png")
        # plt.show()

        print(fir_taps)
        self.taps = fir_taps
        return fir_taps

    def test_filter(self):
        """
        Run a simple test of the filter and generate a plot of the original and filtered signal attenuation.
        """
        freq_MHz = np.array([1, 4, 8, 10, 16, 20, 25, 31.25, 62.5, 100, 200, 250])
        attenuation_dB = np.array([2.0, 3.8, 5.3, 6.0, 7.6, 8.5, 9.5, 10.7, 15.4, 19.8, 29.0, 32.8])
        attenuation_dB /= (100 / self.cable_length)

        freq_Hz = freq_MHz * 1e6
        # nyquist_freq = freq_Hz[-1]
        nyquist_freq = self.nyquist_freq

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
        filtered_signals = np.array([signal.lfilter(self.taps, 1.0, sig) for sig in signals])

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
    
    def simulate(self, example_input: np.ndarray) -> np.ndarray:
        """
        Simulates the FIR filter on the example input.

        Args:
            example_input: The input to the FIR filter.
        """
        filtered_signals = signal.lfilter(self.taps, 1.0, example_input)
        return filtered_signals
    
    def simulate_q(self, example_input: np.ndarray, s: float = 1e3) -> np.ndarray:
        """
        Simulates the FIR filter on the example input and quantizes the input and taps.

        Args:
            example_input: The input to the FIR filter.
            s: The scaling factor.
        """
        golden = self.simulate(example_input)

        example_input_q = self.quantize(example_input, s)
        taps_q = self.quantize(self.taps, s)

        assert np.all(taps_q >= np.iinfo(np.int8).min) and np.all(taps_q <= np.iinfo(np.int8).max), f"Taps out of bounds of int8: [{np.min(taps_q)}, {np.max(taps_q)}]"
        assert np.all(example_input_q >= np.iinfo(np.int8).min) and np.all(example_input_q <= np.iinfo(np.int8).max), f"Input out of bounds of int8: [{np.min(example_input_q)}, {np.max(example_input_q)}]"

        print("  Quantized taps:", taps_q)
        actual_q = signal.lfilter(taps_q, 1.0, example_input_q).astype(np.int32)
        print("  Result signals:", actual_q)

        actual = signal.lfilter(taps_q / 128, 1.0, example_input_q / 128)
        # actual = self.dequantize(actual_q, s**2)

        print("Golden:", golden)
        print("Actual:", actual)
        print("Error:", np.max(np.abs(golden - actual)))

        return actual_q


if __name__ == "__main__":
    gen = FirFilterGenerator()
    gen.calculate_taps()
    gen.test_filter()
