import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


class Pam5SymbolGenerator:
    @staticmethod
    def add_zeros(symbols: np.ndarray, prepend_zeros: int, append_zeros: int) -> np.ndarray:
        return np.concatenate([np.zeros(prepend_zeros), symbols, np.zeros(append_zeros)])

    @staticmethod
    def random(n_symbols: int, prepend_zeros: int = 10, append_zeros: int = 10) -> np.ndarray:
        """
        Generate a random PAM5 symbol sequence
        """
        symbols = np.random.randint(-2, 3, n_symbols, dtype=np.int8).astype(np.float32)
        return Pam5SymbolGenerator.add_zeros(symbols, prepend_zeros, append_zeros)

    @staticmethod
    def alternate_01(n_symbols: int, prepend_zeros: int = 10, append_zeros: int = 10) -> np.ndarray:
        """
        Generate a PAM5 symbol sequence that alternates between {-1, 0, 1, 0}
        """
        symbols = np.zeros(n_symbols, dtype=np.float32)
        symbols[0::4] = -1
        symbols[1::4] = 0
        symbols[2::4] = 1
        symbols[3::4] = 0
        return Pam5SymbolGenerator.add_zeros(symbols, prepend_zeros, append_zeros)

    @staticmethod
    def alternate_02(n_symbols: int, prepend_zeros: int = 10, append_zeros: int = 10) -> np.ndarray:
        """
        Generate a PAM5 symbol sequence that alternates between {-2, 0, 2, 0}
        """
        symbols = np.zeros(n_symbols, dtype=np.float32)
        symbols[0::4] = -2
        symbols[1::4] = 0
        symbols[2::4] = 2
        symbols[3::4] = 0
        return Pam5SymbolGenerator.add_zeros(symbols, prepend_zeros, append_zeros)

    @staticmethod
    def alternate_012(n_symbols: int, prepend_zeros: int = 10, append_zeros: int = 10) -> np.ndarray:
        """
        Generate a PAM5 symbol sequence that alternates between {-2, -1, 0, 1, 2, 1, 0, -1}
        """
        symbols = np.zeros(n_symbols, dtype=np.float32)
        symbols[0::8] = -2
        symbols[1::8] = -1
        symbols[2::8] = 0
        symbols[3::8] = 1
        symbols[4::8] = 2
        symbols[5::8] = 1
        symbols[6::8] = 0
        symbols[7::8] = -1
        return Pam5SymbolGenerator.add_zeros(symbols, prepend_zeros, append_zeros)

    @staticmethod
    def sine_wave(n_symbols: int, prepend_zeros: int = 10, append_zeros: int = 10) -> np.ndarray:
        symbols = np.zeros(n_symbols, dtype=np.float32)
        symbols[0::8] = np.sin(0 * np.pi / 4)
        symbols[1::8] = np.sin(1 * np.pi / 4)
        symbols[2::8] = np.sin(2 * np.pi / 4)
        symbols[3::8] = np.sin(3 * np.pi / 4)
        symbols[4::8] = np.sin(4 * np.pi / 4)
        symbols[5::8] = np.sin(5 * np.pi / 4)
        symbols[6::8] = np.sin(6 * np.pi / 4)
        symbols[7::8] = np.sin(7 * np.pi / 4)
        return Pam5SymbolGenerator.add_zeros(symbols, prepend_zeros, append_zeros)
    
    @staticmethod
    def sawtooth_wave(n_symbols: int, prepend_zeros: int = 10, append_zeros: int = 10) -> np.ndarray:
        symbols = np.zeros(n_symbols, dtype=np.float32)
        symbols[0::5] = -2
        symbols[1::5] = -1
        symbols[2::5] = 0
        symbols[3::5] = 1
        symbols[4::5] = 2
        return Pam5SymbolGenerator.add_zeros(symbols, prepend_zeros, append_zeros)