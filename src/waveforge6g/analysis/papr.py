"""Periodic complex-baseband Fourier interpolation and empirical PAPR CCDF.

Interpolation is an explicit reconstruction assumption, not an analog RF PA
model. Even-N Nyquist is assigned to the negative-frequency bin, consistent
with numpy.fft.fftfreq. It is not split as in real-signal interpolation.
Useful samples exclude prefix energy; all compared waveforms use the same rule.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from waveforge6g.core.bits import random_bits
from waveforge6g.core.metrics import ccdf, papr_db
from waveforge6g.core.modulation import bits_per_symbol, modulate
from waveforge6g.receivers.detectors import WaveformLike


def oversample(signal: ArrayLike, factor: int = 4) -> NDArray[np.complex128]:
    """Periodic bandlimited interpolation from N to factor*N complex samples.

    With signed DFT indices k=floor-centered bins, evaluate the same Fourier
    series at the denser sample times. sqrt(factor) corrects unitary-IFFT
    scaling. Thus y[::factor]=x and ensemble/sample mean power is preserved.
    """
    samples = np.asarray(signal, dtype=np.complex128)
    if samples.ndim != 1 or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("signal must be a finite nonempty vector")
    if isinstance(factor, bool) or not isinstance(factor, int | np.integer) or factor < 1:
        raise ValueError("oversampling factor must be a positive integer")
    if factor == 1:
        return samples.copy()
    n = samples.size
    frequency_indices = np.rint(np.fft.fftfreq(n) * n).astype(int)
    spectrum = np.zeros(n * factor, dtype=np.complex128)
    spectrum[frequency_indices % (n * factor)] = np.fft.fft(samples, norm="ortho") * np.sqrt(factor)
    return np.fft.ifft(spectrum, norm="ortho")


def papr_samples(
    waveform: WaveformLike,
    modulation: str,
    n_frames: int,
    rng: np.random.Generator,
    oversampling: int = 4,
) -> NDArray[np.float64]:
    """Generate per-frame PAPR values in dB using useful-block samples only."""
    if not isinstance(n_frames, int | np.integer) or n_frames < 0:
        raise ValueError("n_frames must be a nonnegative integer")
    values = np.empty(n_frames, dtype=float)
    bit_count = waveform.n_symbols * bits_per_symbol(modulation)
    for frame in range(n_frames):
        symbols = modulate(random_bits(bit_count, rng), modulation)
        values[frame] = papr_db(oversample(waveform.synthesis(symbols), oversampling))
    return values


def papr_ccdf(samples_db: ArrayLike, thresholds_db: ArrayLike) -> NDArray[np.float64]:
    """Strict P(PAPR_dB > threshold_dB), NaN when no frames were observed."""
    return ccdf(samples_db, thresholds_db)
