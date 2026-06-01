"""
MMG feature extraction for neuromuscular readiness state estimation.

Mechanomyography (MMG) captures the mechanical vibrations of contracting
muscle fibres using an accelerometer. Unlike EMG, it is unaffected by
FES electrical artefacts (Ibitoye et al. 2014; Beck et al. 2005).

Features extracted:
    RMS amplitude   : global signal energy, sensitive to contraction intensity
    Mean power freq : MPF decreases with neuromuscular fatigue
    Wavelet energy  : per-band energy via discrete wavelet transform (db4)

Signal model follows Hamzaid et al. (2017) Sensors 17(7):1627.
"""

from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------------
# Physiological signal model for four readiness states
# ---------------------------------------------------------------------------

_STATE_PARAMS: dict[str, dict] = {
    "high_readiness": {
        "amplitude": 1.0,
        "dominant_freq": 22.0,    # Hz — fresh muscle, high MPF
        "noise_level": 0.08,
        "drift_rate": 0.0,
    },
    "moderate_readiness": {
        "amplitude": 0.75,
        "dominant_freq": 18.0,
        "noise_level": 0.10,
        "drift_rate": 0.001,
    },
    "fatigue_overload": {
        "amplitude": 0.45,
        "dominant_freq": 12.0,    # Hz — fatigue shifts MPF downward
        "noise_level": 0.14,
        "drift_rate": 0.005,
    },
    "low_engagement": {
        "amplitude": 0.20,
        "dominant_freq": 15.0,
        "noise_level": 0.06,
        "drift_rate": 0.0,
    },
}


def simulate_mmg(
    state: str = "high_readiness",
    duration: float = 10.0,
    fs: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise a physiologically plausible MMG signal for a readiness state.

    The model superposes:
    - A dominant oscillation at the state's characteristic frequency
    - Harmonic content (2nd and 3rd harmonics at reduced amplitude)
    - Gaussian noise proportional to the state's noise level
    - A linear amplitude drift term for fatigue states

    Parameters
    ----------
    state    : one of "high_readiness", "moderate_readiness",
               "fatigue_overload", "low_engagement"
    duration : seconds
    fs       : sampling frequency (Hz)
    rng      : reproducible random state

    Returns
    -------
    time : (N,) seconds
    mmg  : (N,) acceleration amplitude (normalised, a.u.)
    """
    if state not in _STATE_PARAMS:
        raise ValueError(f"Unknown state '{state}'. Choose from {list(_STATE_PARAMS)}")

    if rng is None:
        rng = np.random.default_rng(42)

    p = _STATE_PARAMS[state]
    t = np.arange(int(duration * fs)) / fs

    f0 = p["dominant_freq"]
    amp = p["amplitude"]
    drift = 1.0 - p["drift_rate"] * t * fs / len(t) * duration

    signal = (
        amp * drift * (
            np.sin(2 * np.pi * f0 * t)
            + 0.35 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
        )
        + rng.normal(0, p["noise_level"], len(t))
    )

    signal = _bandpass(signal, fs, low=5.0, high=100.0)
    return t, signal


def _bandpass(x: np.ndarray, fs: int, low: float, high: float) -> np.ndarray:
    sos = butter(4, [low, high], btype="band", fs=fs, output="sos")
    from scipy.signal import sosfiltfilt
    return sosfiltfilt(sos, x)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _rms_amplitude(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def _mean_power_frequency(x: np.ndarray, fs: int) -> float:
    """Spectral first moment — decreases reliably with fatigue."""
    from scipy.signal import welch
    freqs, psd = welch(x, fs=fs, nperseg=min(512, len(x)))
    mask = (freqs >= 5) & (freqs <= 100)
    total_power = np.trapezoid(psd[mask], freqs[mask])
    if total_power == 0:
        return np.nan
    return float(np.trapezoid(freqs[mask] * psd[mask], freqs[mask]) / total_power)


def _wavelet_energy(x: np.ndarray, wavelet: str = "db4", levels: int = 5) -> dict[str, float]:
    """Return normalised energy in each DWT sub-band (approximation + details)."""
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=levels)
    energies = {f"WE_D{levels - i}": float(np.sum(c ** 2)) for i, c in enumerate(coeffs[1:], 0)}
    energies["WE_A"] = float(np.sum(coeffs[0] ** 2))
    total = sum(energies.values())
    if total > 0:
        energies = {k: v / total for k, v in energies.items()}
    return energies


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_mmg_features(
    mmg: np.ndarray,
    fs: int,
    wavelet: str = "db4",
    wavelet_levels: int = 5,
) -> dict[str, float]:
    """Extract the full MMG feature set from a raw accelerometer signal.

    Parameters
    ----------
    mmg           : raw MMG signal array
    fs            : sampling frequency (Hz)
    wavelet       : PyWavelets mother wavelet name
    wavelet_levels: DWT decomposition depth

    Returns
    -------
    Flat dict with RMS_amplitude, MPF_Hz, and per-band wavelet energies.
    """
    features: dict[str, float] = {
        "RMS_amplitude": _rms_amplitude(mmg),
        "MPF_Hz": _mean_power_frequency(mmg, fs),
    }
    features.update(_wavelet_energy(mmg, wavelet=wavelet, levels=wavelet_levels))
    return features
