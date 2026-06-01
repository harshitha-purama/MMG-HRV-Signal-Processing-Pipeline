"""
HRV feature extraction for neurophysiological readiness state estimation.

Implements time-domain, frequency-domain, and nonlinear HRV features
as described in Task-Scheidt et al. (2021) and Shaffer & Ginsberg (2017).

Features extracted:
    Time-domain : RMSSD, SDNN, pNN50, mean RR
    Frequency   : LF power, HF power, LF/HF ratio  (Welch PSD on interpolated tachogram)
    Nonlinear   : Approximate Entropy (ApEn)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d
import neurokit2 as nk
import antropy


# ---------------------------------------------------------------------------
# Signal simulation
# ---------------------------------------------------------------------------

def simulate_ecg(
    duration: float = 60.0,
    fs: int = 1000,
    heart_rate: float = 70.0,
    noise_amplitude: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a synthetic single-lead ECG and its time axis.

    Parameters
    ----------
    duration : seconds of signal to generate
    fs       : sampling frequency (Hz)
    heart_rate : mean heart rate (bpm)
    noise_amplitude : Gaussian noise std relative to signal amplitude
    rng      : reproducible random state

    Returns
    -------
    time : (N,) seconds
    ecg  : (N,) normalised ECG amplitude
    """
    if rng is None:
        rng = np.random.default_rng(42)

    ecg_df, _ = nk.ecg_process(
        nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=heart_rate, noise=noise_amplitude),
        sampling_rate=fs,
    )
    time = np.arange(len(ecg_df)) / fs
    ecg = ecg_df["ECG_Clean"].to_numpy()
    return time, ecg


# ---------------------------------------------------------------------------
# R-peak detection → RR intervals
# ---------------------------------------------------------------------------

def detect_rpeaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """Return R-peak sample indices from a cleaned ECG signal."""
    _, info = nk.ecg_peaks(ecg, sampling_rate=fs)
    return info["ECG_R_Peaks"]


def rr_intervals(r_peaks: np.ndarray, fs: int) -> np.ndarray:
    """Return inter-beat intervals in milliseconds."""
    return np.diff(r_peaks) / fs * 1000.0


# ---------------------------------------------------------------------------
# Time-domain features
# ---------------------------------------------------------------------------

def _rmssd(rr: np.ndarray) -> float:
    successive_diffs = np.diff(rr)
    return float(np.sqrt(np.mean(successive_diffs ** 2)))


def _sdnn(rr: np.ndarray) -> float:
    return float(np.std(rr, ddof=1))


def _pnn50(rr: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(rr)) > 50.0) * 100.0)


def _mean_rr(rr: np.ndarray) -> float:
    return float(np.mean(rr))


# ---------------------------------------------------------------------------
# Frequency-domain features (Welch PSD on uniformly-resampled tachogram)
# ---------------------------------------------------------------------------

_LF_BAND = (0.04, 0.15)   # Hz
_HF_BAND = (0.15, 0.40)   # Hz
_RESAMPLE_FS = 4.0          # Hz — standard for HRV frequency analysis


def _band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _frequency_features(rr: np.ndarray, r_peaks: np.ndarray, fs: int) -> dict[str, float]:
    cumtime = np.cumsum(rr) / 1000.0
    cumtime = np.insert(cumtime, 0, 0.0)

    t_uniform = np.arange(0, cumtime[-1], 1.0 / _RESAMPLE_FS)
    interp = interp1d(cumtime, np.append(rr[0], rr), kind="cubic", fill_value="extrapolate")
    tachogram = interp(t_uniform)

    freqs, psd = welch(tachogram, fs=_RESAMPLE_FS, nperseg=min(256, len(tachogram)))

    lf = _band_power(freqs, psd, _LF_BAND)
    hf = _band_power(freqs, psd, _HF_BAND)
    lf_hf = lf / hf if hf > 0 else np.nan

    return {"LF_power_ms2": lf, "HF_power_ms2": hf, "LF_HF_ratio": lf_hf}


# ---------------------------------------------------------------------------
# Nonlinear features
# ---------------------------------------------------------------------------

def _apen(rr: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    r = r_factor * np.std(rr)
    return float(antropy.app_entropy(rr, order=m, metric="chebyshev"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_hrv_features(
    ecg: np.ndarray,
    fs: int,
) -> dict[str, float]:
    """Extract the full HRV feature set from a cleaned ECG signal.

    Returns a flat dict with all time-domain, frequency-domain, and
    nonlinear HRV features used for readiness state estimation.
    """
    r_peaks = detect_rpeaks(ecg, fs)
    rr = rr_intervals(r_peaks, fs)

    if len(rr) < 10:
        raise ValueError(f"Too few RR intervals ({len(rr)}) to extract HRV features.")

    features: dict[str, float] = {
        "RMSSD_ms": _rmssd(rr),
        "SDNN_ms": _sdnn(rr),
        "pNN50_pct": _pnn50(rr),
        "mean_RR_ms": _mean_rr(rr),
        "ApEn": _apen(rr),
    }
    features.update(_frequency_features(rr, r_peaks, fs))
    return features
