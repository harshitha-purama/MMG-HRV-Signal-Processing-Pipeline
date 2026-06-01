"""
PhysioNet data loader for the MMG-HRV pipeline.

Downloads ECG segments from the Normal Sinus Rhythm Database (NSRDB)
via the wfdb library and caches them locally in data/.

NSRDB (physionet.org/content/nsrdb/1.0.0):
    18 long-term Holter ECG recordings from healthy subjects
    Sampling frequency: 128 Hz
    Two leads: ECG1, ECG2
    Duration: 24 hours per subject

Usage
-----
    from src.data_loader import load_nsrdb_segment
    fs, ecg = load_nsrdb_segment(record='16265', start_min=5, duration_s=60)
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path

_CACHE_DIR = Path(__file__).parent.parent / "data" / "nsrdb_cache"
_NSRDB_FS = 128  # Hz


def load_nsrdb_segment(
    record: str = "16265",
    start_min: float = 5.0,
    duration_s: float = 60.0,
    lead: int = 0,
) -> tuple[int, np.ndarray]:
    """Download and return a short ECG segment from the NSRDB.

    The segment is cached as a .npy file after first download so
    subsequent calls are instant.

    Parameters
    ----------
    record     : NSRDB record ID (e.g. '16265', '16272', '16273')
    start_min  : start time in minutes from the beginning of the recording
    duration_s : length of segment in seconds
    lead       : channel index (0 = ECG1, 1 = ECG2)

    Returns
    -------
    fs  : sampling frequency (128 Hz)
    ecg : (N,) ECG amplitude in physical units (mV)
    """
    import wfdb

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{record}_start{start_min:.0f}_dur{duration_s:.0f}_lead{lead}.npy"

    if cache_file.exists():
        return _NSRDB_FS, np.load(cache_file)

    start_sample = int(start_min * 60 * _NSRDB_FS)
    end_sample   = start_sample + int(duration_s * _NSRDB_FS)

    record_obj = wfdb.rdrecord(
        record,
        pn_dir="nsrdb",
        sampfrom=start_sample,
        sampto=end_sample,
        channels=[lead],
    )
    ecg = record_obj.p_signal[:, 0].astype(np.float64)
    np.save(cache_file, ecg)
    return _NSRDB_FS, ecg


def available_records() -> list[str]:
    """Return the list of available NSRDB record IDs."""
    import wfdb
    return wfdb.get_record_list("nsrdb")


def build_hrv_pool(
    n_windows: int = 270,
    duration_s: float = 20.0,
    cache_path: str | None = None,
) -> "pandas.DataFrame":
    """Extract real HRV features from multiple NSRDB subjects and time windows.

    Downloads ECG segments spread across 6 subjects and many time points,
    extracts the full HRV feature set from each, and returns a DataFrame
    sorted by RMSSD. Subsequent calls load from cache instantly.

    The pool is used in the multimodal classifier notebook to supply
    real autonomic data for training, in place of simulated ECG.

    Parameters
    ----------
    n_windows  : total number of windows to extract (spread evenly across subjects)
    duration_s : each window length in seconds (20s gives ~30 RR intervals at 128 Hz)
    cache_path : optional override for the cache CSV path

    Returns
    -------
    DataFrame with columns matching extract_hrv_features() output,
    plus 'record' and 'start_min', sorted by RMSSD_ms ascending.
    """
    import pandas as pd
    import neurokit2 as nk
    from .hrv_features import extract_hrv_features

    cache_file = Path(cache_path) if cache_path else (
        _CACHE_DIR / f"hrv_pool_n{n_windows}_dur{duration_s:.0f}s.csv"
    )

    if cache_file.exists():
        return pd.read_csv(cache_file)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Use 6 well-known NSRDB records; spread windows evenly across each
    records = ["16265", "16272", "16273", "16420", "16483", "16539"]
    windows_per_record = n_windows // len(records)

    # Sample start times spread across 0–300 minutes (first 5 hours)
    # to capture natural circadian HRV variation
    start_times = [float(t) for t in range(5, 300, 300 // windows_per_record)][:windows_per_record]

    rows = []
    for rec in records:
        for start_min in start_times:
            try:
                fs, ecg_raw = load_nsrdb_segment(rec, start_min=start_min, duration_s=duration_s)
                ecg = nk.ecg_clean(ecg_raw, sampling_rate=fs)
                feats = extract_hrv_features(ecg, fs)
                feats["record"] = rec
                feats["start_min"] = start_min
                rows.append(feats)
            except Exception:
                pass  # skip windows with too few R-peaks

    df = pd.DataFrame(rows).sort_values("RMSSD_ms").reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    return df
