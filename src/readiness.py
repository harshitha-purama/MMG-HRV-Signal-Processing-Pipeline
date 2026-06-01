"""
Neurophysiological readiness state classification from fused MMG-HRV features.

Four states are defined operationally following the criteria proposed in the
MMG-HRV closed-loop FES research framework (Hamzaid & Naeem, 2026):

    HIGH_READINESS   — elevated vagal tone + preserved neuromuscular function
    MODERATE         — intermediate autonomic and neuromuscular indicators
    FATIGUE_OVERLOAD — declining MMG amplitude/MPF + reduced HRV
    LOW_ENGAGEMENT   — low MMG amplitude without fatigue markers + low HRV variability

This module provides:
    - A threshold-based rule classifier (interpretable baseline)
    - A scikit-learn SVM wrapper for data-driven classification
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class ReadinessState(str, Enum):
    HIGH_READINESS = "high_readiness"
    MODERATE = "moderate_readiness"
    FATIGUE_OVERLOAD = "fatigue_overload"
    LOW_ENGAGEMENT = "low_engagement"

    def fes_action(self) -> str:
        """Recommended FES controller action for this state."""
        actions = {
            ReadinessState.HIGH_READINESS:   "Increase stimulation intensity — optimal neuroplasticity window",
            ReadinessState.MODERATE:         "Maintain current stimulation parameters",
            ReadinessState.FATIGUE_OVERLOAD: "Reduce or pause stimulation — risk of overload",
            ReadinessState.LOW_ENGAGEMENT:   "Apply arousal protocol — insufficient cortical engagement",
        }
        return actions[self]


@dataclass
class ReadinessFeatureVector:
    """Combined MMG + HRV feature vector for a single assessment window."""
    RMSSD_ms: float
    HF_power_ms2: float
    LF_HF_ratio: float
    RMS_amplitude: float
    MPF_Hz: float
    ApEn: float = np.nan


# ---------------------------------------------------------------------------
# Threshold-based classifier (interpretable, no training data required)
# ---------------------------------------------------------------------------

# Thresholds are set to physiologically representative values for a healthy
# adult at rest. Patient-specific normalisation should be applied before
# calling this function in clinical use (divide by individual baseline).

_THRESHOLDS = {
    "high": {
        "RMSSD_ms_min":     40.0,
        "HF_power_ms2_min": 300.0,
        "LF_HF_ratio_max":  1.5,
        "RMS_amplitude_min": 0.60,
        "MPF_Hz_min":       18.0,
    },
    "fatigue": {
        "RMS_amplitude_max": 0.35,
        "MPF_Hz_max":        14.0,
        "RMSSD_ms_max":      25.0,
    },
    "low_engagement": {
        "RMS_amplitude_max": 0.30,
        "MPF_Hz_min":        14.0,   # absence of fatigue frequency shift
        "RMSSD_ms_max":      28.0,
    },
}


def classify_readiness(fv: ReadinessFeatureVector) -> ReadinessState:
    """Rule-based readiness classification from a combined MMG-HRV feature vector.

    Uses a priority hierarchy:
        1. FATIGUE_OVERLOAD  (safety-critical — checked first)
        2. HIGH_READINESS
        3. LOW_ENGAGEMENT
        4. MODERATE          (default)
    """
    th = _THRESHOLDS

    fatigue_amplitude = fv.RMS_amplitude < th["fatigue"]["RMS_amplitude_max"]
    fatigue_freq      = fv.MPF_Hz < th["fatigue"]["MPF_Hz_max"]
    fatigue_hrv       = fv.RMSSD_ms < th["fatigue"]["RMSSD_ms_max"]
    if fatigue_amplitude and fatigue_freq and fatigue_hrv:
        return ReadinessState.FATIGUE_OVERLOAD

    high_hrv  = (fv.RMSSD_ms >= th["high"]["RMSSD_ms_min"]
                 and fv.HF_power_ms2 >= th["high"]["HF_power_ms2_min"]
                 and fv.LF_HF_ratio <= th["high"]["LF_HF_ratio_max"])
    high_mmg  = (fv.RMS_amplitude >= th["high"]["RMS_amplitude_min"]
                 and fv.MPF_Hz >= th["high"]["MPF_Hz_min"])
    if high_hrv and high_mmg:
        return ReadinessState.HIGH_READINESS

    low_amp = fv.RMS_amplitude < th["low_engagement"]["RMS_amplitude_max"]
    no_fatigue_freq = fv.MPF_Hz >= th["low_engagement"]["MPF_Hz_min"]
    low_hrv = fv.RMSSD_ms < th["low_engagement"]["RMSSD_ms_max"]
    if low_amp and no_fatigue_freq and low_hrv:
        return ReadinessState.LOW_ENGAGEMENT

    return ReadinessState.MODERATE


# ---------------------------------------------------------------------------
# SVM-based classifier (data-driven, for Phase 2 of the research protocol)
# ---------------------------------------------------------------------------

def build_svm_classifier() -> Pipeline:
    """Return an untrained SVM pipeline with standard scaling.

    Intended for use in the supervised classification experiments described
    in Phase 2 of the research protocol (n=15 healthy volunteers).

    Usage
    -----
    clf = build_svm_classifier()
    clf.fit(X_train, y_train)       # y: ReadinessState string labels
    y_pred = clf.predict(X_test)
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
    ])


def feature_vector_to_array(fv: ReadinessFeatureVector) -> np.ndarray:
    """Convert a ReadinessFeatureVector to a numpy row vector for sklearn."""
    return np.array([
        fv.RMSSD_ms,
        fv.HF_power_ms2,
        fv.LF_HF_ratio,
        fv.RMS_amplitude,
        fv.MPF_Hz,
        fv.ApEn if not np.isnan(fv.ApEn) else 0.0,
    ])
