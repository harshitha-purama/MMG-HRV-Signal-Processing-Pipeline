"""
Generate all results from real PhysioNet NSRDB data and save to results/.

Data provenance
---------------
All HRV analysis in this script uses real ECG recordings from the
Normal Sinus Rhythm Database (NSRDB), PhysioNet.
  Subjects : 6 (records 16265, 16272, 16273, 16420, 16483, 16539)
  Sampling : 128 Hz
  Windows  : 45 x 20-second non-overlapping segments per subject
             sampled across the first 5 hours of each recording
  Total    : 270 real ECG windows

Outputs
-------
results/
  hrv_features_real_nsrdb.csv          — all 270 windows, all HRV features
  hrv_subject_summary_real_nsrdb.csv   — per-subject mean ± SD
  figures/hrv_feature_distributions.png
  figures/hrv_subject_comparison.png
  figures/hrv_psd_multisubject.png
  figures/hrv_rmssd_timecourse.png
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.interpolate import interp1d
import neurokit2 as nk
from pathlib import Path

from src.data_loader import load_nsrdb_segment, build_hrv_pool
from src.hrv_features import detect_rpeaks, rr_intervals, extract_hrv_features

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

RECORDS = ["16265", "16272", "16273", "16420", "16483", "16539"]
FS = 128

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ── 1. Load real HRV feature pool ────────────────────────────────────────────
print("Loading real HRV feature pool (270 NSRDB windows)...")
df = build_hrv_pool(n_windows=270, duration_s=20.0)
# Force record column to string — pandas reads numeric-looking IDs as int from CSV
df["record"] = df["record"].astype(str)
df.to_csv(RESULTS / "hrv_features_real_nsrdb.csv", index=False)
print(f"  Saved: hrv_features_real_nsrdb.csv  ({len(df)} rows)")

# ── 2. Per-subject summary ────────────────────────────────────────────────────
feat_cols = ["RMSSD_ms", "SDNN_ms", "pNN50_pct", "mean_RR_ms",
             "LF_power_ms2", "HF_power_ms2", "LF_HF_ratio", "ApEn"]

summary = df.groupby("record")[feat_cols].agg(["mean", "std"]).round(2)
summary.columns = ["_".join(c) for c in summary.columns]
summary.index.name = "Subject (NSRDB)"
summary.to_csv(RESULTS / "hrv_subject_summary_real_nsrdb.csv")
print(f"  Saved: hrv_subject_summary_real_nsrdb.csv")

# ── 3. Figure: HRV feature distributions across all real windows ──────────────
print("Generating figures...")
features_to_plot = [
    ("RMSSD_ms",      "RMSSD (ms)",        "#2196F3"),
    ("SDNN_ms",       "SDNN (ms)",         "#4CAF50"),
    ("HF_power_ms2",  "HF Power (ms²/Hz)", "#E91E63"),
    ("LF_HF_ratio",   "LF/HF Ratio",       "#FF9800"),
    ("ApEn",          "ApEn",              "#9C27B0"),
    ("mean_RR_ms",    "Mean RR (ms)",      "#00BCD4"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()

for ax, (col, label, colour) in zip(axes, features_to_plot):
    ax.hist(df[col], bins=25, color=colour, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(df[col].median(), color="k", ls="--", lw=1.2, label=f"Median: {df[col].median():.1f}")
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "HRV Feature Distributions — Real Data\n"
    "PhysioNet NSRDB · 6 subjects · 270 × 20-second windows",
    fontweight="bold", fontsize=11
)
plt.tight_layout()
fig.savefig(FIGURES / "hrv_feature_distributions.png", bbox_inches="tight")
plt.close()
print("  Saved: figures/hrv_feature_distributions.png")

# ── 4. Figure: Per-subject HRV comparison ────────────────────────────────────
key_features    = ["RMSSD_ms", "HF_power_ms2", "LF_HF_ratio", "ApEn"]
key_labels      = ["RMSSD (ms)", "HF Power (ms²/Hz)", "LF/HF Ratio", "ApEn"]
subject_colours = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0", "#00BCD4"]
colour_map      = dict(zip(RECORDS, subject_colours))

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for ax, feat, flabel in zip(axes, key_features, key_labels):
    data_per_subject = [df[df["record"] == rec][feat].values for rec in RECORDS]
    bp = ax.boxplot(data_per_subject, patch_artist=True, widths=0.55,
                    medianprops=dict(color="k", lw=1.8),
                    whiskerprops=dict(color="#555"),
                    capprops=dict(color="#555"),
                    flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for patch, colour in zip(bp["boxes"], subject_colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(RECORDS) + 1))
    ax.set_xticklabels(RECORDS, rotation=35, fontsize=8)
    ax.set_xlabel("NSRDB Subject")
    ax.set_ylabel(flabel)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Per-Subject HRV Variability — Real NSRDB Data\n"
    "45 windows per subject · box = IQR · whiskers = 1.5×IQR",
    fontweight="bold", fontsize=11
)
plt.tight_layout()
fig.savefig(FIGURES / "hrv_subject_comparison.png", bbox_inches="tight")
plt.close()
print("  Saved: figures/hrv_subject_comparison.png")

# ── 5. Figure: RMSSD time-course across recording (circadian trend) ───────────
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=False)
axes = axes.flatten()

for ax, (rec, colour) in zip(axes, zip(RECORDS, subject_colours)):
    rec_df = df[df["record"] == rec].sort_values("start_min")
    ax.plot(rec_df["start_min"], rec_df["RMSSD_ms"], "o-", color=colour,
            lw=1.2, ms=4, alpha=0.8)
    ax.fill_between(rec_df["start_min"], rec_df["RMSSD_ms"], alpha=0.12, color=colour)
    ax.set_title(f"Subject {rec}", fontsize=10)
    ax.set_xlabel("Time in recording (min)")
    ax.set_ylabel("RMSSD (ms)")
    ax.grid(alpha=0.3)

fig.suptitle(
    "RMSSD Time-Course — Real NSRDB Data (first 5 hours)\n"
    "Vagal tone variation across subjects and recording windows",
    fontweight="bold", fontsize=11
)
plt.tight_layout()
fig.savefig(FIGURES / "hrv_rmssd_timecourse.png", bbox_inches="tight")
plt.close()
print("  Saved: figures/hrv_rmssd_timecourse.png")

# ── 6. Figure: Multi-subject HRV PSD overlay ─────────────────────────────────
print("Computing PSD from raw ECG (downloads from cache)...")
RESAMPLE_FS = 4.0
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
axes = axes.flatten()

for ax, (rec, colour) in zip(axes, zip(RECORDS, subject_colours)):
    psds = []
    start_mins = list(range(5, 300, 300 // 9))[:9]

    for start_min in start_mins:
        try:
            _, ecg_raw = load_nsrdb_segment(rec, start_min=start_min, duration_s=20)
            ecg = nk.ecg_clean(ecg_raw, sampling_rate=FS)
            r_peaks = detect_rpeaks(ecg, FS)
            rr = rr_intervals(r_peaks, FS)
            if len(rr) < 10:
                continue
            cumtime = np.insert(np.cumsum(rr) / 1000.0, 0, 0.0)
            t_uni = np.arange(0, cumtime[-1], 1.0 / RESAMPLE_FS)
            tacho = interp1d(cumtime, np.append(rr[0], rr), kind="cubic",
                             fill_value="extrapolate")(t_uni)
            freqs, psd = welch(tacho, fs=RESAMPLE_FS, nperseg=min(64, len(tacho)))
            psds.append(psd)
        except Exception:
            pass

    if psds:
        psd_arr = np.array(psds)
        psd_mean = psd_arr.mean(axis=0)
        psd_std  = psd_arr.std(axis=0)
        ax.semilogy(freqs, psd_mean, color=colour, lw=1.5, label=f"Mean (n={len(psds)})")
        ax.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std,
                        alpha=0.2, color=colour)
        ax.axvspan(0.04, 0.15, alpha=0.12, color="orange", label="LF")
        ax.axvspan(0.15, 0.40, alpha=0.12, color="green",  label="HF")
        ax.set_xlim(0, 0.5)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (ms²/Hz)")
        ax.set_title(f"Subject {rec}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

fig.suptitle(
    "HRV Power Spectral Density — Real NSRDB Data\n"
    "Mean ± SD across 9 windows per subject · LF=orange · HF=green",
    fontweight="bold", fontsize=11
)
plt.tight_layout()
fig.savefig(FIGURES / "hrv_psd_multisubject.png", bbox_inches="tight")
plt.close()
print("  Saved: figures/hrv_psd_multisubject.png")

# ── 7. Print summary table ────────────────────────────────────────────────────
print("\n── Real NSRDB HRV Summary ──────────────────────────────────────────")
print(df.groupby("record")[["RMSSD_ms", "SDNN_ms", "HF_power_ms2", "LF_HF_ratio"]].mean().round(1).to_string())
print(f"\nOverall RMSSD: {df.RMSSD_ms.mean():.1f} ± {df.RMSSD_ms.std():.1f} ms  "
      f"(range {df.RMSSD_ms.min():.1f}–{df.RMSSD_ms.max():.1f} ms)")
print(f"Overall LF/HF: {df.LF_HF_ratio.mean():.2f} ± {df.LF_HF_ratio.std():.2f}")

print("\nAll results saved to results/")
