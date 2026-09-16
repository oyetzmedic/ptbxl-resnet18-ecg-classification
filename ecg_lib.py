"""
ECG preprocessing and classical feature extraction for PTB-XL.

Pure numpy/scipy so everything here is unit-testable without data downloads.
Design notes are in comments because an interviewer will ask "why".

Dataset: PTB-XL (Wagner et al., 2020), 100 Hz version. 12-lead, 10 s records.
Task here: 5-class diagnostic superclass classification (NORM, MI, STTC, CD,
HYP), restricted to records carrying exactly one superclass to keep this a
clean single-label problem — a documented simplification, not a hidden one.
"""
import numpy as np
import pandas as pd
from scipy import signal as sps

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

FS = 100          # PTB-XL low-rate sampling frequency (Hz)
LEAD_II = 1       # conventional rhythm lead index in PTB-XL ordering
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ---------------------------------------------------------------------------
# Filtering — remove baseline wander (<0.5 Hz) and mains/high-frequency noise
# (>40 Hz) while preserving the QRS band. Butterworth for flat passband;
# filtfilt for zero phase shift so R-peak timing is not distorted.
# ---------------------------------------------------------------------------
def butter_bandpass(low=0.5, high=40.0, fs=FS, order=3):
    nyq = fs / 2.0
    return sps.butter(order, [low / nyq, high / nyq], btype="band")


def bandpass_filter(x, low=0.5, high=40.0, fs=FS, order=3):
    """x: (n_samples,) or (n_samples, n_leads). Zero-phase bandpass."""
    b, a = butter_bandpass(low, high, fs, order)
    return sps.filtfilt(b, a, x, axis=0)


def znormalise(x, eps=1e-8):
    """Per-lead z-normalisation. Amplitude varies with electrode contact and
    body habitus; class information lives in morphology, not raw millivolts."""
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)


# ---------------------------------------------------------------------------
# R-peak detection — deliberately simple and transparent (band-limited energy
# + distance-constrained peak picking), not a black box. Good enough for
# rate/rhythm features; limitations stated in the README.
# ---------------------------------------------------------------------------
def detect_r_peaks(sig, fs=FS):
    """sig: 1-D single lead. Returns sample indices of R peaks."""
    filt = bandpass_filter(sig, low=5.0, high=15.0, fs=fs)  # QRS energy band
    energy = filt ** 2
    # Refractory constraint: physiological upper bound ~200 bpm -> 0.3 s apart
    min_dist = int(0.3 * fs)
    thresh = 0.35 * float(np.max(energy)) if np.max(energy) > 0 else 0.0
    peaks, _ = sps.find_peaks(energy, distance=min_dist, height=thresh)
    return peaks


def heart_rate_features(sig, fs=FS):
    """Rate/rhythm descriptors from lead II. Returns dict of scalars."""
    peaks = detect_r_peaks(sig, fs)
    if len(peaks) < 3:
        return {"hr_mean": 0.0, "hr_std": 0.0, "rr_mean": 0.0,
                "rr_std": 0.0, "rr_cv": 0.0, "n_beats": float(len(peaks))}
    rr = np.diff(peaks) / fs                      # seconds between beats
    hr = 60.0 / rr
    return {
        "hr_mean": float(hr.mean()), "hr_std": float(hr.std()),
        "rr_mean": float(rr.mean()), "rr_std": float(rr.std()),
        "rr_cv": float(rr.std() / (rr.mean() + 1e-8)),
        "n_beats": float(len(peaks)),
    }


def spectral_band_powers(sig, fs=FS):
    """Relative Welch band powers — crude but honest frequency descriptors."""
    f, pxx = sps.welch(sig, fs=fs, nperseg=min(256, len(sig)))
    total = _trapz(pxx, f) + 1e-12
    bands = {"p_0_4": (0, 4), "p_4_15": (4, 15), "p_15_40": (15, 40)}
    return {k: float(_trapz(pxx[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]) / total)
            for k, (lo, hi) in bands.items()}


def classical_feature_vector(record):
    """record: (1000, 12) filtered+normalised ECG. Returns 1-D feature array:
    lead-II rate/rhythm + per-lead moments + lead-II band powers."""
    feats = []
    hr = heart_rate_features(record[:, LEAD_II])
    feats.extend(hr[k] for k in sorted(hr))
    for lead in range(record.shape[1]):
        x = record[:, lead]
        feats.extend([x.mean(), x.std(), float(pd.Series(x).skew()),
                      float(pd.Series(x).kurt())])
    bp = spectral_band_powers(record[:, LEAD_II])
    feats.extend(bp[k] for k in sorted(bp))
    return np.asarray(feats, dtype=np.float64)


# ---------------------------------------------------------------------------
# Labels and splits
# ---------------------------------------------------------------------------
def single_superclass_labels(df, agg_df):
    """df: ptbxl_database with scp_codes (dict per row). agg_df: scp_statements
    indexed by code with diagnostic_class. Keeps records mapping to EXACTLY one
    diagnostic superclass; returns df with a 'superclass' column."""
    diag = agg_df[agg_df.diagnostic == 1]

    def to_super(codes):
        s = {diag.loc[c].diagnostic_class for c in codes if c in diag.index}
        return s.pop() if len(s) == 1 else None

    out = df.copy()
    out["superclass"] = out.scp_codes.apply(to_super)
    return out[out.superclass.isin(SUPERCLASSES)]


def patientwise_fold_split(df):
    """PTB-XL ships strat_fold 1..10, stratified and PATIENT-DISJOINT by
    construction. Recommended: 1-8 train, 9 val, 10 test. We use it as-is —
    inventing our own split would risk the exact leakage this portfolio
    exists to avoid."""
    return (df[df.strat_fold <= 8], df[df.strat_fold == 9], df[df.strat_fold == 10])


def assert_patient_disjoint(train_df, test_df):
    """Hard guarantee: no patient_id in both partitions. Raises on violation."""
    overlap = set(train_df.patient_id) & set(test_df.patient_id)
    if overlap:
        raise ValueError(f"PATIENT LEAKAGE: {len(overlap)} patient(s) in both "
                         f"partitions, e.g. {sorted(overlap)[:5]}")
    return True


# ---------------------------------------------------------------------------
# Time-frequency representation for the fine-tuning route
# ---------------------------------------------------------------------------
def spectrogram_image(sig, fs=FS, out_hw=(224, 224)):
    """Lead signal -> log-spectrogram resized to out_hw, scaled to [0,1].
    Pure scipy/numpy (nearest-neighbour resize) so it is testable here;
    finetune.py converts to a 3-channel tensor."""
    f, t, sxx = sps.spectrogram(sig, fs=fs, nperseg=64, noverlap=48)
    img = np.log1p(sxx)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    ridx = (np.linspace(0, img.shape[0] - 1, out_hw[0])).astype(int)
    cidx = (np.linspace(0, img.shape[1] - 1, out_hw[1])).astype(int)
    return img[np.ix_(ridx, cidx)].astype(np.float32)
