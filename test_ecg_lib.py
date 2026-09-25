"""
Hermetic tests for ecg_lib — no downloads, no torch, no PTB-XL required.
Synthetic signals with known properties verify the machinery before it ever
touches real data.

Run:  python -m pytest test_ecg_lib.py -q
"""
import numpy as np
import pandas as pd
from scipy import signal as sps
import pytest

from ecg_lib import (FS, butter_bandpass, znormalise,
                     heart_rate_features, classical_feature_vector,
                     single_superclass_labels, patientwise_fold_split,
                     assert_patient_disjoint, spectrogram_image)


def synthetic_ecg(bpm=60, seconds=10, fs=FS):
    """Impulse train at the given rate convolved with a narrow Gaussian —
    a crude but sufficient stand-in for QRS complexes at a KNOWN heart rate."""
    n = seconds * fs
    x = np.zeros(n)
    step = int(fs * 60.0 / bpm)
    x[step // 2::step] = 1.0
    kernel = sps.windows.gaussian(int(0.08 * fs), std=2)
    return np.convolve(x, kernel, mode="same")


def test_bandpass_attenuates_drift_and_mains_but_passes_qrs_band():
    b, a = butter_bandpass()
    w, h = sps.freqz(b, a, fs=FS)
    gain = np.abs(h)
    def g_at(freq):
        return gain[np.argmin(np.abs(w - freq))]
    assert g_at(0.05) < 0.1      # baseline wander crushed
    assert g_at(10.0) > 0.9      # QRS band preserved
    assert g_at(49.0) < 0.35     # mains-adjacent strongly attenuated


def test_r_peak_detection_recovers_known_rate():
    sig = synthetic_ecg(bpm=72)
    feats = heart_rate_features(sig)
    assert abs(feats["hr_mean"] - 72) < 3          # within 3 bpm
    assert feats["n_beats"] >= 10


def test_znormalise_zero_mean_unit_var():
    rng = np.random.default_rng(0)
    x = rng.normal(5, 3, size=(1000, 12))
    z = znormalise(x)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(z.std(axis=0), 1, atol=1e-3)


def test_feature_vector_is_finite_and_fixed_length():
    rec = znormalise(np.column_stack([synthetic_ecg(70) for _ in range(12)]))
    v1 = classical_feature_vector(rec)
    v2 = classical_feature_vector(rec)
    assert v1.shape == v2.shape == (57,)
    assert np.all(np.isfinite(v1))
    assert np.allclose(v1, v2)                      # deterministic


def test_single_superclass_filter_keeps_only_unambiguous_records():
    agg = pd.DataFrame({"diagnostic": [1, 1, 1],
                        "diagnostic_class": ["NORM", "MI", "STTC"]},
                       index=["NORM", "IMI", "NST_"])
    df = pd.DataFrame({
        "scp_codes": [{"NORM": 100}, {"IMI": 80}, {"IMI": 50, "NST_": 50},
                      {"XYZ": 100}],
        "patient_id": [1, 2, 3, 4], "strat_fold": [1, 9, 10, 5]})
    out = single_superclass_labels(df, agg)
    assert list(out.superclass) == ["NORM", "MI"]   # multi-class + unknown dropped


def test_patient_disjoint_guard_fires_on_leakage():
    train = pd.DataFrame({"patient_id": [1, 2, 3]})
    test = pd.DataFrame({"patient_id": [3, 4]})
    with pytest.raises(ValueError, match="PATIENT LEAKAGE"):
        assert_patient_disjoint(train, test)


def test_recommended_folds_are_patient_disjoint_on_toy_data():
    df = pd.DataFrame({"patient_id": [10, 11, 12, 13, 14, 15],
                       "strat_fold": [1, 5, 8, 9, 10, 10]})
    tr, va, te = patientwise_fold_split(df)
    assert set(tr.patient_id) == {10, 11, 12}
    assert set(va.patient_id) == {13}
    assert set(te.patient_id) == {14, 15}
    assert assert_patient_disjoint(tr, va)
    assert assert_patient_disjoint(tr, te)
    assert assert_patient_disjoint(va, te)


def test_spectrogram_image_shape_range_determinism():
    sig = synthetic_ecg(65)
    img = spectrogram_image(sig)
    assert img.shape == (224, 224)
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert np.allclose(img, spectrogram_image(sig))
