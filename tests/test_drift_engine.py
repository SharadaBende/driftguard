import numpy as np
import pandas as pd

from src.drift_engine import (
    cramers_v,
    psi_categorical,
    psi_numeric,
)


def test_psi_numeric_identical_is_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 50_000)
    assert psi_numeric(x, x) < 0.001


def test_psi_numeric_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 50_000)
    cur = rng.normal(1.5, 1, 50_000)
    assert psi_numeric(ref, cur) > 0.25


def test_psi_numeric_small_noise_is_stable():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 50_000)
    cur = rng.normal(0, 1, 50_000)
    assert psi_numeric(ref, cur) < 0.1


def test_psi_categorical_detects_shift():
    ref = pd.Series([1] * 8000 + [2] * 2000)
    cur = pd.Series([1] * 2000 + [2] * 8000)
    assert psi_categorical(ref, cur) > 0.25


def test_psi_categorical_handles_new_category():
    ref = pd.Series([1] * 1000)
    cur = pd.Series([1] * 500 + [9] * 500)
    assert psi_categorical(ref, cur) > 0.25


def test_cramers_v_identical_is_zero():
    s = pd.Series([1, 2, 3] * 1000)
    assert cramers_v(s, s) < 0.01