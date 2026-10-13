"""
Swing detector tests using the recorded-style data in data/sample_swings.csv
(regenerate with tests/make_sample_data.py).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from swing import SwingDetector, load_csv  # noqa: E402

SAMPLE = os.path.join(HERE, '..', 'data', 'sample_swings.csv')


def detect_all(samples, gain, threshold):
    det = SwingDetector(gain, threshold)
    return [p for t, ax, ay, az in samples if (p := det.feed(t, ax, ay, az)) is not None]


def test_gravity_estimate_settles_and_survives_swings():
    det = SwingDetector(gain=1 / 1500, threshold=0.35)
    for t, ax, ay, az in load_csv(SAMPLE):
        det.feed(t, ax, ay, az)
    gx, gy, gz = det.gravity   # file has gravity on z = 1000
    assert abs(gx) < 30 and abs(gy) < 30 and abs(gz - 1000) < 30


def test_finds_three_swings_with_right_power():
    samples = load_csv(SAMPLE)
    swings = detect_all(samples, gain=1 / 1500, threshold=0.35)
    assert len(swings) == 3
    # Extra peaks in the file are 800, 1500, 2500 -> powers ~0.53, 1.0, 1.67.
    # 50 Hz sampling can land slightly off the true peak, so allow some slack.
    for got, want in zip(swings, [800 / 1500, 1.0, 2500 / 1500]):
        assert abs(got - want) < 0.15 * want


def test_rest_noise_is_not_a_swing():
    samples = [s for s in load_csv(SAMPLE) if s[0] < 3.0]
    assert detect_all(samples, gain=1 / 1500, threshold=0.35) == []


def test_gain_scales_power():
    samples = load_csv(SAMPLE)
    low = detect_all(samples, gain=1 / 1500, threshold=0.35)
    high = detect_all(samples, gain=2 / 1500, threshold=0.70)  # same swings, doubled power
    assert len(low) == len(high) == 3
    for a, b in zip(low, high):
        assert abs(b - 2 * a) < 1e-9


def test_refractory_merges_a_double_peak():
    # Two spikes 0.1 s apart count as one swing.
    det = SwingDetector(gain=1 / 1000, threshold=0.3, refractory_sec=0.35)
    out = []
    for i, extra in enumerate([0, 0, 900, 0, 0, 0, 800, 0, 0]):
        p = det.feed(i * 0.02, 1000 + extra, 0, 0)
        if p is not None:
            out.append(p)
    assert len(out) == 1
