"""
Turn raw accelerometer samples into swing events. Pure math, no hardware,
so it can be tested with recorded CSVs (see tests/test_swing.py).

Gravity is tracked as a slowly-moving average of the accel vector. A swing
is fast, so it shows up as the difference between the current reading and
that average:  power = |accel - gravity| * gain.  This works no matter which
way the motor is held or swung.
"""

import csv
import math


def magnitude(ax, ay, az):
    return math.sqrt(ax * ax + ay * ay + az * az)


def load_csv(path):
    """Read a recording made by accel_tuner.py: rows of t, ax, ay, az."""
    with open(path) as f:
        return [(float(r['t']), float(r['ax']), float(r['ay']), float(r['az']))
                for r in csv.DictReader(f)]


def save_csv(path, samples):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'ax', 'ay', 'az'])
        w.writerows(samples)


class SwingDetector:
    """
    Feed accelerometer samples in; get one power value back per swing.

    A swing starts when power rises above `threshold`. Its power is the peak
    seen until it drops back below half the threshold (or `max_sec` passes).
    After that, new swings are ignored for `refractory_sec`.
    """

    def __init__(self, gain, threshold, max_sec=0.15, refractory_sec=0.35, gravity_tau=0.5):
        self.gain = gain
        self.threshold = threshold
        self.max_sec = max_sec
        self.refractory_sec = refractory_sec
        self.gravity_tau = gravity_tau  # seconds; how fast the gravity estimate follows
        self.gravity = None             # (gx, gy, gz) estimate
        self.level = 0.0                # latest power, for the on-screen meter
        self.raw_level = 0.0            # latest |accel - gravity| before gain
        self._last_t = None
        self._start = None              # time the current swing started, or None
        self._peak = 0.0
        self._quiet_until = -math.inf

    def feed(self, t, ax, ay, az):
        """Returns the swing's power when a swing finishes, otherwise None."""
        if self.gravity is None:
            self.gravity = (ax, ay, az)
            self._last_t = t
            return None

        gx, gy, gz = self.gravity
        self.raw_level = magnitude(ax - gx, ay - gy, az - gz)
        self.level = self.raw_level * self.gain

        # Only let gravity follow the reading when not mid-swing, so the
        # swing itself doesn't get averaged into "gravity".
        dt = max(t - self._last_t, 0.0)
        self._last_t = t
        if self._start is None and self.level < self.threshold:
            a = dt / (self.gravity_tau + dt) if dt > 0 else 0.0
            self.gravity = (gx + a * (ax - gx), gy + a * (ay - gy), gz + a * (az - gz))

        if self._start is None:
            if self.level >= self.threshold and t >= self._quiet_until:
                self._start = t
                self._peak = self.level
            return None

        self._peak = max(self._peak, self.level)
        if self.level < self.threshold * 0.5 or t - self._start >= self.max_sec:
            self._start = None
            self._quiet_until = t + self.refractory_sec
            return self._peak
        return None
