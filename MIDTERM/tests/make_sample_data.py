"""
Generate data/sample_swings.csv: a fake Double Motor recording so the swing
tests and `accel_tuner.py --replay` work before you've recorded real swings.

    3 s at rest (|accel| = 1000, i.e. 1 g in milli-g, plus a little noise),
    then 3 swings with extra peaks of 800, 1500 and 2500, 1.5 s apart.

    .venv/bin/python tests/make_sample_data.py
"""

import csv
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'sample_swings.csv')

RATE_HZ = 50          # matches NOTIFICATION_DELAY_MS = 20
GRAVITY = 1000.0
SWING_PEAKS = [800, 1500, 2500]
SWING_TIMES = [3.5, 5.0, 6.5]
SWING_WIDTH = 0.04    # seconds (gaussian sigma)


def main():
    rng = random.Random(1)
    rows = []
    for i in range(int(8.0 * RATE_HZ)):
        t = i / RATE_HZ
        extra = sum(p * math.exp(-((t - st) ** 2) / (2 * SWING_WIDTH ** 2))
                    for p, st in zip(SWING_PEAKS, SWING_TIMES))
        # Gravity on z, swing acceleration mostly along x
        ax = extra + rng.gauss(0, 10)
        ay = rng.gauss(0, 10)
        az = GRAVITY + rng.gauss(0, 10)
        rows.append((round(t, 3), round(ax), round(ay), round(az)))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'ax', 'ay', 'az'])
        w.writerows(rows)
    print(f'Wrote {len(rows)} samples to {os.path.normpath(OUT)}')


if __name__ == '__main__':
    main()
