"""
Dial in the accelerometer gain so a normal swing = power 1.0 in the game.

    python accel_tuner.py                               # live, with the motor
    python accel_tuner.py --replay data/sample_swings.csv   # no motor needed

Live:
  1. Hold the motor still for a second (gravity estimate settles).
  2. Make ~10 normal, comfortable swings. Each one shows up as a dot + number.
  3. Press s to save calibration.json (main.py loads it automatically).

Keys: s save | c clear swings | + / - raise / lower detection threshold | q quit
On quit, every raw sample is saved to data/swings_<time>.csv for --replay.
"""

import argparse
import json
import os
import statistics
import time

import cv2
import numpy as np

import config as C
from swing import SwingDetector, load_csv, magnitude, save_csv

W, H = 900, 500
HISTORY_SEC = 6.0


def suggest(peaks, threshold_frac):
    med = statistics.median(peaks)
    return {'ACCEL_GAIN': 1.0 / med, 'SWING_THRESHOLD': threshold_frac,
            'median_raw_peak': med, 'raw_peaks': [round(p, 1) for p in peaks]}


def save(cal):
    with open(C.CALIBRATION_FILE, 'w') as f:
        json.dump(cal, f, indent=2)
    print(f'Saved {C.CALIBRATION_FILE}: ACCEL_GAIN={cal["ACCEL_GAIN"]:.6f} '
          f'SWING_THRESHOLD={cal["SWING_THRESHOLD"]}')


def print_summary(peaks, threshold_frac):
    if not peaks:
        print('No swings detected. Try a lower --raw-threshold.')
        return None
    cal = suggest(peaks, threshold_frac)
    print(f'{len(peaks)} swings, raw peaks: {cal["raw_peaks"]}')
    print(f'median raw peak = {cal["median_raw_peak"]:.1f}')
    print(f'suggested ACCEL_GAIN = 1/{cal["median_raw_peak"]:.0f} = {cal["ACCEL_GAIN"]:.6f}')
    print('with that gain your swings would be powers: '
          + ', '.join(f'{p * cal["ACCEL_GAIN"]:.2f}' for p in peaks))
    return cal


def replay(path, raw_threshold, threshold_frac, do_save):
    det = SwingDetector(1.0, raw_threshold, C.SWING_MAX_SEC, C.SWING_REFRACTORY_SEC)
    peaks = []
    for t, ax, ay, az in load_csv(path):
        p = det.feed(t, ax, ay, az)
        if p is not None:
            peaks.append(p)
            print(f'  swing at t={t:.2f}s  raw peak {p:.1f}')
    g = magnitude(*det.gravity)
    print(f'gravity estimate |g| = {g:.1f} raw units (so 1 g = {g:.0f})')
    cal = print_summary(peaks, threshold_frac)
    if cal and do_save:
        save(cal)


def live(raw_threshold, threshold_frac):
    from sensors import Paddle   # only needs the motor in live mode

    paddle = Paddle(gain=1.0, threshold=raw_threshold)   # gain 1 -> powers are raw units
    paddle.record = True
    det = paddle.detector
    history = []          # (t, raw_level)
    peaks = []            # (t, raw_peak)
    t0 = time.monotonic()
    msg = ''

    try:
        while True:
            now = time.monotonic() - t0
            history.append((now, det.raw_level))
            history = [h for h in history if now - h[0] < HISTORY_SEC]
            swing = paddle.get_swing()
            if swing is not None:
                peaks.append((now, swing))
                print(f'swing #{len(peaks)}: raw peak {swing:.1f}')

            img = np.full((H, W, 3), 25, np.uint8)
            top_val = max([det.threshold * 3] + [v for _, v in history] + [p for _, p in peaks[-10:]])

            def pt(t, v):
                return (int(W - (now - t) / HISTORY_SEC * W), int(H - 40 - (H - 140) * v / top_val))

            for (ta, va), (tb, vb) in zip(history, history[1:]):
                cv2.line(img, pt(ta, va), pt(tb, vb), (0, 220, 255), 2)
            ty = pt(now, det.threshold)[1]
            cv2.line(img, (0, ty), (W, ty), (120, 120, 120), 1)
            cv2.putText(img, f'detect threshold {det.threshold:.0f}', (10, ty - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
            for t, p in peaks:
                if now - t < HISTORY_SEC:
                    x, y = pt(t, p)
                    cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
                    cv2.putText(img, f'{p:.0f}', (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            imu = paddle.latest()
            g = magnitude(*det.gravity) if det.gravity else float('nan')
            lines = [
                f'accel x/y/z: {imu.accelerometerX} {imu.accelerometerY} {imu.accelerometerZ}   '
                f'gyro: {imu.gyroscopeX} {imu.gyroscopeY} {imu.gyroscopeZ}',
                f'gravity |g| = {g:.0f} raw units   live |a - g| = {det.raw_level:.0f}',
                f'swings: {len(peaks)}',
            ]
            if peaks:
                cal = suggest([p for _, p in peaks], threshold_frac)
                lines[2] += (f'   median peak {cal["median_raw_peak"]:.0f}   -> ACCEL_GAIN '
                             f'{cal["ACCEL_GAIN"]:.6f}   (current config: {C.ACCEL_GAIN:.6f})')
            lines.append('s save | c clear | +/- threshold | q quit   ' + msg)
            for i, line in enumerate(lines):
                cv2.putText(img, line, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow('Accel Tuner', img)

            key = cv2.waitKey(15) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('c'):
                peaks, msg = [], 'cleared'
            elif key in (ord('+'), ord('=')):
                det.threshold *= 1.2
            elif key in (ord('-'), ord('_')):
                det.threshold /= 1.2
            elif key == ord('s'):
                if len(peaks) < 3:
                    msg = 'need at least 3 swings to save'
                else:
                    save(suggest([p for _, p in peaks], threshold_frac))
                    msg = 'SAVED calibration.json'
    finally:
        paddle.close()
        cv2.destroyAllWindows()
        if paddle.samples:
            os.makedirs(os.path.join(C.HERE, 'data'), exist_ok=True)
            path = os.path.join(C.HERE, 'data', time.strftime('swings_%Y%m%d_%H%M%S.csv'))
            t_start = paddle.samples[0][0]
            save_csv(path, [(round(t - t_start, 4), ax, ay, az) for t, ax, ay, az in paddle.samples])
            print(f'Raw samples saved to {path}  (replay with --replay)')
        print_summary([p for _, p in peaks], threshold_frac)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--replay', metavar='CSV', help='analyze a recording instead of the live motor')
    ap.add_argument('--save', action='store_true', help='with --replay: write calibration.json')
    ap.add_argument('--raw-threshold', type=float, default=C.SWING_THRESHOLD / C.ACCEL_GAIN,
                    help='raw |a - g| that starts a swing (default: current config threshold)')
    ap.add_argument('--threshold-frac', type=float, default=0.35,
                    help='SWING_THRESHOLD to save, as a fraction of a normal swing (default 0.35)')
    args = ap.parse_args()

    if args.replay:
        replay(args.replay, args.raw_threshold, args.threshold_frac, args.save)
    else:
        live(args.raw_threshold, args.threshold_frac)


if __name__ == '__main__':
    main()
