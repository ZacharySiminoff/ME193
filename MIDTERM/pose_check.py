"""
Check pose tracking before playing: is the right hand found, and where does
it land in the game's paddle zone?

    python pose_check.py
    python pose_check.py --camera 0 --hand left

Left: camera with skeleton (red dot = arm end used as the paddle).
Right: the paddle zone from the game with where your paddle would be.
Keys: h switch hand | a switch index/wrist point | q quit
"""

import argparse
import time

import cv2
import numpy as np

import config as C
from sensors import Vision, map_to_rect

FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--camera', type=int, default=C.CAMERA_INDEX)
    ap.add_argument('--hand', choices=['right', 'left'], default=C.HAND)
    args = ap.parse_args()

    vision = Vision(args.camera, hand=args.hand, use_tags=False)
    zx0, zy0, zx1, zy1 = C.PADDLE_ZONE
    zone_w, zone_h = zx1 - zx0, zy1 - zy0
    trail = []
    last, fps, found_frames, frames = time.monotonic(), 0.0, 0, 0

    try:
        while True:
            res = vision.read()
            if res is None:
                print('No frame from camera.')
                break
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            frames += 1
            found_frames += res.arm_point is not None

            cam = cv2.resize(res.preview, (640, 360))
            # Draw the part of the camera that maps onto the paddle zone
            r0x, r0y, r1x, r1y = C.CAMERA_PADDLE_REGION
            cv2.rectangle(cam, (int(r0x * 640), int(r0y * 360)), (int(r1x * 640), int(r1y * 360)), (0, 255, 255), 1)

            zone = np.full((360, int(360 * zone_w / zone_h), 3), 40, np.uint8)
            if res.arm_point is not None:
                gx, gy = map_to_rect(res.arm_point, C.CAMERA_PADDLE_REGION, C.PADDLE_ZONE)
                zx = int((gx - zx0) / zone_w * zone.shape[1])
                zy = int((gy - zy0) / zone_h * zone.shape[0])
                trail = (trail + [(zx, zy)])[-30:]
            for (ax, ay), (bx, by) in zip(trail, trail[1:]):
                cv2.line(zone, (ax, ay), (bx, by), (60, 60, 160), 2)
            if trail and res.arm_point is not None:
                cv2.ellipse(zone, trail[-1], (8, 20), 0, 0, 360, (60, 60, 230), -1)
            # Table height inside the zone, for reference
            ty = int((C.TABLE_Y - zy0) / zone_h * zone.shape[0])
            cv2.line(zone, (0, ty), (zone.shape[1], ty), (90, 130, 40), 3)
            cv2.putText(zone, 'paddle zone', (5, 20), FONT, 0.5, (200, 200, 200), 1)

            img = cv2.copyMakeBorder(np.hstack([cam, zone]), 0, 80, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(25, 25, 25))
            p = res.arm_point
            lines = [
                f'{fps:4.1f} fps   hand: {vision.hand}   point: {vision.arm_point_kind}',
                f'pose: {"found" if res.pose_found else "NOT FOUND"}   arm end: '
                + (f'({p[0]:.2f}, {p[1]:.2f})' if p else 'not visible'),
                f'tracked {100 * found_frames / frames:.0f}% of frames   h hand | a point | q quit',
            ]
            for i, line in enumerate(lines):
                cv2.putText(img, line, (10, 382 + 22 * i), FONT, 0.55, (255, 255, 255), 1)
            cv2.imshow('Pose Check', img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('h'):
                vision.hand = 'left' if vision.hand == 'right' else 'right'
                trail = []
            elif key == ord('a'):
                vision.arm_point_kind = 'wrist' if vision.arm_point_kind == 'index' else 'index'
    finally:
        vision.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
