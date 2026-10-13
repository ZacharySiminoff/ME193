"""
Check the AprilTag cursor before playing.

    python tag_check.py               # live: tag moves a cursor over the game's buttons
    python tag_check.py --make-tag 0  # save a printable tag36h11 #0 as tag36h11_0.png

Hold the tag up to the camera; the magenta cross follows it. Hold it on a
button for DWELL_SEC to "click" (prints in the terminal). Uses the same
buttons, mapping and dwell logic as the game.
Keys: q quit
"""

import argparse
import os
import time

import cv2
import numpy as np

import config as C
from game import Game
from sensors import Vision, map_to_rect
from ui import DwellClicker, Renderer, make_buttons


def make_tag(tag_id):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(d, tag_id, 600)
    tag = cv2.copyMakeBorder(tag, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)  # white margin
    path = os.path.join(C.HERE, f'tag36h11_{tag_id}.png')
    cv2.imwrite(path, tag)
    print(f'Saved {path}. Print it (about 10 cm wide or bigger works well).')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--camera', type=int, default=C.CAMERA_INDEX)
    ap.add_argument('--make-tag', type=int, metavar='ID', help='save a printable tag image and exit')
    args = ap.parse_args()

    if args.make_tag is not None:
        make_tag(args.make_tag)
        return

    game = Game()                     # only used for button labels / actions
    clicker = DwellClicker(make_buttons(game))
    renderer = Renderer()
    vision = Vision(args.camera, use_pose=False)
    seen = 0
    frames = 0

    try:
        while True:
            res = vision.read()
            if res is None:
                print('No frame from camera.')
                break
            frames += 1
            cursor = None
            if res.tag_point is not None:
                seen += 1
                cursor = map_to_rect(res.tag_point, C.CAMERA_TAG_REGION, (0, 0, C.WIDTH, C.HEIGHT))
            clicked = clicker.update(cursor, time.monotonic())
            if clicked:
                print(f'CLICK -> {clicked.label_fn()}')

            img = renderer.draw(game, clicker, cursor, res.preview, show_debug=False,
                                info=f'tag: {res.tag_id if res.tag_id is not None else "none"}   '
                                     f'seen {100 * seen / frames:.0f}% of frames   '
                                     f'cursor: {tuple(int(v) for v in cursor) if cursor else "-"}   q quit')
            # Bigger camera view than the game uses, so the tag is easy to see
            cam = cv2.resize(res.preview, (480, 270))
            img[C.TOP_BAR_H + 10:C.TOP_BAR_H + 280, C.WIDTH - 490:C.WIDTH - 10] = cam
            cv2.imshow('Tag Check', img)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        vision.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
