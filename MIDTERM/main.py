"""
LEGO Motor Ping Pong.

3D view from behind you. Hold the Double Motor like a paddle. The camera
tracks the end of your arm (MediaPipe Pose) to move the paddle left/right
and up/down; swinging the motor (accelerometer) hits the ball, harder swings
hit deeper, and hitting off-center aims left/right. Hold an AprilTag over a button
at the top to click it.

    python main.py                        # full game (camera + motor)
    python main.py --no-motor             # space / 1 / 2 / 3 = fake swings
    python main.py --no-camera            # mouse moves the paddle
    python main.py --no-motor --no-camera # no hardware at all

Keys: q quit | s serve | d difficulty | l left/right hand | h hide/show debug | r r reset high score
"""

import argparse
import json
import os
import time

import cv2

import config as C
from game import Game, paddle_from_camera
from mqttlib import MQTTClient
from sensors import FakePaddle, Paddle, Vision, map_to_rect
from ui import DwellClicker, Renderer, make_buttons

WINDOW = 'LEGO Ping Pong'


def load_high_score():
    if not os.path.exists(C.HIGHSCORE_FILE):
        return 0
    with open(C.HIGHSCORE_FILE) as f:
        return int(json.load(f).get('high_score', 0))


class Scoreboard:
    """Saves the high score to disk and publishes it as a float over MQTT."""

    def __init__(self):
        self.client = None
        try:
            self.client = MQTTClient(C.MQTT_BROKER, C.MQTT_PORT)
            self.client.connect()
            print(f'MQTT connected to {C.MQTT_BROKER}, publishing high score to {C.MQTT_TOPIC}')
        except Exception as exc:
            print(f'WARNING: MQTT unavailable ({exc}); high score will only be saved locally.')
            self.client = None

    def new_high(self, high):
        with open(C.HIGHSCORE_FILE, 'w') as f:
            json.dump({'high_score': high}, f)
        self.publish(high)

    def publish(self, high):
        if self.client:
            # retain=True so anyone who subscribes later still gets the latest value
            self.client.publish(C.MQTT_TOPIC, f'{float(high):.1f}', retain=True)

    def close(self):
        if self.client:
            time.sleep(0.3)   # let the last publish go out
            self.client.disconnect()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-motor', action='store_true', help='keyboard swings instead of the LEGO motor')
    ap.add_argument('--no-camera', action='store_true', help='mouse moves the paddle instead of pose')
    ap.add_argument('--camera', type=int, default=C.CAMERA_INDEX)
    ap.add_argument('--hand', choices=['right', 'left'], default=C.HAND)
    args = ap.parse_args()

    scoreboard = Scoreboard()
    game = Game(high_score=load_high_score(), on_new_high=scoreboard.new_high, hand=args.hand.upper())
    scoreboard.publish(game.high_score)

    vision = None
    if not args.no_camera:
        try:
            vision = Vision(args.camera, hand=args.hand)
        except RuntimeError as exc:
            print(f'WARNING: {exc}\nFalling back to mouse control.')

    paddle = FakePaddle()
    if not args.no_motor:
        try:
            paddle = Paddle()
        except Exception as exc:
            print(f'WARNING: {exc}\nFalling back to keyboard swings (space / 1 / 2 / 3).')
            paddle = FakePaddle()
    fake = isinstance(paddle, FakePaddle)

    buttons = make_buttons(game)
    clicker = DwellClicker(buttons)
    renderer = Renderer()
    mouse = {'pos': None}

    def on_mouse(event, x, y, flags, param):
        mouse['pos'] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            for b in buttons:
                if b.contains((x, y)):
                    b.action()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    smoothed = None
    last_seen = time.monotonic()      # last time the tracked hand was found
    show_debug = True
    reset_armed_until = 0.0
    last = time.monotonic()
    fps = 0.0

    try:
        while True:
            now = time.monotonic()
            dt = min(now - last, 0.1)   # cap so a hiccup doesn't teleport the ball
            last = now
            fps = 0.9 * fps + 0.1 * (1 / dt if dt > 0 else 0)

            # --- inputs ---
            # The camera tracks whichever hand is picked (HAND button / 'l').
            if vision and vision.hand != game.hand.lower():
                vision.hand = game.hand.lower()
                smoothed = None

            paddle_xy, cursor, preview, pose_found = None, None, None, False
            if vision:
                res = vision.read()
                if res is not None:
                    preview, pose_found = res.preview, res.pose_found
                    if res.arm_point is not None:
                        target = paddle_from_camera(res.arm_point)
                        a = C.PADDLE_SMOOTHING
                        smoothed = target if smoothed is None else (
                            smoothed[0] + a * (target[0] - smoothed[0]),
                            smoothed[1] + a * (target[1] - smoothed[1]))
                        paddle_xy = smoothed
                        last_seen = now
                    if res.tag_point is not None:
                        cursor = map_to_rect(res.tag_point, C.CAMERA_TAG_REGION, (0, 0, C.WIDTH, C.HEIGHT))
            elif mouse['pos'] is not None:
                # The paddle goes right under the mouse pointer
                paddle_xy = renderer.view.unproject(*mouse['pos'], C.PLAYER_Z)
            clicker.update(cursor, now)
            game.update(dt, paddle_xy, paddle.get_swing())

            # --- draw ---
            status = [f'{fps:4.1f} fps',
                      'motor: KEYBOARD' if fake else 'motor: ok',
                      ('pose: ok' if pose_found else 'pose: NOT FOUND') if vision else 'paddle: MOUSE',
                      'mqtt: ok' if scoreboard.client else 'mqtt: off']
            if time.monotonic() < reset_armed_until:
                status.append('press r again to reset high score')
            # Big on-screen warning when tracking isn't working, so it's not only in the terminal
            alert = ''
            if not vision and not args.no_camera:
                alert = 'NO CAMERA - using the mouse (see terminal)'
            elif preview is not None and preview[::16, ::16].mean() < 1.0:
                alert = 'CAMERA IS BLACK - quit Photo Booth / FaceTime / Zoom, or check camera permission'
            elif vision and now - last_seen > 1.0:
                alert = (f"CAN'T SEE YOUR {game.hand} HAND - keep it in view of the camera" if pose_found
                         else "CAN'T SEE YOU - step back so your upper body is in view")
            img = renderer.draw(game, clicker, cursor, preview, meter=paddle.level,
                                show_debug=show_debug, info='  |  '.join(status), alert=alert)
            cv2.imshow(WINDOW, img)

            # --- keys ---
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                game.toggle_server()
            elif key == ord('d'):
                game.cycle_difficulty()
            elif key == ord('l'):
                game.toggle_hand()
            elif key == ord('h'):
                show_debug = not show_debug
            elif key == ord('r'):
                if time.monotonic() < reset_armed_until:
                    game.reset_high_score()
                    reset_armed_until = 0.0
                else:
                    reset_armed_until = time.monotonic() + 2.0
            elif fake and key == ord(' '):
                paddle.trigger(C.FAKE_SWING_POWER)
            elif fake and key in C.FAKE_SWING_KEYS:
                paddle.trigger(C.FAKE_SWING_KEYS[key])
    finally:
        paddle.close()
        if vision:
            vision.close()
        scoreboard.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
