"""
Every tunable setting for the ping pong game lives here.

accel_tuner.py writes calibration.json next to this file; any keys in it
(ACCEL_GAIN, SWING_THRESHOLD) override the defaults below.
"""

import json
import os

import legoeducation as le

HERE = os.path.dirname(os.path.abspath(__file__))

# --- LEGO Double Motor (the paddle) ---
# Match the Connection Card on YOUR Double Motor (color + serial on the card).
CARD_COLOR = le.LEGO_COLOR_PURPLE
CARD_SERIAL = '6235'
# Milliseconds between IMU updates from the motor (library allows 15-1000).
# Lower = swings are sampled more finely.
NOTIFICATION_DELAY_MS = 20

# --- Accelerometer -> swing power ---
# power = |accel - gravity| * ACCEL_GAIN   (gravity is tracked automatically, see swing.py)
# The goal (set by accel_tuner.py) is that a normal swing gives power ~1.0.
ACCEL_GAIN = 1 / 1500      # guess assuming raw units are milli-g; run accel_tuner.py
SWING_THRESHOLD = 0.35     # power needed to count as a swing at all
SWING_MAX_SEC = 0.15       # a swing's peak is taken over at most this long
SWING_REFRACTORY_SEC = 0.35  # ignore new swings for this long after one

# Fake swing powers for --no-motor mode (keys: space / 1 / 2 / 3)
FAKE_SWING_POWER = 1.0
FAKE_SWING_KEYS = {ord('1'): 0.25, ord('2'): 1.0, ord('3'): 2.5}

# --- Camera / pose ---
CAMERA_INDEX = 1           # run list_cameras.py to find yours
HAND = 'right'             # which hand holds the motor: 'right' or 'left'
ARM_POINT = 'index'        # 'index' (end of the hand) or 'wrist'
MIN_VISIBILITY = 0.5       # landmarks less visible than this are ignored
POSE_MODEL_COMPLEXITY = 1  # 1 ships with mediapipe; 0 (faster) and 2 download a model on first use
PADDLE_SMOOTHING = 0.5     # 0-1, higher = more responsive / jittery
# Part of the camera image (normalized x0, y0, x1, y1) that maps onto the
# paddle zone. Shrink it if you can't reach the edges of the zone.
CAMERA_PADDLE_REGION = (0.1, 0.1, 0.9, 0.95)

# --- AprilTag cursor ---
TAG_ID = None              # None = any 36h11 tag works, or set e.g. 0
CAMERA_TAG_REGION = (0.15, 0.15, 0.85, 0.85)  # camera area that maps to the full screen
DWELL_SEC = 1.0            # hold the cursor on a button this long to click
CLICK_COOLDOWN_SEC = 1.5

# --- Window / layout (pixels) ---
WIDTH, HEIGHT = 1280, 720
TOP_BAR_H = 90
PREVIEW_W, PREVIEW_H = 320, 180
TABLE_LEFT, TABLE_RIGHT = 240, 1040
TABLE_Y = 520              # top surface of the table
NET_X = (TABLE_LEFT + TABLE_RIGHT) // 2
NET_H = 45
FLOOR_Y = HEIGHT
# Where the player's paddle can be (x0, y0, x1, y1)
PADDLE_ZONE = (40, 230, 360, 680)
PLAYER_HIT_X = 360         # ball must be left of this to be hittable
CPU_HIT_X = TABLE_RIGHT + 40

# --- Physics ---
GRAVITY = 1400.0           # px/s^2
RESTITUTION = 0.85         # bounce height kept after hitting the table
BALL_RADIUS = 10
SERVE_DELAY_SEC = 1.5      # pause before each serve
NET_CLEARANCE = 20         # shots are aimed to pass at least this far over the net

# --- Difficulty presets ---
# hit_radius:   how close (px) the paddle must be to the ball for a swing to hit
# cpu_flight:   seconds for the CPU's shot to reach your side (lower = faster)
# player_flight_slow/fast: your shot's flight time at min/max power
# min_power:    weaker swings than this hit the net
# max_power:    stronger swings than this fly off the end of the table
DIFFICULTIES = {
    'EASY':   dict(hit_radius=140, cpu_flight=1.35, player_flight_slow=1.3,
                   player_flight_fast=0.8, min_power=0.30, max_power=2.2),
    'MEDIUM': dict(hit_radius=100, cpu_flight=1.05, player_flight_slow=1.1,
                   player_flight_fast=0.65, min_power=0.45, max_power=1.7),
    'HARD':   dict(hit_radius=70,  cpu_flight=0.80, player_flight_slow=0.9,
                   player_flight_fast=0.5, min_power=0.60, max_power=1.4),
}
DIFFICULTY_ORDER = ['EASY', 'MEDIUM', 'HARD']
START_DIFFICULTY = 'MEDIUM'
START_SERVER = 'PLAYER'    # 'PLAYER' or 'CPU'

# --- MQTT / high score ---
MQTT_BROKER = 'test.mosquitto.org'
MQTT_PORT = 1883
MQTT_TOPIC = 'ME193/Rogers/Zack'
HIGHSCORE_FILE = os.path.join(HERE, 'highscore.json')
CALIBRATION_FILE = os.path.join(HERE, 'calibration.json')


def _load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        return {}
    with open(CALIBRATION_FILE) as f:
        data = json.load(f)
    allowed = {'ACCEL_GAIN', 'SWING_THRESHOLD'}
    return {k: v for k, v in data.items() if k in allowed}


globals().update(_load_calibration())
