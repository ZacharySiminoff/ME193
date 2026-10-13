"""
Hardware wrappers: camera + MediaPipe pose + AprilTags (Vision), and the
LEGO Double Motor used as a paddle (Paddle). FakePaddle stands in for the
motor when testing with the keyboard.
"""

import logging
import math
import threading
import time

import cv2
import mediapipe as mp

import config as C
from swing import SwingDetector

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# MediaPipe Pose landmark indices: (wrist, index finger) for each hand.
# These are the PERSON's left/right, not the image's.
ARM_LANDMARKS = {
    'right': (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX),
    'left': (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX),
}


def _try_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        # 720p is plenty for pose + tags and much faster than 1080p
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, _ = cap.read()
        if ok:
            return cap
    cap.release()
    return None


def open_camera(index=C.CAMERA_INDEX):
    """Open `index`; if that fails, try the other indices before giving up."""
    cap = _try_camera(index)
    if cap is not None:
        return cap
    for other in range(4):
        if other == index:
            continue
        cap = _try_camera(other)
        if cap is not None:
            print(f'WARNING: camera {index} did not work, using camera {other} instead. '
                  f'Set CAMERA_INDEX = {other} in config.py to skip this.')
            return cap
    raise RuntimeError(
        f'Could not open any camera (tried {index}, then 0-3). Check that no other app '
        '(Zoom, Photo Booth, ...) is using it and that System Settings > Privacy & '
        'Security > Camera allows this terminal / VS Code.')


def map_to_rect(point, region, rect):
    """Map a normalized camera point inside `region` onto the pixel `rect`."""
    nx, ny = point
    r0x, r0y, r1x, r1y = region
    x0, y0, x1, y1 = rect
    fx = min(max((nx - r0x) / (r1x - r0x), 0.0), 1.0)
    fy = min(max((ny - r0y) / (r1y - r0y), 0.0), 1.0)
    return (x0 + fx * (x1 - x0), y0 + fy * (y1 - y0))


class VisionResult:
    def __init__(self):
        self.preview = None        # mirrored camera image with overlays (BGR)
        self.arm_point = None      # (x, y) normalized, mirrored, or None
        self.tag_point = None      # (x, y) normalized, mirrored, or None
        self.tag_id = None
        self.pose_found = False


class Vision:
    """
    One camera feeding both the pose tracker and the AprilTag detector.

    Detection runs on the raw (un-mirrored) frame so MediaPipe's left/right
    labels are correct; the preview and returned points are mirrored so the
    screen behaves like a mirror for the player.
    """

    def __init__(self, camera_index=C.CAMERA_INDEX, hand=C.HAND, arm_point=C.ARM_POINT,
                 use_pose=True, use_tags=True):
        self.cap = open_camera(camera_index)
        self.hand = hand
        self.arm_point_kind = arm_point
        self.pose = mp_pose.Pose(model_complexity=C.POSE_MODEL_COMPLEXITY,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5) if use_pose else None
        self.tags = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
            cv2.aruco.DetectorParameters()) if use_tags else None
        self.frames = 0

    def close(self):
        self.cap.release()
        if self.pose:
            self.pose.close()

    def read(self):
        """Grab a frame and run detection. Returns a VisionResult, or None if no frame."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.frames += 1
        if self.frames == 30 and frame.mean() < 1.0:
            print('WARNING: camera frames are black. Usually camera permission was just '
                  'granted: fully quit and reopen Terminal / VS Code. Also try another '
                  'CAMERA_INDEX (list_cameras.py).')

        res = VisionResult()
        w = frame.shape[1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # before any overlay is drawn

        if self.pose:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            out = self.pose.process(rgb)
            if out.pose_landmarks:
                res.pose_found = True
                mp_drawing.draw_landmarks(frame, out.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                p = self._arm_end(out.pose_landmarks.landmark)
                if p is not None:
                    res.arm_point = (1.0 - p[0], p[1])   # mirror
                    cv2.circle(frame, (int(p[0] * w), int(p[1] * frame.shape[0])), 12, (0, 0, 255), -1)

        if self.tags:
            corners, ids, _ = self.tags.detectMarkers(gray)
            if ids is not None:
                for c, tag_id in zip(corners, ids.flatten()):
                    if C.TAG_ID is not None and tag_id != C.TAG_ID:
                        continue
                    cx, cy = c[0].mean(axis=0)
                    cv2.polylines(frame, [c.astype(int)], True, (255, 0, 255), 3)
                    if res.tag_point is None:   # first matching tag is the cursor
                        res.tag_point = (1.0 - float(cx) / w, float(cy) / frame.shape[0])
                        res.tag_id = int(tag_id)

        res.preview = cv2.flip(frame, 1)
        if res.tag_point is not None:
            # Tag label drawn after the flip so the text isn't mirrored
            tx, ty = int(res.tag_point[0] * w), int(res.tag_point[1] * frame.shape[0])
            cv2.putText(res.preview, f'tag {res.tag_id}', (tx - 30, ty - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
        return res

    def _arm_end(self, landmarks):
        """End of the chosen arm (normalized, un-mirrored), or None if not visible."""
        wrist_i, index_i = ARM_LANDMARKS[self.hand]
        order = [index_i, wrist_i] if self.arm_point_kind == 'index' else [wrist_i]
        for i in order:
            lm = landmarks[i]
            if lm.visibility >= C.MIN_VISIBILITY:
                return (lm.x, lm.y)
        return None


class Paddle:
    """
    The LEGO Double Motor held as a paddle. A background thread watches the
    IMU and runs the SwingDetector; the game calls get_swing() each frame.
    """

    def __init__(self, gain=C.ACCEL_GAIN, threshold=C.SWING_THRESHOLD):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        import legoeducation as le
        self.robot = le.DoubleMotor()
        print('Scanning for Double Motor (up to 10 s)...')
        self.robot.connect(card_color=C.CARD_COLOR, card_serial=C.CARD_SERIAL,
                           device_notification_delay=C.NOTIFICATION_DELAY_MS)
        if not self.robot.connected:
            raise RuntimeError(
                'Could not connect to the Double Motor. Check it is charged, on and '
                'blinking, CARD_COLOR / CARD_SERIAL in config.py match its card, and '
                'Bluetooth permission is on for this terminal. Or run with --no-motor.')
        print('Connected to Double Motor.')

        self.detector = SwingDetector(gain, threshold, C.SWING_MAX_SEC, C.SWING_REFRACTORY_SEC)
        self.samples = []            # (t, ax, ay, az) -- used by accel_tuner.py
        self.record = False
        self._swings = []
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    @property
    def level(self):
        return self.detector.level

    def get_swing(self):
        """Power of the strongest swing since the last call, or None."""
        with self._lock:
            swings, self._swings = self._swings, []
        return max(swings) if swings else None

    def latest(self):
        """Most recent raw IMU notification object."""
        return self.robot.imu_device

    def _poll(self):
        last = None
        while self._running:
            imu = self.robot.imu_device   # the library swaps in a new object per update
            if imu is not last:
                last = imu
                ax, ay, az = imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ
                if not any(math.isnan(v) for v in (ax, ay, az)):
                    t = time.monotonic()
                    if self.record:
                        self.samples.append((round(t, 4), ax, ay, az))
                    power = self.detector.feed(t, ax, ay, az)
                    if power is not None:
                        with self._lock:
                            self._swings.append(power)
            time.sleep(0.003)

    def close(self):
        self._running = False
        self._thread.join(timeout=1)
        try:
            self.robot.disconnect()
        except Exception as exc:
            print(f'Error disconnecting motor: {exc}')


class FakePaddle:
    """Keyboard stand-in for the motor: main.py calls trigger(power) on a key press."""

    def __init__(self):
        self._swing = None
        self._level = 0.0
        self._level_t = 0.0

    @property
    def level(self):
        # Meter decays after a fake swing so it looks like the real thing
        return self._level * max(0.0, 1 - (time.monotonic() - self._level_t) / 0.4)

    def trigger(self, power):
        self._swing = power
        self._level, self._level_t = power, time.monotonic()

    def get_swing(self):
        s, self._swing = self._swing, None
        return s

    def close(self):
        pass
