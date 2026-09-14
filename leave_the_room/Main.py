# Gesture-controlled tank-steering robot using the LEGO Education Python API
# and MediaPipe Pose.
#
# Control scheme (one arm = one wheel, like a pair of throttle levers):
#   - Raise your right arm above shoulder height  -> right motor drives forward
#   - Lower your right arm below shoulder height   -> right motor drives backward
#   - Same for your left arm and the left motor
#   - Both arms at shoulder height                 -> robot stops
#   - Arms raised/lowered by different amounts      -> robot turns
#
# Library docs: https://github.com/LEGO/LEGOEducation
# Install with: pip install legoeducation mediapipe opencv-python

import logging

import cv2
import legoeducation as le
import mediapipe as mp

# Surface legoeducation's scan/connect log messages (e.g. "Could not find
# device matching ...") instead of letting them stay silent.
logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- Robot connection settings ---
# Update these to match the Double Motor's Connection Card (color + serial
# number printed on the card). See connect.md in the LEGOEducation repo.
CARD_COLOR = le.LEGO_COLOR_MAGENTA
CARD_SERIAL = '0998'

# --- Camera settings ---
CAMERA_INDEX = 1

# --- Gesture tuning ---
# Normalized arm-raise distance (in torso lengths) needed to reach full speed.
ACTIVE_RANGE = 0.9
# Normalized arm-raise distance treated as "neutral" (stopped).
DEAD_ZONE = 0.15
# Exponential smoothing factor for motor speed (0-1). Higher = more responsive,
# lower = smoother but laggier.
SMOOTHING = 0.4
MAX_SPEED = 100

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def apply_deadzone(value: float, deadzone: float) -> float:
    """Rescale value so |value| < deadzone maps to 0 and the rest ramps smoothly to 1."""
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def arm_raise_ratio(shoulder, wrist, torso_length: float) -> float:
    """How far a wrist is above (+) or below (-) its shoulder, in torso lengths."""
    if torso_length < 1e-6:
        return 0.0
    return (shoulder.y - wrist.y) / torso_length


def speed_from_ratio(ratio: float) -> float:
    normalized = max(-1.0, min(1.0, ratio / ACTIVE_RANGE))
    normalized = apply_deadzone(normalized, DEAD_ZONE)
    return normalized * MAX_SPEED


def main():
    doublemotor = le.DoubleMotor()
    print('Scanning for Double Motor (up to 10s)...')
    doublemotor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
    if not doublemotor.connected:
        print(
            'Error connecting to Double Motor. Check that: the motor is '
            'charged, powered on, and broadcasting (blinking light); '
            'CARD_COLOR/CARD_SERIAL match ITS connection card (not another '
            "motor's); and Bluetooth permission is granted to this app/terminal "
            'in System Settings > Privacy & Security > Bluetooth.'
        )
        return

    print('Connected to Double Motor.')

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(
            'Error opening webcam. If a permission prompt did not appear, grant '
            'Camera access to this app/terminal in System Settings > Privacy & '
            'Security > Camera, then try again.'
        )
        doublemotor.disconnect()
        return

    print("Camera opened. Showing 'Gesture Control' window - move your arms.")

    smoothed_left = 0.0
    smoothed_right = 0.0
    frame_count = 0

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                success, frame = cap.read()
                if not success:
                    print('Error reading from webcam.')
                    break

                if frame_count == 30 and frame.mean() < 1.0:
                    print(
                        "WARNING: camera frames are completely black. This is "
                        "usually macOS Camera permission that was granted but "
                        "not yet applied - fully QUIT and reopen the app you're "
                        "running this from (Terminal/VS Code), not just re-run "
                        "the script. Also check no other app (Zoom, Photo Booth, "
                        "Camo, OBS) is holding the camera, and try CAMERA_INDEX "
                        "= 1 in case another virtual camera is device 0."
                    )

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = pose.process(rgb_frame)

                target_left = 0.0
                target_right = 0.0

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    lm = mp_pose.PoseLandmark

                    left_shoulder = landmarks[lm.LEFT_SHOULDER]
                    right_shoulder = landmarks[lm.RIGHT_SHOULDER]
                    left_wrist = landmarks[lm.LEFT_WRIST]
                    right_wrist = landmarks[lm.RIGHT_WRIST]
                    left_hip = landmarks[lm.LEFT_HIP]
                    right_hip = landmarks[lm.RIGHT_HIP]

                    shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
                    hip_mid_y = (left_hip.y + right_hip.y) / 2
                    torso_length = abs(hip_mid_y - shoulder_mid_y)

                    left_ratio = arm_raise_ratio(left_shoulder, left_wrist, torso_length)
                    right_ratio = arm_raise_ratio(right_shoulder, right_wrist, torso_length)

                    target_left = speed_from_ratio(left_ratio)
                    target_right = speed_from_ratio(right_ratio)

                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Smooth to avoid jittery motor commands from noisy landmarks.
                smoothed_left += (target_left - smoothed_left) * SMOOTHING
                smoothed_right += (target_right - smoothed_right) * SMOOTHING

                try:
                    doublemotor.movement_move_tank(
                        speed_left=int(smoothed_left),
                        speed_right=int(smoothed_right),
                        blocking=False,
                    )
                except Exception as exc:
                    print(f'Error sending motor command: {exc}')

                # Print status periodically so you can confirm gestures are
                # being detected and speeds are changing, even without
                # watching the numbers overlaid on the video window.
                frame_count += 1
                if frame_count % 15 == 0:
                    print(
                        f'pose detected: {bool(results.pose_landmarks)}  '
                        f'L: {int(smoothed_left):4d}%  R: {int(smoothed_right):4d}%  '
                        f'connected: {doublemotor.connected}'
                    )

                cv2.putText(
                    frame,
                    f'L: {int(smoothed_left):4d}%  R: {int(smoothed_right):4d}%',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Press 'q' to quit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                )
                cv2.imshow('Gesture Control', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        doublemotor.movement_stop()
        doublemotor.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        print('Disconnected.')


if __name__ == '__main__':
    main()
