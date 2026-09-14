# Quick diagnostic: try camera indices 0-4 and report whether each gives a
# real (non-black) image, so you can find the right CAMERA_INDEX for Main.py.

import cv2

for index in range(5):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f'Index {index}: could not open')
        cap.release()
        continue

    success, frame = cap.read()
    if not success:
        print(f'Index {index}: opened but failed to read a frame')
    else:
        brightness = frame.mean()
        print(
            f'Index {index}: opened, resolution {frame.shape[1]}x{frame.shape[0]}, '
            f'mean brightness {brightness:.1f} '
            f'{"(looks BLACK)" if brightness < 1.0 else "(looks like a real image)"}'
        )
    cap.release()
