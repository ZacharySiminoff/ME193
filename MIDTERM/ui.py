"""
Everything drawn on screen (OpenCV), plus the on-screen buttons that the
AprilTag cursor (or mouse) clicks by hovering.
"""

import cv2
import numpy as np

import config as C

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
BG = (60, 35, 20)            # dark blue (BGR)
TABLE = (90, 130, 40)        # green
PADDLE = (60, 60, 230)       # red
CPU = (230, 160, 60)         # blue
YELLOW = (0, 220, 255)
CURSOR = (255, 0, 255)       # magenta, same as the tag outline in the preview


class Button:
    def __init__(self, rect, label_fn, action):
        self.rect = rect             # (x0, y0, x1, y1)
        self.label_fn = label_fn     # () -> text, so the label shows the current setting
        self.action = action

    def contains(self, p):
        x0, y0, x1, y1 = self.rect
        return p is not None and x0 <= p[0] <= x1 and y0 <= p[1] <= y1


class DwellClicker:
    """
    Click a button by holding the cursor on it for DWELL_SEC. After a click
    the cursor has to leave the button (and the cooldown pass) before it can
    click again, so a tag left in place doesn't keep toggling.
    """

    def __init__(self, buttons, dwell_sec=C.DWELL_SEC, cooldown_sec=C.CLICK_COOLDOWN_SEC):
        self.buttons = buttons
        self.dwell_sec = dwell_sec
        self.cooldown_sec = cooldown_sec
        self.hover = None
        self.hover_start = 0.0
        self.progress = 0.0          # 0..1 for the ring
        self._locked = None          # button just clicked; must leave it first
        self._cooldown_until = 0.0

    def update(self, cursor, now):
        """Returns the Button that got clicked this frame, or None."""
        over = next((b for b in self.buttons if b.contains(cursor)), None)
        if over is not self._locked:
            self._locked = None
        if over is None or over is self._locked or now < self._cooldown_until:
            self.hover, self.progress = None, 0.0
            return None
        if over is not self.hover:
            self.hover, self.hover_start = over, now
        self.progress = min((now - self.hover_start) / self.dwell_sec, 1.0)
        if self.progress >= 1.0:
            self._locked, self._cooldown_until = over, now + self.cooldown_sec
            self.hover, self.progress = None, 0.0
            over.action()
            return over
        return None


def make_buttons(game):
    return [
        Button((20, 15, 300, 75), lambda: f'SERVE: {"YOU" if game.server == "PLAYER" else "CPU"}',
               game.toggle_server),
        Button((320, 15, 620, 75), lambda: f'LEVEL: {game.difficulty}', game.cycle_difficulty),
    ]


def text_center(img, text, cx, y, scale, color, thick=2):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, (int(cx - tw / 2), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


def draw_game(game, clicker, cursor=None, preview=None, meter=0.0, show_debug=True, info=''):
    img = np.full((C.HEIGHT, C.WIDTH, 3), BG, np.uint8)
    s = game.settings

    # Table, legs, net
    cv2.rectangle(img, (C.TABLE_LEFT, C.TABLE_Y), (C.TABLE_RIGHT, C.TABLE_Y + 14), TABLE, -1)
    cv2.line(img, (C.TABLE_LEFT, C.TABLE_Y), (C.TABLE_RIGHT, C.TABLE_Y), WHITE, 2)
    for x in (C.TABLE_LEFT + 40, C.TABLE_RIGHT - 40):
        cv2.line(img, (x, C.TABLE_Y + 14), (x, C.HEIGHT), (40, 40, 40), 8)
    cv2.line(img, (C.NET_X, C.TABLE_Y - C.NET_H), (C.NET_X, C.TABLE_Y), WHITE, 4)

    # Paddle zone + paddle (+ hit radius when debugging)
    x0, y0, x1, y1 = C.PADDLE_ZONE
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 70, 50), 1)
    px, py = int(game.paddle[0]), int(game.paddle[1])
    if show_debug:
        cv2.circle(img, (px, py), s['hit_radius'], (80, 80, 160), 1)
    cv2.ellipse(img, (px, py), (14, 34), 0, 0, 360, PADDLE, -1)

    # CPU paddle tracks the ball's height when the ball is coming its way
    b = game.ball
    cpu_y = C.TABLE_Y - 60
    if b is not None and game.last_hitter == 'PLAYER':
        cpu_y = int(min(max(b.y, C.TABLE_Y - 250), C.TABLE_Y + 40))
    cv2.ellipse(img, (C.CPU_HIT_X + 15, cpu_y), (14, 34), 0, 0, 360, CPU, -1)

    if b is not None:
        cv2.circle(img, (int(b.x), int(b.y)), C.BALL_RADIUS, WHITE, -1, cv2.LINE_AA)
        if game._hittable():
            cv2.circle(img, (int(b.x), int(b.y)), C.BALL_RADIUS + 5, YELLOW, 2)

    # Top bar: buttons, score, high score
    cv2.rectangle(img, (0, 0), (C.WIDTH, C.TOP_BAR_H), (30, 20, 10), -1)
    for btn in clicker.buttons:
        bx0, by0, bx1, by1 = btn.rect
        hot = btn is clicker.hover
        cv2.rectangle(img, (bx0, by0), (bx1, by1), (90, 90, 90) if hot else (60, 60, 60), -1)
        cv2.rectangle(img, (bx0, by0), (bx1, by1), YELLOW if hot else GREY, 2)
        text_center(img, btn.label_fn(), (bx0 + bx1) / 2, (by0 + by1) / 2 + 10, 0.8, WHITE)
        if hot:   # dwell progress bar along the bottom of the button
            cv2.rectangle(img, (bx0, by1 - 6), (int(bx0 + (bx1 - bx0) * clicker.progress), by1), YELLOW, -1)
    text_center(img, f'SCORE {game.score}', 830, 62, 1.4, WHITE, 3)
    text_center(img, f'HIGH {game.high_score}', 1130, 62, 1.1, YELLOW, 2)

    # Message under the bar, fades after 2 s
    age = game.time - game.message_time
    if game.message and age < 2.0:
        fade = 1.0 - max(0.0, age - 1.0)
        text_center(img, game.message, C.WIDTH / 2, C.TOP_BAR_H + 60, 1.2,
                    tuple(int(c * fade) for c in WHITE), 3)

    draw_power_meter(img, game, meter)

    if preview is not None:
        small = cv2.resize(preview, (C.PREVIEW_W, C.PREVIEW_H))
        px0, py0 = C.WIDTH - C.PREVIEW_W - 10, C.HEIGHT - C.PREVIEW_H - 10
        img[py0:py0 + C.PREVIEW_H, px0:px0 + C.PREVIEW_W] = small
        cv2.rectangle(img, (px0, py0), (px0 + C.PREVIEW_W, py0 + C.PREVIEW_H), GREY, 1)

    if info:
        cv2.putText(img, info, (10, C.HEIGHT - 10), FONT, 0.5, GREY, 1, cv2.LINE_AA)

    if cursor is not None:
        cx, cy = int(cursor[0]), int(cursor[1])
        cv2.drawMarker(img, (cx, cy), CURSOR, cv2.MARKER_CROSS, 30, 3)
        if clicker.progress > 0:
            cv2.ellipse(img, (cx, cy), (26, 26), -90, 0, 360 * clicker.progress, YELLOW, 4)
    return img


def draw_power_meter(img, game, level):
    """Vertical bar: live swing power, with the threshold and min/max power marks."""
    s = game.settings
    x0, x1, top, bot = 10, 30, C.TOP_BAR_H + 100, C.TABLE_Y
    full = 2.5   # power at the top of the bar

    def y_of(p):
        return int(bot - (bot - top) * min(p, full) / full)

    cv2.rectangle(img, (x0, top), (x1, bot), (40, 40, 40), -1)
    cv2.rectangle(img, (x0, y_of(level)), (x1, bot), YELLOW, -1)
    for p, color in ((C.SWING_THRESHOLD, GREY), (s['min_power'], (0, 0, 255)), (s['max_power'], (0, 0, 255))):
        cv2.line(img, (x0 - 4, y_of(p)), (x1 + 4, y_of(p)), color, 2)
    if game.last_swing_power is not None:
        cv2.putText(img, f'{game.last_swing_power:.2f}', (x0 - 5, top - 10), FONT, 0.5, WHITE, 1)

