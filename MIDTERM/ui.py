"""
Everything drawn on screen (OpenCV), plus the on-screen buttons that the
AprilTag cursor (or mouse) clicks by hovering.

Renderer draws one frame per call:
  cached arena background (per side) -> ball / paddles / effects -> glow
  -> screen shake -> HUD (top bar, score, meter, camera preview, cursor)

The game runs in "player on the left" coordinates; everything here converts
x with mirror_x() so the player can be on either side of the screen.
"""

import math
import random
from collections import deque

import cv2
import numpy as np

import config as C
from game import WAITING, mirror_x

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA
W, H = C.WIDTH, C.HEIGHT

# Colors (BGR)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
GOLD = (40, 200, 255)
ORANGE = (30, 140, 255)
CYAN = (255, 220, 80)
MAGENTA = (255, 60, 255)
GOOD = (120, 255, 140)
BAD = (90, 90, 255)
PLAYER_RUBBER = (55, 45, 215)    # red
CPU_RUBBER = (190, 110, 30)      # blue
WOOD = (90, 150, 205)
TABLE_TOP = (165, 85, 25)        # tournament blue
TABLE_SIDE = (100, 48, 12)

FLOOR_TOP = 600                  # where the back wall meets the floor
BARRIER = (418, 482)             # y range of the advertising boards behind the table
GLOW_SCALE = 4                   # glow layer is drawn at 1/4 resolution, then blurred


# ---------------------------------------------------------------- buttons

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
        Button((16, 14, 246, 76), lambda: f'SERVE: {"YOU" if game.server == "PLAYER" else "CPU"}',
               game.toggle_server),
        Button((258, 14, 488, 76), lambda: f'LEVEL: {game.difficulty}', game.cycle_difficulty),
        Button((500, 14, 730, 76), lambda: f'HAND: {game.side}', game.toggle_side),
    ]


# ---------------------------------------------------------------- drawing helpers

def text(img, s, pos, scale, color, thick=2, anchor='center', font=FONT, shadow=True):
    (tw, th), _ = cv2.getTextSize(s, font, scale, thick)
    x, y = pos
    if anchor == 'center':
        x -= tw // 2
    elif anchor == 'right':
        x -= tw
    x, y = int(x), int(y)
    if shadow:
        cv2.putText(img, s, (x + 2, y + 3), font, scale, (0, 0, 0), thick + 2, AA)
    cv2.putText(img, s, (x, y), font, scale, color, thick, AA)
    return tw, th


def rounded_rect(img, rect, r, color, thickness=-1):
    x0, y0, x1, y1 = [int(v) for v in rect]
    r = int(min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, AA)
    else:
        for (ax, ay), (bx, by) in (((x0 + r, y0), (x1 - r, y0)), ((x0 + r, y1), (x1 - r, y1)),
                                   ((x0, y0 + r), (x0, y1 - r)), ((x1, y0 + r), (x1, y1 - r))):
            cv2.line(img, (ax, ay), (bx, by), color, thickness, AA)
        for (cx, cy), start in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270),
                                ((x1 - r, y1 - r), 0), ((x0 + r, y1 - r), 90)):
            cv2.ellipse(img, (cx, cy), (r, r), 0, start, start + 90, color, thickness, AA)


def blend(img, rect, draw_fn, alpha):
    """Draw something semi-transparent inside rect (only that region is blended)."""
    x0, y0, x1, y1 = [int(v) for v in rect]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, img.shape[1]), min(y1, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    layer = roi.copy()
    draw_fn(layer, (x0, y0))
    cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, roi)


def scale_color(color, f):
    return tuple(int(c * f) for c in color)


def dashed_circle(img, center, r, color, dashes=24, thick=1):
    step = 360 / dashes
    for i in range(dashes):
        a = i * step
        cv2.ellipse(img, center, (r, r), 0, a, a + step * 0.55, color, thick, AA)


# ---------------------------------------------------------------- background

def build_background(side):
    """Static arena: lit back wall, boards, floor, table and net. Cached per side."""
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]
    img = np.zeros((H, W, 3), np.float32)

    # Back wall gradient + floor gradient
    wall_top, wall_bot = np.array([40, 14, 22]), np.array([78, 36, 40])
    t = np.clip(ys / FLOOR_TOP, 0, 1)[..., None]
    img[:] = wall_top * (1 - t) + wall_bot * t
    ft = np.clip((ys - FLOOR_TOP) / (H - FLOOR_TOP), 0, 1)[..., None]
    floor = np.array([48, 32, 34]) * (1 - ft) + np.array([22, 14, 16]) * ft
    img = np.where(ys[..., None] >= FLOOR_TOP, floor, img)

    # Spotlight over the table and its pool of light on the floor
    d = ((xs - W / 2) / 650) ** 2 + ((ys - 180) / 520) ** 2
    img += np.clip(1 - d, 0, 1)[..., None] ** 2 * np.array([70, 55, 50])
    d = ((xs - W / 2) / 620) ** 2 + ((ys - 670) / 70) ** 2
    img += np.clip(1 - d, 0, 1)[..., None] * np.array([40, 34, 34]) * (ys[..., None] >= FLOOR_TOP)

    # Soft bokeh lights high on the wall
    bokeh = np.zeros((H, W, 3), np.float32)
    rng = random.Random(7)
    for _ in range(45):
        c = rng.choice([(255, 180, 90), (255, 120, 200), (120, 200, 255), (200, 255, 255)])
        cv2.circle(bokeh, (rng.randrange(W), rng.randrange(20, 380)), rng.randrange(3, 16),
                   [v * rng.uniform(0.08, 0.25) for v in c], -1, AA)
    img += cv2.GaussianBlur(bokeh, (0, 0), 6)

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Perspective floor lines
    vx, vy = W / 2, 330
    for i in range(-12, 13):
        bx = W / 2 + i * 140
        fx = vx + (bx - vx) * (FLOOR_TOP - vy) / (H - vy)
        cv2.line(img, (int(fx), FLOOR_TOP), (int(bx), H), (62, 44, 46), 1, AA)
    for k in range(1, 7):
        y = int(FLOOR_TOP + (H - FLOOR_TOP) * (k / 6) ** 1.8)
        cv2.line(img, (0, y), (W, y), (58, 40, 42), 1, AA)

    # Advertising boards behind the table
    by0, by1 = BARRIER
    cv2.rectangle(img, (0, by0), (W, by1), (34, 22, 28), -1)
    panel_colors = [(120, 40, 20), (40, 30, 110), (90, 30, 90), (30, 90, 110)]
    for i, x in enumerate(range(0, W, 160)):
        cv2.rectangle(img, (x + 4, by0 + 6), (x + 156, by1 - 6), panel_colors[i % 4], -1)
    cv2.line(img, (0, by0), (W, by0), (120, 100, 110), 2)
    cv2.line(img, (0, by1), (W, by1), (20, 12, 16), 2)

    # Table shadow on the floor
    blend(img, (0, 640, W, H), lambda l, o: cv2.ellipse(
        l, (W // 2 - o[0], 692 - o[1]), (470, 20), 0, 0, 360, (5, 5, 5), -1, AA), 0.6)

    # Legs (far pair darker, near pair lighter)
    L, R, T = C.TABLE_LEFT, C.TABLE_RIGHT, C.TABLE_Y
    for x, shade in ((L + 90, (45, 45, 50)), (R - 90, (45, 45, 50))):
        cv2.line(img, (x, T + 16), (x + 8, 676), shade, 9, AA)
    for x, shade in ((L + 55, (80, 80, 88)), (R - 55, (80, 80, 88))):
        cv2.line(img, (x, T + 16), (x - 6, 694), shade, 11, AA)
        cv2.line(img, (x - 3, T + 16), (x - 9, 694), (130, 130, 140), 2, AA)

    # Table: front apron + top face (slight perspective) + white lines
    cv2.rectangle(img, (L, T), (R, T + 16), TABLE_SIDE, -1)
    top = np.array([(L, T), (R, T), (R - 16, T - 12), (L + 16, T - 12)], np.int32)
    cv2.fillPoly(img, [top], TABLE_TOP, AA)
    cv2.polylines(img, [top], True, WHITE, 2, AA)
    cv2.line(img, (L + 8, T - 6), (R - 8, T - 6), (235, 235, 235), 1, AA)   # center line
    cv2.line(img, (L, T + 16), (R, T + 16), (60, 30, 8), 2)

    # Net: skewed mesh panel, white tape on top, posts
    nx, nh = C.NET_X, C.NET_H
    net = np.array([(nx, T), (nx, T - nh), (nx + 7, T - nh - 12), (nx + 7, T - 12)], np.int32)
    blend(img, (nx - 10, T - nh - 20, nx + 20, T + 5),
          lambda l, o: cv2.fillPoly(l, [net - np.array(o, np.int32)], (30, 30, 30), AA), 0.55)
    for yy in range(T - nh, T, 5):
        cv2.line(img, (nx, yy), (nx + 7, yy - 12), (200, 200, 200), 1, AA)
    for k in range(0, 8, 3):
        cv2.line(img, (nx + k, T - k * 12 // 7), (nx + k, T - nh - k * 12 // 7), (200, 200, 200), 1, AA)
    cv2.line(img, (nx, T - nh), (nx + 7, T - nh - 12), WHITE, 3, AA)
    cv2.line(img, (nx, T + 2), (nx, T - nh - 2), (230, 230, 230), 3, AA)

    # Vignette
    d = ((xs - W / 2) / (W * 0.75)) ** 2 + ((ys - H / 2) / (H * 0.8)) ** 2
    img = (img * np.clip(1 - 0.6 * d, 0.3, 1)[..., None]).astype(np.uint8)

    if side == 'RIGHT':
        img = cv2.flip(img, 1)

    # Board text after flipping so it's readable on both sides
    for i, word in enumerate(['ME193', 'TUFTS', 'AI ROBOTICS', 'LEGO', 'PING PONG',
                              'ME193', 'TUFTS', 'MEDIAPIPE']):
        text(img, word, (i * 160 + 80, by0 + 42), 0.62, (225, 225, 235), 2, font=FONT_BOLD, shadow=False)
    return img


def make_red_vignette():
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]
    d = np.clip(((xs - W / 2) / (W / 2)) ** 2 + ((ys - H / 2) / (H / 2)) ** 2, 0, 2) / 2
    return (d[..., None] ** 1.5 * np.array([20, 20, 255], np.float32)).astype(np.float32)


def make_meter_gradient(h, w):
    t = np.linspace(1, 0, h, dtype=np.float32)[:, None, None]   # top = strongest
    green, yellow, red = np.array([90, 230, 90]), np.array([40, 220, 255]), np.array([60, 60, 255])
    col = np.where(t < 0.5, green + (yellow - green) * (t / 0.5), yellow + (red - yellow) * ((t - 0.5) / 0.5))
    return np.repeat(col, w, axis=1).astype(np.uint8)


# ---------------------------------------------------------------- renderer

class Renderer:
    def __init__(self):
        self._bg = {}
        self._red = make_red_vignette()
        self._meter_h = C.TABLE_Y - (C.TOP_BAR_H + 120)
        self._meter = make_meter_gradient(self._meter_h, 16)
        self.trail = deque(maxlen=16)
        self.particles = []          # [x, y, vx, vy, life, max_life, color, size]
        self.ripples = []            # [x, t0]
        self.popups = []             # [text, x, y, t0, color, scale]
        self.shake = 0.0
        self.flash = 0.0
        self.swing_t = -10.0         # player paddle swing animation start
        self.swing_power = 1.0
        self.cpu_swing_t = -10.0
        self.score_t = -10.0         # score pulse start
        self.banner_t = -10.0        # NEW HIGH SCORE banner start
        self._high_announced = False
        self._last_t = None
        self._rng = random.Random()

    # ----- effects from game events
    def _spark(self, x, y, n, colors, speed=420, size=(2, 5), up=True):
        for _ in range(n):
            a = self._rng.uniform(0, 2 * math.pi)
            v = self._rng.uniform(0.3, 1.0) * speed
            life = self._rng.uniform(0.35, 0.8)
            self.particles.append([x, y, math.cos(a) * v, math.sin(a) * v - (200 if up else 0),
                                   life, life, self._rng.choice(colors), self._rng.uniform(*size)])

    def _handle_events(self, game, now):
        S = lambda x: mirror_x(x, game.side)  # noqa: E731
        for kind, e in game.pop_events():
            if kind == 'swing':
                self.swing_t, self.swing_power = now, e['power']
            elif kind == 'hit':
                x, y, p = S(e['x']), e['y'], e['power']
                if e['quality'] == 'good':
                    self._spark(x, y, int(14 + 18 * min(p, 2)), [ORANGE, GOLD, WHITE], 380 + 200 * p)
                    self.popups.append(['+1', x, y - 30, now, GOOD, 1.4])
                    self.score_t = now
                else:
                    self._spark(x, y, 10, [GREY, BAD])
                    self.popups.append(['TOO SOFT' if e['quality'] == 'soft' else 'TOO HARD',
                                        x, y - 30, now, BAD, 0.9])
                self.shake = max(self.shake, 3 + 5 * min(p, 2))
            elif kind == 'whiff':
                px, py = game.paddle
                self.popups.append(['WHIFF', S(px), py - 50, now, GREY, 0.8])
            elif kind == 'cpu_hit':
                self.cpu_swing_t = now
                self._spark(S(e['x']), e['y'], 10, [CYAN, WHITE], 280)
            elif kind == 'bounce':
                self.ripples.append([S(e['x']), now])
                self._spark(S(e['x']), C.TABLE_Y - 4, 5, [WHITE, CYAN], 160, (1, 3))
            elif kind == 'lost':
                self.flash = 1.0
                self.shake = max(self.shake, 9)
                self._spark(S(e['x']), min(e['y'], H - 10), 18, [BAD, (40, 40, 200)], 300)
                self._high_announced = False
                self.trail.clear()
            elif kind == 'new_high' and not self._high_announced and e['score'] > 1:
                self.banner_t = now
                self._high_announced = True
            elif kind == 'serve':
                self.trail.clear()
                self._high_announced = False

    # ----- paddle
    def _draw_paddle(self, img, glow, gx, gy, lean, facing, rubber, side):
        """Paddle at game coords (gx, gy). lean = degrees tilted toward the opponent."""
        head = cv2.ellipse2Poly((0, 0), (17, 27), int(lean), 0, 360, 12).astype(np.float32)
        a = math.radians(lean)
        hd = (-math.sin(a), math.cos(a))   # handle direction (head's "down" axis)
        handle = np.array([(hd[0] * 22, hd[1] * 22), (hd[0] * 50, hd[1] * 50)], np.float32)

        def to_screen(pts):
            pts = pts.copy()
            pts[:, 0] = mirror_x(gx + facing * pts[:, 0], side)
            pts[:, 1] += gy
            return pts.astype(np.int32)

        h = to_screen(handle)
        cv2.line(img, tuple(h[0]), tuple(h[1]), (40, 70, 100), 12, AA)
        cv2.line(img, tuple(h[0]), tuple(h[1]), WOOD, 8, AA)
        poly = to_screen(head)
        cv2.fillPoly(img, [poly], rubber, AA)
        cv2.polylines(img, [poly], True, (225, 225, 225), 2, AA)   # sponge edge
        hl = to_screen(head * 0.55 + np.array([-4 * facing, -6], np.float32))
        cv2.polylines(img, [hl], True, scale_color(rubber, 1.4), 1, AA)
        c = to_screen(np.zeros((1, 2), np.float32))[0]
        cv2.circle(glow, (c[0] // GLOW_SCALE, c[1] // GLOW_SCALE), 9, scale_color(rubber, 0.5), -1)

    @staticmethod
    def _swing_lean(t_since, power):
        if t_since > 0.32:
            return 0.0
        return math.sin(math.pi * t_since / 0.32) * (35 + 30 * min(power, 2) / 2)

    # ----- main draw
    def draw(self, game, clicker, cursor=None, preview=None, meter=0.0, show_debug=True, info=''):
        now = game.time
        dt = 0.0 if self._last_t is None else max(0.0, min(now - self._last_t, 0.1))
        self._last_t = now
        side = game.side
        S = lambda x: int(mirror_x(x, side))  # noqa: E731

        self._handle_events(game, now)
        if side not in self._bg:
            self._bg[side] = build_background(side)
        img = self._bg[side].copy()
        glow = np.zeros((H // GLOW_SCALE, W // GLOW_SCALE, 3), np.uint8)

        def g(x, y, r, color):
            cv2.circle(glow, (int(x) // GLOW_SCALE, int(y) // GLOW_SCALE), max(1, int(r) // GLOW_SCALE),
                       color, -1, AA)

        # Bounce ripples on the table top
        for rp in self.ripples:
            age = now - rp[1]
            r = int(10 + age * 120)
            fade = max(0.0, 1 - age / 0.45)
            cv2.ellipse(img, (int(rp[0]), C.TABLE_Y - 5), (r, max(2, r // 5)), 0, 0, 360,
                        scale_color((255, 230, 200), fade), 2, AA)
        self.ripples = [rp for rp in self.ripples if now - rp[1] < 0.45]

        # Paddle zone
        x0, y0, x1, y1 = C.PADDLE_ZONE
        zx0, zx1 = sorted((S(x0), S(x1)))
        rounded_rect(img, (zx0, y0, zx1, y1), 18, (90, 70, 80), 1)
        text(img, 'YOUR ZONE', ((zx0 + zx1) // 2, y0 + 22), 0.5, (140, 120, 130), 1, shadow=False)

        b = game.ball
        R = C.BALL_RADIUS

        # Ball shadow (on the table when above it, otherwise on the floor)
        if b is not None:
            bx = S(b.x)
            if C.TABLE_LEFT <= b.x <= C.TABLE_RIGHT and b.y <= C.TABLE_Y + 1:
                hgt = C.TABLE_Y - b.y
                sy, sa = C.TABLE_Y - 5, 0.55 * max(0.25, 1 - hgt / 350)
            else:
                hgt = 690 - b.y
                sy, sa = 690, 0.45 * max(0.2, 1 - hgt / 500)
            sw = int(R + 4 + hgt * 0.03)
            blend(img, (bx - sw - 2, sy - 6, bx + sw + 2, sy + 6), lambda l, o: cv2.ellipse(
                l, (bx - o[0], sy - o[1]), (sw, 4), 0, 0, 360, (0, 0, 0), -1, AA), sa)

        # CPU paddle follows the ball when it's coming its way
        cpu_y = C.TABLE_Y - 70
        if b is not None and game.last_hitter == 'PLAYER':
            cpu_y = min(max(b.y, C.TABLE_Y - 260), C.TABLE_Y + 30)
        self._draw_paddle(img, glow, C.CPU_HIT_X + 18, cpu_y, 10 + self._swing_lean(now - self.cpu_swing_t, 1.0),
                          -1, CPU_RUBBER, side)

        # Player paddle (+ dashed hit radius when debugging)
        px, py = game.paddle
        if show_debug:
            dashed_circle(img, (S(px), int(py)), game.settings['hit_radius'], (110, 90, 170))
        self._draw_paddle(img, glow, px, py, 12 + self._swing_lean(now - self.swing_t, self.swing_power),
                          1, PLAYER_RUBBER, side)

        # Ball: trail, glow, body, highlight
        if b is not None:
            pos = (S(b.x), int(b.y - R))
            self.trail.append(pos)
            n = len(self.trail)
            for i, (tx, ty) in enumerate(self.trail):
                f = (i + 1) / n
                cv2.circle(img, (tx, ty), max(1, int(R * f * 0.8)), scale_color(ORANGE, 0.25 + 0.5 * f), -1, AA)
                g(tx, ty, R * 1.6 * f, scale_color(ORANGE, 0.5 * f))
            g(pos[0], pos[1], R * 3, (20, 90, 200))
            cv2.circle(img, pos, R, (20, 110, 230), -1, AA)
            cv2.circle(img, (pos[0] - 1, pos[1] - 1), R - 2, ORANGE, -1, AA)
            cv2.circle(img, (pos[0] - 3, pos[1] - 4), 3, (210, 235, 255), -1, AA)
            if game._hittable():
                pulse = 0.6 + 0.4 * math.sin(now * 18)
                cv2.circle(img, pos, R + 7, scale_color(GOLD, pulse), 2, AA)
                g(pos[0], pos[1], R + 12, scale_color(GOLD, 0.6))

        # Particles
        alive = []
        for p in self.particles:
            p[4] -= dt
            if p[4] <= 0:
                continue
            p[2] *= 0.985
            p[3] += 900 * dt
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            f = p[4] / p[5]
            cv2.circle(img, (int(p[0]), int(p[1])), max(1, int(p[7] * f)), scale_color(p[6], 0.4 + 0.6 * f), -1, AA)
            g(p[0], p[1], p[7] * 3 * f, scale_color(p[6], f))
            alive.append(p)
        self.particles = alive

        # Composite glow
        glow = cv2.GaussianBlur(glow, (0, 0), 2.2)
        img = cv2.add(img, cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR))

        # Screen shake + red flash on a lost ball
        if self.shake > 0.3:
            sx = self._rng.uniform(-1, 1) * self.shake
            sy = self._rng.uniform(-1, 1) * self.shake
            img = cv2.warpAffine(img, np.float32([[1, 0, sx], [0, 1, sy]]), (W, H),
                                 borderMode=cv2.BORDER_REPLICATE)
        self.shake *= math.exp(-dt * 10)
        if self.flash > 0.02:
            img = np.clip(img + self._red * self.flash, 0, 255).astype(np.uint8)
        self.flash *= math.exp(-dt * 4)

        self._draw_hud(img, game, clicker, now, meter, side)

        # Floating popups ("+1", "TOO HARD", ...)
        for pu in self.popups:
            age = now - pu[3]
            f = max(0.0, 1 - age / 1.0)
            text(img, pu[0], (pu[1], pu[2] - age * 70), pu[5] * (1 + 0.3 * f), scale_color(pu[4], f), 2,
                 font=FONT_BOLD)
        self.popups = [pu for pu in self.popups if now - pu[3] < 1.0]

        # Camera preview, in the bottom corner on the CPU's side
        if preview is not None:
            small = cv2.resize(preview, (C.PREVIEW_W, C.PREVIEW_H))
            vx0 = W - C.PREVIEW_W - 12 if side == 'LEFT' else 12
            vy0 = H - C.PREVIEW_H - 26
            rounded_rect(img, (vx0 - 4, vy0 - 4, vx0 + C.PREVIEW_W + 4, vy0 + C.PREVIEW_H + 4), 8, (20, 20, 20))
            img[vy0:vy0 + C.PREVIEW_H, vx0:vx0 + C.PREVIEW_W] = small
            rounded_rect(img, (vx0 - 4, vy0 - 4, vx0 + C.PREVIEW_W + 4, vy0 + C.PREVIEW_H + 4), 8, (120, 120, 130), 2)
            text(img, 'CAMERA', (vx0 + 8, vy0 + 22), 0.5, WHITE, 1, anchor='left')

        if info:
            text(img, info, (W // 2, H - 8), 0.45, (170, 170, 180), 1, shadow=True)

        if cursor is not None:
            cx, cy = int(cursor[0]), int(cursor[1])
            cv2.circle(img, (cx, cy), 16, MAGENTA, 2, AA)
            cv2.drawMarker(img, (cx, cy), MAGENTA, cv2.MARKER_CROSS, 40, 2, AA)
            if clicker.progress > 0:
                cv2.ellipse(img, (cx, cy), (26, 26), -90, 0, 360 * clicker.progress, GOLD, 5, AA)
        return img

    # ----- HUD
    def _draw_hud(self, img, game, clicker, now, meter, side):
        # Glassy top bar with a neon line under it
        bar = img[:C.TOP_BAR_H]
        cv2.addWeighted(bar, 0.35, np.full_like(bar, (28, 14, 18)), 0.65, 0, bar)
        cv2.line(img, (0, C.TOP_BAR_H), (W, C.TOP_BAR_H), (200, 90, 220), 2, AA)
        cv2.line(img, (0, C.TOP_BAR_H + 2), (W, C.TOP_BAR_H + 2), (100, 40, 110), 2, AA)

        for btn in clicker.buttons:
            x0, y0, x1, y1 = btn.rect
            hot = btn is clicker.hover
            rounded_rect(img, btn.rect, 14, (70, 50, 60) if hot else (50, 34, 42))
            if hot:   # dwell progress fills the button
                fill_x = int(x0 + (x1 - x0) * clicker.progress)
                if fill_x > x0 + 4:
                    rounded_rect(img, (x0, y0, fill_x, y1), 14, (40, 120, 160))
            rounded_rect(img, btn.rect, 14, GOLD if hot else (150, 120, 140), 2)
            text(img, btn.label_fn(), ((x0 + x1) / 2, (y0 + y1) / 2 + 9), 0.7, WHITE, 2, font=FONT_BOLD)

        # Score (pulses on every hit) and best
        pulse = math.exp(-(now - self.score_t) * 7)
        text(img, 'SCORE', (905, 30), 0.5, (190, 170, 190), 1)
        text(img, str(game.score), (905, 78), 1.5 + 0.6 * pulse,
             tuple(int(255 - (255 - c) * pulse) for c in GOOD), 3, font=FONT_BOLD)
        text(img, 'BEST', (1140, 30), 0.5, (190, 170, 190), 1)
        text(img, str(game.high_score), (1140, 78), 1.5, GOLD, 3, font=FONT_BOLD)

        # Message under the bar, colored by kind, fades after 2 s
        age = game.time - game.message_time
        if game.message and age < 2.0:
            f = 1.0 - max(0.0, age - 1.2) / 0.8
            color = {'good': GOOD, 'bad': BAD}.get(game.message_kind, WHITE)
            text(img, game.message.upper(), (W // 2, C.TOP_BAR_H + 52), 1.0, scale_color(color, f), 2,
                 font=FONT_BOLD)

        # Serve countdown
        if game.phase == WAITING:
            frac = min(game.serve_in / C.SERVE_DELAY_SEC, 1.0)
            who = 'YOUR SERVE' if game.server == 'PLAYER' else 'CPU SERVES'
            text(img, who, (W // 2, 300), 1.3, WHITE, 3, font=FONT_BOLD)
            bw = 260
            rounded_rect(img, (W // 2 - bw // 2, 320, W // 2 + bw // 2, 330), 5, (60, 50, 60))
            if frac > 0.03:
                rounded_rect(img, (W // 2 - bw // 2, 320, W // 2 - bw // 2 + int(bw * frac), 330), 5, GOLD)

        # NEW HIGH SCORE banner
        age = now - self.banner_t
        if age < 1.8:
            pop = min(1.0, age / 0.15)
            f = 1.0 if age < 1.3 else max(0.0, 1 - (age - 1.3) / 0.5)
            text(img, 'NEW HIGH SCORE!', (W // 2, 230), (1.0 + 0.6 * pop) * (1 + 0.05 * math.sin(now * 20)),
                 scale_color(GOLD, f), 4, font=FONT_BOLD)

        self._draw_meter(img, game, meter, side)

    def _draw_meter(self, img, game, level, side):
        """Power meter on the player's edge: live swing power + threshold / min / max marks."""
        s = game.settings
        mh = self._meter_h
        top, bot = C.TOP_BAR_H + 120, C.TOP_BAR_H + 120 + mh
        x0 = 16 if side == 'LEFT' else W - 32
        x1 = x0 + 16
        full = 2.5

        def y_of(p):
            return int(bot - mh * min(max(p, 0), full) / full)

        rounded_rect(img, (x0 - 3, top - 3, x1 + 3, bot + 3), 8, (25, 18, 22))
        ly = y_of(level)
        if ly < bot:
            img[ly:bot, x0:x1] = self._meter[ly - top:mh]
        rounded_rect(img, (x0 - 3, top - 3, x1 + 3, bot + 3), 8, (150, 130, 150), 1)
        for p, color, label in ((C.SWING_THRESHOLD, GREY, ''), (s['min_power'], WHITE, 'min'),
                                (s['max_power'], WHITE, 'max')):
            yy = y_of(p)
            cv2.line(img, (x0 - 6, yy), (x1 + 6, yy), color, 2, AA)
            if label:
                lx = x1 + 10 if side == 'LEFT' else x0 - 10
                text(img, label, (lx, yy + 5), 0.4, color, 1, anchor='left' if side == 'LEFT' else 'right')
        cx = (x0 + x1) // 2
        text(img, 'POWER', (cx + (18 if side == 'LEFT' else -18), top - 14), 0.45, (200, 180, 200), 1)
        if game.last_swing_power is not None:
            text(img, f'{game.last_swing_power:.2f}', (cx + (18 if side == 'LEFT' else -18), bot + 24),
                 0.55, WHITE, 1)
