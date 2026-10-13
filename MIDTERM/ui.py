"""
Everything drawn on screen (OpenCV), plus the on-screen buttons that the
AprilTag cursor (or mouse) clicks by hovering.

The 3D view is a simple pinhole camera (View) that projects world points
(meters) to pixels, and things are drawn back-to-front:
  cached arena (floor, wall, table) -> shadows -> CPU paddle -> net / ball
  (whichever is farther first) -> sparks -> your paddle (see-through)
  -> glow -> screen shake -> HUD (top bar, score, meter, camera preview, cursor)
"""

import math
import random
from collections import deque

import cv2
import numpy as np

import config as C
from game import WAITING

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

WALL_Z = 6.5                     # back wall
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
        Button((500, 14, 730, 76), lambda: f'HAND: {game.hand}', game.toggle_hand),
    ]


# ---------------------------------------------------------------- 3D view

class View:
    """Pinhole camera: world (meters) -> screen (pixels)."""

    def __init__(self, eye=C.VIEW_EYE, target=C.VIEW_TARGET, focal=C.VIEW_FOCAL, center=C.VIEW_CENTER):
        self.eye = np.array(eye, np.float64)
        f = np.array(target, np.float64) - self.eye
        f /= np.linalg.norm(f)
        r = np.cross([0.0, 1.0, 0.0], f)
        r /= np.linalg.norm(r)
        u = np.cross(f, r)
        self.rot = np.stack([r, u, f])     # rows: camera right, up, forward
        self.focal = focal
        self.cx, self.cy = center

    def project(self, pts):
        """(N, 3) world points -> (N, 2) pixels and (N,) depths."""
        c = (np.asarray(pts, np.float64) - self.eye) @ self.rot.T
        z = np.maximum(c[:, 2], 0.05)
        return np.stack([self.cx + self.focal * c[:, 0] / z, self.cy - self.focal * c[:, 1] / z], 1), z

    def pt(self, x, y, z):
        """One world point -> ((px, py) ints, depth)."""
        p, d = self.project([(x, y, z)])
        return (int(p[0, 0]), int(p[0, 1])), float(d[0])

    def unproject(self, px, py, z_plane):
        """Screen pixel -> world (x, y) on the plane z = z_plane (for the mouse)."""
        d = self.rot.T @ np.array([(px - self.cx) / self.focal, -(py - self.cy) / self.focal, 1.0])
        t = (z_plane - self.eye[2]) / d[2]
        p = self.eye + t * d
        return float(p[0]), float(p[1])

    def size(self, meters, depth):
        return self.focal * meters / depth

    def poly(self, pts):
        return self.project(pts)[0].astype(np.int32)


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


def blend_poly(img, poly, color, alpha):
    """Fill a polygon semi-transparently (only its bounding box is blended)."""
    x0, y0 = np.maximum(poly.min(0) - 2, 0)
    x1, y1 = np.minimum(poly.max(0) + 2, (img.shape[1], img.shape[0]))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    layer = roi.copy()
    cv2.fillPoly(layer, [poly - np.array([x0, y0])], color, AA)
    cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, roi)


def blend_draw(img, rect, draw_fn, alpha):
    """Run draw_fn(layer, offset) on a copy of rect and blend it back in."""
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


def circle_on_plane(cx, y, cz, r, n=20):
    """Points of a flat horizontal circle (on the table or floor)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), np.full(n, y), cz + r * np.sin(a)], 1)


def circle_facing(cx, cy, cz, r, n=32):
    """Points of an upright circle facing the camera (in a z = const plane)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.full(n, cz)], 1)


# ---------------------------------------------------------------- background

def build_background(view):
    """Static arena: lit wall, boards, floor grid, table with legs and shadow."""
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]
    floor_y = view.pt(0, 0, WALL_Z)[0][1]          # where the floor meets the back wall

    img = np.zeros((H, W, 3), np.float32)
    t = np.clip(ys / floor_y, 0, 1)[..., None]
    img[:] = np.array([40, 14, 22]) * (1 - t) + np.array([78, 36, 40]) * t
    ft = np.clip((ys - floor_y) / (H - floor_y), 0, 1)[..., None]
    floor = np.array([52, 34, 36]) * (1 - ft) + np.array([24, 15, 17]) * ft
    img = np.where(ys[..., None] >= floor_y, floor, img)

    # Spotlight from above the table, and its pool of light on the floor
    d = ((xs - W / 2) / 700) ** 2 + ((ys - 150) / 480) ** 2
    img += np.clip(1 - d, 0, 1)[..., None] ** 2 * np.array([70, 55, 50])
    d = ((xs - W / 2) / 650) ** 2 + ((ys - (floor_y + 160)) / 170) ** 2
    img += np.clip(1 - d, 0, 1)[..., None] * np.array([38, 32, 32]) * (ys[..., None] >= floor_y)

    # Soft bokeh lights high on the wall
    bokeh = np.zeros((H, W, 3), np.float32)
    rng = random.Random(7)
    for _ in range(45):
        c = rng.choice([(255, 180, 90), (255, 120, 200), (120, 200, 255), (200, 255, 255)])
        cv2.circle(bokeh, (rng.randrange(W), rng.randrange(20, max(40, floor_y - 120))), rng.randrange(3, 14),
                   [v * rng.uniform(0.08, 0.25) for v in c], -1, AA)
    img += cv2.GaussianBlur(bokeh, (0, 0), 6)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Floor grid (1 m squares)
    for gx in range(-8, 9):
        a, b = view.poly([(gx, 0, -2.9), (gx, 0, WALL_Z)])
        cv2.line(img, tuple(a), tuple(b), (66, 46, 48), 1, AA)
    for gz in range(-2, int(WALL_Z) + 1):
        a, b = view.poly([(-9, 0, gz), (9, 0, gz)])
        cv2.line(img, tuple(a), tuple(b), (62, 44, 46), 1, AA)

    # Advertising boards along the back wall
    board_h = 0.9
    panel_colors = [(120, 40, 20), (40, 30, 110), (90, 30, 90), (30, 90, 110)]
    words = ['ME193', 'TUFTS', 'AI ROBOTICS', 'LEGO', 'PING PONG', 'MEDIAPIPE', 'ME193', 'TUFTS']
    cv2.fillPoly(img, [view.poly([(-9, 0, WALL_Z), (9, 0, WALL_Z), (9, board_h, WALL_Z), (-9, board_h, WALL_Z)])],
                 (34, 22, 28), AA)
    for i in range(8):
        x0 = -8 + 2 * i
        poly = view.poly([(x0 + 0.05, 0.08, WALL_Z), (x0 + 1.95, 0.08, WALL_Z),
                          (x0 + 1.95, board_h - 0.08, WALL_Z), (x0 + 0.05, board_h - 0.08, WALL_Z)])
        cv2.fillPoly(img, [poly], panel_colors[i % 4], AA)
        (px, py), depth = view.pt(x0 + 1.0, board_h / 2 - 0.12, WALL_Z)
        text(img, words[i], (px, py), view.size(0.0085, depth), (225, 225, 235), 2, font=FONT_BOLD, shadow=False)
    a, b = view.poly([(-9, board_h, WALL_Z), (9, board_h, WALL_Z)])
    cv2.line(img, tuple(a), tuple(b), (120, 100, 110), 2, AA)

    hw, hl, th = C.TABLE_HALF_W, C.TABLE_HALF_L, C.TABLE_H

    # Table shadow on the floor
    shadow = view.poly([(-hw - 0.1, 0, -hl - 0.1), (hw + 0.1, 0, -hl - 0.1), (hw + 0.1, 0, hl + 0.1),
                        (-hw - 0.1, 0, hl + 0.1)])
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [shadow], 255, AA)
    mask = cv2.GaussianBlur(mask, (0, 0), 12).astype(np.float32)[..., None] / 255
    img = (img * (1 - 0.6 * mask)).astype(np.uint8)

    # Legs (far ones first)
    for lz in (hl - 0.25, -hl + 0.25):
        for lx in (-hw + 0.12, hw - 0.12):
            (ax, ay), d = view.pt(lx, th - 0.03, lz)
            (bx, by), _ = view.pt(lx, 0, lz)
            wid = max(2, int(view.size(0.05, d)))
            cv2.line(img, (ax, ay), (bx, by), (70, 70, 78), wid, AA)
            cv2.line(img, (ax - wid // 3, ay), (bx - wid // 3, by), (125, 125, 135), max(1, wid // 4), AA)

    # Table: front edge face, then the top, then white lines
    front = view.poly([(-hw, th, -hl), (hw, th, -hl), (hw, th - 0.05, -hl), (-hw, th - 0.05, -hl)])
    cv2.fillPoly(img, [front], TABLE_SIDE, AA)
    top = view.poly([(-hw, th, -hl), (hw, th, -hl), (hw, th, hl), (-hw, th, hl)])
    cv2.fillPoly(img, [top], TABLE_TOP, AA)
    cv2.polylines(img, [top], True, WHITE, 3, AA)
    a, b = view.poly([(0, th, -hl), (0, th, hl)])
    cv2.line(img, tuple(a), tuple(b), (235, 235, 235), 2, AA)    # center line

    # Vignette
    d = ((xs - W / 2) / (W * 0.75)) ** 2 + ((ys - H / 2) / (H * 0.8)) ** 2
    return (img * np.clip(1 - 0.6 * d, 0.3, 1)[..., None]).astype(np.uint8)


def draw_net(img, view):
    hw, th, nh = C.TABLE_HALF_W + C.NET_OVERHANG, C.TABLE_H, C.NET_H
    blend_poly(img, view.poly([(-hw, th, 0), (hw, th, 0), (hw, th + nh, 0), (-hw, th + nh, 0)]), (25, 25, 25), 0.5)
    for k in range(1, 6):
        y = th + nh * k / 6
        a, b = view.poly([(-hw, y, 0), (hw, y, 0)])
        cv2.line(img, tuple(a), tuple(b), (170, 170, 170), 1, AA)
    for x in np.arange(-hw, hw + 1e-6, 0.08):
        a, b = view.poly([(x, th, 0), (x, th + nh, 0)])
        cv2.line(img, tuple(a), tuple(b), (150, 150, 150), 1, AA)
    a, b = view.poly([(-hw, th + nh, 0), (hw, th + nh, 0)])
    cv2.line(img, tuple(a), tuple(b), WHITE, 4, AA)                     # tape
    for x in (-hw, hw):                                                  # posts
        a, b = view.poly([(x, th - 0.02, 0), (x, th + nh + 0.01, 0)])
        cv2.line(img, tuple(a), tuple(b), (60, 60, 60), 5, AA)


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
    def __init__(self, view=None):
        self.view = view or View()
        self._bg = build_background(self.view)
        self._red = make_red_vignette()
        self._meter_top = C.TOP_BAR_H + 130
        self._meter_h = 330
        self._meter = make_meter_gradient(self._meter_h, 16)
        self.trail = deque(maxlen=14)    # recent ball positions (world)
        self.particles = []              # [x, y, z, vx, vy, vz, life, max_life, color, size_m]
        self.ripples = []                # [x, z, t0]
        self.popups = []                 # [text, (x, y, z), t0, color, scale]
        self.shake = 0.0
        self.flash = 0.0
        self.swing_t = -10.0             # player paddle swing animation start
        self.swing_power = 1.0
        self.cpu_swing_t = -10.0
        self.score_t = -10.0             # score pulse start
        self.banner_t = -10.0            # NEW HIGH SCORE banner start
        self._high_announced = False
        self._last_t = None
        self._rng = random.Random()

    # ----- effects from game events
    def _spark(self, pos, n, colors, speed=3.0, size=(0.008, 0.02)):
        for _ in range(n):
            # random direction on a sphere, biased upward
            u, a = self._rng.uniform(-1, 1), self._rng.uniform(0, 2 * math.pi)
            s = math.sqrt(1 - u * u)
            v = self._rng.uniform(0.3, 1.0) * speed
            life = self._rng.uniform(0.35, 0.8)
            self.particles.append([*pos, s * math.cos(a) * v, u * v + 1.2, s * math.sin(a) * v,
                                   life, life, self._rng.choice(colors), self._rng.uniform(*size)])

    def _handle_events(self, game, now):
        for kind, e in game.pop_events():
            pos = (e.get('x', 0), e.get('y', 0), e.get('z', 0))
            if kind == 'swing':
                self.swing_t, self.swing_power = now, e['power']
            elif kind == 'hit':
                p = e['power']
                if e['quality'] == 'good':
                    self._spark(pos, int(16 + 18 * min(p, 2)), [ORANGE, GOLD, WHITE], 2.0 + 1.2 * p)
                    self.popups.append(['+1', (pos[0], pos[1] + 0.15, pos[2]), now, GOOD, 1.5])
                    self.score_t = now
                else:
                    self._spark(pos, 10, [GREY, BAD], 1.5)
                    self.popups.append(['TOO SOFT' if e['quality'] == 'soft' else 'TOO HARD',
                                        (pos[0], pos[1] + 0.15, pos[2]), now, BAD, 1.0])
                self.shake = max(self.shake, 3 + 5 * min(p, 2))
            elif kind == 'whiff':
                px, py = game.paddle
                self.popups.append(['WHIFF', (px, py + 0.2, C.PLAYER_Z), now, GREY, 0.9])
            elif kind == 'cpu_hit':
                self.cpu_swing_t = now
                self._spark(pos, 12, [CYAN, WHITE], 1.6)
            elif kind == 'bounce':
                self.ripples.append([pos[0], pos[2], now])
                self._spark((pos[0], C.TABLE_H + 0.01, pos[2]), 6, [WHITE, CYAN], 1.0, (0.005, 0.01))
            elif kind == 'lost':
                self.flash = 1.0
                self.shake = max(self.shake, 9)
                self._spark((pos[0], max(pos[1], 0.05), pos[2]), 20, [BAD, (40, 40, 200)], 2.0)
                self._high_announced = False
                self.trail.clear()
            elif kind == 'new_high' and not self._high_announced and e['score'] > 1:
                self.banner_t = now
                self._high_announced = True
            elif kind == 'serve':
                self.trail.clear()
                self._high_announced = False

    # ----- paddles
    @staticmethod
    def _swing_lean(t_since, power):
        if t_since > 0.32:
            return 0.0
        return math.sin(math.pi * t_since / 0.32) * (35 + 30 * min(power, 2) / 2)

    def _draw_paddle(self, img, glow, x, y, z, lean, rubber, alpha=1.0):
        """Round paddle facing the camera at world (x, y, z), handle pointing down."""
        v = self.view
        (cx, cy), depth = v.pt(x, y, z)
        r = v.size(C.PADDLE_RADIUS, depth)
        a = math.radians(lean)
        hd = (-math.sin(a), math.cos(a))       # handle direction on screen
        h0 = (int(cx + hd[0] * r * 0.85), int(cy + hd[1] * r * 0.85))
        h1 = (int(cx + hd[0] * r * 1.9), int(cy + hd[1] * r * 1.9))
        head = cv2.ellipse2Poly((cx, cy), (max(2, int(r * 0.92)), max(2, int(r))), int(lean), 0, 360, 10)
        hw = max(2, int(r * 0.32))

        def draw(layer, o):
            off = np.array(o)
            cv2.line(layer, tuple(np.subtract(h0, o)), tuple(np.subtract(h1, o)), (40, 70, 100), hw + 4, AA)
            cv2.line(layer, tuple(np.subtract(h0, o)), tuple(np.subtract(h1, o)), WOOD, hw, AA)
            cv2.fillPoly(layer, [head - off], rubber, AA)
            cv2.polylines(layer, [head - off], True, (225, 225, 225), max(1, int(r * 0.08)), AA)
            hl = ((head - (cx, cy)) * 0.55 + (cx - r * 0.15, cy - r * 0.2)).astype(np.int32)
            cv2.polylines(layer, [hl - off], True, scale_color(rubber, 1.4), 1, AA)

        pad = int(r * 2.2) + 8
        if alpha >= 1.0:
            draw(img, (0, 0))
        else:
            blend_draw(img, (cx - pad, cy - pad, cx + pad, cy + pad), draw, alpha)
        cv2.circle(glow, (cx // GLOW_SCALE, cy // GLOW_SCALE), max(1, int(r * 0.7) // GLOW_SCALE),
                   scale_color(rubber, 0.5), -1)

    def _surface_shadow(self, img, x, y, z, radius, strength=0.55):
        """Dark blob under (x, y, z): on the table if above it, else on the floor."""
        over = abs(x) <= C.TABLE_HALF_W and abs(z) <= C.TABLE_HALF_L and y >= C.TABLE_H
        ground = C.TABLE_H + 0.002 if over else 0.0
        hgt = max(0.0, y - ground)
        poly = self.view.poly(circle_on_plane(x, ground, z, radius * (1 + hgt * 0.8), 16))
        blend_poly(img, poly, (0, 0, 0), strength * max(0.2, 1 - hgt / 1.2))

    # ----- main draw
    def draw(self, game, clicker, cursor=None, preview=None, meter=0.0, show_debug=True, info='', alert=''):
        v = self.view
        now = game.time
        dt = 0.0 if self._last_t is None else max(0.0, min(now - self._last_t, 0.1))
        self._last_t = now
        self._handle_events(game, now)

        img = self._bg.copy()
        glow = np.zeros((H // GLOW_SCALE, W // GLOW_SCALE, 3), np.uint8)

        def g(px, py, r, color):
            cv2.circle(glow, (int(px) // GLOW_SCALE, int(py) // GLOW_SCALE), max(1, int(r) // GLOW_SCALE),
                       color, -1, AA)

        b = game.ball
        R = C.BALL_RADIUS
        px, py = game.paddle

        # Bounce ripples on the table
        for rp in self.ripples:
            age = now - rp[2]
            poly = v.poly(circle_on_plane(rp[0], C.TABLE_H + 0.003, rp[1], 0.04 + age * 0.5, 24))
            cv2.polylines(img, [poly], True, scale_color((255, 230, 200), max(0.0, 1 - age / 0.45)), 2, AA)
        self.ripples = [rp for rp in self.ripples if now - rp[2] < 0.45]

        # Shadows: ball, both paddles
        if b is not None:
            self._surface_shadow(img, b.x, b.y, b.z, R * 1.1)
        self._surface_shadow(img, px, py, C.PLAYER_Z, C.PADDLE_RADIUS, 0.35)
        cpu_y = C.TABLE_H + 0.3 if b is None else min(max(b.y, C.TABLE_H + 0.1), C.TABLE_H + 0.8)
        self._surface_shadow(img, game.cpu_x, cpu_y, C.CPU_Z, C.PADDLE_RADIUS, 0.35)

        # CPU paddle (farthest)
        self._draw_paddle(img, glow, game.cpu_x, cpu_y, C.CPU_Z,
                          -10 - self._swing_lean(now - self.cpu_swing_t, 1.0), CPU_RUBBER)

        # Net and ball, farthest first
        def draw_ball():
            if b is None:
                return
            self.trail.append((b.x, b.y, b.z))
            pts, depths = v.project(list(self.trail))
            n = len(pts)
            for i, ((tx, ty), d) in enumerate(zip(pts, depths)):
                f = (i + 1) / n
                r = v.size(R, d)
                cv2.circle(img, (int(tx), int(ty)), max(1, int(r * f * 0.8)), scale_color(ORANGE, 0.25 + 0.5 * f), -1, AA)
                g(tx, ty, r * 1.6 * f, scale_color(ORANGE, 0.5 * f))
            (bx, by), d = v.pt(b.x, b.y, b.z)
            r = max(3, int(v.size(R, d)))
            g(bx, by, r * 3, (20, 90, 200))
            cv2.circle(img, (bx, by), r, (20, 110, 230), -1, AA)
            cv2.circle(img, (bx - r // 10, by - r // 10), max(1, r - 2), ORANGE, -1, AA)
            cv2.circle(img, (bx - r // 3, by - r // 3), max(1, r // 3), (210, 235, 255), -1, AA)
            if game.hittable():
                pulse = 0.6 + 0.4 * math.sin(now * 18)
                cv2.circle(img, (bx, by), r + 6, scale_color(GOLD, pulse), 2, AA)
                g(bx, by, r + 12, scale_color(GOLD, 0.6))

        if b is not None and b.z > 0:
            draw_ball()
            draw_net(img, v)
        else:
            draw_net(img, v)
            draw_ball()

        # Particles
        alive = []
        for p in self.particles:
            p[6] -= dt
            if p[6] <= 0:
                continue
            p[4] -= 9.8 * dt
            p[0] += p[3] * dt
            p[1] += p[4] * dt
            p[2] += p[5] * dt
            (sx, sy), d = v.pt(p[0], p[1], p[2])
            f = p[6] / p[7]
            r = v.size(p[9], d) * f
            cv2.circle(img, (sx, sy), max(1, int(r)), scale_color(p[8], 0.4 + 0.6 * f), -1, AA)
            g(sx, sy, r * 3, scale_color(p[8], f))
            alive.append(p)
        self.particles = alive

        # Aim helpers in your paddle's plane: the hit-radius ring, and a marker
        # showing where the incoming ball lines up with it.
        coming = b is not None and game.last_hitter != 'PLAYER' and b.z < 0.5
        if show_debug or coming:
            ring = v.poly(circle_facing(px, py, C.PLAYER_Z, game.settings['hit_radius']))
            col = GOLD if game.hittable() else (110, 90, 170)
            for i in range(0, len(ring), 2):   # dashed
                cv2.line(img, tuple(ring[i]), tuple(ring[(i + 1) % len(ring)]), col, 1, AA)
        if coming:
            (gx, gy), d = v.pt(b.x, b.y, C.PLAYER_Z)
            cv2.drawMarker(img, (gx, gy), GOLD if game.hittable() else (180, 160, 200),
                           cv2.MARKER_TILTED_CROSS, 14, 2, AA)

        # Your paddle, see-through so the ball stays visible behind it
        self._draw_paddle(img, glow, px, py, C.PLAYER_Z,
                          12 + self._swing_lean(now - self.swing_t, self.swing_power), PLAYER_RUBBER, alpha=0.6)

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

        self._draw_hud(img, game, clicker, now, meter)

        # Floating popups ("+1", "TOO HARD", ...) anchored in the world
        for pu in self.popups:
            age = now - pu[2]
            f = max(0.0, 1 - age / 1.0)
            (sx, sy), _ = v.pt(*pu[1])
            text(img, pu[0], (sx, sy - age * 70), pu[4] * (1 + 0.3 * f), scale_color(pu[3], f), 2, font=FONT_BOLD)
        self.popups = [pu for pu in self.popups if now - pu[2] < 1.0]

        # Camera preview, bottom-right
        if preview is not None:
            small = cv2.resize(preview, (C.PREVIEW_W, C.PREVIEW_H))
            vx0, vy0 = W - C.PREVIEW_W - 12, H - C.PREVIEW_H - 26
            rounded_rect(img, (vx0 - 4, vy0 - 4, vx0 + C.PREVIEW_W + 4, vy0 + C.PREVIEW_H + 4), 8, (20, 20, 20))
            img[vy0:vy0 + C.PREVIEW_H, vx0:vx0 + C.PREVIEW_W] = small
            rounded_rect(img, (vx0 - 4, vy0 - 4, vx0 + C.PREVIEW_W + 4, vy0 + C.PREVIEW_H + 4), 8, (120, 120, 130), 2)
            text(img, 'CAMERA', (vx0 + 8, vy0 + 22), 0.5, WHITE, 1, anchor='left')

        if info:
            text(img, info, (W // 2, H - 8), 0.45, (170, 170, 180), 1)

        if alert:   # tracking problem: red box under the top bar
            (tw, _), _ = cv2.getTextSize(alert, FONT_BOLD, 0.75, 2)
            y0 = C.TOP_BAR_H + 72
            rounded_rect(img, (W // 2 - tw // 2 - 16, y0, W // 2 + tw // 2 + 16, y0 + 40), 10, (30, 30, 150))
            text(img, alert, (W // 2, y0 + 29), 0.75, WHITE, 2, font=FONT_BOLD, shadow=False)

        if cursor is not None:
            cx, cy = int(cursor[0]), int(cursor[1])
            cv2.circle(img, (cx, cy), 16, MAGENTA, 2, AA)
            cv2.drawMarker(img, (cx, cy), MAGENTA, cv2.MARKER_CROSS, 40, 2, AA)
            if clicker.progress > 0:
                cv2.ellipse(img, (cx, cy), (26, 26), -90, 0, 360 * clicker.progress, GOLD, 5, AA)
        return img

    # ----- HUD
    def _draw_hud(self, img, game, clicker, now, meter):
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
            text(img, who, (W // 2, 215), 1.3, WHITE, 3, font=FONT_BOLD)
            bw = 260
            rounded_rect(img, (W // 2 - bw // 2, 235, W // 2 + bw // 2, 245), 5, (60, 50, 60))
            if frac > 0.03:
                rounded_rect(img, (W // 2 - bw // 2, 235, W // 2 - bw // 2 + int(bw * frac), 245), 5, GOLD)

        # NEW HIGH SCORE banner
        age = now - self.banner_t
        if age < 1.8:
            pop = min(1.0, age / 0.15)
            f = 1.0 if age < 1.3 else max(0.0, 1 - (age - 1.3) / 0.5)
            text(img, 'NEW HIGH SCORE!', (W // 2, 205), (1.0 + 0.6 * pop) * (1 + 0.05 * math.sin(now * 20)),
                 scale_color(GOLD, f), 4, font=FONT_BOLD)

        self._draw_meter(img, game, meter)

    def _draw_meter(self, img, game, level):
        """Power meter on the left edge: live swing power + threshold / min / max marks."""
        s = game.settings
        mh, top = self._meter_h, self._meter_top
        bot = top + mh
        x0, x1 = 22, 38
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
                text(img, label, (x1 + 10, yy + 5), 0.4, color, 1, anchor='left')
        cx = (x0 + x1) // 2
        text(img, 'POWER', (cx + 10, top - 14), 0.45, (200, 180, 200), 1)
        if game.last_swing_power is not None:
            text(img, f'{game.last_swing_power:.2f}', (cx + 10, bot + 24), 0.55, WHITE, 1)
