"""
Ping pong game logic in 3D: ball physics, CPU, serving, scoring, difficulty.

No OpenCV, camera, or motor in here -- main.py feeds in the paddle position
and swing events each frame, so this file can be unit-tested with made-up
inputs (tests/test_game.py).

World coordinates are meters (see config.py):
    x = left/right, y = up, z = along the table (player at -z, CPU at +z)
The player's paddle moves in the plane z = PLAYER_Z, so it has an (x, y)
position -- exactly how your hand moves in front of the webcam.

The game also queues `events` (hits, bounces, lost balls, ...) for the
renderer to turn into sparks, shakes, etc.
"""

import math
import random

import config as C

# Phases
WAITING = 'waiting'   # pause before a serve
PLAYING = 'playing'


class Ball:
    def __init__(self, x, y, z, vx=0.0, vy=0.0, vz=0.0):
        self.x, self.y, self.z = x, y, z
        self.vx, self.vy, self.vz = vx, vy, vz


def paddle_from_camera(point, region=C.CAMERA_PADDLE_REGION):
    """Normalized (mirrored) camera point -> paddle (x, y) in meters."""
    nx, ny = point
    r0x, r0y, r1x, r1y = region
    fx = min(max((nx - r0x) / (r1x - r0x), 0.0), 1.0)
    fy = min(max((ny - r0y) / (r1y - r0y), 0.0), 1.0)
    x0, y0, x1, y1 = C.PADDLE_ZONE
    return (x0 + fx * (x1 - x0), y1 - fy * (y1 - y0))   # image y grows down, world y up


def solve_shot(p0, target, flight_time, clear_net=True):
    """
    Velocity (vx, vy, vz) that takes the ball from p0 to target (x, y, z) in
    `flight_time` seconds under gravity. If that path would clip the net,
    lob it higher (longer flight time) until it clears.
    """
    x0, y0, z0 = p0
    x1, y1, z1 = target
    t = flight_time
    top = C.TABLE_H + C.NET_H + C.BALL_RADIUS + C.NET_CLEARANCE
    for _ in range(20 if clear_net else 1):
        vx, vz = (x1 - x0) / t, (z1 - z0) / t
        vy = (y1 - y0 + 0.5 * C.GRAVITY * t * t) / t
        if vz == 0 or z0 * z1 > 0:                # doesn't cross the net
            break
        tn = -z0 / vz                             # time when z == 0
        if y0 + vy * tn - 0.5 * C.GRAVITY * tn * tn >= top:
            break
        t *= 1.1
    return vx, vy, vz


class Game:
    def __init__(self, difficulty=C.START_DIFFICULTY, server=C.START_SERVER,
                 high_score=0, on_new_high=None, rng=None, hand=C.HAND.upper()):
        self.difficulty = difficulty
        self.server = server               # 'PLAYER' or 'CPU'
        self.hand = hand                   # 'RIGHT' or 'LEFT': which hand the camera tracks
        self.score = 0
        self.high_score = high_score
        self.on_new_high = on_new_high     # called with the new high score
        self.rng = rng or random.Random()

        self.time = 0.0
        self.paddle = (0.0, C.TABLE_H + 0.25)   # (x, y) in the PLAYER_Z plane
        self.cpu_x = 0.0                         # CPU paddle slides left/right
        self.ball = None
        self.phase = WAITING
        self._phase_until = C.SERVE_DELAY_SEC
        self.last_hitter = None            # 'PLAYER', 'CPU', or None (serve toss)
        self.bounces = {'PLAYER': 0, 'CPU': 0}

        self.message = ''                  # short text for the UI ("HIT!", "NET", ...)
        self.message_time = -10.0
        self.message_kind = 'info'         # 'good', 'bad' or 'info' (for the text color)
        self.last_swing_power = None
        self.events = []                   # (kind, data dict) for the renderer

    # ---------- settings (buttons) ----------
    @property
    def settings(self):
        return C.DIFFICULTIES[self.difficulty]

    def toggle_server(self):
        self.server = 'CPU' if self.server == 'PLAYER' else 'PLAYER'
        self._restart_point()

    def cycle_difficulty(self):
        order = C.DIFFICULTY_ORDER
        self.difficulty = order[(order.index(self.difficulty) + 1) % len(order)]
        self._restart_point()

    def toggle_hand(self):
        self.hand = 'LEFT' if self.hand == 'RIGHT' else 'RIGHT'
        self._restart_point()

    @property
    def serve_in(self):
        """Seconds until the next serve (only meaningful while WAITING)."""
        return max(0.0, self._phase_until - self.time)

    def pop_events(self):
        events, self.events = self.events, []
        return events

    def reset_high_score(self):
        self.high_score = 0
        if self.on_new_high:
            self.on_new_high(0)

    def hittable(self):
        b = self.ball
        return (b is not None and self.last_hitter != 'PLAYER'
                and C.PLAYER_REACH_Z <= b.z <= C.PLAYER_HIT_Z)

    # ---------- per-frame update ----------
    def update(self, dt, paddle_xy=None, swing_power=None):
        """
        dt:          seconds since the last frame
        paddle_xy:   paddle (x, y) in meters, or None if not seen
        swing_power: power of a swing that just finished, or None
        """
        self.time += dt
        if paddle_xy is not None:
            x0, y0, x1, y1 = C.PADDLE_ZONE
            self.paddle = (min(max(paddle_xy[0], x0), x1), min(max(paddle_xy[1], y0), y1))

        # CPU paddle drifts toward the ball's x when the ball is coming its way
        target = self.ball.x if self.ball is not None and self.last_hitter == 'PLAYER' else 0.0
        self.cpu_x += (target - self.cpu_x) * min(1.0, dt * 6)

        if self.phase == WAITING:
            if self.time >= self._phase_until:
                self._serve()
            elif swing_power is not None:
                self.last_swing_power = swing_power
                self._emit('swing', power=swing_power)
            return

        if swing_power is not None:
            self._player_swing(swing_power)

        # Small physics steps so bounces / net hits aren't skipped at low FPS.
        steps = max(1, math.ceil(dt / (1 / 240)))
        h = dt / steps
        for _ in range(steps):
            if self.phase != PLAYING:
                break
            self._step(h)

    # ---------- internals ----------
    def _say(self, text, kind='info'):
        self.message = text
        self.message_kind = kind
        self.message_time = self.time

    def _emit(self, kind, **data):
        self.events.append((kind, data))

    def _restart_point(self):
        self.score = 0       # also on setting changes, so a rally can't span difficulties
        self.ball = None
        self.phase = WAITING
        self._phase_until = self.time + C.SERVE_DELAY_SEC

    def _serve(self):
        self.phase = PLAYING
        self.bounces = {'PLAYER': 0, 'CPU': 0}
        if self.server == 'PLAYER':
            # Toss the ball up just in front of the player, in line with the paddle.
            x = min(max(self.paddle[0], -0.6), 0.6)
            self.ball = Ball(x, C.TABLE_H + 0.2, C.PLAYER_Z + 0.25, 0.0, 2.6, 0.0)
            self.last_hitter = None
            self._say('YOUR SERVE - swing!')
            self._emit('serve', by='PLAYER')
        else:
            self.ball = Ball(self.rng.uniform(-0.4, 0.4), C.TABLE_H + 0.3, C.CPU_HIT_Z)
            self.cpu_x = self.ball.x
            self._cpu_hit()
            self._say('CPU SERVES')
            self._emit('serve', by='CPU')

    def _point_over(self, reason):
        b = self.ball
        self._emit('lost', x=b.x, y=b.y, z=b.z, reason=reason, score=self.score)
        self._say(reason, 'bad')
        self._restart_point()

    def _player_swing(self, power):
        self.last_swing_power = power
        self._emit('swing', power=power)
        if not self.hittable():
            self._say('swing - ball not in reach')
            return
        b = self.ball
        dist = math.hypot(b.x - self.paddle[0], b.y - self.paddle[1])
        if dist > self.settings['hit_radius']:
            self._say(f'MISS - {100 * dist:.0f} cm away', 'bad')
            self._emit('whiff', dist=dist)
            return
        self._player_hit(power)

    def _player_hit(self, power):
        s = self.settings
        b = self.ball
        self.last_hitter = 'PLAYER'
        self.bounces = {'PLAYER': 0, 'CPU': 0}

        # Hitting the ball off-center steers it: paddle left of the ball -> goes right.
        off = (b.x - self.paddle[0]) / s['hit_radius']
        aim_x = min(max(off * C.AIM_STRENGTH, -C.TABLE_HALF_W + 0.1), C.TABLE_HALF_W - 0.1)

        if power < s['min_power']:
            # Too soft: straight into the net.
            b.vx, b.vy, b.vz = solve_shot((b.x, b.y, b.z), (aim_x, C.TABLE_H + C.NET_H / 2, 0.02),
                                          s['player_flight_slow'], clear_net=False)
            self._say(f'too soft ({power:.2f})', 'bad')
            quality = 'soft'
        else:
            if power > s['max_power']:
                # Too hard: sails past the end of the table.
                target_z, flight = C.TABLE_HALF_L + 0.9, s['player_flight_fast']
                self._say(f'too hard ({power:.2f})', 'bad')
                quality = 'hard'
            else:
                # In range: harder = deeper and faster.
                f = (power - s['min_power']) / (s['max_power'] - s['min_power'])
                target_z = 0.35 + f * (C.TABLE_HALF_L - 0.5)
                flight = s['player_flight_slow'] + f * (s['player_flight_fast'] - s['player_flight_slow'])
                self._say(f'HIT! power {power:.2f}', 'good')
                quality = 'good'
            b.vx, b.vy, b.vz = solve_shot((b.x, b.y, b.z), (aim_x, C.TABLE_H + C.BALL_RADIUS, target_z), flight)

        # Score counts every hit you make; it's cleared when the ball is lost.
        self.score += 1
        self._emit('hit', x=b.x, y=b.y, z=b.z, power=power, quality=quality, score=self.score)
        if self.score > self.high_score:
            self.high_score = self.score
            self._emit('new_high', score=self.score)
            if self.on_new_high:
                self.on_new_high(self.high_score)

    def _cpu_hit(self):
        b = self.ball
        self.last_hitter = 'CPU'
        self.bounces = {'PLAYER': 0, 'CPU': 0}
        self._emit('cpu_hit', x=b.x, y=b.y, z=b.z)
        target = (self.rng.uniform(-0.55, 0.55), C.TABLE_H + C.BALL_RADIUS,
                  self.rng.uniform(-C.TABLE_HALF_L + 0.2, -0.4))
        b.vx, b.vy, b.vz = solve_shot((b.x, b.y, b.z), target, self.settings['cpu_flight'])

    def _step(self, h):
        b = self.ball
        pz, py = b.z, b.y
        b.vy -= C.GRAVITY * h
        b.x += b.vx * h
        b.y += b.vy * h
        b.z += b.vz * h
        R = C.BALL_RADIUS
        over_table = abs(b.x) <= C.TABLE_HALF_W and abs(b.z) <= C.TABLE_HALF_L

        # Net: crossed z = 0 below the top of the net
        if (pz * b.z <= 0 and pz != b.z and abs(b.x) <= C.TABLE_HALF_W + C.NET_OVERHANG
                and C.TABLE_H - R < b.y - R < C.TABLE_H + C.NET_H):
            self._point_over('NET')
            return

        # Table bounce
        surface = C.TABLE_H + R
        if py > surface >= b.y and b.vy < 0 and over_table:
            b.y = surface
            b.vy = -b.vy * C.RESTITUTION
            side = 'PLAYER' if b.z < 0 else 'CPU'
            self.bounces[side] += 1
            self._emit('bounce', x=b.x, y=C.TABLE_H, z=b.z)
            if self.last_hitter == 'PLAYER' and side == 'PLAYER':
                self._point_over('bounced on your side')
                return
            if self.last_hitter == 'CPU' and self.bounces['PLAYER'] >= 2:
                self._point_over('MISSED - double bounce')
                return

        # CPU returns any ball that bounced once on its side
        if self.last_hitter == 'PLAYER' and self.bounces['CPU'] == 1 and b.z >= C.CPU_HIT_Z:
            self.cpu_x = b.x
            self._cpu_hit()
            return

        # Floor, behind you, or way off the table
        if b.y < R or b.z < C.PLAYER_REACH_Z - 0.5 or b.z > C.CPU_Z + 1.5 or abs(b.x) > 3:
            if self.last_hitter == 'PLAYER':
                self._point_over('OUT - off the table')
            else:
                self._point_over('MISSED')
