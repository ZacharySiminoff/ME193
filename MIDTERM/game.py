"""
Ping pong game logic: ball physics, CPU, serving, scoring, difficulty.

No OpenCV, camera, or motor in here -- main.py feeds in the paddle position
and swing events each frame, so this file can be unit-tested with made-up
inputs (tests/test_game.py).

Side view, pixels, y grows downward. In game coordinates the player is
ALWAYS on the left and the CPU on the right; when the player picks the right
side (left/right handed), only the drawing and the paddle input are mirrored
with mirror_x(). That keeps the physics and tests the same for both sides.

The game also queues `events` (hits, bounces, lost balls, ...) for the
renderer to turn into sparks, sounds, shakes, etc.
"""

import math
import random

import config as C

# Phases
WAITING = 'waiting'   # pause before a serve
PLAYING = 'playing'


def mirror_x(x, side):
    """Game x <-> screen x. Same function both ways."""
    return x if side == 'LEFT' else C.WIDTH - x


class Ball:
    def __init__(self, x, y, vx=0.0, vy=0.0):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy


def solve_shot(x0, y0, x1, y1, flight_time, clear_net=True):
    """
    Velocity that takes the ball from (x0, y0) to (x1, y1) in `flight_time`
    seconds under gravity. If that path would clip the net, lob it higher
    (longer flight time) until it clears.
    """
    t = flight_time
    for _ in range(20 if clear_net else 1):
        vx = (x1 - x0) / t
        vy = (y1 - y0 - 0.5 * C.GRAVITY * t * t) / t
        if vx == 0 or (C.NET_X - x0) / vx < 0:   # path never crosses the net
            return vx, vy
        tn = (C.NET_X - x0) / vx                  # time when x == net
        if tn > t:                                # lands before reaching the net
            return vx, vy
        y_net = y0 + vy * tn + 0.5 * C.GRAVITY * tn * tn
        if y_net <= C.TABLE_Y - C.NET_H - C.NET_CLEARANCE:
            return vx, vy
        t *= 1.1
    return vx, vy


class Game:
    def __init__(self, difficulty=C.START_DIFFICULTY, server=C.START_SERVER,
                 high_score=0, on_new_high=None, rng=None, side=C.START_SIDE):
        self.difficulty = difficulty
        self.side = side                   # 'LEFT' or 'RIGHT': where the player is on screen
        self.server = server               # 'PLAYER' or 'CPU'
        self.score = 0
        self.high_score = high_score
        self.on_new_high = on_new_high     # called with the new high score
        self.rng = rng or random.Random()

        self.time = 0.0
        self.paddle = (C.PADDLE_ZONE[0] + 100, C.TABLE_Y - 60)
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

    def toggle_side(self):
        self.side = 'RIGHT' if self.side == 'LEFT' else 'LEFT'
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

    # ---------- per-frame update ----------
    def update(self, dt, paddle_xy=None, swing_power=None):
        """
        dt:          seconds since the last frame
        paddle_xy:   paddle position in screen pixels, or None if not seen
        swing_power: power of a swing that just finished, or None
        """
        self.time += dt
        if paddle_xy is not None:
            x0, y0, x1, y1 = C.PADDLE_ZONE
            self.paddle = (min(max(paddle_xy[0], x0), x1), min(max(paddle_xy[1], y0), y1))

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
            # Toss the ball straight up in front of the player.
            self.ball = Ball(C.TABLE_LEFT - 70, C.TABLE_Y - 40, 0.0, -650.0)
            self.last_hitter = None
            self._say('YOUR SERVE - swing!')
            self._emit('serve', by='PLAYER')
        else:
            self.ball = Ball(C.CPU_HIT_X, C.TABLE_Y - 100)
            self._cpu_hit()
            self._say('CPU SERVES')
            self._emit('serve', by='CPU')

    def _point_over(self, reason):
        b = self.ball
        self._emit('lost', x=b.x, y=b.y, reason=reason, score=self.score)
        self._say(reason, 'bad')
        self._restart_point()

    def _hittable(self):
        b = self.ball
        return b is not None and self.last_hitter != 'PLAYER' and b.x <= C.PLAYER_HIT_X

    def _player_swing(self, power):
        self.last_swing_power = power
        self._emit('swing', power=power)
        if not self._hittable():
            self._say('swing - ball not in reach')
            return
        b = self.ball
        dist = math.hypot(b.x - self.paddle[0], b.y - self.paddle[1])
        if dist > self.settings['hit_radius']:
            self._say(f'MISS - {int(dist)}px away', 'bad')
            self._emit('whiff', dist=dist)
            return
        self._player_hit(power)

    def _player_hit(self, power):
        s = self.settings
        b = self.ball
        self.last_hitter = 'PLAYER'
        self.bounces = {'PLAYER': 0, 'CPU': 0}

        if power < s['min_power']:
            # Too soft: aim straight into the net.
            b.vx, b.vy = solve_shot(b.x, b.y, C.NET_X + 2, C.TABLE_Y - C.NET_H / 2,
                                    s['player_flight_slow'], clear_net=False)
            self._say(f'too soft ({power:.2f})', 'bad')
            quality = 'soft'
        elif power > s['max_power']:
            # Too hard: sails past the end of the table.
            target_x, flight = C.TABLE_RIGHT + 250, s['player_flight_fast']
            self._say(f'too hard ({power:.2f})', 'bad')
            quality = 'hard'
        else:
            # In range: harder = deeper and faster.
            f = (power - s['min_power']) / (s['max_power'] - s['min_power'])
            target_x = C.NET_X + 40 + f * (C.TABLE_RIGHT - C.NET_X - 70)
            flight = s['player_flight_slow'] + f * (s['player_flight_fast'] - s['player_flight_slow'])
            self._say(f'HIT! power {power:.2f}', 'good')
            quality = 'good'
        if power >= s['min_power']:
            b.vx, b.vy = solve_shot(b.x, b.y, target_x, C.TABLE_Y, flight)

        # Score counts every hit you make; it's cleared when the ball is lost.
        self.score += 1
        self._emit('hit', x=b.x, y=b.y, power=power, quality=quality, score=self.score)
        if self.score > self.high_score:
            self.high_score = self.score
            self._emit('new_high', score=self.score)
            if self.on_new_high:
                self.on_new_high(self.high_score)

    def _cpu_hit(self):
        b = self.ball
        self.last_hitter = 'CPU'
        self.bounces = {'PLAYER': 0, 'CPU': 0}
        self._emit('cpu_hit', x=b.x, y=b.y)
        target_x = self.rng.uniform(C.TABLE_LEFT + 60, C.NET_X - 80)
        b.vx, b.vy = solve_shot(b.x, b.y, target_x, C.TABLE_Y, self.settings['cpu_flight'])

    def _step(self, h):
        b = self.ball
        px, py = b.x, b.y
        b.vy += C.GRAVITY * h
        b.x += b.vx * h
        b.y += b.vy * h

        # Net: crossed the net's x while below the top of the net
        if ((px - C.NET_X) * (b.x - C.NET_X) <= 0 and px != b.x
                and C.TABLE_Y - C.NET_H < b.y <= C.TABLE_Y):
            self._point_over('NET')
            return

        # Table bounce
        if py < C.TABLE_Y <= b.y and b.vy > 0 and C.TABLE_LEFT <= b.x <= C.TABLE_RIGHT:
            b.y = C.TABLE_Y
            b.vy = -b.vy * C.RESTITUTION
            side = 'PLAYER' if b.x < C.NET_X else 'CPU'
            self.bounces[side] += 1
            self._emit('bounce', x=b.x, y=b.y)
            if self.last_hitter == 'PLAYER' and side == 'PLAYER':
                self._point_over('bounced on your side')
                return
            if self.last_hitter == 'CPU' and self.bounces['PLAYER'] >= 2:
                self._point_over('MISSED - double bounce')
                return

        # CPU returns any ball that bounced once on its side
        if self.last_hitter == 'PLAYER' and self.bounces['CPU'] == 1 and b.x >= C.CPU_HIT_X:
            self._cpu_hit()
            return

        # Off the table / off screen
        if b.y > C.FLOOR_Y or b.x < -50 or b.x > C.WIDTH + 50:
            if self.last_hitter == 'PLAYER':
                self._point_over('OUT - off the table')
            else:
                self._point_over('MISSED')
