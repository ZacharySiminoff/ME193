"""
Game logic tests with made-up inputs (no camera / motor needed).

    .venv/bin/python -m pytest tests/
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C  # noqa: E402
from game import PLAYING, WAITING, Game, paddle_from_camera, solve_shot  # noqa: E402

DT = 1 / 60
GOOD_POWER = 1.0


def make_game(**kw):
    kw.setdefault('rng', random.Random(0))
    return Game(**kw)


def run_until(game, cond, max_sec=10, paddle=None):
    """Step the game until cond(game) is true. Returns True if it happened."""
    for _ in range(int(max_sec / DT)):
        game.update(DT, paddle_xy=paddle)
        if cond(game):
            return True
    return False


def wait_for_serve(game):
    assert run_until(game, lambda g: g.phase == PLAYING)


def wait_hittable(game):
    assert run_until(game, lambda g: g.hittable())


def swing_at_ball(game, power=GOOD_POWER, offset=(0, 0)):
    b = game.ball
    game.update(DT, paddle_xy=(b.x + offset[0], b.y + offset[1]), swing_power=power)


def fly(p0, v, until_y, dt=0.0005):
    """Integrate a shot until it comes back down to until_y. Returns (x, z)."""
    (x, y, z), (vx, vy, vz) = p0, v
    for _ in range(20000):
        vy -= C.GRAVITY * dt
        x, y, z = x + vx * dt, y + vy * dt, z + vz * dt
        if y <= until_y and vy < 0:
            return x, z
    raise AssertionError('never landed')


def test_solve_shot_lands_on_target():
    p0, target = (0.3, 1.0, -1.6), (-0.4, C.TABLE_H, 0.9)
    x, z = fly(p0, solve_shot(p0, target, 1.0), C.TABLE_H)
    assert abs(x - target[0]) < 0.02 and abs(z - target[2]) < 0.02


def test_solve_shot_clears_net():
    # A low, fast shot would hit the net; solve_shot must lob it over.
    p0 = (0.0, C.TABLE_H + 0.05, -1.0)
    vx, vy, vz = solve_shot(p0, (0.0, C.TABLE_H, 0.6), 0.3)
    tn = -p0[2] / vz
    y_net = p0[1] + vy * tn - 0.5 * C.GRAVITY * tn * tn
    assert y_net > C.TABLE_H + C.NET_H + C.BALL_RADIUS


def test_paddle_from_camera_maps_corners():
    r = C.CAMERA_PADDLE_REGION
    x0, y0, x1, y1 = C.PADDLE_ZONE
    assert paddle_from_camera((r[0], r[1])) == (x0, y1)    # top-left of camera = high and left
    assert paddle_from_camera((r[2], r[3])) == (x1, y0)
    assert paddle_from_camera((-5, 5)) == (x0, y0)          # clamped


def test_player_serve_hit_scores_and_cpu_returns():
    g = make_game(server='PLAYER')
    wait_for_serve(g)
    swing_at_ball(g)
    assert g.score == 1 and g.high_score == 1
    assert g.last_hitter == 'PLAYER' and g.ball.vz > 0
    assert run_until(g, lambda g: g.last_hitter == 'CPU'), 'CPU should return the ball'
    assert g.ball.vz < 0


def test_swing_too_far_away_is_a_miss_and_resets():
    g = make_game(server='PLAYER')
    wait_for_serve(g)
    swing_at_ball(g, offset=(C.DIFFICULTIES['MEDIUM']['hit_radius'] + 0.1, 0))
    assert g.score == 0 and g.message.startswith('MISS')
    assert run_until(g, lambda g: g.phase == WAITING)  # ball falls to the floor
    assert g.message == 'MISSED'


def test_swing_when_ball_out_of_reach_does_nothing():
    g = make_game(server='CPU')
    wait_for_serve(g)
    assert not g.hittable()        # ball is still on the CPU's side
    g.update(DT, paddle_xy=g.paddle, swing_power=GOOD_POWER)
    assert g.score == 0 and g.phase == PLAYING


def test_too_soft_hits_net():
    g = make_game(server='PLAYER')
    wait_for_serve(g)
    swing_at_ball(g, power=C.DIFFICULTIES['MEDIUM']['min_power'] - 0.1)
    assert g.score == 1
    assert run_until(g, lambda g: g.phase == WAITING)
    assert g.message == 'NET'
    assert g.score == 0 and g.high_score == 1   # high score survives the reset


def test_too_hard_goes_off_the_table():
    g = make_game(server='PLAYER')
    wait_for_serve(g)
    swing_at_ball(g, power=C.DIFFICULTIES['MEDIUM']['max_power'] + 0.5)
    assert run_until(g, lambda g: g.phase == WAITING)
    assert g.message.startswith('OUT')
    assert g.score == 0 and g.high_score == 1


def test_off_center_hit_aims_sideways():
    r = C.DIFFICULTIES['MEDIUM']['hit_radius']
    for offset, sign in ((-0.8 * r, 1), (0.8 * r, -1)):   # paddle left of ball -> ball goes right
        g = make_game(server='PLAYER')
        wait_for_serve(g)
        swing_at_ball(g, offset=(offset, 0))
        assert g.ball.vx * sign > 0


def test_missing_cpu_shot_resets_score():
    g = make_game(server='CPU')
    wait_for_serve(g)
    assert run_until(g, lambda g: g.phase == WAITING)  # never swing
    assert g.message.startswith('MISSED') and g.score == 0


def test_long_rally_and_high_score_callback():
    highs = []
    g = make_game(server='CPU', on_new_high=highs.append)
    wait_for_serve(g)
    for _ in range(5):
        wait_hittable(g)
        swing_at_ball(g)
    assert g.score == 5
    assert highs == [1, 2, 3, 4, 5]

    # Lose the point, then make a shorter rally: no new-high callbacks.
    assert run_until(g, lambda g: g.phase == WAITING)
    assert g.score == 0 and g.high_score == 5
    wait_for_serve(g)
    wait_hittable(g)
    swing_at_ball(g)
    assert g.score == 1 and highs == [1, 2, 3, 4, 5]


def test_every_difficulty_is_playable():
    for level in C.DIFFICULTY_ORDER:
        for seed in range(5):
            g = make_game(server='CPU', difficulty=level, rng=random.Random(seed))
            wait_for_serve(g)
            for _ in range(4):
                wait_hittable(g)
                swing_at_ball(g)
            assert g.score == 4, (level, seed)


def test_ball_stays_in_reach_long_enough():
    # After the CPU's shot bounces on your side there must be time to swing.
    for level in C.DIFFICULTY_ORDER:
        g = make_game(server='CPU', difficulty=level)
        wait_for_serve(g)
        wait_hittable(g)
        frames = 0
        while g.hittable():
            g.update(DT)
            frames += 1
        assert frames * DT > 0.25, level


def test_toggle_server_difficulty_and_hand():
    g = make_game(server='PLAYER', difficulty='EASY', hand='RIGHT')
    g.toggle_server()
    assert g.server == 'CPU' and g.phase == WAITING
    g.cycle_difficulty()
    assert g.difficulty == 'MEDIUM'
    g.cycle_difficulty()
    g.cycle_difficulty()
    assert g.difficulty == 'EASY'
    g.toggle_hand()
    assert g.hand == 'LEFT'
    wait_for_serve(g)
    assert g.last_hitter == 'CPU' and g.ball.vz < 0


def test_changing_settings_resets_score_not_high_score():
    g = make_game(server='PLAYER')
    wait_for_serve(g)
    swing_at_ball(g)
    assert g.score == 1
    g.toggle_hand()
    assert g.score == 0 and g.high_score == 1 and g.phase == WAITING


def test_paddle_is_clamped_to_zone():
    g = make_game()
    g.update(DT, paddle_xy=(50, -50))
    x0, y0, x1, y1 = C.PADDLE_ZONE
    assert g.paddle == (x1, y0)


def test_reset_high_score():
    highs = []
    g = make_game(high_score=7, on_new_high=highs.append)
    g.reset_high_score()
    assert g.high_score == 0 and highs == [0]


def test_events_for_renderer():
    g = make_game(server='PLAYER')
    wait_for_serve(g)
    swing_at_ball(g)
    kinds = [k for k, _ in g.pop_events()]
    assert kinds[:4] == ['serve', 'swing', 'hit', 'new_high']
    assert g.pop_events() == []          # popping clears the queue
    assert run_until(g, lambda g: g.last_hitter == 'CPU')
    kinds = [k for k, _ in g.pop_events()]
    assert 'bounce' in kinds and 'cpu_hit' in kinds
