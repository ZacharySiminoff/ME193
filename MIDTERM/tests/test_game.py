"""
Game logic tests with made-up inputs (no camera / motor needed).

    .venv/bin/python -m pytest tests/
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C  # noqa: E402
from game import PLAYING, WAITING, Game, solve_shot  # noqa: E402

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
    assert run_until(game, lambda g: g._hittable())


def swing_at_ball(game, power=GOOD_POWER, offset=(0, 0)):
    b = game.ball
    game.update(DT, paddle_xy=(b.x + offset[0], b.y + offset[1]), swing_power=power)


def test_solve_shot_lands_on_target():
    vx, vy = solve_shot(200, 450, 800, C.TABLE_Y, 1.0)
    # Walk the arc until it comes back down to table height
    x, y, t = 200, 450, 0.0
    while t < 5:
        t += 0.001
        vy += C.GRAVITY * 0.001
        x += vx * 0.001
        y += vy * 0.001
        if y >= C.TABLE_Y and vy > 0:
            break
    assert abs(x - 800) < 5


def test_solve_shot_clears_net():
    # A low, fast shot would hit the net; solve_shot must lob it over.
    x0, y0 = 300, C.TABLE_Y - 5
    vx, vy = solve_shot(x0, y0, 700, C.TABLE_Y, 0.3)
    tn = (C.NET_X - x0) / vx
    y_net = y0 + vy * tn + 0.5 * C.GRAVITY * tn * tn
    assert y_net < C.TABLE_Y - C.NET_H


def test_player_serve_hit_scores_and_cpu_returns():
    g = make_game(server='PLAYER')
    wait_for_serve(g)
    swing_at_ball(g)
    assert g.score == 1 and g.high_score == 1
    assert g.last_hitter == 'PLAYER' and g.ball.vx > 0
    assert run_until(g, lambda g: g.last_hitter == 'CPU'), 'CPU should return the ball'
    assert g.ball.vx < 0


def test_swing_too_far_away_is_a_miss_and_resets():
    g = make_game(server='PLAYER')
    wait_for_serve(g)
    swing_at_ball(g, offset=(0, C.DIFFICULTIES['MEDIUM']['hit_radius'] + 50))
    assert g.score == 0 and g.message.startswith('MISS')
    assert run_until(g, lambda g: g.phase == WAITING)  # ball falls to the floor
    assert g.message == 'MISSED'


def test_swing_when_ball_out_of_reach_does_nothing():
    g = make_game(server='CPU')
    wait_for_serve(g)
    assert not g._hittable()        # ball is still on the CPU's side
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
        g = make_game(server='CPU', difficulty=level)
        wait_for_serve(g)
        for _ in range(3):
            wait_hittable(g)
            swing_at_ball(g)
        assert g.score == 3, level


def test_toggle_server_and_difficulty():
    g = make_game(server='PLAYER', difficulty='EASY')
    g.toggle_server()
    assert g.server == 'CPU' and g.phase == WAITING
    g.cycle_difficulty()
    assert g.difficulty == 'MEDIUM'
    g.cycle_difficulty()
    g.cycle_difficulty()
    assert g.difficulty == 'EASY'
    wait_for_serve(g)
    assert g.last_hitter == 'CPU' and g.ball.vx < 0


def test_changing_settings_resets_score_not_high_score():
    g = make_game(server='PLAYER')
    wait_for_serve(g)
    swing_at_ball(g)
    assert g.score == 1
    g.cycle_difficulty()
    assert g.score == 0 and g.high_score == 1


def test_paddle_is_clamped_to_zone():
    g = make_game()
    g.update(DT, paddle_xy=(5000, -5000))
    x0, y0, x1, y1 = C.PADDLE_ZONE
    assert g.paddle == (x1, y0)


def test_reset_high_score():
    highs = []
    g = make_game(high_score=7, on_new_high=highs.append)
    g.reset_high_score()
    assert g.high_score == 0 and highs == [0]
