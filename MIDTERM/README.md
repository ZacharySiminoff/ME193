# MIDTERM — LEGO Motor Ping Pong

Hold a LEGO Double Motor like a paddle and play ping pong against the CPU.

- **Where the paddle is**: the camera + MediaPipe Pose track the end of your
  arm (index-finger landmark, falls back to the wrist). That point moves the
  red paddle inside YOUR ZONE at your end of the table.
- **Either side / either hand**: the `HAND: RIGHT/LEFT` button (or `l`)
  puts you on that side of the table and tracks that hand. The camera view
  is mirrored, so right-handed players are on the right and left-handed on
  the left. Default is `HAND` in `config.py` (or `--hand left`).
- **Hitting**: swing the motor. Its accelerometer detects the swing and how
  hard it was. A swing only **hits** if the paddle is close enough to the ball
  (the dashed circle around the paddle) while the ball is on your side
  (it gets a pulsing gold ring when it's hittable).
- **How hard**: power ~1.0 is a normal swing. Too soft → the ball hits the
  net. Too hard → it flies off the end of the table. In between, harder =
  deeper and faster. The meter on the left shows live power; red lines are
  the min/max for the current difficulty.
- **Score** = hits in a row. Any lost ball (net, out, missed, double
  bounce) resets it to 0. **High score** is kept in `highscore.json` and
  published over MQTT.
- **Buttons** (top left): `SERVE: YOU/CPU`, `LEVEL: EASY/MEDIUM/HARD` and `HAND: RIGHT/LEFT`.
  Hold an AprilTag (36h11 family) in front of the camera — it becomes the
  magenta cursor — and keep it on a button for 1 s to click. Mouse clicks
  work too. Changing a setting restarts the rally.
- The small camera view (bottom corner on the CPU's side) shows the
  skeleton, the arm-end dot (red) and detected tags (magenta outline + ID).
- **Effects**: sparks and screen shake on hits (bigger for harder swings),
  ripples on bounces, a ball trail and shadow, "+1" / "TOO HARD" popups,
  a red flash when the ball is lost, a NEW HIGH SCORE banner and a serve
  countdown. All drawing is in `ui.py` (`Renderer`); the static arena is
  cached so it stays fast (~3 ms per frame).

## Setup

```bash
cd MIDTERM
python3.12 -m venv .venv          # mediapipe 0.10.14 needs Python <= 3.12
source .venv/bin/activate
pip install -r requirements.txt
```

Set your motor's card in `config.py` (`CARD_COLOR`, `CARD_SERIAL`) and find
your camera with `python list_cameras.py` (`CAMERA_INDEX`).

## Run order (first time)

| Step | Command | What to check |
|---|---|---|
| 1 | `python -m pytest tests/` | game + swing logic pass (no hardware needed) |
| 2 | `python accel_tuner.py` | hold still, make ~10 normal swings, press **s** → writes `calibration.json` |
| 3 | `python pose_check.py` | "pose: found", red dot on the right hand, paddle covers the zone |
| 4 | `python tag_check.py --make-tag 0` then `python tag_check.py` | print the tag; the cross follows it and dwell-clicks print `CLICK` |
| 5 | `python mqtt_check.py` (separate terminal) | shows the high score when the game starts / beats it |
| 6 | `python main.py` | play |

## Game controls

| Key | Action |
|---|---|
| `q` / Esc | quit |
| `s` | switch who serves |
| `d` | cycle difficulty |
| `l` | switch hand / side (right ⇄ left) |
| `h` | hide/show the hit-radius circle |
| `r` `r` | reset high score (press twice within 2 s) |
| `space`, `1`, `2`, `3` | fake swings in `--no-motor` mode (normal, too soft, normal, too hard) |

Testing without hardware:

```bash
python main.py --no-motor              # camera for the paddle, keyboard swings
python main.py --no-camera             # motor swings, mouse moves the paddle
python main.py --no-motor --no-camera  # no hardware at all
```

If the motor or camera can't be opened, the game says so in the terminal and
falls back to keyboard / mouse automatically.

## Tuning (everything is in `config.py`)

- **Swings too weak/strong in game** → rerun `accel_tuner.py`. It sets
  `ACCEL_GAIN` so your median swing = power 1.0, and `SWING_THRESHOLD`
  (0.35 = 35% of a normal swing counts as a swing). `calibration.json`
  overrides the values in `config.py`; delete it to go back to the defaults.
- **Can't reach the edges of the paddle box** → shrink `CAMERA_PADDLE_REGION`.
  **Paddle jittery** → lower `PADDLE_SMOOTHING`.
- **Holding the motor in the other hand** → HAND button / `l` in game, or
  `HAND = 'left'` in config (or `--hand left`).
- **Tag cursor can't reach the buttons** → shrink `CAMERA_TAG_REGION`.
  Only one tag should count → set `TAG_ID`.
- **Game too hard/easy** → edit `DIFFICULTIES` (hit radius, ball speed,
  power window).

### How swing power is computed (`swing.py`)

Gravity is tracked as a slow moving average of the accelerometer vector. A
swing is fast, so it shows up as `|accel − gravity|`, which works whichever
way the motor is held. `power = |accel − gravity| × ACCEL_GAIN`. A swing is
the peak power from when it crosses `SWING_THRESHOLD` until it drops back
(max 0.15 s), then swings are ignored for 0.35 s so one swing = one hit.

## Test data

- `data/sample_swings.csv` — a made-up recording (3 s rest, then swings of
  0.5×, 1×, 1.7× normal), made by `tests/make_sample_data.py`. Used by
  `tests/test_swing.py` and `python accel_tuner.py --replay data/sample_swings.csv`.
- Every live `accel_tuner.py` session saves its raw samples to
  `data/swings_<time>.csv`. Replay one to re-tune without the motor:
  `python accel_tuner.py --replay data/swings_....csv --save`.
- `tests/test_game.py` — hits, misses, net, out, double bounce, long
  rallies on every difficulty, high-score callbacks, settings buttons.

## MQTT

- Broker: `test.mosquitto.org:1883` (via `mqttlib.py`, same wrapper as `MQTT/`).
- Topic: **`ME193/Rogers/Zack`**.
- Payload: the high score as a float string, e.g. `"12.0"`. It is sent when
  the game starts and every time the high score goes up (or is reset to
  `"0.0"`), with `retain=True` so new subscribers get the current value
  right away.
- Listen: `python mqtt_check.py`, or `python "../MQTT Listener/listener.py" ME193/Rogers/Zack`.
  Send a test value: `python mqtt_check.py --send 5`.

## Files

| File | What it is |
|---|---|
| `main.py` | the game loop |
| `game.py` | game rules + physics (no hardware; unit tested) |
| `swing.py` | accelerometer → swing power (no hardware; unit tested) |
| `sensors.py` | camera + pose + AprilTag (`Vision`), motor (`Paddle`), keyboard stand-in (`FakePaddle`) |
| `ui.py` | `Renderer` (arena, effects, HUD) + dwell-click buttons |
| `config.py` | all settings |
| `accel_tuner.py`, `pose_check.py`, `tag_check.py`, `mqtt_check.py`, `list_cameras.py` | calibration / check tools |
| `mqttlib.py` | MQTT wrapper (copied from `MQTT/`) |
