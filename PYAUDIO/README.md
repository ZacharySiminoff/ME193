# PYAUDIO

Listens to the microphone and drives a robot running
[`mqtt_double_motor_game.py`](https://github.com/JulianGM9/ME193-AIRobotics/blob/main/MQTT_scripts/mqtt_double_motor_game.py)
over MQTT: pitch steers, volume/tonality gates whether a tone counts as
deliberate, and a couple of reserved high/low tones fire one-shot
`win`/`lose` events. A live scrolling spectrogram shows what's being
picked up, with the detected peak marked on it.

## Files

- `main.py` — the robot script. Asks whether you're the `shooter` or
  `goalie`, announces that role on `ME193/Julian`, waits for a `"start"`
  message on `ME193/Rogers`, calibrates the ambient noise floor, then
  opens a live scrolling spectrogram window and runs the mic → pitch →
  drive-command → MQTT loop. The spectrogram is annotated with
  translucent stripes for the LOSE/STEER/WIN frequency bands, a dashed
  line at the "straight ahead" pitch, and a red dot marking the detected
  peak frequency each chunk; the plot title shows the exact payload just
  published.
- `send_start.py` — test helper that publishes `"start"` to `ME193/Rogers`
  so you can unblock a waiting `main.py` without a separate referee script.
- `mqttlib.py` — small wrapper around `paho-mqtt` (connect/publish/subscribe)
  shared with the other MQTT-based scripts in this repo.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS also needs the PortAudio system library for `pyaudio` to build:

```bash
brew install portaudio
```

## Usage

1. Start the robot script and answer the role prompt:

   ```bash
   python main.py
   ```

   It will connect to MQTT, announce its role on `ME193/Julian`, print
   `Waiting for 'start' on 'ME193/Rogers'...`, and block there.

2. In another terminal, trigger the start:

   ```bash
   python send_start.py
   ```

   `main.py` should unblock, calibrate against ambient noise for about a
   second, open the spectrogram window, and start publishing to
   `ME193/Julian` every chunk. Close the plot window (or Ctrl+C in the
   terminal) to stop.

## MQTT protocol (matches `mqtt_double_motor_game.py`)

- `ME193/Rogers` — local-only "start" gate. This script waits here for
  the literal message `"start"` before it begins processing audio; the
  receiving robot doesn't listen on this topic at all.
- `ME193/Julian` — everything the receiving robot actually listens to:
  - the initial role (`"shooter"` or `"goalie"`)
  - `"win"` / `"lose"` (sent once each time a WIN/LOSE tone starts)
  - drive commands as a stringified `(speed, turn)` tuple, e.g.
    `"(0.80, -0.35)"`, applied by the receiver as
    `left = speed + turn`, `right = speed - turn` — so positive turn
    steers right, negative turn steers left.

Something is published to `ME193/Julian` on every single chunk (never
silently skipped), including `"(0.00, 0.00)"` when nothing is detected,
so the robot reliably stops instead of coasting on its last command
forever (the receiver has no message timeout of its own).

## Frequency layout

| Frequency range | Meaning |
|---|---|
| 600–850 Hz | `LOSE` event — publishes `"lose"` once |
| 850–1000 Hz | guard gap — treated like silence |
| 1000–1600 Hz | `STEER` — turn maps linearly from -1.0 (1000 Hz) through 0.0 (1300 Hz, straight) to +1.0 (1600 Hz), at a constant forward speed |
| 1600–1700 Hz | guard gap — treated like silence |
| 1700–2200 Hz | `WIN` event — publishes `"win"` once |
| anything else / no clear tone | `"(0.00, 0.00)"` |

A tone only counts as deliberate (not background noise) once the
microphone's ambient noise floor has been calibrated at startup and the
chunk's volume and spectral "tonality" (how sharply peaked vs. broadband
the sound is) both clear their thresholds — see the constants at the top
of `main.py` (`TONALITY_THRESHOLD`, `SILENCE_MARGIN_MULT`,
`SILENCE_MARGIN_ADD`, `FORWARD_SPEED`, and the band edges) if you need to
retune by ear.

Both scripts use the public `test.mosquitto.org` broker by default (see
`mqttlib.MQTTClient`).
