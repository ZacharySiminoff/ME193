import threading

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.animation import FuncAnimation

from mqttlib import MQTTClient

# ME193/Rogers is only used to gate the local start signal (see
# wait_for_start). ME193/Julian is the topic the receiving robot
# (mqtt_double_motor_game.py) actually subscribes to for its role and
# every drive/event message.
ROGERS_TOPIC = "ME193/Rogers"
DRIVE_TOPIC = "ME193/Julian"

RATE = 44100
CHUNK = 2048

# Steering: pitch maps linearly to turn, 1300 Hz = straight ahead.
STEER_FREQ_MIN = 1000   # -> turn = -1.0 (left)
STEER_FREQ_MAX = 1600   # -> turn = +1.0 (right)
FORWARD_SPEED = 0.8     # constant forward speed while steering; tune/replace later

# Reserved bands outside the steering range that fire one-shot events
# instead of driving. Julian's receiver only understands the literal
# strings "win" and "lose" besides the driving tuple.
LOSE_FREQ_MIN, LOSE_FREQ_MAX = 600, 850
WIN_FREQ_MIN, WIN_FREQ_MAX = 1700, 2200

# Band searched for the dominant pitch / tonality check (wide enough to
# cover all the bands above with margin on either side).
ANALYSIS_FREQ_MIN = 200
ANALYSIS_FREQ_MAX = 2500

# How many times louder the peak frequency must be than the average of
# the analysis band to count as a deliberate pitch rather than
# broadband noise (talking, chair scraping, room hum).
TONALITY_THRESHOLD = 4.0

# The quiet gate sits at max(floor * this multiplier, floor + this margin).
SILENCE_MARGIN_MULT = 1.5
SILENCE_MARGIN_ADD = 40.0
CALIBRATION_CHUNKS = 20

# Scrolling spectrogram display
HISTORY = 200       # number of FFT columns kept on screen
DISPLAY_MAX_FREQ = 3000  # Hz, just for the y-axis zoom (peak detection still uses the full spectrum)

# Bands shown as reference stripes on the plot
COMMAND_BANDS = [
    ("LOSE", LOSE_FREQ_MIN, LOSE_FREQ_MAX),
    ("STEER", STEER_FREQ_MIN, STEER_FREQ_MAX),
    ("WIN", WIN_FREQ_MIN, WIN_FREQ_MAX),
]


# -------------------------
# Role
# -------------------------

def ask_role():
    while True:
        role = input("Are you the shooter or goalie? ").strip().lower()
        if role in ("shooter", "goalie"):
            return role
        print("Please type 'shooter' or 'goalie'.")


# -------------------------
# Pitch / tone analysis
# -------------------------

def clamp_map(value, in_min, in_max, out_min=-1.0, out_max=1.0):
    if value <= in_min:
        return out_min
    if value >= in_max:
        return out_max
    fraction = (value - in_min) / (in_max - in_min)
    return out_min + fraction * (out_max - out_min)


def calibrate_noise_floor(stream):
    print("Calibrating ambient noise level - stay quiet for a second...")
    floor_samples = []
    for _ in range(CALIBRATION_CHUNKS):
        data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        floor_samples.append(float(np.sqrt(np.mean(samples ** 2))))
    floor = float(np.mean(floor_samples))
    quiet_threshold = max(floor * SILENCE_MARGIN_MULT, floor + SILENCE_MARGIN_ADD)
    print(f"Ambient noise floor rms={floor:.1f}, quiet threshold rms={quiet_threshold:.1f}")
    return quiet_threshold


# -------------------------
# Wait for start signal
# -------------------------

def wait_for_start(client):
    started = threading.Event()

    def on_message(topic, payload):
        if payload.strip().lower() == "start":
            started.set()

    client.subscribe(ROGERS_TOPIC, on_message)
    print(f"Waiting for 'start' on '{ROGERS_TOPIC}'...")
    started.wait()
    print("Received start - beginning audio processing.")


# -------------------------
# Plot
# -------------------------

def build_plot(frequencies):
    fig, ax = plt.subplots(figsize=(10, 6))

    times = np.linspace(-HISTORY * CHUNK / RATE, 0, HISTORY)
    spec_buffer = np.full((len(frequencies), HISTORY), -100.0)

    image = ax.imshow(
        spec_buffer,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
        cmap="magma",
    )
    fig.colorbar(image, ax=ax, label="Magnitude (dB)")

    for name, lo, hi in COMMAND_BANDS:
        ax.axhspan(lo, hi, color="cyan", alpha=0.15, linewidth=0)
        ax.text(0.99, (lo + hi) / 2, name, color="cyan", fontsize=9,
                va="center", ha="right", transform=ax.get_yaxis_transform())

    straight_freq = (STEER_FREQ_MIN + STEER_FREQ_MAX) / 2
    ax.axhline(straight_freq, color="white", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(0.01, straight_freq, "straight", color="white", fontsize=8,
            va="bottom", ha="left", transform=ax.get_yaxis_transform())

    peak_scatter = ax.scatter(times, np.full(HISTORY, np.nan), s=15, color="red")

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, DISPLAY_MAX_FREQ)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    title = ax.set_title("(0.00, 0.00)")

    return fig, ax, image, peak_scatter, title, times, spec_buffer


def main():
    role = ask_role()

    client = MQTTClient()
    client.connect()

    client.publish(DRIVE_TOPIC, role)
    print(f"Announced role '{role}' on '{DRIVE_TOPIC}'")

    wait_for_start(client)

    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    quiet_threshold = calibrate_noise_floor(stream)

    frequencies = np.fft.rfftfreq(CHUNK, 1 / RATE)
    analysis_band = (frequencies >= ANALYSIS_FREQ_MIN) & (frequencies <= ANALYSIS_FREQ_MAX)
    analysis_freqs = frequencies[analysis_band]
    window = np.hanning(CHUNK)

    fig, _ax, image, peak_scatter, title, times, spec_buffer = build_plot(frequencies)
    peak_freqs = np.full(HISTORY, np.nan)
    active_zone = None  # None, "lose", "win", "steer", or "none" (guard gap)

    def band_zone(freq):
        if LOSE_FREQ_MIN <= freq <= LOSE_FREQ_MAX:
            return "lose"
        if STEER_FREQ_MIN <= freq <= STEER_FREQ_MAX:
            return "steer"
        if WIN_FREQ_MIN <= freq <= WIN_FREQ_MAX:
            return "win"
        return "none"

    def update(frame):
        nonlocal active_zone

        # Get microphone data
        raw_data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)

        # FFT (windowed, for both the display and the pitch/tonality analysis)
        magnitudes = np.abs(np.fft.rfft(samples * window))
        rms = float(np.sqrt(np.mean(samples ** 2)))

        band_magnitudes = magnitudes[analysis_band]
        peak_index = np.argmax(band_magnitudes)
        peak_frequency = float(analysis_freqs[peak_index])
        peak_magnitude = float(band_magnitudes[peak_index])
        mean_magnitude = float(np.mean(band_magnitudes))
        tonality = peak_magnitude / (mean_magnitude + 1e-9)

        is_tone = rms > quiet_threshold and tonality > TONALITY_THRESHOLD

        if not is_tone:
            payload = "(0.00, 0.00)"
            active_zone = None
        else:
            zone = band_zone(peak_frequency)
            if zone == "lose":
                payload = "lose" if active_zone != "lose" else "(0.00, 0.00)"
                active_zone = "lose"
            elif zone == "win":
                payload = "win" if active_zone != "win" else "(0.00, 0.00)"
                active_zone = "win"
            elif zone == "steer":
                turn = clamp_map(peak_frequency, STEER_FREQ_MIN, STEER_FREQ_MAX)
                payload = f"({FORWARD_SPEED:.2f}, {turn:.2f})"
                active_zone = "steer"
            else:
                payload = "(0.00, 0.00)"
                active_zone = "none"

        client.publish(DRIVE_TOPIC, payload)

        # Scroll the spectrogram buffer and peak marker history
        spec_db = 20 * np.log10(magnitudes + 1e-6)
        spec_buffer[:, :-1] = spec_buffer[:, 1:]
        spec_buffer[:, -1] = spec_db
        peak_freqs[:-1] = peak_freqs[1:]
        peak_freqs[-1] = peak_frequency if is_tone else np.nan

        image.set_data(spec_buffer)
        current_max = spec_buffer.max()
        image.set_clim(vmin=current_max - 80, vmax=current_max)

        peak_scatter.set_offsets(np.column_stack((times, peak_freqs)))
        title.set_text(payload)

        return image, peak_scatter, title

    anim = FuncAnimation(fig, update, interval=1, blit=False, cache_frame_data=False)

    try:
        plt.show()

    except KeyboardInterrupt:
        pass

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        client.disconnect()


if __name__ == "__main__":
    main()
