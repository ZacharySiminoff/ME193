import paho.mqtt.client as mqtt
import pyaudio
import numpy as np


# -------------------------
# MQTT
# -------------------------

client = mqtt.Client()
client.connect("BROKER_ADDRESS", 1883)
client.loop_start()


# -------------------------
# Audio
# -------------------------

RATE = 44100
CHUNK = 2048

p = pyaudio.PyAudio()

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)


# -------------------------
# Decision policy
# -------------------------

def decide(freq):

    if 700 < freq < 900:
        return "STOP"

    elif 1000 < freq < 1300:
        return "LEFT"

    elif 1400 < freq < 1700:
        return "RIGHT"

    elif 2000 < freq < 2400:
        return "FORWARD"

    else:
        return "NONE"


# -------------------------
# Main loop
# -------------------------

while True:

    # Get microphone data
    raw_data = stream.read(CHUNK)

    samples = np.frombuffer(
        raw_data,
        dtype=np.int16
    )

    # FFT
    spectrum = np.abs(np.fft.rfft(samples))

    frequencies = np.fft.rfftfreq(
        len(samples),
        1 / RATE
    )

    # dominant frequency
    peak_index = np.argmax(spectrum)

    peak_frequency = frequencies[peak_index]

    # Make decision
    command = decide(peak_frequency)

    print(
        f"Frequency: {peak_frequency:.0f} Hz "
        f"→ {command}"
    )

    # Send command
    if command != "NONE":
        client.publish(
            "ME193/myrobot/control",
            command
        )