"""
Listen on an MQTT topic and send votes back to it.

Type a vote and press Enter to publish it; everyone else's votes print as they
arrive. Ctrl-C to quit and see the tally.

    python vote.py           # vote on the default ME193 topic
    python vote.py robots    # vote on the 'robots' topic
"""

import sys
import threading
import time

from mqttlib import MQTTClient

TOPIC = sys.argv[1] if len(sys.argv) > 1 else "ME193"
PROMPT = "vote> "

tally = {}
lock = threading.Lock()  # on_message runs on mqttlib's background network thread
waiting = threading.Event()  # set while the main thread sits in input()


def on_message(topic, payload):
    vote = payload.strip()
    if not vote:
        return

    with lock:
        tally[vote] = tally.get(vote, 0) + 1
        count = tally[vote]

    # \r + clear-to-end-of-line wipes the prompt so the vote lands on a clean
    # line, then redraw the prompt under it if we interrupted someone typing
    print(f"\r\033[K[{topic}] {vote}  ({count})")
    if waiting.is_set():
        print(PROMPT, end="", flush=True)


def show_tally():
    with lock:
        if not tally:
            print("\nNo votes.")
            return
        print("\nFinal tally:")
        for vote, count in sorted(tally.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {count:3d}  {vote}")


with MQTTClient() as client:
    client.subscribe(TOPIC, on_message)
    print(f"Listening on '{TOPIC}'. Type a vote and press Enter, Ctrl-C to quit.")

    try:
        while True:
            waiting.set()
            try:
                vote = input(PROMPT).strip()
            finally:
                waiting.clear()

            if vote:
                client.publish(TOPIC, vote)  # echoes back through on_message
    except (KeyboardInterrupt, EOFError):
        time.sleep(0.5)  # let votes already in flight arrive before disconnecting
        show_tally()
