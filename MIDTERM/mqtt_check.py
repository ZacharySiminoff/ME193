"""
Check the high score MQTT topic.

    python mqtt_check.py              # print every message on ME193/Rogers/Zack
    python mqtt_check.py --send 12.0  # publish a test value and exit

The game publishes the high score as a float string (e.g. "12.0") with
retain=True, so a fresh listener immediately gets the latest value.
"""

import argparse
import time

import config as C
from mqttlib import MQTTClient


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--send', type=float, metavar='VALUE', help='publish VALUE and exit')
    ap.add_argument('--topic', default=C.MQTT_TOPIC)
    args = ap.parse_args()

    with MQTTClient(C.MQTT_BROKER, C.MQTT_PORT) as client:
        if args.send is not None:
            client.publish(args.topic, f'{args.send:.1f}', retain=True)
            time.sleep(0.5)   # let it go out before disconnecting
            print(f'Published {args.send:.1f} to {args.topic}')
            return

        client.subscribe(args.topic, lambda topic, payload: print(f'[{topic}] {payload}', flush=True))
        print(f'Listening on {args.topic} at {C.MQTT_BROKER}. Ctrl-C to quit.')
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
