import time

from mqttlib import MQTTClient

TOPIC = "ME193/Rogers"

with MQTTClient() as client:
    client.publish(TOPIC, "start")
    print(f"Published 'start' to '{TOPIC}'")

    time.sleep(1)  # give the message time to go out before disconnecting
