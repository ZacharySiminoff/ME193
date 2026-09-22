'''
Easiest way to use:

from mqttlib import MQTTClient

with MQTTClient() as client:
    client.subscribe("ME193", lambda topic, payload: print(topic, payload))
    client.publish("ME193", "hello world")
'''


import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883


class MQTTClient:
    def __init__(self, broker=BROKER, port=PORT):
        self.broker = broker
        self.port = port
        self.callbacks = {}
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_message = self._on_message

    def connect(self):
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_start()

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def _on_message(self, client, userdata, message):
        payload = message.payload.decode()
        for pattern, callback in self.callbacks.items():
            if mqtt.topic_matches_sub(pattern, message.topic):
                callback(message.topic, payload)

    def subscribe(self, topic, callback):
        self.callbacks[topic] = callback
        self.client.subscribe(topic)

    def unsubscribe(self, topic):
        self.callbacks.pop(topic, None)
        self.client.unsubscribe(topic)

    def publish(self, topic, payload):
        self.client.publish(topic, payload)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
        return False
