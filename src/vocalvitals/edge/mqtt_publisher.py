"""Publish synthetic demo events to an MQTT broker."""

from __future__ import annotations

import json

import paho.mqtt.client as mqtt

from vocalvitals.api.models import EventIn


def publish_demo_event(
    event: EventIn,
    host: str,
    port: int = 1883,
    topic: str = "vocalvitals/demo/events",
) -> None:
    if event.source != "synthetic_demo":
        raise ValueError("Only synthetic_demo events may be published by this helper.")
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.connect(host, port, keepalive=30)
    info = client.publish(topic, json.dumps(event.model_dump(mode="json")), qos=1)
    info.wait_for_publish(timeout=10)
    client.disconnect()
