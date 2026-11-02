"""Optional MQTT-to-API bridge for synthetic demo events."""

from __future__ import annotations

import logging

import paho.mqtt.client as mqtt

from vocalvitals.api.models import EventIn
from vocalvitals.api.store import EventStore

LOGGER = logging.getLogger(__name__)


class MqttEventBridge:
    def __init__(
        self,
        store: EventStore,
        host: str,
        port: int = 1883,
        topic: str = "vocalvitals/demo/events",
    ) -> None:
        self.store = store
        self.host = host
        self.port = port
        self.topic = topic
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="vocalvitals-api-bridge",
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: object,
        flags: object,
        reason_code: mqtt.ReasonCode,
        properties: object,
    ) -> None:
        if reason_code.is_failure:
            LOGGER.error("MQTT bridge connection failed: %s", reason_code)
            return
        client.subscribe(self.topic, qos=1)
        LOGGER.info("MQTT bridge subscribed to %s", self.topic)

    def _on_message(self, client: mqtt.Client, userdata: object, message: mqtt.MQTTMessage) -> None:
        try:
            event = EventIn.model_validate_json(message.payload)
            if event.source != "synthetic_demo":
                LOGGER.warning("Ignoring non-synthetic MQTT event")
                return
            self.store.add(event)
        except Exception:
            LOGGER.exception("Rejected invalid MQTT demo event")

    def start(self) -> None:
        self.client.connect_async(self.host, self.port, keepalive=60)
        self.client.loop_start()

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
