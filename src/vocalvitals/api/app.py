"""FastAPI endpoints for the synthetic demonstration pipeline only."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from vocalvitals.api.models import EventIn, EventOut, PatientSummary
from vocalvitals.api.mqtt_bridge import MqttEventBridge
from vocalvitals.api.store import EventStore

SAFETY_DESCRIPTION = (
    "Synthetic demonstration API only. Normal / Delayed / Aspirating are "
    "artificial demo labels, not clinical outcomes or medical advice."
)


def create_app(store: EventStore | None = None, mqtt_host: str | None = None) -> FastAPI:
    event_store = store or EventStore(max_events_per_patient=500)
    broker_host = mqtt_host or os.getenv("VOCALVITALS_MQTT_HOST")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bridge = None
        if broker_host:
            bridge = MqttEventBridge(event_store, broker_host)
            bridge.start()
            app.state.mqtt_bridge = bridge
        try:
            yield
        finally:
            if bridge:
                bridge.stop()

    app = FastAPI(
        title="VocalVitals Synthetic Demo API",
        version="0.1.0",
        description=SAFETY_DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.event_store = event_store

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "synthetic_demo_only"}

    @app.post("/event", response_model=EventOut, status_code=201)
    def create_event(event: EventIn) -> EventOut:
        if event.source != "synthetic_demo":
            raise HTTPException(
                status_code=400,
                detail="This API accepts synthetic_demo events only.",
            )
        return event_store.add(event)

    @app.get("/patients", response_model=list[PatientSummary])
    def list_patients() -> list[PatientSummary]:
        return event_store.patients()

    @app.get("/patients/{patient_id}/trend", response_model=list[EventOut])
    def patient_trend(patient_id: str) -> list[EventOut]:
        events = event_store.trend(patient_id)
        if not events:
            raise HTTPException(status_code=404, detail="Patient has no demo events")
        return events

    return app


app = create_app()
