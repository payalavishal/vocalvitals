from fastapi.testclient import TestClient

from vocalvitals.api.app import create_app
from vocalvitals.api.store import EventStore


def test_health_reports_synthetic_mode() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["mode"] == "synthetic_demo_only"


def test_event_round_trip_and_patient_trend() -> None:
    client = TestClient(create_app())
    created = client.post(
        "/event",
        json={
            "patient_id": "DEMO-001",
            "label": "Delayed",
            "confidence": 0.82,
            "source": "synthetic_demo",
        },
    )
    assert created.status_code == 201
    assert created.json()["event_id"]

    patients = client.get("/patients")
    assert patients.status_code == 200
    assert patients.json()[0]["event_count"] == 1

    trend = client.get("/patients/DEMO-001/trend")
    assert trend.status_code == 200
    assert trend.json()[0]["label"] == "Delayed"


def test_api_rejects_non_synthetic_event_source() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/event",
        json={
            "patient_id": "P001",
            "label": "Normal",
            "confidence": 0.5,
            "source": "real_binary_model",
        },
    )
    assert response.status_code == 400


def test_event_store_keeps_only_configured_history() -> None:
    from vocalvitals.api.models import EventIn

    store = EventStore(max_events_per_patient=2)
    for confidence in (0.1, 0.2, 0.3):
        store.add(
            EventIn(
                patient_id="DEMO-002",
                label="Normal",
                confidence=confidence,
            )
        )
    events = store.trend("DEMO-002")
    assert len(events) == 2
    assert [event.confidence for event in events] == [0.2, 0.3]
