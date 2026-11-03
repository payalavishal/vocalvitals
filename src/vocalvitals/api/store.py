"""Thread-safe in-memory store for demo events."""

from __future__ import annotations

from collections import defaultdict, deque
from threading import RLock
from uuid import uuid4

from vocalvitals.api.models import EventIn, EventOut, PatientSummary


class EventStore:
    """Volatile store; not a durable patient record system."""

    def __init__(self, max_events_per_patient: int = 500) -> None:
        self._events: dict[str, deque[EventOut]] = defaultdict(
            lambda: deque(maxlen=max_events_per_patient)
        )
        self._lock = RLock()

    def add(self, event: EventIn) -> EventOut:
        record = EventOut(event_id=str(uuid4()), **event.model_dump())
        with self._lock:
            self._events[event.patient_id].append(record)
        return record

    def patients(self) -> list[PatientSummary]:
        with self._lock:
            summaries = []
            for patient_id, events in self._events.items():
                latest = events[-1] if events else None
                summaries.append(
                    PatientSummary(
                        patient_id=patient_id,
                        event_count=len(events),
                        latest_event=latest.timestamp if latest else None,
                        latest_label=latest.label if latest else None,
                    )
                )
        return sorted(summaries, key=lambda item: item.patient_id)

    def trend(self, patient_id: str) -> list[EventOut]:
        with self._lock:
            return list(self._events.get(patient_id, ()))

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
