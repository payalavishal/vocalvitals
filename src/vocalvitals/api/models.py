"""API schemas for the synthetic demonstration pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SyntheticLabel(StrEnum):
    NORMAL = "Normal"
    DELAYED = "Delayed"
    ASPIRATING = "Aspirating"


class EventIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(min_length=1, max_length=64)
    label: SyntheticLabel
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = Field(default="synthetic_demo", max_length=64)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("patient_id")
    @classmethod
    def strip_patient_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("patient_id cannot be blank")
        return cleaned


class EventOut(EventIn):
    event_id: str


class PatientSummary(BaseModel):
    patient_id: str
    event_count: int
    latest_event: datetime | None
    latest_label: SyntheticLabel | None
