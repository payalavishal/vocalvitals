"""Streamlit dashboard for synthetic demo events only."""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="VocalVitals Demo", layout="wide")
st.title("VocalVitals - Synthetic Demo Dashboard")
st.caption(
    "Demonstration only. Normal / Delayed / Aspirating are artificial labels, "
    "not clinical findings, diagnosis, or caregiver alerts."
)

api_url = st.sidebar.text_input(
    "API base URL", os.getenv("VOCALVITALS_API_URL", "http://127.0.0.1:8000")
).rstrip("/")
if st.sidebar.button("Refresh"):
    st.rerun()

try:
    response = requests.get(f"{api_url}/patients", timeout=5)
    response.raise_for_status()
    patients = response.json()
except requests.RequestException as error:
    st.error(f"Cannot reach the demo API at {api_url}: {error}")
    st.stop()

total_events = sum(patient["event_count"] for patient in patients)
col1, col2 = st.columns(2)
col1.metric("Demo patients", len(patients))
col2.metric("Stored demo events", total_events)

if not patients:
    st.info("No synthetic demo events yet. Send events to POST /event.")
    st.stop()

st.subheader("Patients")
st.dataframe(pd.DataFrame(patients), use_container_width=True, hide_index=True)

patient_ids = [patient["patient_id"] for patient in patients]
selected = st.selectbox("View event trend", patient_ids)
try:
    trend_response = requests.get(f"{api_url}/patients/{selected}/trend", timeout=5)
    trend_response.raise_for_status()
    events = trend_response.json()
except requests.RequestException as error:
    st.error(f"Could not load event history: {error}")
    st.stop()

history = pd.DataFrame(events)
if not history.empty:
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    st.subheader(f"Recent synthetic events - {selected}")
    st.dataframe(
        history[["timestamp", "label", "confidence", "source"]],
        use_container_width=True,
        hide_index=True,
    )
    counts = history.groupby("label").size().rename("events")
    st.bar_chart(counts)
