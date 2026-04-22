"""
Shared pytest fixtures for the Healthcare MAS test suite.
Available to all test modules via automatic discovery.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture()
def sample_patient_data() -> dict:
    """Return a fully valid patient data dictionary."""
    return {
        "patient_id": "PT999",
        "name": "Test Patient",
        "age": 30,
        "gender": "Female",
        "blood_type": "O+",
        "chief_complaint": "Fever and body aches",
        "symptoms": ["high fever", "body aches", "fatigue", "chills"],
        "medical_history": [],
        "current_medications": [],
        "allergies": [],
        "vital_signs": {
            "temperature_celsius": 38.9,
            "blood_pressure_mmhg": "120/80",
            "heart_rate_bpm": 95,
            "respiratory_rate_rpm": 18,
            "oxygen_saturation_percent": 98,
        },
        "contact": {
            "phone": "000-000-0000",
            "emergency_contact": "Next of Kin",
            "emergency_phone": "000-000-0001",
        },
        "registration_date": "2026-04-22",
    }


@pytest.fixture()
def valid_patient_file(tmp_path, sample_patient_data) -> str:
    """Write a valid patient JSON to a temp file and return the path."""
    path = tmp_path / "patient_TEST.json"
    path.write_text(json.dumps(sample_patient_data), encoding="utf-8")
    return str(path)


@pytest.fixture()
def patient_pt001_path() -> str:
    """Return the path to the bundled PT001 sample patient file."""
    here = os.path.dirname(__file__)
    return os.path.join(here, "data", "patients", "patient_PT001.json")


@pytest.fixture()
def patient_pt002_path() -> str:
    """Return the path to the bundled PT002 sample patient file."""
    here = os.path.dirname(__file__)
    return os.path.join(here, "data", "patients", "patient_PT002.json")
