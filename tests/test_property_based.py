"""
tests/test_property_based.py

Property-based tests for the Healthcare MAS tool layer.
Uses the Hypothesis library to generate diverse inputs and verify
invariants that must hold for any valid or invalid input.

Invariants tested:
    Tool 1 — patient_record_reader:
        1. Return schema is always complete (all 7 keys present).
        2. success=False implies validation_passed=False.
        3. A fully valid synthesised patient always passes validation.
        4. An out-of-range age always triggers a validation error.
        5. An invalid blood type always triggers a validation error.
        6. An out-of-range vital sign always triggers a validation error.

    Tool 2 — symptom_analyzer:
        7.  Return schema is always complete (all 6 keys present).
        8.  confidence_score is always in [0.0, 100.0].
        9.  possible_conditions is always sorted descending by confidence_score.
        10. matched_symptoms is always a subset of normalised_symptoms.
        11. risk_level is always one of the four valid strings.
        12. len(possible_conditions) never exceeds top_n.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.tool_patient_reader import (
    REQUIRED_FIELDS,
    VALID_BLOOD_TYPES,
    VALID_GENDERS,
    VITAL_SIGN_RANGES,
    patient_record_reader,
)
from tools.tool_symptom_analyzer import symptom_analyzer

# ── Hypothesis strategies ─────────────────────────────────────────────────────

_valid_blood_type_st = st.sampled_from(VALID_BLOOD_TYPES)

_invalid_blood_type_st = st.text(
    alphabet="ABCOZ+-0123456789",
    min_size=1,
    max_size=5,
).filter(lambda s: s not in VALID_BLOOD_TYPES and s.strip() != "")

_valid_gender_st = st.sampled_from(VALID_GENDERS)

_out_of_range_age_st = st.one_of(
    st.integers(min_value=151, max_value=9999),
    st.integers(min_value=-9999, max_value=-1),
)

_valid_age_st = st.integers(min_value=0, max_value=150)

_symptom_str_st = (
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
        min_size=2,
        max_size=40,
    )
    .map(str.strip)
    .filter(lambda s: len(s) >= 2)
)

_symptom_list_st = st.lists(_symptom_str_st, min_size=1, max_size=8)

_KNOWN_SYMPTOMS = [
    "high fever", "fever", "body aches", "chills", "severe fatigue",
    "cough", "runny nose", "sneezing", "chest pain", "difficulty breathing",
    "productive cough", "nausea", "vomiting", "headache", "fatigue",
    "burning urination", "frequent urination", "shortness of breath",
    "sweating", "loss of appetite",
]

_known_symptom_list_st = st.lists(
    st.sampled_from(_KNOWN_SYMPTOMS),
    min_size=1,
    max_size=6,
    unique=True,
)

_vital_sign_key_st = st.sampled_from(list(VITAL_SIGN_RANGES.keys()))

_out_of_range_value_st = st.one_of(
    st.floats(min_value=500.0, max_value=9999.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-9999.0, max_value=-0.1, allow_nan=False, allow_infinity=False),
)

_VALID_RISK_LEVELS = {"low", "moderate", "high", "critical", "unknown"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_patient_file(tmp_path, data: dict) -> str:
    """Write patient dict to a temp JSON file and return the path string."""
    p = tmp_path / "hyp_patient.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def _make_valid_patient(
    age: int = 30,
    gender: str = "Female",
    blood_type: str = "O+",
    symptoms: list | None = None,
) -> dict:
    """Build a minimal valid patient dict."""
    return {
        "patient_id": "PT_HYP",
        "name": "Hypothesis Patient",
        "age": age,
        "gender": gender,
        "blood_type": blood_type,
        "chief_complaint": "test complaint",
        "symptoms": symptoms or ["fever"],
        "medical_history": [],
        "current_medications": [],
        "allergies": [],
        "vital_signs": {
            "temperature_celsius": 37.0,
            "blood_pressure_mmhg": "120/80",
            "heart_rate_bpm": 75,
            "respiratory_rate_rpm": 16,
            "oxygen_saturation_percent": 98,
        },
        "contact": {
            "phone": "000-000-0000",
            "emergency_contact": "Next of Kin",
            "emergency_phone": "000-000-0001",
        },
        "registration_date": "2026-01-01",
    }


# ── Tool 1: patient_record_reader property tests ──────────────────────────────

class TestPatientRecordReaderProperties:
    """Property-based invariant tests for patient_record_reader."""

    @given(path=st.text(min_size=0, max_size=200))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_return_schema_always_complete(self, path: str) -> None:
        """
        INVARIANT: For any path string (even gibberish), the tool always
        returns a dict containing all 7 required top-level keys.
        """
        result = patient_record_reader(path)
        required_keys = {
            "success", "patient_data", "validation_passed",
            "validation_errors", "validation_warnings",
            "field_statuses", "loaded_at",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    @given(path=st.text(min_size=1, max_size=200))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_success_false_implies_validation_passed_false(self, path: str) -> None:
        """
        INVARIANT: Whenever success=False, validation_passed must also be False.
        No valid scenario exists where loading failed but validation passed.
        """
        result = patient_record_reader(path)
        if not result["success"]:
            assert result["validation_passed"] is False

    @given(
        age=_valid_age_st,
        gender=_valid_gender_st,
        blood_type=_valid_blood_type_st,
        symptoms=st.lists(
            st.text(min_size=2, max_size=30).filter(lambda s: len(s.strip()) >= 2),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_fully_valid_patient_always_passes(
        self, tmp_path, age: int, gender: str, blood_type: str, symptoms: list
    ) -> None:
        """
        INVARIANT: A patient dict with valid age [0,150], valid gender, valid
        blood_type, and a non-empty symptoms list always passes validation with
        no errors.
        """
        data = _make_valid_patient(
            age=age, gender=gender, blood_type=blood_type, symptoms=symptoms
        )
        path = _write_patient_file(tmp_path, data)
        result = patient_record_reader(path)
        assert result["success"] is True
        assert result["validation_passed"] is True, (
            f"Unexpected errors for valid patient: {result['validation_errors']}"
        )

    @given(age=_out_of_range_age_st)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_out_of_range_age_always_fails(self, tmp_path, age: int) -> None:
        """
        INVARIANT: Any age outside [0, 150] always produces at least one
        validation error containing the word 'age'.
        """
        data = _make_valid_patient(age=age)
        path = _write_patient_file(tmp_path, data)
        result = patient_record_reader(path)
        assert result["validation_passed"] is False
        assert any("age" in e.lower() for e in result["validation_errors"]), (
            f"No age error for age={age}; errors={result['validation_errors']}"
        )

    @given(blood_type=_invalid_blood_type_st)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_invalid_blood_type_always_fails(self, tmp_path, blood_type: str) -> None:
        """
        INVARIANT: Any blood_type not in VALID_BLOOD_TYPES always produces
        a validation error containing 'blood_type'.
        """
        data = _make_valid_patient(blood_type=blood_type)
        path = _write_patient_file(tmp_path, data)
        result = patient_record_reader(path)
        assert result["validation_passed"] is False
        assert any("blood_type" in e for e in result["validation_errors"]), (
            f"No blood_type error for blood_type={blood_type!r}; "
            f"errors={result['validation_errors']}"
        )

    @given(vital_sign=_vital_sign_key_st, out_of_range_value=_out_of_range_value_st)
    @settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_out_of_range_vital_sign_always_fails(
        self, tmp_path, vital_sign: str, out_of_range_value: float
    ) -> None:
        """
        INVARIANT: Any numeric vital sign value far outside its documented
        range always fails validation and includes the sign key in the error.
        """
        data = _make_valid_patient()
        data["vital_signs"][vital_sign] = out_of_range_value
        path = _write_patient_file(tmp_path, data)
        result = patient_record_reader(path)
        assert result["validation_passed"] is False
        assert any(vital_sign in e for e in result["validation_errors"]), (
            f"No error for vital_sign={vital_sign} value={out_of_range_value}; "
            f"errors={result['validation_errors']}"
        )


# ── Tool 2: symptom_analyzer property tests ───────────────────────────────────

class TestSymptomAnalyzerProperties:
    """Property-based invariant tests for symptom_analyzer."""

    @given(symptoms=_symptom_list_st)
    @settings(max_examples=80, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_return_schema_always_complete(self, symptoms: list) -> None:
        """
        INVARIANT: For any list of symptom strings, the tool always returns
        a dict containing all 6 required keys.
        """
        result = symptom_analyzer(symptoms)
        required_keys = {
            "success", "possible_conditions", "risk_level",
            "emergency_flags", "normalised_symptoms", "error",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    @given(symptoms=_known_symptom_list_st)
    @settings(max_examples=80, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_confidence_scores_always_in_range(self, symptoms: list) -> None:
        """
        INVARIANT: Every confidence_score in possible_conditions is always a
        float in [0.0, 100.0] regardless of which symptoms are provided.
        """
        result = symptom_analyzer(symptoms)
        assert result["success"] is True
        for cond in result["possible_conditions"]:
            score = cond["confidence_score"]
            assert isinstance(score, float), (
                f"Score is not float: {type(score)}"
            )
            assert 0.0 <= score <= 100.0, (
                f"Score {score} out of range for condition '{cond['name']}'"
            )

    @given(symptoms=_known_symptom_list_st)
    @settings(max_examples=80, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_conditions_sorted_descending_by_confidence(self, symptoms: list) -> None:
        """
        INVARIANT: possible_conditions is always sorted from highest to lowest
        confidence_score — the top diagnosis is always the best match.
        """
        result = symptom_analyzer(symptoms)
        assert result["success"] is True
        scores = [c["confidence_score"] for c in result["possible_conditions"]]
        assert scores == sorted(scores, reverse=True), (
            f"Conditions not sorted descending: {scores}"
        )

    @given(symptoms=_known_symptom_list_st)
    @settings(max_examples=80, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_matched_symptoms_subset_of_normalised_input(self, symptoms: list) -> None:
        """
        INVARIANT: For every matched condition, each string in matched_symptoms
        must appear in normalised_symptoms. The tool must not fabricate symptom
        strings that were not in the patient's input.
        """
        result = symptom_analyzer(symptoms)
        assert result["success"] is True
        normalised_set = set(result["normalised_symptoms"])
        for cond in result["possible_conditions"]:
            for ms in cond["matched_symptoms"]:
                assert ms in normalised_set, (
                    f"Fabricated matched symptom '{ms}' not in normalised input "
                    f"{normalised_set} for condition '{cond['name']}'"
                )

    @given(symptoms=_symptom_list_st)
    @settings(max_examples=80, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_risk_level_always_valid_string(self, symptoms: list) -> None:
        """
        INVARIANT: risk_level is always one of the four documented values or
        'unknown' for degenerate inputs.
        """
        result = symptom_analyzer(symptoms)
        assert result["risk_level"] in _VALID_RISK_LEVELS, (
            f"Unexpected risk_level={result['risk_level']!r}"
        )

    @given(
        symptoms=_known_symptom_list_st,
        top_n=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_top_n_cap_always_respected(self, symptoms: list, top_n: int) -> None:
        """
        INVARIANT: The number of returned conditions never exceeds top_n
        regardless of how many conditions match the symptom input.
        """
        result = symptom_analyzer(symptoms, top_n=top_n)
        assert len(result["possible_conditions"]) <= top_n, (
            f"Returned {len(result['possible_conditions'])} conditions "
            f"but top_n={top_n}"
        )
