"""
tests/test_agent1_patient_intake.py
Member: 

Test suite for the Patient Intake Agent and its patient_record_reader tool.
Covers:
    - Property-based unit tests for every validation rule in the tool.
    - Agent integration tests using GlobalState.
    - Edge cases: missing fields, invalid ranges, malformed JSON, empty symptoms.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.tool_patient_reader import patient_record_reader
from agents.agent_patient_intake import PatientIntakeAgent
from config.state import reset_state


class TestPatientRecordReaderTool:
    """Unit and property tests for patient_record_reader."""

    def test_returns_all_required_keys(self, valid_patient_file: str) -> None:
        """Tool must return all expected top-level keys."""
        result = patient_record_reader(valid_patient_file)
        expected_keys = {
            "success", "patient_data", "validation_passed",
            "validation_errors", "validation_warnings",
            "field_statuses", "loaded_at",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    def test_valid_file_passes_validation(self, valid_patient_file: str) -> None:
        """A well-formed patient file must pass all validation checks."""
        result = patient_record_reader(valid_patient_file)
        assert result["success"] is True
        assert result["validation_passed"] is True
        assert result["validation_errors"] == []

    def test_file_not_found_returns_failure(self) -> None:
        """Non-existent file path must return success=False, not raise."""
        result = patient_record_reader("/nonexistent/path/patient.json")
        assert result["success"] is False
        assert result["validation_passed"] is False
        assert len(result["validation_errors"]) >= 1

    def test_empty_path_returns_failure(self) -> None:
        """Empty string path must return success=False."""
        result = patient_record_reader("")
        assert result["success"] is False

    def test_invalid_json_returns_failure(self, tmp_path) -> None:
        """A file containing invalid JSON must return success=False."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ this is not valid json }", encoding="utf-8")
        result = patient_record_reader(str(bad_file))
        assert result["success"] is False
        assert any("JSON" in e or "json" in e.lower() for e in result["validation_errors"])

    def test_missing_required_field_detected(self, tmp_path, sample_patient_data) -> None:
        """Removing a required field must produce a validation error."""
        del sample_patient_data["age"]
        path = tmp_path / "missing_age.json"
        path.write_text(json.dumps(sample_patient_data), encoding="utf-8")
        result = patient_record_reader(str(path))
        assert result["validation_passed"] is False
        assert any("age" in e for e in result["validation_errors"])

    def test_invalid_age_detected(self, tmp_path, sample_patient_data) -> None:
        """An age of -5 must be flagged as invalid."""
        sample_patient_data["age"] = -5
        path = tmp_path / "bad_age.json"
        path.write_text(json.dumps(sample_patient_data), encoding="utf-8")
        result = patient_record_reader(str(path))
        assert result["validation_passed"] is False
        assert any("age" in e.lower() for e in result["validation_errors"])

    def test_invalid_blood_type_detected(self, tmp_path, sample_patient_data) -> None:
        """An unrecognised blood type must be flagged."""
        sample_patient_data["blood_type"] = "Z-"
        path = tmp_path / "bad_blood.json"
        path.write_text(json.dumps(sample_patient_data), encoding="utf-8")
        result = patient_record_reader(str(path))
        assert result["validation_passed"] is False
        assert any("blood_type" in e for e in result["validation_errors"])

    def test_empty_symptoms_detected(self, tmp_path, sample_patient_data) -> None:
        """An empty symptoms list must be flagged as a validation error."""
        sample_patient_data["symptoms"] = []
        path = tmp_path / "no_symptoms.json"
        path.write_text(json.dumps(sample_patient_data), encoding="utf-8")
        result = patient_record_reader(str(path))
        assert result["validation_passed"] is False
        assert any("symptoms" in e.lower() for e in result["validation_errors"])

    def test_out_of_range_vital_sign_detected(self, tmp_path, sample_patient_data) -> None:
        """Temperature of 60 C must be flagged as out of range."""
        sample_patient_data["vital_signs"]["temperature_celsius"] = 60.0
        path = tmp_path / "bad_temp.json"
        path.write_text(json.dumps(sample_patient_data), encoding="utf-8")
        result = patient_record_reader(str(path))
        assert result["validation_passed"] is False
        assert any("temperature" in e.lower() for e in result["validation_errors"])

    def test_patient_data_preserved_correctly(self, valid_patient_file: str) -> None:
        """All patient fields must be returned unchanged in patient_data."""
        result = patient_record_reader(valid_patient_file)
        assert result["patient_data"]["patient_id"] == "PT999"
        assert result["patient_data"]["name"] == "Test Patient"
        assert result["patient_data"]["age"] == 30

    def test_field_statuses_populated(self, valid_patient_file: str) -> None:
        """field_statuses must contain an entry for every required field."""
        result = patient_record_reader(valid_patient_file)
        required = {
            "patient_id", "name", "age", "gender", "blood_type",
            "chief_complaint", "symptoms", "medical_history",
            "current_medications", "allergies", "vital_signs",
        }
        for field in required:
            assert field in result["field_statuses"], (
                f"field_statuses missing key: '{field}'"
            )

    def test_loaded_at_is_iso_timestamp(self, valid_patient_file: str) -> None:
        """loaded_at must be a non-empty ISO-8601-style timestamp string."""
        result = patient_record_reader(valid_patient_file)
        assert isinstance(result["loaded_at"], str)
        assert len(result["loaded_at"]) >= 19

    def test_multiple_missing_fields_all_reported(self, tmp_path, sample_patient_data) -> None:
        """All missing fields must appear in validation_errors, not just the first."""
        del sample_patient_data["name"]
        del sample_patient_data["age"]
        del sample_patient_data["gender"]
        path = tmp_path / "multi_missing.json"
        path.write_text(json.dumps(sample_patient_data), encoding="utf-8")
        result = patient_record_reader(str(path))
        assert result["validation_passed"] is False
        assert len(result["validation_errors"]) >= 3


class TestPatientIntakeAgentIntegration:
    """Integration tests for PatientIntakeAgent with GlobalState."""

    def test_agent_populates_patient_info(self, valid_patient_file: str) -> None:
        """Agent must populate state.patient_info after a successful run."""
        state = reset_state()
        state.patient_file_path = valid_patient_file
        result_state = PatientIntakeAgent().run(state)
        assert isinstance(result_state.patient_info, dict)
        assert result_state.patient_info.get("patient_id") == "PT999"

    def test_agent_sets_is_valid_true_for_good_file(self, valid_patient_file: str) -> None:
        """state.is_valid must be True after processing a valid patient file."""
        state = reset_state()
        state.patient_file_path = valid_patient_file
        result_state = PatientIntakeAgent().run(state)
        assert result_state.is_valid is True

    def test_agent_sets_is_valid_false_for_bad_file(self, tmp_path, sample_patient_data) -> None:
        """state.is_valid must be False when validation errors are present."""
        sample_patient_data["age"] = -1
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(sample_patient_data), encoding="utf-8")
        state = reset_state()
        state.patient_file_path = str(path)
        result_state = PatientIntakeAgent().run(state)
        assert result_state.is_valid is False

    def test_agent_handles_missing_file_without_crash(self) -> None:
        """Agent must return state cleanly when the file does not exist."""
        state = reset_state()
        state.patient_file_path = "/no/such/file.json"
        result_state = PatientIntakeAgent().run(state)
        assert result_state is not None
        assert result_state.is_valid is False

    def test_agent_handles_no_path_without_crash(self) -> None:
        """Agent must return state cleanly when patient_file_path is empty."""
        state = reset_state()
        result_state = PatientIntakeAgent().run(state)
        assert result_state is not None

    def test_agent_creates_log_entry(self, valid_patient_file: str) -> None:
        """Agent must append at least one entry to state.agent_logs."""
        state = reset_state()
        state.patient_file_path = valid_patient_file
        result_state = PatientIntakeAgent().run(state)
        assert len(result_state.agent_logs) >= 1
        assert result_state.agent_logs[-1].agent_name == "PatientIntakeAgent"

    def test_agent_processes_real_pt001(self, patient_pt001_path: str) -> None:
        """Agent must successfully process the bundled PT001 sample patient."""
        state = reset_state()
        state.patient_file_path = patient_pt001_path
        result_state = PatientIntakeAgent().run(state)
        assert result_state.patient_info.get("patient_id") == "PT001"
        assert result_state.is_valid is True
