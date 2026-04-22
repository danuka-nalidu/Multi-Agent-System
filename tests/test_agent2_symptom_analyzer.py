"""
tests/test_agent2_symptom_analyzer.py
Member: 

Test suite for the Symptom Analyzer Agent and its symptom_analyzer tool.
Covers:
    - Correct symptom-to-condition matching and ranking.
    - Emergency indicator detection and risk escalation.
    - Edge cases: no symptoms, unknown symptoms, single symptom.
    - Agent integration with GlobalState.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.tool_symptom_analyzer import symptom_analyzer
from agents.agent_symptom_analyzer import SymptomAnalyzerAgent
from config.state import reset_state


class TestSymptomAnalyzerTool:
    """Unit and property tests for symptom_analyzer."""

    def test_returns_all_required_keys(self) -> None:
        """Tool must always return all required top-level keys."""
        result = symptom_analyzer(["fever"])
        required = {
            "success", "possible_conditions", "risk_level",
            "emergency_flags", "normalised_symptoms", "error",
        }
        assert required.issubset(result.keys())

    def test_influenza_symptoms_match_influenza(self) -> None:
        """Classic influenza symptoms must rank Influenza as the top condition."""
        result = symptom_analyzer(["high fever", "body aches", "chills", "severe fatigue"])
        assert result["success"] is True
        names = [c["name"] for c in result["possible_conditions"]]
        assert "Influenza" in names, f"Influenza not in results: {names}"
        assert result["possible_conditions"][0]["name"] == "Influenza"

    def test_uti_symptoms_match_uti(self) -> None:
        """Urinary symptoms must rank UTI highly."""
        result = symptom_analyzer(["burning urination", "frequent urination", "cloudy urine"])
        assert result["success"] is True
        names = [c["name"] for c in result["possible_conditions"]]
        assert "Urinary Tract Infection" in names

    def test_empty_symptoms_returns_failure(self) -> None:
        """Empty symptom list must return success=False."""
        result = symptom_analyzer([])
        assert result["success"] is False
        assert result["error"] != ""

    def test_whitespace_only_symptoms_treated_as_empty(self) -> None:
        """Symptoms that are only whitespace must be filtered out."""
        result = symptom_analyzer(["   ", "  "])
        assert result["success"] is False

    def test_conditions_sorted_by_confidence_descending(self) -> None:
        """Conditions must be returned in descending confidence score order."""
        result = symptom_analyzer(["high fever", "body aches", "chills"])
        scores = [c["confidence_score"] for c in result["possible_conditions"]]
        assert scores == sorted(scores, reverse=True), (
            "Conditions are not sorted by confidence descending"
        )

    def test_confidence_score_between_0_and_100(self) -> None:
        """All confidence scores must be in the range [0, 100]."""
        result = symptom_analyzer(["fever", "cough", "headache"])
        for cond in result["possible_conditions"]:
            assert 0.0 <= cond["confidence_score"] <= 100.0, (
                f"Score {cond['confidence_score']} out of range for {cond['name']}"
            )

    def test_matched_symptoms_subset_of_input(self) -> None:
        """matched_symptoms in each condition must be a subset of the input symptoms."""
        symptoms = ["high fever", "body aches", "cough", "nausea"]
        result = symptom_analyzer(symptoms)
        normalised = {s.strip().lower() for s in symptoms}
        for cond in result["possible_conditions"]:
            for ms in cond["matched_symptoms"]:
                assert ms in normalised, (
                    f"Matched symptom '{ms}' was not in the original input list"
                )

    def test_unknown_symptom_returns_empty_conditions(self) -> None:
        """A completely unknown symptom should match no conditions."""
        result = symptom_analyzer(["xyzzy_nonexistent_symptom_12345"])
        assert result["success"] is True
        assert result["possible_conditions"] == []

    def test_risk_level_is_valid_string(self) -> None:
        """risk_level must always be one of the four defined values."""
        valid_risk_levels = {"low", "moderate", "high", "critical", "unknown"}
        result = symptom_analyzer(["fever", "cough"])
        assert result["risk_level"] in valid_risk_levels

    def test_pneumonia_triggers_critical_risk(self) -> None:
        """Pneumonia symptoms with 'difficulty breathing' must raise risk to critical."""
        result = symptom_analyzer([
            "high fever", "productive cough", "chest pain",
            "difficulty breathing", "fatigue", "chills",
        ])
        assert result["risk_level"] in {"critical", "high"}, (
            f"Expected critical/high risk for pneumonia symptoms, got: {result['risk_level']}"
        )

    def test_emergency_flags_are_list(self) -> None:
        """emergency_flags must always be a list (even when empty)."""
        result = symptom_analyzer(["mild cough"])
        assert isinstance(result["emergency_flags"], list)

    def test_top_n_limits_results(self) -> None:
        """Requesting top_n=2 must return at most 2 conditions."""
        result = symptom_analyzer(
            ["fever", "cough", "fatigue", "nausea", "headache", "dizziness"],
            top_n=2,
        )
        assert len(result["possible_conditions"]) <= 2

    def test_condition_has_required_fields(self) -> None:
        """Each returned condition must contain all required sub-fields."""
        result = symptom_analyzer(["high fever", "body aches"])
        required_fields = {
            "name", "icd10_code", "category", "severity",
            "confidence_score", "matched_symptoms", "matched_count",
        }
        for cond in result["possible_conditions"]:
            assert required_fields.issubset(cond.keys()), (
                f"Condition '{cond.get('name')}' missing fields: "
                f"{required_fields - set(cond.keys())}"
            )

    def test_normalised_symptoms_are_lowercase(self) -> None:
        """normalised_symptoms must all be lowercase and stripped."""
        result = symptom_analyzer(["  HIGH FEVER  ", " Body Aches "])
        for ns in result["normalised_symptoms"]:
            assert ns == ns.lower().strip(), f"Not normalised: '{ns}'"

    def test_mild_cold_symptoms_give_low_or_moderate_risk(self) -> None:
        """Classic cold symptoms without emergency flags must not give critical risk."""
        result = symptom_analyzer(["runny nose", "sneezing", "mild cough"])
        assert result["risk_level"] in {"low", "moderate"}, (
            f"Unexpected risk level for mild cold: {result['risk_level']}"
        )


class TestSymptomAnalyzerAgentIntegration:
    """Integration tests for SymptomAnalyzerAgent with GlobalState."""

    def test_agent_populates_possible_conditions(self, valid_patient_file: str) -> None:
        """Agent must populate state.possible_conditions after running."""
        from agents.agent_patient_intake import PatientIntakeAgent
        state = reset_state()
        state.patient_file_path = valid_patient_file
        state = PatientIntakeAgent().run(state)
        state = SymptomAnalyzerAgent().run(state)
        assert isinstance(state.possible_conditions, list)
        assert len(state.possible_conditions) >= 1

    def test_agent_sets_top_diagnosis(self, valid_patient_file: str) -> None:
        """Agent must set state.top_diagnosis to the highest-confidence condition."""
        from agents.agent_patient_intake import PatientIntakeAgent
        state = reset_state()
        state.patient_file_path = valid_patient_file
        state = PatientIntakeAgent().run(state)
        state = SymptomAnalyzerAgent().run(state)
        assert state.top_diagnosis != ""
        assert state.top_diagnosis == state.possible_conditions[0]["name"]

    def test_agent_sets_risk_level(self, valid_patient_file: str) -> None:
        """Agent must set state.risk_level to a non-empty string."""
        from agents.agent_patient_intake import PatientIntakeAgent
        state = reset_state()
        state.patient_file_path = valid_patient_file
        state = PatientIntakeAgent().run(state)
        state = SymptomAnalyzerAgent().run(state)
        assert state.risk_level in {"low", "moderate", "high", "critical"}

    def test_agent_aborts_gracefully_without_patient_info(self) -> None:
        """Agent must return state cleanly when patient_info is empty."""
        state = reset_state()
        result_state = SymptomAnalyzerAgent().run(state)
        assert result_state is not None
        assert result_state.possible_conditions == []

    def test_agent_creates_log_entry(self, valid_patient_file: str) -> None:
        """Agent must append at least one entry to state.agent_logs."""
        from agents.agent_patient_intake import PatientIntakeAgent
        state = reset_state()
        state.patient_file_path = valid_patient_file
        state = PatientIntakeAgent().run(state)
        log_count_before = len(state.agent_logs)
        state = SymptomAnalyzerAgent().run(state)
        assert len(state.agent_logs) > log_count_before

    def test_agent_processes_pt002_uti(self, patient_pt002_path: str) -> None:
        """Agent must identify UTI-related conditions for PT002."""
        from agents.agent_patient_intake import PatientIntakeAgent
        state = reset_state()
        state.patient_file_path = patient_pt002_path
        state = PatientIntakeAgent().run(state)
        state = SymptomAnalyzerAgent().run(state)
        names = [c["name"] for c in state.possible_conditions]
        assert "Urinary Tract Infection" in names
