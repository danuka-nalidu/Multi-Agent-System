"""
tests/test_pipeline_integration.py
Group Integration Tests for the Full 4-Agent Pipeline
Member:

End-to-end integration tests that validate the complete 4-agent pipeline
as a unified system. These tests confirm that:
    - State is preserved and accumulated correctly across all agent handoffs.
    - No context is lost between agents.
    - Both sample patient files produce valid, complete reports.
    - The pipeline handles edge cases gracefully without crashing.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.agent_patient_intake import PatientIntakeAgent
from agents.agent_symptom_analyzer import SymptomAnalyzerAgent
from agents.agent_treatment_planner import TreatmentPlannerAgent
from agents.agent_report_generator import MedicalReportAgent
from config.state import reset_state, GlobalState


def run_full_pipeline(patient_file_path: str) -> GlobalState:
    """Execute all 4 agents in sequence and return the final GlobalState."""
    state = reset_state()
    state.patient_file_path = patient_file_path
    state = PatientIntakeAgent().run(state)
    state = SymptomAnalyzerAgent().run(state)
    state = TreatmentPlannerAgent().run(state)
    state = MedicalReportAgent().run(state)
    return state


class TestFullPipelinePT001:
    """End-to-end tests for the full pipeline using patient PT001 (Influenza case)."""

    def test_pipeline_runs_without_exception(self, patient_pt001_path: str) -> None:
        """The complete 4-agent pipeline must complete without raising any exception."""
        state = run_full_pipeline(patient_pt001_path)
        assert state is not None

    def test_patient_info_populated_after_pipeline(self, patient_pt001_path: str) -> None:
        """state.patient_info must be populated after the full run."""
        state = run_full_pipeline(patient_pt001_path)
        assert state.patient_info.get("patient_id") == "PT001"
        assert state.patient_info.get("name") == "Amara Perera"

    def test_conditions_found_after_pipeline(self, patient_pt001_path: str) -> None:
        """state.possible_conditions must contain at least one entry."""
        state = run_full_pipeline(patient_pt001_path)
        assert len(state.possible_conditions) >= 1

    def test_influenza_is_top_diagnosis_for_pt001(self, patient_pt001_path: str) -> None:
        """PT001 classic influenza symptoms must produce Influenza as top diagnosis."""
        state = run_full_pipeline(patient_pt001_path)
        assert state.top_diagnosis == "Influenza", (
            f"Expected 'Influenza', got '{state.top_diagnosis}'"
        )

    def test_medications_recommended_for_pt001(self, patient_pt001_path: str) -> None:
        """At least one medication must be recommended for PT001."""
        state = run_full_pipeline(patient_pt001_path)
        assert len(state.recommended_medications) >= 1

    def test_penicillin_allergy_respected_for_pt001(self, patient_pt001_path: str) -> None:
        """PT001 is allergic to penicillin — Amoxicillin must not appear."""
        state = run_full_pipeline(patient_pt001_path)
        names = [m["name"].lower() for m in state.recommended_medications]
        assert "amoxicillin" not in names

    def test_report_file_created_for_pt001(self, patient_pt001_path: str) -> None:
        """A report file must be saved to disk after the full pipeline."""
        state = run_full_pipeline(patient_pt001_path)
        assert state.report_path != ""
        assert os.path.isfile(state.report_path)

    def test_all_four_agents_logged_for_pt001(self, patient_pt001_path: str) -> None:
        """All four agent names must appear in state.agent_logs."""
        state = run_full_pipeline(patient_pt001_path)
        logged_agents = {log.agent_name for log in state.agent_logs}
        expected = {
            "PatientIntakeAgent", "SymptomAnalyzerAgent",
            "TreatmentPlannerAgent", "MedicalReportAgent",
        }
        assert expected.issubset(logged_agents)

    def test_state_to_dict_is_json_serialisable(self, patient_pt001_path: str) -> None:
        """state.to_dict() must produce a JSON-serialisable dictionary."""
        import json
        state = run_full_pipeline(patient_pt001_path)
        try:
            serialised = json.dumps(state.to_dict())
            assert len(serialised) > 100
        except (TypeError, ValueError) as exc:
            pytest.fail(f"state.to_dict() is not JSON-serialisable: {exc}")

    def test_pipeline_end_timestamp_set(self, patient_pt001_path: str) -> None:
        """state.pipeline_end must be set after the pipeline completes."""
        state = run_full_pipeline(patient_pt001_path)
        assert state.pipeline_end is not None
        assert len(state.pipeline_end) >= 19


class TestFullPipelinePT002:
    """End-to-end tests using patient PT002 (UTI case with existing medications)."""

    def test_pipeline_runs_without_exception_pt002(self, patient_pt002_path: str) -> None:
        """Pipeline must complete without exception for PT002."""
        state = run_full_pipeline(patient_pt002_path)
        assert state is not None

    def test_uti_detected_for_pt002(self, patient_pt002_path: str) -> None:
        """PT002 UTI symptoms must result in UTI as a top diagnosis."""
        state = run_full_pipeline(patient_pt002_path)
        names = [c["name"] for c in state.possible_conditions]
        assert "Urinary Tract Infection" in names

    def test_sulfonamide_allergy_respected_for_pt002(self, patient_pt002_path: str) -> None:
        """PT002 is allergic to sulfonamide — Co-trimoxazole must not be recommended."""
        state = run_full_pipeline(patient_pt002_path)
        names = [m["name"].lower() for m in state.recommended_medications]
        assert "co-trimoxazole" not in names

    def test_current_medications_not_duplicated_for_pt002(
        self, patient_pt002_path: str
    ) -> None:
        """PT002 takes metformin and lisinopril — neither must be re-recommended."""
        state = run_full_pipeline(patient_pt002_path)
        names = [m["name"].lower() for m in state.recommended_medications]
        assert "metformin" not in names
        assert "lisinopril" not in names

    def test_report_file_created_for_pt002(self, patient_pt002_path: str) -> None:
        """A report file must be saved to disk after processing PT002."""
        state = run_full_pipeline(patient_pt002_path)
        assert os.path.isfile(state.report_path)


class TestPipelineStateManagement:
    """Tests specifically validating GlobalState integrity across agent handoffs."""

    def test_state_is_passed_by_reference_through_pipeline(
        self, patient_pt001_path: str
    ) -> None:
        """Each agent must return the same state object, not a copy."""
        state = reset_state()
        state.patient_file_path = patient_pt001_path
        id_before = id(state)

        state = PatientIntakeAgent().run(state)
        assert id(state) == id_before, "Agent 1 must return the same state object"

        state = SymptomAnalyzerAgent().run(state)
        assert id(state) == id_before, "Agent 2 must return the same state object"

    def test_agent1_output_consumed_by_agent2(self, patient_pt001_path: str) -> None:
        """Agent 2 must use the patient_info set by Agent 1."""
        state = reset_state()
        state.patient_file_path = patient_pt001_path
        state = PatientIntakeAgent().run(state)
        patient_name = state.patient_info.get("name")
        state = SymptomAnalyzerAgent().run(state)
        assert state.patient_info.get("name") == patient_name, (
            "Agent 2 must not overwrite patient_info set by Agent 1"
        )

    def test_agent2_output_consumed_by_agent3(self, patient_pt001_path: str) -> None:
        """Agent 3 must treat the conditions identified by Agent 2."""
        state = reset_state()
        state.patient_file_path = patient_pt001_path
        state = PatientIntakeAgent().run(state)
        state = SymptomAnalyzerAgent().run(state)
        top_condition = state.top_diagnosis
        state = TreatmentPlannerAgent().run(state)
        plan_conditions = [c.lower() for c in state.treatment_plan.get("primary_conditions", [])]
        assert any(top_condition.lower() in c or c in top_condition.lower()
                   for c in plan_conditions), (
            f"Agent 3 must treat Agent 2's top condition '{top_condition}'"
        )

    def test_no_agent_logs_lost_between_handoffs(self, patient_pt001_path: str) -> None:
        """Agent logs must accumulate monotonically — earlier logs must not be dropped."""
        state = reset_state()
        state.patient_file_path = patient_pt001_path
        state = PatientIntakeAgent().run(state)
        count_after_1 = len(state.agent_logs)
        state = SymptomAnalyzerAgent().run(state)
        assert len(state.agent_logs) >= count_after_1, (
            "Agent 2 must not discard Agent 1's log entries"
        )
        state = TreatmentPlannerAgent().run(state)
        count_after_3 = len(state.agent_logs)
        state = MedicalReportAgent().run(state)
        assert len(state.agent_logs) >= count_after_3, (
            "Agent 4 must not discard earlier log entries"
        )
