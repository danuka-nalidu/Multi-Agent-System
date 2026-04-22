"""
tests/test_agent4_report_generator.py
Member: 

Test suite for the Medical Report Agent and its medical_report_generator tool.
Covers:
    - Report file creation and persistence.
    - Markdown structure validation (all required sections present).
    - JSON summary creation.
    - Executive summary content correctness.
    - Agent integration: full pipeline run and report verification.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.tool_report_generator import medical_report_generator
from agents.agent_report_generator import MedicalReportAgent
from agents.agent_patient_intake import PatientIntakeAgent
from agents.agent_symptom_analyzer import SymptomAnalyzerAgent
from agents.agent_treatment_planner import TreatmentPlannerAgent
from config.state import reset_state


SAMPLE_PATIENT_INFO = {
    "patient_id": "PT999",
    "name": "Test Patient",
    "age": 30,
    "gender": "Female",
    "blood_type": "O+",
    "chief_complaint": "Fever and aches",
    "symptoms": ["high fever", "body aches", "fatigue"],
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
    "registration_date": "2026-04-22",
}

SAMPLE_VALIDATION_REPORT = {
    "validation_passed": True,
    "validation_errors": [],
    "validation_warnings": [],
    "field_statuses": {},
    "loaded_at": "2026-04-22T12:00:00",
}

SAMPLE_CONDITIONS = [
    {
        "name": "Influenza",
        "icd10_code": "J11.1",
        "category": "Respiratory",
        "severity": "moderate",
        "confidence_score": 75.0,
        "matched_symptoms": ["high fever", "body aches", "fatigue"],
        "matched_count": 3,
        "requires_urgent_care": False,
        "typical_duration": "1-2 weeks",
        "description": "Influenza infection.",
    }
]

SAMPLE_MEDICATIONS = [
    {
        "name": "Paracetamol",
        "generic_name": "Acetaminophen",
        "category": "Analgesic",
        "conditions_treated": ["influenza"],
        "adult_dosage": "500-1000mg every 4-6 hours",
        "max_daily_dose": "4000mg",
        "route": "Oral",
        "side_effects": ["rare liver toxicity"],
        "notes": "First-line antipyretic.",
        "interaction_warnings": [],
    }
]


class TestMedicalReportGeneratorTool:
    """Unit tests for medical_report_generator (Student 4)."""

    def test_returns_all_required_keys(self, tmp_path) -> None:
        """Tool must return all required keys."""
        result = medical_report_generator(
            patient_info=SAMPLE_PATIENT_INFO,
            validation_report=SAMPLE_VALIDATION_REPORT,
            possible_conditions=SAMPLE_CONDITIONS,
            risk_level="moderate",
            emergency_flags=[],
            treatment_plan={},
            recommended_medications=SAMPLE_MEDICATIONS,
            contraindicated_medications=[],
            lifestyle_recommendations=["Rest and hydrate."],
            output_dir=str(tmp_path),
        )
        required = {
            "success", "report_path", "json_path",
            "executive_summary", "total_conditions",
            "total_medications", "generated_at", "error",
        }
        assert required.issubset(result.keys())

    def test_report_file_is_created(self, tmp_path) -> None:
        """Tool must create a Markdown report file on disk."""
        result = medical_report_generator(
            patient_info=SAMPLE_PATIENT_INFO,
            validation_report=SAMPLE_VALIDATION_REPORT,
            possible_conditions=SAMPLE_CONDITIONS,
            risk_level="moderate",
            emergency_flags=[],
            treatment_plan={},
            recommended_medications=SAMPLE_MEDICATIONS,
            contraindicated_medications=[],
            lifestyle_recommendations=[],
            output_dir=str(tmp_path),
        )
        assert result["success"] is True
        assert os.path.isfile(result["report_path"]), (
            f"Report file not found at: {result['report_path']}"
        )

    def test_report_file_ends_with_md(self, tmp_path) -> None:
        """Report file must have a .md extension."""
        result = medical_report_generator(
            patient_info=SAMPLE_PATIENT_INFO,
            validation_report=SAMPLE_VALIDATION_REPORT,
            possible_conditions=SAMPLE_CONDITIONS,
            risk_level="low",
            emergency_flags=[],
            treatment_plan={},
            recommended_medications=[],
            contraindicated_medications=[],
            lifestyle_recommendations=[],
            output_dir=str(tmp_path),
        )
        assert result["report_path"].endswith(".md")

    def test_json_summary_file_is_created(self, tmp_path) -> None:
        """Tool must also create a JSON summary file."""
        result = medical_report_generator(
            patient_info=SAMPLE_PATIENT_INFO,
            validation_report=SAMPLE_VALIDATION_REPORT,
            possible_conditions=SAMPLE_CONDITIONS,
            risk_level="moderate",
            emergency_flags=[],
            treatment_plan={},
            recommended_medications=SAMPLE_MEDICATIONS,
            contraindicated_medications=[],
            lifestyle_recommendations=[],
            output_dir=str(tmp_path),
        )
        assert os.path.isfile(result["json_path"])
        with open(result["json_path"], "r", encoding="utf-8") as fh:
            summary = json.load(fh)
        assert summary["patient_id"] == "PT999"

    def test_report_contains_patient_name(self, tmp_path) -> None:
        """The generated Markdown must mention the patient's name."""
        result = medical_report_generator(
            patient_info=SAMPLE_PATIENT_INFO,
            validation_report=SAMPLE_VALIDATION_REPORT,
            possible_conditions=SAMPLE_CONDITIONS,
            risk_level="moderate",
            emergency_flags=[],
            treatment_plan={},
            recommended_medications=SAMPLE_MEDICATIONS,
            contraindicated_medications=[],
            lifestyle_recommendations=[],
            output_dir=str(tmp_path),
        )
        with open(result["report_path"], "r", encoding="utf-8") as fh:
            content = fh.read()
        assert "Test Patient" in content

    def test_report_contains_diagnosis_section(self, tmp_path) -> None:
        """Report must contain the Differential Diagnosis Analysis section."""
        result = medical_report_generator(
            patient_info=SAMPLE_PATIENT_INFO,
            validation_report=SAMPLE_VALIDATION_REPORT,
            possible_conditions=SAMPLE_CONDITIONS,
            risk_level="moderate",
            emergency_flags=[],
            treatment_plan={},
            recommended_medications=SAMPLE_MEDICATIONS,
            contraindicated_medications=[],
            lifestyle_recommendations=[],
            output_dir=str(tmp_path),
        )
        with open(result["report_path"], "r", encoding="utf-8") as fh:
            content = fh.read()
        assert "Differential Diagnosis" in content

    def test_report_contains_treatment_section(self, tmp_path) -> None:
        """Report must contain a Treatment Plan section."""
        result = medical_report_generator(
            patient_info=SAMPLE_PATIENT_INFO,
            validation_report=SAMPLE_VALIDATION_REPORT,
            possible_conditions=SAMPLE_CONDITIONS,
            risk_level="moderate",
            emergency_flags=[],
            treatment_plan={},
            recommended_medications=SAMPLE_MEDICATIONS,
            contraindicated_medications=[],
            lifestyle_recommendations=[],
            output_dir=str(tmp_path),
        )
        with open(result["report_path"], "r", encoding="utf-8") as fh:
            content = fh.read()
        assert "Treatment Plan" in content

    def test_executive_summary_contains_top_diagnosis(self, tmp_path) -> None:
        """Executive summary must mention the top diagnosis condition name."""
        result = medical_report_generator(
            patient_info=SAMPLE_PATIENT_INFO,
            validation_report=SAMPLE_VALIDATION_REPORT,
            possible_conditions=SAMPLE_CONDITIONS,
            risk_level="moderate",
            emergency_flags=[],
            treatment_plan={},
            recommended_medications=SAMPLE_MEDICATIONS,
            contraindicated_medications=[],
            lifestyle_recommendations=[],
            output_dir=str(tmp_path),
        )
        assert "Influenza" in result["executive_summary"]

    def test_total_conditions_matches_input(self, tmp_path) -> None:
        """total_conditions in result must equal len(possible_conditions)."""
        result = medical_report_generator(
            patient_info=SAMPLE_PATIENT_INFO,
            validation_report=SAMPLE_VALIDATION_REPORT,
            possible_conditions=SAMPLE_CONDITIONS,
            risk_level="low",
            emergency_flags=[],
            treatment_plan={},
            recommended_medications=[],
            contraindicated_medications=[],
            lifestyle_recommendations=[],
            output_dir=str(tmp_path),
        )
        assert result["total_conditions"] == len(SAMPLE_CONDITIONS)

    def test_total_medications_matches_input(self, tmp_path) -> None:
        """total_medications in result must equal len(recommended_medications)."""
        result = medical_report_generator(
            patient_info=SAMPLE_PATIENT_INFO,
            validation_report=SAMPLE_VALIDATION_REPORT,
            possible_conditions=[],
            risk_level="low",
            emergency_flags=[],
            treatment_plan={},
            recommended_medications=SAMPLE_MEDICATIONS,
            contraindicated_medications=[],
            lifestyle_recommendations=[],
            output_dir=str(tmp_path),
        )
        assert result["total_medications"] == len(SAMPLE_MEDICATIONS)

    def test_empty_output_dir_is_created_automatically(self, tmp_path) -> None:
        """Tool must create output_dir if it does not exist."""
        new_dir = str(tmp_path / "auto_created_dir")
        result = medical_report_generator(
            patient_info=SAMPLE_PATIENT_INFO,
            validation_report=SAMPLE_VALIDATION_REPORT,
            possible_conditions=[],
            risk_level="low",
            emergency_flags=[],
            treatment_plan={},
            recommended_medications=[],
            contraindicated_medications=[],
            lifestyle_recommendations=[],
            output_dir=new_dir,
        )
        assert result["success"] is True
        assert os.path.isdir(new_dir)


class TestMedicalReportAgentIntegration:
    """Integration tests for MedicalReportAgent with full pipeline GlobalState."""

    def _run_full_pipeline(self, patient_file_path: str):
        """Helper: run all four agents and return the final state."""
        state = reset_state()
        state.patient_file_path = patient_file_path
        state = PatientIntakeAgent().run(state)
        state = SymptomAnalyzerAgent().run(state)
        state = TreatmentPlannerAgent().run(state)
        state = MedicalReportAgent().run(state)
        return state

    def test_full_pipeline_creates_report_file(self, patient_pt001_path: str) -> None:
        """Full pipeline must save a report file that exists on disk."""
        state = self._run_full_pipeline(patient_pt001_path)
        assert state.report_path != ""
        assert os.path.isfile(state.report_path), (
            f"Report file not found: {state.report_path}"
        )

    def test_full_pipeline_sets_executive_summary(self, patient_pt001_path: str) -> None:
        """Full pipeline must set a non-empty executive_summary in state."""
        state = self._run_full_pipeline(patient_pt001_path)
        assert state.executive_summary != ""
        assert len(state.executive_summary) > 50

    def test_full_pipeline_creates_log_entries_for_all_agents(
        self, patient_pt001_path: str
    ) -> None:
        """All four agents must have entries in state.agent_logs."""
        state = self._run_full_pipeline(patient_pt001_path)
        agent_names = {log.agent_name for log in state.agent_logs}
        expected = {
            "PatientIntakeAgent",
            "SymptomAnalyzerAgent",
            "TreatmentPlannerAgent",
            "MedicalReportAgent",
        }
        assert expected.issubset(agent_names), (
            f"Missing log entries for: {expected - agent_names}"
        )

    def test_full_pipeline_saves_trace_to_logs(self, patient_pt001_path: str) -> None:
        """Full pipeline must create at least one trace JSON file in logs/."""
        state = self._run_full_pipeline(patient_pt001_path)
        assert os.path.isdir("logs"), "logs/ directory must exist after pipeline run"
        trace_files = [f for f in os.listdir("logs") if f.startswith("trace_")]
        assert len(trace_files) >= 1
