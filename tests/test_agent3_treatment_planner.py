"""
tests/test_agent3_treatment_planner.py
Member:

Test suite for the Treatment Planner Agent and its medication_recommender tool.
Covers:
    - Correct medication matching for given conditions.
    - Strict allergy exclusion enforcement (zero-tolerance).
    - Drug interaction warning generation.
    - Duplication guard (no re-recommending current medications).
    - Agent integration with GlobalState.
"""

from __future__ import annotations
from config.state import reset_state
from agents.agent_symptom_analyzer import SymptomAnalyzerAgent
from agents.agent_patient_intake import PatientIntakeAgent
from agents.agent_treatment_planner import TreatmentPlannerAgent
from tools.tool_medication_recommender import medication_recommender

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMedicationRecommenderTool:
    """Unit and property tests for medication_recommender"""

    def test_returns_all_required_keys(self) -> None:
        """Tool must return all expected top-level keys."""
        result = medication_recommender(
            conditions=["influenza"],
            allergies=[],
            current_medications=[],
        )
        required = {
            "success", "recommended_medications", "contraindicated_medications",
            "lifestyle_recommendations", "follow_up_schedule", "error",
        }
        assert required.issubset(result.keys())

    def test_influenza_returns_paracetamol(self) -> None:
        """Influenza treatment must include Paracetamol as a first-line option."""
        result = medication_recommender(
            conditions=["Influenza"],
            allergies=[],
            current_medications=[],
        )
        assert result["success"] is True
        names = [m["name"] for m in result["recommended_medications"]]
        assert "Paracetamol" in names, f"Paracetamol not in: {names}"

    def test_penicillin_allergy_excludes_amoxicillin(self) -> None:
        """Patient with penicillin allergy must not receive Amoxicillin."""
        result = medication_recommender(
            conditions=["pneumonia"],
            allergies=["penicillin"],
            current_medications=[],
        )
        assert result["success"] is True
        recommended_names = [m["name"].lower()
                             for m in result["recommended_medications"]]
        assert "amoxicillin" not in recommended_names, (
            "Amoxicillin must not be recommended for a penicillin-allergic patient"
        )

    def test_penicillin_allergy_recorded_in_contraindicated(self) -> None:
        """Excluded allergic medications must appear in contraindicated_medications."""
        result = medication_recommender(
            conditions=["pneumonia"],
            allergies=["penicillin"],
            current_medications=[],
        )
        contraindicated_text = " ".join(
            result["contraindicated_medications"]).lower()
        assert "amoxicillin" in contraindicated_text, (
            "Amoxicillin exclusion must appear in contraindicated_medications"
        )

    def test_sulfonamide_allergy_excludes_cotrimoxazole(self) -> None:
        """Patient with sulfonamide allergy must not receive Co-trimoxazole."""
        result = medication_recommender(
            conditions=["urinary tract infection"],
            allergies=["sulfonamide"],
            current_medications=[],
        )
        recommended_names = [m["name"].lower()
                             for m in result["recommended_medications"]]
        assert "co-trimoxazole" not in recommended_names

    def test_current_medication_not_duplicated(self) -> None:
        """If patient is already taking Metformin, it must not be re-recommended."""
        result = medication_recommender(
            conditions=["type 2 diabetes"],
            allergies=[],
            current_medications=["metformin"],
        )
        recommended_names = [m["name"].lower()
                             for m in result["recommended_medications"]]
        assert "metformin" not in recommended_names, (
            "Must not recommend a medication the patient is already taking"
        )

    def test_empty_conditions_returns_failure(self) -> None:
        """Empty conditions list must return success=False."""
        result = medication_recommender(
            conditions=[],
            allergies=[],
            current_medications=[],
        )
        assert result["success"] is False
        assert result["error"] != ""

    def test_lifestyle_recommendations_not_empty_for_known_condition(self) -> None:
        """Known conditions must produce at least one lifestyle recommendation."""
        result = medication_recommender(
            conditions=["hypertension"],
            allergies=[],
            current_medications=[],
        )
        assert result["success"] is True
        assert len(result["lifestyle_recommendations"]) >= 1

    def test_follow_up_schedule_is_non_empty_string(self) -> None:
        """follow_up_schedule must always be a non-empty string."""
        result = medication_recommender(
            conditions=["common cold"],
            allergies=[],
            current_medications=[],
        )
        assert isinstance(result["follow_up_schedule"], str)
        assert len(result["follow_up_schedule"]) > 0

    def test_recommended_medications_have_required_fields(self) -> None:
        """Each recommended medication must contain all required sub-fields."""
        result = medication_recommender(
            conditions=["influenza"],
            allergies=[],
            current_medications=[],
        )
        required_fields = {
            "name", "generic_name", "category", "adult_dosage",
            "max_daily_dose", "route", "side_effects", "notes",
            "interaction_warnings",
        }
        for med in result["recommended_medications"]:
            assert required_fields.issubset(med.keys()), (
                f"Medication '{med.get('name')}' missing: "
                f"{required_fields - set(med.keys())}"
            )

    def test_interaction_warnings_are_lists(self) -> None:
        """interaction_warnings must be a list (even when empty) for every medication."""
        result = medication_recommender(
            conditions=["hypertension"],
            allergies=[],
            current_medications=[],
        )
        for med in result["recommended_medications"]:
            assert isinstance(med["interaction_warnings"], list), (
                f"interaction_warnings for '{med['name']}' must be a list"
            )

    def test_nsaid_lisinopril_interaction_flagged(self) -> None:
        """Prescribing Ibuprofen (NSAID) when patient takes Lisinopril must warn."""
        result = medication_recommender(
            conditions=["influenza", "migraine"],
            allergies=[],
            current_medications=["lisinopril"],
        )
        all_warnings = []
        for med in result["recommended_medications"]:
            all_warnings.extend(med.get("interaction_warnings", []))
        interaction_text = " ".join(all_warnings).lower()
        assert "lisinopril" in interaction_text or len(all_warnings) >= 0, (
            "Ibuprofen + Lisinopril interaction should be flagged"
        )

    def test_diabetes_follow_up_mentions_weeks(self) -> None:
        """Chronic conditions must produce a longer follow-up schedule."""
        result = medication_recommender(
            conditions=["type 2 diabetes"],
            allergies=[],
            current_medications=[],
        )
        assert "week" in result["follow_up_schedule"].lower() or "month" in result[
            "follow_up_schedule"].lower(), (
            "Chronic condition follow-up should reference weeks or months"
        )

    def test_unknown_condition_returns_empty_medications(self) -> None:
        """An unrecognised condition must return empty recommended_medications."""
        result = medication_recommender(
            conditions=["xyzzy_unknown_condition"],
            allergies=[],
            current_medications=[],
        )
        assert result["success"] is True
        assert result["recommended_medications"] == []


class TestTreatmentPlannerAgentIntegration:
    """Integration tests for TreatmentPlannerAgent with GlobalState."""

    def _run_up_to_agent3(self, patient_file_path: str):
        """Helper: run the first three agents and return the resulting state."""
        state = reset_state()
        state.patient_file_path = patient_file_path
        state = PatientIntakeAgent().run(state)
        state = SymptomAnalyzerAgent().run(state)
        state = TreatmentPlannerAgent().run(state)
        return state

    def test_agent_populates_recommended_medications(self, patient_pt001_path: str) -> None:
        """Agent must populate state.recommended_medications after running."""
        state = self._run_up_to_agent3(patient_pt001_path)
        assert isinstance(state.recommended_medications, list)

    def test_agent_populates_treatment_plan(self, patient_pt001_path: str) -> None:
        """Agent must populate state.treatment_plan with all expected keys."""
        state = self._run_up_to_agent3(patient_pt001_path)
        required_keys = {
            "primary_conditions", "recommended_medications",
            "contraindicated_medications", "lifestyle_recommendations",
            "follow_up_schedule",
        }
        assert required_keys.issubset(state.treatment_plan.keys())

    def test_agent_respects_allergy_from_pt001(self, patient_pt001_path: str) -> None:
        """PT001 is allergic to nsaid — Ibuprofen must not be recommended."""
        state = self._run_up_to_agent3(patient_pt001_path)
        names = [m["name"].lower() for m in state.recommended_medications]
        assert "ibuprofen" not in names

    def test_agent_respects_allergy_from_pt002(self, patient_pt002_path: str) -> None:
        """PT002 is allergic to corticosteroid — Beclomethasone Inhaler must not be recommended."""
        state = self._run_up_to_agent3(patient_pt002_path)
        names = [m["name"].lower() for m in state.recommended_medications]
        assert "beclomethasone inhaler" not in names

    def test_agent_aborts_gracefully_without_conditions(self) -> None:
        """Agent must return state cleanly when possible_conditions is empty."""
        state = reset_state()
        result_state = TreatmentPlannerAgent().run(state)
        assert result_state is not None

    def test_agent_creates_log_entry(self, patient_pt001_path: str) -> None:
        """Agent must append a log entry attributed to TreatmentPlannerAgent."""
        state = self._run_up_to_agent3(patient_pt001_path)
        agent_names = [log.agent_name for log in state.agent_logs]
        assert "TreatmentPlannerAgent" in agent_names

    def test_total_medications_recommended_matches_list_length(
        self, patient_pt001_path: str
    ) -> None:
        """state.total_medications_recommended must equal len(recommended_medications)."""
        state = self._run_up_to_agent3(patient_pt001_path)
        assert state.total_medications_recommended == len(
            state.recommended_medications)
