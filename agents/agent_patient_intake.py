"""
Agent 1: Patient Intake Agent
Member: 

Persona
-------
A meticulous hospital admissions coordinator with 10+ years of experience in
clinical data management.  Obsessed with data completeness, accuracy, and
patient confidentiality.  Never assumes or guesses missing values — every
field is either validated or explicitly flagged as absent.

Responsibilities
----------------
- Read the patient JSON file from disk using the patient_record_reader tool.
- Validate all required fields (demographics, vitals, medical history).
- Populate patient_info, validation_report, is_valid, and validation_errors
  in the shared GlobalState.
- Log all actions for LLMOps observability.
- Pass the structured, validated state to the Symptom Analyzer Agent.

Constraints
-----------
- Never modify or infer missing patient data — only report what is present.
- Must handle file-not-found and JSON parse errors gracefully.
- validation_errors must be human-readable and actionable.
- Always produce a log entry regardless of success or failure.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from config.state import GlobalState
from config.observability import log_agent_start, log_tool_call, log_agent_end, log_error
from tools.tool_patient_reader import patient_record_reader


class PatientIntakeAgent:
    """
    Agent 1 — Patient Intake & Validation.

    System Prompt / Persona
    -----------------------
    You are a senior hospital admissions coordinator. Your singular purpose is
    to receive raw patient intake data and produce a validated, structured
    patient record for downstream clinical agents. You must flag every
    missing, out-of-range, or inconsistent data point with a precise, human-
    readable error message. You do NOT diagnose. You do NOT recommend. You
    ensure data quality so that subsequent agents can operate on a reliable
    foundation. Your output must be machine-parseable and exhaustive.

    Interaction Strategy
    --------------------
    This agent is the first node in a sequential pipeline:
        PatientIntakeAgent ──► SymptomAnalyzerAgent ──► TreatmentPlannerAgent
        ──► MedicalReportAgent

    It writes exclusively to:
        state.patient_info
        state.validation_report
        state.is_valid
        state.validation_errors
    """

    name: str = "PatientIntakeAgent"

    def run(self, state: GlobalState) -> GlobalState:
        """
        Execute patient intake: load, parse, and validate the patient record.

        Steps:
            1. Verify a patient_file_path is present in state.
            2. Call patient_record_reader tool to load and validate the file.
            3. Write structured results into GlobalState.
            4. Log all actions for observability.

        Args:
            state: The shared GlobalState object for this pipeline run.

        Returns:
            Updated GlobalState with patient_info, validation_report,
            is_valid, and validation_errors populated.
        """
        start_time: float = time.time()
        task: str = "Load and validate patient JSON record from disk."

        log_agent_start(
            self.name,
            task,
            f"patient_file_path='{state.patient_file_path}'",
        )

        if not state.patient_file_path:
            log_error(self.name, "No patient_file_path set in state.")
            state.log_agent_action(
                agent_name=self.name,
                action=task,
                input_summary="No file path provided",
                output_summary="Aborted — missing patient_file_path",
                tool_calls=[],
                status="error",
            )
            return state

        try:
            result: Dict[str, Any] = patient_record_reader(
                patient_file_path=state.patient_file_path
            )
            log_tool_call(
                self.name,
                "patient_record_reader",
                state.patient_file_path,
                (
                    f"success={result['success']}, "
                    f"valid={result['validation_passed']}, "
                    f"errors={len(result['validation_errors'])}"
                ),
            )
        except Exception as exc:
            log_error(self.name, f"patient_record_reader raised: {exc}")
            state.log_agent_action(
                agent_name=self.name,
                action=task,
                input_summary=state.patient_file_path,
                output_summary=f"Tool error: {exc}",
                tool_calls=["patient_record_reader"],
                status="error",
            )
            return state

        state.patient_info = result.get("patient_data", {})
        state.validation_report = {
            "validation_passed":  result.get("validation_passed", False),
            "validation_errors":  result.get("validation_errors", []),
            "validation_warnings": result.get("validation_warnings", []),
            "field_statuses":     result.get("field_statuses", {}),
            "loaded_at":          result.get("loaded_at", ""),
        }
        state.is_valid = result.get("validation_passed", False)
        state.validation_errors = result.get("validation_errors", [])

        patient_name: str = state.patient_info.get("name", "Unknown")
        patient_id: str = state.patient_info.get("patient_id", "Unknown")
        symptom_count: int = len(state.patient_info.get("symptoms", []))

        output_summary: str = (
            f"Patient '{patient_name}' (ID: {patient_id}) loaded. "
            f"Validation: {'PASSED' if state.is_valid else 'FAILED'}. "
            f"Symptoms reported: {symptom_count}. "
            f"Errors: {len(state.validation_errors)}."
        )

        state.log_agent_action(
            agent_name=self.name,
            action=task,
            input_summary=f"file='{state.patient_file_path}'",
            output_summary=output_summary,
            tool_calls=["patient_record_reader"],
            status="success" if result.get("success") else "error",
        )

        log_agent_end(self.name, output_summary, time.time() - start_time)
        return state
