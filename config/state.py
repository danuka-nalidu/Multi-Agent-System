"""
Global State Management — Healthcare Multi-Agent System
SE4010 CTSE Assignment 2

Defines the central GlobalState dataclass that is passed between all agents
in the sequential pipeline.  Every agent reads from and writes into this
single, shared object so that no context is ever lost between handoffs.

State lifecycle:
    reset_state()  ──→  PatientIntakeAgent  ──→  SymptomAnalyzerAgent
    ──→  TreatmentPlannerAgent  ──→  MedicalReportAgent  ──→  save_trace()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Observability helper dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentLog:
    """
    Represents one agent's action log entry for LLMOps / AgentOps tracing.

    Attributes:
        agent_name:     Identifier of the agent that produced this entry.
        action:         Short description of the task performed.
        input_summary:  Brief summary of what the agent received as input.
        output_summary: Brief summary of what the agent produced.
        tool_calls:     Ordered list of tool names invoked during this action.
        timestamp:      ISO-8601 timestamp of when the action completed.
        status:         Outcome — 'success' or 'error'.
    """

    agent_name: str
    action: str
    input_summary: str
    output_summary: str
    tool_calls: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "success"


# ─────────────────────────────────────────────────────────────────────────────
# Central Global State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GlobalState:
    """
    Central state object passed through every stage of the Healthcare MAS pipeline.

    Data is accumulated progressively:
        - Agent 1 populates: patient_info, validation_report, is_valid
        - Agent 2 populates: possible_conditions, risk_level, emergency_flags
        - Agent 3 populates: treatment_plan, recommended_medications, lifestyle_recommendations
        - Agent 4 populates: final_report, report_path, executive_summary

    The agent_logs list captures every agent action for full LLMOps traceability.
    """

    # ── Pipeline Input ────────────────────────────────────────────────────────
    patient_file_path: str = ""

    # ── Agent 1 — Patient Intake ──────────
    patient_info: Dict[str, Any] = field(default_factory=dict)
    """Structured patient record loaded from the JSON input file."""

    validation_report: Dict[str, Any] = field(default_factory=dict)
    """Field-by-field validation results produced by the patient_record_reader tool."""

    is_valid: bool = False
    """True only when all required patient fields pass validation."""

    validation_errors: List[str] = field(default_factory=list)
    """Human-readable descriptions of any validation failures found."""

    # ── Agent 2 — Symptom Analyzer ──────────────
    possible_conditions: List[Dict[str, Any]] = field(default_factory=list)
    """
    Ranked list of potential medical conditions, each containing:
    {name, icd10_code, confidence_score, matched_symptoms, severity, requires_urgent_care}
    """

    risk_level: str = "low"
    """Overall patient risk: 'low', 'moderate', 'high', or 'critical'."""

    emergency_flags: List[str] = field(default_factory=list)
    """List of symptom-based emergency indicators that require immediate attention."""

    top_diagnosis: str = ""
    """Name of the highest-confidence condition from Agent 2's analysis."""

    # ── Agent 3 — Treatment Planner ────────
    treatment_plan: Dict[str, Any] = field(default_factory=dict)
    """
    Structured treatment plan containing:
    {primary_diagnosis, recommended_medications, contraindicated_medications,
     lifestyle_recommendations, follow_up_schedule, warnings}
    """

    recommended_medications: List[Dict[str, Any]] = field(default_factory=list)
    """Safe, non-contraindicated medications selected for the patient."""

    contraindicated_medications: List[str] = field(default_factory=list)
    """Medications excluded due to patient allergies or drug interactions."""

    lifestyle_recommendations: List[str] = field(default_factory=list)
    """Non-pharmacological advice for the patient's condition."""

    # ── Agent 4 — Medical Report Generator ───────────
    final_report: str = ""
    """Full Markdown text of the generated medical report."""

    report_path: str = ""
    """Absolute or relative path to the saved report file."""

    executive_summary: str = ""
    """One-paragraph plain-language summary suitable for non-clinical stakeholders."""

    # ── LLM Reasoning Commentary (Ollama) ────────────────────────────────────
    llm_intake_reasoning: str = ""
    """Clinical commentary from Ollama LLM on the patient intake validation result."""

    llm_symptom_reasoning: str = ""
    """Clinical commentary from Ollama LLM on the differential diagnosis findings."""

    llm_treatment_reasoning: str = ""
    """Clinical commentary from Ollama LLM on the treatment plan safety and rationale."""

    llm_report_reasoning: str = ""
    """Clinical commentary from Ollama LLM on the final report and executive summary."""

    # ── LLMOps / AgentOps Observability ──────────────────────────────────────
    agent_logs: List[AgentLog] = field(default_factory=list)
    pipeline_start: str = field(default_factory=lambda: datetime.now().isoformat())
    pipeline_end: Optional[str] = None
    total_conditions_found: int = 0
    total_medications_recommended: int = 0

    def log_agent_action(
        self,
        agent_name: str,
        action: str,
        input_summary: str,
        output_summary: str,
        tool_calls: List[str],
        status: str = "success",
    ) -> None:
        """
        Append a structured log entry for an agent's completed action.

        Args:
            agent_name:     Name of the calling agent.
            action:         Brief task description.
            input_summary:  Summary of what the agent was given.
            output_summary: Summary of what the agent produced.
            tool_calls:     List of tool names the agent invoked.
            status:         'success' or 'error'.
        """
        entry = AgentLog(
            agent_name=agent_name,
            action=action,
            input_summary=input_summary,
            output_summary=output_summary,
            tool_calls=tool_calls,
            status=status,
        )
        self.agent_logs.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the full pipeline state to a JSON-compatible dictionary.

        Returns:
            Dict containing all pipeline inputs, outputs, and agent logs.
        """
        return {
            "pipeline_start": self.pipeline_start,
            "pipeline_end": self.pipeline_end,
            "patient_file_path": self.patient_file_path,
            "patient_name": self.patient_info.get("name", "Unknown"),
            "patient_id": self.patient_info.get("patient_id", "Unknown"),
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "risk_level": self.risk_level,
            "emergency_flags": self.emergency_flags,
            "top_diagnosis": self.top_diagnosis,
            "possible_conditions": self.possible_conditions,
            "recommended_medications": self.recommended_medications,
            "contraindicated_medications": self.contraindicated_medications,
            "lifestyle_recommendations": self.lifestyle_recommendations,
            "total_conditions_found": self.total_conditions_found,
            "total_medications_recommended": self.total_medications_recommended,
            "report_path": self.report_path,
            "executive_summary": self.executive_summary,
            "llm_intake_reasoning":    self.llm_intake_reasoning,
            "llm_symptom_reasoning":   self.llm_symptom_reasoning,
            "llm_treatment_reasoning": self.llm_treatment_reasoning,
            "llm_report_reasoning":    self.llm_report_reasoning,
            "agent_logs": [
                {
                    "agent": log.agent_name,
                    "action": log.action,
                    "input": log.input_summary,
                    "output": log.output_summary,
                    "tools": log.tool_calls,
                    "status": log.status,
                    "timestamp": log.timestamp,
                }
                for log in self.agent_logs
            ],
        }

    def save_trace(self, logs_dir: str = "logs") -> str:
        """
        Persist the complete LLMOps pipeline trace to a timestamped JSON file.

        Args:
            logs_dir: Directory in which to save the trace file.

        Returns:
            File path of the saved trace.
        """
        os.makedirs(logs_dir, exist_ok=True)
        self.pipeline_end = datetime.now().isoformat()
        filename = (
            f"{logs_dir}/trace_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filename, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return filename


# ─────────────────────────────────────────────────────────────────────────────
# Singleton helpers
# ─────────────────────────────────────────────────────────────────────────────

_state: Optional[GlobalState] = None


def get_state() -> GlobalState:
    """
    Return the current global state, initialising it if not yet created.

    Returns:
        The active GlobalState instance.
    """
    global _state
    if _state is None:
        _state = GlobalState()
    return _state


def reset_state() -> GlobalState:
    """
    Create and return a fresh GlobalState, discarding any previous run.

    Returns:
        A newly initialised GlobalState instance.
    """
    global _state
    _state = GlobalState()
    return _state
