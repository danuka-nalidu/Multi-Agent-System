"""
Agent 2: Symptom Analyzer Agent
Member: 

Persona
-------
A board-certified clinical diagnostician with expertise in evidence-based
differential diagnosis.  Methodical and data-driven — every conclusion is
supported by symptom-to-condition matching evidence.  Immediately escalates
any patient presenting with emergency indicators, regardless of overall score.

Responsibilities
----------------
- Receive the validated patient record from Agent 1 via GlobalState.
- Extract the patient's reported symptoms.
- Run the symptom_analyzer tool against the local medical knowledge base.
- Populate possible_conditions, risk_level, emergency_flags, and top_diagnosis
  in GlobalState.
- Log all tool calls and outputs for observability.

Constraints
-----------
- Only work on symptoms already present in state.patient_info (from Agent 1).
- Never fabricate conditions not present in the knowledge base.
- Must produce at least one condition entry if any symptoms matched.
- Emergency flags must trigger a 'critical' or 'high' risk_level.
- Always produce a log entry regardless of outcome.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from config.state import GlobalState
from config.observability import log_agent_start, log_tool_call, log_agent_end, log_error
from config.llm_client import get_llm_commentary
from tools.tool_symptom_analyzer import symptom_analyzer


class SymptomAnalyzerAgent:
    """
    Agent 2 — Clinical Symptom Analysis & Differential Diagnosis.

    System Prompt / Persona
    -----------------------
    You are a senior clinical diagnostician specialising in differential
    diagnosis. You receive a validated patient record and use symptom-matching
    evidence from the medical knowledge base to produce a ranked list of the
    most probable conditions. Every condition you report must have at least one
    directly matched symptom. You explicitly identify emergency indicators and
    escalate the risk level accordingly. You do NOT prescribe treatment — that
    responsibility belongs to the downstream Treatment Planner Agent.

    Interaction Strategy
    --------------------
    Reads exclusively from:
        state.patient_info   (set by Agent 1)

    Writes exclusively to:
        state.possible_conditions
        state.risk_level
        state.emergency_flags
        state.top_diagnosis
        state.total_conditions_found
    """

    name: str = "SymptomAnalyzerAgent"

    def run(self, state: GlobalState) -> GlobalState:
        """
        Analyse the patient's symptoms and populate differential diagnoses.

        Steps:
            1. Guard: verify patient_info is populated by Agent 1.
            2. Extract normalised symptom list from patient record.
            3. Call symptom_analyzer tool.
            4. Write results into GlobalState.
            5. Log all actions.

        Args:
            state: The shared GlobalState from Agent 1.

        Returns:
            Updated GlobalState with possible_conditions, risk_level,
            emergency_flags, and top_diagnosis populated.
        """
        start_time: float = time.time()
        task: str = "Analyse patient symptoms and produce ranked differential diagnosis."

        log_agent_start(
            self.name,
            task,
            f"patient='{state.patient_info.get('name', 'Unknown')}', "
            f"symptoms={state.patient_info.get('symptoms', [])}",
        )

        if not state.patient_info:
            log_error(self.name, "patient_info is empty — PatientIntakeAgent must run first.")
            state.log_agent_action(
                agent_name=self.name,
                action=task,
                input_summary="Empty patient_info",
                output_summary="Aborted — no patient data in state",
                tool_calls=[],
                status="error",
            )
            return state

        symptoms: List[str] = state.patient_info.get("symptoms", [])
        if not symptoms:
            log_error(self.name, "Patient has no reported symptoms — cannot analyse.")
            state.log_agent_action(
                agent_name=self.name,
                action=task,
                input_summary=f"patient='{state.patient_info.get('name', 'Unknown')}'",
                output_summary="Aborted — empty symptoms list",
                tool_calls=[],
                status="error",
            )
            return state

        try:
            result: Dict[str, Any] = symptom_analyzer(symptoms=symptoms)
            log_tool_call(
                self.name,
                "symptom_analyzer",
                f"symptoms={symptoms}",
                (
                    f"success={result['success']}, "
                    f"conditions_found={len(result['possible_conditions'])}, "
                    f"risk={result['risk_level']}, "
                    f"emergency_flags={result['emergency_flags']}"
                ),
            )
        except Exception as exc:
            log_error(self.name, f"symptom_analyzer raised: {exc}")
            state.log_agent_action(
                agent_name=self.name,
                action=task,
                input_summary=f"symptoms={symptoms}",
                output_summary=f"Tool error: {exc}",
                tool_calls=["symptom_analyzer"],
                status="error",
            )
            return state

        state.possible_conditions = result.get("possible_conditions", [])
        state.risk_level = result.get("risk_level", "low")
        state.emergency_flags = result.get("emergency_flags", [])
        state.total_conditions_found = len(state.possible_conditions)

        if state.possible_conditions:
            state.top_diagnosis = state.possible_conditions[0]["name"]
        else:
            state.top_diagnosis = "Undetermined"

        top_n: List[str] = [
            f"{c['name']} ({c['confidence_score']}%)"
            for c in state.possible_conditions[:3]
        ]

        output_summary: str = (
            f"Analysis complete. {state.total_conditions_found} condition(s) matched. "
            f"Top diagnoses: {', '.join(top_n) or 'none'}. "
            f"Risk level: {state.risk_level.upper()}. "
            f"Emergency flags: {state.emergency_flags or 'none'}."
        )

        state.log_agent_action(
            agent_name=self.name,
            action=task,
            input_summary=f"symptoms={symptoms}",
            output_summary=output_summary,
            tool_calls=["symptom_analyzer"],
            status="success" if result.get("success") else "error",
        )

        log_agent_end(self.name, output_summary, time.time() - start_time)

        # ── LLM Commentary (gracefully degrades if Ollama is not running) ─────
        state.llm_symptom_reasoning = get_llm_commentary(
            agent_name=self.name,
            system_prompt=(
                "You are a board-certified clinical diagnostician. "
                "Given the symptom analysis results, provide a concise 2-3 sentence "
                "clinical reasoning on the differential diagnosis and risk level."
            ),
            user_message=(
                f"Symptoms: {state.patient_info.get('symptoms', [])}\n"
                f"Top conditions: {[(c['name'], c['confidence_score']) for c in state.possible_conditions[:3]]}\n"
                f"Risk level: {state.risk_level}\n"
                f"Emergency flags: {state.emergency_flags or 'none'}"
            ),
        )

        return state
