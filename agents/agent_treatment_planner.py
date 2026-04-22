"""
Agent 3: Treatment Planner Agent
Member: 

Persona
-------
A senior clinical pharmacist and treatment coordinator. Combines evidence-
based pharmacological knowledge with a safety-first mindset. Every
medication recommendation is screened against patient allergies, current
medications, and known drug interactions before being approved. Never
recommends treatment for an unconfirmed diagnosis.

Responsibilities
----------------
- Receive differential diagnoses and patient data from GlobalState.
- Identify the top conditions to treat (up to 2 from Agent 2's output).
- Run the medication_recommender tool to find safe, appropriate medications.
- Apply allergy and drug-interaction safety filters.
- Populate treatment_plan, recommended_medications, contraindicated_medications,
  and lifestyle_recommendations in GlobalState.
- Log all tool calls and outputs for observability.

Constraints
-----------
- Only treat conditions already identified in state.possible_conditions.
- Strictly enforce patient allergy exclusions — zero tolerance for allergy violations.
- Record every excluded medication with an explicit, human-readable reason.
- Lifestyle recommendations must be condition-specific, not generic platitudes.
- Always produce a log entry regardless of success or failure.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from config.state import GlobalState
from config.observability import log_agent_start, log_tool_call, log_agent_end, log_error
from tools.tool_medication_recommender import medication_recommender


class TreatmentPlannerAgent:
    """
    Agent 3 — Pharmacological Treatment Planning.

    System Prompt / Persona
    -----------------------
    You are a senior clinical pharmacist responsible for translating a
    differential diagnosis into a safe, evidence-based treatment plan. You
    consult the medication knowledge base, apply every allergy and drug-
    interaction filter, and produce a prioritised list of medications with
    precise dosage instructions. You do not guess. If you cannot find a safe
    medication for a condition, you state that explicitly. Your plan must
    include non-pharmacological lifestyle advice and a clear follow-up schedule.

    Interaction Strategy
    --------------------
    Reads from:
        state.possible_conditions  (set by Agent 2)
        state.patient_info         (set by Agent 1)

    Writes to:
        state.treatment_plan
        state.recommended_medications
        state.contraindicated_medications
        state.lifestyle_recommendations
        state.total_medications_recommended
    """

    name: str = "TreatmentPlannerAgent"

    def run(self, state: GlobalState) -> GlobalState:
        """
        Generate a safe, personalised treatment plan from the differential diagnosis.

        Steps:
            1. Guard: verify possible_conditions is populated by Agent 2.
            2. Extract top condition names, patient allergies, and current meds.
            3. Call medication_recommender tool.
            4. Write results into GlobalState.
            5. Log all actions.

        Args:
            state: The shared GlobalState from Agent 2.

        Returns:
            Updated GlobalState with treatment_plan, recommended_medications,
            contraindicated_medications, and lifestyle_recommendations populated.
        """
        start_time: float = time.time()
        task: str = (
            "Generate a personalised, allergy-screened pharmacological treatment plan."
        )

        log_agent_start(
            self.name,
            task,
            (
                f"conditions={[c['name'] for c in state.possible_conditions[:3]]}, "
                f"allergies={state.patient_info.get('allergies', [])}, "
                f"current_meds={state.patient_info.get('current_medications', [])}"
            ),
        )

        if not state.possible_conditions:
            log_error(self.name, "No conditions in state — SymptomAnalyzerAgent must run first.")
            state.log_agent_action(
                agent_name=self.name,
                action=task,
                input_summary="Empty possible_conditions",
                output_summary="Aborted — no differential diagnoses available",
                tool_calls=[],
                status="error",
            )
            return state

        top_conditions: List[str] = [
            c["name"] for c in state.possible_conditions[:2]
        ]
        allergies: List[str] = state.patient_info.get("allergies", [])
        current_medications: List[str] = state.patient_info.get("current_medications", [])

        try:
            result: Dict[str, Any] = medication_recommender(
                conditions=top_conditions,
                allergies=allergies,
                current_medications=current_medications,
            )
            log_tool_call(
                self.name,
                "medication_recommender",
                (
                    f"conditions={top_conditions}, "
                    f"allergies={allergies}, "
                    f"current_meds={current_medications}"
                ),
                (
                    f"success={result['success']}, "
                    f"recommended={len(result['recommended_medications'])}, "
                    f"contraindicated={len(result['contraindicated_medications'])}"
                ),
            )
        except Exception as exc:
            log_error(self.name, f"medication_recommender raised: {exc}")
            state.log_agent_action(
                agent_name=self.name,
                action=task,
                input_summary=f"conditions={top_conditions}",
                output_summary=f"Tool error: {exc}",
                tool_calls=["medication_recommender"],
                status="error",
            )
            return state

        state.recommended_medications = result.get("recommended_medications", [])
        state.contraindicated_medications = result.get("contraindicated_medications", [])
        state.lifestyle_recommendations = result.get("lifestyle_recommendations", [])
        state.total_medications_recommended = len(state.recommended_medications)

        state.treatment_plan = {
            "primary_conditions":           top_conditions,
            "recommended_medications":      state.recommended_medications,
            "contraindicated_medications":  state.contraindicated_medications,
            "lifestyle_recommendations":    state.lifestyle_recommendations,
            "follow_up_schedule":           result.get("follow_up_schedule", ""),
            "allergy_screen_applied":       allergies,
            "interaction_screen_applied":   current_medications,
        }

        med_names: List[str] = [
            m["name"] for m in state.recommended_medications
        ]

        output_summary: str = (
            f"Treatment plan complete. "
            f"Conditions targeted: {top_conditions}. "
            f"Medications recommended ({state.total_medications_recommended}): "
            f"{', '.join(med_names) or 'none'}. "
            f"Contraindicated: {len(state.contraindicated_medications)}. "
            f"Lifestyle tips: {len(state.lifestyle_recommendations)}."
        )

        state.log_agent_action(
            agent_name=self.name,
            action=task,
            input_summary=(
                f"conditions={top_conditions}, "
                f"allergies={allergies}, "
                f"current_meds={current_medications}"
            ),
            output_summary=output_summary,
            tool_calls=["medication_recommender"],
            status="success" if result.get("success") else "error",
        )

        log_agent_end(self.name, output_summary, time.time() - start_time)
        return state
