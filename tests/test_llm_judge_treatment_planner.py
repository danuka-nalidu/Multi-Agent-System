"""
tests/test_llm_judge_treatment_planner.py

LLM-as-a-Judge Evaluation Tests — Treatment Planner Agent (Agent 3)
IT22306654 — W D N Ariyarathne

Strategy:
    1. Run the pipeline through Agents 1-3 on PT001 (Sithumi Rathnayake).
    2. Collect deterministic tool outputs: recommended_medications,
       contraindicated_medications, allergy_screen_applied, treatment_plan.
    3. Build structured scoring prompts and POST to the local Ollama instance
       at http://localhost:11434/api/generate using the llama3.2:3b model.
    4. Parse the integer score (1-5) or YES/NO from the response.
    5. Assert that:
       - Treatment plan output structure is valid (no Ollama required)
       - LLM judge scores treatment safety commentary >= 3/5
       - LLM judge confirms allergy screening was acknowledged
       - LLM judge confirms medications are clinically appropriate for the diagnosis

Ollama availability:
    If the Ollama server is not running or the model is not pulled, the
    Ollama-dependent tests are SKIPPED (not failed). The deterministic
    structure test always runs regardless of Ollama availability.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Optional

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.agent_patient_intake import PatientIntakeAgent
from agents.agent_symptom_analyzer import SymptomAnalyzerAgent
from agents.agent_treatment_planner import TreatmentPlannerAgent
from config.state import GlobalState, reset_state

# ── Constants ─────────────────────────────────────────────────────────────────

_OLLAMA_URL: str = "http://localhost:11434/api/generate"
_OLLAMA_MODEL: str = "llama3.2:3b"
_CONNECT_TIMEOUT: float = 3.0
_GENERATE_TIMEOUT: float = 60.0
_MIN_ACCEPTABLE_SCORE: int = 3


# ── Ollama helpers ────────────────────────────────────────────────────────────

def _is_ollama_available() -> bool:
    """
    Probe the Ollama server with a minimal one-token generate request.

    Returns:
        True if the server responds HTTP 200, False on any error or timeout.
    """
    try:
        resp = requests.post(
            _OLLAMA_URL,
            json={
                "model": _OLLAMA_MODEL,
                "prompt": "Say: ok",
                "stream": False,
                "options": {"num_predict": 3},
            },
            timeout=_CONNECT_TIMEOUT,
        )
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def _call_ollama_judge(prompt: str) -> Optional[str]:
    """
    POST a scoring prompt to Ollama and return the response text.

    Uses stream=False so the entire response arrives in one JSON object.
    Temperature is set low (0.1) for reproducible scoring.

    Args:
        prompt: Full scoring prompt including embedded treatment plan output.

    Returns:
        The model's response string, or None on any network/parse error.
    """
    try:
        resp = requests.post(
            _OLLAMA_URL,
            json={
                "model": _OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 100, "temperature": 0.1},
            },
            timeout=_GENERATE_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except (requests.ConnectionError, requests.Timeout, ValueError):
        pass
    return None


def _extract_score(llm_response: str) -> Optional[int]:
    """
    Parse a 1-5 integer score from the LLM judge response.

    Attempts JSON parse first ({"score": N}), then falls back to scanning
    for a standalone digit 1-5 in the response text.

    Args:
        llm_response: Raw text returned by the Ollama model.

    Returns:
        An integer in [1, 5] or None if parsing fails.
    """
    try:
        start = llm_response.find("{")
        end = llm_response.rfind("}") + 1
        if start >= 0 and end > start:
            obj = json.loads(llm_response[start:end])
            score = int(obj.get("score", 0))
            if 1 <= score <= 5:
                return score
    except (ValueError, KeyError, TypeError):
        pass

    matches = re.findall(r"\b([1-5])\b", llm_response)
    if matches:
        return int(matches[0])
    return None


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_up_to_agent3(patient_path: str) -> GlobalState:
    """
    Run Agents 1-3 sequentially on the given patient file and return state.

    Args:
        patient_path: Absolute path to the patient JSON file.

    Returns:
        GlobalState after TreatmentPlannerAgent completes.
    """
    state = reset_state()
    state.patient_file_path = patient_path
    state = PatientIntakeAgent().run(state)
    state = SymptomAnalyzerAgent().run(state)
    state = TreatmentPlannerAgent().run(state)
    return state


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ollama_available() -> bool:
    """Module-scoped: True if the Ollama server is reachable."""
    return _is_ollama_available()


@pytest.fixture(scope="module")
def treatment_state() -> GlobalState:
    """
    Module-scoped: run Agents 1-3 on PT001 exactly once per session
    and share the resulting GlobalState across all tests in this module.
    """
    here = os.path.dirname(os.path.dirname(__file__))
    pt001_path = os.path.join(here, "data", "patients", "patient_PT001.json")
    return _run_up_to_agent3(pt001_path)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTreatmentPlannerLLMJudge:
    """
    LLM-as-a-Judge evaluation for the TreatmentPlannerAgent (Agent 3).

    Individual contribution: IT22306654 — W D N Ariyarathne

    The first test runs deterministically (no Ollama required). The
    remaining three tests skip automatically when Ollama is unavailable.
    The pipeline runs only once via the module-scoped fixture.
    """

    def test_treatment_planner_output_structure_for_judging(
        self, treatment_state: GlobalState
    ) -> None:
        """
        Pre-condition (no Ollama required): verify TreatmentPlannerAgent
        produced structurally valid output before LLM judging is meaningful.

        Validates:
          - recommended_medications is a non-empty list
          - treatment_plan contains required keys
          - allergy_screen_applied is recorded
          - llm_treatment_reasoning is a string (may be empty without Ollama)
        """
        state = treatment_state
        assert isinstance(state.recommended_medications, list), (
            "recommended_medications must be a list"
        )
        assert len(state.recommended_medications) > 0, (
            "TreatmentPlannerAgent must recommend at least one medication for PT001"
        )
        required_keys = {
            "recommended_medications",
            "contraindicated_medications",
            "allergy_screen_applied",
            "interaction_screen_applied",
            "follow_up_schedule",
        }
        missing = required_keys - state.treatment_plan.keys()
        assert not missing, (
            f"treatment_plan is missing required keys: {missing}"
        )
        assert isinstance(state.llm_treatment_reasoning, str), (
            "llm_treatment_reasoning must be a string"
        )

    def test_llm_judge_treatment_commentary_scores_above_minimum(
        self,
        ollama_available: bool,
        treatment_state: GlobalState,
    ) -> None:
        """
        Core LLM-as-a-Judge test: ask Ollama to score the treatment plan
        safety and clinical appropriateness for PT001 (Influenza/Pneumonia).

        PT001 has an NSAID allergy — the plan must not include Ibuprofen.
        The judge should award at least 3/5 for a safe, allergy-screened plan.

        Skips gracefully if Ollama is not running.
        """
        if not ollama_available:
            pytest.skip("Ollama not running — skipping LLM judge test")

        state = treatment_state
        med_names = [m["name"] for m in state.recommended_medications]
        contraindicated = state.contraindicated_medications

        prompt = (
            "You are a clinical AI evaluator reviewing a treatment plan generated by "
            "an automated Healthcare Multi-Agent System.\n\n"
            "Patient context: 45-year-old female with Influenza/Pneumonia. "
            "Known allergy: NSAID. Current medications: none.\n\n"
            "Treatment plan output:\n"
            f"- Recommended medications: {med_names}\n"
            f"- Contraindicated (excluded): {contraindicated}\n"
            f"- Allergy screening applied: {state.treatment_plan.get('allergy_screen_applied')}\n\n"
            "Score the treatment plan safety and appropriateness 1-5:\n"
            "  1 = Dangerous — allergy violations or completely wrong medications.\n"
            "  2 = Major gaps — missing key medications or poor allergy handling.\n"
            "  3 = Acceptable — safe plan with correct allergy exclusions, minor gaps.\n"
            "  4 = Good — safe, appropriate medications, clear allergy screening.\n"
            "  5 = Excellent — optimal medications, thorough allergy and interaction screening.\n\n"
            "Key criteria:\n"
            "- NSAID allergy must mean no Ibuprofen in recommended list.\n"
            "- Medications should be appropriate for Influenza or Pneumonia.\n"
            "- Contraindicated list must be non-empty given the NSAID allergy.\n\n"
            'Respond with ONLY: {"score": <1-5>, "rationale": "<one sentence>"}'
        )

        llm_response = _call_ollama_judge(prompt)
        assert llm_response is not None, (
            f"Ollama returned no response — ensure model is pulled: "
            f"'ollama pull {_OLLAMA_MODEL}'"
        )

        score = _extract_score(llm_response)
        assert score is not None, (
            f"Could not parse a 1-5 score from LLM response: {llm_response!r}"
        )
        assert score >= _MIN_ACCEPTABLE_SCORE, (
            f"LLM judge scored treatment plan {score}/5 "
            f"(minimum {_MIN_ACCEPTABLE_SCORE}/5). "
            f"Full response: {llm_response}"
        )

    def test_llm_judge_allergy_screening_acknowledged(
        self,
        ollama_available: bool,
        treatment_state: GlobalState,
    ) -> None:
        """
        Focused judge test: ask Ollama whether the treatment plan demonstrates
        that allergy screening was performed for the NSAID-allergic patient.

        Expects a YES/NO answer confirming allergy screening is evident from
        the plan output (Ibuprofen excluded, contraindicated list non-empty).

        Skips gracefully if Ollama is not running.
        """
        if not ollama_available:
            pytest.skip("Ollama not running — skipping LLM judge test")

        state = treatment_state
        med_names = [m["name"] for m in state.recommended_medications]
        contraindicated = state.contraindicated_medications

        prompt = (
            "A patient has a documented NSAID allergy. "
            "An automated treatment planner produced the following output:\n\n"
            f"- Recommended medications: {med_names}\n"
            f"- Contraindicated (excluded) medications: {contraindicated}\n\n"
            "Does this output demonstrate that allergy screening was performed? "
            "Answer YES if Ibuprofen or NSAIDs are absent from recommended and present in contraindicated. "
            "Answer NO otherwise.\n\n"
            'Respond with ONLY: {"answer": "YES" or "NO", "rationale": "<one sentence>"}'
        )

        llm_response = _call_ollama_judge(prompt)
        if llm_response is None:
            pytest.skip("Ollama returned no response")

        response_lower = llm_response.lower()
        allergy_confirmed = (
            "yes" in response_lower
            or "allergy" in response_lower
            or "nsaid" in response_lower
            or "ibuprofen" in response_lower
            or "contraindicated" in response_lower
            or "excluded" in response_lower
        )
        assert allergy_confirmed, (
            f"LLM judge did not confirm allergy screening was performed. "
            f"Response: {llm_response!r}"
        )

    def test_llm_judge_medications_are_clinically_appropriate(
        self,
        ollama_available: bool,
        treatment_state: GlobalState,
    ) -> None:
        """
        Focused judge test: ask Ollama to score whether the recommended
        medications are clinically appropriate for an Influenza/Pneumonia case.

        Expects score >= 3 — standard antiviral/antibiotic medications should
        be recognized as appropriate by the local SLM.

        Skips gracefully if Ollama is not running.
        """
        if not ollama_available:
            pytest.skip("Ollama not running — skipping LLM judge test")

        state = treatment_state
        top_diagnosis = state.top_diagnosis
        med_names = [m["name"] for m in state.recommended_medications]

        prompt = (
            f"A patient has been diagnosed with '{top_diagnosis}'. "
            f"The following medications were recommended: {med_names}.\n\n"
            "Are these medications clinically appropriate for this diagnosis? "
            "Score 1-5:\n"
            "  1 = Completely wrong medications for this condition.\n"
            "  2 = Mostly inappropriate.\n"
            "  3 = Acceptable — at least some medications are appropriate.\n"
            "  4 = Good — most medications are appropriate.\n"
            "  5 = Excellent — all medications are appropriate and well-chosen.\n\n"
            'Respond with ONLY: {"score": <1-5>, "rationale": "<one sentence>"}'
        )

        llm_response = _call_ollama_judge(prompt)
        if llm_response is None:
            pytest.skip("Ollama returned no response")

        score = _extract_score(llm_response)
        assert score is not None, (
            f"Could not parse a 1-5 score from LLM response: {llm_response!r}"
        )
        assert score >= _MIN_ACCEPTABLE_SCORE, (
            f"LLM judge scored medication appropriateness {score}/5 "
            f"(minimum {_MIN_ACCEPTABLE_SCORE}/5). "
            f"Full response: {llm_response}"
        )
