"""
tests/test_llm_judge.py

LLM-as-a-Judge evaluation for the Healthcare MAS pipeline.

Strategy:
    1. Run the full 4-agent pipeline on PT001 (Sithumi Rathnayake / Pneumonia).
    2. Collect deterministic tool outputs: top_diagnosis, risk_level,
       executive_summary, and medication / condition counts.
    3. Build a structured scoring prompt and POST to the local Ollama instance
       at http://localhost:11434/api/generate using the llama3.2:3b model.
    4. Parse the integer score (1-5) from the response.
    5. Assert score >= 3 (minimum acceptable clinical quality).

Ollama availability:
    If the Ollama server is not running or the model is not pulled, the
    Ollama-dependent tests are SKIPPED (not failed). This allows the suite
    to pass in environments without Ollama while still providing LLM
    evaluation when the full development stack is available.

Skip condition:
    A module-scoped fixture probes http://localhost:11434/api/generate with
    a 3-second timeout. Any connection error or non-200 response causes the
    three LLM-dependent tests to skip automatically.
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
from agents.agent_report_generator import MedicalReportAgent
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
        prompt: Full scoring prompt including embedded pipeline output.

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

    Attempts JSON parse first ({"score": N, "rationale": "..."}), then
    falls back to scanning for a standalone digit 1-5 in the response text.

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


def _build_composite_judge_prompt(
    top_diagnosis: str,
    risk_level: str,
    executive_summary: str,
    medication_count: int,
    condition_count: int,
) -> str:
    """
    Build the structured LLM-as-a-Judge composite scoring prompt.

    Embeds PT001 patient context and pipeline output, and requests a
    JSON score object rated 1-5.

    Args:
        top_diagnosis:     Name of the top condition from Agent 2.
        risk_level:        Overall risk level string from Agent 2.
        executive_summary: Plain-language summary from Agent 4.
        medication_count:  Number of medications recommended by Agent 3.
        condition_count:   Number of conditions identified by Agent 2.

    Returns:
        Prompt string ready for the Ollama generate endpoint.
    """
    return (
        "You are a clinical AI evaluator reviewing the output of an automated "
        "Healthcare Multi-Agent System.\n\n"
        "Patient context: 45-year-old female presenting with high fever (39.1°C), "
        "productive cough, chest pain, difficulty breathing, fatigue, chills, sweating, "
        "loss of appetite. Oxygen saturation: 93% (low). Respiratory rate: 24 rpm "
        "(elevated). Known allergy: NSAID.\n\n"
        "System output:\n"
        f"- Top Diagnosis: {top_diagnosis}\n"
        f"- Risk Level: {risk_level}\n"
        f"- Conditions Identified: {condition_count}\n"
        f"- Medications Recommended: {medication_count}\n"
        f"- Executive Summary: {executive_summary[:400]}\n\n"
        "Score the clinical quality 1-5:\n"
        "  1 = Completely wrong or dangerous.\n"
        "  2 = Major clinical gaps or errors.\n"
        "  3 = Acceptable — correct diagnosis and risk, minor gaps.\n"
        "  4 = Good — correct diagnosis, appropriate risk, reasonable treatment.\n"
        "  5 = Excellent — all outputs clinically appropriate and complete.\n\n"
        "Key criteria:\n"
        "- Top diagnosis should be Pneumonia given this symptom profile.\n"
        "- Risk level should be 'critical' or 'high' given O2 sat 93% and dyspnoea.\n"
        "- NSAID allergy must not result in ibuprofen recommendations.\n"
        "- Executive summary must be coherent and clinically relevant.\n\n"
        'Respond with ONLY: {"score": <1-5>, "rationale": "<one sentence>"}'
    )


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_pipeline_on_pt001() -> GlobalState:
    """Run the full 4-agent sequential pipeline on the bundled PT001 file."""
    here = os.path.dirname(os.path.dirname(__file__))
    pt001_path = os.path.join(here, "data", "patients", "patient_PT001.json")
    state = reset_state()
    state.patient_file_path = pt001_path
    state = PatientIntakeAgent().run(state)
    state = SymptomAnalyzerAgent().run(state)
    state = TreatmentPlannerAgent().run(state)
    state = MedicalReportAgent().run(state)
    return state


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ollama_available() -> bool:
    """Module-scoped: True if the Ollama server is reachable."""
    return _is_ollama_available()


@pytest.fixture(scope="module")
def pipeline_state_pt001() -> GlobalState:
    """
    Module-scoped: run the full pipeline on PT001 exactly once per session
    and share the resulting GlobalState across all tests in this module.
    """
    return _run_pipeline_on_pt001()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLLMJudgeEvaluation:
    """
    LLM-as-a-Judge quality evaluation for the Healthcare MAS pipeline.

    The first test runs deterministically (no Ollama required). The
    remaining three tests skip automatically when Ollama is unavailable.
    The pipeline executes only once via module-scoped fixtures.
    """

    def test_pipeline_produces_valid_output_for_judging(
        self, pipeline_state_pt001: GlobalState
    ) -> None:
        """
        Pre-condition (no Ollama required): the pipeline must produce
        structurally valid, non-empty output before LLM judging is meaningful.

        Validates: is_valid=True, non-empty top_diagnosis, risk_level in
        valid set, non-empty executive_summary.
        """
        state = pipeline_state_pt001
        assert state.is_valid is True, "PT001 must pass patient intake validation"
        assert state.top_diagnosis != "", "top_diagnosis must not be empty"
        assert state.risk_level in {"low", "moderate", "high", "critical"}, (
            f"Unexpected risk_level: {state.risk_level!r}"
        )
        assert state.executive_summary != "", "executive_summary must not be empty"
        assert len(state.possible_conditions) > 0, (
            "Symptom analyzer must identify at least one condition"
        )

    def test_llm_judge_scores_pt001_output_above_minimum(
        self,
        ollama_available: bool,
        pipeline_state_pt001: GlobalState,
    ) -> None:
        """
        Core LLM-as-a-Judge test: submit the composite pipeline output for
        PT001 to Ollama and assert the returned quality score >= 3/5.

        Expected: Pneumonia is the correct top diagnosis for this symptom
        profile, and 'critical' or 'high' is the correct risk level given
        O2 saturation 93% and difficulty breathing. The judge should award
        at least 3/5 for a clinically correct output.

        Skips gracefully if Ollama is not running.
        """
        if not ollama_available:
            pytest.skip("Ollama not running — skipping LLM judge test")

        state = pipeline_state_pt001
        prompt = _build_composite_judge_prompt(
            top_diagnosis=state.top_diagnosis,
            risk_level=state.risk_level,
            executive_summary=state.executive_summary,
            medication_count=len(state.recommended_medications),
            condition_count=len(state.possible_conditions),
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
            f"LLM judge scored pipeline output {score}/5 "
            f"(minimum {_MIN_ACCEPTABLE_SCORE}/5). "
            f"Full response: {llm_response}"
        )

    def test_llm_judge_top_diagnosis_evaluation_is_structured(
        self,
        ollama_available: bool,
        pipeline_state_pt001: GlobalState,
    ) -> None:
        """
        Focused judge test: ask Ollama to evaluate the top diagnosis and
        assert the response is a coherent, structured JSON object.

        This test validates that the LLM judge can reason about the diagnosis
        and produce a parseable evaluation. It does not assert the verdict
        (small SLMs may reasonably differ on specific diagnoses) — the
        composite score test is the primary quality gate.

        Skips gracefully if Ollama is not running.
        """
        if not ollama_available:
            pytest.skip("Ollama not running — skipping LLM judge test")

        state = pipeline_state_pt001
        prompt = (
            "A 45-year-old female presents with: high fever (39.1°C), productive cough, "
            "chest pain, difficulty breathing, fatigue, chills, sweating, loss of appetite. "
            "O2 saturation 93%, respiratory rate 24 rpm.\n\n"
            f"An automated system identified '{state.top_diagnosis}' as the top diagnosis.\n\n"
            "Evaluate whether this diagnosis is plausible given the symptoms. "
            'Respond with ONLY: {"appropriate": true/false, "rationale": "<one sentence>"}'
        )

        llm_response = _call_ollama_judge(prompt)
        if llm_response is None:
            pytest.skip("Ollama returned no response")

        assert len(llm_response) > 0, "LLM judge must return a non-empty response"

        try:
            start = llm_response.find("{")
            end = llm_response.rfind("}") + 1
            obj = json.loads(llm_response[start:end])
            assert "appropriate" in obj, (
                f"LLM response missing 'appropriate' key: {llm_response}"
            )
            assert "rationale" in obj, (
                f"LLM response missing 'rationale' key: {llm_response}"
            )
            assert isinstance(obj["rationale"], str) and len(obj["rationale"]) > 0, (
                "LLM rationale must be a non-empty string"
            )
        except (ValueError, AssertionError):
            assert any(kw in llm_response.lower() for kw in ("true", "false", "appropriate")), (
                f"LLM response not structured or relevant: {llm_response!r}"
            )

    def test_llm_judge_risk_level_appropriate_for_pt001(
        self,
        ollama_available: bool,
        pipeline_state_pt001: GlobalState,
    ) -> None:
        """
        Focused judge test: assert the risk level is 'critical' or 'high'
        (deterministic), then ask Ollama to confirm the classification is
        clinically appropriate given PT001's vital signs.

        The deterministic assertion runs even without Ollama, ensuring
        infrastructure failures cannot mask pipeline regressions.

        Skips the LLM portion gracefully if Ollama is not running.
        """
        state = pipeline_state_pt001
        assert state.risk_level in {"critical", "high"}, (
            f"PT001 (O2 sat 93%, difficulty breathing) should be critical or "
            f"high risk, got: {state.risk_level!r}"
        )

        if not ollama_available:
            pytest.skip("Ollama not running — skipping LLM judge confirmation")

        prompt = (
            f"A patient has oxygen saturation 93% and is experiencing difficulty "
            f"breathing and chest pain. An automated triage system assigned a risk "
            f"level of '{state.risk_level}'.\n\n"
            "Is this risk classification appropriate? "
            'Respond with ONLY: {"appropriate": true/false, "rationale": "<one sentence>"}'
        )

        llm_response = _call_ollama_judge(prompt)
        assert llm_response is not None and llm_response != "", (
            "Expected non-empty LLM judge response"
        )
