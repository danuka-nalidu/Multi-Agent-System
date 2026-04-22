"""
Tool: symptom_analyzer
Member: 
Agent:  Symptom Analyzer Agent

Matches a patient's reported symptoms against a local medical knowledge base
(symptoms_db.json) using a weighted overlap scoring algorithm and returns a
ranked list of probable conditions with confidence percentages.

The tool also detects emergency indicators — symptoms that, when present,
should trigger an immediate escalation regardless of overall score.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

_DEFAULT_DB_PATH: str = os.path.join(
    os.path.dirname(__file__), "..", "data", "symptoms_db.json"
)


def symptom_analyzer(
    symptoms: List[str],
    symptoms_db_path: str = _DEFAULT_DB_PATH,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Analyse a patient's reported symptoms and rank possible medical conditions.

    Algorithm
    ---------
    For each condition in the knowledge base a *match score* is calculated as:

        score = matched_count / max(total_patient_symptoms, total_condition_symptoms)

    This is a normalised overlap that penalises both over-specific matches
    (many condition symptoms not present in the patient) and under-specific
    ones (patient has many symptoms the condition does not explain).

    Emergency detection
    -------------------
    Each condition declares a list of ``emergency_indicators``.  If any of the
    patient's symptoms textually overlap with an indicator the function flags
    that symptom and raises the overall ``risk_level`` accordingly.

    Args:
        symptoms:          List of symptom strings exactly as reported by the patient.
        symptoms_db_path:  Path to the JSON knowledge base.  Defaults to the
                           bundled ``data/symptoms_db.json``.
        top_n:             Maximum number of conditions to return (default 5).

    Returns:
        Dict with the following keys:
            - success (bool):                 True when the DB loaded successfully.
            - possible_conditions (List[Dict]): Ranked conditions (highest score first).
              Each entry contains:
                  {name, icd10_code, category, severity, confidence_score,
                   matched_symptoms, matched_count, requires_urgent_care}
            - risk_level (str):               Overall risk: 'low', 'moderate',
                                              'high', or 'critical'.
            - emergency_flags (List[str]):     Symptom strings that matched an
                                              emergency indicator.
            - normalised_symptoms (List[str]): Lower-cased, stripped input list.
            - error (str):                    Non-empty only when success is False.

    Example:
        >>> result = symptom_analyzer(["high fever", "body aches", "chills"])
        >>> result["possible_conditions"][0]["name"]
        'Influenza'
        >>> result["risk_level"]
        'moderate'
    """
    normalised: List[str] = [s.strip().lower() for s in symptoms if s.strip()]

    if not normalised:
        return _error_result("No symptoms provided — list is empty after normalisation.")

    db_path = os.path.normpath(symptoms_db_path)
    if not os.path.isfile(db_path):
        return _error_result(f"Symptoms knowledge base not found at: '{db_path}'")

    try:
        with open(db_path, "r", encoding="utf-8") as fh:
            db: Dict[str, Any] = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return _error_result(f"Failed to load symptoms database: {exc}")

    conditions: List[Dict[str, Any]] = db.get("conditions", [])
    if not conditions:
        return _error_result("Symptoms database contains no conditions.")

    scored_conditions: List[Dict[str, Any]] = []
    all_emergency_flags: List[str] = []

    for cond in conditions:
        cond_symptoms: List[str] = [s.lower() for s in cond.get("symptoms", [])]
        emergency_indicators: List[str] = [
            e.lower() for e in cond.get("emergency_indicators", [])
        ]

        matched: List[str] = [
            s for s in normalised if _symptom_matches(s, cond_symptoms)
        ]
        matched_count: int = len(matched)

        if matched_count == 0:
            continue

        denominator = max(len(normalised), len(cond_symptoms))
        score: float = matched_count / denominator if denominator > 0 else 0.0
        confidence_pct: float = round(score * 100, 1)

        emergency_hits: List[str] = [
            s for s in normalised if _symptom_matches(s, emergency_indicators)
        ]
        all_emergency_flags.extend(
            [h for h in emergency_hits if h not in all_emergency_flags]
        )

        scored_conditions.append({
            "name":                cond.get("name", "Unknown"),
            "icd10_code":          cond.get("icd10_code", "N/A"),
            "category":            cond.get("category", "Unknown"),
            "severity":            cond.get("severity", "unknown"),
            "confidence_score":    confidence_pct,
            "matched_symptoms":    matched,
            "matched_count":       matched_count,
            "typical_duration":    cond.get("typical_duration", "N/A"),
            "requires_urgent_care": cond.get("requires_urgent_care", False),
            "description":         cond.get("description", ""),
        })

    scored_conditions.sort(key=lambda c: c["confidence_score"], reverse=True)
    top_conditions: List[Dict[str, Any]] = scored_conditions[:top_n]

    risk_level: str = _compute_risk_level(
        top_conditions, bool(all_emergency_flags)
    )

    return {
        "success": True,
        "possible_conditions": top_conditions,
        "risk_level": risk_level,
        "emergency_flags": all_emergency_flags,
        "normalised_symptoms": normalised,
        "error": "",
    }


def _symptom_matches(patient_symptom: str, condition_symptoms: List[str]) -> bool:
    """
    Return True when a patient symptom string overlaps with any condition symptom.

    Matching rules (evaluated in order — first match wins):
    1. Exact string equality.
    2. Patient symptom is a substring of the condition symptom.
    3. Condition symptom is a substring of the patient symptom.

    Args:
        patient_symptom:    Normalised (lower-case, stripped) patient symptom.
        condition_symptoms: List of normalised condition symptoms from the DB.

    Returns:
        True if the patient symptom matches at least one condition symptom.
    """
    for cs in condition_symptoms:
        if patient_symptom == cs:
            return True
        if patient_symptom in cs:
            return True
        if cs in patient_symptom:
            return True
    return False


def _compute_risk_level(
    conditions: List[Dict[str, Any]],
    has_emergency_flags: bool,
) -> str:
    """
    Derive an overall risk level from the top-ranked conditions and emergency flags.

    Risk mapping:
        critical  — any HIGH-CONFIDENCE condition (≥25%) requires urgent care,
                    OR emergency flags are present in the patient's symptoms
        high      — top condition severity is 'severe'
        moderate  — top condition severity is 'moderate'
        low       — top condition severity is 'mild' or no strong match

    A minimum confidence threshold of 25% is applied before a condition's
    ``requires_urgent_care`` flag is considered, preventing incidental
    low-confidence matches from triggering a false critical alert.

    Args:
        conditions:          Ranked list of matched conditions.
        has_emergency_flags: True when any emergency indicator was detected.

    Returns:
        Risk level string: 'critical', 'high', 'moderate', or 'low'.
    """
    URGENT_CARE_MIN_CONFIDENCE: float = 25.0

    if has_emergency_flags:
        return "critical"

    if not conditions:
        return "low"

    if any(
        c.get("requires_urgent_care") and c.get("confidence_score", 0) >= URGENT_CARE_MIN_CONFIDENCE
        for c in conditions
    ):
        return "critical"

    top_severity: str = conditions[0].get("severity", "mild")
    if top_severity == "severe":
        return "high"
    if top_severity == "moderate":
        return "moderate"
    return "low"


def _error_result(message: str) -> Dict[str, Any]:
    """
    Build a standardised error result for symptom_analyzer failures.

    Args:
        message: Human-readable description of the error.

    Returns:
        Dict matching the symptom_analyzer return schema with success=False.
    """
    return {
        "success": False,
        "possible_conditions": [],
        "risk_level": "unknown",
        "emergency_flags": [],
        "normalised_symptoms": [],
        "error": message,
    }
