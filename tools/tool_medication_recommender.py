"""
Tool: medication_recommender
Member: 
Agent:  Treatment Planner Agent

Queries the local medications knowledge base (medications_db.json) to
recommend appropriate pharmacological treatments for a patient's diagnosed
conditions.  The tool enforces allergy safety and flags known drug interactions
with the patient's current medications before returning a ranked list of
safe, actionable recommendations.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

_DEFAULT_DB_PATH: str = os.path.join(
    os.path.dirname(__file__), "..", "data", "medications_db.json"
)

LIFESTYLE_RECOMMENDATIONS_MAP: Dict[str, List[str]] = {
    "influenza": [
        "Rest at home and avoid contact with others to limit transmission.",
        "Maintain adequate hydration — at least 2 litres of fluid per day.",
        "Use a humidifier to relieve nasal congestion and throat irritation.",
        "Monitor temperature every 4–6 hours; seek care if fever exceeds 40 °C.",
    ],
    "common cold": [
        "Stay well hydrated with warm fluids (herbal teas, broths).",
        "Rest and avoid strenuous physical activity until symptoms resolve.",
        "Saline nasal rinses can relieve congestion safely.",
    ],
    "pneumonia": [
        "Complete the full antibiotic course even if symptoms improve early.",
        "Avoid smoking and second-hand smoke throughout recovery.",
        "Follow up with a chest X-ray in 6–8 weeks to confirm resolution.",
        "Seek emergency care immediately if breathlessness worsens.",
    ],
    "hypertension": [
        "Adopt a low-sodium diet (< 2 g/day) and increase potassium-rich foods.",
        "Engage in at least 150 minutes of moderate aerobic exercise per week.",
        "Limit alcohol to ≤ 2 standard drinks per day.",
        "Monitor blood pressure at home and keep a log for clinic visits.",
        "Reduce and manage psychological stress through relaxation techniques.",
    ],
    "type 2 diabetes": [
        "Follow a low glycaemic index diet; reduce refined carbohydrates and sugars.",
        "Perform at least 150 minutes of moderate exercise per week.",
        "Monitor blood glucose levels as directed by your healthcare provider.",
        "Attend regular foot examinations to detect early neuropathy.",
        "Maintain a healthy body weight; even 5–10% weight loss improves control.",
    ],
    "urinary tract infection": [
        "Increase fluid intake — aim for 2–3 litres of water per day.",
        "Avoid holding urine; urinate as soon as the urge arises.",
        "After sexual activity, urinate promptly to reduce bacterial introduction.",
        "Avoid irritants such as caffeine, alcohol, and spicy foods until resolved.",
    ],
    "migraine": [
        "Keep a migraine diary to identify and avoid personal triggers.",
        "Maintain a consistent sleep schedule — irregular sleep is a common trigger.",
        "Limit caffeine and alcohol consumption.",
        "Practice stress-reduction techniques such as yoga or mindfulness.",
        "Rest in a quiet, darkened room during acute episodes.",
    ],
    "asthma": [
        "Identify and avoid personal asthma triggers (dust, pollen, cold air, smoke).",
        "Always carry your reliever inhaler and ensure it is not expired.",
        "Use a spacer device with metered-dose inhalers to improve drug delivery.",
        "Follow an Asthma Action Plan discussed with your clinician.",
        "Do not smoke and avoid environments with tobacco smoke.",
    ],
    "gastroenteritis": [
        "Prioritise oral rehydration — small frequent sips of ORS or clear fluids.",
        "Reintroduce bland foods (BRAT diet: bananas, rice, applesauce, toast) gradually.",
        "Practise strict hand hygiene with soap and water after toilet use.",
        "Avoid dairy products, high-fat foods, and caffeine until fully recovered.",
    ],
    "anxiety disorder": [
        "Practise evidence-based relaxation techniques (deep breathing, progressive muscle relaxation).",
        "Engage in regular aerobic exercise — 30 minutes most days reduces anxiety symptoms.",
        "Limit caffeine and alcohol, which can worsen anxiety.",
        "Consider referral to a registered psychologist for Cognitive Behavioural Therapy (CBT).",
        "Maintain a regular daily routine to provide structure and predictability.",
    ],
}

DEFAULT_LIFESTYLE_ADVICE: List[str] = [
    "Maintain adequate rest and sleep (7–9 hours per night).",
    "Stay well hydrated throughout the day.",
    "Follow up with your healthcare provider as scheduled.",
]


def medication_recommender(
    conditions: List[str],
    allergies: List[str],
    current_medications: List[str],
    medications_db_path: str = _DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """
    Recommend safe medications for the patient's diagnosed conditions.

    The function applies three safety layers before returning results:

    1. **Allergy filter** — Any medication whose ``allergy_class`` matches a
       patient allergy (case-insensitive, normalised) is excluded.
    2. **Interaction check** — Current medications are compared against each
       candidate medication's ``known_interactions`` list; flagged combinations
       are recorded but the medication is still returned with a warning.
    3. **Duplication guard** — If the patient is already taking a medication,
       it is excluded from recommendations to avoid double-dosing.

    Args:
        conditions:           List of condition names to treat (e.g. ``["Influenza"]``).
        allergies:            Patient allergy strings (e.g. ``["penicillin"]``).
        current_medications:  Medications the patient is currently taking.
        medications_db_path:  Path to ``medications_db.json``. Defaults to the
                              bundled data file.

    Returns:
        Dict with the following keys:
            - success (bool):                       True when DB loaded successfully.
            - recommended_medications (List[Dict]): Safe medications for the
              patient's conditions, each containing:
              {name, generic_name, category, conditions_treated, adult_dosage,
               max_daily_dose, route, side_effects, notes, interaction_warnings}
            - contraindicated_medications (List[str]): Names of excluded drugs
              and the reason for exclusion.
            - lifestyle_recommendations (List[str]): Non-pharmacological advice.
            - follow_up_schedule (str):              Suggested follow-up timeframe.
            - error (str):                           Non-empty only when success is False.

    Example:
        >>> result = medication_recommender(
        ...     conditions=["Influenza"],
        ...     allergies=["penicillin"],
        ...     current_medications=[],
        ... )
        >>> result["recommended_medications"][0]["name"]
        'Paracetamol'
    """
    if not conditions:
        return _error_result("No conditions provided — cannot generate recommendations.")

    db_path = os.path.normpath(medications_db_path)
    if not os.path.isfile(db_path):
        return _error_result(
            f"Medications database not found at: '{db_path}'"
        )

    try:
        with open(db_path, "r", encoding="utf-8") as fh:
            db: Dict[str, Any] = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return _error_result(f"Failed to load medications database: {exc}")

    all_medications: List[Dict[str, Any]] = db.get("medications", [])
    if not all_medications:
        return _error_result("Medications database contains no entries.")

    normalised_conditions: List[str] = [c.strip().lower() for c in conditions]
    normalised_allergies: List[str] = [a.strip().lower() for a in allergies]
    normalised_current: List[str] = [m.strip().lower() for m in current_medications]

    recommended: List[Dict[str, Any]] = []
    contraindicated: List[str] = []
    seen_names: List[str] = []

    for med in all_medications:
        med_conditions: List[str] = [
            c.lower() for c in med.get("conditions_treated", [])
        ]

        if not any(
            _condition_matches(nc, med_conditions)
            for nc in normalised_conditions
        ):
            continue

        med_name_lower: str = med.get("name", "").lower()
        generic_lower: str = med.get("generic_name", "").lower()
        allergy_class: str = med.get("allergy_class", "none").lower()

        if med_name_lower in seen_names:
            continue

        is_allergy_contraindicated: bool = (
            allergy_class in normalised_allergies
            or allergy_class in normalised_current
            or any(
                a in allergy_class or allergy_class in a
                for a in normalised_allergies
            )
        )
        if is_allergy_contraindicated:
            contraindicated.append(
                f"{med['name']} — excluded: patient allergy to '{allergy_class}'"
            )
            continue

        is_already_taking: bool = (
            med_name_lower in normalised_current
            or generic_lower in normalised_current
        )
        if is_already_taking:
            contraindicated.append(
                f"{med['name']} — excluded: patient already taking this medication"
            )
            continue

        interaction_warnings: List[str] = _check_interactions(
            med, normalised_current, normalised_allergies
        )

        seen_names.append(med_name_lower)
        recommended.append({
            "name":                med.get("name"),
            "generic_name":        med.get("generic_name"),
            "category":            med.get("category"),
            "conditions_treated":  med.get("conditions_treated"),
            "adult_dosage":        med.get("adult_dosage"),
            "max_daily_dose":      med.get("max_daily_dose"),
            "route":               med.get("route"),
            "side_effects":        med.get("side_effects", []),
            "notes":               med.get("notes", ""),
            "interaction_warnings": interaction_warnings,
        })

    lifestyle: List[str] = _build_lifestyle_recommendations(normalised_conditions)
    follow_up: str = _determine_follow_up(normalised_conditions)

    return {
        "success": True,
        "recommended_medications": recommended,
        "contraindicated_medications": contraindicated,
        "lifestyle_recommendations": lifestyle,
        "follow_up_schedule": follow_up,
        "error": "",
    }


def _condition_matches(patient_condition: str, med_conditions: List[str]) -> bool:
    """
    Return True when the patient's condition string overlaps with a medication's
    treatment list (exact, substring, or partial token match).

    Args:
        patient_condition: Normalised condition name.
        med_conditions:    Normalised list of conditions the medication treats.

    Returns:
        True if the condition is covered by this medication.
    """
    for mc in med_conditions:
        if patient_condition == mc:
            return True
        if patient_condition in mc:
            return True
        if mc in patient_condition:
            return True
    return False


def _check_interactions(
    med: Dict[str, Any],
    current_medications: List[str],
    allergies: List[str],
) -> List[str]:
    """
    Identify potential drug interactions between a candidate medication and
    the patient's current medications.

    Args:
        med:                 Candidate medication record from the DB.
        current_medications: Normalised names of medications the patient takes.
        allergies:           Normalised patient allergy strings.

    Returns:
        List of human-readable interaction warning strings (may be empty).
    """
    warnings: List[str] = []
    known_interactions: List[str] = [
        i.lower() for i in med.get("known_interactions", [])
    ]

    for interaction in known_interactions:
        for current in current_medications:
            if interaction in current or current in interaction:
                warnings.append(
                    f"Potential interaction between {med['name']} and "
                    f"'{current}' — monitor patient closely."
                )
    return warnings


def _build_lifestyle_recommendations(conditions: List[str]) -> List[str]:
    """
    Compile condition-specific lifestyle advice, falling back to generic advice
    when no specific guidance is available.

    Args:
        conditions: Normalised list of patient condition names.

    Returns:
        Deduplicated list of lifestyle recommendation strings.
    """
    advice: List[str] = []
    matched_any: bool = False

    for condition in conditions:
        for key, recs in LIFESTYLE_RECOMMENDATIONS_MAP.items():
            if key in condition or condition in key:
                for rec in recs:
                    if rec not in advice:
                        advice.append(rec)
                matched_any = True

    if not matched_any:
        return list(DEFAULT_LIFESTYLE_ADVICE)

    return advice


def _determine_follow_up(conditions: List[str]) -> str:
    """
    Suggest a follow-up timeframe based on condition severity.

    Args:
        conditions: Normalised list of condition names.

    Returns:
        A plain-English follow-up schedule string.
    """
    chronic_conditions = {"hypertension", "type 2 diabetes", "asthma", "anxiety disorder"}
    acute_severe = {"pneumonia"}

    for cond in conditions:
        if cond in acute_severe or any(ac in cond for ac in acute_severe):
            return "Follow up within 48–72 hours or attend Emergency if symptoms worsen."
        if cond in chronic_conditions or any(cc in cond for cc in chronic_conditions):
            return "Review in 4–6 weeks to assess medication response and disease control."

    return "Follow up in 7–10 days if symptoms have not resolved or worsen."


def _error_result(message: str) -> Dict[str, Any]:
    """
    Build a standardised error result for medication_recommender failures.

    Args:
        message: Human-readable description of the error.

    Returns:
        Dict matching the medication_recommender return schema with success=False.
    """
    return {
        "success": False,
        "recommended_medications": [],
        "contraindicated_medications": [],
        "lifestyle_recommendations": [],
        "follow_up_schedule": "",
        "error": message,
    }
