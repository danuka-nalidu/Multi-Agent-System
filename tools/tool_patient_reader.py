"""
Tool: patient_record_reader
Member: 
Agent:  Patient Intake Agent

Reads a patient JSON file from disk, validates every required field, and
returns a structured result containing the patient record and a detailed
validation report.  The tool never raises on bad data — it captures all
issues and returns them as structured errors so the upstream agent can make
an informed decision about whether to continue the pipeline.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

REQUIRED_FIELDS: List[str] = [
    "patient_id",
    "name",
    "age",
    "gender",
    "blood_type",
    "chief_complaint",
    "symptoms",
    "medical_history",
    "current_medications",
    "allergies",
    "vital_signs",
]

VALID_BLOOD_TYPES: List[str] = [
    "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-",
]

VALID_GENDERS: List[str] = ["Male", "Female", "Other", "Prefer not to say"]

VITAL_SIGN_RANGES: Dict[str, Tuple[float, float]] = {
    "temperature_celsius":        (34.0, 42.5),
    "heart_rate_bpm":             (30.0, 200.0),
    "respiratory_rate_rpm":       (8.0,  40.0),
    "oxygen_saturation_percent":  (70.0, 100.0),
}


def patient_record_reader(patient_file_path: str) -> Dict[str, Any]:
    """
    Read, parse, and validate a patient JSON record from disk.

    Performs the following checks in order:
    1. File existence and readability.
    2. JSON syntax validity.
    3. Presence of all required top-level fields.
    4. Type and range validation for age, gender, blood type, and vital signs.
    5. Symptoms list is non-empty.

    Args:
        patient_file_path: Absolute or relative path to the patient JSON file.

    Returns:
        Dict with the following keys:
            - success (bool):               True when the file was parsed without
                                            critical errors.
            - patient_data (Dict):          The raw patient record (empty on file error).
            - validation_passed (bool):     True when ALL validation rules pass.
            - validation_errors (List[str]): Human-readable list of failures found.
            - validation_warnings (List[str]): Non-blocking notices (e.g. empty history).
            - field_statuses (Dict[str, str]): Per-field 'ok' / 'missing' / 'invalid'.
            - loaded_at (str):              ISO-8601 timestamp of when the file was read.

    Example:
        >>> result = patient_record_reader("data/patients/patient_PT001.json")
        >>> result["validation_passed"]
        True
        >>> result["patient_data"]["name"]
        'Amara Perera'
    """
    loaded_at: str = datetime.now().isoformat()
    validation_errors: List[str] = []
    validation_warnings: List[str] = []
    field_statuses: Dict[str, str] = {}

    if not patient_file_path:
        return _error_result("No patient file path provided.", loaded_at)

    if not os.path.isfile(patient_file_path):
        return _error_result(
            f"File not found: '{patient_file_path}'", loaded_at
        )

    try:
        with open(patient_file_path, "r", encoding="utf-8") as fh:
            raw: str = fh.read()
    except OSError as exc:
        return _error_result(f"Cannot read file: {exc}", loaded_at)

    try:
        patient_data: Dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _error_result(f"Invalid JSON syntax: {exc}", loaded_at)

    if not isinstance(patient_data, dict):
        return _error_result("Patient file must contain a JSON object.", loaded_at)

    for field_name in REQUIRED_FIELDS:
        if field_name not in patient_data or patient_data[field_name] is None:
            validation_errors.append(f"Required field missing or null: '{field_name}'")
            field_statuses[field_name] = "missing"
        else:
            field_statuses[field_name] = "ok"

    age = patient_data.get("age")
    if age is not None:
        if not isinstance(age, (int, float)) or not (0 <= age <= 150):
            validation_errors.append(
                f"Field 'age' must be a number between 0 and 150 (got {age!r})."
            )
            field_statuses["age"] = "invalid"

    gender = patient_data.get("gender")
    if gender is not None and gender not in VALID_GENDERS:
        validation_errors.append(
            f"Field 'gender' must be one of {VALID_GENDERS} (got {gender!r})."
        )
        field_statuses["gender"] = "invalid"

    blood_type = patient_data.get("blood_type")
    if blood_type is not None and blood_type not in VALID_BLOOD_TYPES:
        validation_errors.append(
            f"Field 'blood_type' must be one of {VALID_BLOOD_TYPES} (got {blood_type!r})."
        )
        field_statuses["blood_type"] = "invalid"

    symptoms = patient_data.get("symptoms")
    if isinstance(symptoms, list) and len(symptoms) == 0:
        validation_errors.append("Field 'symptoms' must not be an empty list.")
        field_statuses["symptoms"] = "invalid"

    vital_signs = patient_data.get("vital_signs")
    if isinstance(vital_signs, dict):
        for sign_key, (low, high) in VITAL_SIGN_RANGES.items():
            value = vital_signs.get(sign_key)
            if value is not None:
                if not isinstance(value, (int, float)):
                    validation_errors.append(
                        f"Vital sign '{sign_key}' must be numeric (got {value!r})."
                    )
                elif not (low <= float(value) <= high):
                    validation_errors.append(
                        f"Vital sign '{sign_key}' = {value} is outside expected range "
                        f"[{low}, {high}]."
                    )
    elif vital_signs is not None:
        validation_errors.append("Field 'vital_signs' must be a JSON object.")

    medical_history = patient_data.get("medical_history", [])
    if isinstance(medical_history, list) and len(medical_history) == 0:
        validation_warnings.append(
            "No medical history recorded — assumed none."
        )

    current_meds = patient_data.get("current_medications", [])
    if isinstance(current_meds, list) and len(current_meds) == 0:
        validation_warnings.append(
            "No current medications recorded — assumed none."
        )

    validation_passed: bool = len(validation_errors) == 0

    return {
        "success": True,
        "patient_data": patient_data,
        "validation_passed": validation_passed,
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
        "field_statuses": field_statuses,
        "loaded_at": loaded_at,
    }


def _error_result(message: str, loaded_at: str) -> Dict[str, Any]:
    """
    Build a standardised error result when file loading fails critically.

    Args:
        message:   Human-readable error description.
        loaded_at: ISO-8601 timestamp of the read attempt.

    Returns:
        Dict matching the schema of patient_record_reader but with success=False.
    """
    return {
        "success": False,
        "patient_data": {},
        "validation_passed": False,
        "validation_errors": [message],
        "validation_warnings": [],
        "field_statuses": {},
        "loaded_at": loaded_at,
    }
