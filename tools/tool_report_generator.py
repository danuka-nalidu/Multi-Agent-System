"""
Tool: medical_report_generator
Member: 
Agent:  Medical Report Agent

Compiles all outputs from the upstream agents (patient intake, symptom
analysis, and treatment planning) into a single, professionally formatted
Markdown medical report and saves it to the reports/ directory.

The report follows clinical documentation standards:
    - Executive Summary
    - Patient Demographics
    - Symptom & Differential Diagnosis Analysis
    - Treatment Plan
    - Medication Details
    - Follow-up & Warnings
    - Pipeline Trace Metadata
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List


def medical_report_generator(
    patient_info: Dict[str, Any],
    validation_report: Dict[str, Any],
    possible_conditions: List[Dict[str, Any]],
    risk_level: str,
    emergency_flags: List[str],
    treatment_plan: Dict[str, Any],
    recommended_medications: List[Dict[str, Any]],
    contraindicated_medications: List[str],
    lifestyle_recommendations: List[str],
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """
    Generate a comprehensive Markdown medical report from pipeline outputs.

    The report is saved to ``<output_dir>/<patient_id>_medical_report_<timestamp>.md``.
    A separate machine-readable JSON summary is saved alongside it.

    Args:
        patient_info:               Patient demographics and chief complaint.
        validation_report:          Output from the patient_record_reader tool.
        possible_conditions:        Ranked list of differential diagnoses from
                                    the symptom_analyzer tool.
        risk_level:                 Overall risk classification ('low' to 'critical').
        emergency_flags:            Symptom strings that triggered emergency alerts.
        treatment_plan:             Structured treatment plan from the
                                    medication_recommender tool.
        recommended_medications:    Safe medications selected for the patient.
        contraindicated_medications: Excluded medications with reasons.
        lifestyle_recommendations:  Non-pharmacological advice list.
        output_dir:                 Directory in which to save the report.
                                    Created automatically if absent.

    Returns:
        Dict with the following keys:
            - success (bool):       True when the report was saved without error.
            - report_path (str):    Path to the saved Markdown report file.
            - json_path (str):      Path to the saved JSON summary file.
            - executive_summary (str): One-paragraph plain-language summary.
            - total_conditions (int): Number of differential diagnoses included.
            - total_medications (int): Number of medications recommended.
            - generated_at (str):   ISO-8601 timestamp of report generation.
            - error (str):          Non-empty only when success is False.

    Example:
        >>> result = medical_report_generator(
        ...     patient_info={"patient_id": "PT001", "name": "Amara Perera"},
        ...     validation_report={"validation_passed": True, "validation_errors": []},
        ...     possible_conditions=[],
        ...     risk_level="low",
        ...     emergency_flags=[],
        ...     treatment_plan={},
        ...     recommended_medications=[],
        ...     contraindicated_medications=[],
        ...     lifestyle_recommendations=[],
        ... )
        >>> result["success"]
        True
    """
    generated_at: str = datetime.now().isoformat()
    os.makedirs(output_dir, exist_ok=True)

    patient_id: str = patient_info.get("patient_id", "UNKNOWN")
    patient_name: str = patient_info.get("name", "Unknown Patient")
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename: str = f"{patient_id}_medical_report_{timestamp}"

    top_condition: str = (
        possible_conditions[0]["name"] if possible_conditions else "Undetermined"
    )
    top_confidence: float = (
        possible_conditions[0]["confidence_score"] if possible_conditions else 0.0
    )

    executive_summary: str = _build_executive_summary(
        patient_name=patient_name,
        patient_id=patient_id,
        top_condition=top_condition,
        top_confidence=top_confidence,
        risk_level=risk_level,
        medication_count=len(recommended_medications),
        has_emergency=bool(emergency_flags),
    )

    markdown: str = _render_markdown(
        patient_info=patient_info,
        validation_report=validation_report,
        possible_conditions=possible_conditions,
        risk_level=risk_level,
        emergency_flags=emergency_flags,
        recommended_medications=recommended_medications,
        contraindicated_medications=contraindicated_medications,
        lifestyle_recommendations=lifestyle_recommendations,
        treatment_plan=treatment_plan,
        executive_summary=executive_summary,
        generated_at=generated_at,
    )

    report_path: str = os.path.join(output_dir, f"{base_filename}.md")
    json_path: str = os.path.join(output_dir, f"{base_filename}.json")

    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(markdown)
    except OSError as exc:
        return _error_result(f"Failed to write Markdown report: {exc}")

    json_summary: Dict[str, Any] = {
        "generated_at": generated_at,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "risk_level": risk_level,
        "top_diagnosis": top_condition,
        "top_confidence_pct": top_confidence,
        "emergency_flags": emergency_flags,
        "total_conditions_identified": len(possible_conditions),
        "total_medications_recommended": len(recommended_medications),
        "total_contraindicated": len(contraindicated_medications),
        "executive_summary": executive_summary,
        "report_path": report_path,
    }

    try:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(json_summary, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        return _error_result(f"Failed to write JSON summary: {exc}")

    return {
        "success": True,
        "report_path": report_path,
        "json_path": json_path,
        "executive_summary": executive_summary,
        "total_conditions": len(possible_conditions),
        "total_medications": len(recommended_medications),
        "generated_at": generated_at,
        "error": "",
    }


def _build_executive_summary(
    patient_name: str,
    patient_id: str,
    top_condition: str,
    top_confidence: float,
    risk_level: str,
    medication_count: int,
    has_emergency: bool,
) -> str:
    """
    Compose a one-paragraph executive summary for non-clinical stakeholders.

    Args:
        patient_name:     Full name of the patient.
        patient_id:       Unique patient identifier.
        top_condition:    Most likely diagnosis.
        top_confidence:   Confidence percentage of the top diagnosis.
        risk_level:       Overall risk classification.
        medication_count: Number of medications being recommended.
        has_emergency:    True if emergency indicators were detected.

    Returns:
        Plain-English executive summary string (one paragraph).
    """
    urgency: str = (
        "URGENT ATTENTION IS REQUIRED — one or more emergency indicators were detected."
        if has_emergency
        else "No immediate emergency indicators were detected."
    )
    return (
        f"Patient {patient_name} (ID: {patient_id}) was assessed by the Healthcare "
        f"Multi-Agent System. The clinical symptom analysis identified "
        f"'{top_condition}' as the most probable diagnosis (confidence: "
        f"{top_confidence:.1f}%). The overall patient risk level has been classified "
        f"as '{risk_level.upper()}'. A total of {medication_count} medication(s) have "
        f"been recommended after allergy and interaction screening. {urgency} "
        f"This report should be reviewed and validated by a licensed clinician "
        f"before any treatment is administered."
    )


def _render_markdown(
    patient_info: Dict[str, Any],
    validation_report: Dict[str, Any],
    possible_conditions: List[Dict[str, Any]],
    risk_level: str,
    emergency_flags: List[str],
    recommended_medications: List[Dict[str, Any]],
    contraindicated_medications: List[str],
    lifestyle_recommendations: List[str],
    treatment_plan: Dict[str, Any],
    executive_summary: str,
    generated_at: str,
) -> str:
    """
    Render the full Markdown report string from all pipeline data.

    Returns:
        Complete Markdown report as a single string.
    """
    vitals: Dict[str, Any] = patient_info.get("vital_signs", {})
    risk_badge: str = {
        "low":      "GREEN  LOW",
        "moderate": "YELLOW MODERATE",
        "high":     "ORANGE HIGH",
        "critical": "RED    CRITICAL",
    }.get(risk_level, risk_level.upper())

    lines: List[str] = [
        "# Healthcare MAS — Medical Assessment Report",
        f"> **Generated:** {generated_at}  ",
        "> **System:** CTSE SE4010 — Multi-Agent Healthcare System  ",
        "> **Disclaimer:** This report is AI-generated and must be reviewed by a "
        "licensed clinician before clinical use.",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        executive_summary,
        "",
        "---",
        "",
        "## 1. Patient Demographics",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| **Patient ID** | {patient_info.get('patient_id', 'N/A')} |",
        f"| **Name** | {patient_info.get('name', 'N/A')} |",
        f"| **Age** | {patient_info.get('age', 'N/A')} years |",
        f"| **Gender** | {patient_info.get('gender', 'N/A')} |",
        f"| **Blood Type** | {patient_info.get('blood_type', 'N/A')} |",
        f"| **Registration Date** | {patient_info.get('registration_date', 'N/A')} |",
        "",
        f"**Chief Complaint:** {patient_info.get('chief_complaint', 'N/A')}",
        "",
        "### Vital Signs",
        "",
        "| Measurement | Value |",
        "|-------------|-------|",
        f"| Temperature | {vitals.get('temperature_celsius', 'N/A')} C |",
        f"| Blood Pressure | {vitals.get('blood_pressure_mmhg', 'N/A')} mmHg |",
        f"| Heart Rate | {vitals.get('heart_rate_bpm', 'N/A')} bpm |",
        f"| Respiratory Rate | {vitals.get('respiratory_rate_rpm', 'N/A')} rpm |",
        f"| O2 Saturation | {vitals.get('oxygen_saturation_percent', 'N/A')}% |",
        "",
        "### Medical History & Current Medications",
        "",
        f"- **Known Allergies:** "
        f"{', '.join(patient_info.get('allergies', [])) or 'None reported'}",
        f"- **Medical History:** "
        f"{', '.join(patient_info.get('medical_history', [])) or 'None reported'}",
        f"- **Current Medications:** "
        f"{', '.join(patient_info.get('current_medications', [])) or 'None'}",
        "",
        "---",
        "",
        "## 2. Reported Symptoms",
        "",
    ]

    symptoms: List[str] = patient_info.get("symptoms", [])
    for symptom in symptoms:
        lines.append(f"- {symptom}")

    if emergency_flags:
        lines += [
            "",
            "### Emergency Indicators Detected",
            "",
            "> The following reported symptoms match known emergency indicators "
            "and require immediate clinical assessment:",
            "",
        ]
        for flag in emergency_flags:
            lines.append(f"- **{flag}**")

    lines += [
        "",
        "---",
        "",
        "## 3. Differential Diagnosis Analysis",
        "",
        f"**Overall Patient Risk Level:** {risk_badge}",
        "",
        "| Rank | Condition | ICD-10 | Confidence | Severity | Matched Symptoms |",
        "|------|-----------|--------|------------|----------|-----------------|",
    ]

    for i, cond in enumerate(possible_conditions, start=1):
        matched: str = ", ".join(cond.get("matched_symptoms", []))
        urgent_marker: str = " [URGENT]" if cond.get("requires_urgent_care") else ""
        lines.append(
            f"| {i} | {cond['name']}{urgent_marker} | {cond['icd10_code']} | "
            f"{cond['confidence_score']}% | {cond['severity'].capitalize()} | {matched} |"
        )

    if not possible_conditions:
        lines.append("| -- | No matching conditions found | -- | -- | -- | -- |")

    lines += [
        "",
        "---",
        "",
        "## 4. Treatment Plan",
        "",
        "### 4.1 Recommended Medications",
        "",
    ]

    if recommended_medications:
        for med in recommended_medications:
            warnings_list: List[str] = med.get("interaction_warnings", [])
            lines += [
                f"#### {med['name']} ({med.get('generic_name', '')})",
                "",
                f"- **Category:** {med.get('category', 'N/A')}",
                f"- **Route:** {med.get('route', 'N/A')}",
                f"- **Adult Dosage:** {med.get('adult_dosage', 'N/A')}",
                f"- **Maximum Daily Dose:** {med.get('max_daily_dose', 'N/A')}",
                f"- **Side Effects:** "
                f"{', '.join(med.get('side_effects', [])) or 'None significant'}",
                f"- **Notes:** {med.get('notes', 'N/A')}",
            ]
            if warnings_list:
                lines.append("- **Interaction Warnings:**")
                for w in warnings_list:
                    lines.append(f"  - {w}")
            lines.append("")
    else:
        lines.append(
            "_No medications recommended — consult a clinician for manual prescribing._"
        )
        lines.append("")

    if contraindicated_medications:
        lines += [
            "### 4.2 Contraindicated / Excluded Medications",
            "",
        ]
        for excluded in contraindicated_medications:
            parts = excluded.split(" — ", 1)
            name_part = parts[0]
            reason_part = parts[1] if len(parts) > 1 else excluded
            lines.append(f"- ~~{name_part}~~ — {reason_part}")
        lines.append("")

    lines += [
        "### 4.3 Lifestyle & Non-Pharmacological Recommendations",
        "",
    ]
    for advice in lifestyle_recommendations:
        lines.append(f"- {advice}")

    follow_up: str = treatment_plan.get(
        "follow_up_schedule",
        "Follow up with your healthcare provider as directed.",
    )
    lines += [
        "",
        "### 4.4 Follow-up Schedule",
        "",
        f"> {follow_up}",
        "",
        "---",
        "",
        "## 5. Validation & Data Quality",
        "",
        "| Check | Result |",
        "|-------|--------|",
        f"| Patient Record Valid | "
        f"{'Yes' if validation_report.get('validation_passed') else 'No'} |",
        f"| Validation Errors | "
        f"{len(validation_report.get('validation_errors', []))} |",
        f"| Validation Warnings | "
        f"{len(validation_report.get('validation_warnings', []))} |",
    ]

    errors: List[str] = validation_report.get("validation_errors", [])
    if errors:
        lines += ["", "**Validation Errors:**", ""]
        for err in errors:
            lines.append(f"- {err}")

    lines += [
        "",
        "---",
        "",
        "## 6. Pipeline Metadata",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Report Generated | {generated_at} |",
        "| System | CTSE SE4010 Healthcare MAS |",
        "| Framework | Custom Sequential Multi-Agent Pipeline |",
        "| Agents | PatientIntakeAgent -> SymptomAnalyzerAgent -> "
        "TreatmentPlannerAgent -> MedicalReportAgent |",
        "| Knowledge Base | symptoms_db.json v1.0 + medications_db.json v1.0 |",
        "",
        "---",
        "",
        "*This report was generated automatically by the CTSE SE4010 Healthcare "
        "Multi-Agent System. It does not constitute medical advice. Always consult "
        "a qualified healthcare professional before making any clinical decision.*",
        "",
    ]

    return "\n".join(lines)


def _error_result(message: str) -> Dict[str, Any]:
    """
    Build a standardised error result for medical_report_generator failures.

    Args:
        message: Human-readable description of the error.

    Returns:
        Dict matching the medical_report_generator return schema with success=False.
    """
    return {
        "success": False,
        "report_path": "",
        "json_path": "",
        "executive_summary": "",
        "total_conditions": 0,
        "total_medications": 0,
        "generated_at": datetime.now().isoformat(),
        "error": message,
    }
