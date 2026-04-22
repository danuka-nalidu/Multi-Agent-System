"""
CTSE SE4010 — Assignment 2
Healthcare Multi-Agent System (MAS)

Domain: Healthcare — Patient Intake, Diagnosis & Treatment Pipeline

Sequential Pipeline Architecture:
    PatientIntakeAgent  ──►  SymptomAnalyzerAgent
    ──►  TreatmentPlannerAgent  ──►  MedicalReportAgent

Student Contributions:
    Agent 1 / Tool 1: PatientIntakeAgent
    Agent 2 / Tool 2: SymptomAnalyzerAgent
    Agent 3 / Tool 3: TreatmentPlannerAgent
    Agent 4 / Tool 4: MedicalReportAgent

Usage:
    python main.py --patient data/patients/patient_PT001.json
    python main.py --patient data/patients/patient_PT002.json
"""

from __future__ import annotations
from rich.panel import Panel
from agents.agent_report_generator import MedicalReportAgent
from agents.agent_treatment_planner import TreatmentPlannerAgent
from agents.agent_symptom_analyzer import SymptomAnalyzerAgent
from agents.agent_patient_intake import PatientIntakeAgent
from config.observability import console
from config.state import reset_state

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def run_pipeline(patient_file_path: str) -> None:
    """
    Execute the full 4-agent Healthcare MAS pipeline on a patient record.

    Pipeline stages:
        1. PatientIntakeAgent    — Validate and load patient JSON record.
        2. SymptomAnalyzerAgent  — Match symptoms to probable conditions.
        3. TreatmentPlannerAgent — Recommend safe, allergy-screened medications.
        4. MedicalReportAgent    — Generate and save the final Markdown report.

    Args:
        patient_file_path: Path to the patient JSON file to process.
    """
    console.print(
        Panel(
            f"[bold cyan]Patient File:[/] {patient_file_path}\n"
            f"[bold cyan]Pipeline:[/]  PatientIntakeAgent "
            f"→ SymptomAnalyzerAgent "
            f"→ TreatmentPlannerAgent "
            f"→ MedicalReportAgent",
            title="[bold green]CTSE SE4010 — Healthcare Multi-Agent System[/]",
            border_style="green",
        )
    )

    state = reset_state()
    state.patient_file_path = patient_file_path

    state = PatientIntakeAgent().run(state)

    if not state.is_valid:
        console.print(
            f"[yellow]Warning:[/] Patient record has validation issues: "
            f"{state.validation_errors}. Pipeline will continue with available data."
        )

    state = SymptomAnalyzerAgent().run(state)

    state = TreatmentPlannerAgent().run(state)

    state = MedicalReportAgent().run(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CTSE SE4010 — Healthcare Multi-Agent System"
    )
    parser.add_argument(
        "--patient",
        type=str,
        default="data/patients/patient_PT001.json",
        help="Path to the patient JSON file (default: data/patients/patient_PT001.json)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.patient):
        console.print(f"[red]Error: Patient file not found: {args.patient}[/]")
        sys.exit(1)

    run_pipeline(patient_file_path=args.patient)
