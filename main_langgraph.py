"""
CTSE SE4010 — Assignment 2
Healthcare Multi-Agent System — LangGraph Orchestration Entry Point

This module replaces the manual sequential pipeline in main.py with a
LangGraph StateGraph.  All four agents are wrapped as LangGraph nodes and
the graph is compiled and invoked via LangGraph's runtime.

The original main.py is preserved and unchanged.

Usage:
    python main_langgraph.py --patient data/patients/patient_PT001.json
    python main_langgraph.py --patient data/patients/patient_PT002.json

LangGraph compatibility:
    LangGraph requires a TypedDict as state schema.  Since GlobalState is a
    dataclass, it is wrapped in a thin TypedDict:
        {"global_state": GlobalState}
    Each node extracts the dataclass, calls agent.run(), and returns the dict.

Ollama LLM integration:
    Each agent calls get_llm_commentary() after its Python tool completes.
    If Ollama is not running, those calls silently return "" and the pipeline
    continues with deterministic tool outputs only.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from rich.panel import Panel

from agents.agent_patient_intake import PatientIntakeAgent
from agents.agent_symptom_analyzer import SymptomAnalyzerAgent
from agents.agent_treatment_planner import TreatmentPlannerAgent
from agents.agent_report_generator import MedicalReportAgent
from config.observability import console
from config.state import GlobalState, reset_state

sys.path.insert(0, os.path.dirname(__file__))


# ── LangGraph State Schema ────────────────────────────────────────────────────

class PipelineState(TypedDict):
    """
    Thin TypedDict wrapper around GlobalState for LangGraph compatibility.

    LangGraph manages graph state as a dict; GlobalState is stored under the
    'global_state' key to avoid field-level merge logic.
    """
    global_state: GlobalState


# ── Node Functions ─────────────────────────────────────────────────────────────

def node_patient_intake(state: PipelineState) -> PipelineState:
    """LangGraph node: run PatientIntakeAgent and return updated state."""
    updated = PatientIntakeAgent().run(state["global_state"])
    return {"global_state": updated}


def node_symptom_analyzer(state: PipelineState) -> PipelineState:
    """LangGraph node: run SymptomAnalyzerAgent and return updated state."""
    updated = SymptomAnalyzerAgent().run(state["global_state"])
    return {"global_state": updated}


def node_treatment_planner(state: PipelineState) -> PipelineState:
    """LangGraph node: run TreatmentPlannerAgent and return updated state."""
    updated = TreatmentPlannerAgent().run(state["global_state"])
    return {"global_state": updated}


def node_medical_report(state: PipelineState) -> PipelineState:
    """LangGraph node: run MedicalReportAgent and return updated state."""
    updated = MedicalReportAgent().run(state["global_state"])
    return {"global_state": updated}


# ── Graph Construction ─────────────────────────────────────────────────────────

def build_healthcare_graph():
    """
    Construct and compile the LangGraph StateGraph for the healthcare pipeline.

    Topology:
        START → intake → symptom → treatment → report → END
    """
    graph = StateGraph(PipelineState)

    graph.add_node("intake",    node_patient_intake)
    graph.add_node("symptom",   node_symptom_analyzer)
    graph.add_node("treatment", node_treatment_planner)
    graph.add_node("report",    node_medical_report)

    graph.add_edge(START,       "intake")
    graph.add_edge("intake",    "symptom")
    graph.add_edge("symptom",   "treatment")
    graph.add_edge("treatment", "report")
    graph.add_edge("report",    END)

    return graph.compile()


# ── Pipeline Runner ───────────────────────────────────────────────────────────

def run_pipeline_langgraph(patient_file_path: str) -> None:
    """
    Execute the full 4-agent Healthcare MAS pipeline via LangGraph.

    Args:
        patient_file_path: Path to the patient JSON file to process.
    """
    console.print(
        Panel(
            f"[bold cyan]Patient File:[/] {patient_file_path}\n"
            f"[bold cyan]Orchestrator:[/] LangGraph StateGraph\n"
            f"[bold cyan]Pipeline:[/]  PatientIntakeAgent "
            f"→ SymptomAnalyzerAgent "
            f"→ TreatmentPlannerAgent "
            f"→ MedicalReportAgent",
            title="[bold green]CTSE SE4010 — Healthcare MAS (LangGraph)[/]",
            border_style="green",
        )
    )

    gs: GlobalState = reset_state()
    gs.patient_file_path = patient_file_path

    app = build_healthcare_graph()
    final_state: PipelineState = app.invoke({"global_state": gs})

    fgs: GlobalState = final_state["global_state"]

    if any([
        fgs.llm_intake_reasoning,
        fgs.llm_symptom_reasoning,
        fgs.llm_treatment_reasoning,
        fgs.llm_report_reasoning,
    ]):
        console.print(
            Panel(
                f"[bold]Intake:[/]\n{fgs.llm_intake_reasoning}\n\n"
                f"[bold]Symptom:[/]\n{fgs.llm_symptom_reasoning}\n\n"
                f"[bold]Treatment:[/]\n{fgs.llm_treatment_reasoning}\n\n"
                f"[bold]Report:[/]\n{fgs.llm_report_reasoning}",
                title="[yellow]LLM Clinical Reasoning (Ollama)[/]",
                border_style="yellow",
            )
        )


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CTSE SE4010 — Healthcare MAS (LangGraph Orchestration)"
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

    run_pipeline_langgraph(patient_file_path=args.patient)
