"""
Agent 4: Medical Report Agent
Member: 

Persona
-------
A senior medical documentation specialist with clinical audit experience.
Synthesises the complete outputs of all upstream agents into a single,
structured, auditable Markdown medical report. Prioritises clarity for both
clinical and non-clinical readers. Never omits a finding, a warning, or a
contraindication — completeness is non-negotiable.

Responsibilities
----------------
- Consume all data accumulated in GlobalState by Agents 1, 2, and 3.
- Call the medical_report_generator tool to produce the Markdown report.
- Save the report to the reports/ directory.
- Finalize the GlobalState with report_path and executive_summary.
- Persist the full LLMOps pipeline trace to logs/.
- Print the pipeline execution summary table to the console.

Constraints
-----------
- Must include every finding from every agent — no selective reporting.
- The executive summary must be understandable by a non-clinical manager.
- The report file must be saved before this agent returns.
- Always print the summary table, even on partial failure.
- Always produce a log entry regardless of success or failure.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from config.state import GlobalState
from config.llm_client import get_llm_commentary
from config.observability import (
    log_agent_start,
    log_tool_call,
    log_agent_end,
    log_error,
    print_summary_table,
    save_full_trace,
)
from tools.tool_report_generator import medical_report_generator


class MedicalReportAgent:
    """
    Agent 4 — Medical Documentation & Report Generation.

    System Prompt / Persona
    -----------------------
    You are the final agent in the Healthcare MAS pipeline. You receive a
    fully populated GlobalState and synthesise every upstream finding into a
    single, professionally formatted Markdown medical report. Your report
    serves both as a clinical handover document and as an LLMOps audit trail.
    You NEVER add new clinical findings — you only document what the previous
    agents have established. Every section must be present, every medication
    must be listed with its dosage, and every contraindication must be
    explained. The executive summary must be written in plain language
    accessible to a hospital administrator.

    Interaction Strategy
    --------------------
    Reads from:
        state.patient_info
        state.validation_report
        state.possible_conditions
        state.risk_level
        state.emergency_flags
        state.treatment_plan
        state.recommended_medications
        state.contraindicated_medications
        state.lifestyle_recommendations

    Writes to:
        state.final_report
        state.report_path
        state.executive_summary
    """

    name: str = "MedicalReportAgent"

    def run(self, state: GlobalState) -> GlobalState:
        """
        Generate and save the final comprehensive medical report.

        Steps:
            1. Call medical_report_generator with all accumulated state data.
            2. Write report_path, executive_summary, and final_report to state.
            3. Save the LLMOps trace to disk.
            4. Print the pipeline execution summary table.

        Args:
            state: The fully populated GlobalState from Agents 1–3.

        Returns:
            Updated GlobalState with report_path, executive_summary, and
            final_report populated.
        """
        start_time: float = time.time()
        task: str = (
            "Synthesise all pipeline outputs into a structured Markdown medical report."
        )

        log_agent_start(
            self.name,
            task,
            (
                f"patient='{state.patient_info.get('name', 'Unknown')}', "
                f"conditions={len(state.possible_conditions)}, "
                f"medications={len(state.recommended_medications)}"
            ),
        )

        # ── LLM Commentary for this agent (before tool so it goes into report) ─
        state.llm_report_reasoning = get_llm_commentary(
            agent_name=self.name,
            system_prompt=(
                "You are a senior medical documentation specialist. "
                "Given the final report summary, provide a concise 2-3 sentence "
                "commentary on case quality, report completeness, and key clinical takeaways."
            ),
            user_message=(
                f"Patient: {state.patient_info.get('name', 'Unknown')}\n"
                f"Top diagnosis: {state.top_diagnosis}\n"
                f"Risk: {state.risk_level}, Conditions: {len(state.possible_conditions)}, "
                f"Medications: {len(state.recommended_medications)}\n"
                f"Emergency flags: {state.emergency_flags or 'none'}"
            ),
        )

        try:
            result: Dict[str, Any] = medical_report_generator(
                patient_info=state.patient_info,
                validation_report=state.validation_report,
                possible_conditions=state.possible_conditions,
                risk_level=state.risk_level,
                emergency_flags=state.emergency_flags,
                treatment_plan=state.treatment_plan,
                recommended_medications=state.recommended_medications,
                contraindicated_medications=state.contraindicated_medications,
                lifestyle_recommendations=state.lifestyle_recommendations,
                output_dir="reports",
                llm_sections={
                    "intake":    state.llm_intake_reasoning,
                    "symptom":   state.llm_symptom_reasoning,
                    "treatment": state.llm_treatment_reasoning,
                    "report":    state.llm_report_reasoning,
                },
            )
            log_tool_call(
                self.name,
                "medical_report_generator",
                (
                    f"patient={state.patient_info.get('patient_id', 'Unknown')}, "
                    f"conditions={len(state.possible_conditions)}, "
                    f"medications={len(state.recommended_medications)}"
                ),
                (
                    f"success={result['success']}, "
                    f"report_path='{result['report_path']}'"
                ),
            )
        except Exception as exc:
            log_error(self.name, f"medical_report_generator raised: {exc}")
            state.log_agent_action(
                agent_name=self.name,
                action=task,
                input_summary=f"patient={state.patient_info.get('name', 'Unknown')}",
                output_summary=f"Tool error: {exc}",
                tool_calls=["medical_report_generator"],
                status="error",
            )
            print_summary_table()
            return state

        state.report_path = result.get("report_path", "")
        state.executive_summary = result.get("executive_summary", "")
        state.final_report = f"[Report saved to: {state.report_path}]"

        output_summary: str = (
            f"Report generated: '{state.report_path}'. "
            f"Conditions: {result['total_conditions']}, "
            f"Medications: {result['total_medications']}."
        )

        state.log_agent_action(
            agent_name=self.name,
            action=task,
            input_summary=(
                f"patient={state.patient_info.get('name', 'Unknown')}, "
                f"risk={state.risk_level}"
            ),
            output_summary=output_summary,
            tool_calls=["medical_report_generator"],
            status="success" if result.get("success") else "error",
        )

        log_agent_end(self.name, output_summary, time.time() - start_time)

        trace_path: str = state.save_trace(logs_dir="logs")
        save_full_trace()

        print_summary_table()

        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(
            Panel(
                f"[bold]Executive Summary:[/]\n{state.executive_summary}\n\n"
                f"[bold]Report:[/] [underline]{state.report_path}[/]\n"
                f"[bold]Trace:[/]  [underline]{trace_path}[/]",
                title="[green]Pipeline Complete — Healthcare MAS[/]",
                border_style="green",
            )
        )

        return state
