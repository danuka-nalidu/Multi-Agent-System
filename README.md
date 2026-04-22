# CTSE MAS — Healthcare Multi-Agent System

**SE4010 – Cloud Technologies and Software Engineering | Assignment 2**

A locally-hosted **Multi-Agent System (MAS)** that automates the full patient-care pipeline: intake, symptom analysis, treatment planning, and report generation — all without any cloud API calls. Features **LangGraph orchestration** and **Ollama LLM integration** for AI-assisted clinical reasoning at every stage.

---

## Team Members & Responsibilities

| Student ID | Name | Agent                 | Tool                          |
| ---------- | ---- | --------------------- | ----------------------------- |
|            |      | PatientIntakeAgent    | `tool_patient_reader`         |
|            |      | SymptomAnalyzerAgent  | `tool_symptom_analyzer`       |
|            |      | TreatmentPlannerAgent | `tool_medication_recommender` |
|            |      | MedicalReportAgent    | `tool_report_generator`       |

---

## Problem Domain

**Healthcare — Automated Patient Intake, Diagnosis & Treatment Pipeline**

Managing patient records, matching symptoms to conditions, and generating treatment plans are time-intensive tasks prone to human error. This MAS automates the full clinical workflow: load patient data → analyse symptoms → plan treatment → generate report, providing a consistent, structured medical assessment for any patient record.

---

## System Architecture

```
Patient JSON File
        │
        ├─────────────────────────────────────┐
        │                                     │
        ▼                                     ▼
    main.py                          main_langgraph.py
(Sequential)                        (LangGraph Graph)
        │                                     │
        ▼                                     ▼
┌────────────────────────┐           ┌──────────────────┐
│   PatientIntakeAgent   │           │  START Node      │
│ Tool: patient_reader   │           └────────┬─────────┘
│ LLM: Ollama reasoning  │                    │ invoke()
└──────────┬─────────────┘                    ▼
           │                         ┌──────────────────────────┐
           │ GlobalState             │   intake node            │
           │ (llm_intake_reasoning)  │ PatientIntakeAgent + LLM │
           ▼                         └────────┬─────────────────┘
┌────────────────────────────┐               │
│   SymptomAnalyzerAgent     │               ▼
│ Tool: symptom_analyzer     │       ┌──────────────────────────┐
│ LLM: Ollama reasoning      │       │   symptom node           │
└──────────┬─────────────────┘       │ SymptomAnalyzerAgent     │
           │ GlobalState             │ + LLM reasoning          │
           │ (llm_symptom_reasoning) └────────┬─────────────────┘
           ▼                                  │
┌─────────────────────────────┐              ▼
│   TreatmentPlannerAgent     │      ┌──────────────────────────┐
│ Tool: medication_recommender │     │   treatment node         │
│ LLM: Ollama reasoning       │     │ TreatmentPlannerAgent    │
└──────────┬──────────────────┘     │ + LLM reasoning          │
           │ GlobalState            └────────┬─────────────────┘
           │ (llm_treatment_reasoning)       │
           ▼                                 ▼
┌──────────────────────────┐         ┌──────────────────────────┐
│   MedicalReportAgent     │         │   report node            │
│ Tool: report_generator   │         │ MedicalReportAgent       │
│ LLM: Ollama reasoning    │         │ + LLM reasoning          │
└──────────┬───────────────┘         └────────┬─────────────────┘
           │ GlobalState (llm_report_reasoning)│
           │                                   ▼
           ▼                           ┌──────────────────┐
┌────────────────────┐                │    END Node      │
│  reports/report_   │                └──────────────────┘
│   *.md (incl. LLM  │
│  Section 7: AI     │
│ Reasoning)         │
│  logs/trace_*.json │
└────────────────────┘
```

**Orchestration Pattern:** 
- **Sequential Pipeline (main.py):** Coordinator-Worker pattern. Each agent reads from and writes to the shared `GlobalState` object — no data is lost between handoffs.
- **LangGraph Orchestration (main_langgraph.py):** StateGraph with 4 nodes (intake, symptom, treatment, report) connected in a linear DAG. Each node wraps an agent and executes via LangGraph runtime.

**LLM Integration:**
- **Ollama (llama3.2:3b)** runs locally on port 11434
- Each agent calls `get_llm_commentary()` after its tool completes for clinical reasoning
- LLM output is stored in `llm_*_reasoning` fields (gracefully degrades to "" if Ollama unavailable)
- Section 7 in generated reports includes all 4 agents' clinical reasoning commentary

---

## Project Structure

```
ctse_mas/
├── main.py                               # Original sequential pipeline entry point
├── main_langgraph.py                     # LangGraph StateGraph orchestration entry point
├── sample_buggy_code.py                  # Demo patient record (buggy version)
├── requirements.txt
├── conftest.py
│
├── config/
│   ├── state.py                          # GlobalState dataclass + llm_*_reasoning fields
│   ├── llm_client.py                     # Ollama LLM wrapper (graceful degradation)
│   └── observability.py                  # LLMOps logging & Rich console tracing
│
├── agents/
│   ├── agent_patient_intake.py           # Agent 1 + Ollama LLM reasoning
│   ├── agent_symptom_analyzer.py         # Agent 2 + Ollama LLM reasoning
│   ├── agent_treatment_planner.py        # Agent 3 + Ollama LLM reasoning
│   └── agent_report_generator.py         # Agent 4 + Ollama LLM reasoning
│
├── tools/
│   ├── tool_patient_reader.py            # Tool 1
│   ├── tool_symptom_analyzer.py          # Tool 2
│   ├── tool_medication_recommender.py    # Tool 3
│   └── tool_report_generator.py          # Tool 4
│
├── data/
│   ├── symptoms_db.json                  # 10 medical conditions
│   ├── medications_db.json               # 14 medications
│   └── patients/
│       ├── patient_PT001.json            # Influenza case
│       └── patient_PT002.json            # UTI case (with comorbidities)
│
├── tests/
│   ├── test_agent1_patient_intake.py     # 21 tests
│   ├── test_agent2_symptom_analyzer.py   # 22 tests
│   ├── test_agent3_treatment_planner.py  # 21 tests
│   ├── test_agent4_report_generator.py   # 15 tests
│   └── test_pipeline_integration.py      # Group integration harness (19 tests)
│
├── reports/                              # Generated Markdown reports (incl. Section 7: AI Reasoning)
└── logs/                                 # LLMOps JSON traces
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- No paid API keys required — runs 100% locally
- (Optional) Ollama for LLM reasoning — if not installed, pipeline still works without LLM

### Install Ollama (for AI Clinical Reasoning)

LLM reasoning requires Ollama running locally:

1. **Download & Install Ollama**
   - macOS: `brew install ollama` or download from https://ollama.com
   - Linux: `curl -fsSL https://ollama.ai/install.sh | sh`
   - Windows: Download from https://ollama.com

2. **Pull the Model**
   ```bash
   ollama pull llama3.2:3b
   ```

3. **Start Ollama** (runs as background service on port 11434)
   - macOS/Linux: `ollama serve`
   - Windows: Ollama app starts automatically

4. **Verify** (optional)
   ```bash
   curl http://localhost:11434/api/tags
   ```

**Note:** The pipeline works fully without Ollama running — LLM reasoning gracefully degrades to empty strings.

### Create Virtual Environment & Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Pipeline

**Option 1: LangGraph Orchestration (Recommended with Ollama)**
```bash
# With LangGraph + Ollama LLM reasoning (if Ollama is running)
python main_langgraph.py --patient data/patients/patient_PT001.json

# Process different patient
python main_langgraph.py --patient data/patients/patient_PT002.json
```

**Option 2: Original Sequential Pipeline**
```bash
# Pure Python sequential pipeline (works even without Ollama)
python main.py --patient data/patients/patient_PT001.json

# Process the UTI patient
python main.py --patient data/patients/patient_PT002.json
```

Both entry points produce identical outputs with LLM reasoning included (Section 7 in reports).

### Run All Tests

```bash
cd ctse_mas

# Full test suite (98 tests)
python -m pytest tests/ -v

# Individual agent tests
python -m pytest tests/test_agent1_patient_intake.py -v     
python -m pytest tests/test_agent2_symptom_analyzer.py -v  
python -m pytest tests/test_agent3_treatment_planner.py -v  
python -m pytest tests/test_agent4_report_generator.py -v   

# Full pipeline integration tests
python -m pytest tests/test_pipeline_integration.py -v
```

---

## What Each Agent Does

### Agent 1 — PatientIntakeAgent

- Loads and validates the patient JSON record
- Checks required fields, age range, blood type, vital signs
- Populates `GlobalState` with patient demographics, symptoms, and vitals
- **LLM Reasoning:** Calls Ollama (llama3.2:3b) for clinical commentary on data quality and validation concerns

### Agent 2 — SymptomAnalyzerAgent

- Matches patient symptoms against `symptoms_db.json` (10 conditions)
- Scores and ranks probable conditions by confidence percentage
- Detected conditions include: Influenza, UTI, Pneumonia, COVID-19, Diabetes, and more
- **LLM Reasoning:** Calls Ollama for clinical reasoning on differential diagnosis and risk level

### Agent 3 — TreatmentPlannerAgent

- Recommends medications from `medications_db.json` (14 medications)
- Screens against patient allergies before recommending
- Outputs a safe, structured treatment plan with dosage and duration
- **LLM Reasoning:** Calls Ollama for commentary on medication safety, allergy screening, and drug interactions

### Agent 4 — MedicalReportAgent

- Aggregates all pipeline results from `GlobalState`
- Generates a structured Markdown report saved to `reports/`
- Records a full LLMOps execution trace to `logs/`
- **LLM Reasoning:** Calls Ollama for clinical review of report completeness and case quality

---

## Output Files

| File                  | Description                                                                       |
| --------------------- | --------------------------------------------------------------------------------- |
| `reports/report_*.md` | Full patient medical report in Markdown (incl. **Section 7: AI Clinical Reasoning**) |
| `logs/trace_*.json`   | LLMOps execution trace (all agent actions, tool calls, timestamps)                |

### Report Sections

Generated Markdown reports include 7 sections:
1. **Executive Summary** — Plain-language overview for non-clinical readers
2. **Patient Demographics** — ID, name, age, gender, blood type, vital signs
3. **Reported Symptoms** — List of symptoms and emergency indicators detected
4. **Differential Diagnosis** — Ranked conditions with ICD-10 codes and confidence scores
5. **Treatment Plan** — Recommended medications with dosage, contraindications, lifestyle advice
6. **Pipeline Metadata** — Framework info (LangGraph, Ollama, knowledge base versions)
7. **AI Clinical Reasoning (Ollama LLM)** — Commentary from all 4 agents on data quality, diagnosis, treatment safety, and report completeness

---

## Technical Requirements Met

| Requirement                      | Implementation                                                                                |
| -------------------------------- | --------------------------------------------------------------------------------------------- |
| ✅ 4 Distinct Agents             | PatientIntakeAgent, SymptomAnalyzerAgent, TreatmentPlannerAgent, MedicalReportAgent           |
| ✅ Custom Python Tools           | 4 tools with type hints, docstrings, and error handling                                       |
| ✅ State Management              | `GlobalState` dataclass with 4 new `llm_*_reasoning` fields for LLM commentary storage        |
| ✅ LLMOps / Observability        | `observability.py` logs every agent start/end/tool call + JSON trace                          |
| ✅ No Paid APIs                  | Runs entirely locally — no OpenAI/Anthropic keys needed                                       |
| ✅ Individual Agent + Tool       | Each member owns one agent and one corresponding tool                                         |
| ✅ Testing & Evaluation          | 98 tests: unit, integration, and full end-to-end pipeline (all passing)                       |
| ✅ **LangGraph Orchestration**   | `main_langgraph.py` — StateGraph with 4 agent nodes (intake, symptom, treatment, report)     |
| ✅ **Ollama LLM Integration**    | `config/llm_client.py` — llama3.2:3b local LLM, 4 agents call `get_llm_commentary()` per tool |
| ✅ **Graceful LLM Degradation**  | Pipeline runs fully without Ollama; LLM blocks return "" safely on failure                     |
