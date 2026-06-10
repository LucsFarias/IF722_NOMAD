# NOMAD: Multi-Agent LLM System for UML Class Diagram Generation

This repository is an academic extension of the NOMAD paper, focused on one bottleneck: the original multi-agent coordination is mostly sequential/cascade. Our contribution is an iterative rethink loop with validation and targeted correction.

## 1. Project Objective

The goal of this project is to generate UML class diagrams in PlantUML from natural-language requirements using reproducible multi-agent experiments.

The implementation supports three execution modes:

- `SingleAgentBaseline`
- `CascadeNOMAD`
- `RethinkNOMAD`

The code is designed for reproducible experimentation, mocked testing, and offline evaluation.

## 2. NOMAD Paper Summary

The NOMAD paper proposes a multi-agent LLM system for UML class diagram generation from textual requirements. The task is decomposed into specialized stages such as concept extraction, attribute inference, relationship inference, integration, and PlantUML generation.

The paper’s central idea is that decomposition helps structure the task, but the coordination is still largely sequential. This repository keeps the task scope from the paper and extends the coordination mechanism.

Paper:

- [arXiv](https://arxiv.org/abs/2511.22409)
- [PDF](https://arxiv.org/pdf/2511.22409)

## 3. Chosen Bottleneck

NOMAD is multi-agent, but its coordination is mostly sequential/cascade.

Our improvement adds iterative rethink:

- `ValidatorAgent` identifies structured issues.
- `RethinkRouter` assigns each issue to the responsible agent.
- The relevant agent revises only the affected part of the model.
- The model is regenerated and revalidated.

This closes the loop instead of stopping after one pass.

## 4. Scope

This project focuses only on:

- natural language requirements
- UML class diagram generation
- PlantUML output

It does not implement:

- reverse engineering from database tables
- Northwind reconstruction as the main experiment

The main dataset is:

- `data/small_models/data.jsonl`

## 5. Architecture

### SingleAgentBaseline

One LLM call receives the requirements and generates PlantUML directly.

### CascadeNOMAD

Sequential pipeline:

`ConceptAgent -> AttributeSpecialistAgent -> RelationshipAgent -> ModelIntegratorAgent -> PlantUMLAgent`

### RethinkNOMAD

Iterative pipeline:

`CascadeNOMAD -> ValidatorAgent -> RethinkRouter -> targeted correction -> regenerate -> revalidate`

Default:

- `max_rethink_iterations=2`

## 6. LLM Backends

Supported providers:

- Gemini Flash via `ChatGoogleGenerativeAI`
- Ollama via `ChatOllama`

Defaults:

- `LLM_PROVIDER=gemini`
- `LLM_MODEL=gemini-3.1-flash-lite`
- `GOOGLE_API_KEY=<your key>`
- `GOOGLE_GENAI_USE_VERTEXAI=false`
- `GEMINI_REQUEST_DELAY_SECONDS=5`
- `GEMINI_MAX_RETRIES=1`

For local inference:

- `LLM_PROVIDER=ollama`
- `LLM_MODEL=<your local ollama model>`
- `OLLAMA_BASE_URL=http://localhost:11434`

## 7. Setup

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pytest
```

Install the runtime dependencies required by your environment if they are not already available.

### Environment file

Copy `.env.example` to `.env` and adjust values as needed.

`.env.example`:

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.1-flash-lite
LLM_TEMPERATURE=0
GOOGLE_API_KEY=
GOOGLE_GENAI_USE_VERTEXAI=false
GEMINI_REQUEST_DELAY_SECONDS=5
GEMINI_MAX_RETRIES=1
OLLAMA_BASE_URL=http://localhost:11434
```

### Gemini

Set:

```bash
export LLM_PROVIDER=gemini
export LLM_MODEL=gemini-3.1-flash-lite
export GOOGLE_API_KEY=your_api_key
export GOOGLE_GENAI_USE_VERTEXAI=false
export GEMINI_REQUEST_DELAY_SECONDS=5
export GEMINI_MAX_RETRIES=1
```

You can swap `LLM_MODEL` for another Gemini model available on your account. Use the exact model name shown in AI Studio.

### Ollama

Set:

```bash
export LLM_PROVIDER=ollama
export LLM_MODEL=llama3.1
export OLLAMA_BASE_URL=http://localhost:11434
```

## 8. Reproducible Experiments

All experiment runs save per-case artifacts under the output directory.

### Experiment 1

Compare `single`, `cascade`, and `rethink` on the main dataset.

```bash
python scripts/run_experiment.py \
  --dataset data/small_models/data.jsonl \
  --mode single \
  --provider ollama \
  --model llama3.1 \
  --output results/exp1/single

python scripts/run_experiment.py \
  --dataset data/small_models/data.jsonl \
  --mode cascade \
  --provider ollama \
  --model llama3.1 \
  --output results/exp1/cascade

python scripts/run_experiment.py \
  --dataset data/small_models/data.jsonl \
  --mode rethink \
  --provider ollama \
  --model llama3.1 \
  --max-rethink-iterations 2 \
  --output results/exp1/rethink
```

### Experiment 2

Run `rethink` with 0, 1, and 2 iterations.

```bash
python scripts/run_experiment.py \
  --dataset data/small_models/data.jsonl \
  --mode rethink \
  --provider ollama \
  --model llama3.1 \
  --max-rethink-iterations 0 \
  --output results/exp2/rethink_0

python scripts/run_experiment.py \
  --dataset data/small_models/data.jsonl \
  --mode rethink \
  --provider ollama \
  --model llama3.1 \
  --max-rethink-iterations 1 \
  --output results/exp2/rethink_1

python scripts/run_experiment.py \
  --dataset data/small_models/data.jsonl \
  --mode rethink \
  --provider ollama \
  --model llama3.1 \
  --max-rethink-iterations 2 \
  --output results/exp2/rethink_2
```

### Experiment 3

Compare Gemini vs Ollama on at least 3 cases.

```bash
python scripts/run_experiment.py \
  --dataset data/small_models/data.jsonl \
  --mode rethink \
  --provider gemini \
  --model gemini-3.1-flash-lite \
  --limit-cases 3 \
  --output results/exp3/gemini

python scripts/run_experiment.py \
  --dataset data/small_models/data.jsonl \
  --mode rethink \
  --provider ollama \
  --model llama3.1 \
  --limit-cases 3 \
  --output results/exp3/ollama
```

### Evaluation

Aggregate all runs into summary files:

```bash
python scripts/evaluate_results.py \
  --runs results \
  --output results/summary.csv
```

This generates:

- `results/summary.csv`
- `results/summary.json`

## 9. Files Generated in `results/`

Each case directory saves:

- `input_requirements.txt`
- `gold.puml`
- `generated_initial.puml`
- `generated_final.puml`
- `intermediate_model.json`
- `validation_report_initial.json`
- `validation_report_final.json`
- `rethink_history.json`
- `metrics.json`

The summary step generates:

- `results/summary.csv`
- `results/summary.json`

The metrics include:

- class precision/recall/F1
- attribute precision/recall/F1
- relationship precision/recall/F1, strict and relaxed
- average F1
- `llm_calls`
- `runtime_seconds`
- `initial_issue_count`
- `final_issue_count`
- `resolved_issue_count`

## 10. Known Limitations

- LLMs can vary even with `temperature=0`.
- Ollama depends on the local model you choose.
- `ValidatorAgent` reduces errors but does not guarantee perfect correction.
- Automatic evaluation does not capture every semantic error.

## 11. Testing

Run the unit tests:

```bash
pytest -q
```

The test suite is fully mocked and does not require real Gemini or Ollama access.

## 12. Implementation Notes

- The main dataset is `data/small_models/data.jsonl`.
- `src/eval_helpers.py` is safe to import without auto-downloading NLTK resources.
- The project keeps failures visible by saving intermediate and final artifacts, including invalid outputs when they occur.
