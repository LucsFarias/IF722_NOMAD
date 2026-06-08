# AGENTS.md

## Project goal

This repository implements an academic extension of the NOMAD paper for UML class diagram generation from natural language requirements.

The goal is not to reproduce the whole paper. The goal is to improve one bottleneck: the original NOMAD architecture is mostly a sequential/cascade multi-agent pipeline. We want to implement a genuinely iterative multi-agent workflow with a rethink loop.

## Scope

Focus only on:

natural language requirements -> UML class diagram -> PlantUML

Do not implement database/table reverse engineering.
Do not make Northwind/database reconstruction the main experiment.

Use data/small_models/data.jsonl as the main dataset.

## Required architecture

Implement three execution modes:

1. SingleAgentBaseline
2. CascadeNOMAD
3. RethinkNOMAD

RethinkNOMAD must include:

- ConceptAgent
- AttributeSpecialistAgent
- RelationshipAgent
- ModelIntegratorAgent
- PlantUMLAgent
- ValidatorAgent
- RethinkRouter

The rethink loop must:

1. Generate an initial UML model.
2. Validate it against the requirements.
3. Produce structured validation issues.
4. Route each issue to the responsible agent.
5. Revise only the necessary part of the model.
6. Regenerate PlantUML.
7. Revalidate.
8. Stop when there are no critical issues or when max_rethink_iterations is reached.

Default max_rethink_iterations must be 2.

## LLM backends

Support only:

- Gemini Flash via LangChain ChatGoogleGenerativeAI.
- Ollama via LangChain ChatOllama.

Do not use GPT-4o as default.

Default:

LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash

Also support:

LLM_PROVIDER=ollama
LLM_MODEL=<local ollama model>

## Reproducibility

All experiments must be reproducible with scripts.

Required scripts:

- scripts/run_experiment.py
- scripts/evaluate_results.py

Each experiment run must save:

- input_requirements.txt
- gold.puml
- generated_initial.puml
- generated_final.puml
- intermediate_model.json
- validation_report_initial.json
- validation_report_final.json
- rethink_history.json
- metrics.json

The final summary must be saved as:

- results/summary.csv
- results/summary.json

Use deterministic settings where possible:

- temperature=0
- fixed prompts
- fixed dataset
- saved outputs
- optional LLM response cache

## Metrics

Compute:

- precision, recall, F1 for classes
- precision, recall, F1 for attributes
- precision, recall, F1 for relationships strict
- precision, recall, F1 for relationships relaxed
- average F1
- number of validation issues
- number of resolved issues
- approximate number of LLM calls
- runtime

## Testing

Create tests with mocked LLMs. Unit tests must not call Gemini or Ollama.

Required tests:

- LLM provider factory
- dataset loader
- PlantUML evaluation on a tiny example
- RethinkRouter with fake validation issues
- Cascade pipeline with mocked LLM responses
- Rethink pipeline with mocked LLM responses

## Important constraints

Do not hide failures.
If JSON parsing fails, retry once, then record the failure.
If PlantUML generation fails, save the invalid output and record the error.
Do not require internet for tests.
Do not require real API keys for tests.