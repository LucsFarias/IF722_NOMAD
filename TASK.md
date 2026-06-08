# TASK.md

## Academic context

We are implementing a final project for a course. The project must satisfy three requirements:

1. Propose how to improve bottlenecks of a paper using multi-agent systems.
2. Run more than one experiment to demonstrate advantages and limitations.
3. Make the experiments reproducible.

The attached paper is NOMAD: A Multi-Agent LLM System for UML Class Diagram Generation from Natural Language Requirements.

## Our interpretation of the bottleneck

The original NOMAD paper decomposes UML generation into multiple agents, but the coordination is mostly sequential/cascade. The paper mentions more advanced coordination mechanisms such as inter-agent feedback as future work.

Our improvement is to implement RethinkNOMAD: a multi-agent iterative architecture where a ValidatorAgent identifies issues and a RethinkRouter sends those issues back to the responsible agent for correction.

## Existing repository state

The current repository already has:

- data/small_models/data.jsonl
- src/eval_helpers.py
- a partial AttributeSpecialistAgent
- a simple orchestrator with stubs

However, the current implementation is incomplete. Concept extraction and relationship comprehension are stubs, and there is no real rethink loop.

## Implementation goal

Refactor the repository into a complete experimental framework for:

natural language requirements -> UML class diagram -> PlantUML

Do not focus on database/table reconstruction.

## Required execution modes

Implement:

1. SingleAgentBaseline
   - A single LLM call or single agent receives the requirements and generates PlantUML directly.

2. CascadeNOMAD
   - Sequential multi-agent pipeline:
     ConceptAgent -> AttributeSpecialistAgent -> RelationshipAgent -> ModelIntegratorAgent -> PlantUMLAgent -> ValidatorAgent
   - No iterative correction.

3. RethinkNOMAD
   - Same pipeline as CascadeNOMAD, but with iterative correction:
     generate -> validate -> route issues -> revise responsible component -> regenerate -> revalidate
   - Default max_rethink_iterations=2.

## Required agents

- ConceptAgent:
  Extracts candidate UML classes from requirements.

- AttributeSpecialistAgent:
  Extracts and revises class attributes.

- RelationshipAgent:
  Infers UML relationships: association, aggregation, composition, inheritance/generalization.

- ModelIntegratorAgent:
  Produces a consistent intermediate UML JSON model.

- PlantUMLAgent:
  Generates final PlantUML.

- ValidatorAgent:
  Compares requirements, intermediate JSON and PlantUML. Returns structured issues.

- RethinkRouter:
  Maps each validation issue to the responsible agent.

## Required schemas

Use Pydantic or dataclasses for:

- UMLAttribute
- UMLClass
- UMLRelationship
- UMLModel
- ValidationIssue
- ValidationReport
- RethinkStep
- ExperimentResult

## Required LLM providers

Implement an LLM factory:

src/llm/provider.py

Supported providers:

- Gemini Flash through ChatGoogleGenerativeAI
- Ollama through ChatOllama

Configuration:

LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=...

or:

LLM_PROVIDER=ollama
LLM_MODEL=<ollama model>
OLLAMA_BASE_URL=http://localhost:11434

Do not use GPT-4o as default.

## Required scripts

Create:

scripts/run_experiment.py
scripts/evaluate_results.py

run_experiment.py must support:

--dataset data/small_models/data.jsonl
--mode single|cascade|rethink
--provider gemini|ollama
--model
--max-rethink-iterations
--output
--limit-cases
--use-cache

evaluate_results.py must support:

--runs results
--output results/summary.csv

## Required experiments

Document commands for:

Experiment 1:
SingleAgentBaseline vs CascadeNOMAD vs RethinkNOMAD on data/small_models/data.jsonl.

Experiment 2:
RethinkNOMAD with max_rethink_iterations=0, 1, and 2.

Experiment 3:
RethinkNOMAD with Gemini Flash vs Ollama on at least 3 small cases.

## Required outputs per case

For each case, save:

- input_requirements.txt
- gold.puml
- generated_initial.puml
- generated_final.puml
- intermediate_model.json
- validation_report_initial.json
- validation_report_final.json
- rethink_history.json
- metrics.json

## Required final outputs

Save:

- results/summary.csv
- results/summary.json

## Required README updates

Update README.md with:

- project goal
- relation to the NOMAD paper
- what bottleneck is being addressed
- architecture diagram in text
- how to configure Gemini
- how to configure Ollama
- how to run each experiment
- how to reproduce results
- known limitations

## Development process

Before making large changes, first inspect the repository and report:

1. Current files.
2. Which parts are reusable.
3. Which parts are stubs.
4. Proposed implementation plan.

Then implement in phases:

Phase 1:
Schemas, dataset loader, LLM provider factory, tests.

Phase 2:
SingleAgentBaseline and CascadeNOMAD with mocked tests.

Phase 3:
RethinkNOMAD, ValidatorAgent and RethinkRouter.

Phase 4:
Experiment scripts and evaluation.

Phase 5:
README and final cleanup.

Do not skip tests.
Do not make unit tests depend on real LLM calls.