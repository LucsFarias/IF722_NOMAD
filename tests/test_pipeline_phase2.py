from types import SimpleNamespace

import src.agents.base as base_module
from src.agents.plantuml_agent import PlantUMLAgent
from src.pipeline import CascadeNOMAD, SingleAgentBaseline
from src.schemas import UMLAttribute, UMLClass, UMLModel, UMLRelationship


class SequencedFakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        return SimpleNamespace(content=response)


def test_single_agent_baseline_generates_plantuml_with_mocked_llm():
    fake_llm = SequencedFakeLLM(
        [
            "```text\n"
            "@startuml\nclass Order\nclass Customer\nCustomer -- Order\n@enduml"
            "\n```"
        ]
    )

    baseline = SingleAgentBaseline(llm=fake_llm)
    plantuml = baseline.run("Orders belong to customers.")

    assert plantuml.startswith("@startuml")
    assert "class Order" in plantuml
    assert "Customer -- Order" in plantuml
    assert len(fake_llm.prompts) == 1


def test_plantuml_agent_renders_model_deterministically_without_llm_calls():
    fake_llm = SequencedFakeLLM([])
    agent = PlantUMLAgent(llm=fake_llm)
    model = UMLModel(
        classes=[
            UMLClass(
                name="Order",
                attributes=[UMLAttribute(name="id", data_type="String")],
            ),
            UMLClass(name="Customer"),
        ],
        relationships=[
            UMLRelationship(
                source="Customer",
                target="Order",
                relationship_type="association",
                source_multiplicity="1",
                target_multiplicity="*",
                label="places",
            )
        ],
    )

    plantuml = agent.generate(model)

    assert "@startuml" in plantuml
    assert "class Order" in plantuml
    assert "class Customer" in plantuml
    assert 'Customer "1" -- "*" Order : places' in plantuml
    assert fake_llm.prompts == []


def test_cascade_pipeline_runs_all_agents_with_mocked_llm(monkeypatch):
    fake_llm = SequencedFakeLLM(
        [
            "```json\n"
            '{"classes": ['
            '{"name": "Order", "description": "An order", "attributes": []},'
            '{"name": "Customer", "description": "A customer", "attributes": []}'
            ']}'
            "\n```",
            "```json\n"
            '{"attributes": [{"name": "id", "data_type": "String", "confidence": 0.8}]}'
            "\n```",
            "```json\n"
            '{"attributes": [{"name": "email", "data_type": "String", "confidence": 0.9}]}'
            "\n```",
            "```json\n"
            '{"relationships": ['
            '{"source": "Customer", "target": "Order", "relationship_type": "association", "source_multiplicity": "1", "target_multiplicity": "*", "label": "places", "metadata": {}}'
            ']}'
            "\n```",
        ]
    )
    monkeypatch.setattr(base_module, "create_llm_provider", lambda **kwargs: fake_llm)

    cascade = CascadeNOMAD()
    result = cascade.run("Customers place orders.")

    assert [uml_class.name for uml_class in result["model"].classes] == ["Customer", "Order"]
    assert len(result["model"].relationships) == 1
    assert "places" in result["plantuml"]
    assert "Customer" in result["plantuml"]
    assert "Order" in result["plantuml"]
    assert len(fake_llm.prompts) == 4
