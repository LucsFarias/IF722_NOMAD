from types import SimpleNamespace

import src.agents.base as base_module
from src.agents.attribute_specialist import AttributeSpecialistAgent


class SequencedFakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        return SimpleNamespace(content=response)


def test_attribute_specialist_agent_uses_injected_factory_and_parses_json(monkeypatch):
    fake_llm = SequencedFakeLLM(
        [
            "```json\n"
            '{"attributes": [{"name": "customerId", "inferred_type": "String", "confidence": 0.9}]}'
            "\n```"
        ]
    )
    monkeypatch.setattr(base_module, "create_llm_provider", lambda **kwargs: fake_llm)

    agent = AttributeSpecialistAgent()
    result = agent.enrich_entities(
        [
            {
                "name": "Order",
                "description": "An order",
                "attributes": [{"name": "id", "data_type": "String"}],
            }
        ],
        requirements="Orders belong to customers.",
    )

    assert len(result) == 1
    assert result[0]["name"] == "Order"
    assert any(attribute["name"] == "customerId" for attribute in result[0]["attributes"])
    assert agent.last_parse_error is None
    assert len(fake_llm.prompts) == 1


def test_attribute_specialist_agent_retries_once_on_invalid_json():
    fake_llm = SequencedFakeLLM(
        [
            "not json",
            "```json\n"
            '{"attributes": [{"name": "phone", "inferred_type": "String", "confidence": 0.7}]}'
            "\n```",
        ]
    )

    agent = AttributeSpecialistAgent(llm=fake_llm)
    result = agent.enrich_entities(
        [
            {
                "name": "Customer",
                "description": "A buyer",
                "attributes": [],
            }
        ],
        requirements="Customers have a phone.",
    )

    assert len(fake_llm.prompts) == 2
    assert result[0]["attributes"][0]["name"] == "phone"
    assert agent.last_parse_error is None
