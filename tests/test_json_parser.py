from src.utils.json_parser import parse_llm_json


def test_parse_llm_json_accepts_pure_json():
    payload = parse_llm_json('{"name": "Order", "count": 3}')

    assert payload["name"] == "Order"
    assert payload["count"] == 3


def test_parse_llm_json_accepts_json_in_markdown_fence():
    payload = parse_llm_json(
        "```json\n"
        '{"classes": [{"name": "Customer"}]}'
        "\n```"
    )

    assert payload["classes"][0]["name"] == "Customer"


def test_parse_llm_json_extracts_text_before_and_after_json():
    payload = parse_llm_json(
        "Aqui está o resultado solicitado:\n"
        '{"issues": [{"id": "i-1", "severity": "minor"}]}\n'
        "Fim."
    )

    assert payload["issues"][0]["id"] == "i-1"
