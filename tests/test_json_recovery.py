from agent.llm_ollama import OllamaLLM

parse = OllamaLLM._parse_json


def test_clean_json():
    assert parse('{"action": "finish", "final": "ok"}')["action"] == "finish"


def test_markdown_fenced_json():
    text = '```json\n{"action": "list_files"}\n```'
    assert parse(text)["action"] == "list_files"


def test_prose_prefixed_json():
    text = 'Sure! Here is the action:\n{"action": "read_file", "path": "x.py"}'
    assert parse(text)["path"] == "x.py"


def test_unrecoverable_output_finishes_gracefully():
    out = parse("I cannot help with that.")
    assert out["action"] == "finish" and "did not return valid JSON" in out["final"]
