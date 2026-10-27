"""Token Factory client with a fake HTTP session (no network)."""
import json

import pytest

from engine.llm import LLMError, TokenFactoryClient, TruncatedOutput, extract_json, role_settings
from engine.config import load_stage_configs

FAKE_KEY = "sk-test-not-a-real-key-123"


class Resp:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload
        self.text = json.dumps(payload)

    def json(self):
        return self._p


class Session:
    def __init__(self, responses):
        self.responses, self.sent = list(responses), []

    def post(self, url, headers=None, data=None, timeout=None):
        self.sent.append((url, headers, json.loads(data)))
        return self.responses.pop(0)

    def get(self, url, headers=None, timeout=None):
        return Resp(200, {"data": [{"id": "nvidia/b"}, {"id": "nvidia/a"}]})


def client(tmp_path, responses):
    c = TokenFactoryClient(api_key=FAKE_KEY, log_path=tmp_path / "calls.jsonl", max_retries=1)
    c.session = Session(responses)
    return c


def ok(content, finish="stop"):
    return Resp(200, {"choices": [{"message": {"content": content}, "finish_reason": finish}],
                      "usage": {"prompt_tokens": 3, "completion_tokens": 2}})


def test_chat_body_thinking_and_log_without_key(tmp_path):
    c = client(tmp_path, [ok('{"a": 1}')])
    res = c.chat("nvidia/x", [{"role": "user", "content": "hi"}], thinking=False, max_tokens=10,
                 response_format={"type": "json_object"}, tag="A")
    assert extract_json(res.content) == {"a": 1}
    url, headers, body = c.session.sent[0]
    assert url.endswith("/chat/completions")
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["max_tokens"] == 10
    log = (tmp_path / "calls.jsonl").read_text()
    assert FAKE_KEY not in log and '"tag": "A"' in log


def test_length_is_failure(tmp_path):
    c = client(tmp_path, [ok('{"a"', finish="length")])
    with pytest.raises(TruncatedOutput):
        c.chat("m", [{"role": "user", "content": "x"}])


def test_http_error_and_retry(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.llm.time.sleep", lambda s: None)
    c = client(tmp_path, [Resp(503, {}), ok("{}")])
    assert c.chat("m", [{"role": "user", "content": "x"}]).content == "{}"
    c = client(tmp_path, [Resp(400, {"error": "bad schema"})])
    with pytest.raises(LLMError, match="HTTP 400"):
        c.chat("m", [{"role": "user", "content": "x"}])


def test_missing_key_message_has_no_secret(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    c = TokenFactoryClient()
    assert not c.has_key
    with pytest.raises(LLMError, match="NEBIUS_API_KEY is not set"):
        c.list_models()


def test_extract_json_variants():
    assert extract_json('```json\n{"x": [1, 2]}\n```') == {"x": [1, 2]}
    assert extract_json('<think>a {b}</think>\n<answer>\n{"k": "v \\"q\\" }"}\n</answer>') == {"k": 'v "q" }'}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_roles_configured():
    cfg = load_stage_configs()["llm"]
    assert role_settings(cfg, "A")["model"] and role_settings(cfg, "B")["repeats"] == 3
    assert role_settings(cfg, "A")["thinking"] is False
    assert role_settings(cfg, "C_fallback")["model"]
