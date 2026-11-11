"""03 scenario proposer backend: OpenAI GPT (default) or Token Factory Nemotron A. Mocked HTTP only."""
import json

import pytest

from engine import nemotron as NM
from engine.config import load_stage_configs
from engine.llm import (DEFAULT_GPT_MODEL, LLMError, proposer_client, proposer_settings,
                        required_llm_keys, resolve_llm_provider, resolve_proposer,
                        resolve_vision_backend, vision_settings)
from tests.test_engine_llm import FAKE_KEY, Session, client, ok


def _llm():
    return load_stage_configs()["llm"]


def test_default_proposer_is_gpt():
    assert resolve_proposer(_llm()) == ("gpt", "gpt-4o")
    assert DEFAULT_GPT_MODEL == "gpt-4o"


def test_resolve_proposer_overrides(monkeypatch):
    cfg = _llm()
    assert resolve_proposer(cfg, "nemotron")[0] == "nemotron"
    assert "Nemotron" in resolve_proposer(cfg, "nemotron")[1]
    assert resolve_proposer(cfg, "gpt", "gpt-4o-mini") == ("gpt", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_A_MODEL_ID", "gpt-4.1")
    assert resolve_proposer(cfg, "gpt") == ("gpt", "gpt-4.1")
    with pytest.raises(ValueError, match="gpt or nemotron"):
        resolve_proposer(cfg, "claude")


def test_required_llm_keys():
    assert required_llm_keys(proposer="gpt", video_on=False, insights_on=False, judge_b="laya") == ["OPENAI_API_KEY"]
    assert required_llm_keys(proposer="nemotron", video_on=False, insights_on=False, judge_b="laya") == ["NEBIUS_API_KEY"]
    both = required_llm_keys(proposer="gpt", video_on=True, insights_on=False, judge_b="laya")
    assert both == ["OPENAI_API_KEY", "NEBIUS_API_KEY"]
    both = required_llm_keys(proposer="gpt", video_on=False, insights_on=False, judge_b="nemotron")
    assert both == ["OPENAI_API_KEY", "NEBIUS_API_KEY"]
    only = required_llm_keys(proposer="gpt", video_on=True, insights_on=True, judge_b="gpt", vision_backend="gpt")
    assert only == ["OPENAI_API_KEY"]


def test_gpt_settings_omit_thinking_and_use_json_object():
    s = proposer_settings(_llm(), "gpt", "gpt-4o")
    r = proposer_settings(_llm(), "gpt", "gpt-4o", refine=True)
    assert s["response_format"] == "json_object" and s.get("thinking") is None
    assert r["max_tokens"] >= s["max_tokens"] and r.get("thinking") is None
    n = proposer_settings(_llm(), "nemotron", "nvidia/Nemotron-3_5-Lightning")
    assert n["thinking"] is False and n.get("response_format") == "json_schema"


def test_gpt_client_request_shape_and_log_hides_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    c = proposer_client(_llm(), "gpt", tmp_path / "calls.jsonl")
    assert c.base_url == "https://api.openai.com/v1" and c.api_key_env == "OPENAI_API_KEY"
    c.session = Session([ok('{"candidates": []}')])
    res = c.chat("gpt-4o", [{"role": "system", "content": "Output a JSON object."},
                            {"role": "user", "content": "propose"}],
                 temperature=1.0, top_p=0.95, max_tokens=6000,
                 response_format={"type": "json_object"}, tag="A")
    url, headers, body = c.session.sent[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer " + FAKE_KEY
    assert body["model"] == "gpt-4o"
    assert body["response_format"] == {"type": "json_object"}
    assert "chat_template_kwargs" not in body
    assert res.content == '{"candidates": []}'
    log = (tmp_path / "calls.jsonl").read_text()
    assert FAKE_KEY not in log and '"tag": "A"' in log


def test_call_a_gpt_uses_json_object_and_validates(tmp_path, monkeypatch):
    """Same prompt/parser/validation as Nemotron A; GPT starts at json_object and pastes the schema."""
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    schema = {"type": "object",
              "properties": {"candidates": {"type": "array", "minItems": 1, "items": {
                  "type": "object", "properties": {"fault_id": {"type": "string"}},
                  "required": ["fault_id"], "additionalProperties": False}}},
              "required": ["candidates"]}
    payload = {"candidates": [{"fault_id": "F-06"}, {"fault_id": "F-04", "extra": 1}]}
    c = client(tmp_path, [ok(json.dumps(payload))])
    c.base_url = "https://api.openai.com/v1"
    c.api_key_env = "OPENAI_API_KEY"
    msgs = NM.a_messages(schema=schema, n=2, context={"note": "unit-test"})
    assert "json_schema" not in json.loads(msgs[-1]["content"])
    settings = {"model": "gpt-4o", "backend": "gpt", "response_format": "json_object", "temperature": 1.0}
    obj, st = NM.call_a(c, settings, msgs, schema)
    url, headers, body = c.session.sent[0]
    assert body["response_format"] == {"type": "json_object"}
    assert "chat_template_kwargs" not in body
    user = json.loads(body["messages"][-1]["content"])
    assert "json_schema" in user
    assert obj["candidates"] == [{"fault_id": "F-06"}]   # invalid extra field dropped
    assert st["items_valid"] == 1 and st["items_returned"] == 2
    assert st["response_format"] == "json_object" and st["backend"] == "gpt" and st["model"]
    assert FAKE_KEY not in (tmp_path / "calls.jsonl").read_text()


def test_call_a_gpt_failure_does_not_switch_format(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setattr("engine.llm.time.sleep", lambda s: None)
    from tests.test_engine_llm import Resp
    c = client(tmp_path, [Resp(400, {"error": "bad request"})])
    schema = {"type": "object", "properties": {"candidates": {"type": "array"}}, "required": ["candidates"]}
    obj, st = NM.call_a(c, {"model": "gpt-4o", "backend": "gpt", "response_format": "json_object"},
                        NM.a_messages(schema=schema, n=1, context={}), schema)
    assert obj is None and st["parse_error"] and "HTTP 400" in st["parse_error"]
    assert st["response_format"] == "json_object"
    assert len(c.session.sent) == 1


def test_missing_openai_key_message_has_no_secret(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = proposer_client(_llm(), "gpt")
    assert not c.has_key
    with pytest.raises(LLMError, match="OPENAI_API_KEY is not set"):
        c.list_models()


def test_mock_loop_records_gpt_proposer(tmp_path):
    from run import Loop
    loop = Loop(mode="mock", out=tmp_path / "o", ablation="X-5", video_on=False, seed=0,
                cold_start=2, batch_size=2, rounds=0, quiet=True)
    assert loop.proposer == "gpt" and loop.a_settings["model"] == "gpt-4o"
    assert loop.a_settings.get("thinking") is None
    loop.run()
    cfg = json.loads((tmp_path / "o" / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["proposer"] == {"backend": "gpt", "model": "gpt-4o", "stub": True}
    stats = json.loads((tmp_path / "o" / "loop_stats.json").read_text(encoding="utf-8"))
    assert stats["proposer"]["backend"] == "gpt" and stats["proposer"]["model"] == "gpt-4o"
    assert stats["proposer"].get("stub") is True
    ev = json.loads((tmp_path / "o" / "evaluation_metrics.json").read_text(encoding="utf-8"))
    assert ev["nemotron_a"]["backend"] == "gpt" and ev["nemotron_a"]["model"] == "gpt-4o"


def test_mock_loop_proposer_nemotron_still_selectable(tmp_path):
    from run import Loop
    loop = Loop(mode="mock", out=tmp_path / "n", ablation="X-5", video_on=False, seed=0,
                cold_start=2, batch_size=2, rounds=0, quiet=True, proposer="nemotron")
    assert loop.proposer == "nemotron" and loop.a_settings["thinking"] is False
    assert "Nemotron" in loop.a_settings["model"]
    loop.run()
    cfg = json.loads((tmp_path / "n" / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["proposer"]["backend"] == "nemotron" and cfg["proposer"]["stub"] is True


def test_openai_only_provider_and_vision(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key-123")
    assert resolve_llm_provider(_llm(), mode="real") == "gpt"
    assert resolve_llm_provider(_llm(), mode="mock") is None
    assert resolve_vision_backend(_llm(), mode="real") == "gpt"
    assert resolve_vision_backend(_llm(), "nemotron", mode="real") == "nemotron"
    s = vision_settings(_llm(), "gpt")
    assert s["model"] == "gpt-4o" and s.get("thinking") is None
    assert "video" not in s["media"] and "frames" in s["media"]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_llm_provider(_llm(), mode="real") is None


def test_chat_logprobs_in_request_body(tmp_path):
    from tests.test_engine_llm import client, ok
    c = client(tmp_path, [ok('{"ok": true}')])
    c.chat("gpt-4o", [{"role": "user", "content": "x"}], logprobs=True, top_logprobs=5, tag="B")
    body = c.session.sent[0][2]
    assert body["logprobs"] is True and body["top_logprobs"] == 5


def test_mock_loop_llm_provider_gpt_records_roles(tmp_path):
    from run import Loop
    loop = Loop(mode="mock", out=tmp_path / "p", ablation="X-5", video_on=False, seed=0,
                cold_start=2, batch_size=2, rounds=0, quiet=True, llm_provider="gpt")
    assert loop.proposer == "gpt" and loop.judge_b == "gpt" and loop.vision_backend == "gpt"
    assert loop.jev_client.stub
    loop.run()
    cfg = json.loads((tmp_path / "p" / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["llm_provider"] == "gpt" and cfg["vision_backend"] == "gpt"
    assert cfg["judge_b"] == "gpt" and cfg["proposer"]["backend"] == "gpt"
