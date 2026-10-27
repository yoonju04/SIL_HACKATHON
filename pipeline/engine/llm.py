"""Nebius Token Factory (OpenAI-compatible) client + offline stub.

* Key from env (NEBIUS_API_KEY by default); never logged or written. Call logs keep model, usage,
  finish_reason, latency only.
* finish_reason == 'length' is a failure (reasoning tokens share max_tokens; Nemotron cards).
* Nemotron reasoning toggle: extra field chat_template_kwargs={"enable_thinking": bool, ...}.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests


class LLMError(RuntimeError):
    pass


class TruncatedOutput(LLMError):
    """finish_reason == 'length'."""


@dataclass
class ChatResult:
    content: str
    reasoning: str | None
    finish_reason: str | None
    usage: dict
    model: str
    latency_s: float
    raw: dict = field(default_factory=dict, repr=False)


def role_settings(llm_cfg: dict, role: str) -> dict:
    s = dict((llm_cfg.get("roles") or {}).get(role) or {})
    env = s.get("model_env_override")
    if env and os.environ.get(env):
        s["model"] = os.environ[env]
    return s


class TokenFactoryClient:
    def __init__(self, base_url: str = "https://api.tokenfactory.nebius.com/v1", api_key_env: str = "NEBIUS_API_KEY",
                 timeout_s: float = 180.0, max_retries: int = 2, log_path: Path | None = None,
                 api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._key = api_key if api_key is not None else os.environ.get(api_key_env)
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.log_path = log_path
        self.session = requests.Session()

    @classmethod
    def from_config(cls, llm_cfg: dict, log_path: Path | None = None, role: str | None = None) -> "TokenFactoryClient":
        p = llm_cfg.get("provider") or {}
        base_url, key_env = p.get("base_url"), p.get("api_key_env", "NEBIUS_API_KEY")
        if role:
            s = (llm_cfg.get("roles") or {}).get(role) or {}
            if s.get("base_url_env_override") and os.environ.get(s["base_url_env_override"]):
                base_url = os.environ[s["base_url_env_override"]]
            if s.get("api_key_env_override") and os.environ.get(s["api_key_env_override"]):
                key_env = s["api_key_env_override"]
        return cls(base_url=base_url, api_key_env=key_env, timeout_s=float(p.get("timeout_s", 180)),
                   max_retries=int(p.get("max_retries", 2)), log_path=log_path)

    @property
    def has_key(self) -> bool:
        return bool(self._key)

    def _headers(self) -> dict:
        if not self._key:
            raise LLMError(f"{self.api_key_env} is not set")
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}

    def list_models(self) -> list[str]:
        r = self.session.get(f"{self.base_url}/models", headers=self._headers(), timeout=self.timeout_s)
        if r.status_code != 200:
            raise LLMError(f"GET /models -> HTTP {r.status_code}: {r.text[:200]}")
        return sorted(m.get("id") for m in r.json().get("data", []))

    def chat(self, model: str, messages: list[dict], *, temperature: float | None = None, top_p: float | None = None,
             max_tokens: int | None = None, response_format: dict | None = None, thinking: bool | None = None,
             chat_template_kwargs: dict | None = None, extra_body: dict | None = None, tag: str = "") -> ChatResult:
        body: dict[str, Any] = {"model": model, "messages": messages}
        for k, v in (("temperature", temperature), ("top_p", top_p), ("max_tokens", max_tokens),
                     ("response_format", response_format)):
            if v is not None:
                body[k] = v
        ctk = dict(chat_template_kwargs or {})
        if thinking is not None:
            ctk["enable_thinking"] = bool(thinking)
        if ctk:
            body["chat_template_kwargs"] = ctk
        body.update(extra_body or {})
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                r = self.session.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                                      data=json.dumps(body), timeout=self.timeout_s)
            except requests.RequestException as e:
                last_err = LLMError(f"network error: {type(e).__name__}")
                time.sleep(2 ** attempt)
                continue
            dt = time.time() - t0
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                last_err = LLMError(f"HTTP {r.status_code}")
                time.sleep(2 ** (attempt + 1))
                continue
            if r.status_code != 200:
                self._log(tag, model, None, {}, dt, f"HTTP {r.status_code}")
                raise LLMError(f"HTTP {r.status_code} for model {model}: {r.text[:300]}")
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            res = ChatResult(content=msg.get("content") or "", reasoning=msg.get("reasoning_content") or msg.get("reasoning"),
                             finish_reason=choice.get("finish_reason"), usage=data.get("usage") or {},
                             model=data.get("model", model), latency_s=dt, raw=data)
            self._log(tag, model, res.finish_reason, res.usage, dt, None)
            if res.finish_reason == "length":
                raise TruncatedOutput(f"{model}: finish_reason=length (raise max_tokens or disable thinking)")
            return res
        raise last_err or LLMError("unknown error")

    def _log(self, tag, model, finish, usage, latency, error) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(ts=time.time(), tag=tag, model=model, finish_reason=finish, usage=usage,
                                     latency_s=round(latency, 3), error=error)) + "\n")


class StubClient:
    """Offline client for --mock. handler(tag, model, messages, kwargs) -> content string (SYNTHETIC)."""

    def __init__(self, handler: Callable[..., str], log_path: Path | None = None) -> None:
        self.handler = handler
        self.log_path = log_path
        self.has_key = True
        self.calls: list[dict] = []

    def list_models(self) -> list[str]:
        return ["stub/A", "stub/B", "stub/C"]

    def chat(self, model: str, messages: list[dict], tag: str = "", **kw) -> ChatResult:
        content = self.handler(tag, model, messages, kw)
        self.calls.append(dict(tag=tag, model=model))
        usage = {"prompt_tokens": sum(len(json.dumps(m.get("content"))) // 4 for m in messages),
                 "completion_tokens": len(content) // 4}
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(ts=time.time(), tag=tag, model=model, stub=True, usage=usage)) + "\n")
        return ChatResult(content=content, reasoning=None, finish_reason="stop", usage=usage, model=model, latency_s=0.0)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply: <answer>..</answer> block, ``` fences, or the first {...} span."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        text = m.group(1)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = _FENCE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth, in_str, esc = 0, False, False
    open_c = text[start]
    close_c = "}" if open_c == "{" else "]"
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON in reply")


def json_schema_format(name: str, schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
