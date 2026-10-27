"""LIVE smoke test (network, costs a few tokens). Runs only when NEBIUS_API_KEY is set. Never prints the key."""
import base64
import io
import os

import pytest

from engine.config import load_stage_configs
from engine.llm import TokenFactoryClient, role_settings

pytestmark = pytest.mark.skipif(not os.environ.get("NEBIUS_API_KEY"), reason="NEBIUS_API_KEY not set")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    cfg = load_stage_configs()["llm"]
    return cfg, TokenFactoryClient.from_config(cfg, tmp_path_factory.mktemp("live") / "calls.jsonl")


def test_models_listed(setup):
    cfg, c = setup
    models = set(c.list_models())
    for role in ("A", "B", "C_fallback"):
        assert role_settings(cfg, role)["model"] in models, role


def test_super_vl_image(setup):
    from PIL import Image, ImageDraw
    cfg, c = setup
    img = Image.new("RGB", (96, 64), "white")
    ImageDraw.Draw(img).rectangle([10, 10, 50, 50], fill="red")
    buf = io.BytesIO(); img.save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    s = role_settings(cfg, "C_fallback")
    res = c.chat(s["model"], [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                            {"type": "text", "text": "What color is the square? One word."}]}],
                 max_tokens=200, thinking=False, tag="smoke")
    assert "red" in res.content.lower()


def test_lightning_json_schema(setup):
    from engine.llm import extract_json, json_schema_format
    cfg, c = setup
    s = role_settings(cfg, "A")
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"], "additionalProperties": False}
    res = c.chat(s["model"], [{"role": "user", "content": "Return JSON {\"n\": 7}. Schema: " + str(schema)}],
                 max_tokens=100, thinking=False, response_format=json_schema_format("t", schema), tag="smoke")
    assert extract_json(res.content) == {"n": 7}
