"""engine/cosmos_power.py with a fake control plane (no network)."""
import pytest

from engine import cosmos_power as CP

RK = "dedicated/nvidia/Cosmos3-Super-Reasoner-test"


class Resp:
    def __init__(self, code, body=None):
        self.status_code, self._body, self.text = code, body or {}, str(body)

    def json(self):
        return self._body


class FakePlane:
    """enabled flag + a scripted sequence of deployment states after a start."""

    def __init__(self, enabled=False, start_states=None, fail_get=False, fail_patch=False):
        self.enabled = enabled
        self.start_states = list(start_states or [("running", "ready")])
        self.dep = ("running", "ready") if enabled else ("stopped", "not ready")
        self.fail_get, self.fail_patch = fail_get, fail_patch
        self.gets, self.patches = 0, []

    def get(self, url, headers=None, timeout=None):
        assert url.endswith("/v0/dedicated_endpoints") and headers["Authorization"] == "Bearer k"
        self.gets += 1
        if self.fail_get:
            return Resp(500, {"detail": "boom"})
        if self.enabled and self.dep[1] != "ready" and self.start_states:
            self.dep = self.start_states.pop(0)
        return Resp(200, {"object": "list", "data": [
            {"id": "other", "routing_key": "dedicated/x", "enabled": True, "deployment": {"status": "running", "readiness": "ready"}},
            {"id": "ep1", "routing_key": RK, "enabled": self.enabled,
             "deployment": {"status": self.dep[0], "readiness": self.dep[1], "ready_replicas": int(self.dep[1] == "ready")}}]})

    def patch(self, url, headers=None, json=None, timeout=None):
        assert url.endswith("/v0/dedicated_endpoints/ep1")
        self.patches.append(json["enabled"])
        if self.fail_patch:
            return Resp(403, {"detail": "forbidden"})
        self.enabled = json["enabled"]
        self.dep = ("starting", "not ready") if self.enabled else ("stopping", "not ready")
        return Resp(200, {})


def make(plane, timeout_s=100.0, poll_s=10.0):
    t = {"now": 0.0}
    logs = []

    def sleep(s):
        t["now"] += s
    p = CP.CosmosPower(RK, "k", session=plane, timeout_s=timeout_s, poll_s=poll_s, log=logs.append,
                       sleep=sleep, clock=lambda: t["now"])
    return p, logs


def test_ensure_on_starts_and_waits_until_ready():
    plane = FakePlane(enabled=False, start_states=[("starting", "not ready"), ("starting", "partially ready"),
                                                   ("running", "ready")])
    p, logs = make(plane)
    assert p.ensure_on("stage_10") is True
    assert plane.patches == [True]
    assert p.initial_state["enabled"] is False
    assert p.events[-1]["event"] == "ready" and p.events[-1]["startup_s"] == 30.0
    assert any("ready after" in m for m in logs)
    assert not any("Bearer" in m or " k " in m for m in logs)     # key never logged


def test_ensure_on_idempotent_when_already_ready():
    plane = FakePlane(enabled=True)
    p, _ = make(plane)
    assert p.ensure_on() is True and p.ensure_on() is True
    assert plane.patches == []
    assert p.initial_state["enabled"] is True


def test_ensure_on_timeout_returns_false():
    plane = FakePlane(enabled=False, start_states=[("starting", "not ready")] * 100)
    p, logs = make(plane, timeout_s=30, poll_s=10)
    assert p.ensure_on() is False
    assert p.events[-1]["event"] == "on_timeout"
    assert any("NOT ready" in m for m in logs)


def test_errors_never_raise():
    p, logs = make(FakePlane(fail_get=True))
    assert p.ensure_on() is False and p.ensure_off() is False and p.status() is None
    p2, logs2 = make(FakePlane(enabled=False, fail_patch=True))
    assert p2.ensure_on() is False
    p3, logs3 = make(FakePlane(enabled=True, fail_patch=True))
    assert p3.ensure_off() is False and any("Nebius console" in m for m in logs3)


def test_ensure_off_idempotent():
    plane = FakePlane(enabled=True)
    p, _ = make(plane)
    assert p.ensure_off("end") is True and p.ensure_off("end") is True
    assert plane.patches == [False]


def test_not_found():
    p, _ = make(FakePlane())
    p.routing_key = "dedicated/missing"
    assert p.ensure_on() is False and p.ensure_off() is False


def test_from_env_requires_dedicated_and_key(monkeypatch):
    monkeypatch.delenv("COSMOS_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    assert CP.CosmosPower.from_env(RK) is None
    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    assert CP.CosmosPower.from_env("nvidia/Cosmos3-Super-Reasoner") is None
    assert CP.CosmosPower.from_env(RK).routing_key == RK


# ---------------------------------------------------------------- run.py wiring
import run as R  # noqa: E402


class SpyPower:
    def __init__(self, on_ok=True):
        self.calls, self.on_ok = [], on_ok

    def ensure_on(self, reason="", timeout_s=None):
        self.calls.append(("on", reason))
        return self.on_ok

    def ensure_off(self, reason=""):
        self.calls.append(("off", reason))
        return True


def test_power_guard_turns_off_in_finally_even_on_error():
    spy = SpyPower()
    g = R.CosmosGuard(spy, log=lambda m: None)
    with pytest.raises(KeyboardInterrupt):
        with g.session("final"):
            raise KeyboardInterrupt
    assert spy.calls == [("on", "final"), ("off", "final")]
    g.shutdown("exit")
    assert spy.calls[-1] == ("off", "exit")


def test_power_guard_keep_on_and_none():
    spy = SpyPower()
    g = R.CosmosGuard(spy, log=lambda m: None)
    with g.session("r0", keep_on=True) as ok:
        assert ok is True
    assert spy.calls == [("on", "r0")]
    none = R.CosmosGuard(None, log=lambda m: None)
    with none.session("x") as ok:
        assert ok is True        # manual mode / mock: assume usable, no API calls
    none.shutdown("exit")


def test_mock_run_makes_no_control_plane_calls(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("control plane called in mock mode")
    monkeypatch.setattr(CP.CosmosPower, "state", boom)
    monkeypatch.setattr(CP.CosmosPower, "from_env", classmethod(lambda cls, *a, **k: boom()))
    assert R.main(["--mock", "--cold-start", "2", "--rounds", "0", "--out", str(tmp_path / "m"),
                   "--cosmos-power", "auto"]) == 0


def test_campaign_insights_skipped_when_cosmos_not_ready(tmp_path):
    spy = SpyPower(on_ok=False)
    g = R.CosmosGuard(spy, log=lambda m: None)
    assert R.campaign_insights(tmp_path, "real", {"llm": {}}, g) is None
    assert spy.calls == [("on", "campaign insights"), ("off", "campaign insights")]
    assert g.failed is True
    with g.session("stage_10 round 1") as ok:      # later uses skip instead of waiting again
        assert ok is False
    assert len(spy.calls) == 2
