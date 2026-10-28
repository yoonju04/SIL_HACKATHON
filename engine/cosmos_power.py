"""Turn the Cosmos Dedicated Endpoint on only while the loop needs it (Nebius Token Factory control plane).

Billing: a dedicated endpoint is billed per GPU-minute only while it is `ready`; a stopped endpoint costs
nothing (docs.tokenfactory.nebius.com/ai-models-inference/dedicated-endpoints/billing-policy).

  GET   {control}/v0/dedicated_endpoints                    -> list (status in .deployment)
  PATCH {control}/v0/dedicated_endpoints/{id} {"enabled": b} -> start / stop

* The endpoint is found by routing_key == COSMOS_MODEL_ID (e.g. dedicated/nvidia/Cosmos3-Super-Reasoner-xxxx).
* Key: COSMOS_API_KEY, else NEBIUS_API_KEY (same resolution as the C_cosmos role). Never logged.
* Every public method catches errors and returns False/None, so a control-plane problem never stops a run.
* Restarting can be slow or fail when GPU capacity is taken (dedicated-endpoints/capacity-and-scaling).

CLI (manual check):  python -m engine.cosmos_power status|on|off [--timeout MIN]
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable

import requests

CONTROL_PLANE_URL = "https://api.tokenfactory.nebius.com"
DEFAULT_TIMEOUT_S = 15 * 60.0
DEFAULT_POLL_S = 10.0
DATA_PLANE_URL = "https://api.tokenfactory.nebius.com/v1"
# BUG-014: right after readiness=ready (status=updating) chat/completions still answered HTTP 404 for the routing key.
# These codes mean "not routed / not serving yet"; anything else (200, 400 ...) means the model answers.
NOT_SERVING_CODES = (404, 429, 500, 502, 503, 504)


def _default_log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def endpoint_state(ep: dict | None) -> dict:
    """Flatten one /v0/dedicated_endpoints item into enabled / status / readiness / ready_replicas."""
    if not ep:
        return {"found": False}
    dep = ep.get("deployment") or {}
    return {"found": True, "id": ep.get("id"), "name": ep.get("name"), "enabled": bool(ep.get("enabled")),
            "status": str(dep.get("status") or "").lower() or None,
            "readiness": str(dep.get("readiness") or "").lower() or None,
            "ready_replicas": dep.get("ready_replicas")}


def is_serving(st: dict) -> bool:
    """Send traffic only when readiness is `ready` (lifecycle-and-status: use Readiness, not Lifecycle)."""
    return bool(st.get("found") and st.get("enabled") and st.get("readiness") == "ready")


def _fmt(st: dict) -> str:
    if not st.get("found"):
        return "not found"
    return (f"enabled={st['enabled']} status={st.get('status')} readiness={st.get('readiness')} "
            f"ready_replicas={st.get('ready_replicas')}")


class CosmosPower:
    def __init__(self, routing_key: str, api_key: str | None, *, control_url: str = CONTROL_PLANE_URL,
                 timeout_s: float = DEFAULT_TIMEOUT_S, poll_s: float = DEFAULT_POLL_S,
                 log: Callable[[str], None] = _default_log, session: Any = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 http_timeout_s: float = 30.0, data_url: str | None = None) -> None:
        self.routing_key = routing_key
        self.data_url = data_url.rstrip("/") if data_url else None   # None: no data-plane probe (unit tests)
        self._key = api_key
        self.control_url = control_url.rstrip("/")
        self.timeout_s, self.poll_s, self.http_timeout_s = float(timeout_s), float(poll_s), float(http_timeout_s)
        self.log, self.sleep, self.clock = log, sleep, clock
        self.session = session or requests.Session()
        self.endpoint_id: str | None = None
        self.initial_state: dict | None = None   # first observation (was it already on when the run started?)
        self.used = False                        # ensure_on() called at least once
        self.on_seconds_requested: list[float] = []
        self.events: list[dict] = []

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_env(cls, model: str | None = None, **kw) -> "CosmosPower | None":
        """None when the Cosmos model is not a dedicated endpoint (nothing to switch) or no key is set."""
        model = model or os.environ.get("COSMOS_MODEL_ID") or ""
        key = os.environ.get("COSMOS_API_KEY") or os.environ.get("NEBIUS_API_KEY")
        if not model.startswith("dedicated/") or not key:
            return None
        url = os.environ.get("NEBIUS_CONTROL_PLANE_URL") or CONTROL_PLANE_URL
        kw.setdefault("data_url", os.environ.get("COSMOS_BASE_URL") or DATA_PLANE_URL)
        return cls(model, key, control_url=url, **kw)

    # ------------------------------------------------------------------ HTTP
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}

    def _list(self) -> list[dict]:
        r = self.session.get(f"{self.control_url}/v0/dedicated_endpoints", headers=self._headers(),
                             timeout=self.http_timeout_s)
        if r.status_code != 200:
            raise RuntimeError(f"GET /v0/dedicated_endpoints -> HTTP {r.status_code}: {str(r.text)[:200]}")
        body = r.json()
        return body.get("data") or [] if isinstance(body, dict) else list(body or [])

    def _patch_enabled(self, enabled: bool) -> None:
        r = self.session.patch(f"{self.control_url}/v0/dedicated_endpoints/{self.endpoint_id}",
                               headers=self._headers(), json={"enabled": enabled}, timeout=self.http_timeout_s)
        if r.status_code not in (200, 202):
            raise RuntimeError(f"PATCH enabled={enabled} -> HTTP {r.status_code}: {str(r.text)[:200]}")

    def _probe(self) -> int | None:
        """One tiny chat call to the data plane. Status code, or None on a network error."""
        try:
            r = self.session.post(f"{self.data_url}/chat/completions", headers=self._headers(),
                                  json={"model": self.routing_key, "max_tokens": 1,
                                        "messages": [{"role": "user", "content": "ping"}]},
                                  timeout=self.http_timeout_s)
            return r.status_code
        except Exception:  # noqa: BLE001
            return None

    def _wait_data_plane(self, t0: float, limit: float, reason: str) -> bool:
        """BUG-014: the control plane says ready before the routing key answers. Wait until a chat call is routed."""
        if not self.data_url:
            return True
        last = "unset"
        while True:
            code = self._probe()
            elapsed = self.clock() - t0
            if code is not None and code not in NOT_SERVING_CODES:
                if last != "unset":
                    self.log(f"[cosmos-power] data plane answers after {elapsed:.0f}s (probe HTTP {code})")
                self._event("data_plane_ready", reason=reason, at_s=round(elapsed, 1), code=code)
                return code < 400 or code in (400, 422)
            if elapsed >= limit:
                self.log(f"[cosmos-power] data plane still HTTP {code} after {elapsed:.0f}s (limit {limit:.0f}s)"
                         " - falling back")
                self._event("data_plane_timeout", reason=reason, waited_s=round(elapsed, 1), code=code)
                return False
            if code != last:
                self.log(f"[cosmos-power]   control plane ready but chat/completions -> HTTP {code} ({elapsed:.0f}s); waiting")
                last = code
            self.sleep(self.poll_s)

    def state(self) -> dict:
        eps = self._list()
        ep = next((e for e in eps if e.get("routing_key") == self.routing_key), None)
        if ep is None and self.endpoint_id:
            ep = next((e for e in eps if e.get("id") == self.endpoint_id), None)
        st = endpoint_state(ep)
        if st.get("found"):
            self.endpoint_id = st["id"]
        if self.initial_state is None:
            self.initial_state = st
        return st

    def _event(self, kind: str, **kw) -> None:
        self.events.append({"t": time.time(), "event": kind, **kw})

    # ------------------------------------------------------------------ public (never raise)
    def status(self) -> dict | None:
        try:
            return self.state()
        except Exception as e:  # noqa: BLE001
            self.log(f"[cosmos-power] status check failed: {type(e).__name__}: {e}")
            return None

    def ensure_on(self, reason: str = "", timeout_s: float | None = None) -> bool:
        """Start the endpoint if needed and wait until it serves. True = Cosmos usable now."""
        self.used = True
        limit = self.timeout_s if timeout_s is None else float(timeout_s)
        t0 = self.clock()
        try:
            st = self.state()
            if not st.get("found"):
                self.log(f"[cosmos-power] endpoint {self.routing_key} not found in /v0/dedicated_endpoints")
                self._event("on_failed", reason="not_found")
                return False
            if is_serving(st):
                self.log(f"[cosmos-power] already on ({reason or 'use'}): {_fmt(st)}")
                self._event("already_on", reason=reason)
                return self._wait_data_plane(t0, limit, reason)
            if not st["enabled"]:
                self.log(f"[cosmos-power] starting Cosmos endpoint ({reason or 'use'}); billing starts when ready")
                self._patch_enabled(True)
                self._event("start_requested", reason=reason)
            else:
                self.log(f"[cosmos-power] endpoint enabled but not ready yet ({_fmt(st)}); waiting")
            last = None
            while True:
                elapsed = self.clock() - t0
                if is_serving(st):
                    self.log(f"[cosmos-power] Cosmos ready after {elapsed:.0f}s ({_fmt(st)})")
                    self.on_seconds_requested.append(elapsed)
                    self._event("ready", reason=reason, startup_s=round(elapsed, 1))
                    return self._wait_data_plane(t0, limit, reason)
                if elapsed >= limit:
                    self.log(f"[cosmos-power] Cosmos NOT ready after {elapsed:.0f}s (limit {limit:.0f}s): {_fmt(st)}"
                             " - falling back")
                    self._event("on_timeout", reason=reason, waited_s=round(elapsed, 1))
                    return False
                cur = _fmt(st)
                if cur != last:
                    self.log(f"[cosmos-power]   waiting ({elapsed:.0f}s): {cur}")
                    last = cur
                self.sleep(self.poll_s)
                st = self.state()
        except Exception as e:  # noqa: BLE001 - control plane problems must not stop the run
            self.log(f"[cosmos-power] ensure_on failed after {self.clock() - t0:.0f}s: {type(e).__name__}: {e}")
            self._event("on_failed", reason=f"{type(e).__name__}")
            return False

    def ensure_off(self, reason: str = "") -> bool:
        """Stop the endpoint (no waiting; billing stops at shutdown). True = off or stop requested."""
        try:
            st = self.state()
            if not st.get("found"):
                self.log(f"[cosmos-power] endpoint {self.routing_key} not found; nothing to stop")
                return False
            if not st["enabled"]:
                self._event("already_off", reason=reason)
                return True
            self._patch_enabled(False)
            self.log(f"[cosmos-power] stop requested ({reason or 'done'}); was {_fmt(st)}")
            self._event("stop_requested", reason=reason)
            return True
        except Exception as e:  # noqa: BLE001
            self.log(f"[cosmos-power] ensure_off FAILED ({reason}): {type(e).__name__}: {e} "
                     "- press Stop in the Nebius console to avoid charges")
            self._event("off_failed", reason=f"{type(e).__name__}")
            return False


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run import load_dotenv  # same .env rules as run.py; values never printed
    ap = argparse.ArgumentParser(description="Cosmos dedicated endpoint power (Nebius Token Factory)")
    ap.add_argument("action", choices=("status", "on", "off"))
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S / 60, help="minutes to wait for ready (on)")
    a = ap.parse_args(argv)
    load_dotenv()
    p = CosmosPower.from_env()
    if p is None:
        print("COSMOS_MODEL_ID is not a dedicated/ model or no COSMOS_API_KEY/NEBIUS_API_KEY")
        return 2
    if a.action == "status":
        st = p.status()
        print(_fmt(st) if st else "status unavailable")
        return 0 if st else 1
    ok = p.ensure_on("cli", timeout_s=a.timeout * 60) if a.action == "on" else p.ensure_off("cli")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
