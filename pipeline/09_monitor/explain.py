"""09b Vision judgment + commentary (Cosmos Reason). The monitor does the numbers; the model does the rest.

Division of work
  monitor.py (numbers)  pass/fail, SR1..SR6, latency, lane margin, collisions, team metrics
  explain.py (vision)   what only the video can tell:
    1. blind judgment   the model sees the clip WITHOUT the monitor's verdict and reports
                        lane behaviour, motion pattern (how it failed), surroundings (what was next to
                        the car: oncoming car, pedestrian, road edge ...), near miss, severity 0..3
    2. comparison       (code, not the model) monitor verdict vs blind severity:
                        agree / monitor may have missed it (pass but severity >= 2) /
                        video shows no risk (fail but severity 0) -> human review queue
                        + on the chase view, the STEERING: MPC/LQR label vs the logged switch time
    3. commentary       a second call gets the monitor facts + its own judgment and explains in Korean
  Nothing here changes the monitor's verdict. Severity / context / pattern go to the failure table
  (stage 3, Nemotron A and B) and to the screen.

Outputs (screen folder)
  clips/<run>.mp4      clip: fault onset - 1 s -> end of graded window (crash + 1 s)
  commentary.jsonl     one line per run: judgment, comparison, commentary, errors
  campaign.html        redrawn (dashboard.py)

Model access (OpenAI-compatible chat API: Nebius Token Factory, or a Cosmos Reason vLLM endpoint):
  NEBIUS_TOKEN_FACTORY_KEY        API key (Token Factory key, starts with "v1.")
  NEBIUS_TOKEN_FACTORY_BASE_URL   default https://api.tokenfactory.nebius.com/v1/
  COSMOS_MODEL                    model id your key lists (run --check). The hosted Cosmos3 reasoner left
                                  the public Token Factory catalog in August 2026, so no id is hard-coded.

  python explain.py --check
  python explain.py --records results/run_records.jsonl --screen results/screen [--failures-only] [--watch]
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from feedback import read_records  # noqa: E402
from monitor import view  # noqa: E402
from video_clip import make_clip  # noqa: E402

DEFAULT_BASE = "https://api.tokenfactory.nebius.com/v1/"
FAILURES = ("F1_no_or_late_switch", "F2_switch_transient", "F3_not_recovered")
OUTCOME_KO = {"F1_no_or_late_switch": "F1 전환 실패·지연", "F2_switch_transient": "F2 전환 순간 불안정",
              "F3_not_recovered": "F3 전환 후 불안정", "pass": "통과", "no_trigger": "시험 안 됨", "invalid": "무효"}
SR_KO = {"SR1": "전환 시간", "SR2": "전환 순간 조향 튐", "SR4": "차선 유지/충돌", "SR5": "회복", "SR6": "조향 한계"}

# closed vocabularies for the blind judgment (anything else -> "unclear")
LANE = ("centered", "drift_in_lane", "touches_line", "crosses_line", "leaves_road", "unclear")
MOTION = ("smooth", "slow_drift", "weaving", "sudden_swerve", "overcorrection_after_switch", "stopped", "unclear")
OBJ = ("vehicle", "oncoming_vehicle", "pedestrian", "cyclist", "road_edge", "barrier", "parked_vehicle")
NEAR = ("far", "near", "very_near")
SIDE = ("left", "right", "front", "rear")

SYSTEM = (
    "너는 자율주행 시뮬레이션 영상을 보는 물리 세계 판단자다. 시험 대상 차량(카메라가 따라가는 차)은 "
    "주 제어기 MPC에 고장이 들어가면 백업 제어기 LQR로 전환한다. 너는 영상에서 보이는 것만으로 판단하고, "
    "보이지 않는 것은 'unclear'로 둔다. 답은 요청한 JSON 하나만 쓴다."
)

JUDGE_TASK = """[과제: 영상만 보고 판단]
고장은 t={onset:.1f} s에 들어갔다. 프레임 시각: {times}.
{label_note}
아래 JSON 하나만 출력해라.
{{
 "lane": "{lane}",
 "motion": "{motion}",
 "surroundings": [{{"type": "{obj}", "side": "{side}", "proximity": "{near}"}}],
 "near_miss": true 또는 false,
 "near_miss_detail": "가까이 간 대상과 순간, 없으면 null",
 "switch_seen_frame": 화면 STEERING 표시가 LQR로 바뀐 첫 프레임 번호(1부터) 또는 null,
 "severity": 0~3 정수,
 "evidence_frames": [판단 근거 프레임 번호],
 "confidence": "low|medium|high"
}}
severity 기준: 0 = 차선 중앙 부근 유지, 위험 없음 / 1 = 차선 안에서 눈에 띄게 밀리거나 흔들림, 주변 위험 없음 /
2 = 차선을 넘거나 다른 물체·도로 끝에 가까워짐, 실제 도로라면 운전자 개입이 필요 / 3 = 충돌 또는 충돌 직전.
surroundings에는 위험과 관련 있는 것만 넣고, 없으면 빈 목록."""

COMMENT_TASK = """[과제: 해설]
[모니터 수치 (시뮬레이터 실측, 판정 확정)]
{facts}

[너의 영상 판단]
{judgment}

[모니터와 영상 판단 비교] {agreement}

엔지니어와 심사위원이 읽을 해설을 한국어 JSON 하나로 써라. 판정은 바꾸지 마라.
{{"summary": "한 문장 요약", "timeline": ["t=..s 무슨 일", "..."], "why": "왜 이 결과인지 1~2문장, 주변 상황 포함",
 "risk_context": "같은 결과라도 이 장면이 더/덜 위험한 이유, 없으면 null"}}"""

AGREEMENT_KO = {
    "agree_risky": "일치 · 둘 다 위험",
    "agree_safe": "일치 · 둘 다 안전",
    "agree_minor": "대체로 일치 · 영상상 경미",
    "monitor_may_miss": "불일치 · 모니터는 통과인데 영상은 위험 (사람 확인 필요)",
    "video_shows_no_risk": "불일치 · 모니터는 실패인데 영상상 위험이 안 보임 (기준·측정 확인)",
    "no_judgment": "영상 판단 없음",
}


# ------------------------------------------------------------------ model client
class ModelClient:
    def __init__(self, key: str | None = None, base_url: str | None = None, model: str | None = None,
                 timeout: float = 120.0):
        self.key = key if key is not None else os.environ.get("NEBIUS_TOKEN_FACTORY_KEY")
        self.base = (base_url or os.environ.get("NEBIUS_TOKEN_FACTORY_BASE_URL") or DEFAULT_BASE).rstrip("/") + "/"
        self.model = model if model is not None else os.environ.get("COSMOS_MODEL")
        self.timeout = timeout

    @property
    def ready(self) -> bool:
        return bool(self.key and self.model)

    def _request(self, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(self.base + path, method="POST" if body is not None else "GET",
                                     data=None if body is None else json.dumps(body).encode("utf-8"),
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        last = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
                if e.code not in (429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError) as e:
                last = f"network: {e}"
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(last or "request failed")

    def models(self) -> list[str]:
        return [m.get("id") for m in self._request("models").get("data", [])]

    def chat(self, messages: list[dict], max_tokens: int = 1200, temperature: float = 0.2) -> str:
        out = self._request("chat/completions", {"model": self.model, "messages": messages,
                                                 "max_tokens": max_tokens, "temperature": temperature})
        return out["choices"][0]["message"]["content"] or ""


# ------------------------------------------------------------------ parsing
def _json_from(text: str) -> dict | None:
    """Drop <think>...</think>, then read the first JSON object."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_judgment(text: str, n_frames: int) -> dict | None:
    """Blind judgment -> validated dict (closed vocabularies, severity 0..3), or None."""
    o = _json_from(text)
    if not isinstance(o, dict) or "severity" not in o:
        return None
    try:
        sev = max(0, min(3, int(o.get("severity"))))
    except (TypeError, ValueError):
        return None
    pick = lambda v, allowed: v if v in allowed else "unclear"  # noqa: E731
    sur = []
    for s in o.get("surroundings") or []:
        if isinstance(s, dict) and s.get("type") in OBJ:
            sur.append({"type": s["type"], "side": pick(s.get("side"), SIDE), "proximity": pick(s.get("proximity"), NEAR)})
    sw = o.get("switch_seen_frame")
    sw = sw if isinstance(sw, int) and 1 <= sw <= n_frames else None
    ev = [f for f in (o.get("evidence_frames") or []) if isinstance(f, int) and 1 <= f <= n_frames]
    return {"lane": pick(o.get("lane"), LANE), "motion": pick(o.get("motion"), MOTION), "surroundings": sur,
            "near_miss": bool(o.get("near_miss")), "near_miss_detail": o.get("near_miss_detail") or None,
            "switch_seen_frame": sw, "severity": sev, "evidence_frames": ev,
            "confidence": o.get("confidence") if o.get("confidence") in ("low", "medium", "high") else "low"}


def parse_answer(text: str) -> dict:
    """Commentary -> {summary, timeline, why, risk_context}. Falls back to the plain text."""
    o = _json_from(text)
    if isinstance(o, dict):
        return {"summary": o.get("summary"), "timeline": [str(x) for x in o.get("timeline") or []],
                "why": o.get("why"), "risk_context": o.get("risk_context") or None}
    plain = text.rsplit("</think>", 1)[-1].strip()
    return {"summary": plain[:400] or None, "timeline": [], "why": None, "risk_context": None}


# ------------------------------------------------------------------ comparison (code, not the model)
def compare(r: dict, j: dict | None, frames: list[dict], label_visible: bool) -> dict:
    """Monitor verdict vs blind severity, and the on-screen switch label vs the logged switch time."""
    if not j:
        return {"agreement": "no_judgment", "switch_check": None}
    fail = r.get("outcome_category") in FAILURES
    sev = j["severity"]
    if fail:
        agreement = "agree_risky" if sev >= 2 else ("agree_minor" if sev == 1 else "video_shows_no_risk")
    else:
        agreement = "monitor_may_miss" if sev >= 2 else "agree_safe"
    check = None
    if label_visible and frames:
        st = r.get("switch_time")
        seen = j.get("switch_seen_frame")
        step = (frames[-1]["t"] - frames[0]["t"]) / max(1, len(frames) - 1) if len(frames) > 1 else 1.0
        if st is None and seen is None:
            check = "consistent"
        elif st is None or seen is None:
            check = "mismatch"
        else:
            check = "consistent" if abs(frames[seen - 1]["t"] - st) <= step + 0.05 else "mismatch"
    return {"agreement": agreement, "switch_check": check}


# ------------------------------------------------------------------ prompts
def facts(r: dict) -> str:
    """The monitor's facts for one run, in plain Korean lines."""
    fp = r.get("fault_plan") or (r.get("parameters") or {}).get("fault_plan") or {}
    env = r.get("environment") or (r.get("parameters") or {}).get("environment") or {}
    sr = r.get("sr") or {}
    m = r.get("sr_measures") or {}
    lines = [f"결과: {OUTCOME_KO.get(r.get('outcome_category'), r.get('outcome_category'))}",
             f"경로: {env.get('route') or env.get('town') or '모름'}, 속도: {env.get('speed_mps', '모름')} m/s",
             f"고장: {fp.get('fault_type', '모름')} (세기 {fp.get('magnitude', '-')}, {fp.get('duration_s', '-')} s), "
             f"시점: {fp.get('onset_event', '모름')}, 실제 시작 t={r.get('actual_fault_onset')} s"]
    if r.get("switch_time") is not None:
        s1 = sr.get("SR1") or {}
        lines.append(f"전환: t={r['switch_time']:.2f} s (고장 후 {r.get('detection_latency_s') or 0:.2f} s, "
                     f"한계 {s1.get('limit_s', '-')} s)")
    else:
        lines.append("전환: 없음 (MPC가 계속 운전)")
    for k in r.get("failed_requirements") or []:
        v = sr.get(k) or {}
        detail = {"SR1": f"지연 {v.get('latency_s')} s / 한계 {v.get('limit_s')} s",
                  "SR2": f"조향 속도 {v.get('max_rate_dps', 0):.0f} °/s / 한계 60 °/s",
                  "SR4": (f"충돌 t={m.get('SR4_collision_t')} s ({m.get('SR4_collision_other') or '상대 모름'})"
                          if v.get("collision") else f"차체가 차선을 {-(v.get('margin_m') or 0):.2f} m 넘음"),
                  "SR6": f"조향 {v.get('max_steer_deg', 0):.0f}°, {v.get('max_rate_dps', 0):.0f} °/s"}.get(k, "")
        lines.append(f"어긴 기준: {k} {SR_KO.get(k, '')} — {detail}")
    return "\n".join(lines)


def _images(frames: list[dict]) -> list[dict]:
    out = []
    for f in frames:                                   # media first, then text (Cosmos Reason prompt guide)
        data = base64.b64encode(Path(f["path"]).read_bytes()).decode()
        out.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
    return out


def judge_messages(r: dict, frames: list[dict], label_visible: bool) -> list[dict]:
    onset = r.get("actual_fault_onset") or r.get("planned_fault_onset") or 0.0
    times = ", ".join(f"{i + 1}번 t={f['t']} s" for i, f in enumerate(frames))
    note = ("화면 왼쪽 위 'STEERING: MPC / LQR (fallback)' 글자가 현재 운전 중인 제어기다." if label_visible
            else "이 영상에는 제어기 표시가 없다. switch_seen_frame은 null로 둔다.")
    text = JUDGE_TASK.format(onset=onset, times=times, label_note=note, lane="|".join(LANE), motion="|".join(MOTION),
                             obj="|".join(OBJ), side="|".join(SIDE), near="|".join(NEAR))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _images(frames) + [
        {"type": "text", "text": text}]}]


def comment_messages(r: dict, frames: list[dict], j: dict | None, cmp: dict) -> list[dict]:
    text = COMMENT_TASK.format(facts=facts(r), judgment=json.dumps(j, ensure_ascii=False) if j else "없음",
                               agreement=AGREEMENT_KO.get(cmp["agreement"], cmp["agreement"]))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _images(frames) + [
        {"type": "text", "text": text}]}]


# ------------------------------------------------------------------ per run
def clip_window(r: dict, before_s: float = 1.0, after_crash_s: float = 1.0) -> tuple[float, float]:
    onset = r.get("actual_fault_onset") or r.get("planned_fault_onset") or 0.0
    end = (r.get("evaluation_window") or [onset, onset + 10.0])[1]
    crash = (r.get("sr_measures") or {}).get("SR4_collision_t")
    if crash is not None:
        end = min(end, crash + after_crash_s)
    return max(0.0, onset - before_s), end


def rid_of(r: dict) -> str:
    rid = str(r.get("run_id") or f"{r.get('scenario_id')}-{r.get('planned_fault_onset')}")
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in rid)


def read_commentary(screen: str | Path) -> dict:
    """run_id -> latest commentary row (the file is append-only)."""
    p = Path(screen) / "commentary.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["run_id"]] = row
    return out


def explain_run(r: dict, screen: str | Path, client: ModelClient, n_frames: int = 8) -> dict:
    screen = Path(screen)
    rid = rid_of(r)
    row = {"run_id": rid, "created": datetime.now().isoformat(timespec="seconds"), "model": client.model,
           "outcome": r.get("outcome_category"), "clip": None, "frames_t": [], "judgment": None,
           "agreement": "no_judgment", "switch_check": None, "summary": None, "timeline": [], "why": None,
           "risk_context": None, "error": None}
    fault_dir = (r.get("run_dirs") or {}).get("fault")
    clip = None
    if fault_dir and Path(fault_dir).exists():
        a, b = clip_window(r)
        clip = make_clip(fault_dir, a, b, screen / "clips" / f"{rid}.mp4", screen / "clips" / f"{rid}_frames",
                         n_frames=n_frames)
    if not clip:
        row["error"] = "영상 없음 (run_benchmark.py --record-front 로 녹화해야 함)"
        return row
    frames = clip["frames"]
    label_visible = clip["source"] == "view.mp4"
    row.update(clip=f"clips/{rid}.mp4", frames_t=[f["t"] for f in frames], video_source=clip["source"],
               exact_timing=clip["exact_timing"])
    if not client.ready:
        row["error"] = "모델 미설정 (NEBIUS_TOKEN_FACTORY_KEY, COSMOS_MODEL)"
        return row
    try:
        text = client.chat(judge_messages(r, frames, label_visible))
        j = parse_judgment(text, len(frames))
        row["judgment"] = j
        row["judge_raw"] = text[-1500:]
        if j is None:
            row["error"] = "영상 판단 형식 오류"
        row.update(compare(r, j, frames, label_visible))
        text2 = client.chat(comment_messages(r, frames, j, row))
        row.update(parse_answer(text2))
    except Exception as e:  # noqa: BLE001 — the screen must still update
        row["error"] = f"모델 호출 실패: {e}"
    return row


def process(records_path, screen, client: ModelClient | None = None, failures_only: bool = False,
            hypotheses=None, patterns=None, redo_errors: bool = False) -> int:
    """Judge + explain every new run (all runs by default: monitor misses hide in passes), redraw the screen."""
    from dashboard import refresh
    client = client or ModelClient()
    screen = Path(screen)
    screen.mkdir(parents=True, exist_ok=True)
    p = Path(records_path)
    done = read_commentary(screen)
    n = 0
    for raw in read_records(p) if p.exists() else []:
        r = view(raw)
        rid = rid_of(r)
        if r.get("outcome_category") in ("invalid", "no_trigger"):
            continue
        if failures_only and r.get("outcome_category") not in FAILURES:
            continue
        if rid in done and not (redo_errors and done[rid].get("error")):
            continue
        row = explain_run(r, screen, client)
        with (screen / "commentary.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
        refresh(p, screen, hypotheses=hypotheses, patterns=patterns)   # show each result as soon as it lands
    refresh(p, screen, hypotheses=hypotheses, patterns=patterns)
    return n


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="09b: vision judgment + commentary + screen")
    ap.add_argument("--check", action="store_true", help="list the models your key can use")
    ap.add_argument("--records")
    ap.add_argument("--screen")
    ap.add_argument("--failures-only", action="store_true", help="skip passing runs (cheaper, but no miss check)")
    ap.add_argument("--redo-errors", action="store_true", help="retry runs whose judgment failed")
    ap.add_argument("--hypotheses", default=None)
    ap.add_argument("--patterns", default=None)
    ap.add_argument("--open", action="store_true", help="open the screen in the browser once")
    ap.add_argument("--watch", action="store_true", help="keep going: process new runs as they are graded")
    a = ap.parse_args(argv)
    client = ModelClient()
    if a.check:
        if not client.key:
            print("NEBIUS_TOKEN_FACTORY_KEY is not set")
            return 1
        ids = client.models()
        cosmos = [i for i in ids if "cosmos" in (i or "").lower()]
        print(f"{len(ids)} models available at {client.base}")
        print("Cosmos models:", ", ".join(cosmos) if cosmos else "none listed for this key")
        print("COSMOS_MODEL =", client.model or "(not set)",
              "" if not client.model else ("(available)" if client.model in ids else "(NOT in the list)"))
        return 0
    if not (a.records and a.screen):
        ap.error("--records and --screen are required")
    kw = dict(hypotheses=a.hypotheses, patterns=a.patterns)
    n = process(a.records, a.screen, client, a.failures_only, redo_errors=a.redo_errors, **kw)
    if a.open:
        import webbrowser
        webbrowser.open((Path(a.screen) / "campaign.html").resolve().as_uri())
    print(f"judged {n} run(s) -> {Path(a.screen) / 'campaign.html'}")
    if a.watch:
        last = None
        try:
            while True:
                p = Path(a.records)
                stamp = p.stat().st_mtime if p.exists() else None
                if stamp != last:
                    process(a.records, a.screen, client, a.failures_only, **kw)
                    last = stamp
                time.sleep(2.0)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
