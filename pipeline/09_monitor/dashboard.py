"""09 Monitor screen for people: one HTML page that updates by itself.

  campaign.html   left  : the scenario that just finished (fault, onset, switch, driving after the switch)
                  right : every run so far (counts + one row per run; click a row to show it on the left)
                  below left: hypothesis decisions (stage 2), and a hypothesis column in the right table
                  failures  : clip + model commentary from explain.py (screen/commentary.jsonl, screen/clips/)

How it is used
  start of campaign : python dashboard.py --records results/run_records.jsonl --out results/screen --open
                      (writes the page and opens the browser once)
  after every run   : explain.py --records results/run_records.jsonl --screen results/screen
                      (cuts failure clips, adds model commentary, then calls refresh())
  watch mode        : python dashboard.py --records ... --out ... --open --watch
                      (re-draws whenever run_records.jsonl changes; use when the runner is not ours)

Plain HTML, no server, no internet, nothing to install. The page reloads itself every few seconds.
"""
from __future__ import annotations

import html
import json
import os
import time
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from feedback import (LANE_KO, MOTION_KO, NEAR_KO, OBJ_KO, SIDE_KO, hypotheses_from_yaml,
                      hypothesis_tracking, read_records)
from monitor import view

REFRESH_S = 5

# result label, colour class, one plain sentence
OUTCOME = {
    "pass": ("통과", "ok", "제때 전환했고, 전환 후에도 안정적으로 주행함"),
    "harmless": ("통과 (전환 불필요)", "ok",
                 "전환하지 않았지만, 이 고장은 아무 대응을 안 해도 차선을 벗어나지 않는 고장이라 문제없음"),
    "no_trigger": ("시험 안 됨", "mut", "고장을 넣었지만 MPC 출력이 바뀌지 않음"),
    "F1_no_or_late_switch": ("F1 전환 실패·지연", "bad", "위험한 고장인데 LQR로 전환을 못 했거나 너무 늦게 함"),
    "F2_switch_transient": ("F2 전환 순간 불안정", "bad", "전환은 했지만 그 순간 조향이 크게 튐"),
    "F3_not_recovered": ("F3 전환 후 불안정", "bad", "전환은 했지만 그 뒤 차선을 벗어나거나 충돌함"),
    "invalid": ("무효", "mut", "시뮬레이션 기록에 문제가 있어 채점하지 않음"),
}
# short Korean summary per vulnerability pattern (02 vulnerability_patterns.yaml); fallback = pattern name
PATTERN_KO = {
    "P-OFF-DESIGN": "LQR 설계 조건에서 먼 상황에서 전환하면 회복이 나쁘다",
    "P-DELAYED-SWITCH": "곡선에 들어간 뒤에 전환되면 전환 전 오차가 커진다",
    "P-CHATTERING": "MPC 계산 시간이 한계 근처면 전환이 반복된다",
    "P-UNRECOVERABLE-SWITCH": "이미 회복하기 어려운 상태에서 전환되면 LQR이 못 살린다",
    "P-BACKUP-SATURATION": "전환 직후 LQR 조향이 한계에 걸리면 회복이 나쁘다",
    "P-MPC-MODEL-MISMATCH": "실제 차와 MPC 모델이 다르면(마찰 등) 오차가 커진다",
    "P-MPC-DEADLINE": "어려운 조건에서 MPC 계산이 제한 시간을 넘긴다",
    "P-MPC-CONSTRAINT-STRESS": "급한 조건에서 MPC 제약 위반이 늘어난다",
}
METRIC_KO = {"recovery_success": "회복 성공률", "recovery_robustness": "회복 여유",
             "pre_switch_max_lateral_error_m": "전환 전 최대 오차", "switch_count": "전환 횟수",
             "lateral_error_rmse_m": "횡오차 RMSE", "deadline_miss_rate": "MPC 시간 초과율",
             "mpc_constraint_violation_rate": "MPC 제약 위반율", "margin_delta_m": "차선 여유 변화"}
HYP_STATUS = {"supported": ("맞음 · 약점 확인", "bad"), "refuted": ("틀림", "ok"),
              "inconclusive": ("판단 불가", "mut"), "testing": ("진행 중", "mut"), "blocked": ("보류 · 정보 부족", "mut")}
ONSET_KO = {"curve_entry": "곡선 진입", "curve_apex": "곡선 정점", "straight": "직선", "curve_exit": "곡선 탈출"}

CSS = """
:root{--bg:#eef1f4;--panel:#fff;--line:#d9dee4;--soft:#f6f8fa;--fg:#1c2530;--mut:#66727f;
--ok:#1a7f37;--okbg:#e6f4ea;--bad:#c4271c;--badbg:#fdecea;--acc:#1f5fbf}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--fg)}
body{font:14px/1.5 "Segoe UI","Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif}
.num{font-variant-numeric:tabular-nums}.mut{color:var(--mut)}.ok{color:var(--ok)}.bad{color:var(--bad)}
header{display:flex;align-items:center;gap:16px;padding:12px 20px;background:var(--panel);border-bottom:1px solid var(--line)}
header h1{font-size:16px;margin:0}header .sp{flex:1}header span{font-size:12px;color:var(--mut)}
.grid{display:grid;grid-template-columns:minmax(340px,5fr) 7fr;gap:16px;padding:16px 20px;align-items:start}
.col{background:var(--panel);border:1px solid var(--line);border-radius:4px;min-width:0}
.run td{overflow-wrap:anywhere}
.col>h2{margin:0;padding:10px 16px;font-size:13px;color:var(--mut);font-weight:600;border-bottom:1px solid var(--line);
display:flex;align-items:center;gap:8px}
.col>h2 button{margin-left:auto;font:inherit;font-size:12px;border:1px solid var(--line);background:var(--soft);
border-radius:3px;padding:2px 8px;cursor:pointer;color:var(--fg)}
.run{padding:14px 16px}.run[hidden]{display:none}
.title{font-size:18px;font-weight:700}.where{font-size:12px;color:var(--mut);margin-bottom:12px}
.result{border-left:4px solid var(--mut);background:var(--soft);padding:10px 12px;margin-bottom:14px}
.result.ok{border-color:var(--ok);background:var(--okbg)}.result.bad{border-color:var(--bad);background:var(--badbg)}
.result b{font-size:16px;display:block;color:var(--fg)}.result p{margin:2px 0 0;font-size:13px;color:var(--fg)}
.blk{margin-top:14px}.blk h3{font-size:12px;color:var(--mut);margin:0 0 6px;font-weight:600}
table{width:100%;border-collapse:collapse}
td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-size:12px;color:var(--mut);font-weight:600;background:var(--soft);white-space:nowrap}
.kv td:first-child{width:90px;color:var(--mut)}
.mark{font-weight:700;width:22px}
.counts{display:flex;flex-wrap:wrap;border-bottom:1px solid var(--line)}
.counts div{flex:1 1 90px;padding:10px 14px;border-right:1px solid var(--line)}.counts div:last-child{border-right:0}
.counts b{display:block;font-size:22px}.counts span{font-size:12px;color:var(--mut)}
.list tr{cursor:pointer}.list tbody tr:hover td{background:var(--soft)}
.list tr.sel td{background:#e8f0fb}
.tag{display:inline-block;font-size:12px;font-weight:600;padding:0 6px;border-radius:3px;white-space:nowrap}
.tag.ok{background:var(--okbg)}.tag.bad{background:var(--badbg)}.tag.mut{background:var(--soft);color:var(--mut)}
.empty{padding:20px 16px;color:var(--mut)}
.wrapx{overflow-x:auto}
.clip{width:100%;display:block;background:#000;border-radius:3px;margin:0 0 4px}
.cap{font-size:12px;color:var(--mut);margin-bottom:12px}
.ai{border:1px solid var(--line);border-radius:4px;padding:10px 12px;margin-bottom:6px}
.ai .src{display:block;font-size:11px;color:var(--mut);margin-bottom:2px}
.ai .sum{font-weight:600;margin:2px 0 6px}.ai ul{margin:0 0 6px 18px;padding:0;font-size:13px}
.ai .why{font-size:13px}.ai .warn{margin-top:8px;padding:6px 8px;background:#fff6e0;border-left:3px solid #c98a00;font-size:12px}
.ai.wait{color:var(--mut);font-size:13px}
.ai h4{margin:0 0 6px;font-size:12px;color:var(--mut);font-weight:600}
.sev{display:inline-block;font-weight:700;padding:1px 8px;border-radius:3px;font-size:13px}
.sev0{background:var(--okbg);color:var(--ok)}.sev1{background:#eef3e2;color:#4d6b16}
.sev2{background:#fff1d6;color:#8a5a00}.sev3{background:var(--badbg);color:var(--bad)}
.facts{display:grid;grid-template-columns:72px 1fr;gap:3px 8px;font-size:13px;margin-top:8px}.facts span:nth-child(odd){color:var(--mut)}
.agree{margin-top:8px;font-size:12px;padding:5px 8px;border-radius:3px;background:var(--soft)}
.agree.dis{background:#fff6e0;border-left:3px solid #c98a00}
.vtag{font-size:11px;color:var(--acc);margin-left:4px}
.stack{display:flex;flex-direction:column;gap:16px;min-width:0}
.hrow{padding:10px 16px;border-bottom:1px solid var(--line)}.hrow:last-child{border-bottom:0}
.hrow .h1{display:flex;align-items:center;gap:8px}.hrow .h1 b{font-size:14px}.hrow .h1 .tag{margin-left:auto}
.hrow .what{margin:2px 0 6px;font-size:13px}.hrow .meta{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:12px;color:var(--mut)}
.hrow .meta b{color:var(--fg);font-weight:600}
.list td{white-space:nowrap;padding:6px 6px;font-size:13px}.list th{padding:6px 6px}.listbox{max-height:calc(100vh - 210px);overflow:auto}
.listbox thead th{position:sticky;top:0;z-index:1}
.bar{display:inline-block;width:60px;height:5px;background:var(--line);border-radius:3px;vertical-align:middle;margin-right:6px}
.bar i{display:block;height:100%;background:var(--acc);border-radius:3px}
@media (max-width:860px){.grid{grid-template-columns:1fr;padding:12px}header{padding:10px 12px;flex-wrap:wrap;gap:4px}
.kv td:first-child{width:70px}}
"""

# click a row -> show that run on the left; the choice survives the auto reload
SCRIPT = """<script>
(function(){
 var KEY='sel:'+location.pathname, latest=document.body.dataset.latest;
 function get(){try{return sessionStorage.getItem(KEY)}catch(e){return null}}
 function put(v){try{v?sessionStorage.setItem(KEY,v):sessionStorage.removeItem(KEY)}catch(e){}}
 function show(id){
  var runs=document.querySelectorAll('.run'), found=false;
  runs.forEach(function(el){var on=el.id==='run-'+id;el.hidden=!on;found=found||on});
  if(!found&&latest){id=latest;var l=document.getElementById('run-'+latest);if(l)l.hidden=false}
  document.querySelectorAll('.list tr[data-id]').forEach(function(tr){tr.classList.toggle('sel',tr.dataset.id===id)});
  var h=document.getElementById('lefttitle'), b=document.getElementById('back');
  if(h)h.textContent=(id===latest)?'방금 돌린 시나리오':'선택한 런';
  if(b)b.hidden=(id===latest);
 }
 document.querySelectorAll('.list tr[data-id]').forEach(function(tr){
  tr.addEventListener('click',function(){var id=tr.dataset.id;put(id===latest?null:id);show(id)})});
 var b=document.getElementById('back');if(b)b.addEventListener('click',function(){put(null);show(latest)});
 show(get()||latest);
 // reload for new results, but never while a video is playing
 var every=+(document.body.dataset.refresh||5)*1000;
 setInterval(function(){
  var playing=[].some.call(document.querySelectorAll('video'),function(v){return !v.paused&&!v.ended});
  if(!playing)location.reload();
 },every);
 try{var y=sessionStorage.getItem('y:'+location.pathname);if(y)scrollTo(0,+y);
  addEventListener('beforeunload',function(){sessionStorage.setItem('y:'+location.pathname,scrollY)})}catch(e){}
})();
</script>"""


# ------------------------------------------------------------------ helpers
def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def _f(x, nd=2, unit="") -> str:
    if x is None:
        return "–"
    try:
        return f"{float(x):.{nd}f}{unit}"
    except (TypeError, ValueError):
        return _e(x)


def _fault(r) -> dict:
    return r.get("fault_plan") or (r.get("parameters") or {}).get("fault_plan") or {}


def _env(r) -> dict:
    return r.get("environment") or (r.get("parameters") or {}).get("environment") or {}


def _onset_ko(ev) -> str:
    return ONSET_KO.get(ev, ev or "–")


def _key(r) -> str:
    """Outcome key for display: 'harmless' is a pass where the switch was not needed."""
    oc = r.get("outcome_category") or "invalid"
    return "harmless" if oc == "pass" and r.get("note") == "undetected_harmless" else oc


def _tag(r) -> str:
    text, cls, _ = OUTCOME.get(_key(r), (r.get("outcome_category"), "mut", ""))
    return f'<span class="tag {cls}">{_e(text)}</span>'


def _rid(r) -> str:
    rid = str(r.get("run_id") or f"{r.get('scenario_id')}-{r.get('planned_fault_onset')}")
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in rid)


def _write(path: Path, text: str):
    """Atomic write: the browser never reloads a half-written page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _switch(r) -> tuple[str, str]:
    """(short text for the list, sentence for the detail panel)."""
    sr1 = (r.get("sr") or {}).get("SR1") or {}
    lat = r.get("detection_latency_s")
    if r.get("switch_time") is not None:
        if lat is None and r.get("actual_fault_onset") is not None:
            lat = r["switch_time"] - r["actual_fault_onset"]
        limit = f" (한계 {_f(sr1.get('limit_s'))} s)" if sr1.get("applies") else ""
        if sr1.get("applies") and not sr1.get("pass"):
            return (f'<span class="bad">✗ 늦음 {_f(lat)} s</span>',
                    f'<span class="bad">✗ 늦게 전환</span> — 고장 후 {_f(lat)} s 만에 LQR로 넘어감{limit}')
        return (f'<span class="ok">✓ {_f(lat)} s</span>',
                f'<span class="ok">✓ 전환 성공</span> — 고장 후 {_f(lat)} s 만에 LQR로 넘어감{limit}')
    if sr1.get("applies"):
        return ('<span class="bad">✗ 안 함</span>', '<span class="bad">✗ 전환 안 함</span> — 위험한 고장인데 MPC가 계속 운전함')
    if r.get("note") == "undetected_harmless":
        return ('<span class="mut">– 불필요</span>',
                '<span class="mut">– 전환 안 함</span> — 대응하지 않아도 안전한 고장이라 괜찮음')
    return ('<span class="mut">–</span>', '<span class="mut">– 해당 없음</span>')


def _checks(r) -> list[tuple]:
    """Driving after the switch: (name, ok/None, measured, limit)."""
    sr, m = r.get("sr") or {}, r.get("sr_measures") or {}
    s2, s4, s5, s6 = (sr.get(k) or {} for k in ("SR2", "SR4", "SR5", "SR6"))
    switched = r.get("switch_time") is not None
    margin = s4.get("margin_m")
    return [
        ("전환 순간 조향 튐", s2.get("pass") if s2.get("applies") else None,
         _f(s2.get("max_rate_dps"), 0, " °/s") if switched else "전환 안 함", "60 °/s 이하"),
        ("차선 유지 (차체 기준)", s4.get("pass"),
         "–" if margin is None else (f"선까지 {_f(margin, 2, ' m')}" if margin >= 0 else f"선 넘음 {_f(-margin, 2, ' m')}"),
         "선 안쪽"),
        ("충돌", (not s4.get("collision")) if s4 else None,
         (f"{_f(m.get('SR4_collision_t'), 1, ' s')} {m.get('SR4_collision_other') or ''}" if s4.get("collision") else "없음"),
         "없음"),
        ("회복 (참고)", s5.get("pass") if s5.get("applies") else None,
         ("회복 못 함" if s5.get("recover_time_s") is None else _f(s5.get("recover_time_s"), 1, " s"))
         if s5.get("applies") else "전환 안 함", "3 s 안에 중심 복귀"),
        ("조향 한계", s6.get("pass") if s6.get("applies") else None,
         f"{_f(s6.get('max_steer_deg'), 0, '°')} · {_f(s6.get('max_rate_dps'), 0, ' °/s')}" if s6.get("applies") else "기록 없음",
         "차량 한계 · 90 °/s"),
    ]


def _stability_short(r) -> str:
    if r.get("outcome_category") in ("invalid", "no_trigger"):
        return '<span class="mut">–</span>'
    names = {"SR2": "조향 튐", "SR4": "차선", "SR6": "조향 한계"}
    bad = [names[k] for k in (r.get("failed_requirements") or []) if k in names]
    if ((r.get("sr") or {}).get("SR4") or {}).get("collision"):
        bad = [b for b in bad if b != "차선"] + ["충돌"]
    return f'<span class="bad">✗ {_e(", ".join(bad))}</span>' if bad else '<span class="ok">✓</span>'


# ------------------------------------------------------------------ left: one run
FAILS = ("F1_no_or_late_switch", "F2_switch_transient", "F3_not_recovered")


AGREE_KO = {"agree_risky": ("일치 · 모니터와 영상 모두 위험", False), "agree_safe": ("일치 · 모니터와 영상 모두 안전", False),
            "agree_minor": ("대체로 일치 · 영상상으로는 경미", False),
            "monitor_may_miss": ("불일치 · 모니터는 통과인데 영상은 위험 → 사람 확인 필요", True),
            "video_shows_no_risk": ("불일치 · 모니터는 실패인데 영상상 위험이 안 보임 → 기준·측정 확인", True)}
SEV_KO = {0: "위험 없음", 1: "경미", 2: "위험", 3: "충돌·직전"}


def _sev(sev) -> str:
    return f'<span class="sev sev{sev}">위험도 {sev}/3 · {SEV_KO.get(sev, "")}</span>'


def _commentary_html(r: dict, c: dict | None, active: bool = False) -> str:
    """Clip + vision judgment (the model, blind to the verdict) + comparison + commentary."""
    if not c:
        return '<div class="ai wait">영상 판단 대기 중… (explain.py)</div>' if active else ""
    out = []
    if c.get("clip"):
        out.append(f'<video class="clip" src="{_e(c["clip"])}" controls muted playsinline preload="metadata"></video>'
                   f'<div class="cap">고장 1 s 전부터 · {_e(c.get("video_source") or "")}'
                   f'{"" if c.get("exact_timing", True) else " · 시각은 근사"}</div>')
    j = c.get("judgment")
    if j:
        sur = "<br>".join(f"{SIDE_KO.get(x['side'], '')} {OBJ_KO.get(x['type'], x['type'])} · {NEAR_KO.get(x['proximity'], '')}"
                          for x in j.get("surroundings") or []) or "관련 물체 없음"
        text, dis = AGREE_KO.get(c.get("agreement"), (c.get("agreement") or "", False))
        sw = ""
        if c.get("switch_check") == "mismatch":
            sw = '<div class="agree dis">화면의 MPC/LQR 표시가 바뀐 시점이 로그의 전환 시각과 다름 → 로그 확인</div>'
        near = f"있음 · {_e(j.get('near_miss_detail') or '')}" if j.get("near_miss") else "없음"
        out.append(f'<div class="ai"><span class="src">AI 판단 · 영상만 보고 (판정 숫자는 안 봄) · {_e(c.get("model") or "")}'
                   f' · 확신 {_e(j.get("confidence"))}</span>{_sev(j["severity"])}'
                   f'<div class="facts"><span>차선</span><span>{_e(LANE_KO.get(j["lane"], "불분명"))}</span>'
                   f'<span>움직임</span><span>{_e(MOTION_KO.get(j["motion"], "불분명"))}</span>'
                   f'<span>주변</span><span>{sur}</span><span>근접</span><span>{near}</span></div>'
                   f'<div class="agree{" dis" if dis else ""}">{_e(text)}</div>{sw}</div>')
    if c.get("summary") or c.get("why"):
        tl = "".join(f"<li>{_e(x)}</li>" for x in c.get("timeline") or [])
        ctx = f'<div class="why" style="margin-top:4px">{_e(c["risk_context"])}</div>' if c.get("risk_context") else ""
        out.append(f'<div class="ai"><span class="src">AI 해설</span>'
                   f'<div class="sum">{_e(c.get("summary") or "")}</div>{f"<ul>{tl}</ul>" if tl else ""}'
                   f'<div class="why">{_e(c.get("why") or "")}</div>{ctx}</div>')
    if c.get("error"):
        out.append(f'<div class="ai wait">{_e(c["error"])}</div>')
    return "".join(out)


def render_run_panel(r: dict, hidden: bool, commentary: dict | None = None, active: bool = False) -> str:
    fp, env = _fault(r), _env(r)
    key = _key(r)
    text, cls, sentence = OUTCOME.get(key, (key, "mut", ""))
    if key == "invalid":
        sentence = f"{sentence} ({r.get('invalid_reason') or ''})"
    where = " · ".join(x for x in (env.get("route") or env.get("town"),
                                   _f(env.get("speed_mps"), 0, " m/s") if env.get("speed_mps") is not None else None,
                                   f"라운드 {r.get('round')}" if r.get("round") is not None else None) if x)
    out = [f'<div class="run" id="run-{_e(_rid(r))}"{" hidden" if hidden else ""}>',
           f'<div class="title">{_e(r.get("scenario_id") or "-")} <span class="mut" style="font-size:13px;font-weight:400">'
           f'{_e(r.get("run_id") or "")}</span></div><div class="where">{_e(where)}</div>',
           f'<div class="result {cls}"><b>{_e(text)}</b><p>{_e(sentence)}</p></div>',
           _commentary_html(r, commentary, active and r.get("outcome_category") not in ("invalid", "no_trigger"))]
    if key == "invalid":
        out.append("</div>")
        return "".join(out)
    mag = fp.get("magnitude", fp.get("strength"))
    dur = fp.get("duration_s")
    out.append(f'''<div class="blk"><h3>고장</h3><table class="kv">
<tr><td>종류</td><td><b>{_e(fp.get("fault_type") or "–")}</b>{f" · 세기 {_e(mag)}" if mag is not None else ""}{f" · {_f(dur, 1, ' s')} 동안" if dur is not None else ""}</td></tr>
<tr><td>시점</td><td>{_e(_onset_ko(fp.get("onset_event")))} <span class="mut num">· {_f(r.get("actual_fault_onset") or r.get("planned_fault_onset"), 2, " s")}</span></td></tr>
</table></div>''')
    out.append(f'<div class="blk"><h3>전환</h3><table class="kv"><tr><td>결과</td><td>{_switch(r)[1]}</td></tr>'
               '</table></div>')
    rows = "".join(
        f'<tr><td class="mark {"ok" if ok else "bad" if ok is False else "mut"}">{"✓" if ok else "✗" if ok is False else "–"}</td>'
        f'<td>{_e(name)}</td><td class="num">{_e(meas)}</td><td class="mut">{_e(lim)}</td></tr>'
        for name, ok, meas, lim in _checks(r))
    out.append(f'<div class="blk"><h3>전환 후 운행 안정성</h3><table>{rows}</table></div></div>')
    return "".join(out)


def _ai_cell(c: dict | None) -> str:
    if not c or not c.get("judgment"):
        return '<span class="mut">–</span>'
    sev = c["judgment"]["severity"]
    flag = AGREE_KO.get(c.get("agreement"), ("", False))[1] or c.get("switch_check") == "mismatch"
    warn = ' <span class="bad" title="모니터와 영상 판단 불일치">⚠</span>' if flag else ""
    return (f'<span class="sev sev{sev}">{sev}</span>'
            f'{warn}')


# ------------------------------------------------------------------ hypotheses
def _hyp_cell(r) -> str:
    """Right-table column: which hypothesis run this is (condition / control), or GP exploration."""
    arms = r.get("hypothesis_arms") or {}
    if not arms:
        return '<span class="mut">탐색</span>'
    return "<br>".join(f'{_e(h)} <span class="mut">{"조건군" if a == "condition" else "대조군" if a == "control" else _e(a)}</span>'
                       for h, a in sorted(arms.items()))


def _hyp_summary(v) -> str:
    pat = v.get("pattern") or v.get("claim") or ""
    for k, t in PATTERN_KO.items():
        if pat.startswith(k):
            return t
    claim = (v.get("claim") or pat or "").strip()
    return claim if len(claim) <= 40 else claim[:38].rstrip() + "…"


def render_hypotheses(records: list[dict], hypotheses: dict | None, delta_min_m=None) -> str:
    ht = hypothesis_tracking(records, hypotheses, delta_min_m)["hypotheses"]
    if not ht:
        return '<div class="empty">가설 전용 런이 아직 없습니다.</div>'
    rows, held = [], []
    for hid, v in ht.items():
        st = v["status"]
        if st == "testing" and not (v["runs_condition"] or v["runs_control"]):
            held.append((hid, v))          # nothing run yet (blocked or not started): one summary line below
            continue
        text, cls = HYP_STATUS.get(st, (st, "mut"))
        n = min(v["runs_condition"], v["runs_control"])
        goal = v.get("next_checkpoint") or 30
        hist = v.get("status_history") or []
        decided_at = hist[-1]["checkpoint"] if st != "testing" and hist else None
        guard = hist[-1].get("guard") if hist else None
        GUARD_KO = {"switch_count": "반복 전환이 한 번도 안 나와서 아직 결론 안 냄",
                    "secondary": "보조 지표가 반대로 나와서 아직 결론 안 냄"}
        guard_ko = next((t for k, t in GUARD_KO.items() if guard and guard.startswith(k)), guard)
        when = f"각 {decided_at}회 기준 판정" if decided_at else f"각 {goal}회가 되면 판정"
        eff = "아직 계산 안 함" if v["d_estimate"] is None else (
            f'<b class="num">{v["d_estimate"]:+.2f}</b> <span class="num">[{v["ci"][0]:+.2f}, {v["ci"][1]:+.2f}]</span>')
        note = f'<div class="meta">{hist[-1]["checkpoint"]}회 판정: {_e(guard_ko)}</div>' if guard_ko and st == "testing" else ""
        rows.append(f'<div class="hrow"><div class="h1"><b>{_e(hid)}</b>'
                    f'<span class="mut" style="font-size:12px">지표: {_e(METRIC_KO.get(v["metric"], v["metric"]))}</span>'
                    f'<span class="tag {cls}">{_e(text)}</span></div>'
                    f'<div class="what">{_e(_hyp_summary(v))}</div>'
                    f'<div class="meta"><span>조건군 / 대조군 <b class="num">{v["runs_condition"]} / {v["runs_control"]}회</b>'
                    f' <span class="bar"><i style="width:{min(100, round(100 * n / goal))}%"></i></span>{_e(when)}</span>'
                    f'<span>효과 (99% 구간) {eff}</span></div>{note}</div>')
    if held:
        blocked = [h for h, v in held if v.get("hypothesis_status") == "blocked"]
        waiting = [h for h, v in held if v.get("hypothesis_status") != "blocked"]
        parts = []
        if blocked:
            parts.append(f'보류 {len(blocked)}개 ({", ".join(blocked)}) · 제어기 정보가 부족해 아직 시험 못 함')
        if waiting:
            parts.append(f'대기 {len(waiting)}개 ({", ".join(waiting)}) · 아직 돌린 런 없음')
        rows.append(f'<div class="hrow meta">{_e(" / ".join(parts))}</div>')
    return "".join(rows)


def render_page(records: list[dict], refresh=REFRESH_S, hypotheses: dict | None = None, delta_min_m=None,
                commentary: dict | None = None) -> str:
    commentary = commentary or {}
    n = len(records)
    cnt = defaultdict(int)
    for r in records:
        cnt[_key(r)] += 1
    latest = _rid(records[-1]) if records else ""
    rounds = [r.get("round") for r in records if r.get("round") is not None]

    active = bool(commentary)
    left = "".join(render_run_panel(r, hidden=(i != n - 1), commentary=commentary.get(_rid(r)), active=active)
                   for i, r in enumerate(records)) or \
        '<div class="empty">아직 끝난 시나리오가 없습니다.</div>'

    counts = [("전체", n, ""), ("통과", cnt["pass"] + cnt["harmless"], "ok"),
              ("F1 전환 실패·지연", cnt["F1_no_or_late_switch"], "bad"),
              ("F2 전환 순간 불안정", cnt["F2_switch_transient"], "bad"),
              ("F3 전환 후 불안정", cnt["F3_not_recovered"], "bad"),
              ("시험 안 됨·무효", cnt["no_trigger"] + cnt["invalid"], ""),
              ("AI 불일치 · 확인 필요", sum(1 for c in commentary.values()
                                         if AGREE_KO.get(c.get("agreement"), ("", False))[1]
                                         or c.get("switch_check") == "mismatch"), "bad")]
    counts_html = "".join(f'<div><b class="num {c if v else ""}">{v}</b><span>{_e(t)}</span></div>' for t, v, c in counts)
    rows = []
    for r in reversed(records):
        fp = _fault(r)
        rows.append(f'<tr data-id="{_e(_rid(r))}"><td class="num mut">{_e(r.get("run_id") or "")}</td>'
                    f'<td>{_e(r.get("scenario_id") or "")}</td><td>{_e(fp.get("fault_type") or "–")}</td>'
                    f'<td>{_e(_onset_ko(fp.get("onset_event")))}</td><td class="num">{_switch(r)[0]}</td>'
                    f'<td>{_stability_short(r)}</td><td>{_hyp_cell(r)}</td><td>{_tag(r)}</td><td>{_ai_cell(commentary.get(_rid(r)))}</td></tr>')
    table = (f'<div class="wrapx listbox"><table class="list"><thead><tr><th>런</th><th>시나리오</th><th>고장</th><th>시점</th>'
             f'<th>전환</th><th>운행 안정성</th><th>가설</th><th>결과</th><th>AI 영상 판단</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
             if rows else '<div class="empty">아직 기록이 없습니다.</div>')

    body = f"""<header><h1>MPC → LQR 전환 안전성 검증</h1><span class="sp"></span>
<span>라운드 {max(rounds) if rounds else "–"} · {n}회 실행 · {datetime.now().strftime("%H:%M:%S")} 갱신 · {refresh}초마다 자동 새로고침</span></header>
<div class="grid">
<div class="stack">
<section class="col"><h2><span id="lefttitle">방금 돌린 시나리오</span><button id="back" hidden>최신으로</button></h2>{left}</section>
<section class="col"><h2>가설 판정 <span style="font-weight:400">· 조건군과 대조군을 각각 10·20·30회 돌릴 때마다 판정</span></h2>{render_hypotheses(records, hypotheses, delta_min_m)}</section>
</div>
<section class="col"><h2>지금까지 돌린 시나리오 ({n})</h2><div class="counts">{counts_html}</div>{table}</section>
</div>"""
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<noscript><meta http-equiv="refresh" content="{refresh}"></noscript><title>전환 안전성 검증</title><style>{CSS}</style></head>'
            f'<body data-latest="{_e(latest)}" data-refresh="{refresh}">{body}{SCRIPT}</body></html>')


# ------------------------------------------------------------------ entry points
def refresh(records_path, out_dir, refresh_s=REFRESH_S, hypotheses=None, patterns=None, delta_min_m=None,
            **_ignored) -> Path:
    """Re-draw the page. Call after every graded run.
    hypotheses: path to 02 hypotheses.yaml (or the dict from feedback.hypotheses_from_yaml); patterns: path to
    vulnerability_patterns.yaml for the per-metric decision thresholds."""
    p = Path(records_path)
    records = [view(r) for r in read_records(p)] if p.exists() else []
    commentary = {}
    cp = Path(out_dir) / "commentary.jsonl"          # written by explain.py (latest line per run wins)
    if cp.exists():
        for line in cp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                commentary[row["run_id"]] = row
    if isinstance(hypotheses, (str, Path)):
        hypotheses = hypotheses_from_yaml(hypotheses, patterns)
    target = Path(out_dir) / "campaign.html"
    _write(target, render_page(records, refresh_s, hypotheses, delta_min_m, commentary))
    return target


def open_screen(records_path, out_dir, **kw) -> Path:
    """Start of a campaign: write the page (even if empty) and open it in the browser once."""
    target = refresh(records_path, out_dir, **kw)
    webbrowser.open(target.resolve().as_uri())
    return target


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="09 Monitor screen (self-refreshing HTML)")
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hypotheses", default=None, help="02_hypotheses/hypotheses.yaml")
    ap.add_argument("--patterns", default=None, help="02_hypotheses/vulnerability_patterns.yaml")
    ap.add_argument("--delta-min", type=float, default=None)
    ap.add_argument("--open", action="store_true", help="open the browser once")
    ap.add_argument("--watch", action="store_true", help="re-draw whenever the records file changes")
    a = ap.parse_args(argv)
    kw = {"hypotheses": a.hypotheses, "patterns": a.patterns, "delta_min_m": a.delta_min}
    target = (open_screen if a.open else refresh)(a.records, a.out, **kw)
    print(f"screen -> {target}")
    if a.watch:
        last = None
        try:
            while True:
                p = Path(a.records)
                stamp = p.stat().st_mtime if p.exists() else None
                if stamp != last:
                    refresh(a.records, a.out, **kw)
                    last = stamp
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
