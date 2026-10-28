"""09 Monitor screen for people: one HTML page that updates by itself.

  campaign.html   left  : the scenario that just finished (fault, onset, switch, driving after the switch)
                  right : every run so far (counts + one row per run; click a row to show it on the left)
                  below left: hypothesis decisions (stage 2), and a hypothesis column in the right table
                  insights  : campaign analysis from monitor numbers (screen/insights.json)

How it is used
  start of campaign : python dashboard.py --records results/run_records.jsonl --out results/screen --open
                      (writes the page and opens the browser once)
  after every run   : explain.py --records results/run_records.jsonl --screen results/screen
                      (analyzes the numerical dataset, then calls refresh())
  watch mode        : python dashboard.py --records ... --out ... --open --watch
                      (re-draws whenever run_records.jsonl changes; use when the runner is not ours)

Plain HTML, no server, no internet, nothing to install. The page reloads itself every few seconds.
"""
from __future__ import annotations

import html
import hashlib
import json
import os
import time
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from feedback import hypotheses_from_yaml, hypothesis_tracking, read_records
from monitor import view

REFRESH_S = 5

# result label, colour class, one plain sentence
OUTCOME = {
    "pass": ("통과", "ok", "제때 전환했고, 전환 후에도 안정적으로 주행함"),
    "harmless": ("통과 (관측 창 내 전환 불필요)", "ok",
                 "평가 창에서 무대응 런의 차선 침범이 관측되지 않음 · 창 밖의 안전성은 미확인"),
    "incomplete": ("판정 미완료", "mut", "필수 측정값이 없어 전체 기준 통과 여부를 판단할 수 없음"),
    "no_trigger": ("시험 안 됨", "mut", "고장을 넣었지만 MPC 출력이 바뀌지 않음"),
    "F1_no_or_late_switch": ("F1 전환 실패·지연", "bad", "위험한 고장인데 LQR로 전환을 못 했거나 너무 늦게 함"),
    "F2_switch_transient": ("F2 전환 순간 불안정", "bad", "전환은 했지만 그 순간 조향이 크게 튐"),
    "F3_not_recovered": ("F3 전환 후 불안정", "bad", "평가 창에서 차선·충돌·설정된 조향 한계를 위반함"),
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
METRIC_KO = {"max_steering_rate_dps": "최대 조향 속도", "recovery_success": "회복 성공률", "recovery_robustness": "회복 여유",
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
.grid{display:grid;grid-template-columns:minmax(400px,7fr) 5fr;gap:16px;padding:16px 20px;align-items:start}
.col{background:var(--panel);border:1px solid var(--line);border-radius:4px;min-width:0}
.run td{overflow-wrap:anywhere}
.col>h2{margin:0;padding:10px 16px;font-size:13px;color:var(--mut);font-weight:600;border-bottom:1px solid var(--line);
display:flex;align-items:center;gap:8px}
.col>h2 button{margin-left:auto;font:inherit;font-size:12px;border:1px solid var(--line);background:var(--soft);
border-radius:3px;padding:2px 8px;cursor:pointer;color:var(--fg)}
.col>h2.section-title{padding:17px 18px 8px;font-size:24px;line-height:1.35;font-weight:700;color:var(--fg);border-bottom:0;flex-wrap:wrap}
.section-label{font-size:11px;letter-spacing:.06em;color:var(--mut);padding:16px 18px 0;font-weight:600}
.section-subtitle{font-size:12px;color:var(--mut);padding:0 18px 12px}
.switch-sequence{list-style:none;display:flex;align-items:center;gap:8px;margin:0 18px 14px;padding:0;font-size:12px;color:var(--mut)}
.switch-sequence li{padding:7px 10px;background:var(--soft);border:1px solid var(--line);border-radius:3px}
.switch-sequence .handover{font-weight:700;color:var(--acc);background:#edf3fc;border-color:#cfdef5}
.scenario-heading{display:flex;align-items:center;gap:10px;padding:9px 16px;font-size:12px;color:var(--mut);background:var(--soft);border-block:1px solid var(--line)}
.scenario-heading button{margin-left:auto;border:1px solid var(--line);background:var(--panel);font:inherit;padding:2px 8px;cursor:pointer;border-radius:3px}
.hypothesis-card{padding:16px 18px;border-bottom:1px solid var(--line)}.hypothesis-card:last-child{border-bottom:0}
.hypothesis-meta{display:flex;justify-content:space-between;align-items:center;font-size:11px;color:var(--mut)}
.hypothesis-card h3{font-size:18px;line-height:1.5;color:var(--fg);margin:8px 0 12px}
.group-definitions{display:grid;grid-template-columns:45px 1fr;gap:5px 10px;font-size:12px;margin:0 0 16px}
.group-definitions dt{color:var(--mut)}.group-definitions dd{margin:0;overflow-wrap:anywhere}
.stat-heading{font-size:12px;font-weight:600;margin:0 0 5px}.hypothesis-stats{font-size:13px}
.observed-effect{display:flex;align-items:baseline;gap:8px;margin-top:12px;font-size:13px}
.observed-effect b{font-size:22px;color:var(--acc)}.observed-effect span{font-size:11px;color:var(--mut)}
.stat-note{font-size:12px;color:var(--mut);line-height:1.6;margin:6px 0 0}
.hypothesis-area{border-top:1px solid var(--line)}
.col>h2.record-title{font-size:18px;font-weight:700;color:var(--fg);padding:16px 16px 8px;border-bottom:0}
.model-credit{font-size:12px;font-weight:600;color:var(--acc);background:#edf3fc;border-radius:3px;padding:4px 7px;margin-left:auto;white-space:nowrap}
.col>h2.insight-title{gap:12px;flex-wrap:wrap}
.record-summary{padding:9px 12px;border-bottom:1px solid var(--line);font-size:12px;color:var(--mut)}
.record-summary strong{font-weight:600;color:var(--fg)}

.col>h2.insight-title{font-size:24px;font-weight:700;color:var(--fg);padding:16px 18px 8px;border-bottom:0}
.insights{padding:0 18px 16px!important}
.insight-item{padding:16px 0;border-top:1px solid var(--line)}
.insight-item:first-of-type{border-top:0;padding-top:10px}
.insight-item h3{font-size:19px;line-height:1.45;font-weight:700;color:var(--acc);margin:0 0 7px}
.insight-item p{font-size:14px;line-height:1.75;margin:0;color:var(--fg)}
.insight-meta{font-size:11px;color:var(--mut);margin-bottom:4px}
.analysis-details,.hypothesis-details{font-size:12px;color:var(--mut);border-top:1px solid var(--line);padding-top:10px;margin-top:8px}
.analysis-details summary,.hypothesis-details summary{cursor:pointer;color:var(--mut);font-size:12px}
.analysis-details h4{font-size:12px;margin:14px 0 4px;color:var(--fg)}
.analysis-details p{overflow-wrap:anywhere}.analysis-details ul{padding-left:18px;line-height:1.7}
.hypothesis-details{margin:0 18px 16px}.hypothesis-details .hrow{padding:12px 0}
.eventline{display:flex;flex-wrap:wrap;gap:8px 18px;padding:10px 0;border-bottom:1px solid var(--line);font-size:12px;color:var(--mut)}
.eventline b{display:block;color:var(--fg);font-size:14px}
@media(max-width:600px){.switch-sequence{flex-wrap:wrap}.observed-effect{flex-wrap:wrap}}
.run,.insights{padding:14px 16px}.run[hidden]{display:none}
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
.list td{white-space:nowrap;padding:5px;font-size:12px}.list th{padding:6px 6px}.listbox{max-height:320px;overflow:auto}
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
  if(h)h.textContent=(id===latest)?'최근 시나리오 결과':'선택한 시나리오 결과';
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
    if oc == "pass" and r.get("assessment_complete") is False:
        return "incomplete"
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
         _f(s2.get("max_rate_dps"), 0, " °/s") if switched else "전환 안 함", f"{_f(s2.get('limit_dps'), 0)} °/s 이하 (설정값)"),
        ("차선 유지 (차체 기준)", s4.get("pass"),
         "–" if margin is None else (f"선까지 {_f(margin, 2, ' m')}" if margin >= 0 else f"선 넘음 {_f(-margin, 2, ' m')}"),
         "선 안쪽"),
        ("충돌", (not s4.get("collision")) if s4 else None,
         (f"{_f(m.get('SR4_collision_t'), 1, ' s')} {m.get('SR4_collision_other') or ''}" if s4.get("collision") else "없음"),
         "없음"),
        ("회복 (참고)", s5.get("pass") if s5.get("applies") else None,
         ("회복 못 함" if s5.get("recover_time_s") is None else _f(s5.get("recover_time_s"), 1, " s"))
         if s5.get("applies") else "전환 안 함", f"{_f(s5.get('within_s'), 1)} s 안에 중심 복귀 (설정값)"),
        ("조향 한계", s6.get("pass") if s6.get("applies") else None,
         f"{_f(s6.get('max_steer_deg'), 0, '°')} · {_f(s6.get('max_rate_dps'), 0, ' °/s')}" if s6.get("applies") else "기록 없음",
         f"설정 한계 · {_f(s6.get('limit_rate_dps'), 0)} °/s"),
    ]


def _stability_short(r) -> str:
    if r.get('assessment_complete') is False:
        return '<span class="mut">미측정</span>'
    if r.get("outcome_category") in ("invalid", "no_trigger"):
        return '<span class="mut">–</span>'
    names = {"SR2": "조향 튐", "SR4": "차선", "SR6": "조향 한계"}
    bad = [names[k] for k in (r.get("failed_requirements") or []) if k in names]
    if ((r.get("sr") or {}).get("SR4") or {}).get("collision"):
        bad = [b for b in bad if b != "차선"] + ["충돌"]
    return f'<span class="bad">✗ {_e(", ".join(bad))}</span>' if bad else '<span class="ok">✓</span>'


# ------------------------------------------------------------------ left: one run
FAILS = ("F1_no_or_late_switch", "F2_switch_transient", "F3_not_recovered")


def render_insights(report: dict | None) -> str:
    """Lead with concise interpretations; keep evidence and experiments available on demand."""
    if not report:
        return '<div class="empty">인사이트 대기 중</div>'
    credit = ('출력 형식 예시 · 실제 Cosmos 호출 없음' if report.get('status') == 'illustrative_demo'
              else 'Cosmos Reason이 모니터 판정 수치에서 도출한 해석' if report.get('status') == 'complete'
              else 'Cosmos Reason · 분석 대기')
    out = ['<div class="insights"><div class="insight-meta">' + _e(credit) + '</div>']
    if report.get("selection") == "failures_only":
        out.append('<div class="cap">실패 사례만 분석</div>')
    if report.get("error"):
        out.append('<div class="ai wait">' + _e(report["error"]) + '</div>')
    items = report.get("insights") or []
    for item in items:
        out.append('<article class="insight-item"><h3>' + _e(item.get("title")) + '</h3><p>'
                   + _e(item.get("interpretation") or item.get("finding")) + '</p></article>')
    if not items and report.get("summary"):
        out.append('<p>' + _e(report["summary"]) + '</p>')
    if items or report.get('next_experiments') or report.get('limitations'):
        out.append('<details class="analysis-details"><summary>분석 상세</summary>')
        if report.get('summary'):
            out.append('<p>' + _e(report['summary']) + '</p>')
        for item in items:
            evidence = '; '.join(str(e['run_id']) + ' · ' + e['metric'] + ' = ' + str(e['value'])
                                 for e in item.get('evidence') or [])
            out.append('<h4>' + _e(item.get('title')) + '</h4><p>' + _e(item.get('finding'))
                       + '</p><p>근거: ' + _e(evidence) + '</p><p>모델 확신: '
                       + _e(item.get('confidence')) + '</p>')
        if report.get('next_experiments'):
            out.append('<h4>다음 실험</h4><ul>' + ''.join('<li>' + _e(x['proposal']) + ' — '
                       + _e(x['rationale']) + '</li>' for x in report['next_experiments']) + '</ul>')
        if report.get('limitations'):
            out.append('<h4>분석 한계</h4><ul>' + ''.join('<li>' + _e(x) + '</li>'
                       for x in report['limitations']) + '</ul>')
        out.append('</details>')
    out.append('</div>')
    return ''.join(out)


def render_run_panel(r: dict, hidden: bool, replay: dict | None = None) -> str:
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
           f'<div class="result {cls}"><b>{_e(text)}</b><p>{_e(sentence)}</p></div>']
    if key == "invalid":
        out.append("</div>")
        return "".join(out)
    if key in FAILS:
        replay = replay or {}
        out.append('<div class="blk"><h3>실패 순간 리플레이 · 고장 → 전환 → 안전 위반</h3>')
        if replay.get('clip'):
            out.append('<video class="clip" src="' + _e(replay['clip']) + '" controls muted playsinline preload="metadata"></video>')
            out.append('<div class="cap">t=' + _f(replay.get('start_s')) + '–' + _f(replay.get('end_s'))
                       + ' s · 주목 이벤트 t=' + _f(replay.get('event_s')) + ' s · '
                       + _e(replay.get('source')) + ('' if replay.get('exact_timing') else ' · 시각은 근사') + '</div>')
        else:
            out.append('<div class="empty">' + _e(replay.get('error') or '녹화 영상 연결 대기 중') + '</div>')
        out.append('</div>')
    out.append('<div class="eventline"><span>고장 주입 <b>' + _f(r.get('actual_fault_onset'), 2, ' s')
               + '</b></span><span>제어권 인계 <b>' + _f(r.get('switch_time'), 2, ' s')
               + '</b></span><span>고장 → 인계 <b>' + _f(r.get('detection_latency_s'), 2, ' s') + '</b></span></div>')
    if r.get('assessment_complete') is False:
        out.append('<p class="cap">미평가 기준: ' + _e(', '.join(r.get('unassessed_requirements') or [])) + '</p>')
    phases = (r.get('diagnostics') or {}).get('phases') or {}
    if phases:
        items = []
        for phase, name in [('transition','전환 순간'),('post_switch','전환 이후')]:
            values = phases.get(phase) or {}
            if values.get('samples',0):
                items.append('<tr><td>' + name + '</td><td>' + _f(values.get('lateral_rmse_m'),3) + '</td><td>'
                             + _f(values.get('max_abs_lateral_accel_mps2'),2) + '</td><td>'
                             + _f(values.get('max_abs_lateral_jerk_mps3'),2) + '</td><td>'
                             + _e(values.get('status')) + '</td></tr>')
        if items:
            out.append('<details class="analysis-details"><summary>주행 성능 지표 · 참고</summary><table><tr><th>구간</th><th>횡오차 RMSE (m)</th><th>|ay| (m/s²)</th><th>|횡jerk| (m/s³)</th><th>관측</th></tr>'
                       + ''.join(items) + '</table><p class="cap">SR3 안정 영역 미검증 · 쾌적/추종 지표는 안전 판정과 분리</p></details>')
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


def _definition(value):
    if isinstance(value, dict):
        return " · ".join(f"{k}: {v}" for k, v in value.items())
    return str(value) if value else "조건 정의 미연결"


METRIC_UNITS = {"max_steering_rate_dps": "°/s", "margin_delta_m": "m", "lateral_error_rmse_m": "m",
                "pre_switch_max_lateral_error_m": "m", "command_jump_rad": "rad", "switch_count": "회"}


def render_hypotheses(records: list[dict], hypotheses: dict | None, delta_min_m=None) -> str:
    ht = hypothesis_tracking(records, hypotheses, delta_min_m)["hypotheses"]
    if not ht:
        return '<div class="empty">연결된 취약 가설이 없습니다.</div>'
    rows = []
    for hid, v in ht.items():
        st = v["status"]
        text, cls = HYP_STATUS.get(st, (st, "mut"))
        observed = v["descriptive"]
        condition, control = observed['condition'], observed['control']
        goal = v.get('next_checkpoint')
        n = min(condition['n'], control['n'])
        if st == 'testing' and goal:
            text = '표본 부족' if not v['status_history'] else '검증 중'
        title = v.get('claim') or PATTERN_KO.get(v.get('pattern')) or v.get('pattern') or hid
        metric = METRIC_KO.get(v['metric'], v['metric'])
        unit = METRIC_UNITS.get(v['metric'], '')
        def measurement(value):
            return _f(value, 2, (' ' + unit) if unit else '')
        delta = observed['mean_difference']
        difference = '비교 불가' if delta is None else f'{delta:+.2f}' + ((' ' + unit) if unit else '')
        group_rows = ''.join(
            '<tr><td>' + label + '</td><td class="num">' + str(group['n']) + '회</td><td class="num">'
            + measurement(group['mean']) + '</td><td class="num">' + measurement(group['min']) + '–'
            + measurement(group['max']) + '</td></tr>'
            for label, group in [('조건군', condition), ('대조군', control)])
        if v.get('ci') is not None:
            ci = v['ci']
            checkpoint = v['status_history'][-1]['checkpoint']
            inference = (f'{round(v["ci_level"] * 100)}% 신뢰구간: '
                         f'[{measurement(ci[0])}, {measurement(ci[1])}] · 각 {checkpoint}회 기준')
        elif goal:
            inference = f'신뢰구간 미산출 · 각 {goal}회 필요 (조건군 {max(0, goal-condition["n"])}회 / 대조군 {max(0, goal-control["n"])}회 추가)'
        else:
            inference = '신뢰구간 미산출'
        hist = v.get('status_history') or []
        guard = hist[-1].get('guard') if hist else None
        note = '<p class="stat-note">판정 보류: ' + _e(guard) + '</p>' if guard else ''
        rows.append('<article class="hypothesis-card"><div class="hypothesis-meta">' + _e(hid)
                    + '<span class="tag ' + cls + '">' + _e(text) + '</span></div>'
                    + '<h3>' + _e(title) + '</h3><dl class="group-definitions"><dt>조건군</dt><dd>'
                    + _e(_definition(v.get('condition_definition'))) + '</dd><dt>대조군</dt><dd>'
                    + _e(_definition(v.get('control_definition'))) + '</dd></dl>'
                    + '<div class="stat-heading">' + _e(metric) + ' · 현재 유효 표본</div>'
                    + '<table class="hypothesis-stats"><thead><tr><th>비교군</th><th>표본</th><th>평균</th><th>최소–최대</th></tr></thead><tbody>'
                    + group_rows + '</tbody></table><div class="observed-effect">관측 차이 <b>'
                    + _e(difference) + '</b><span>조건군 평균 − 대조군 평균</span></div>'
                    + '<p class="stat-note">' + _e(inference) + '</p>' + note
                    + ('<p class="stat-note">무효·결측 제외 ' + str(v['excluded_runs']) + '회</p>' if v['excluded_runs'] else '')
                    + '</article>')
    return ''.join(rows)


def render_page(records: list[dict], refresh=REFRESH_S, hypotheses: dict | None = None, delta_min_m=None,
                insights: dict | None = None, replays: dict | None = None) -> str:
    replays = replays or {}
    n = len(records)
    cnt = defaultdict(int)
    for r in records:
        cnt[_key(r)] += 1
    latest = _rid(records[-1]) if records else ""
    rounds = [r.get("round") for r in records if r.get("round") is not None]

    left = "".join(render_run_panel(r, hidden=(i != n - 1), replay=replays.get(str(r.get("run_id"))))
                   for i, r in enumerate(records)) or \
        '<div class="empty">아직 끝난 시나리오가 없습니다.</div>'

    failed = sum(cnt[k] for k in FAILS)
    counts_html = (f'<strong>{n}건</strong> · 통과 {cnt["pass"] + cnt["harmless"]}'
                   f' · 전환 실패 {failed} · 제외 {cnt["no_trigger"] + cnt["invalid"] + cnt["incomplete"]}')
    rows = []
    for r in reversed(records):
        fp = _fault(r)
        rows.append(f'<tr data-id="{_e(_rid(r))}"><td class="num mut">{_e(r.get("run_id") or "")}</td>'
                    f'<td>{_e(r.get("scenario_id") or "")}</td><td>{_e(fp.get("fault_type") or "–")}</td>'
                    f'<td>{_e(_onset_ko(fp.get("onset_event")))}</td><td class="num">{_switch(r)[0]}</td>'
                    f'<td>{_stability_short(r)}</td><td>{_hyp_cell(r)}</td><td>{_tag(r)}</td></tr>')
    table = (f'<div class="wrapx listbox"><table class="list"><thead><tr><th>런</th><th>시나리오</th><th>고장</th><th>시점</th>'
             f'<th>전환</th><th>운행 안정성</th><th>가설</th><th>결과</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
             if rows else '<div class="empty">아직 기록이 없습니다.</div>')

    body = f"""<header><h1>MPC → LQR 안전성 검증</h1><span class="sp"></span>
<span>{datetime.now().strftime("%H:%M:%S")} 갱신</span></header>
<div class="grid">
<div class="stack" id="focus-column">
<section class="col" id="switch-section">
<div class="section-label">검증 대상</div>
<h2 class="section-title">제어기 전환 순간<span class="model-credit">안전성 판정</span></h2>
<div class="section-subtitle">검증 대상: MPC → LQR 전환 및 전환 직후 안정성</div>
<div class="scenario-heading"><span id="lefttitle">최근 시나리오 결과</span><button id="back" hidden>최신으로</button></div>{left}</section>
<section class="col" id="nemotron-section">
<div class="section-label">NEMOTRON</div>
<h2 class="section-title">취약 가설 · 통계 검증</h2>
<div class="hypothesis-area">{render_hypotheses(records, hypotheses, delta_min_m)}</div></section>
</div>
<div class="stack" id="results-column">
<section class="col" id="verdict-section"><h2 class="record-title">전환 판정</h2><div class="record-summary">{counts_html}</div>{table}</section>
<section class="col" id="insight-section"><h2 class="insight-title">인사이트<span class="model-credit">Cosmos Reason 분석 결과</span></h2>{render_insights(insights)}</section>
</div>
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
    raw_records = read_records(p) if p.exists() else []
    records = [view(r) for r in raw_records]
    from replay import prepare_replays
    replays = prepare_replays(raw_records, out_dir)
    ip = Path(out_dir) / "insights.json"
    insights = json.loads(ip.read_text(encoding="utf-8")) if ip.exists() else None
    # A dashboard-only refresh may be ahead of the last model analysis.
    if insights:
        from explain import build_dataset, dumps
        expected = build_dataset(read_records(p) if p.exists() else [], insights.get("selection") == "failures_only")
        expected_hash = hashlib.sha256(dumps(expected).encode("utf-8")).hexdigest()
        if insights.get("input_hash") != expected_hash:
            insights = {"error": "기록이 변경되어 이전 인사이트를 숨겼습니다. explain.py를 다시 실행하세요.",
                        "statistics": expected["summary"], "selected_runs": expected["selected_runs"]}
    if isinstance(hypotheses, (str, Path)):
        hypotheses = hypotheses_from_yaml(hypotheses, patterns)
    target = Path(out_dir) / "campaign.html"
    _write(target, render_page(records, refresh_s, hypotheses, delta_min_m, insights, replays))
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
