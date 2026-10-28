"""09 Monitor screen for people: one self-refreshing HTML page (ported from junwoo/sim-eval a5242b2 dashboard.py).

  <out>/screen/campaign.html
     left  : the run that just finished (fault, onset, switch, SR checks, our recovery v2 verdict, video explanation)
     right : every run so far (counts by sr_outcome + one row per run; click a row to show it on the left)
     below left: recovery v2 outcome summary (recovery_outcome / final_safe_recovery counts)

Input is OUR run_record (engine.monitor.evaluate: recovery v2 fields + sr_* fields from engine.sr_checks) and
OUR explanations.jsonl (engine.explainer via run.py stage 10). run.py redraws it after every round.

  python -m engine.dashboard --records outputs/<run>/run_records.jsonl --out outputs/<run>/screen [--open] [--watch]

Plain HTML, no server, no internet. The page reloads itself every few seconds (not while a video plays).
Dropped from the teammate version: the stage-2 hypothesis tracking panel (it read the old H-01..H-11 hypotheses and
the teammate's feedback.py; our S-set hypotheses are tracked by the loop's own report) and the blind vision
"judgment/agreement" block (our explainer never judges safety; its verification result is shown instead).
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

REFRESH_S = 5

# sr_outcome (engine.sr_checks) -> (label, colour class, one plain sentence)
OUTCOME = {
    "pass": ("통과", "ok", "제때 전환했고, 전환 후에도 안정적으로 주행함 (회복 판정: recovery v2)"),
    "harmless": ("통과 (전환 불필요)", "ok",
                 "전환하지 않았지만, 이 고장은 아무 대응을 안 해도 차선을 벗어나지 않는 고장이라 문제없음"),
    "no_trigger": ("시험 안 됨", "mut", "고장을 넣었지만 MPC 출력이 바뀌지 않음"),
    "F1_no_or_late_switch": ("F1 전환 실패·지연", "bad", "위험한 고장인데 LQR로 전환을 못 했거나 너무 늦게 함"),
    "F2_switch_transient": ("F2 전환 순간 불안정", "bad", "전환은 했지만 그 순간 조향이 크게 튐"),
    "F3_not_recovered": ("F3 전환 후 불안정", "bad", "전환은 했지만 recovery v2 기준으로 안전하게 회복하지 못함"),
    "invalid": ("무효", "mut", "시뮬레이션 기록에 문제가 있어 채점하지 않음"),
}
RECOVERY_KO = {"safe_recovery": "안전 회복", "recovered_after_departure": "이탈 후 복귀", "unsettled": "안정 안 됨",
               "unrecovered": "회복 실패", "no_switch": "전환 없음", "simulation_invalid": "무효"}
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
.insw{padding:16px 20px 0}.ins .isum{padding:10px 16px;border-bottom:1px solid var(--line);font-size:14px}
.ins .irow{padding:8px 16px;border-bottom:1px solid var(--line)}.ins .irow:last-child{border-bottom:0}
.ins .irow b{font-size:14px}.ins .irow p{margin:2px 0;font-size:13px}.ins .ev{font-size:12px;color:var(--mut)}
.ins .ev code{background:var(--soft);padding:0 4px;border-radius:3px}.ins h3{margin:0;padding:8px 16px 0;font-size:12px;color:var(--mut)}
.ins ul{margin:4px 0 8px 34px;padding:0;font-size:13px}.ins .conf{font-size:11px;margin-left:6px}
.ins .note{padding:10px 16px;color:var(--mut);font-size:13px}.ins h2 a{margin-left:auto;font-size:12px;color:var(--acc)}
@media (max-width:860px){.grid{grid-template-columns:1fr;padding:12px}header{padding:10px 12px;flex-wrap:wrap;gap:4px}
.kv td:first-child{width:70px}}
"""


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


def view(rec: dict) -> dict:
    """Our run_record -> the flat fields this page reads (sr_checks details next to recovery v2 fields)."""
    out = dict(rec)
    g = rec.get("sr_checks") or {}
    out["sr"] = g.get("sr") or {}
    out["sr_measures"] = g.get("sr_measures") or {}
    out["note"] = g.get("note")
    out["failed_requirements"] = rec.get("sr_failed_requirements") or []
    oc = rec.get("sr_outcome")
    if rec.get("sim_status") == "invalid" or oc in (None, "invalid"):
        oc = "invalid"
        out["invalid_reason"] = rec.get("invalid_reason") or "; ".join(g.get("invalid_reasons") or []) or None
    out["display_outcome"] = oc
    p = rec.get("parameters") or {}
    f = p.get("fault") or {}
    out["fault_plan"] = {"fault_type": f.get("fault_id") or f.get("mode"), "mode": f.get("mode"),
                         "params": f.get("params"), "onset_event": p.get("onset_event"),
                         "onset_offset_s": p.get("onset_offset_s")}
    env = dict(p.get("environment") or {})
    env.setdefault("route", p.get("route_id"))
    out["environment"] = env
    return out


def _key(r) -> str:
    oc = r.get("display_outcome") or "invalid"
    return "harmless" if oc == "pass" and r.get("note") == "undetected_harmless" else oc


def _tag(r) -> str:
    text, cls, _ = OUTCOME.get(_key(r), (_key(r), "mut", ""))
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


def _limits(spec: dict | None) -> dict:
    rq = (spec or {}).get("requirements") or {}   # SR monitor v3.1 sections (engine/sr_checks.py grades with them)
    ss = (spec or {}).get("safety_specification") or {}
    p02 = next((p for p in ss.get("properties") or [] if p.get("id") == "P-02"), {})
    return {"sr2": (rq.get("SR2") or {}).get("limit_dps", 60.0), "sr6": (rq.get("SR6") or {}).get("max_rate_dps", 90.0),
            "acc": p02.get("lateral_acc_threshold_mps2", 4.41), "win": ss.get("after_switch_window_s", 3.0),
            "rj": ss.get("recovery_judgment") or {}}


def _switch(r) -> tuple[str, str]:
    """(short text for the list, sentence for the detail panel)."""
    sr1 = (r.get("sr") or {}).get("SR1") or {}
    lat = r.get("detection_latency_s")
    if r.get("switch_time") is not None:
        if lat is None and r.get("actual_fault_onset") is not None:
            lat = r["switch_time"] - r["actual_fault_onset"]
        limit = f" (한계 {_f(sr1.get('limit_s'))} s, {_e(sr1.get('limit_source'))})" if sr1.get("applies") else ""
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


def _checks(r, lim: dict) -> list[tuple]:
    """Driving after the switch: (name, ok/None, measured, limit)."""
    sr, m = r.get("sr") or {}, r.get("sr_measures") or {}
    s2, s4, s6 = (sr.get(k) or {} for k in ("SR2", "SR4", "SR6"))
    switched = r.get("switch_time") is not None
    margin = s4.get("margin_m")
    labels = r.get("recovery_labels") or {}
    acc = (labels.get("lat_acc_excess") or {}).get("primary")
    ro = r.get("recovery_outcome")
    return [
        ("전환 순간 조향 튐 (SR2)", s2.get("pass") if s2.get("applies") else None,
         _f(s2.get("max_rate_dps"), 0, " °/s") if switched else "전환 안 함", f"{_f(lim['sr2'], 0)} °/s 이하"),
        ("차선 유지 · 차체 사각형 (SR4)", s4.get("pass"),
         "–" if margin is None else (f"선까지 {_f(margin, 2, ' m')}" if margin >= 0 else f"선 넘음 {_f(-margin, 2, ' m')}"),
         "선 안쪽"),
        ("충돌", (not s4.get("collision")) if s4 else None,
         (f"{_f(m.get('SR4_collision_t'), 1, ' s')} {m.get('SR4_collision_other') or ''}" if s4.get("collision") else "없음"),
         "없음"),
        ("회복 (recovery v2)", None if ro in (None, "no_switch", "simulation_invalid") else ro == "safe_recovery",
         RECOVERY_KO.get(ro, ro or "–") + (f" · lrr {_f(r.get('lane_recovery_robustness'))}" if switched else ""),
         f"{_f(lim['win'], 0)} s 창, {_f(lim['rj'].get('e_settle_m', 0.3), 1)} m · {_f(lim['rj'].get('psi_settle_rad', 0.1), 1)} rad · "
         f"{_f(lim['rj'].get('settle_hold_s', 1.0), 0)} s 유지"),
        ("횡가속도 (P-02, baseline 대비)", None if acc is None else not acc,
         _f(r.get("lat_acc_excess_mps2"), 2, " m/s²"), f"{_f(lim['acc'], 2)} m/s² 이하"),
        ("조향 한계 (SR6)", s6.get("pass") if s6.get("applies") else None,
         f"{_f(s6.get('max_steer_deg'), 0, '°')} · {_f(s6.get('max_rate_dps'), 0, ' °/s')}" if s6.get("applies") else "기록 없음",
         f"차량 한계 · {_f(lim['sr6'], 0)} °/s"),
    ]


def _stability_short(r) -> str:
    if _key(r) in ("invalid", "no_trigger"):
        return '<span class="mut">–</span>'
    names = {"SR2": "조향 튐", "SR4": "차선", "SR5": "회복", "SR6": "조향 한계"}
    bad = [names[k] for k in (r.get("failed_requirements") or []) if k in names]
    if ((r.get("sr") or {}).get("SR4") or {}).get("collision"):
        bad = [b for b in bad if b != "차선"] + ["충돌"]
    return f'<span class="bad">✗ {_e(", ".join(bad))}</span>' if bad else '<span class="ok">✓</span>'


# ------------------------------------------------------------------ left: one run
def _rel(path: str | None, out_dir: Path) -> str | None:
    if not path:
        return None
    try:
        return os.path.relpath(path, out_dir)
    except ValueError:
        return path


def _commentary_html(r: dict, c: dict | None, out_dir: Path, active: bool = False) -> str:
    """Clip + our explainer's mechanism text + its verification against monitor timestamps."""
    if not c:
        return '<div class="ai wait">영상 해설 없음 (실패·경계 근처 런만 해설)</div>' if active else ""
    out = []
    clip = c.get("clip") or {}
    if clip.get("video"):
        out.append(f'<video class="clip" src="{_e(_rel(clip["video"], out_dir))}" controls muted playsinline preload="metadata"></video>'
                   f'<div class="cap">{_f(clip.get("start_s"), 1)}–{_f(clip.get("end_s"), 1)} s · {_e(clip.get("source") or "")}'
                   f' · {_e(clip.get("alignment") or "")}</div>')
    ex = c.get("explanation") or {}
    v = c.get("verification") or {}
    if ex:
        tags = ", ".join(ex.get("mechanism_tags") or [])
        ok = v.get("accepted")
        failed = [k for k, x in (v.get("checks") or {}).items() if not x.get("pass")]
        out.append(f'<div class="ai"><span class="src">영상 해설 · {_e(c.get("model") or "")} · 판정은 바꾸지 않음</span>'
                   f'<div class="sum">{_e(ex.get("mechanism") or "")}</div>'
                   f'<div class="why">태그: {_e(tags or "–")}</div>'
                   f'<div class="agree{"" if ok else " dis"}">{"검증 통과 (모니터 시각 ±허용오차 일치)" if ok else "검증 실패: " + _e(", ".join(failed) or v.get("reason") or "")}</div></div>')
    elif v.get("reason"):
        out.append(f'<div class="ai wait">{_e(v["reason"])}</div>')
    return "".join(out)


def render_run_panel(r: dict, hidden: bool, lim: dict, out_dir: Path, commentary: dict | None = None,
                     active: bool = False) -> str:
    fp, env = r.get("fault_plan") or {}, r.get("environment") or {}
    key = _key(r)
    text, cls, sentence = OUTCOME.get(key, (key, "mut", ""))
    if key == "invalid":
        sentence = f"{sentence} ({r.get('invalid_reason') or ''})"
    where = " · ".join(x for x in (env.get("route") or env.get("town"),
                                   _f(env.get("max_speed_kmh"), 0, " km/h") if env.get("max_speed_kmh") is not None else None,
                                   f"라운드 {r.get('round')}" if r.get("round") is not None else None) if x)
    final = r.get("final_safe_recovery")
    verdict = {True: "final_safe_recovery = true", False: "final_safe_recovery = false", None: "final_safe_recovery = 판정 불가"}[final]
    out = [f'<div class="run" id="run-{_e(_rid(r))}"{" hidden" if hidden else ""}>',
           f'<div class="title">{_e(r.get("scenario_id") or "-")} <span class="mut" style="font-size:13px;font-weight:400">'
           f'{_e(r.get("run_id") or "")}</span></div><div class="where">{_e(where)}</div>',
           f'<div class="result {cls}"><b>{_e(text)}</b><p>{_e(sentence)}</p>'
           f'<p class="mut">우리 판정: {_e(RECOVERY_KO.get(r.get("recovery_outcome"), r.get("recovery_outcome")))} · {_e(verdict)}'
           f' · {_e(r.get("outcome_category"))}</p></div>',
           _commentary_html(r, commentary, out_dir, active and key not in ("invalid", "no_trigger"))]
    if key == "invalid":
        out.append("</div>")
        return "".join(out)
    params = fp.get("params") or {}
    pt = ", ".join(f"{k} {_f(v, 3) if isinstance(v, (int, float)) else _e(v)}" for k, v in params.items())
    out.append(f'''<div class="blk"><h3>고장</h3><table class="kv">
<tr><td>종류</td><td><b>{_e(fp.get("fault_type") or "–")}</b> {_e(fp.get("mode") or "")}{f" · {pt}" if pt else ""}</td></tr>
<tr><td>시점</td><td>{_e(ONSET_KO.get(fp.get("onset_event"), fp.get("onset_event") or "–"))} {_f(fp.get("onset_offset_s"), 2, " s")} <span class="mut num">· {_f(r.get("actual_fault_onset") or r.get("planned_fault_onset"), 2, " s")}</span></td></tr>
</table></div>''')
    out.append(f'<div class="blk"><h3>전환</h3><table class="kv"><tr><td>결과</td><td>{_switch(r)[1]}</td></tr>'
               f'<tr><td>사유</td><td>{_e(r.get("trigger_source") or "–")}</td></tr></table></div>')
    rows = "".join(
        f'<tr><td class="mark {"ok" if ok else "bad" if ok is False else "mut"}">{"✓" if ok else "✗" if ok is False else "–"}</td>'
        f'<td>{_e(name)}</td><td class="num">{_e(meas)}</td><td class="mut">{_e(lm)}</td></tr>'
        for name, ok, meas, lm in _checks(r, lim))
    out.append(f'<div class="blk"><h3>전환 후 운행 안정성</h3><table>{rows}</table></div></div>')
    return "".join(out)


def _ai_cell(c: dict | None) -> str:
    if not c or not c.get("explanation"):
        return '<span class="mut">–</span>'
    return '<span class="ok">✓</span>' if (c.get("verification") or {}).get("accepted") else '<span class="bad">✗</span>'


def render_recovery_summary(records: list[dict]) -> str:
    cnt = defaultdict(int)
    for r in records:
        cnt[r.get("recovery_outcome") or "–"] += 1
    safe = sum(1 for r in records if r.get("final_safe_recovery") is True)
    unsafe = sum(1 for r in records if r.get("final_safe_recovery") is False)
    if not records:
        return '<div class="empty">아직 기록이 없습니다.</div>'
    rows = "".join(f'<div class="hrow"><div class="h1"><b>{_e(RECOVERY_KO.get(k, k))}</b>'
                   f'<span class="tag mut">{v}</span></div></div>' for k, v in sorted(cnt.items()))
    return (f'<div class="hrow meta"><span>final_safe_recovery: <b>{safe}</b> 안전 · <b>{unsafe}</b> 불안전</span></div>' + rows)


CONF_KO = {"high": ("확신 높음", "ok"), "medium": ("확신 중간", "mut"), "low": ("확신 낮음", "mut")}


def load_insights(out_dir: Path) -> dict | None:
    """<results>/insights/insights.json (engine/insights.py, 09 explain.py) next to the screen folder."""
    p = Path(out_dir).parent / "insights" / "insights.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (OSError, json.JSONDecodeError):
        return None


def render_insights(ins: dict | None, out_dir: Path = Path(".")) -> str:
    """Cosmos campaign insights (text-only analysis of SR-monitor numbers). Hypotheses, not verdicts."""
    detail = Path(out_dir).parent / "insights" / "campaign.html"
    link = (f'<a href="{_e(os.path.relpath(detail, out_dir).replace(os.sep, "/"))}">자세히 보기</a>'
            if detail.is_file() else "")
    head = f'<h2>Cosmos 캠페인 인사이트 (SR 모니터 수치 분석, 참고용){link}</h2>'
    if not ins or ins.get("status") != "complete":
        why = ("캠페인이 끝나면 생성됩니다." if not ins else
               f"생성 실패: {ins.get('error') or ins.get('status')}")
        return f'<div class="insw"><section class="col ins">{head}<div class="note">아직 인사이트가 없습니다. {_e(why)}</div></section></div>'
    st = ins.get("statistics") or {}
    rate = st.get("failure_rate")
    stat = (f'<div class="note">SR 판정 분포 {_e(", ".join(f"{k} {v}" for k, v in (st.get("outcome_counts") or {}).items()))}'
            f' · 실패율 {_f(rate * 100 if rate is not None else None, 1, "%")} (채점된 런 {_e(st.get("graded_triggered_runs"))}개)'
            f' · 모델 {_e(ins.get("model"))} · {_e(ins.get("created"))}</div>')
    rows = []
    for it in ins.get("insights") or []:
        label, cls = CONF_KO.get(it.get("confidence"), (it.get("confidence") or "–", "mut"))
        ev = " · ".join(f'<code>{_e(e.get("run_id"))}</code> {_e(e.get("metric"))} = <b class="num">{_e(e.get("value"))}</b>'
                        for e in it.get("evidence") or [])
        rows.append(f'<div class="irow"><b>{_e(it.get("title"))}</b><span class="tag {cls} conf">{_e(label)}</span>'
                    f'<p>관측: {_e(it.get("finding"))}</p><p class="mut">해석(가설): {_e(it.get("interpretation"))}</p>'
                    f'<div class="ev">근거: {ev}</div></div>')
    nxt = "".join(f'<li>{_e(x.get("proposal"))} <span class="mut">— {_e(x.get("rationale"))}'
                  f'{(" (근거 " + _e(", ".join(x.get("run_ids") or [])) + ")") if x.get("run_ids") else ""}</span></li>'
                  for x in ins.get("next_experiments") or [])
    lim = "".join(f"<li>{_e(x)}</li>" for x in ins.get("limitations") or [])
    return (f'<div class="insw"><section class="col ins">{head}<div class="isum">{_e(ins.get("summary"))}</div>{stat}'
            f'{"".join(rows) or "<div class=note>근거가 있는 인사이트가 없습니다.</div>"}'
            + (f"<h3>다음 실험 제안</h3><ul>{nxt}</ul>" if nxt else "")
            + (f"<h3>한계</h3><ul>{lim}</ul>" if lim else "") + "</section></div>")


def render_page(records: list[dict], refresh=REFRESH_S, commentary: dict | None = None, spec: dict | None = None,
                out_dir: Path = Path("."), insights: dict | None = None) -> str:
    commentary = commentary or {}
    lim = _limits(spec)
    n = len(records)
    cnt = defaultdict(int)
    for r in records:
        cnt[_key(r)] += 1
    latest = _rid(records[-1]) if records else ""
    rounds = [r.get("round") for r in records if r.get("round") is not None]
    active = bool(commentary)
    left = "".join(render_run_panel(r, hidden=(i != n - 1), lim=lim, out_dir=out_dir,
                                    commentary=commentary.get(r.get("run_id")), active=active)
                   for i, r in enumerate(records)) or '<div class="empty">아직 끝난 시나리오가 없습니다.</div>'
    counts = [("전체", n, ""), ("통과", cnt["pass"] + cnt["harmless"], "ok"),
              ("F1 전환 실패·지연", cnt["F1_no_or_late_switch"], "bad"),
              ("F2 전환 순간 불안정", cnt["F2_switch_transient"], "bad"),
              ("F3 전환 후 불안정", cnt["F3_not_recovered"], "bad"),
              ("시험 안 됨·무효", cnt["no_trigger"] + cnt["invalid"], ""),
              ("해설 검증 실패", sum(1 for c in commentary.values() if c.get("explanation")
                                and not (c.get("verification") or {}).get("accepted")), "bad")]
    counts_html = "".join(f'<div><b class="num {c if v else ""}">{v}</b><span>{_e(t)}</span></div>' for t, v, c in counts)
    rows = []
    for r in reversed(records):
        fp = r.get("fault_plan") or {}
        rows.append(f'<tr data-id="{_e(_rid(r))}"><td class="num mut">{_e(r.get("run_id") or "")}</td>'
                    f'<td>{_e(r.get("scenario_id") or "")}</td><td>{_e(fp.get("fault_type") or "–")}</td>'
                    f'<td>{_e(ONSET_KO.get(fp.get("onset_event"), fp.get("onset_event") or "–"))}</td><td class="num">{_switch(r)[0]}</td>'
                    f'<td>{_stability_short(r)}</td><td>{_e(RECOVERY_KO.get(r.get("recovery_outcome"), r.get("recovery_outcome") or "–"))}</td>'
                    f'<td>{_tag(r)}</td><td>{_ai_cell(commentary.get(r.get("run_id")))}</td></tr>')
    table = (f'<div class="wrapx listbox"><table class="list"><thead><tr><th>런</th><th>시나리오</th><th>고장</th><th>시점</th>'
             f'<th>전환</th><th>운행 안정성</th><th>회복 (v2)</th><th>SR 결과</th><th>영상 해설</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
             if rows else '<div class="empty">아직 기록이 없습니다.</div>')
    body = f"""<header><h1>MPC → LQR 전환 안전성 검증</h1><span class="sp"></span>
<span>라운드 {max(rounds) if rounds else "–"} · {n}회 실행 · {datetime.now().strftime("%H:%M:%S")} 갱신 · {refresh}초마다 자동 새로고침</span></header>
{render_insights(insights, out_dir)}
<div class="grid">
<div class="stack">
<section class="col"><h2><span id="lefttitle">방금 돌린 시나리오</span><button id="back" hidden>최신으로</button></h2>{left}</section>
<section class="col"><h2>회복 판정 (recovery v2) 요약</h2>{render_recovery_summary(records)}</section>
</div>
<section class="col"><h2>지금까지 돌린 시나리오 ({n})</h2><div class="counts">{counts_html}</div>{table}</section>
</div>"""
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<noscript><meta http-equiv="refresh" content="{refresh}"></noscript><title>전환 안전성 검증</title><style>{CSS}</style></head>'
            f'<body data-latest="{_e(latest)}" data-refresh="{refresh}">{body}{SCRIPT}</body></html>')


# ------------------------------------------------------------------ entry points
def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def refresh(records_path, out_dir, refresh_s=REFRESH_S, explanations=None, spec=None) -> Path:
    """Re-draw the page from run_records.jsonl (+ explanations.jsonl). run.py calls this after every round."""
    out_dir = Path(out_dir)
    records = [view(r) for r in _read_jsonl(Path(records_path))]
    commentary = {}
    for row in _read_jsonl(Path(explanations)) if explanations else []:
        commentary[row.get("run_id")] = row          # latest line per run wins
    if spec is None or isinstance(spec, (str, Path)):
        from .monitor import load_safety_spec
        spec = load_safety_spec(spec)
    target = out_dir / "campaign.html"
    _write(target, render_page(records, refresh_s, commentary, spec, out_dir, insights=load_insights(out_dir)))
    return target


def open_screen(records_path, out_dir, **kw) -> Path:
    target = refresh(records_path, out_dir, **kw)
    webbrowser.open(target.resolve().as_uri())
    return target


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m engine.dashboard", description="09 Monitor screen (self-refreshing HTML)")
    ap.add_argument("--records", required=True, help="run_records.jsonl (run.py output folder)")
    ap.add_argument("--out", required=True, help="screen folder (campaign.html is written here)")
    ap.add_argument("--explanations", default=None, help="explanations.jsonl (default: next to --records)")
    ap.add_argument("--spec", default=None, help="09 safety_spec.yaml (limits shown on the page)")
    ap.add_argument("--open", action="store_true", help="open the browser once")
    ap.add_argument("--watch", action="store_true", help="re-draw whenever the records file changes")
    a = ap.parse_args(argv)
    expl = a.explanations or str(Path(a.records).with_name("explanations.jsonl"))
    kw = {"explanations": expl, "spec": a.spec}
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
