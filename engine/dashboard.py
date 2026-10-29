"""09 Monitor screen for people: one self-refreshing HTML page (ported from junwoo/sim-eval a5242b2 dashboard.py).

  <out>/screen/campaign.html

One fixed screen (100vh, no page scroll, no click needed), concise professional English:
  header: topic "MPC → LQR Handover Stability Evaluation", method band + 3 columns (Latest Scenario / scenario table /
          Hypotheses + Insights)
  band  : method flow Nemotron code analysis → Hypothesis → Fault scenario generation → CARLA simulation →
          Statistical verification (counts from the files)
  left  : latest run (or the clicked one): verdict, one terse line from recorded times, facts, video, and the full
          post-handover check table (Item | Measured | Threshold)
  middle: SR count strip + one dense row per run (latest 20, "+k more" beyond)
  right : hypotheses tested (02_objectives question, 00_prior_payload testable hypotheses + status from runs) and
          the Cosmos campaign insights ("Driving Safety" = recovery v2 outcome)

Input is OUR run_record (engine.monitor.evaluate: recovery v2 fields + sr_* fields from engine.sr_checks) and
OUR explanations.jsonl (engine.explainer via run.py stage 10). run.py redraws it after every run.
Every number on the page comes from those records; a missing value is shown as "Unknown", never guessed.
All UI text is English; Korean can only appear when it comes from the data (e.g. a Cosmos insight title without
structured evidence), which is why Pretendard stays in the font stack as a fallback.

  python -m engine.dashboard --records outputs/<run>/run_records.jsonl --out outputs/<run>/screen [--open] [--watch]

Plain HTML, no server, no internet. The page reloads itself every few seconds (not while a video plays) and keeps
the selected run, the scroll position and every opened details block across reloads.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import os
import time
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REFRESH_S = 5

# sr_outcome (engine.sr_checks) -> (label, colour class, one plain sentence)
OUTCOME = {
    "pass": ("Pass", "ok", "Switched in time and kept the lane after the switch"),
    "harmless": ("Pass (no switch needed)", "ok", "No switch, but the car kept its lane (not a failure)"),
    "no_trigger": ("Not tested", "mut", "Fault injected, but the MPC output did not change"),
    "F1_no_or_late_switch": ("Controller handover failed/late", "bad", "Hazardous fault, but no switch to LQR or too late"),
    "F2_switch_transient": ("Unstable at handover", "bad", "Switched, but steering spiked at the moment of the switch"),
    "F3_not_recovered": ("Unstable after handover", "bad", "Switched, but could not hold the lane stably"),
    "invalid": ("Not graded (invalid, not a failure)", "mut",
                "Recording or baseline conditions not met; not graded and not counted as a safety failure"),
}
# recovery v2 outcome = "Driving Safety" (lane keeping after the handover) on the page
RECOVERY_KO = {"safe_recovery": "Lane kept", "recovered_after_departure": "Back after departure", "unsettled": "Not settled",
               "unrecovered": "Not recovered", "no_switch": "No switch · held", "simulation_invalid": "Invalid"}
ONSET_KO = {"curve_entry": "Curve entry", "curve_apex": "Curve apex", "straight": "Straight", "curve_exit": "Curve exit"}

# result bucket -> (score label, short note, colour class)
BUCKET = {
    "safe": ("Safe response", "Detected, switched and kept the lane", "g"),
    "danger": ("Unsafe", "Late or missed detection, lane departure, collision", "r"),
    "harmless": ("Undetected · harmless", "Not detected but kept the lane (not a failure)", "n"),
    "na": ("Not graded", "Recording conditions not met (not a failure)", "x"),
}
KEY_BUCKET = {"pass": "safe", "harmless": "harmless", "F1_no_or_late_switch": "danger", "F2_switch_transient": "danger",
              "F3_not_recovered": "danger", "invalid": "na", "no_trigger": "na"}
# fault mode -> (short name, what it does). Checked against scenario_space_av_stack.yaml (F-01..F-07).
FAULT_KO = {
    "solver_fail": ("Solver Failure", "MPC reports that it found no solution"),
    "stale": ("Stale Command", "The previous command is resent instead of a new one"),
    "output_delay": ("Output Delay", "A steering command from several steps earlier is sent as current"),
    "bias": ("Steering Bias", "A constant angle is added to the steering command"),
    "drift": ("Ramp Bias / Drift", "Steering error grows over time"),
    "saturate": ("Steering Saturation", "The steering command is pinned toward the maximum angle"),
    "intermittent_bias": ("Intermittent Bias", "A bias switches on and off periodically"),
}
WHEN_KO = {"curve_entry": "Curve entry", "curve_apex": "Curve apex", "straight": "Straight", "curve_exit": "Curve exit"}
UNKNOWN = "Unknown"
# Screen wording of the testable prior-knowledge hypotheses in 00_prior_payload.json (others shown as written)
PK_KO = {"PK-002": "Fixed-gain LQR degrades as the state moves away from its design point (larger error at the switch)",
         "PK-003": "Longer detection and handover latency leaves less lane-keeping margin"}
# English wording of the research question when 02_objectives.json only carries question_ko
QUESTION_EN = {"S1": "How far off the path can the car be at the switch (error state), and how hard can LQR commands hit "
                     "the steering-angle and steering-rate limits, before LQR fails to recover within the lane?"}
MAX_ROWS = 20


CSS = """
@font-face{font-family:"Inter";src:url("fonts/Inter.ttf") format("truetype");font-weight:100 900;font-style:normal;font-display:swap}
@font-face{font-family:"Pretendard Variable";src:url("fonts/PretendardVariable.woff2") format("woff2-variations"),url("fonts/PretendardVariable.woff2") format("woff2");font-weight:45 920;font-style:normal;font-display:swap}
:root{
--bg:#f3f4f6;--surface:#ffffff;--line:#e3e6ea;--line-soft:#eceef1;
--fg:#111827;--fg2:#374151;--mut:#6b7280;--mut2:#9ca3af;--acc:#2f5bd3;--acc-soft:#e8eefb;
--ok:#167a3e;--ok-bg:#eef7f1;--ok-line:#cde6d6;
--bad:#b42318;--bad-bg:#fdf1f0;--bad-line:#f1cfca;
--neu:#4b5563;--neu-bg:#f3f4f6;--neu-line:#dfe2e6;
--t-left:#fcfcfd;--t-mid:#f8fafd;--t-hyp:#fcfaf6;--t-ins:#f6faf8;
--d-left:#6b7280;--d-mid:#2f5bd3;--d-hyp:#b7791f;--d-ins:#1f8a5b;
--r-card:.5rem;--r-chip:.25rem;--s1:.25rem;--s2:.5rem;--s3:.75rem;--s4:1rem;--s6:1.5rem}
*{box-sizing:border-box}
html{font-size:clamp(11px,1.48vh,16px)}
html,body{margin:0;height:100%;overflow:hidden;background:var(--bg);color:var(--fg)}
body{font:400 .9375rem/1.45 "Inter","Pretendard Variable","Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR","Noto Sans CJK KR",system-ui,sans-serif;
-webkit-font-smoothing:antialiased;word-break:keep-all;overflow-wrap:anywhere;display:grid;grid-template-rows:auto auto minmax(0,1fr);grid-template-columns:minmax(0,1fr);
height:100vh;padding:0 var(--s4) var(--s4);gap:var(--s3)}
b,strong{font-weight:600}
.num{font-variant-numeric:tabular-nums}.mut{color:var(--mut)}.ok{color:var(--ok)}.bad{color:var(--bad)}
/* ---------- header ---------- */
header{display:flex;align-items:flex-end;gap:var(--s4);padding:var(--s4) var(--s1) 0}
header .ttl{flex:1 1 auto;min-width:0}
header h1{margin:0;font-size:2.25rem;line-height:1.15;font-weight:600;letter-spacing:-.025em;white-space:nowrap;color:var(--fg)}
header .meta{display:flex;align-items:center;gap:var(--s3);font-size:.8rem;color:var(--mut);white-space:nowrap;padding-bottom:.35rem}
header .live{font-weight:500;color:var(--fg2);background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:.1rem .6rem;max-width:18rem;overflow:hidden;text-overflow:ellipsis}
/* ---------- method stepper: one surface, steps separated by chevrons ---------- */
.flow{display:flex;align-items:stretch;background:var(--surface);border:1px solid var(--line);border-radius:var(--r-card);padding:var(--s3) var(--s4);min-width:0}
.flow .step{flex:1 1 0;min-width:0;display:flex;gap:var(--s3);align-items:flex-start;padding:0 var(--s2)}
.flow .sn{flex:none;width:1.6rem;height:1.6rem;border-radius:50%;border:1.5px solid var(--acc);color:var(--acc);font-size:.8rem;font-weight:600;display:flex;align-items:center;justify-content:center;margin-top:.05rem}
.flow .st{min-width:0;display:flex;flex-direction:column;gap:.1rem}
.flow .st b{font-size:1rem;font-weight:600;color:var(--fg);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.flow .sd{font-size:.84rem;color:var(--fg2);line-height:1.35;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.flow .sc2{font-size:.76rem;color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-variant-numeric:tabular-nums}
.flow .arr{flex:none;align-self:center;font-style:normal;color:var(--mut2);font-size:1.1rem;padding:0 var(--s1)}
/* ---------- grid + cards ---------- */
.grid{display:grid;grid-template-columns:minmax(0,29fr) minmax(0,41fr) minmax(0,30fr);gap:var(--s3);min-height:0;min-width:0}
.stack{display:flex;flex-direction:column;gap:var(--s3);min-height:0;min-width:0}
.col{border:1px solid var(--line);border-radius:var(--r-card);min-width:0;min-height:0;display:flex;flex-direction:column;overflow:hidden}
.col.left{background:var(--t-left)}.col.mid{background:var(--t-mid)}.col.hypo{background:var(--t-hyp);flex:none}.col.ins{background:var(--t-ins);flex:1}
.col>h2{margin:0;padding:var(--s3) var(--s4) var(--s2);font-size:1.05rem;font-weight:600;letter-spacing:-.01em;display:flex;align-items:center;gap:var(--s2);flex:none;color:var(--fg)}
.col>h2::before{content:"";width:.5rem;height:.5rem;border-radius:50%;background:var(--d-left);flex:none}
.col.mid>h2::before{background:var(--d-mid)}.col.hypo>h2::before{background:var(--d-hyp)}.col.ins>h2::before{background:var(--d-ins)}
.col>h2 .src{margin-left:auto;font-size:.72rem;font-weight:500;color:var(--mut);border:1px solid var(--line);background:var(--surface);border-radius:var(--r-chip);padding:.05rem .45rem;letter-spacing:0}
.col>h2 button{margin-left:auto;font:inherit;font-size:.75rem;font-weight:500;border:1px solid var(--line);background:var(--surface);color:var(--fg2);border-radius:var(--r-chip);padding:.1rem .5rem;cursor:pointer}
.ibody{padding:0 var(--s4) var(--s4);overflow:hidden;flex:1;min-height:0;display:flex;flex-direction:column;gap:var(--s3)}
.empty,.insbadge{padding:var(--s4);color:var(--mut);font-size:.9rem}
/* ---------- chips ---------- */
.pill,.conf,.ck,.pk,.chip{display:inline-flex;align-items:center;white-space:nowrap;font-size:.72rem;font-weight:500;line-height:1.5;border-radius:var(--r-chip);padding:0 .4rem;border:1px solid var(--neu-line);background:var(--neu-bg);color:var(--neu)}
.pill.r,.ck.bad{background:var(--bad-bg);border-color:var(--bad-line);color:var(--bad)}
.pill.g,.ck.ok,.conf.ok{background:var(--ok-bg);border-color:var(--ok-line);color:var(--ok)}
.chip{background:var(--acc-soft);border-color:#cfdaf6;color:var(--acc)}
/* ---------- left: one run ---------- */
.lbody{flex:1;min-height:0;overflow:hidden;display:flex;flex-direction:column}
.run{padding:0 var(--s4) var(--s4);display:flex;flex-direction:column;gap:var(--s3);flex:1;min-height:0;overflow:hidden}.run[hidden]{display:none}
.verdict{padding-left:var(--s3);border-left:3px solid var(--mut2)}
.verdict.r{border-color:var(--bad)}.verdict.g{border-color:var(--ok)}
.verdict .vh{display:block;font-size:1.4rem;line-height:1.25;font-weight:600;letter-spacing:-.015em;color:var(--neu)}
.verdict.r .vh{color:var(--bad)}.verdict.g .vh{color:var(--ok)}
.verdict .vs{display:block;font-size:.92rem;color:var(--fg2);margin-top:.15rem}
.fault{font-size:.88rem;color:var(--mut)}.fault b{font-size:1.05rem;font-weight:600;color:var(--fg);margin-right:.35rem}.fault small{display:block;font-size:.76rem;color:var(--mut2);margin-top:.05rem}
.facts{display:grid;grid-template-columns:auto 1fr auto 1fr;gap:.2rem var(--s3);font-size:.88rem;padding-top:var(--s2);border-top:1px solid var(--line-soft)}
.facts span{color:var(--mut);white-space:nowrap}.facts b{font-weight:500;color:var(--fg);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.clip.none{display:flex;align-items:center;justify-content:center;color:#9aa1ab;font-size:.85rem;max-height:8rem}
.clip{display:block;width:100%;flex:1 1 auto;min-height:5rem;max-height:30vh;background:#0f1115;border-radius:.375rem;object-fit:contain}
.cap{font-size:.74rem;color:var(--mut);margin-top:calc(-1 * var(--s2))}
.tech{flex:none;display:flex;flex-direction:column;gap:var(--s2)}
.tech table{width:100%;border-collapse:collapse}
.tech table.kv{font-size:.86rem}
.tech table.kv td{padding:.22rem 0;border-bottom:1px solid var(--line-soft);vertical-align:top}
.tech table.kv tr:first-child td{border-top:1px solid var(--line-soft)}
.tech table.kv td:first-child{width:5.6rem;color:var(--mut);padding-right:var(--s3);white-space:nowrap}
.tech .chkh{margin:var(--s1) 0 0;font-size:.84rem;font-weight:600;color:var(--fg)}.tech .chkh .mut{font-weight:400}
/* ---------- gridded tables (scenario list + check table share one style) ---------- */
table.list,.tech table.chk{border-collapse:separate;border-spacing:0;table-layout:fixed;width:100%;background:var(--surface);border:1px solid var(--line);border-radius:.375rem;overflow:hidden}
table.list th,.tech table.chk th{background:#f5f6f8;color:var(--mut);font-size:.76rem;font-weight:500;text-align:left;padding:.4rem .6rem;border-bottom:1px solid var(--line);white-space:nowrap}
table.list td,.tech table.chk td{padding:.4rem .6rem;border-bottom:1px solid var(--line-soft);vertical-align:middle;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
table.list th+th,table.list td+td,.tech table.chk th+th,.tech table.chk td+td{border-left:1px solid var(--line-soft)}
table.list tbody tr:last-child td,.tech table.chk tr:last-child td{border-bottom:0}
.tech table.chk{font-size:.8rem}.tech table.chk td,.tech table.chk th{padding:.26rem .5rem}
.tech table.chk col.c0{width:1.9rem}.tech table.chk col.c1{width:40%}.tech table.chk col.c2{width:28%}
.tech table.chk td.mark{text-align:center;font-weight:600}.mark.ok{color:var(--ok)}.mark.bad{color:var(--bad)}.mark.mut{color:var(--mut2)}
/* ---------- middle: counts + scenario table ---------- */
.counts{display:grid;grid-template-columns:.7fr .7fr 1.25fr 1.25fr 1.1fr 1fr;margin:0 var(--s4);background:#edf2f9;border-radius:.375rem;flex:none}
.counts div{padding:var(--s2) var(--s3);min-width:0;border-left:1px solid #dbe3ef}.counts div:first-child{border-left:0}
.counts b{display:block;font-size:1.45rem;line-height:1.15;font-weight:600;font-variant-numeric:tabular-nums;color:var(--fg)}
.counts b.ok{color:var(--ok)}.counts b.bad{color:var(--bad)}
.counts span{display:block;font-size:.74rem;color:var(--mut);line-height:1.3;margin-top:.1rem}
.listbox{flex:1;min-height:0;overflow:hidden;margin:var(--s3) var(--s4) var(--s4)}
table.list{height:100%;font-size:.9rem}
table.list thead th{height:2.1rem}
table.list .scentd{width:20%}table.list .faulttd{width:26%}table.list .swtd{width:15%}table.list .safetd{width:18%}table.list .cliptd{width:21%}
table.list tbody tr{cursor:pointer}
table.list tbody tr.g td{background:#f2f9f4}table.list tbody tr.r td{background:#fdf4f3}
table.list tbody tr.n td,table.list tbody tr.x td{background:#f7f8f9}
table.list tbody tr:hover td{background-image:linear-gradient(#0000000a,#0000000a)}
table.list tbody tr.sel td{background-image:linear-gradient(#2f5bd312,#2f5bd312)}
table.list tbody tr.sel td:first-child{box-shadow:inset 3px 0 0 var(--acc)}
.sid{font-weight:500;font-variant-numeric:tabular-nums;line-height:1.3}.scen{display:flex;margin-top:.2rem;min-width:0}.scen .pill{max-width:100%;overflow:hidden;text-overflow:ellipsis}
.smeta{font-size:.76rem;color:var(--mut);line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:.1rem}
.more-runs{font-size:.76rem;color:var(--mut);padding:0 var(--s4) var(--s3);margin-top:calc(-1 * var(--s3));flex:none}
/* ---------- hypotheses ---------- */
.hypo .q .lab{display:block;font-size:.74rem;font-weight:500;color:var(--mut);margin-bottom:.15rem;text-transform:none}
.hypo .q .ql{display:block;font-size:1rem;font-weight:500;line-height:1.45;color:var(--fg)}
.hypo ol.hyp{margin:0;padding:0;list-style:none;display:flex;flex-direction:column}
.hypo ol.hyp li{padding:var(--s2) 0;border-top:1px solid #ece6da}
.hypo ol.hyp li:last-child{border-bottom:1px solid #ece6da}
.hypo .pk{background:#f6efe1;border-color:#e8dcc3;color:#8a5a12;margin-bottom:.25rem}
.hypo .ht{display:block;font-size:.95rem;line-height:1.45;color:var(--fg)}
.hypo .hs{display:flex;gap:.3rem;flex-wrap:wrap;margin-top:.35rem}
.hypo .none{font-size:.82rem;color:var(--mut2)}
.hypo .hkey{margin:0;font-size:.9rem;color:var(--fg2)}.hypo .hkey b{color:var(--fg)}
/* ---------- insights ---------- */
.ins .ikey{margin:0;font-size:1.15rem;color:var(--fg2)}.ins .ikey b{font-size:1.35rem;color:var(--fg);font-weight:600;font-variant-numeric:tabular-nums}
.ins ol.itop{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;flex:1;min-height:0;overflow:hidden}
.ins ol.itop li{padding:var(--s4) 0;border-top:1px solid #dde9e2}
.ins .ih{display:flex;align-items:baseline;gap:var(--s2)}
.ins .it{flex:1;min-width:0;font-size:1.06rem;font-weight:500;line-height:1.4;color:var(--fg)}
.ins .conf{flex:none}
.ins ol.itop .fd{font-size:.94rem;color:var(--mut);margin-top:.2rem;line-height:1.45;font-variant-numeric:tabular-nums;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
/* long lists (12+ rows): one line per row */
.listbox.dense table.list{font-size:.84rem}
.listbox.dense table.list td{padding:.12rem .55rem}
.listbox.dense table.list thead th{height:1.9rem}
.listbox.dense .smeta{display:none}
.listbox.dense table.list .scentd{width:26%}.listbox.dense table.list .faulttd{width:22%}.listbox.dense table.list .swtd{width:15%}.listbox.dense table.list .safetd{width:16%}.listbox.dense table.list .cliptd{width:21%}
.listbox.dense td.scentd{display:table-cell}.listbox.dense .sid{display:inline}.listbox.dense .scen{display:inline-flex;margin:0 0 0 .4rem;vertical-align:middle}
@media (max-width:1400px){header h1{font-size:1.9rem}.facts{grid-template-columns:auto 1fr auto 1fr}}
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
  document.querySelectorAll('tr[data-id],.rcard[data-id]').forEach(function(c){c.classList.toggle('sel',c.dataset.id===id)});
  var h=document.getElementById('lefttitle'), b=document.getElementById('back');
  if(h)h.textContent=(id===latest)?'Latest Scenario':'Selected Scenario';
  if(b)b.hidden=(id===latest);
 }
 document.querySelectorAll('tr[data-id],.rcard[data-id]').forEach(function(c){
  function pick(){var id=c.dataset.id;put(id===latest?null:id);show(id);
   if(innerWidth<860){var p=document.getElementById('leftpanel');if(p)p.scrollIntoView()}}
  c.addEventListener('click',pick);
  c.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();pick()}})});
 var b=document.getElementById('back');if(b)b.addEventListener('click',function(){put(null);show(latest)});
 show(get()||latest);
 // keep opened details open across the auto reload
 document.querySelectorAll('details[data-k]').forEach(function(d){
  var k='det:'+d.dataset.k;try{if(sessionStorage.getItem(k)==='1')d.open=true}catch(e){}
  d.addEventListener('toggle',function(){try{sessionStorage.setItem(k,d.open?'1':'')}catch(e){}})});
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


def _ensure_fonts(out_dir: Path):
    """Copy the bundled fonts (Inter for the UI, Pretendard Variable as a fallback for any Korean in the data, both OFL)
    next to campaign.html so the offline page renders the same on every PC."""
    dst = Path(out_dir) / "fonts"
    here = Path(__file__).resolve().parent / "static"
    for name in ("Inter.ttf", "Inter-LICENSE.txt", "PretendardVariable.woff2", "Pretendard-LICENSE.txt"):
        src_f, target = here / name, dst / name
        if src_f.is_file() and not (target.is_file() and target.stat().st_size == src_f.stat().st_size):
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, target)


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
        limit = f" (limit {_f(sr1.get('limit_s'))} s, {_e(sr1.get('limit_source'))})" if sr1.get("applies") else ""
        if sr1.get("applies") and not sr1.get("pass"):
            return (f'<span class="bad">✗ Late {_f(lat)} s</span>',
                    f'<span class="bad">✗ Late switch</span> — handed over to LQR {_f(lat)} s after the fault{limit}')
        return (f'<span class="ok">✓ {_f(lat)} s</span>',
                f'<span class="ok">✓ Switched</span> — handed over to LQR {_f(lat)} s after the fault{limit}')
    if sr1.get("applies"):
        return ('<span class="bad">✗ None</span>', '<span class="bad">✗ No switch</span> — hazardous fault, but MPC kept driving')
    if r.get("note") == "undetected_harmless":
        return ('<span class="mut">– Not needed</span>',
                '<span class="mut">– No switch</span> — harmless fault, no response needed')
    return ('<span class="mut">–</span>', '<span class="mut">– N/A</span>')


def _checks(r, lim: dict) -> list[tuple]:
    """Driving after the switch: (name, ok/None, measured, limit)."""
    sr, m = r.get("sr") or {}, r.get("sr_measures") or {}
    s2, s4, s6 = (sr.get(k) or {} for k in ("SR2", "SR4", "SR6"))
    switched = r.get("switch_time") is not None
    margin = s4.get("margin_m")
    labels = r.get("recovery_labels") or {}
    acc = (labels.get("lat_acc_excess") or {}).get("primary")
    ro = r.get("recovery_outcome")
    rows = [
        ("Steering spike (SR2)", s2.get("pass") if s2.get("applies") else None,
         _f(s2.get("max_rate_dps"), 0, " °/s") if switched else "No switch", f"≤ {_f(lim['sr2'], 0)} °/s"),
        ("Lane margin (SR4)", s4.get("pass"),
         "–" if margin is None else (f"Margin {_f(margin, 2, ' m')}" if margin >= 0 else f"Over line {_f(-margin, 2, ' m')}"),
         "Margin ≥ 0 m"),
        ("Collision", (not s4.get("collision")) if s4 else None,
         (f"{_f(m.get('SR4_collision_t'), 1, ' s')} {m.get('SR4_collision_other') or ''}" if s4.get("collision") else "None"),
         "None"),
        ("Driving safety (v2)", None if ro in (None, "no_switch", "simulation_invalid") else ro == "safe_recovery",
         RECOVERY_KO.get(ro, ro or "–") + (f" · score {_f(r.get('lane_recovery_robustness'))}" if switched else ""),
         f"{_f(lim['rj'].get('e_settle_m', 0.3), 1)} m · {_f(lim['rj'].get('psi_settle_rad', 0.1), 1)} rad · "
         f"{_f(lim['rj'].get('settle_hold_s', 1.0), 0)} s hold in {_f(lim['win'], 0)} s"),
        ("Lateral accel. (P-02)", None if acc is None else not acc,
         _f(r.get("lat_acc_excess_mps2"), 2, " m/s²"), f"≤ {_f(lim['acc'], 2)} m/s²"),
        ("Steering limits (SR6)", s6.get("pass") if s6.get("applies") else None,
         f"{_f(s6.get('max_steer_deg'), 0, '°')} · {_f(s6.get('max_rate_dps'), 0, ' °/s')}" if s6.get("applies") else "No record",
         f"Vehicle · ≤ {_f(lim['sr6'], 0)} °/s"),
    ]
    if ro == "no_switch":
        # fallback never engaged: no green ✓ (nothing was graded after a switch); a real lane departure (SR4) or
        # collision stays ✗, every other row is shown muted with its observed value.
        rows = [(n, False if (i in (1, 2) and ok is False) else None, meas, lm) for i, (n, ok, meas, lm) in enumerate(rows)]
    return rows


def _stability_short(r) -> str:
    if _key(r) in ("invalid", "no_trigger"):
        return '<span class="mut">–</span>'
    names = {"SR2": "Steering spike", "SR4": "Lane", "SR5": "Driving safety", "SR6": "Steering limit"}
    bad = [names[k] for k in (r.get("failed_requirements") or []) if k in names]
    s4 = (r.get("sr") or {}).get("SR4") or {}
    if s4.get("collision"):
        bad = [b for b in bad if b != "Lane"] + ["Collision"]
    if r.get("recovery_outcome") == "no_switch":
        # fallback never engaged: nothing to grade after a switch. Only real collision / lane departure count.
        bad = [b for b in bad if b in ("Lane", "Collision")]
        if s4.get("pass") is False and "Lane" not in bad and "Collision" not in bad:
            bad.append("Lane")
        return (f'<span class="bad">✗ {_e(", ".join(bad))}</span>' if bad
                else '<span class="mut">No switch · held</span>')
    return f'<span class="bad">✗ {_e(", ".join(bad))}</span>' if bad else '<span class="ok">✓</span>'



# ------------------------------------------------------------------ short wording (first screen)
def _num(x):
    try:
        return None if x is None or isinstance(x, bool) else float(x)
    except (TypeError, ValueError):
        return None


def _sec(x) -> str:
    v = _num(x)
    if v is None:
        return UNKNOWN
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{'0' if s in ('-0', '') else s} s"


def _deg(rad) -> str | None:
    v = _num(rad)
    return None if v is None else f"{abs(v) * 57.29577951308232:.1f}°"


def _bucket(r) -> str:
    return KEY_BUCKET.get(_key(r), "na")


def fault_plain(r) -> tuple[str, str, str]:
    """(short fault name, what it does, strength) from parameters.fault; unknown parts -> Unknown."""
    fp = r.get("fault_plan") or {}
    mode = fp.get("mode") or fp.get("fault_type")
    name, what = FAULT_KO.get(mode, (str(mode) if mode else UNKNOWN, ""))
    p = fp.get("params") or {}
    parts = []
    if mode == "saturate":
        if _num(p.get("magnitude")) is not None:
            parts.append(f"{_num(p['magnitude']) * 100:.0f}% of max angle")
        if _num(p.get("duration_s")) is not None:
            parts.append(f"for {_sec(p['duration_s'])}")
    elif mode == "bias" and _deg(p.get("magnitude_rad")):
        parts.append(f"{_deg(p['magnitude_rad'])}")
    elif mode == "drift" and _deg(p.get("rate_rad_s")):
        parts.append(f"+{_deg(p['rate_rad_s'])}/s")
    elif mode == "intermittent_bias":
        if _deg(p.get("magnitude_rad")):
            parts.append(f"{_deg(p['magnitude_rad'])}")
        if _num(p.get("on_s")) is not None and _num(p.get("off_s")) is not None:
            parts.append(f"{_sec(p['on_s'])} on/{_sec(p['off_s'])} off")
    elif mode in ("stale", "solver_fail") and _num(p.get("duration_s")) is not None:
        parts.append(f"for {_sec(p['duration_s'])}")
    elif mode == "output_delay" and _num(p.get("delay_ticks")) is not None:
        parts.append(f"{_num(p['delay_ticks']):.0f}-step delay")
    return name, what, " · ".join(parts) or UNKNOWN


def road_plain(r) -> str:
    env = r.get("environment") or {}
    route = str(env.get("route") or env.get("town") or "")
    if not route:
        return UNKNOWN
    if route.startswith("bench2drive:"):
        return f"Road #{route.split(':', 1)[1]}"
    if route.startswith("town10"):
        return "Town10" + (" long route" if "long" in route else "")
    return route


def times(r) -> dict:
    """Seconds after the fault started (from the run record only): notice (switch), left (first SR4 lane violation),
    crash (SR4 collision). None when not recorded / did not happen."""
    onset = _num(r.get("actual_fault_onset"))
    if onset is None:
        onset = _num(r.get("planned_fault_onset"))
    s4 = (r.get("sr") or {}).get("SR4") or {}
    m = r.get("sr_measures") or {}
    out = {"onset": onset, "notice": None, "left": None, "crash": None, "crashed": bool(s4.get("collision"))}
    if r.get("switch_time") is not None:
        lat = _num(r.get("detection_latency_s"))
        if lat is None and onset is not None and _num(r.get("switch_time")) is not None:
            lat = _num(r["switch_time"]) - onset
        out["notice"] = lat
    if onset is not None and _num(s4.get("first_violation_t")) is not None:
        out["left"] = _num(s4["first_violation_t"]) - onset
    if out["crashed"] and onset is not None and _num(m.get("SR4_collision_t")) is not None:
        out["crash"] = _num(m["SR4_collision_t"]) - onset
    return out


def story(r) -> tuple[str, str]:
    """(headline, terse one-line statement) from recorded values only."""
    key, b = _key(r), _bucket(r)
    if key == "invalid":
        return "Not graded (invalid, not a failure)", "Recording or baseline conditions not met — excluded from grading"
    if key == "no_trigger":
        return "Not graded (fault not triggered)", "Steering command unchanged after injection — excluded from grading"
    t = times(r)
    switched = r.get("switch_time") is not None
    n, left = t["notice"], t["left"]
    ev = []
    if left is not None:
        ev.append((left, "Lane departure" if left >= 0 else "Out of lane before the fault"))
    if switched:
        late = left is not None and n is not None and left < n
        ev.append((n if n is not None else 9e9, ("Detected and switched immediately" if n is not None and abs(n) < 1e-6
                                                 else "detected and switched") + (" late" if late else "")))
    if t["crashed"]:
        ev.append((t["crash"] if t["crash"] is not None else 9e9, "collision"))
    ev.sort(key=lambda e: e[0])
    segs = []
    for i, (tt, label) in enumerate(ev):
        if label.startswith("Detected and switched immediately") or label.startswith("Out of lane") or tt >= 9e9:
            segs.append(label[:1].upper() + label[1:])
        elif not segs:
            segs.append(f"{label[:1].upper() + label[1:]} {_sec(tt)} after the fault")
        else:
            segs.append(f"{label} at {_sec(tt)}")
    if not switched:
        segs.append("Not detected")
    if left is None and not t["crashed"]:
        segs.append("Lane kept" + (" (harmless)" if b == "harmless" else " (rule requires detection)" if b == "danger" and not switched else ""))
    ro = r.get("recovery_outcome")
    if switched and not t["crashed"]:
        extra = {"recovered_after_departure": "Back in lane", "unrecovered": "Not recovered", "unsettled": "Not settled"}.get(ro)
        if extra:
            segs.append(extra)
    if key == "F2_switch_transient":
        s2 = (r.get("sr") or {}).get("SR2") or {}
        rate, lim = _num(s2.get("max_rate_dps")), _num(s2.get("limit_dps"))
        segs.append(f"Steering {rate:.0f}°/s at handover (limit ≤ {lim:.0f}°/s)" if rate is not None and lim is not None
                    else "Steering jump at handover")
    head = {"pass": "Pass", "harmless": "Undetected · harmless",
            "F1_no_or_late_switch": "Controller handover failed/late",
            "F2_switch_transient": "Unstable at handover",
            "F3_not_recovered": "Unstable after handover"}.get(key)
    if head is None:
        head = {"safe": "Pass", "harmless": "Undetected · harmless"}.get(b, "Not graded")
    return head, " · ".join(segs)


def lane_plain(r) -> str:
    s4 = (r.get("sr") or {}).get("SR4") or {}
    m = _num(s4.get("margin_m"))
    if s4.get("collision"):
        return "Departed, then collided"
    if m is None:
        return UNKNOWN
    return f"Departed by up to {-m:.2f} m" if m < 0 else f"Kept (min. margin {m:.2f} m)"


# ------------------------------------------------------------------ left: one run
def _rel(path: str | None, out_dir: Path) -> str | None:
    """Path of a clip relative to the screen folder. Result folders get moved (results/<x> -> result/result_N), so
    when the recorded path is gone, fall back to <campaign>/clips/<run>/<file> next to the screen folder."""
    if not path:
        return None
    p = str(path)
    if not os.path.exists(p):
        bits = [b for b in p.replace("\\", "/").split("/") if b]
        if len(bits) >= 2:
            alt = Path(out_dir).parent / "clips" / bits[-2] / bits[-1]
            if alt.is_file():
                p = str(alt)
    try:
        return os.path.relpath(p, out_dir).replace(os.sep, "/")
    except ValueError:
        return p


def _clip_html(c: dict | None, out_dir: Path) -> str:
    clip = (c or {}).get("clip") or {}
    if not clip.get("video"):
        return ""
    when = (f" ({_f(clip.get('start_s'), 1)}–{_f(clip.get('end_s'), 1)} s)"
            if clip.get("start_s") is not None and clip.get("end_s") is not None else "")
    rel = _rel(clip["video"], out_dir)
    if not (os.path.exists(str(clip["video"])) or (Path(out_dir) / rel).is_file()):
        # recorded clip is gone (moved / not copied): say so instead of an empty black player
        return f'<div class="clip none">No video file</div><div class="cap">Recorded path: {_e(clip["video"])}</div>'
    return (f'<video class="clip" src="{_e(rel)}" controls muted playsinline preload="metadata"></video>'
            f'<div class="cap">Video: around the fault{_e(when)}</div>')


def _commentary_html(r: dict, c: dict | None, out_dir: Path, active: bool = False) -> str:
    """Our explainer's mechanism text + its verification against monitor timestamps (the clip is shown separately)."""
    if not c:
        return '<div class="ai wait">No video commentary (only failure or near-boundary runs)</div>' if active else ""
    out = []
    ex = c.get("explanation") or {}
    v = c.get("verification") or {}
    clip = c.get("clip") or {}
    if clip.get("video"):
        out.append(f'<div class="cap">Clip {_f(clip.get("start_s"), 1)}–{_f(clip.get("end_s"), 1)} s · '
                   f'{_e(clip.get("source") or "")} · {_e(clip.get("alignment") or "")}</div>')
    if ex:
        tags = ", ".join(ex.get("mechanism_tags") or [])
        ok = v.get("accepted")
        failed = [k for k, x in (v.get("checks") or {}).items() if not x.get("pass")]
        out.append(f'<div class="ai"><span class="src">AI video commentary · {_e(c.get("model") or "")} · does not change the verdict</span>'
                   f'<div class="sum">{_e(ex.get("mechanism") or "")}</div>'
                   f'<div class="why">Tags: {_e(tags or "–")}</div>'
                   f'<div class="agree{"" if ok else " dis"}">{"Verified (matches monitor times within tolerance)" if ok else "Verification failed: " + _e(", ".join(failed) or v.get("reason") or "") + " — does not fully match the record; reference only"}</div></div>')
    elif v.get("reason"):
        out.append(f'<div class="ai wait">{_e(v["reason"])}</div>')
    return "".join(out)


SELECT_KO = {"cold_start_nemotron_a": "Cold start · proposed by Nemotron A", "cold_start_space_filling": "Cold start · space filling",
             "acquisition": "Picked by GP (near boundary)", "random_space_filling": "Random sampling"}


def _tech_html(r: dict, lim: dict, out_dir: Path, commentary: dict | None, active: bool) -> str:
    """Everything technical for one run (SR ids, F codes, recovery v2, lrr, selector, raw params)."""
    fp = r.get("fault_plan") or {}
    key = _key(r)
    text, _, sentence = OUTCOME.get(key, (key, "mut", ""))
    final = r.get("final_safe_recovery")
    verdict = {True: "true", False: "false", None: "not graded"}[final]
    sel = str(r.get("selector") or "")
    sel_txt = SELECT_KO.get(sel) or ("Replacement (earlier candidate skipped)" if sel.startswith("replacement_for_") else sel or "–")
    rows = [("Run", f'{_e(r.get("run_id") or "–")} · {_e(r.get("scenario_id") or "–")}'
             + (f' · round {_e(r.get("round"))}' if r.get("round") is not None else "")),
            ("SR result", f'<b>{_e(text)}</b> — {_e(sentence)}'),
            ("Driving safety (recovery v2)", f'{_e(RECOVERY_KO.get(r.get("recovery_outcome"), r.get("recovery_outcome") or "–"))} · '
                                   f'final_safe_recovery = {_e(verdict)} · {_e(r.get("outcome_category") or "–")}'),
            ("Rules not met", _e(", ".join(r.get("failed_requirements") or []) or "None")),
            ("Selection", f'{_e(sel_txt)} <span class="mut">({_e(sel or "–")})</span>')]
    if key == "invalid":
        rows.append(("Invalid reason", _e(r.get("invalid_reason") or UNKNOWN)))
    params = fp.get("params") or {}
    pt = ", ".join(f"{k} {_f(v, 3) if isinstance(v, (int, float)) and not isinstance(v, bool) else _e(v)}" for k, v in params.items())
    rows += [("Fault code", f'{_e(fp.get("fault_type") or "–")} {_e(fp.get("mode") or "")}{f" · {_e(pt)}" if pt else ""}'),
             ("Fault onset", f'{_e(ONSET_KO.get(fp.get("onset_event"), fp.get("onset_event") or "–"))} {_f(fp.get("onset_offset_s"), 2, " s")} · '
                       f'{_f(r.get("actual_fault_onset") or r.get("planned_fault_onset"), 2, " s")}'),
             ("Handover", f'{_switch(r)[1]} · trigger {_e(r.get("trigger_source") or "–")}')]
    if r.get("switch_time") is not None:
        rows.append(("lrr", _f(r.get("lane_recovery_robustness"))))
    kv = "".join(f"<tr><td>{_e(k)}</td><td>{v}</td></tr>" for k, v in rows)
    out = [f'<table class="kv">{kv}</table>']
    if key != "invalid":
        chk = "".join(
            f'<tr><td class="mark {"ok" if ok else "bad" if ok is False else "mut"}">{"✓" if ok else "✗" if ok is False else "–"}</td>'
            f'<td>{_e(name)}</td><td class="num">{_e(meas)}</td><td class="mut">{_e(lm)}</td></tr>'
            for name, ok, meas, lm in _checks(r, lim))
        held = ' <span class="mut" style="font-weight:400">· No switch · held (whole-run observations below)</span>' \
            if r.get("recovery_outcome") == "no_switch" else ""
        out.append(f'<div class="blk"><h3>Driving stability after handover{held}</h3><table>{chk}</table></div>')
    ai = _commentary_html(r, commentary, out_dir, active and key not in ("invalid", "no_trigger"))
    if ai:
        out.append(f'<div class="blk"><h3>AI video commentary</h3>{ai}</div>')
    return "".join(out)


def _tech_simple(r: dict, lim: dict) -> str:
    """Always-visible technical block: summary lines + the full post-switch check table (mark | Item | Measured | Threshold)."""
    key = _key(r)
    text, _, sentence = OUTCOME.get(key, (key, "mut", ""))
    rec = RECOVERY_KO.get(r.get("recovery_outcome"), r.get("recovery_outcome") or "–")
    sw_short, _ = _switch(r)
    fp = r.get("fault_plan") or {}
    params = fp.get("params") or {}
    pt = ", ".join(f"{k} {_f(v, 3) if isinstance(v, (int, float)) and not isinstance(v, bool) else v}"
                   for k, v in params.items())
    name = FAULT_KO.get(fp.get("mode"), (fp.get("mode") or "–",))[0]
    fault = f'{_e(name)} <span class="mut">{_e(fp.get("fault_type") or "")}{" · " + _e(pt) if pt else ""}</span>'
    failed = ", ".join(r.get("failed_requirements") or []) or "None"
    rows = [("Verdict", f"{_e(text)} <span class='mut'>— {_e(sentence)}</span>"),
            ("Driving safety", _e(rec)), ("Handover", sw_short), ("Fault", fault), ("Rules not met", _e(failed))]
    if key == "invalid":
        rows.append(("Invalid reason", _e(r.get("invalid_reason") or UNKNOWN)))
    kv = "".join(f"<tr><td>{_e(k)}</td><td>{v}</td></tr>" for k, v in rows)
    out = f'<table class="kv">{kv}</table>'
    if key not in ("invalid", "no_trigger"):
        chk = "".join(
            f'<tr><td class="mark {"ok" if ok else "bad" if ok is False else "mut"}">{"✓" if ok else "✗" if ok is False else "–"}</td>'
            f'<td title="{_e(n)}">{_e(n)}</td><td class="num" title="{_e(meas)}">{_e(meas)}</td>'
            f'<td class="mut" title="{_e(lm)}">{_e(lm)}</td></tr>'
            for n, ok, meas, lm in _checks(r, lim))
        held = (' <span class="mut">· No switch · held (whole-run observations below)</span>'
                if r.get("recovery_outcome") == "no_switch" else "")
        out += (f'<p class="chkh">Driving stability after handover{held}</p>'
                f'<table class="chk"><colgroup><col class="c0"><col class="c1"><col class="c2"><col class="c3"></colgroup>'
                f'<tr><th></th><th>Item</th><th>Measured</th><th>Threshold</th></tr>{chk}</table>')
    return f'<div class="tech">{out}</div>'

def render_run_panel(r: dict, hidden: bool, lim: dict, out_dir: Path, commentary: dict | None = None,
                     active: bool = False) -> str:
    b = _bucket(r)
    head, line = story(r)
    name, what, strength = fault_plain(r)
    env = r.get("environment") or {}
    fp = r.get("fault_plan") or {}
    speed = _num(env.get("max_speed_kmh"))
    fric = _num(env.get("tire_friction_scale"))
    facts = [("Strength", strength), ("Injected at", WHEN_KO.get(fp.get("onset_event"), fp.get("onset_event") or UNKNOWN)),
             ("Road", road_plain(r)), ("Max speed", f"{speed:.0f} km/h" if speed is not None else UNKNOWN),
             ("Tire grip", f"{fric * 100:.0f}%" if fric is not None else UNKNOWN)]
    rid = _rid(r)
    return (f'<div class="run" id="run-{_e(rid)}"{" hidden" if hidden else ""}>'
            f'<div class="verdict {BUCKET[b][2]}"><span class="vh">{_e(head)}</span><span class="vs">{_e(line)}</span></div>'
            f'<div class="fault"><b>{_e(name)}</b> <span class="mut">{_e(what)}</span> <small>{_e(r.get("run_id") or "")}</small></div>'
            '<div class="facts">' + "".join(f"<span>{_e(k)}</span><b>{_e(v)}</b>" for k, v in facts) + "</div>"
            + _clip_html(commentary, out_dir)
            + _tech_simple(r, lim) + "</div>")


def render_run_card(r: dict, no: int, commentary: dict | None = None) -> str:
    """One table row: Scenario | Fault | Handover | Driving Safety | Video Commentary."""
    key, b = _key(r), _bucket(r)
    name, _, strength = fault_plain(r)
    fp = r.get("fault_plan") or {}
    env = r.get("environment") or {}
    when = WHEN_KO.get(fp.get("onset_event"), fp.get("onset_event") or "")
    speed = _num(env.get("max_speed_kmh"))
    road = road_plain(r)
    rid = _rid(r)
    # Scenario: run id + small SR chip (road and speed go under the fault)
    # short chip text (the full verdict is in the left panel and the row tooltip)
    pill = {"pass": "Pass", "harmless": "Harmless", "F1_no_or_late_switch": "Failed/late",
            "F2_switch_transient": "Unstable at switch", "F3_not_recovered": "Unstable after",
            "invalid": "Not graded", "no_trigger": "Not graded"}.get(key, {"safe": "Pass", "danger": "Failed/late",
            "harmless": "Harmless", "na": "Not graded"}[b])
    meta = " · ".join(x for x in (road, f"{speed:.0f} km/h" if speed is not None else None) if x)
    sid = re.sub(r"^(R\d+-\d+)-.*$", r"\1", rid)
    scen = (f'<div class="sid">{_e(sid)}</div>'
            f'<div class="scen"><span class="pill {BUCKET[b][2]}">{_e(pill)}</span></div>')
    # Fault: professional name + optional onset
    fault = _e(name)
    sub = " · ".join(x for x in (when, meta) if x)
    if sub:
        fault += f'<div class="smeta">{_e(sub)}</div>'
    # Handover
    lat = r.get("detection_latency_s")
    if r.get("switch_time") is not None:
        if lat is None and r.get("actual_fault_onset") is not None:
            lat = r["switch_time"] - r["actual_fault_onset"]
        sr1 = (r.get("sr") or {}).get("SR1") or {}
        late = sr1.get("applies") and not sr1.get("pass")
        if late:
            sw = '<span class="bad">Late' + (f' t={_f(lat)}s' if lat is not None else "") + '</span>'
        else:
            sw = '<span class="ok">✓' + (f' t={_f(lat)}s' if lat is not None else " Switched") + '</span>'
    else:
        sw = '<span class="mut">No switch</span>' if key not in ("invalid", "no_trigger") else '<span class="mut">–</span>'
    # Driving Safety
    rec = RECOVERY_KO.get(r.get("recovery_outcome"), r.get("recovery_outcome") or "–")
    if r.get("recovery_outcome") == "no_switch":
        rec = "Held"
    elif r.get("recovery_outcome") == "recovered_after_departure":
        rec = "Back in lane"
    if key in ("invalid", "no_trigger"):
        safe = f'<span class="mut">{_e(rec if rec != "–" else "–")}</span>'
    elif r.get("recovery_outcome") in ("unrecovered", "unsettled"):
        safe = f'<span class="bad">{_e(rec)}</span>'
    elif r.get("recovery_outcome") in ("safe_recovery", "recovered_after_departure"):
        safe = f'<span class="ok">{_e(rec)}</span>'
    else:
        safe = f'<span class="mut">{_e(rec)}</span>'
    # Video Commentary
    c = commentary or {}
    has_clip = bool((c.get("clip") or {}).get("video"))
    if c.get("explanation"):
        ok = (c.get("verification") or {}).get("accepted")
        expl = '<span class="ok">Verified ✓</span>' if ok else '<span class="bad">Rejected ✗</span>'
    else:
        expl = '<span class="mut">–</span>'
    if has_clip:
        expl += ' <span class="chip">Video</span>'
    title = f"{rid} · {OUTCOME.get(key, (pill,))[0]} · {name} · {rec}"
    return (f'<tr class="{BUCKET[b][2]}" data-id="{_e(rid)}" tabindex="0" role="button" title="{_e(title)}">'
            f'<td class="scentd">{scen}</td>'
            f'<td class="faulttd">{fault}</td>'
            f'<td class="swtd">{sw}</td>'
            f'<td class="safetd">{safe}</td>'
            f'<td class="cliptd">{expl}</td></tr>')



def _ai_cell(c: dict | None) -> str:
    if not c or not c.get("explanation"):
        return '<span class="mut">–</span>'
    return '<span class="ok">✓</span>' if (c.get("verification") or {}).get("accepted") else '<span class="bad">✗</span>'


CONF_KO = {"high": ("High confidence", "ok"), "medium": ("Medium confidence", "mut"), "low": ("Low confidence", "mut")}


def load_insights(out_dir: Path) -> dict | None:
    """<results>/insights/insights.json (engine/insights.py, 09 explain.py) next to the screen folder."""
    p = Path(out_dir).parent / "insights" / "insights.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (OSError, json.JSONDecodeError):
        return None


# ---- readable words for the AI insight text (Cosmos writes metric names and fault codes; facts and numbers kept) ----
# longest first so "metrics.lateral_error_rmse_m" wins over "lateral_error_rmse_m"
TERM_KO = {
    "metrics.": "", "fault_plan.": "",
    "lateral_error_rmse_m": "lateral error RMS", "pre_switch_max_lateral_error_m": "max lateral error before the switch",
    "margin_delta_m": "lane margin change vs. baseline", "detection_latency_s": "detection latency",
    "lane_recovery_robustness": "lane recovery score", "lat_acc_excess_mps2": "excess lateral acceleration",
    "lat_jerk_excess_mps3": "excess lateral jerk", "switch_count": "switch count",
    "ftti_s": "fault tolerant time interval", "recovery_success": "recovery success rate",
    "max_steering_rate_dps": "max steering rate", "command_jump_rad": "command jump",
    "post_switch_saturation_ratio": "post-switch saturation ratio", "tire_friction_scale": "tire grip",
    "deadline_miss_rate": "deadline miss rate", "onset_event": "fault onset", "monitor_outcome": "verdict",
    "F1_no_or_late_switch": "Controller handover failed/late", "F2_switch_transient": "Unstable at handover",
    "F3_not_recovered": "Unstable after handover", "duration_s": "duration", "magnitude_rad": "bias magnitude",
    "delay_ticks": "delay steps", "on_s": "on time", "off_s": "off time", "rate_rad_s": "bias growth rate",
    "intermittent_bias": "Intermittent Bias", "output_delay": "Output Delay",
    "solver_fail": "Solver Failure", "saturate": "Steering Saturation", "stale": "Stale Command", "drift": "Ramp Bias / Drift", "bias": "Steering Bias",
    "curve_entry": "curve entry", "curve_apex": "curve apex", "curve_exit": "curve exit", "straight": "straight",
    "pass": "Pass", "F1": "failure", "F2": "failure", "F3": "failure",
}


def plain_text(s) -> str:
    """Swap technical identifiers in model text for readable English; numbers, run ids and facts stay as written."""
    if not s:
        return ""
    s = str(s)
    keys = sorted(TERM_KO, key=len, reverse=True)
    pat = re.compile("|".join(re.escape(k) for k in keys))
    out, i = [], 0
    for m in pat.finditer(s):
        k = m.group(0)
        # don't touch words inside run ids / longer identifiers (R00-0002-stale, sr_bias_x)
        before = s[m.start() - 1] if m.start() else " "
        after = s[m.end()] if m.end() < len(s) else " "
        if k[0].isascii() and k[0].isalnum() and (before.isalnum() and before.isascii() or before in "-_"):
            continue
        if k[-1].isascii() and k[-1].isalnum() and (after.isascii() and after.isalnum() or after in "_"):
            continue
        out.append(s[i:m.start()] + TERM_KO[k])
        i = m.end()
    out.append(s[i:])
    return "".join(out)


METRIC_KO = {"lateral_error_rmse_m": "Lateral error RMS", "margin_delta_m": "Lane margin change",
             "pre_switch_max_lateral_error_m": "Pre-switch max error", "detection_latency_s": "Detection latency",
             "lat_acc_excess_mps2": "Lat. accel. excess", "lat_jerk_excess_mps3": "Lat. jerk excess",
             "max_steering_rate_dps": "Max steering rate", "lane_recovery_robustness": "Lane recovery score",
             "post_switch_saturation_ratio": "Post-switch saturation", "command_jump_rad": "Command jump",
             "ftti_s": "Fault tolerant time", "switch_count": "Switch count", "recovery_success": "Recovery success",
             "tire_friction_scale": "Tire grip", "deadline_miss_rate": "Deadline miss rate"}
UNIT = (("_mps2", " m/s²"), ("_mps3", " m/s³"), ("_dps", " °/s"), ("_rad", " rad"), ("_m", " m"), ("_s", " s"))
SHORT_OUT = {"pass": "Pass", "harmless": "Pass", "F1_no_or_late_switch": "Controller handover failed/late",
             "F2_switch_transient": "Unstable at handover", "F3_not_recovered": "Unstable after handover",
             "invalid": "Not graded", "no_trigger": "Not graded"}


def _subject(title: str):
    """Fault mode or onset phase named in a generated insight title (raw keys), first occurrence."""
    best = None
    for k in sorted(list(FAULT_KO) + list(WHEN_KO), key=len, reverse=True):
        mt = re.search(rf"(?<![A-Za-z_]){re.escape(k)}(?![A-Za-z_])", title or "")
        if mt and (best is None or mt.start() < best[0]):
            best = (mt.start(), k)
    if not best:
        return None
    k = best[1]
    return FAULT_KO[k][0] if k in FAULT_KO else f"{WHEN_KO[k]} faults"


def _num_txt(v: float) -> str:
    return (f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}").replace("-", "−")


def insight_view(it: dict, outcomes: dict) -> tuple[str, str]:
    """(headline, evidence) for one generated insight, from its structured evidence; falls back to the text."""
    ev = [e for e in it.get("evidence") or [] if isinstance(e, dict)]
    if not ev:
        return plain_text(it.get("title")), plain_text(it.get("finding"))
    runs = list(dict.fromkeys(str(e.get("run_id")) for e in ev if e.get("run_id")))
    res = {}
    for rid in runs:
        o = outcomes.get(rid)
        if o is None:
            o = next((e.get("value") for e in ev if e.get("run_id") == rid and e.get("metric") == "monitor_outcome"), None)
        if o is not None:
            res[rid] = SHORT_OUT.get(o, o)
    verdict = ""
    if res and len(res) == len(runs):
        kinds = list(dict.fromkeys(res.values()))
        if len(kinds) == 1:
            verdict = kinds[0] if len(runs) == 1 else f"{kinds[0]} in {len(runs)}/{len(runs)} runs"
        else:
            verdict = " · ".join(f"{k} {sum(1 for v in res.values() if v == k)}" for k in kinds)
    subj = _subject(it.get("title") or "")
    head = f"{subj}: {verdict}" if subj and verdict else plain_text(it.get("title"))
    vals = {}
    for e in ev:
        name = str(e.get("metric") or "").split(".")[-1]
        v = e.get("value")
        if name in METRIC_KO and isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.setdefault(name, []).append(float(v))
    parts = []
    for name, vs in list(vals.items())[:3]:
        unit = next((u for suf, u in UNIT if name.endswith(suf)), "")
        avg = sum(vs) / len(vs)
        parts.append(f"{METRIC_KO[name]} {'avg ' if len(vs) > 1 else ''}{_num_txt(avg)}{unit}")
    ids = [re.sub(r"^(R\d+-\d+).*$", r"\1", r) for r in runs[:4]]
    parts.append(", ".join(ids) + (f" +{len(runs) - 4} more" if len(runs) > 4 else ""))
    return head, " · ".join(p for p in parts if p)


def render_insights(ins: dict | None, out_dir: Path = Path("."), records: list | None = None) -> str:
    """Insights panel (Cosmos): failure rate + top 3 findings as headline / evidence numbers, always open."""
    if not ins or ins.get("status") != "complete":
        why = "generated after the campaign ends" if not ins else f"generation failed: {ins.get('error') or ins.get('status')}"
        return (f'<section class="col ins" id="insbadge"><h2>Insights <span class="src">by Cosmos</span></h2>'
                f'<div class="insbadge" title="{_e(why)}">Not available yet ({_e(why[:60])})</div></section>')
    st = ins.get("statistics") or {}
    rate, graded, fails = st.get("failure_rate"), st.get("graded_triggered_runs"), st.get("failure_count")
    key = (f'Failure rate <b>{_f(rate * 100, 1, "%")}</b> <span class="mut">({_e(fails)} of {_e(graded)} graded runs failed)</span>'
           if rate is not None else "No graded runs")
    outcomes = {str(r.get("run_id")): _key(r) for r in records or []}
    items = sorted(ins.get("insights") or [], key=lambda it: {"high": 0, "medium": 1, "low": 2}.get(it.get("confidence"), 3))
    lis = ""
    for it in items[:3]:
        head, evid = insight_view(it, outcomes)
        conf, cls = CONF_KO.get(it.get("confidence"), (it.get("confidence") or "–", "mut"))
        lis += (f'<li><div class="ih"><span class="it">{_e(head)}</span> <span class="conf {cls}">{_e(conf)}</span></div> '
                f'<div class="fd">{_e(evid)}</div></li>\n')
    return (f'<section class="col ins" id="ins"><h2>Insights <span class="src">by Cosmos</span></h2>'
            f'<div class="ibody"><p class="ikey">{key}</p><ol class="itop">{lis}</ol></div></section>')

SIDE_KO = {"recover": "Lane kept", "fail": "Fail", "boundary": "Boundary"}


def load_method(out_dir: Path) -> dict:
    """How this campaign worked, from the files run.py already writes next to screen/ (no new outputs needed).

    code: 01_profile_summary.json (code observations Nemotron A reads) + round_*/05b_answers.json (files Nemotron B cited)
    hypotheses: Nemotron A proposals in round_*/03_04_pool.json (expected_side + expected_mechanism)
    check: run_records.jsonl outcomes vs expected_side, evaluation_metrics.json GP leave-one-out sign accuracy."""
    camp = Path(out_dir).parent

    def rj(path, default=None):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default
    prof = rj(camp / "01_profile_summary.json", {}) or {}
    rounds = sorted(camp.glob("round_*"))
    b_files, b_cands, a_props = set(), 0, {}
    for rd in rounds:
        for key, row in (rj(rd / "05b_answers.json", {}) or {}).items():
            b_cands += not str(key).startswith("__")      # __static__ = once-per-campaign questions (integration3 #6)
            for ans in (row.get("answers") or {}).values():
                for ev in ans.get("evidence") or []:
                    if ev.get("file"):
                        b_files.add(ev["file"])
        for c in rj(rd / "03_04_pool.json", []) or []:
            if str(c.get("source", "")).startswith("nemotron_a"):
                a_props.setdefault(c["scenario_id"], c)
    recs = _read_jsonl(camp / "run_records.jsonl")
    tests = []
    for r in recs:
        c = a_props.get(r.get("scenario_id"))
        kind = "A original"
        if c is None:
            c, kind = a_props.get(r.get("seed_scenario_id")), "A variant"
        if c is None:
            continue
        exp = c.get("expected_side")
        lrr = r.get("lane_recovery_robustness")
        if r.get("recovery_outcome") == "no_switch" or r.get("final_safe_recovery") is None:
            actual, hit = "No switch", None                  # S1 (LQR recovery) not exercised
        else:
            actual = "Lane kept" if r.get("final_safe_recovery") else "Fail"
            if exp == "boundary":
                hit = lrr is not None and abs(lrr) < 0.2
            else:
                hit = (exp == "recover") == bool(r.get("final_safe_recovery"))
        tests.append({"run_id": r.get("run_id"), "kind": kind, "expected": SIDE_KO.get(exp, exp or "–"), "actual": actual,
                      "hit": hit, "lrr": lrr, "why": c.get("expected_mechanism"),
                      "evidence": list(c.get("evidence_ids") or [])})
    gp = (rj(camp / "evaluation_metrics.json", {}) or {}).get("gp_boundary_sign_accuracy") or {}
    return {"n_obs": prof.get("observations"), "n_verified": prof.get("verified"), "b_files": sorted(b_files),
            "b_cands": b_cands, "n_props": len(a_props),
            "sides": {SIDE_KO.get(k, k): sum(1 for c in a_props.values() if c.get("expected_side") == k)
                      for k in ("recover", "fail", "boundary")},
            "tests": tests, "gp": gp}


def render_method_details(m: dict) -> str:
    """Optional folded detail under the hypotheses card: how each step ran, and every A prediction vs the run."""
    t = m["tests"]
    gp = m["gp"]
    gp_txt = (f'GP leave-one-out (LOO) boundary sign accuracy {_f(gp["accuracy"] * 100, 0, "%")} ({_e(gp.get("n"))} runs)'
              if gp.get("accuracy") is not None else "GP accuracy is computed after the campaign ends")
    sides = " · ".join(f"{k} {v}" for k, v in m["sides"].items() if v)
    steps = (
        f'<li><b>Nemotron code analysis</b>: Nemotron A reads '
        f'{_e(m["n_obs"] if m["n_obs"] is not None else "–")} code observations extracted with file and line evidence '
        f'(not yet human-reviewed, used as hints only) plus prior knowledge. Nemotron B reads '
        f'{_e(len(m["b_files"]) or "–")} av_stack source files ({_e(", ".join(Path(f).name for f in m["b_files"]) or "–")}) '
        f'with line numbers and answers 3 questions 3 times for each of {_e(m["b_cands"])} candidates.</li>'
        f'<li><b>Hypothesis</b>: A attaches a lane-kept / fail / boundary prediction and a reason to each of '
        f'{_e(m["n_props"])} scenarios{f" ({_e(sides)})" if sides else ""}.</li>'
        f'<li><b>Statistical verification</b>: each run is paired in CARLA with a fault-free baseline and graded by the '
        f'monitor rules. The GP estimates the boundary from the results and picks the next scenarios. {gp_txt}.</li>')
    rows = ""
    for x in t:
        margin = "" if x["lrr"] is None else " (%+.2f)" % x["lrr"]
        hit = "–" if x["hit"] is None else ("Hit" if x["hit"] else "Miss")
        rows += (f'<tr><td>{_e(x["run_id"])}</td><td>{_e(x["kind"])}</td><td>{_e(x["expected"])}</td>'
                 f'<td>{_e(x["actual"])}{margin}</td><td>{hit}</td></tr>')
    table = (f'<table class="kv mtab"><tr><th>Run</th><th>Source</th><th>Expected</th><th>Actual (lrr)</th><th>Result</th></tr>{rows}</table>'
             if rows else "<p>None yet</p>")
    return (f'<details data-k="methsteps"><summary>Details (per step · per prediction)</summary><ol class="msteps">{steps}</ol>{table}'
            '<p>Hits are counted only for runs with a switch (without one, the LQR response is not tested). A boundary '
            'prediction is a hit when |lrr| &lt; 0.2. "A variant" is an A scenario slightly changed by the GP, so its '
            'original prediction is compared for reference only.</p></details>')


def render_method(out_dir: Path = Path(".")) -> str:
    """Prediction-hit line + method details (kept for tests / optional use)."""
    m = load_method(out_dir)
    t = m["tests"]
    graded = [x for x in t if x["hit"] is not None]
    key = (f"Prediction hits {sum(1 for x in graded if x['hit'])}/{len(graded)}" if graded else "No graded predictions")
    n_untested = sum(1 for x in t if x["hit"] is None)
    key += f" · no switch {n_untested}" if n_untested else ""
    return f'<div class="meth" id="meth"><p class="hkey">{_e(key)}</p>{render_method_details(m)}</div>'


def load_hypotheses(out_dir: Path) -> dict:
    """What this campaign tests, from files run.py writes next to screen/ (nothing is made up):
    question  : 02_objectives.json  <S>_objective.question_en (else QUESTION_EN[<S>], else question_ko as written)
    prior     : 00_prior_payload.json  domain_prior_knowledge.principles[kind == testable_hypothesis]
    status    : runs whose Nemotron A scenario cites that id (round_*/03_04_pool.json evidence_ids), scored like load_method."""
    camp = Path(out_dir).parent

    def rj(path, default=None):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default
    obj = rj(camp / "02_objectives.json", {}) or {}
    def _question(sid):
        o = obj.get(f"{sid}_objective") or {}
        return str(o.get("question_en") or QUESTION_EN.get(sid) or o.get("question_ko") or "").strip()
    questions = [(sid, _question(sid)) for sid in obj.get("active") or []]
    questions = [q for q in questions if q[1]]
    prior = []
    pay = rj(camp / "00_prior_payload.json", {}) or {}
    for msg in pay.get("messages") or []:
        try:
            u = json.loads(msg.get("content") or "")
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(u, dict):
            for p in ((u.get("domain_prior_knowledge") or {}).get("principles") or []):
                if p.get("kind") == "testable_hypothesis":
                    prior.append({"id": p.get("id"), "text": PK_KO.get(p.get("id")) or p.get("statement") or p.get("title")})
    m = load_method(out_dir)
    for h in prior:
        ts = [x for x in m["tests"] if h["id"] in (x.get("evidence") or [])]
        h["n"] = len(ts)
        h["hit"] = sum(1 for x in ts if x["hit"] is True)
        h["miss"] = sum(1 for x in ts if x["hit"] is False)
        h["na"] = sum(1 for x in ts if x["hit"] is None)
    return {"questions": questions, "prior": prior, "method": m}


def _question_lines(text: str) -> list[str]:
    """Break a long research question at its clause joints for reading (wording unchanged)."""
    parts = re.split(r"(?<=,)\s*(?=(?:and|그리고) )|(?<=\)),\s*(?!\s*and )", text)
    return [p.strip() for p in parts if p.strip()] or [text]


def render_hypotheses(out_dir: Path = Path(".")) -> str:
    """Research question + testable hypotheses (prior-knowledge file) with Nemotron A prediction results."""
    h = load_hypotheses(out_dir)
    m = h["method"]
    t = m["tests"]
    graded = [x for x in t if x["hit"] is not None]
    hits = sum(1 for x in graded if x["hit"])
    n_untested = sum(1 for x in t if x["hit"] is None)
    q = "".join('<div class="q"><span class="lab">Evaluation question</span> '
                + "".join(f'<span class="ql">{_e(s)}</span> ' for s in _question_lines(text)) + '</div>'
                for _, text in h["questions"][:1])
    rows = ""
    for p in h["prior"][:2]:
        if p["n"]:
            res = (f'<span class="ck">Tests {p["n"]}</span> <span class="ck ok">Hit {p["hit"]}</span> '
                   f'<span class="ck bad">Miss {p["miss"]}</span> <span class="ck">Unverified {p["na"]}</span>')
        else:
            res = '<span class="none">No test targeted this hypothesis yet</span>'
        rows += (f'<li><span class="pk">{_e(p["id"])}</span> <span class="ht">{_e(p["text"])}</span> '
                 f'<div class="hs">{res}</div></li>\n')
    key = (f'Nemotron A prediction hits <b>{hits}/{len(graded)}</b>' if graded else "No graded predictions")
    if n_untested:
        key += f' <span class="mut">· unverified (no switch) {n_untested}</span>'
    body = q + (f'<ol class="hyp">{rows}</ol>' if rows else '<p class="mut">No hypothesis file</p>')
    return ('<section class="col hypo" id="hypo"><h2>Hypotheses <span class="src">Prior knowledge · predictions by Nemotron A</span></h2>'
            f'<div class="ibody">{body}<p class="hkey">{key}</p></div></section>')

def render_flow(out_dir: Path = Path("."), n_runs: int = 0, planned: int | None = None) -> str:
    """How the system works, in order: numbered steps with arrows, one line of what happens + counts from files."""
    m = load_method(out_dir)
    graded = [x for x in m["tests"] if x["hit"] is not None]
    gp = m["gp"]
    stat = (f"GP boundary accuracy {_f(gp['accuracy'] * 100, 0, '%')}" if gp.get("accuracy") is not None
            else f"Prediction hits {sum(1 for x in graded if x['hit'])}/{len(graded)}" if graded else "Grading in progress")
    obs = m["n_obs"] if m["n_obs"] is not None else "–"
    steps = [("Nemotron code analysis", "Reads supervisor and controller code, cites evidence",
              f"{obs} observations · {len(m['b_files']) or '–'} source files"),
             ("Hypothesis", "Nemotron A predicts outcomes from prior knowledge",
              f"{m['n_props']} scenarios with predictions"),
             ("Fault scenario generation", "7 faults × onset × conditions, GP picks next",
              "Steering fault injection plan"),
             ("CARLA simulation", "Fault run and fault-free baseline, same conditions",
              f"{n_runs}" + (f"/{planned}" if planned else "") + " runs"),
             ("Statistical verification", "SR monitor grading, prediction check, boundary", stat)]
    cells = '<i class="arr">→</i>'.join(
        f'<div class="step"><span class="sn">{k + 1}</span><div class="st"><b>{_e(a)}</b>'
        f'<span class="sd">{_e(d)}</span><span class="sc2">{_e(c)}</span></div></div>'
        for k, (a, d, c) in enumerate(steps))
    return f'<div class="flow" id="flow">{cells}</div>'

def render_counts(records: list[dict], commentary: dict) -> str:
    cnt = defaultdict(int)
    for r in records:
        cnt[_key(r)] += 1
    counts = [("Total", len(records), ""), ("Pass", cnt["pass"] + cnt["harmless"], "ok"),
              ("Controller handover failed/late", cnt["F1_no_or_late_switch"], "bad"),
              ("Unstable at handover", cnt["F2_switch_transient"], "bad"),
              ("Unstable after handover", cnt["F3_not_recovered"], "bad"),
              ("Not tested · invalid", cnt["no_trigger"] + cnt["invalid"], "")]
    return '<div class="counts">' + "".join(
        f'<div><b class="num {c if v else ""}">{v}</b><span>{_e(t)}</span></div>' for t, v, c in counts) + "</div>"


def load_planned(out_dir: Path) -> int | None:
    """Planned number of runs = cold_start + rounds x batch_size from <campaign>/run_config.json (None if unknown)."""
    try:
        rc = json.loads((Path(out_dir).parent / "run_config.json").read_text(encoding="utf-8"))
        return int(rc["cold_start"]) + int(rc["rounds"]) * int(rc["batch_size"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def render_scorecards(records: list[dict], planned: int | None = None) -> str:
    cnt = defaultdict(int)
    for r in records:
        cnt[_bucket(r)] += 1
    n = len(records)
    cards = "".join(f'<div class="sc {cls}" title="{_e(sub)}"><b class="num">{cnt[k]}</b><span class="l">{_e(label)}</span></div>'
                    for k, (label, sub, cls) in BUCKET.items())
    pct = min(100.0, n / planned * 100) if planned else None
    prog = (f'<div class="sc prog"><b class="num">{n}/{planned if planned else "?"}</b><span class="l">Progress</span>'
            + (f'<div class="pbar"><i style="width:{pct:.0f}%"></i></div>' if pct is not None else "") + "</div>")
    return f'<div class="cards">{cards}{prog}</div>'


def render_page(records: list[dict], refresh=REFRESH_S, commentary: dict | None = None, spec: dict | None = None,
                out_dir: Path = Path("."), insights: dict | None = None, status: str | None = None,
                planned: int | None = None) -> str:
    commentary = commentary or {}
    lim = _limits(spec)
    n = len(records)
    latest = _rid(records[-1]) if records else ""
    active = bool(commentary)
    left = "".join(render_run_panel(r, hidden=(i != n - 1), lim=lim, out_dir=out_dir,
                                    commentary=commentary.get(r.get("run_id")), active=active)
                   for i, r in enumerate(records)) or '<div class="empty">No finished scenarios yet.</div>'
    shown = list(enumerate(records))[-MAX_ROWS:]
    rows = "".join(render_run_card(r, i + 1, commentary.get(r.get("run_id"))) for i, r in reversed(shown))
    more = f'<div class="more-runs">+{n - len(shown)} more (earlier runs)</div>' if n > len(shown) else ""
    listing = (f'<div class="listbox{" dense" if len(shown) > 11 else ""}"><table class="list"><thead><tr>'
               f'<th class="scentd">Scenario</th><th class="faulttd">Fault</th><th class="swtd">Handover</th>'
               f'<th class="safetd">Driving Safety</th><th class="cliptd">Video Commentary</th>'
               f'</tr></thead><tbody>{rows}</tbody></table></div>{more}'
               if rows else '<div class="empty">No records yet.</div>')
    live = f'<span class="live">{_e(status)}</span>' if status else ""
    body = f"""<header><div class="ttl"><h1>MPC → LQR Handover Stability Evaluation</h1>
</div>
<div class="meta">{live}<span>Updated {datetime.now().strftime("%H:%M:%S")} · auto-refresh every {refresh} s</span></div></header>
{render_flow(out_dir, n, planned)}
<div class="grid">
<section class="col left" id="leftpanel"><h2><span id="lefttitle">Latest Scenario</span><button id="back" hidden>Back to latest</button></h2><div class="lbody">{left}</div></section>
<section class="col mid"><h2>Scenarios Run ({n})</h2>{render_counts(records, commentary)}{listing}</section>
<div class="stack">
{render_hypotheses(out_dir)}
{render_insights(insights, out_dir, records)}
</div>
</div>"""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<noscript><meta http-equiv="refresh" content="{refresh}"></noscript><title>MPC → LQR Handover Stability Evaluation</title><style>{CSS}</style></head>'
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


def refresh(records_path, out_dir, refresh_s=REFRESH_S, explanations=None, spec=None, status=None) -> Path:
    """Re-draw the page from run_records.jsonl (+ explanations.jsonl). run.py calls this after every run and at
    every stage change (status = what the loop is doing now, shown in the header)."""
    out_dir = Path(out_dir)
    records = [view(r) for r in _read_jsonl(Path(records_path))]
    commentary = {}
    for row in _read_jsonl(Path(explanations)) if explanations else []:
        commentary[row.get("run_id")] = row          # latest line per run wins
    if spec is None or isinstance(spec, (str, Path)):
        from .monitor import load_safety_spec
        spec = load_safety_spec(spec)
    target = out_dir / "campaign.html"
    _ensure_fonts(out_dir)
    _write(target, render_page(records, refresh_s, commentary, spec, out_dir, insights=load_insights(out_dir), status=status,
                               planned=load_planned(out_dir)))
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

