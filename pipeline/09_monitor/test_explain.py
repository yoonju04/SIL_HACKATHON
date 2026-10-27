"""Tests for video_clip.py + explain.py with a synthetic video and a local fake chat API:
python pipeline/09_monitor/test_explain.py"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explain  # noqa: E402
import monitor  # noqa: E402
from test_monitor import CFG, ONSET, dangerous_no_reaction, drift, trace  # noqa: E402
from video_clip import ffmpeg_exe, make_clip  # noqa: E402

JUDGE = {"severity": 3}           # tests change this to steer the fake model's blind judgment


def judge_answer():
    sev = JUDGE["severity"]
    return ("<think>프레임 4부터 차가 오른쪽 선을 넘는다</think>" + json.dumps({
        "lane": "crosses_line" if sev >= 2 else "centered", "motion": "slow_drift" if sev else "smooth",
        "surroundings": [{"type": "oncoming_vehicle", "side": "left", "proximity": "very_near"},
                         {"type": "spaceship", "side": "up", "proximity": "near"}],
        "near_miss": sev >= 2, "near_miss_detail": "t=11.5s 맞은편 차와 1 m 이내" if sev >= 2 else None,
        "switch_seen_frame": None, "severity": sev, "evidence_frames": [4, 5, 99], "confidence": "medium"},
        ensure_ascii=False))


ANSWER = ('<think>프레임 3에서 차가 오른쪽으로 밀린다...</think>\n'
          '{"summary": "bias 고장 뒤 차가 오른쪽 차선으로 밀렸고 LQR 전환이 없었다.", '
          '"timeline": ["t=10.0s 고장 시작", "t=11.6s 차체가 차선을 넘음"], '
          '"why": "위험한 고장인데 MPC가 계속 운전했다.", "mismatch": null}')


class FakeAPI(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *a):
        pass

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"data": [{"id": "nvidia/Cosmos-Reason2-8B"}, {"id": "openai/gpt-oss-120b"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeAPI.calls.append({"auth": self.headers.get("Authorization"), "body": body})
        text = body["messages"][1]["content"][-1]["text"]
        content = judge_answer() if "[과제: 영상만 보고 판단]" in text else ANSWER
        self._send({"choices": [{"message": {"content": content}}]})


def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1/"


def synthetic_run(folder: Path, seconds=20.0, fps=20):
    """A run folder with front.mp4 (test pattern) + front_frames.jsonl (sim time per frame)."""
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=640x360:rate={fps}:duration={seconds}", "-c:v", "mpeg4", str(folder / "front.mp4")],
                   check=True)
    with (folder / "front_frames.jsonl").open("w") as fh:
        for k in range(int(seconds * fps)):
            fh.write(json.dumps({"sensor_frame": k, "t": round(k / fps, 3)}) + "\n")


def _probe_codec(path):
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    return "h264" in out


def test_parse_answer_drops_think_and_reads_json():
    a = explain.parse_answer(ANSWER)
    assert a["summary"].startswith("bias 고장") and len(a["timeline"]) == 2
    b = explain.parse_answer("<think>x</think> 그냥 문장으로 답함")
    assert b["summary"] == "그냥 문장으로 답함" and b["timeline"] == []


def test_clip_is_h264_and_frames_carry_sim_time():
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "run"
        synthetic_run(run)
        c = make_clip(run, 9.0, 15.0, Path(d) / "clip.mp4", Path(d) / "frames", n_frames=6)
        assert c and _probe_codec(c["clip"]) and c["exact_timing"] and c["source"] == "front.mp4"
        assert len(c["frames"]) == 6 and 9.0 < c["frames"][0]["t"] < c["frames"][-1]["t"] < 15.0


def _records(d, video_for_pass=False):
    run = Path(d) / "fault"
    synthetic_run(run)
    info = {"run_id": "r1", "scenario_id": "S1", "environment": {"route": "Town04"},
            "fault_plan": {"fault_type": "bias", "onset_event": "curve_entry", "magnitude": 0.05}}
    fail = monitor.evaluate_pair(trace(drift(1.0, 2.0), fault_at=ONSET), trace(), ONSET, CFG,
                                 dangerous_no_reaction(), info)
    fail["metrics"]["monitor"]["run_dirs"] = {"fault": str(run)}
    ok = monitor.evaluate_pair(trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2), trace(), ONSET, CFG,
                               dangerous_no_reaction(), {**info, "run_id": "r2", "scenario_id": "S2"})
    if video_for_pass:
        ok["metrics"]["monitor"]["run_dirs"] = {"fault": str(run)}
    rr = Path(d) / "rr.jsonl"
    rr.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in (fail, ok)), encoding="utf-8")
    return rr


def test_parse_judgment_keeps_only_known_words():
    j = explain.parse_judgment(judge_answer(), n_frames=8)
    assert j["lane"] == "crosses_line" and j["severity"] == 3
    assert [x["type"] for x in j["surroundings"]] == ["oncoming_vehicle"]          # "spaceship" dropped
    assert j["evidence_frames"] == [4, 5]                                         # frame 99 does not exist
    assert explain.parse_judgment('{"lane": "x", "severity": 9}', 8)["severity"] == 3
    assert explain.parse_judgment("no json here", 8) is None


def test_compare_monitor_vs_video():
    fail, ok = {"outcome_category": "F1_no_or_late_switch"}, {"outcome_category": "pass"}
    c = lambda r, sev: explain.compare(r, {"severity": sev, "switch_seen_frame": None}, [], False)["agreement"]  # noqa: E731
    assert c(fail, 3) == "agree_risky" and c(fail, 1) == "agree_minor" and c(fail, 0) == "video_shows_no_risk"
    assert c(ok, 2) == "monitor_may_miss" and c(ok, 0) == "agree_safe"
    frames = [{"t": 10.0 + 0.5 * i} for i in range(8)]
    sw = {"outcome_category": "pass", "switch_time": 11.0}
    assert explain.compare(sw, {"severity": 0, "switch_seen_frame": 3}, frames, True)["switch_check"] == "consistent"
    assert explain.compare(sw, {"severity": 0, "switch_seen_frame": 7}, frames, True)["switch_check"] == "mismatch"


def test_blind_judgment_then_commentary_on_every_run():
    srv, base = fake_server()
    FakeAPI.calls.clear()
    JUDGE["severity"] = 3
    try:
        with tempfile.TemporaryDirectory() as d:
            rr = _records(d, video_for_pass=True)
            screen = Path(d) / "screen"
            client = explain.ModelClient(key="v1.test", base_url=base, model="nvidia/Cosmos-Reason2-8B")
            assert "nvidia/Cosmos-Reason2-8B" in client.models()
            assert explain.process(rr, screen, client) == 2                 # the pass is judged too
            judge = [c for c in FakeAPI.calls if "[과제: 영상만 보고 판단]" in c["body"]["messages"][1]["content"][-1]["text"]]
            assert len(judge) == 2 and len(FakeAPI.calls) == 4              # 2 runs x (judge + commentary)
            jt = judge[0]["body"]["messages"][1]["content"][-1]["text"]
            assert "결과:" not in jt and "F1" not in jt and "어긴 기준" not in jt   # blind: no verdict given
            parts = judge[0]["body"]["messages"][1]["content"]
            assert sum(p["type"] == "image_url" for p in parts) == 8 and FakeAPI.calls[0]["auth"] == "Bearer v1.test"
            ct = FakeAPI.calls[1]["body"]["messages"][1]["content"][-1]["text"]
            assert "결과: F1" in ct and "crosses_line" in ct                  # commentary sees both
            rows = explain.read_commentary(screen)
            assert rows["r1"]["agreement"] == "agree_risky" and rows["r2"]["agreement"] == "monitor_may_miss"
            assert rows["r1"]["judgment"]["near_miss"] and rows["r1"]["summary"].startswith("bias 고장")
            page = (screen / "campaign.html").read_text(encoding="utf-8")
            assert '<video class="clip" src="clips/r1.mp4"' in page and "위험도 3/3" in page and "불일치" in page
            # stage 3: video severity and context reach the failure table; the pass that looked risky -> review
            import feedback
            ft = feedback.failure_table([json.loads(x) for x in rr.read_text(encoding="utf-8").splitlines()],
                                        min_reproduction=1, judgments=rows)
            row = ft["candidates"][0] if ft["candidates"] else ft["verified"][0]
            assert row["vision"]["severity"] == 3 and "맞은편 차량" in row["summary"]
            assert [x["run_id"] for x in ft["needs_review"]] == ["r2"]
            assert explain.process(rr, screen, client) == 0                 # nothing new -> no more calls
    finally:
        srv.shutdown()


def test_without_key_the_clip_still_shows():
    with tempfile.TemporaryDirectory() as d:
        rr = _records(d)
        screen = Path(d) / "screen"
        explain.process(rr, screen, explain.ModelClient(key="", model=""), failures_only=True)
        row = explain.read_commentary(screen)["r1"]
        assert row["clip"] and "모델 미설정" in row["error"]
        assert '<video class="clip"' in (screen / "campaign.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    if ffmpeg_exe() is None:
        print("SKIP: no ffmpeg (install ffmpeg or `pip install imageio-ffmpeg`)")
        sys.exit(0)
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
