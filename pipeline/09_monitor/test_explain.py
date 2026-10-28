"""Numerical dataset -> Token Factory -> validated insights. No videos or real API key.
Run: python pipeline/09_monitor/test_explain.py
"""
import copy
import json
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import explain
import monitor
from test_monitor import CFG, ONSET, dangerous_no_reaction, drift, trace


def records():
    info = {"run_id": "r1", "scenario_id": "S1", "environment": {"route": "Town04", "speed_mps": 15},
            "fault_plan": {"fault_type": "bias", "onset_event": "curve_entry", "magnitude": 0.05}}
    fail = monitor.evaluate_pair(trace(drift(1.0, 2.0), fault_at=ONSET), trace(), ONSET, CFG,
                                 dangerous_no_reaction(), info)
    ok = monitor.evaluate_pair(trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2), trace(), ONSET, CFG,
                               dangerous_no_reaction(), {**info, "run_id": "r2", "scenario_id": "S2"})
    return [fail, ok]


def answer(payload):
    if "records" in payload:
        r = payload["records"][0]
        ev = {"run_id": r["run_id"], "metric": "metrics.margin_delta_m", "value": r["metrics"]["margin_delta_m"]}
    else:
        reports = payload["reports"]
        ev = reports[0]["insights"][0]["evidence"][0]
    return {"summary": "수치 기반 캠페인 분석", "insights": [{"title": "차선 여유 감소", "finding": "차선 여유가 감소했다.",
            "evidence": [ev], "interpretation": "bias와 차선 여유의 관계를 통제 실험으로 확인한다.", "confidence": "low"}],
            "next_experiments": [{"proposal": "속도를 고정하고 bias만 변화", "rationale": "조건 효과 분리", "run_ids": [ev["run_id"]]}],
            "limitations": ["표본이 작아 인과관계를 확정할 수 없음"]}


class FakeAPI(BaseHTTPRequestHandler):
    calls = []
    mode = "ok"

    def log_message(self, *args):
        pass

    def send(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send({"data": [{"id": "test/Cosmos-Reason"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append({"body": body, "auth": self.headers.get("Authorization"), "path": self.path})
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        report = answer(payload)
        if self.mode == "bad_ref":
            report["insights"][0]["evidence"][0]["run_id"] = "invented"
        if self.mode == "cjk_once" and len(body["messages"]) == 2:
            report["insights"][0]["finding"] = "检测 지연이 컸다."
        content = "not JSON" if self.mode == "bad_json" else "<think>private</think>" + json.dumps(report, ensure_ascii=False)
        self.send({"choices": [{"finish_reason": "length" if self.mode == "truncated" else "stop",
                                "message": {"content": content}}]})


class EnvFileTests(unittest.TestCase):
    def test_env_file_loads_key_model_quotes_and_comments(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {}, clear=True):
            p = Path(d) / '.env'
            p.write_text('# local settings\nexport NEBIUS_TOKEN_FACTORY_KEY="v1.test#literal" # comment\n'
                         "COSMOS_MODEL='test/Cosmos-Reason'\n"
                         'NEBIUS_TOKEN_FACTORY_BASE_URL=https://example.invalid/v1/ # endpoint\n'
                         'UNRELATED_SECRET=ignored\n', encoding='utf-8-sig')
            c = explain.ModelClient(env_file=p)
            self.assertEqual(c.key, 'v1.test#literal')
            self.assertEqual(c.model, 'test/Cosmos-Reason')
            self.assertEqual(c.base, 'https://example.invalid/v1/')
            self.assertTrue(c.ready)
            self.assertNotIn('UNRELATED_SECRET', explain.load_env_file(p))
            self.assertNotIn('NEBIUS_TOKEN_FACTORY_KEY', os.environ)

    def test_environment_and_explicit_arguments_override_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / '.env'
            p.write_text('NEBIUS_TOKEN_FACTORY_KEY=v1.file\nCOSMOS_MODEL=file/Cosmos\n')
            with patch.dict(os.environ, {'NEBIUS_TOKEN_FACTORY_KEY': 'v1.shell', 'COSMOS_MODEL': 'shell/Cosmos'}, clear=True):
                c = explain.ModelClient(env_file=p)
                self.assertEqual((c.key, c.model), ('v1.shell', 'shell/Cosmos'))
                c = explain.ModelClient(key='v1.explicit', model='explicit/Cosmos', env_file=p)
                self.assertEqual((c.key, c.model), ('v1.explicit', 'explicit/Cosmos'))

    def test_default_path_is_independent_of_working_directory_and_file_edits_reload(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {}, clear=True):
            p = Path(d) / '.env'
            p.write_text('NEBIUS_TOKEN_FACTORY_KEY=v1.first\nCOSMOS_MODEL=test/Cosmos\n')
            elsewhere = Path(d) / 'elsewhere'
            elsewhere.mkdir()
            before = Path.cwd()
            try:
                os.chdir(elsewhere)
                with patch.object(explain, 'DEFAULT_ENV_FILE', p):
                    self.assertEqual(explain.ModelClient().key, 'v1.first')
                    p.write_text('NEBIUS_TOKEN_FACTORY_KEY=v1.second\nCOSMOS_MODEL=test/Cosmos\n')
                    self.assertEqual(explain.ModelClient().key, 'v1.second')
            finally:
                os.chdir(before)

    def test_missing_or_empty_file_and_safe_parse_error(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {}, clear=True):
            p = Path(d) / '.env'
            self.assertFalse(explain.ModelClient(env_file=p).ready)
            p.write_text('NEBIUS_TOKEN_FACTORY_KEY=\nCOSMOS_MODEL=\n')
            self.assertFalse(explain.ModelClient(env_file=p).ready)
            p.write_text('NEBIUS_TOKEN_FACTORY_KEY="v1.private-unclosed\n')
            with self.assertRaises(ValueError) as caught:
                explain.ModelClient(env_file=p)
            self.assertNotIn('private', str(caught.exception))


class DatasetTests(unittest.TestCase):
    def test_real_runner_projection_preserves_both_verdicts_and_baseline(self):
        raw = records()
        before = copy.deepcopy(raw)
        ds = explain.build_dataset(raw)
        self.assertEqual(raw, before)
        row = ds["records"][0]
        self.assertEqual(row["monitor_outcome"], "F1_no_or_late_switch")
        self.assertEqual(row["runner_outcome"], raw[0]["outcome_category"])
        self.assertEqual(row["baseline_metrics"], explain.clean(raw[0]["baseline_metrics"]))
        self.assertTrue(row["sr"])
        self.assertEqual(row["environment"]["speed_mps"], 15)
        self.assertNotIn("monitor", row["metrics"])
        self.assertNotIn("run_dirs", row)

    def test_denominator_excludes_invalid_and_no_trigger_and_preserves_null(self):
        rows = explain.build_dataset(records())["records"]
        for rid, outcome in [("r3", "invalid"), ("r4", "no_trigger")]:
            rows.append({**copy.deepcopy(rows[0]), "run_id": rid, "monitor_outcome": outcome})
        rows[0]["metrics"]["missing"] = None
        rows[1]["metrics"]["missing"] = 2.0
        s = explain.summarize(rows)
        self.assertEqual(s["total_runs"], 4)
        self.assertEqual(s["graded_triggered_runs"], 2)
        self.assertEqual(s["failure_rate"], 0.5)
        self.assertEqual(s["metrics"]["missing"]["mean"], 2)
        self.assertEqual(s["metrics"]["missing"]["missing"], 1)
        raw = records()
        raw[0]["metrics"]["not_finite"] = float("inf")
        self.assertIsNone(explain.build_dataset(raw)["records"][0]["metrics"]["not_finite"])

    def test_failures_only_keeps_whole_campaign_denominator(self):
        ds = explain.build_dataset(records(), failures_only=True)
        self.assertEqual(ds["selected_runs"], 1)
        self.assertEqual(ds["summary"]["total_runs"], 2)
        self.assertEqual(ds["selection"], "failures_only")

    def test_duplicate_or_missing_ids_are_rejected(self):
        raw = records()
        with self.assertRaises(ValueError):
            explain.build_dataset([raw[0], raw[0]])
        raw[0]["run_id"] = None
        with self.assertRaises(ValueError):
            explain.build_dataset(raw)

    def test_evidence_must_match_id_path_value_and_type(self):
        ds = explain.build_dataset(records())
        report = answer(ds)
        for field, value in [("run_id", "fake"), ("metric", "metrics.fake"), ("value", 12345), ("value", None)]:
            changed = copy.deepcopy(report)
            changed["insights"][0]["evidence"][0][field] = value
            with self.assertRaises(ValueError):
                explain.parse_insights(json.dumps(changed), ds["records"])
        self.assertEqual(explain.parse_insights("```json\n" + json.dumps(report) + "\n```", ds["records"]), report)
        with self.assertRaises(ValueError):
            explain.parse_insights("plain answer", ds["records"])

    def test_batch_size_limit_never_silently_drops_an_oversized_record(self):
        with self.assertRaises(ValueError):
            list(explain._batches([{"text": "x" * 2000}], {}, "records", 40, 1000))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rr = self.root / "run_records.jsonl"
        self.raw = records()
        self.write_records()
        self.screen = self.root / "screen"
        FakeAPI.calls.clear()
        FakeAPI.mode = "ok"
        self.srv = HTTPServer(("127.0.0.1", 0), FakeAPI)
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()
        self.client = explain.ModelClient(key="v1.test", base_url=f"http://127.0.0.1:{self.srv.server_port}/v1/", model="test/Cosmos-Reason")

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join()
        self.temp.cleanup()

    def write_records(self):
        self.rr.write_text("".join(json.dumps(r) + "\n" for r in self.raw), encoding="utf-8")

    def report(self):
        return json.loads((self.screen / "insights.json").read_text(encoding="utf-8"))

    def test_text_only_http_contract_and_dashboard_without_video(self):
        original = self.rr.read_bytes()
        self.assertIn(self.client.model, self.client.models())
        self.assertEqual(explain.process(self.rr, self.screen, self.client), 1)
        self.assertEqual(self.rr.read_bytes(), original)
        call = FakeAPI.calls[0]
        self.assertEqual(call["path"], "/v1/chat/completions")
        self.assertEqual(call["auth"], "Bearer v1.test")
        self.assertTrue(all(isinstance(m["content"], str) for m in call["body"]["messages"]))
        self.assertNotIn("image_url", json.dumps(call["body"]))
        self.assertEqual(self.report()["status"], "complete")
        page = (self.screen / "campaign.html").read_text(encoding="utf-8")
        self.assertIn('class="insight-title">인사이트', page)
        self.assertIn("차선 여유 감소", page)
        self.assertIn('<div class="insights">', page)
        self.assertNotIn("AI 영상 판단", page)
        self.assertNotIn("<video", page)
        self.assertFalse((self.screen / "clips").exists())

    def test_cache_data_change_and_model_change(self):
        self.assertEqual(explain.process(self.rr, self.screen, self.client), 1)
        self.assertEqual(explain.process(self.rr, self.screen, self.client), 0)
        self.client.model = "test/Other-Cosmos"
        self.assertEqual(explain.process(self.rr, self.screen, self.client), 1)
        self.raw[0]["metrics"]["margin_delta_m"] -= 0.01
        self.write_records()
        self.assertEqual(explain.process(self.rr, self.screen, self.client), 1)
        self.assertEqual(len(FakeAPI.calls), 3)

    def test_multiple_batches_and_synthesis_cover_all_runs(self):
        for i in range(3, 6):
            self.raw.append({**copy.deepcopy(self.raw[0]), "run_id": f"r{i}"})
        self.write_records()
        calls = explain.process(self.rr, self.screen, self.client, batch_runs=2)
        self.assertEqual(calls, 6)  # 3 raw batches -> 2 intermediate -> 1 final
        seen = []
        for c in FakeAPI.calls:
            payload = json.loads(c["body"]["messages"][1]["content"].split("\n", 1)[1])
            seen.extend(r["run_id"] for r in payload.get("records", []))
            self.assertLessEqual(len(explain.dumps(payload)), explain.MAX_INPUT_CHARS)
        self.assertEqual(seen, [f"r{i}" for i in range(1, 6)])
        self.assertEqual(self.report()["status"], "complete")

    def test_malformed_and_hallucinated_responses_are_errors_and_retryable(self):
        for mode in ("bad_json", "bad_ref", "truncated"):
            FakeAPI.mode = mode
            explain.process(self.rr, self.screen, self.client, force=True)
            self.assertEqual(self.report()["status"], "error")
            self.assertEqual(self.report()["insights"], [])
        FakeAPI.mode = "ok"
        self.assertEqual(explain.process(self.rr, self.screen, self.client, redo_errors=True), 1)
        self.assertEqual(self.report()["status"], "complete")

    def test_rejected_answer_is_reasked_with_reason(self):
        FakeAPI.mode = "cjk_once"
        self.assertEqual(explain.process(self.rr, self.screen, self.client), 2)
        self.assertEqual(self.report()["status"], "complete")
        retry = FakeAPI.calls[1]["body"]["messages"]
        self.assertEqual(len(retry), 4)
        self.assertIn("거절", retry[-1]["content"])

    def test_no_key_still_writes_dataset_and_clear_dashboard_error(self):
        explain.process(self.rr, self.screen, explain.ModelClient(key="", model=""))
        self.assertEqual(self.report()["status"], "error")
        self.assertTrue((self.screen / "insight_dataset.json").exists())
        self.assertEqual(len(FakeAPI.calls), 0)

    def test_empty_campaign_does_not_call_model(self):
        self.raw = []
        self.write_records()
        self.assertEqual(explain.process(self.rr, self.screen, self.client), 0)
        self.assertEqual(self.report()["status"], "empty")
        self.assertEqual(len(FakeAPI.calls), 0)

    def test_dashboard_hides_insights_when_records_change(self):
        import dashboard
        explain.process(self.rr, self.screen, self.client)
        self.raw[0]["metrics"]["margin_delta_m"] -= 0.2
        self.write_records()
        # A new input artifact may already be written while the model is still running.
        explain._write_json(self.screen / "insight_dataset.json", explain.build_dataset(self.raw))
        dashboard.refresh(self.rr, self.screen)
        page = (self.screen / "campaign.html").read_text(encoding="utf-8")
        self.assertIn("기록이 변경", page)
        self.assertNotIn("차선 여유 감소", page)


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.ds = explain.build_dataset(records())
        self.report = answer(self.ds)
        self.value = self.report["insights"][0]["evidence"][0]["value"]

    def check(self, report):
        return explain.parse_insights(json.dumps(report, ensure_ascii=False), self.ds["records"], self.ds["summary"])

    def test_facts_count_scenarios_routes_and_faults(self):
        f = self.ds["summary"]["facts"]
        self.assertEqual((f["n_runs"], f["n_scenarios"], f["routes"], f["fault_types"]), (2, 2, ["Town04"], ["bias"]))

    def test_clean_answer_with_rounded_evidence_number_passes(self):
        r = copy.deepcopy(self.report)
        r["insights"][0]["finding"] = f"차선 여유 변화가 {self.value:.2f}m였다."
        self.assertEqual(self.check(r)["insights"][0]["finding"], r["insights"][0]["finding"])

    def test_han_or_kana_is_rejected(self):
        for text in ("检测 지연", "テスト 결과"):
            r = copy.deepcopy(self.report)
            r["insights"][0]["interpretation"] = text
            with self.assertRaises(ValueError):
                self.check(r)

    def test_number_not_in_evidence_or_statistics_is_rejected(self):
        r = copy.deepcopy(self.report)
        r["insights"][0]["finding"] = "차선 여유가 123.457m였다."
        with self.assertRaises(ValueError) as caught:
            self.check(r)
        self.assertIn("123.457", str(caught.exception))
        r = copy.deepcopy(self.report)
        r["summary"] = "평균 98.765m 감소"
        with self.assertRaises(ValueError):
            self.check(r)

    def test_relaxing_safety_threshold_is_rejected(self):
        r = copy.deepcopy(self.report)
        r["next_experiments"][0]["proposal"] = "SR4 기준을 0.5m로 완화한 실험"
        with self.assertRaises(ValueError):
            self.check(r)

    def test_same_scenario_claim_rejected_only_when_dataset_has_several(self):
        r = copy.deepcopy(self.report)
        r["limitations"] = ["모든 실험은 동일한 시나리오에서 수행되었다."]
        with self.assertRaises(ValueError):
            self.check(r)
        explain.parse_insights(json.dumps(r, ensure_ascii=False), self.ds["records"])  # no facts -> not checked


if __name__ == "__main__":
    unittest.main()
