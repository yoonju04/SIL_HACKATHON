"""run.py redraws screen/campaign.html after every run and at stage changes; the browser opens at most once."""
import run as R


def test_mock_run_redraws_screen_per_run_and_opens_once(tmp_path, monkeypatch):
    statuses, opened = [], []
    real_refresh = R.refresh_dashboard

    def spy(*a, **k):
        statuses.append(k.get("status"))
        return real_refresh(*a, **k)
    monkeypatch.setattr(R, "refresh_dashboard", spy)
    monkeypatch.setattr(R.Loop, "open_browser", lambda self: opened.append(1))
    out = tmp_path / "live"
    assert R.main(["--mock", "--cold-start", "4", "--rounds", "1", "--batch-size", "2", "--out", str(out)]) == 0
    n_runs = sum(1 for line in (out / "run_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    per_run = [s for s in statuses if s and " 끝 (" in s]
    assert len(per_run) == n_runs >= 4
    assert statuses[0].startswith("시작") and statuses[-1] == "완료"
    assert any("Nemotron A" in s for s in statuses if s)
    assert opened == [1]
    assert '<span class="live">완료</span>' in (out / "screen" / "campaign.html").read_text(encoding="utf-8")


def test_open_screen_off(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(R.Loop, "open_browser", lambda self: opened.append(1))
    assert R.main(["--mock", "--cold-start", "2", "--rounds", "0", "--out", str(tmp_path / "o"), "--open-screen", "off"]) == 0
    assert opened == []
