"""08 CARLA paired run: subprocess wrapper around sim-eval av_stack run_benchmark.py (+ mock runner).

Per candidate: a no-fault baseline (cached by the 08 match key) and a fault run, both with
--planner pdm --controller supervised --record-front; optionally (08 paired_runs.yaml no_reaction_run) a
no-reaction run with the same fault and --controller mpc_unprotected, used by engine.sr_checks SR1 to measure FTTI. The baseline also gives the onset_event times
(engine.scenarios.detect_route_events). sim-eval code is called, never modified.

run_benchmark.py arguments used (read from sim-eval 9b51151 run_benchmark.py:21-58):
  --routes, --route-id, --max-speed, --no-view, --green-lights, --controller, --planner, --record-front,
  --out (must be a new folder), --host, --port, --traffic-manager-port, --seed, --timeout,
  --mpc-fault, --fault-magnitude, --fault-on, --fault-off, --fault-start, --fault-duration, --stop-after-fault
  integration3: --saturate-level, --fault-ramp (saturate), --tire-friction-scale (baseline, fault and no-reaction runs)
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import judgment as J
from .config import import_scenario_space_tool
from .scenarios import fault_cli_local, fault_dict_for_sim, friction_of

NO_REACTION_CONTROLLER = "mpc_unprotected"  # run_benchmark.py --controller choice (sim-eval PR #1): MPC, no supervisor


def baseline_key(c: dict, route: dict, seed: int, planner: str = "pdm") -> dict:
    """08 paired_runs.match fields (BASELINE_MATCH_FIELDS); axes the runner does not vary are None."""
    vals = {f: None for f in J.BASELINE_MATCH_FIELDS}
    vals.update(map=route.get("town"), route_id=c["route_id"], seed=seed, max_speed_kmh=c["environment"]["max_speed_kmh"],
                planner=planner, tire_friction_scale=friction_of(c))   # integration3: baseline at the same friction
    return vals


def key_hash(key: dict) -> str:
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass
class RunResult:
    run_dir: Path
    returncode: int
    command: list[str]
    seconds: float
    log: Path | None = None


class BaseRunner:
    def __init__(self, out_root: Path, routes: list[dict], seed: int, space: dict, stop_after_fault_s: float,
                 supervisor: str | None = None):
        self.out_root = Path(out_root)
        # integration3 #8: run_benchmark.py --supervisor tight|legacy (None = run_benchmark default, tight)
        self.supervisor = supervisor
        self.routes = {r["route_id"]: r for r in routes}
        self.seed = seed
        self.space = space
        self.stop_after_fault_s = stop_after_fault_s

    def baseline_dir(self, c: dict) -> Path:
        k = baseline_key(c, self.routes[c["route_id"]], self.seed)
        return self.out_root / "baselines" / key_hash(k)

    def ensure_baseline(self, c: dict) -> tuple[Path, bool]:
        d = self.baseline_dir(c)
        if (d / "steps.jsonl").is_file() and (d / ".complete").is_file():
            return d, True
        if (d / "results.json").is_file():
            # BUG-013: this baseline already ran and failed (no .complete). Same seed/route/speed -> same result;
            # rerunning would also hit _run's fresh --out check and crash the campaign. Caller rejects it.
            return d, True
        res = self._run(c, d, fault_args=None)
        if res.returncode == 0:
            (d / ".complete").write_text(json.dumps(baseline_key(c, self.routes[c["route_id"]], self.seed)))
        return d, False

    def no_reaction_dir(self, c: dict, onset_s: float) -> Path:
        k = dict(baseline_key(c, self.routes[c["route_id"]], self.seed), controller=NO_REACTION_CONTROLLER,
                 fault=c["fault"], onset_s=round(float(onset_s), 2))
        return self.out_root / "no_reaction" / key_hash(k)

    def ensure_no_reaction(self, c: dict, onset_s: float) -> tuple[Path, bool]:
        """SR1 FTTI run: same route/env/fault/onset, no supervisor (`--controller mpc_unprotected`). Cached."""
        d = self.no_reaction_dir(c, onset_s)
        if (d / "steps.jsonl").is_file() and (d / ".complete").is_file():
            return d, True
        if (d / "results.json").is_file():   # BUG-013: failed earlier, do not rerun into the same --out
            return d, True
        res = self.run_no_reaction(c, d, onset_s)
        if res.returncode == 0:
            (d / ".complete").write_text(json.dumps({"controller": NO_REACTION_CONTROLLER, "onset_s": onset_s}))
        return d, False

    def run_fault(self, c: dict, run_dir: Path, onset_s: float) -> RunResult:
        raise NotImplementedError

    def run_no_reaction(self, c: dict, run_dir: Path, onset_s: float) -> RunResult:
        raise NotImplementedError

    def _run(self, c: dict, out: Path, fault_args) -> RunResult:
        raise NotImplementedError


class MockRunner(BaseRunner):
    """SYNTHETIC: writes run folders with engine/fixtures/synthetic_sim.py (no CARLA)."""

    def _run(self, c, out, fault_args=None, onset_s=None, controller="supervised"):
        from .fixtures.synthetic_sim import write_run_folder
        t0 = time.time()
        fault = None if onset_s is None else fault_dict_for_sim(self.space, c, onset_s)
        write_run_folder(out, route=c["route_id"], max_speed_kmh=c["environment"]["max_speed_kmh"], fault=fault,
                         tire_friction_scale=friction_of(c),
                         seed=self.seed, stop_after_fault_s=self.stop_after_fault_s if fault else None,
                         record_front=fault is not None and controller == "supervised", controller=controller,
                         supervisor=self.supervisor or "tight")
        return RunResult(out, 0, ["<synthetic_sim>", c["route_id"], controller], time.time() - t0)

    def run_fault(self, c, run_dir, onset_s):
        return self._run(c, run_dir, onset_s=onset_s)

    def run_no_reaction(self, c, run_dir, onset_s):
        return self._run(c, run_dir, onset_s=onset_s, controller=NO_REACTION_CONTROLLER)


class CarlaRunner(BaseRunner):
    """Real mode: one CARLA server already running (tools/wait_carla.py, start_carla_stable.ps1 in sim-eval)."""

    def __init__(self, out_root, routes, seed, space, stop_after_fault_s, supervisor=None, *, av_stack: Path, python_exe: str | None,
                 host: str = "localhost", port: int = 2000, tm_port: int = 8050, timeout_s: float = 120.0,
                 wall_timeout_s: float = 1800.0, chase_view: bool = False):
        super().__init__(out_root, routes, seed, space, stop_after_fault_s, supervisor)
        self.av_stack = Path(av_stack)
        self.python = python_exe or default_python(self.av_stack)
        self.host, self.port, self.tm_port = host, port, tm_port
        self.timeout_s, self.wall_timeout_s = timeout_s, wall_timeout_s
        self.chase_view = chase_view
        self.tool = import_scenario_space_tool()

    def command(self, c: dict, out: Path, fault_args: list[str] | None, controller: str = "supervised") -> list[str]:
        r = self.routes[c["route_id"]]
        cmd = [self.python, str(self.av_stack / "run_benchmark.py"), "--routes", r["routes_arg"]]
        if r.get("route_id_arg"):
            cmd += ["--route-id", str(r["route_id_arg"])]
        cmd += ["--planner", "pdm", "--controller", controller] + (["--record-front"] if controller == "supervised" else []) + [
                "--max-speed", "%.1f" % c["environment"]["max_speed_kmh"], "--seed", str(self.seed),
                "--host", self.host, "--port", str(self.port), "--traffic-manager-port", str(self.tm_port),
                "--timeout", str(self.timeout_s), "--out", str(out)]
        fr = friction_of(c)
        if abs(fr - 1.0) > 1e-9:   # integration3: same tyres for the baseline, fault and no-reaction runs
            cmd += ["--tire-friction-scale", "%.2f" % fr]
        if self.supervisor:   # integration3 #8: explicit, so agent_config.json always records the profile
            cmd += ["--supervisor", str(self.supervisor)]
        if not self.chase_view:
            cmd.append("--no-view")   # view.mp4 needs the chase window (chase_view.py); front.mp4 does not
        cmd += list(r.get("extra_args") or [])
        if fault_args:
            # --max-speed comes from the environment mapping too; keep only the fault part of to_cli
            fa = list(fault_args)
            for flag in ("--max-speed", "--tire-friction-scale"):   # both already set above from the environment
                if flag in fa:
                    i = fa.index(flag); del fa[i:i + 2]
            cmd += fa + ["--stop-after-fault", "%.1f" % self.stop_after_fault_s]
        return cmd

    def fault_args(self, c: dict, onset_s: float) -> list[str]:
        if self.tool is not None:
            scen = {"fault_id": c["fault"]["fault_id"], "params": c["fault"]["params"], "environment": c["environment"]}
            return self.tool.to_cli(self.space, scen, onset_s)
        return fault_cli_local(self.space, c, onset_s)

    def _run(self, c, out, fault_args=None, controller="supervised"):
        out = Path(out).resolve()   # run_benchmark.py runs with cwd=av_stack, so a relative --out would land there
        if (out / "results.json").exists():
            raise FileExistsError(f"{out} already has results.json (run_benchmark.py needs a fresh --out)")
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.command(c, out, fault_args, controller)
        log = out.parent / (out.name + ".log")
        t0 = time.time()
        env = dict(os.environ)
        with log.open("w", encoding="utf-8") as fh:
            try:
                proc = subprocess.run(cmd, cwd=str(self.av_stack), stdout=fh, stderr=subprocess.STDOUT,
                                      timeout=self.wall_timeout_s, env=env)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = -9
        return RunResult(out, rc, cmd, time.time() - t0, log)

    def run_fault(self, c, run_dir, onset_s):
        return self._run(c, run_dir, self.fault_args(c, onset_s))

    def run_no_reaction(self, c, run_dir, onset_s):
        return self._run(c, run_dir, self.fault_args(c, onset_s), controller=NO_REACTION_CONTROLLER)


def default_python(av_stack: Path) -> str:
    win = av_stack / ".venv-tfpp" / "Scripts" / "python.exe"
    nix = av_stack / ".venv-tfpp" / "bin" / "python"
    if win.is_file():
        return str(win)
    if nix.is_file():
        return str(nix)
    return os.environ.get("AV_STACK_PYTHON", sys.executable)


def carla_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
