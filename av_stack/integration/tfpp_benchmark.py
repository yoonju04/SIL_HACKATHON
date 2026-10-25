"""Windows existing-server adapter for the unchanged Bench2Drive evaluator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys


ROOT = Path(__file__).resolve().parents[1]
GARAGE = ROOT / "third_party" / "carla_garage"
BENCH = GARAGE / "Bench2Drive"


def carla_root() -> Path:
    """CARLA_ROOT if set, else the CARLA folder next to the repository (either spelling)."""
    if os.environ.get("CARLA_ROOT"):
        return Path(os.environ["CARLA_ROOT"])
    for name in ("CARLA_0.9.15", "CALRA 0.9.15"):
        candidate = ROOT.parents[1] / name / "WindowsNoEditor"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Set CARLA_ROOT to the CARLA 0.9.15 WindowsNoEditor folder")


def prepare_imports() -> None:
    """Select one consistent upstream leaderboard/scenario-runner package pair."""
    paths = (BENCH / "leaderboard", BENCH / "scenario_runner", GARAGE / "team_code",
             carla_root() / "PythonAPI" / "carla", ROOT)
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    os.environ["SCENARIO_RUNNER_ROOT"] = str(BENCH / "scenario_runner")
    os.environ["LEADERBOARD_ROOT"] = str(BENCH / "leaderboard")
    os.environ["WORK_DIR"] = str(BENCH)


def evaluate(options: argparse.Namespace, agent_config: Path) -> int:
    prepare_imports()
    import carla
    from leaderboard import leaderboard_evaluator as upstream
    from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
    from integration.carla_io import flatten_steering_curve, vehicle_geometry
    from leaderboard.utils.route_indexer import RouteIndexer
    from leaderboard.utils.statistics_manager import StatisticsManager
    from leaderboard.scenarios.scenario_manager import ScenarioManager

    if not getattr(ScenarioManager, '_stop_request_patch', False):
        upstream_tick = ScenarioManager._tick_scenario

        def tick_until_agent_requests_stop(manager) -> None:
            """Demo option --stop-after-fault: the agent sets stop_requested; scoring is unchanged."""
            upstream_tick(manager)
            if getattr(getattr(manager._agent_wrapper, '_agent', None), 'stop_requested', False):
                manager._running = False

        ScenarioManager._tick_scenario = tick_until_agent_requests_stop
        ScenarioManager._stop_request_patch = True
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

    class ExistingServerEvaluator(LeaderboardEvaluator):
        """Reuse official route/scenario logic while owning no CARLA process."""

        def _setup_simulation(self, args: argparse.Namespace) -> tuple:
            client = carla.Client(args.host, args.port)
            client.set_timeout(args.timeout)
            if client.get_server_version() != "0.9.15":
                raise RuntimeError("This Bench2Drive integration requires CARLA server 0.9.15")
            world = client.get_world()
            self.original_settings = world.get_settings()
            self.world = world
            traffic = client.get_trafficmanager(args.traffic_manager_port)
            return client, args.timeout, traffic

        def _load_and_wait_for_world(self, args: argparse.Namespace, town: str) -> None:
            current = self.client.get_world()
            if current.get_map().name.rsplit("/", 1)[-1] != town:
                self.world = self.client.load_world(town, reset_settings=False)
            else:
                self.world = current
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.frame_rate
            settings.deterministic_ragdolls = True
            settings.spectator_as_ego = False
            settings.tile_stream_distance = 650
            settings.actor_active_distance = 650
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.world.reset_all_traffic_lights()
            CarlaDataProvider.set_client(self.client)
            CarlaDataProvider.set_traffic_manager_port(args.traffic_manager_port)
            CarlaDataProvider.set_world(self.world)
            self.traffic_manager.set_random_device_seed(args.traffic_manager_seed)
            self.world.tick()
            if CarlaDataProvider.get_map().name.rsplit("/", 1)[-1] != town:
                raise RuntimeError(f"The published route requires map {town}")

        def _load_and_run_scenario(self, args: argparse.Namespace, config) -> bool:
            original_scenario = upstream.RouteScenario

            class CalibratedRouteScenario(original_scenario):
                def __init__(self, world: carla.World, config, debug_mode: int = 0) -> None:
                    super().__init__(world=world, config=config, debug_mode=debug_mode)
                    ego = self.ego_vehicles[0]
                    if options.controller != "original":  # TF++ PID was tuned on the default curve
                        flatten_steering_curve(ego, carla)
                    payload = json.loads(agent_config.read_text(encoding="utf-8"))
                    payload["geometry"] = vehicle_geometry(ego)
                    agent_config.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            # Calibration reads only fixed ego intrinsics; agent receives no actor handle.
            upstream.RouteScenario = CalibratedRouteScenario
            try:
                return super()._load_and_run_scenario(args, config)
            finally:
                upstream.RouteScenario = original_scenario

        def _register_statistics(self, route_index, entry_status, crash_message='') -> None:
            super()._register_statistics(route_index, entry_status, crash_message)
            if self.route_scenario is not None:
                events = [dict(criterion=node.name, frame=event.get_frame(),
                               event_type=event.get_type().name, message=event.get_message())
                          for node in self.route_scenario.get_criteria() for event in node.events]
                (options.out / 'evaluation_events.json').write_text(
                    json.dumps(events, indent=2), encoding='utf-8')

        def _cleanup(self) -> None:
            # Upstream's final sweep destroys unrelated sensors on a shared server.
            # AgentWrapper and CarlaDataProvider already own this evaluation's actors.
            try:
                CarlaDataProvider.cleanup()
                if self._agent_watchdog:
                    self._agent_watchdog.stop()
                if self.agent_instance:
                    self.agent_instance.destroy()
                    self.agent_instance = None
            finally:
                try:
                    if self.route_scenario:
                        self.route_scenario.remove_all_actors()
                        self.route_scenario = None
                        self.statistics_manager.remove_scenario()
                finally:
                    if self.manager:
                        self._client_timed_out = not self.manager.get_running_status()
                        self.manager.cleanup()

        def _reset_world_settings(self) -> None:
            try:
                self.traffic_manager.set_synchronous_mode(False)
                self.traffic_manager.set_hybrid_physics_mode(False)
            finally:
                self.world.apply_settings(self.original_settings)

    args = argparse.Namespace(host=options.host, port=options.port, timeout=options.timeout,
        traffic_manager_port=options.traffic_manager_port, traffic_manager_seed=options.seed,
        agent=str(ROOT / ("integration/tcp_agent.py" if options.planner == "tcp" else
                          "integration/pdm_agent.py" if options.planner == "pdm" else
                          "integration/tfpp_original_agent.py" if options.controller == "original" else
                          "integration/tfpp_agent.py")), agent_config=str(agent_config),
        debug=0, track="MAP" if options.planner == "pdm" else "SENSORS", record="", repetitions=1, routes_subset=options.route_id,
        routes=str(options.routes), resume=False,
        checkpoint=str(options.out / "results.json"), debug_checkpoint=str(options.out / "live_results.txt"))
    indexer = RouteIndexer(args.routes, 1, args.routes_subset)
    if indexer.total != 1:
        raise RuntimeError("Exactly one published route must be selected")
    statistics = StatisticsManager(args.checkpoint, args.debug_checkpoint)
    previous_signal = signal.getsignal(signal.SIGINT)
    evaluator = None
    try:
        evaluator = ExistingServerEvaluator(args, statistics)
        statistics.clear_records()
        statistics.save_progress(0, 1)
        statistics.write_statistics()
        crashed = evaluator._load_and_run_scenario(args, indexer.get_next_config())
        statistics.save_progress(indexer.index, indexer.total)
        statistics.write_statistics()
        if not crashed:
            statistics.compute_global_statistics()
        statistics.validate_and_write_statistics(evaluator.sensors_initialized, crashed)
        return 1 if crashed else 0
    finally:
        try:
            if evaluator is not None:
                try:
                    evaluator._cleanup()
                finally:
                    evaluator._reset_world_settings()
        finally:
            signal.signal(signal.SIGINT, previous_signal)
