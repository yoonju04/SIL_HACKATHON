"""Bench2Drive route + PDM-Lite expert, with av_stack MPC/LQR as the only steering owner.

PDM-Lite (DriveLM, Sima et al. ECCV 2024; the expert that generated the CARLA Garage /
TransFuser++ training data) is used unchanged for route planning, hazard handling
(IDM car following, traffic lights, stop signs, pedestrians, scenario obstacles) and
throttle/brake. Only its lateral PID (_get_steer) is replaced: the dense map route it
plans each tick becomes the reference of our MPC / LQR / supervised switch.

PDM-Lite is privileged (simulator ground truth), so perception is out of scope here.
"""
from __future__ import annotations

from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from integration.tfpp_bootstrap import prepare

prepare()
import carla
import leaderboard.autoagents as _autoagents
import leaderboard.autoagents.autonomous_agent as _b2d_autonomous_agent

# PDM-Lite imports leaderboard_autopilot's autonomous_agent_local; Bench2Drive's base class
# provides the same interface (dense route via set_global_plan, run_step(input_data, timestamp)).
sys.modules.setdefault('leaderboard.autoagents.autonomous_agent_local', _b2d_autonomous_agent)
_autoagents.autonomous_agent_local = _b2d_autonomous_agent
os.environ.pop('SAVE_PATH', None)  # PDM-Lite writes datasets when SAVE_PATH is set
from autopilot import AutoPilot
import privileged_route_planner as _route_planner_module
from integration.tfpp_bootstrap import GARAGE

_upstream_speed_limits = _route_planner_module.PrivilegedRoutePlanner.compute_speed_limits


def _speed_limits_from_garage(self, carla_map):
    # Upstream loads 'team_code/speed_limits/<Town>_speed_limits.npy' relative to the carla_garage root.
    # '<Town>_Opt' (layered map) shares the speed-limit table of '<Town>'.
    name = carla_map.name.split('/')[-1]
    if name.endswith('_Opt') and not (GARAGE/'team_code'/'speed_limits'/f'{name}_speed_limits.npy').exists():
        class _BaseMapName:
            def __init__(self, inner):
                self.inner, self.name = inner, inner.name[:-len('_Opt')]

            def __getattr__(self, attribute):
                return getattr(self.inner, attribute)
        carla_map = _BaseMapName(carla_map)
    cwd = os.getcwd()
    os.chdir(GARAGE)
    try:
        return _upstream_speed_limits(self, carla_map)
    finally:
        os.chdir(cwd)


_route_planner_module.PrivilegedRoutePlanner.compute_speed_limits = _speed_limits_from_garage

from integration.behavior_control import update_reference
from integration.benchmark_recording import FrontRecorder, SensorFrame
from integration.carla_io import read_state, to_carla_steer, vehicle_geometry
from integration.episode import build_controller, step_controller
from integration.fault_injection import FaultConfig
from integration.chase_view import ChaseView
from integration.live_control import CapturingLongitudinal, ComfortPedals, LiveFaultCommands, SupervisorConsole
from integration.monitor_logs import MonitorLogs
from srunner.scenariomanager.timer import GameTime
from integration.supervised_lateral import SupervisedLateral
from integration.unprotected_lateral import UnprotectedMPC
from lqr_controller import LQRConfig
from planner import Trajectory
from planner.cubic_spline import CubicSpline2D

REFERENCE_AHEAD_M = 80.0
REFERENCE_BEHIND_M = 3.0
# LQR has no preview; PDM-Lite turns junctions at 8-11 m/s, so the curvature feed-forward is read
# 0.2 s ahead (experiments/pdm_turn_sweep.py: max path error 0.26-0.70 m -> 0.08-0.12 m on R=6-20 m
# turns). Lateral weight 0.05 -> 0.5 (Apollo default favours heading; the fallback felt sluggish):
# worst takeover error 0.11 -> 0.07 m on R=8-80 m; 1.0 starts to oscillate at R=12 m.
LQR_CONFIG = LQRConfig(q=(0.5, 0.0, 1.0, 0.0), preview_time_s=.2)
TAKEOVER_RATE_DPS = 90.0  # LQR steering-rate limit after takeover; SR2 bound (road wheel)


def get_entry_point() -> str:
    return 'PDMLiteOwnSteerAgent'


def route_to_reference(route_xy: np.ndarray, speed: float, behind: int = 0) -> Trajectory | None:
    """PDM-Lite route points (CARLA frame, 10 per metre) -> right-handed reference.

    route_xy[behind] is the ego's matched route point; the points before it are the real
    route behind the vehicle. A straight stub is only added when none exist (route start):
    joining a straight stub to a curving route makes the spline wiggle right at the ego
    (seen as a wrong-signed curvature mid-turn).
    """
    route_xy = np.asarray(route_xy, float)
    start = behind % 10
    points = route_xy[start:behind + int(REFERENCE_AHEAD_M * 10):10, :2]  # 1 m spacing
    if len(points) - (behind - start) // 10 < 4:
        return None
    points = np.column_stack((points[:, 0], -points[:, 1]))  # left-handed CARLA -> right-handed
    if behind < 10 * REFERENCE_BEHIND_M:
        direction = points[1] - points[0]
        direction /= max(np.linalg.norm(direction), 1e-6)
        stub = points[0] - np.outer(np.arange(int(REFERENCE_BEHIND_M), 0, -1), direction)
        points = np.vstack((stub, points))
    spline = CubicSpline2D(points[:, 0], points[:, 1])
    s = np.linspace(0, spline.length, math.ceil(spline.length / .25) + 1)
    x, y = spline.position(s)
    v = np.full_like(s, max(speed, .5))
    return Trajectory(s, x, y, np.unwrap(spline.yaw(s)), spline.curvature(s), v, s / v[0],
                      x.copy(), y.copy(), dict(source='PDM-Lite privileged route'))


class PDMLiteOwnSteerAgent(AutoPilot):
    def setup(self, path_to_conf_file: str, route_index=None, traffic_manager=None) -> None:
        self.options = json.loads(Path(path_to_conf_file.split('+')[0]).read_text(encoding='utf-8'))
        super().setup(path_to_conf_file, route_index, traffic_manager)
        self.config.steer_noise = 0.  # PDM-Lite adds 1e-3 noise for data diversity; not wanted here
        self.dt = float(self.options.get('dt', .05))
        self.fault = FaultConfig(self.options['mpc_fault'], float(self.options['fault_start']),
                                 float(self.options['fault_duration']),
                                 float(self.options.get('fault_magnitude', .05)))
        self.mode = self.options['controller']
        self.lateral = None
        self.reference = None
        self.first_timestamp = None
        self.elapsed = 0.
        self.last_lateral = None
        self.resources = ExitStack()
        out = Path(self.options['out'])
        out.mkdir(parents=True, exist_ok=True)
        self.control_log = self.resources.enter_context((out/'agent_controls.jsonl').open('w', encoding='utf-8'))
        self.recorder = FrontRecorder(out) if self.options.get('record_front', False) else None
        if self.recorder is not None:
            self.resources.callback(self.recorder.close)
        self.max_speed = float(self.options.get('max_speed_kmh', 40.)) / 3.6
        self.comfort = ComfortPedals(max_speed=self.max_speed, dt=self.dt) if self.options.get('comfort', True) else None
        self.live = LiveFaultCommands()
        self.console = SupervisorConsole(float(self.options.get('monitor_period', 0.)))
        print(f'[av_stack] PDM-Lite + {self.mode} steering | max speed {self.max_speed * 3.6:.0f} km/h | '
              f'comfort pedals {"on" if self.comfort else "off"} | '
              f'traffic lights {"green for the ego" if self.options.get("green_lights") else "normal"}',
              flush=True)

    def _init(self, hd_map):
        super()._init(hd_map)
        self._longitudinal_controller = CapturingLongitudinal(self._longitudinal_controller, self.max_speed)
        self.green_lights = bool(self.options.get('green_lights', False))
        self._hold_lights_green()
        self.view = None
        if self.options.get('view', True):
            video = Path(self.options['out']) / 'view.mp4' if self.options.get('record_front', False) else None
            self.view = ChaseView(carla, self._world, self._vehicle, video)
            self.resources.callback(self.view.close)
        self.monitor_logs = None
        if self.options.get('monitor_logs', True):
            # Intended lane for the 09 Monitor: the route before PDM-Lite's obstacle shifts, 1 m spacing.
            route = np.array(self._waypoint_planner.original_route_points[::10, :2], float)
            self.monitor_logs = MonitorLogs(carla, self._world, self._vehicle, route, Path(self.options['out']),
                                            meta=dict(planner='PDM-Lite', controller=self.mode,
                                                      fault_mode=self.fault.mode, fault_start=self.fault.at_s,
                                                      fault_magnitude=self.fault.magnitude))
            self.resources.callback(self.monitor_logs.close)

    def _hold_lights_green(self) -> None:
        """--green-lights: the light the ego approaches turns green, its junction's others red.

        All lights green at once gridlocked the junctions, so only the ego's approach is
        switched (within 40 m on the planned route); other traffic keeps obeying the cycle,
        which resumes once the ego has passed. Also undoes a freeze left by an older run.
        """
        if not getattr(self, '_lights_unfrozen', False):
            self._world.freeze_all_traffic_lights(False)
            self._lights_unfrozen = True
        if not self.green_lights:
            return
        planner = self._waypoint_planner
        index = min(planner.route_index, len(planner.next_traffic_lights) - 1)
        light = planner.next_traffic_lights[index]
        if light is None or planner.distances_to_next_traffic_lights[index] > 40.                 or light.get_state() == carla.TrafficLightState.Green:
            return
        for other in light.get_group_traffic_lights():
            if other.id != light.id:
                other.set_state(carla.TrafficLightState.Red)
        light.set_state(carla.TrafficLightState.Green)

    def __call__(self):
        # Upstream prints a wall-clock line every tick; keep the console for the supervisor monitor.
        input_data = self.sensor_interface.get_data(GameTime.get_frame())
        control = self.run_step(input_data, GameTime.get_time())
        control.manual_gear_shift = False
        return control

    def sensors(self):
        specs = super().sensors()
        if self.options.get('record_front', False):
            # Behind-the-roof view for the demo video (leaderboard limits sensors to a 3 m radius).
            specs.append(dict(type='sensor.camera.rgb', x=-2., y=0., z=2.2, roll=0., pitch=-12., yaw=0.,
                              width=1280, height=720, fov=100, id='rgb_front'))
        return specs

    def _get_steer(self, route_points, current_position, current_heading, current_speed):
        # PDM-Lite's save() logs the aim point its PID used; keep it filled (not used for control).
        self.aim_wp = route_points[min(int(10 * max(current_speed, 3.)), len(route_points) - 1)]
        self.angle = self._get_angle_to(current_position, current_heading, self.aim_wp) / 90
        x, y, yaw, vx, vy, yaw_rate = read_state(self._vehicle)
        # route_points starts at the matched index; take the planner's real route behind it too.
        planner = self._waypoint_planner
        behind = min(planner.route_index, int(10 * REFERENCE_BEHIND_M))
        route = planner.route_points[planner.route_index - behind:]
        reference = route_to_reference(route, vx, behind)
        if reference is not None:
            self.reference = reference
        if self.reference is None:
            return 0.
        if self.lateral is None:
            geometry = vehicle_geometry(self._vehicle)
            self.max_steer = geometry['max_steer_rad']
            if self.mode == 'supervised':
                self.lateral = SupervisedLateral(geometry, self.dt, self.reference, self.fault,
                                                 lqr_config=LQR_CONFIG)
                self.lateral.rate_dps = TAKEOVER_RATE_DPS
            elif self.mode == 'mpc_unprotected':
                self.lateral = UnprotectedMPC(geometry, self.dt, self.reference, self.fault)
            else:
                self.lateral = build_controller(self.mode, geometry, self.dt, self.reference,
                                                lqr_cfg=LQR_CONFIG if self.mode == 'lqr' else None)
        speed_cap = max(vx, .5)
        if self.mode in ('supervised', 'mpc_unprotected'):
            self.lateral.set_reference(self.reference, speed_cap)
            result = self.lateral.step(self.elapsed, x, y, yaw, vx, vy, yaw_rate)
        else:
            update_reference(self.lateral, self.reference, speed_cap)
            result = step_controller(self.mode, self.lateral, self.elapsed, x, y, yaw, vx, vy, yaw_rate)
        self.last_lateral = result
        return to_carla_steer(result.steer or 0., self.max_steer)

    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        self.elapsed = timestamp - self.first_timestamp
        speed = float(input_data['speed'][1]['speed'])
        if self.recorder is not None and 'rgb_front' in input_data:
            self.recorder.capture(SensorFrame(int(input_data['rgb_front'][0]), self.elapsed, speed,
                                              input_data['rgb_front'][1]))
        started = perf_counter()
        if getattr(self, 'green_lights', False):
            self._hold_lights_green()
        command = self.live.poll(self.elapsed)
        if command is not None:
            if self.mode in ('supervised', 'mpc_unprotected'):
                self.fault = command
                if self.lateral is not None:
                    self.lateral.injector.config = command
                self.console.injected(self.elapsed, command)
            else:
                print('[inject] ignored: faults need --controller supervised or mpc_unprotected', flush=True)
        control = super().run_step(input_data, timestamp, sensors, plant)
        pdm_pedals = (float(control.throttle), float(control.brake))
        if self.comfort is not None and getattr(self, '_longitudinal_controller', None) is not None:
            hazard, target, _ = self._longitudinal_controller.last
            blocked = self.ego_blocked_for_ticks >= self.config.max_blocked_ticks  # PDM-Lite's unstick throttle
            if not blocked:
                control.throttle, control.brake = self.comfort(hazard or control.brake >= 1. and target < .05,
                                                               target, speed, float(control.throttle))
                if control.throttle == 0 and speed < self.config.minimum_speed_to_prevent_rolling_back:
                    control.brake = 1.
        safe_stop = bool(getattr(self.lateral, 'stop_required', False))
        if safe_stop:
            control.throttle, control.brake = 0., 1.
        result = self.last_lateral
        active = getattr(self.lateral, 'active', self.mode)
        self.console.update(self.elapsed, active, self.lateral.record() if hasattr(self.lateral, 'record') else {},
                            None if result is None else result.e_y, None if result is None else result.e_psi, speed)
        if getattr(self, 'view', None) is not None:
            self.view.show(active)
        record = dict(t=self.elapsed, steer=float(control.steer), throttle=float(control.throttle),
                      brake=float(control.brake), measured_speed=speed, planner='PDM-Lite',
                      source=getattr(self.lateral, 'active', self.mode),
                      tracking_error=None if result is None else float(result.e_y),
                      heading_error=None if result is None else float(result.e_psi),
                      lateral_status=None if result is None else result.status,
                      fault_injected=self.fault.active(self.elapsed), safe_stop=safe_stop,
                      fault_mode=self.fault.mode, fault_magnitude=self.fault.magnitude,
                      pdm_throttle=pdm_pedals[0], pdm_brake=pdm_pedals[1],
                      vehicle_hazard=bool(self.vehicle_hazard), walker_hazard=bool(self.walker_hazard),
                      traffic_light_hazard=bool(self.traffic_light_hazard), junction=bool(self.junction),
                      step_time_s=perf_counter()-started)
        if hasattr(self.lateral, 'record'):
            record.update(self.lateral.record())
        self.control_log.write(json.dumps(record)+chr(10))
        self.control_log.flush()
        if getattr(self, 'monitor_logs', None) is not None and result is not None:
            _, _, _, vx, vy, yaw_rate = read_state(self._vehicle)
            self.monitor_logs.record(self.elapsed, vx, vy, yaw_rate, float(result.e_y), float(result.e_psi),
                                     active, getattr(self.lateral, 'reason', None),
                                     self.fault.active(self.elapsed), safe_stop)
        stop_after = self.options.get('stop_after_fault')
        self.stop_requested = bool(stop_after is not None and self.fault.config_active_since(self.elapsed, stop_after))
        return control

    def destroy(self, results=None) -> None:
        if hasattr(self, 'resources'):
            self.resources.close()
        super().destroy(results)
