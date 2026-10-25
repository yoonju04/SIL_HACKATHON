"""Bench2Drive sensor agent whose only actuator owners are av_stack MPC/LQR/PI."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from integration.tfpp_bootstrap import prepare

prepare()
import carla
from integration.fault_injection import FaultConfig
from integration.tfpp_predictor import TrajectoryOnlySensorAgent, SensorData
from integration.tfpp_reference import checkpoints_to_reference
from integration.tfpp_control import PredictionController, SensorState
from integration.tfpp_lidar_guard import lidar_guard
from integration.benchmark_recording import FrontRecorder, SensorFrame
from integration.live_control import LiveFaultCommands, SupervisorConsole
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime


def get_entry_point() -> str:
    return 'TFPPAgent'


class TFPPAgent(TrajectoryOnlySensorAgent):
    def setup(self, path_to_conf_file: str, route_index=None, traffic_manager=None) -> None:
        config = json.loads(Path(path_to_conf_file.split('+')[0]).read_text(encoding='utf-8'))
        self.options = config
        self.setup_model(Path(config['model']))
        self.dt = float(config.get('dt', .05))
        self.fault = FaultConfig(config['mpc_fault'], float(config['fault_start']), float(config['fault_duration']),
                                 float(config.get('fault_magnitude', .05)))
        self.own_controller = None
        self.first_timestamp = None
        self.resources = ExitStack()
        out = Path(config['out'])
        out.mkdir(parents=True, exist_ok=True)
        self.control_log = self.resources.enter_context((out/'agent_controls.jsonl').open('w', encoding='utf-8'))
        self.plan_log = self.resources.enter_context((out/'agent_plans.jsonl').open('w', encoding='utf-8'))
        self.recorder = FrontRecorder(out) if config.get('record_front', False) else None
        if self.recorder is not None:
            self.resources.callback(self.recorder.close)
        self.log_count = 0
        self.live = LiveFaultCommands()
        self.console = SupervisorConsole(float(config.get('monitor_period', 0.)))
        self.view = None
        print(f'[av_stack] {config.get("planner", "transfuserpp")} (camera+LiDAR) + {config["controller"]} steering | '
              f'traffic lights {"green for the ego" if config.get("green_lights") else "normal"}', flush=True)

    def __call__(self):
        # Upstream prints a wall-clock line every tick; keep the console for the supervisor banners.
        control = self.run_step(self.sensor_interface.get_data(GameTime.get_frame()), GameTime.get_time())
        control.manual_gear_shift = False
        return control

    def _demo_hooks(self, elapsed: float) -> None:
        """Live fault commands, chase-camera window and --green-lights.

        The simulator handle is used only for display and signal control, never as model input.
        """
        command = self.live.poll(elapsed)
        if command is not None:
            if self.options['controller'] in ('supervised', 'mpc_unprotected'):
                self.fault = command
                managed = None if self.own_controller is None else self.own_controller.managed
                if managed is not None:
                    managed.injector.config = command
                self.console.injected(elapsed, command)
            else:
                print('[inject] ignored: faults need --controller supervised or mpc_unprotected', flush=True)
        # Only when explicitly enabled (run_benchmark sets view=True unless --no-view): the sensor
        # agent otherwise never touches a simulator handle (tests/test_tfpp_gpu.py tripwire).
        if not (self.options.get('view', False) or self.options.get('green_lights', False)):
            return
        hero = CarlaDataProvider.get_hero_actor()
        if hero is None:
            return
        if self.view is None and self.options.get('view', False):
            from integration.chase_view import ChaseView
            video = Path(self.options['out']) / 'view.mp4' if self.options.get('record_front', False) else None
            self.view = ChaseView(carla, CarlaDataProvider.get_world(), hero, video)
            self.resources.callback(self.view.close)
        if self.options.get('green_lights') and hero.is_at_traffic_light():
            light = hero.get_traffic_light()
            if light is not None and light.get_state() != carla.TrafficLightState.Green:
                for other in light.get_group_traffic_lights():
                    if other.id != light.id:
                        other.set_state(carla.TrafficLightState.Red)
                light.set_state(carla.TrafficLightState.Green)

    def run_step(self, input_data: SensorData, timestamp: float, sensors=None) -> carla.VehicleControl:
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        elapsed = timestamp - self.first_timestamp
        if self.recorder is not None:
            self.recorder.capture(SensorFrame(int(input_data['rgb_front'][0]), elapsed,
                                             float(input_data['speed'][1]['speed']), input_data['rgb_front'][1]))
        self._demo_hooks(elapsed)
        prediction_started = perf_counter()
        prediction = self.predict(input_data, timestamp)
        prediction_seconds = perf_counter() - prediction_started
        if prediction is None:
            self.control = carla.VehicleControl(brake=1)
            return self.control
        state = SensorState(prediction.pose[0], -prediction.pose[1], -prediction.pose[2],
                            float(input_data['speed'][1]['speed']), -float(input_data['imu'][1][5]))
        # 'model': TF++ target speed as is (original; tailgates, rear-ended on route 26406).
        # 'follow' (default): + LiDAR gap keeping. 'guarded': + curvature cap (hesitant in merges).
        policy = self.options.get('speed_policy', 'follow')
        guarded = policy == 'guarded'
        guard = lidar_guard(self.lidar_buffer[-1], state.vx)
        desired_speed = prediction.target_speed
        if policy != 'model' and guard.speed_cap is not None:
            desired_speed = min(desired_speed, guard.speed_cap)
        try:
            reference = checkpoints_to_reference(prediction.checkpoints, prediction.pose, desired_speed,
                                                 state.vx, self.options.get('mpc_extension', 'linear'))
            lqr_reference = checkpoints_to_reference(prediction.checkpoints, prediction.pose,
                                                     desired_speed, state.vx, 'quadratic')
        except ValueError as error:
            self.control = carla.VehicleControl(brake=1)
            self.control_log.write(json.dumps(dict(t=elapsed, source='model_plan_invalid', error=str(error)))+chr(10))
            self.control_log.flush()
            return self.control
        if self.own_controller is None:
            self.own_controller = PredictionController(self.options['controller'], self.options['geometry'],
                                                       self.dt, reference, self.fault, self.throttle_law(),
                                                       curvature_speed_cap=guarded)
        control_started = perf_counter()
        command = self.own_controller.command(reference, desired_speed, state, elapsed, lqr_reference)
        control_seconds = perf_counter() - control_started
        self.control = carla.VehicleControl(throttle=command.throttle, brake=command.brake, steer=command.steer)
        record = dict(t=elapsed, sensor_frame=int(input_data['rgb_front'][0]), **asdict(command), perception=self.options.get('planner', 'transfuserpp'), original_control=False,
                      fault_injected=self.fault.active(elapsed), measured_speed=state.vx,
                      yaw_rate=state.yaw_rate, prediction_time_s=prediction_seconds,
                      control_time_s=control_seconds, model_target_speed=prediction.target_speed,
                      lidar_brake=guard.brake, lidar_clearance_m=guard.clearance_m)
        if self.own_controller.supervised is not None:
            record.update(self.own_controller.supervised.record())
        if self.own_controller.unprotected is not None:
            record.update(self.own_controller.unprotected.record())
        managed = self.own_controller.managed
        self.console.update(elapsed, command.source, managed.record() if managed is not None else {},
                            command.tracking_error, None, state.vx)
        if self.view is not None:
            self.view.show(command.source)
        self.control_log.write(json.dumps(record, allow_nan=False)+chr(10))
        self.plan_log.write(json.dumps(dict(t=elapsed, capture_pose=prediction.pose,
                                           checkpoints=prediction.checkpoints.tolist(), target_speed=prediction.target_speed,
                                           waypoint_interval=prediction.waypoint_interval))+chr(10))
        self.control_log.flush()
        self.plan_log.flush()
        self.log_count += 1
        stop_after = self.options.get('stop_after_fault')
        self.stop_requested = bool(stop_after is not None and self.fault.config_active_since(elapsed, stop_after))
        return self.control

    def throttle_law(self):
        """TF++'s own longitudinal law (nav_planner.get_throttle); steering stays with MPC/LQR.

        The model's target speed depends on the ego speed it observes, so a slower PI than the
        one it was evaluated with feeds back into lower predicted speeds."""
        config = getattr(self, 'config', None)
        if self.options.get('longitudinal', 'tfpp') != 'tfpp' or not hasattr(config, 'longitudinal_params'):
            return None
        from nav_planner import get_throttle

        def law(target: float, speed: float, stop: bool) -> tuple[float, float]:
            if stop or target < .01:
                return 0., 1.
            if speed / target > config.brake_ratio:
                # Graded instead of TF++'s full brake: a small overspeed (e.g. a lowered gap cap)
                # must not stop the car mid-manoeuvre. 1 m/s over target -> brake 0.3.
                return 0., float(min(1., .3 * (speed - target)))
            throttle, control_brake = get_throttle(config, False, target, speed)
            return float(min(max(throttle, 0.), config.clip_throttle)), float(bool(control_brake))
        return law

    def destroy(self) -> None:
        if hasattr(self, 'resources'):
            self.resources.close()
        self.nets = []
