"""Baseline only: unchanged TransFuser++ run_step with its own PID, stop-sign and creep logic.

Used to separate model/route failures from av_stack controller failures on the same
Bench2Drive route. No av_stack MPC/LQR/PI or LiDAR guard is involved in this mode.
"""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
from time import perf_counter

from integration.tfpp_bootstrap import prepare

prepare()
import carla
from sensor_agent import SensorAgent
from integration.tfpp_predictor import TrajectoryOnlySensorAgent, SensorData
from integration.benchmark_recording import FrontRecorder, SensorFrame


def get_entry_point() -> str:
    return 'TFPPOriginalAgent'


class TFPPOriginalAgent(TrajectoryOnlySensorAgent):
    def setup(self, path_to_conf_file: str, route_index=None, traffic_manager=None) -> None:
        config = json.loads(Path(path_to_conf_file.split('+')[0]).read_text(encoding='utf-8'))
        # Upstream evaluate_routes_slurm_tfpp.py settings; STOP_CONTROL keeps its upstream default (1).
        os.environ.pop('SAVE_PATH', None)
        os.environ.update(DIRECT='1', COMPILE='0', DEBUG_CHALLENGE='0', TUNED_AIM_DISTANCE='0',
                          SLOWER='0', UNCERTAINTY_WEIGHT='1', STOP_AFTER_METER='-1', STOP_CONTROL='1')
        SensorAgent.setup(self, config['model'])
        self.first_timestamp = None
        self.resources = ExitStack()
        out = Path(config['out'])
        out.mkdir(parents=True, exist_ok=True)
        self.control_log = self.resources.enter_context((out/'agent_controls.jsonl').open('w', encoding='utf-8'))
        self.recorder = FrontRecorder(out) if config.get('record_front', False) else None
        if self.recorder is not None:
            self.resources.callback(self.recorder.close)

    def run_step(self, input_data: SensorData, timestamp: float, sensors=None) -> carla.VehicleControl:
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        elapsed = timestamp - self.first_timestamp
        speed = float(input_data['speed'][1]['speed'])
        if self.recorder is not None:
            self.recorder.capture(SensorFrame(int(input_data['rgb_front'][0]), elapsed, speed, input_data['rgb_front'][1]))
        started = perf_counter()
        control = SensorAgent.run_step(self, input_data, timestamp, sensors)
        self.control_log.write(json.dumps(dict(
            t=elapsed, sensor_frame=int(input_data['rgb_front'][0]), throttle=float(control.throttle),
            brake=float(control.brake), steer=float(control.steer), source='original_tfpp',
            perception='TF++ RGB+LiDAR', original_control=True, measured_speed=speed,
            stuck_detector=int(self.stuck_detector), force_move=int(self.force_move),
            step_time_s=perf_counter()-started))+chr(10))
        self.control_log.flush()
        return control

    def destroy(self) -> None:
        if hasattr(self, 'resources'):
            self.resources.close()
        self.nets = []
