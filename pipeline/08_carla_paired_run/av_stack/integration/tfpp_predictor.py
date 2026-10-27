"""Sensor preprocessing and pretrained forward pass only; upstream actuation is disabled."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Mapping
import os

import numpy as np

from integration.tfpp_bootstrap import prepare

prepare()
import carla
import torch
import torch.nn.functional as F
from sensor_agent import SensorAgent
from leaderboard.autoagents.autonomous_agent import Track
from leaderboard.envs.sensor_interface import SensorInterface

SensorData = Mapping[str, tuple[int, np.ndarray | Mapping[str, float]]]


@dataclass(frozen=True)
class LearnedPrediction:
    checkpoints: np.ndarray
    target_speed: float
    pose: tuple[float, float, float]
    timestamp: float
    waypoint_interval: float | None = None


def forbidden_controller(*args, **kwargs) -> None:
    raise RuntimeError('Original TransFuser++ controller is disabled; use av_stack MPC/LQR/PI.')


class TrajectoryOnlySensorAgent(SensorAgent):
    def __init__(self, carla_host: str = 'localhost', carla_port: int = 2000, debug: bool = False) -> None:
        self.track = Track.SENSORS
        self._global_plan = None
        self._global_plan_world_coord = None
        self.sensor_interface = SensorInterface()
        self.wallclock_t0 = None

    def setup_model(self, model_folder: Path) -> None:
        os.environ.pop('SAVE_PATH', None)
        os.environ.update(STOP_CONTROL='0', DEBUG_CHALLENGE='0', COMPILE='0', DIRECT='1',
                          HISTOGRAM='0', STOP_AFTER_METER='-1', SLOWER='0')
        SensorAgent.setup(self, str(model_folder))
        if not self.config.use_controller_input_prediction:
            raise ValueError('Checkpoint lacks path and target-speed prediction heads.')
        for net in self.nets:
            for name in ('turn_controller', 'speed_controller', 'turn_controller_direct',
                         'speed_controller_direct', 'lateral_pid_controller'):
                delattr(net, name)
            net.control_pid = forbidden_controller
            net.control_pid_direct = forbidden_controller
        self.control = carla.VehicleControl(brake=1)

    def run_step(self, input_data: SensorData, timestamp: float):
        forbidden_controller()

    @torch.inference_mode()
    def predict(self, input_data: SensorData, timestamp: float) -> LearnedPrediction | None:
        self.step += 1
        first = not self.initialized
        if first:
            self._init()
        data = self.tick(input_data)
        current = deepcopy(data['lidar'])
        if first:
            self.lidar_last = current
            return None
        previous_pose, pose = self.state_log[-2], self.state_log[-1]
        previous = self.align_lidar(self.lidar_last, *previous_pose[:3], *pose[:3])
        self.lidar_buffer.append(np.concatenate((current, previous), axis=0))
        self.lidar_last = current
        if len(self.lidar_buffer) < self.config.lidar_seq_len*self.config.data_save_freq:
            return None
        if self.config.lidar_seq_len != 1:
            raise ValueError('This adapter requires the published single-frame TF++ checkpoint.')
        histogram = self.data.lidar_to_histogram_features(self.lidar_buffer[-1],
                                                       use_ground_plane=self.config.use_ground_plane)
        lidar = torch.from_numpy(histogram).unsqueeze(0).to(self.device, dtype=torch.float32)
        probabilities, paths = [], []
        for net in self.nets:
            result = net.forward(rgb=data['rgb'], lidar_bev=lidar, target_point=data['target_point'],
                                 target_point_next=data.get('target_point_next'), ego_vel=data['speed'].reshape(1, 1),
                                 command=data['command'])
            probabilities.append(F.softmax(result[1][0], dim=0))
            paths.append(result[2][0])
        probability = torch.stack(probabilities).mean(0).cpu().numpy()
        speed = (self.inference_target_speeds[0] if probability[0] > self.config.brake_uncertainty_threshold
                 else float(probability @ np.asarray(self.inference_target_speeds)))
        points = torch.stack(paths).mean(0).cpu().numpy().copy()
        return LearnedPrediction(points, float(speed), tuple(float(value) for value in pose[:3]), timestamp)

    def destroy(self) -> None:
        self.nets = []
