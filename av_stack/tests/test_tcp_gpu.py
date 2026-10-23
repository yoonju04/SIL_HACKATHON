from __future__ import annotations

import math
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.tcp_predictor import TCPOnlySensorAgent, tcp_waypoint_speed
from integration.tfpp_predictor import forbidden_controller
from integration.tfpp_bootstrap import PROJECT
import carla
import torch
from agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def test_temporal_waypoints_preserve_speed_units() -> None:
    points = np.array([[2., 0.], [4., 0.], [6., 0.], [8., 0.]])
    assert tcp_waypoint_speed(points) == 4.


def test_real_tcp_checkpoint_sensor_forward_without_world_or_control() -> None:
    with patch.object(carla, 'Client', forbidden_controller), \
         patch.object(CarlaDataProvider, 'get_world', forbidden_controller), \
         patch.object(CarlaDataProvider, 'get_map', forbidden_controller), \
         patch.object(CarlaDataProvider, 'get_hero_actor', forbidden_controller):
        agent = TCPOnlySensorAgent()
        gps = [({'lat': 0., 'lon': i*.001, 'z': 0.}, RoadOption.LANEFOLLOW) for i in range(4)]
        world = [(carla.Transform(carla.Location(x=i*111.319490793)), RoadOption.LANEFOLLOW) for i in range(4)]
        agent.set_global_plan(gps, world)
        agent.setup_model(PROJECT/'models/tcp_b2d.ckpt')
        image = np.zeros((900, 1600, 4), dtype=np.uint8)
        sensors = {name: (1, image) for name in ('rgb_front', 'rgb_front_left', 'rgb_front_right')}
        sensors.update(gps=(1, np.zeros(3)), imu=(1, np.array([0., 0., 0., 0., 0., 0., math.pi/2])),
                       speed=(1, {'speed': 0.}), lidar=(1, np.zeros((0, 4), dtype=np.float32)))
        with patch.object(agent.net, 'forward', forbidden_controller), \
             patch.object(agent.net, 'process_action', forbidden_controller), \
             patch.object(agent.net, 'control_pid', forbidden_controller):
            result = agent.predict(sensors, .05)
            durations = []
            for frame in range(5):
                started = perf_counter()
                result = agent.predict(sensors, .1+frame*.05)
                torch.cuda.synchronize()
                durations.append(perf_counter()-started)
        assert result.checkpoints.shape == (4, 2)
        assert np.isfinite(result.checkpoints).all()
        assert result.waypoint_interval == .5
        assert result.target_speed == tcp_waypoint_speed(result.checkpoints)
        assert np.allclose(result.pose, [0., 0., 0.], atol=1e-5)
        assert not hasattr(agent.net, 'speed_controller')
        assert agent.last_rgb_shape == (1, 3, 256, 900)
        print(f'Warm full sensor preprocessing + GPU prediction: mean={np.mean(durations):.4f}s max={max(durations):.4f}s')
        print('PASS official TCP checkpoint -> three-camera trajectory-only GPU inference; world, PID, full control forward blocked')
        agent.destroy()


if __name__ == '__main__':
    test_temporal_waypoints_preserve_speed_units()
    test_real_tcp_checkpoint_sensor_forward_without_world_or_control()
