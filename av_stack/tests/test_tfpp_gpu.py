from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from tempfile import TemporaryDirectory
from dataclasses import asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.tfpp_bootstrap import PROJECT
from integration.tfpp_predictor import TrajectoryOnlySensorAgent, forbidden_controller
import carla
import torch
from agents.navigation.local_planner import RoadOption
from sensor_agent import SensorAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from integration.tfpp_agent import TFPPAgent
from sim.plant import PlantParams


def test_real_pretrained_gpu_predictions_without_world_access_or_original_control() -> None:
    assert torch.cuda.is_available()
    carla.Client = forbidden_controller
    for name in ('get_world', 'get_map', 'get_hero_actor', 'get_all_actors'):
        setattr(CarlaDataProvider, name, forbidden_controller)
    SensorAgent.run_step = forbidden_controller
    predictor = TrajectoryOnlySensorAgent()
    predictor.setup_model(PROJECT/'models/transfuserpp/pretrained_models/all_towns')
    gps = [({'lat': 0., 'lon': i*.001, 'z': 0.}, RoadOption.LANEFOLLOW) for i in range(4)]
    world = [(carla.Transform(carla.Location(x=i*111.319)), RoadOption.LANEFOLLOW) for i in range(4)]
    predictor.set_global_plan(gps, world)
    for net in predictor.nets:
        assert not hasattr(net, 'lateral_pid_controller')
        assert not hasattr(net, 'speed_controller')
        try:
            net.control_pid_direct(None, None, None)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Original PID entry point was not disabled.')
    images = np.zeros((512, 1024, 4), dtype=np.uint8)
    lidar = np.zeros((100, 4), dtype=np.float32)
    lidar[:, 0] = np.linspace(1, 30, 100)
    started = time.perf_counter()
    prediction = None
    for frame in range(7):
        sensors = {'rgb_front': (frame, images), 'gps': (frame, np.zeros(3)),
                   'imu': (frame, np.array([0, 0, 0, 0, 0, 0, math.pi/2])),
                   'speed': (frame, {'speed': 0.}), 'lidar': (frame, lidar)}
        prediction = predictor.predict(sensors, frame*.05)
    torch.cuda.synchronize()
    assert prediction is not None
    assert prediction.checkpoints.shape == (10, 2)
    assert np.isfinite(prediction.checkpoints).all()
    assert math.isfinite(prediction.target_speed) and prediction.target_speed >= 0
    result = dict(device=torch.cuda.get_device_name(), checkpoint='all_towns/model_0030_0.pth',
                  checkpoint_shape=list(prediction.checkpoints.shape), target_speed=prediction.target_speed,
                  original_controllers='disabled', ground_truth_access='blocked by test tripwires',
                  source='synthetic sensor smoke test; not a live driving score', elapsed_s=time.perf_counter()-started)
    output = Path('C:/Users/dongh/Documents/Codex/2026-10-22/new-chat/outputs/tfpp_gpu_smoke.json')
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('PASS pretrained TF++ GPU inference, no world access, original PID blocked')
    print(json.dumps(result, indent=2))
    predictor.destroy()


def test_sensor_prediction_drives_own_controller_and_fault_switch_only() -> None:
    agent = TFPPAgent()
    geometry = asdict(PlantParams())
    geometry.pop('max_steer_rate_rad_s')
    with TemporaryDirectory(prefix='tfpp-control-') as temporary:
        output = Path(temporary)
        config = dict(model=str(PROJECT/'models/transfuserpp/pretrained_models/all_towns'),
                      controller='supervised', mpc_fault='solver_fail', fault_start=.35,
                      fault_duration=2, out=str(output), geometry=geometry)
        config_path = output/'config.json'
        config_path.write_text(json.dumps(config), encoding='utf-8')
        agent.setup(str(config_path))
        gps = [({'lat': 0., 'lon': i*.001, 'z': 0.}, RoadOption.LANEFOLLOW) for i in range(4)]
        world = [(carla.Transform(carla.Location(x=i*111.319)), RoadOption.LANEFOLLOW) for i in range(4)]
        agent.set_global_plan(gps, world)
        for frame in range(10):
            sensors = {'rgb_front': (frame, np.zeros((512, 1024, 4), dtype=np.uint8)),
                       'gps': (frame, np.zeros(3)),
                       'imu': (frame, np.array([0, 0, 0, 0, 0, 0, math.pi/2])),
                       'speed': (frame, {'speed': 0.}),
                       'lidar': (frame, np.zeros((100, 4), dtype=np.float32))}
            command = agent.run_step(sensors, frame*.05)
            assert math.isfinite(command.steer) and -1 <= command.steer <= 1
        agent.destroy()
        rows = [json.loads(row) for row in (output/'agent_controls.jsonl').read_text().splitlines()]
        assert any(row.get('source') == 'mpc' for row in rows)
        assert any(row.get('source') == 'lqr' for row in rows)
        assert all(row.get('original_control') is False for row in rows)
        assert rows[-1]['switch_count'] == 1
        assert rows[-1]['supervisor_reason'] == 'solver_failure'
        print('PASS actual pretrained sensors -> path/speed -> own MPC -> fault -> own LQR')


if __name__ == '__main__':
    test_real_pretrained_gpu_predictions_without_world_access_or_original_control()
    test_sensor_prediction_drives_own_controller_and_fault_switch_only()
