"""Official Bench2Drive TCP camera preprocessing and trajectory head, without actuator heads."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import math
import sys

import cv2
import numpy as np
from scipy.optimize import fsolve

from integration.tfpp_bootstrap import PROJECT
from integration.tfpp_predictor import LearnedPrediction, SensorData, TrajectoryOnlySensorAgent, forbidden_controller

TCP_SOURCE = PROJECT/'third_party'/'bench2drive_tcp'
if str(TCP_SOURCE) not in sys.path:
    sys.path.insert(0, str(TCP_SOURCE))
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TV
import TCP.resnet as tcp_resnet

# The CC BY-NC-ND upstream source stays unmodified: ImageNet weights are skipped here
# at import time; the strict checkpoint load replaces every parameter afterwards.
_upstream_resnet34 = tcp_resnet.resnet34
tcp_resnet.resnet34 = lambda pretrained=False, **kwargs: _upstream_resnet34(pretrained=False, **kwargs)
from TCP.config import GlobalConfig
from TCP.model import TCP
from third_party.bench2drive_tcp.team_code.planner import RoutePlanner


def tcp_waypoint_speed(points: np.ndarray) -> float:
    """TCP targets are 0.5 seconds apart; auxiliary pred_speed estimates current speed."""
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).mean() / .5)


class TCPOnlySensorAgent(TrajectoryOnlySensorAgent):
    def setup_model(self, checkpoint: Path) -> None:
        self.net = TCP(GlobalConfig())
        weights = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
        self.net.load_state_dict({key.removeprefix('model.'): value for key, value in weights.items()}, strict=True)
        self.net.cuda().eval()
        del self.net.turn_controller
        del self.net.speed_controller
        self.net.control_pid = forbidden_controller
        self.net.process_action = forbidden_controller
        self.initialized = False
        self.state_log = deque(maxlen=2)
        self.lidar_buffer = deque(maxlen=1)
        self.nets = [self.net]

    def sensors(self):
        cameras = [('rgb_front', .8, 0., 0.), ('rgb_front_left', .27, -.55, -55.),
                   ('rgb_front_right', .27, .55, 55.)]
        result = [dict(type='sensor.camera.rgb', x=x, y=y, z=1.6, roll=0., pitch=0., yaw=yaw,
                       width=1600, height=900, fov=70, id=name) for name, x, y, yaw in cameras]
        result.extend([
            dict(type='sensor.other.imu', x=-1.4, y=0., z=0., roll=0., pitch=0., yaw=0., sensor_tick=.05, id='imu'),
            dict(type='sensor.other.gnss', x=-1.4, y=0., z=0., roll=0., pitch=0., yaw=0., sensor_tick=.01, id='gps'),
            dict(type='sensor.speedometer', reading_frequency=20, id='speed'),
            dict(type='sensor.lidar.ray_cast', x=0., y=0., z=2.5, roll=0., pitch=0., yaw=0., id='lidar'),
        ])
        return result

    def _init_route(self) -> None:
        location = self._global_plan_world_coord[0][0].location
        gps = self._global_plan[0][0]
        radius = 6378137.

        def equations(reference: np.ndarray) -> list[float]:
            lat, lon = reference
            scale = math.cos(math.radians(lat))
            x = scale * radius * math.radians(gps['lon'] - lon)
            y = scale * radius * (math.log(math.tan(math.radians(90 + lat)/2)) -
                                  math.log(math.tan(math.radians(90 + gps['lat'])/2)))
            return [x-location.x, y-location.y]

        self.lat_ref, self.lon_ref = fsolve(equations, [0., 0.])
        self._route_planner = RoutePlanner(4., 50., lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    @torch.inference_mode()
    def predict(self, input_data: SensorData, timestamp: float) -> LearnedPrediction:
        if not self.initialized:
            self._init_route()
        images = []
        for name in ('rgb_front_left', 'rgb_front', 'rgb_front_right'):
            image = cv2.cvtColor(input_data[name][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, compressed = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 20])
            image = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
            start = 200 if name != 'rgb_front_left' else 0
            stop = 1400 if name != 'rgb_front_right' else 1600
            images.append(image[:, start:stop])
        pixels = torch.from_numpy(np.concatenate(images, axis=1)).permute(2, 0, 1).unsqueeze(0).float()
        pixels = F.interpolate(pixels, size=(256, 900), mode='bilinear', align_corners=False)
        rgb = pixels.byte().float().div(255.).cuda()
        rgb = TV.normalize(rgb, mean=[.485, .456, .406], std=[.229, .224, .225])
        self.last_rgb_shape = tuple(rgb.shape)
        position = self._route_planner.gps_to_location(input_data['gps'][1][:2])
        compass = float(input_data['imu'][1][-1])
        if not math.isfinite(compass):
            compass = math.pi/2
        yaw = compass-math.pi/2
        speed = float(input_data['speed'][1]['speed'])
        self.state_log.append(np.array([*position, yaw, speed]))
        cloud = input_data['lidar'][1].copy()
        cloud[:, 2] += 2.5
        self.lidar_buffer.append(cloud)
        waypoint, command = self._route_planner.run_step(position)
        delta = waypoint-position
        rotation = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
        target = torch.tensor(rotation @ delta, dtype=torch.float32, device='cuda').reshape(1, 2)
        command_index = max(command.value, 1)-1 if command.value >= 0 else 3
        command_vector = F.one_hot(torch.tensor([command_index], device='cuda'), num_classes=6).float()
        state = torch.cat([torch.tensor([[speed/12]], device='cuda'), target, command_vector], dim=1)
        # This is exactly the original trajectory branch; action logits and actuator policies are never evaluated.
        feature, _ = self.net.perception(rgb)
        measurement = self.net.measurements(state)
        hidden = self.net.join_traj(torch.cat([feature, measurement], dim=1))
        point = torch.zeros((1, 2), device='cuda')
        points = []
        for _ in range(self.net.config.pred_len):
            hidden = self.net.decoder_traj(torch.cat([point, target], dim=1), hidden)
            point = point+self.net.output_traj(hidden)
            points.append(point)
        checkpoints = torch.stack(points, dim=1)[0].cpu().numpy()
        return LearnedPrediction(checkpoints, tcp_waypoint_speed(checkpoints),
                                 (float(position[0]), float(position[1]), yaw), timestamp, waypoint_interval=.5)

    def destroy(self) -> None:
        self.nets = []
        if hasattr(self, 'net'):
            del self.net
