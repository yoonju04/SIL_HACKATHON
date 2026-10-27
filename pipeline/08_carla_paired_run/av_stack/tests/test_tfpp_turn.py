import math
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import BicyclePlant, PlantParams
from integration.tfpp_reference import checkpoints_to_reference
from integration.tfpp_control import PredictionController, SensorState
from integration.fault_injection import FaultConfig


def test_mpc_tracks_a_moving_predicted_turn_on_independent_plant() -> None:
    parameters = PlantParams()
    geometry = asdict(parameters)
    geometry.pop('max_steer_rate_rad_s')
    plant = BicyclePlant(parameters)
    plant.s.vx = 4.
    controller = None
    for frame in range(160):
        state = plant.s
        angle = math.atan2(state.x, 12+state.y) + np.linspace(2.5, 11.5, 10)/12
        dx, dy = 12*np.sin(angle)-state.x, 12*np.cos(angle)-12-state.y
        points = np.column_stack((math.cos(state.yaw)*dx+math.sin(state.yaw)*dy,
                                  math.sin(state.yaw)*dx-math.cos(state.yaw)*dy))
        reference = checkpoints_to_reference(points, (state.x, -state.y, -state.yaw), 4., state.vx)
        if controller is None:
            controller = PredictionController('mpc', geometry, .05, reference, FaultConfig())
        command = controller.command(reference, 4., SensorState(state.x, state.y, state.yaw,
                                                               state.vx, state.r), frame*.05)
        plant.step(-command.steer*parameters.max_steer_rad, 0., .05)
    offset = abs(math.hypot(plant.s.x, plant.s.y+12)-12)
    assert offset < .1, offset
    print('PASS moving predicted turn; final radial error', offset)


if __name__ == '__main__':
    test_mpc_tracks_a_moving_predicted_turn_on_independent_plant()
