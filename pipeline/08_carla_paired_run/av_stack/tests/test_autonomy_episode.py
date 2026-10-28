"""Native-map closed-loop tests; run in CARLA Python without a server."""
import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from behavior_fixture import Actor, World, carla
from integration.autonomy_episode import EpisodeSettings, run_episode
from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
from sim.plant import BicyclePlant, PlantParams


class TickWorld(World):
    """Advance only local physics and release a hazard after two simulated seconds."""
    def __init__(self) -> None:
        super().__init__()
        self.ticks = 0
        self.vehicle: Optional['PlantVehicle'] = None
        self.hazard: Optional[Actor] = None

    def tick(self) -> int:
        assert self.vehicle is not None
        self.vehicle.advance(0.05)
        self.ticks += 1
        if self.ticks == 40 and self.hazard is not None:
            self.hazard.state = carla.TrafficLightState.Green
            self.hazard.x = 400.0
        return self.ticks


class PlantVehicle(Actor):
    """CARLA actuator/state boundary backed by the existing nonlinear bicycle plant."""
    def __init__(self, world: TickWorld) -> None:
        super().__init__(world, 'vehicle.ego', 20.0)
        self.tick_world = world
        self.params = PlantParams()
        self.plant = BicyclePlant(self.params)
        self.plant.s.x = self.x
        self.plant.s.y = -1.75
        self.control = carla.VehicleControl()
        world.vehicle = self

    def get_world(self) -> TickWorld:
        return self.tick_world

    def get_location(self) -> carla.Location:
        state = self.plant.s
        return carla.Location(state.x, -state.y, 0.0)

    def get_transform(self) -> carla.Transform:
        return carla.Transform(self.get_location(), carla.Rotation(yaw=-math.degrees(self.plant.s.yaw)))

    def get_velocity(self) -> carla.Vector3D:
        state = self.plant.s
        vx = state.vx * math.cos(state.yaw) - state.vy * math.sin(state.yaw)
        vy = state.vx * math.sin(state.yaw) + state.vy * math.cos(state.yaw)
        return carla.Vector3D(vx, -vy, 0.0)

    def get_angular_velocity(self) -> carla.Vector3D:
        return carla.Vector3D(0.0, 0.0, -math.degrees(self.plant.s.r))

    def get_control(self) -> carla.VehicleControl:
        return self.control

    def get_physics_control(self) -> carla.VehiclePhysicsControl:
        physics = carla.VehiclePhysicsControl()
        physics.mass = self.params.mass
        physics.center_of_mass = carla.Vector3D()
        transform = self.get_transform()
        forward = transform.get_forward_vector()
        wheels = []
        for distance in (self.params.lf, self.params.lf, -self.params.lr, -self.params.lr):
            wheel = carla.WheelPhysicsControl()
            wheel.max_steer_angle = math.degrees(self.params.max_steer_rad) if distance > 0 else 0.0
            wheel.position = carla.Vector3D(
                (transform.location.x + distance * forward.x) * 100.0,
                (transform.location.y + distance * forward.y) * 100.0, 0.0)
            wheels.append(wheel)
        physics.wheels = wheels
        return physics

    def apply_control(self, control: carla.VehicleControl) -> None:
        assert -1.0 <= control.steer <= 1.0
        assert 0.0 <= control.throttle <= 1.0
        assert 0.0 <= control.brake <= 1.0
        self.control = control

    def advance(self, dt: float) -> None:
        steer = -self.control.steer * self.params.max_steer_rad
        accel = 4.0 * self.control.throttle - 8.0 * self.control.brake - 0.001 * self.plant.s.vx ** 2
        self.plant.step(steer, accel, dt)


def exercise_episode(controller: str, hazard_type: str) -> None:
    # Given: native map/agent, actual lateral controller, and nonlinear local physics.
    world = TickWorld()
    vehicle = PlantVehicle(world)
    if hazard_type:
        hazard = Actor(world, hazard_type, 24.0)
        if hazard_type.startswith('walker'):
            hazard.x = 26.0
            hazard.bounding_box = carla.BoundingBox(carla.Location(), carla.Vector3D(0.3, 0.3, 0.9))
        world.hazard = hazard
    bridge = BehaviorBridge(vehicle, BehaviorConfig(speed_mps=5.0, route_length_m=40.0))
    with TemporaryDirectory(prefix='av-stack-autonomy-') as temporary:
        output = Path(temporary)
        settings = EpisodeSettings(controller=controller, duration=25.0, dt=0.05, out=output)
        # When: the production episode runs to destination with logging enabled.
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_episode(vehicle, bridge, settings)
        records = [json.loads(line) for line in (output / 'behavior.jsonl').read_text().splitlines()]
        # Then: arrival stops inside the documented radius, controlled by MPC/LQR only.
        assert summary['route_completed'], summary
        assert records[-1]['goal_distance_m'] <= 3.0
        assert vehicle.plant.s.vx < 0.3
        assert max(abs(vehicle.plant.s.y + 1.75), summary['max_abs_e_y_m']) < 0.1
        assert {record['steer_source'] for record in records} == {controller}
        assert records[-1]['throttle'] == 0.0
        assert records[-1]['brake'] > 0.0
        if hazard_type:
            held = [record for record in records if record['t'] < 2.0]
            released = [record for record in records if 2.0 <= record['t'] < 4.0]
            assert held and all(record['stop'] for record in held)
            assert all(record['throttle'] == 0.0 and record['brake'] > 0.0 for record in held)
            assert any(record['throttle'] > 0.0 and not record['stop'] for record in released)
        print('{} {}: complete at {:.2f}s, goal {:.3f}m, speed {:.3f}m/s'.format(
            controller, hazard_type or 'clear', records[-1]['t'],
            records[-1]['goal_distance_m'], vehicle.plant.s.vx))


def test_lqr_clear_route_arrives_stopped() -> None:
    exercise_episode('lqr', '')


def test_mpc_clear_route_arrives_stopped() -> None:
    exercise_episode('mpc', '')


def test_lqr_red_light_hold_then_release() -> None:
    exercise_episode('lqr', 'traffic.traffic_light')


def test_mpc_red_light_hold_then_release() -> None:
    exercise_episode('mpc', 'traffic.traffic_light')


def test_lqr_pedestrian_hold_then_release() -> None:
    exercise_episode('lqr', 'walker.pedestrian.0001')


def test_mpc_pedestrian_hold_then_release() -> None:
    exercise_episode('mpc', 'walker.pedestrian.0001')


def test_cli_help_without_carla_server() -> None:
    # Given: the real autonomy entrypoint in the CARLA environment.
    root = Path(__file__).resolve().parents[1]
    # When: requesting command-line help in a fresh process.
    result = subprocess.run([sys.executable, str(root / 'run_autonomy.py'), '--help'],
                            capture_output=True, text=True, check=True, timeout=30)
    # Then: CLI options are exposed without a simulator connection.
    assert '--controller' in result.stdout
    assert '--behavior' in result.stdout


if __name__ == '__main__':
    checks = [value for name, value in list(globals().items())
              if name.startswith('test_') and callable(value)]
    for check in checks:
        check()
        print('PASS ' + check.__name__)
    print('{} closed-loop/CLI checks passed'.format(len(checks)))
