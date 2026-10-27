from __future__ import annotations

from test_autonomy_episode import Actor, PlantVehicle, TickWorld
from integration.urban_signals import UrbanSignals


def signal_fixture(hazard_id: int) -> UrbanSignals:
    world = TickWorld()
    signal = UrbanSignals.__new__(UrbanSignals)
    signal.ego = PlantVehicle(world)
    signal.light = Actor(world, "traffic.traffic_light", 50.0)
    signal.station = 50.0
    signal.phase = "red"
    signal.started = 0.0
    signal.wait_started = None
    signal.reason = lambda: "red_light"
    signal.hazard_id = lambda: hazard_id
    signal.log = lambda *args, **kwargs: None
    return signal


def test_red_stop_requires_the_selected_signal_actor() -> None:
    signal = signal_fixture(9999)
    signal.tick(1.0, 40.0)
    assert signal.wait_started is None


def test_matching_red_stop_is_recorded() -> None:
    signal = signal_fixture(2)
    signal.tick(1.0, 40.0)
    assert signal.wait_started == 1.0


def test_pedestrian_stop_is_not_a_red_light_stop() -> None:
    signal = signal_fixture(2)
    signal.reason = lambda: "pedestrian"
    signal.tick(1.0, 40.0)
    assert signal.wait_started is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
