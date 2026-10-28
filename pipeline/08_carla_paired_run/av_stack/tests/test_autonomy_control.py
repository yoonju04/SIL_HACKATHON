"""Server-free actuator ownership and stop/resume regression tests."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_stop_clears_acceleration_integrator() -> None:
    from integration.behavior_control import BehaviorSpeedControl
    controller = BehaviorSpeedControl(0.05)
    for _ in range(100):
        controller.command(0.0, 8.0, False)
    pedals = controller.command(8.0, 8.0, True)
    assert pedals.throttle == 0.0 and pedals.brake == 1.0
    pedals = controller.command(8.0, 8.0, False)
    assert pedals.throttle == 0.0


def test_zero_target_holds_brake_at_standstill() -> None:
    from integration.behavior_control import BehaviorSpeedControl
    pedals = BehaviorSpeedControl(0.05).command(0.0, 0.0, False)
    assert pedals.throttle == 0.0 and pedals.brake == 1.0


def test_clear_road_releases_brake() -> None:
    from integration.behavior_control import BehaviorSpeedControl
    controller = BehaviorSpeedControl(0.05)
    controller.command(0.0, 0.0, True)
    pedals = controller.command(0.0, 5.0, False)
    assert pedals.throttle > 0.0 and pedals.brake == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
