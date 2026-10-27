from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import carla
from integration.behavior_agent import RoadOption
from integration.urban_route import UrbanRoute, quick_fault_route


class Waypoint:
    def __init__(self, x: float) -> None:
        self.transform = carla.Transform(carla.Location(x=x))


def test_quick_route_starts_110m_before_first_turn() -> None:
    plan = [(Waypoint(float(i*2)), RoadOption.LEFT if 100 <= i <= 120 else RoadOption.LANEFOLLOW)
            for i in range(250)]
    route = quick_fault_route(UrbanRoute(30, plan, 498, 1, 0, 3))
    assert abs(route.plan[0][0].transform.location.x - 90) < 2.1
    assert route.length_m < 230
    assert route.left_turns == 1 and route.right_turns == 0
    assert route.signal_count == 0


if __name__ == '__main__':
    test_quick_route_starts_110m_before_first_turn()
    print('PASS test_quick_route_starts_110m_before_first_turn')
