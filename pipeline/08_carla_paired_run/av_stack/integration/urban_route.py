from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable

import carla
import networkx as nx

from integration.behavior_agent import RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner


@dataclass(frozen=True)
class UrbanRoute:
    spawn_index: int
    plan: list[tuple[carla.Waypoint, RoadOption]]
    length_m: float
    left_turns: int
    right_turns: int
    signal_count: int
    light_ids: tuple[int, ...] = ()


def count_turns(options: Iterable[RoadOption]) -> tuple[int, int]:
    runs = []
    for option in options:
        if not runs or option != runs[-1]:
            runs.append(option)
    return runs.count(RoadOption.LEFT), runs.count(RoadOption.RIGHT)


def quick_fault_route(route: UrbanRoute) -> UrbanRoute:
    stations = [0.0]
    for (a, _), (b, _) in zip(route.plan, route.plan[1:]):
        stations.append(stations[-1] + a.transform.location.distance(b.transform.location))
    start_turn = next(i for i, (_, option) in enumerate(route.plan)
                      if option in (RoadOption.LEFT, RoadOption.RIGHT) and stations[i] >= 110)
    end_turn = start_turn
    while end_turn + 1 < len(route.plan) and route.plan[end_turn+1][1] == route.plan[start_turn][1]:
        end_turn += 1
    start = min(range(start_turn+1), key=lambda i: abs(stations[i] - (stations[start_turn]-110)))
    end = min(range(end_turn, len(route.plan)), key=lambda i: abs(stations[i] - (stations[end_turn]+70)))
    plan = route.plan[start:end+1]
    left, right = count_turns(option for _, option in plan)
    return UrbanRoute(route.spawn_index, plan, _length(plan), left, right, 0)


def _length(plan: list[tuple[carla.Waypoint, RoadOption]]) -> float:
    return sum(a.transform.location.distance(b.transform.location)
               for (a, _), (b, _) in zip(plan, plan[1:]))


def _settled_end(plan: list[tuple[carla.Waypoint, RoadOption]]) -> bool:
    return len(plan) >= 16 and all(not wp.is_junction for wp, _ in plan[-16:])


def _signal_ids(plan: list[tuple[carla.Waypoint, RoadOption]],
                stops: list[tuple[int, carla.Waypoint]]) -> tuple[int, ...]:
    lanes: dict[tuple[int, int], list[carla.Waypoint]] = {}
    for waypoint, _ in plan:
        lanes.setdefault((waypoint.road_id, waypoint.lane_id), []).append(waypoint)
    return tuple(sorted({identifier for identifier, stop in stops
                         if any(abs(stop.s - wp.s) < 5.0
                                for wp in lanes.get((stop.road_id, stop.lane_id), []))}))


def choose_urban_route(world_map: carla.Map, lights: list[carla.TrafficLight],
                       seed: int = 17) -> UrbanRoute:
    """Select a continuous native-map route with repeated turns and signal stops."""
    planner = GlobalRoutePlanner(world_map, 2.0)
    # Lateral graph links skip lane samples; exclude them from a spline reference.
    graph = planner._graph
    lane_changes = (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT)
    graph.remove_edges_from([(a, b) for a, b, data in graph.edges(data=True)
                             if data['type'] in lane_changes])
    stops = [(light.id, wp) for light in lights for wp in light.get_stop_waypoints()]
    spawns = world_map.get_spawn_points()
    indices = list(range(len(spawns)))
    Random(seed).shuffle(indices)
    candidates = []
    for start in indices[:30]:
        for end in indices[:45]:
            if start == end:
                continue
            try:
                plan = planner.trace_route(spawns[start].location, spawns[end].location)
            except nx.NetworkXNoPath:
                continue
            length = _length(plan)
            left, right = count_turns(option for _, option in plan)
            signals = _signal_ids(plan, stops)
            if (700.0 <= length <= 1600.0 and min(left, right) >= 2
                    and len(signals) >= 2 and _settled_end(plan)):
                return UrbanRoute(start, plan, length, left, right, len(signals), signals)
            if 250.0 < length < 1050.0:
                candidates.append((start, end, plan))
    for start, end, first in candidates:
        for destination in indices[:45]:
            try:
                second = planner.trace_route(first[-1][0].transform.location,
                                             spawns[destination].location)
            except nx.NetworkXNoPath:
                continue
            plan = first + second[1:]
            length = _length(plan)
            left, right = count_turns(option for _, option in plan)
            signals = _signal_ids(plan, stops)
            first_ids = {wp.id for wp, _ in first[:-20]}
            overlap = sum(wp.id in first_ids for wp, _ in second)
            if (700.0 <= length <= 1600.0 and min(left, right) >= 2
                    and len(signals) >= 2 and overlap <= 10 and _settled_end(plan)):
                return UrbanRoute(start, plan, length, left, right, len(signals), signals)
    raise RuntimeError('No continuous urban route with two left/right turns and signals was found.')
