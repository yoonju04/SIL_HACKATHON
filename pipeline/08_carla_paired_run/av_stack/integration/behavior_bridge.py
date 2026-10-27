"""Translate official behavior routes and speed decisions into av_stack references."""
from __future__ import annotations

from dataclasses import dataclass
import math

import carla

from integration.behavior_agent import PlanningAgent, RoadOption
from planner import PlannerConfig, ReferencePlanner, Trajectory


@dataclass(frozen=True)  # Python 3.8 CARLA environment does not support dataclass slots.
class BehaviorConfig:
    speed_mps: float = 8.0
    dt: float = 0.05
    behavior: str = "cautious"
    route_length_m: float = 500.0
    destination: carla.Location | None = None
    a_lat: float = 3.0
    mu: float = 0.9

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.speed_mps, self.dt, self.route_length_m, self.a_lat, self.mu)):
            raise ValueError("Behavior speeds, timestep, route length, and physical limits must be positive and finite.")
        if self.route_length_m < 10.0 or self.dt > 0.1:
            raise ValueError("Use route length >= 10 m and 0 < dt <= 0.1 s.")
        if self.behavior not in ("cautious", "normal", "aggressive"):
            raise ValueError("Unknown CARLA behavior profile.")


@dataclass(frozen=True)
class BehaviorDecision:
    speed_mps: float
    emergency_stop: bool
    reason: str
    route_revision: int
    hazard_actor_id: int | None = None


class BehaviorBridge:
    """Maintain a map-anchored reference; only upstream route replacement rebuilds it."""

    def __init__(self, vehicle: carla.Vehicle, config: BehaviorConfig) -> None:
        self.config = config
        self.agent = PlanningAgent(vehicle, config.dt, config.behavior)
        self.revision = 0
        world_map = vehicle.get_world().get_map()
        if config.destination is not None:
            self.agent.set_destination(config.destination)
        else:
            waypoint = world_map.get_waypoint(vehicle.get_location())
            route = [(waypoint, RoadOption.LANEFOLLOW)]
            for _ in range(int(config.route_length_m / 2.0)):
                following = waypoint.next(2.0)
                if not following:
                    break
                waypoint = following[0]
                route.append((waypoint, RoadOption.LANEFOLLOW))
            self.agent.set_global_plan(route)
        self.trajectory = self._reference()

    def _reference(self) -> Trajectory:
        local = self.agent.get_local_planner()
        locations = [waypoint.transform.location for waypoint, _ in local.route]
        cfg = self.config
        planner = ReferencePlanner(PlannerConfig(
            v_max=cfg.speed_mps, a_lat_max=cfg.a_lat, mu=cfg.mu,
            v_start=0.0 if self.revision == 0 else cfg.speed_mps, v_end=0.0))
        trajectory = planner.plan([p.x for p in locations], [-p.y for p in locations])
        self.revision = local.revision
        trajectory.meta.update(source="CARLA BehaviorAgent 0.9.15", route_revision=self.revision)
        return trajectory

    def step(self) -> BehaviorDecision:
        """Run behavior only; return a speed cap and stop intent, never steering."""
        local = self.agent.get_local_planner()
        if self.agent.done():
            return BehaviorDecision(0.0, True, "destination", self.revision)
        control = self.agent.run_step()
        if local.revision != self.revision:
            self.trajectory = self._reference()
        stopping = control.brake > 0.0
        reason = "destination" if self.agent.done() else self.agent.reason
        speed = 0.0 if stopping else min(local.capture.speed_mps, self.config.speed_mps)
        return BehaviorDecision(speed, stopping, reason, self.revision, self.agent.hazard_actor_id)
