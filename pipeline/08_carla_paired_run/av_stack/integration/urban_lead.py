from __future__ import annotations

from typing import Callable

import carla

from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.local_planner import RoadOption


class UrbanLead:
    def __init__(self, ego: carla.Vehicle, traffic_manager: carla.TrafficManager, plan: list) -> None:
        self.ego = ego
        self.tm = traffic_manager
        self.actor = None
        self.phase = "driving"
        self.started = 0.0
        self.wait_started = None
        self.log: Callable = lambda *args: None
        self.reason: Callable[[], str] = lambda: ""
        self.hazard_id: Callable = lambda: None
        world = ego.get_world()
        blueprint = world.get_blueprint_library().find("vehicle.audi.a2")
        blueprint.set_attribute("role_name", "urban_lead")
        self.path = [waypoint.transform.location for waypoint, _ in plan]
        start = next(i for i, p in enumerate(self.path) if p.distance(self.path[0]) > 35.0)
        pose = plan[start][0].transform
        pose.location.z += 0.6
        self.actor = world.try_spawn_actor(blueprint, pose)
        if self.actor is None:
            raise RuntimeError("The planned lead-vehicle spawn is occupied.")
        self.agent = BehaviorAgent(self.actor, behavior="cautious")
        self.agent._behavior.max_speed = 15.0
        lead_plan = list(plan[start:])
        tail = lead_plan[-1][0]
        for _ in range(60):
            following = tail.next(2.0)
            if not following:
                break
            tail = following[0]
            lead_plan.append((tail, RoadOption.LANEFOLLOW))
        self.agent.set_global_plan(lead_plan)

    def tick(self, t: float, progress: float) -> None:
        gap = self.actor.get_location().distance(self.ego.get_location())
        if self.phase == "driving" and t > 12.0 and progress > 35.0 and gap < 35.0:
            self.phase = "stopped"
            self.started = t
            self.actor.apply_control(carla.VehicleControl(brake=1.0))
            self.log("lead_braking", t, gap_m=gap)
        if self.phase == "stopped":
            map_instance = self.ego.get_world().get_map()
            ego_lane = map_instance.get_waypoint(self.ego.get_location())
            lead_lane = map_instance.get_waypoint(self.actor.get_location())
            same_lane = (ego_lane.road_id, ego_lane.lane_id) == (lead_lane.road_id, lead_lane.lane_id)
            if (self.reason() == "vehicle" and self.hazard_id() == self.actor.id and same_lane and self.ego.get_velocity().length() < 0.4
                    and self.actor.get_velocity().length() < 0.4 and gap < 20.0 and self.wait_started is None):
                self.wait_started = t
                self.log("ego_stopped_behind_lead", t, gap_m=gap)
            waited = self.wait_started is not None and t - self.wait_started > 5.0
            if waited or t - self.started > 35.0:
                self.phase = "resumed"
                self.agent._behavior.max_speed = 22.0
                self.log("lead_resumed", t, ego_wait_observed=self.wait_started is not None)
        if self.phase != "stopped":
            control = carla.VehicleControl(brake=1.0) if self.agent.done() else self.agent.run_step()
            self.actor.apply_control(control)

    def close(self) -> None:
        if self.actor is not None and self.actor.is_alive:
            self.actor.destroy()
