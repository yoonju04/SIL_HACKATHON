from __future__ import annotations

import json
from pathlib import Path

import carla
import numpy as np

from integration.behavior_agent import RoadOption
from integration.urban_camera import UrbanCamera
from integration.urban_crossings import UrbanCrossings
from integration.urban_lead import UrbanLead
from integration.urban_route import UrbanRoute
from integration.urban_signals import UrbanSignals


class UrbanScenario:
    def __init__(self, ego: carla.Vehicle, route: UrbanRoute, out: Path) -> None:
        self.ego = ego
        self.map_name = ego.get_world().get_map().name
        self.route = route
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.camera = None
        self.lead = None
        self.crossings = None
        self.signals = None
        self.index = 0
        self.turns = []
        locations = [waypoint.transform.location for waypoint, _ in route.plan]
        self.xy = np.array([(p.x, p.y) for p in locations])
        self.stations = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))]
        self.progress = 0.0
        self.turn_indices = [i for i, (_, option) in enumerate(route.plan)
                             if option in (RoadOption.LEFT, RoadOption.RIGHT)
                             and (i == 0 or option != route.plan[i - 1][1])]
        self.turn_ends = {}
        for index in self.turn_indices:
            end = index
            while end + 1 < len(route.plan) and route.plan[end + 1][1] == route.plan[index][1]:
                end += 1
            self.turn_ends[index] = end
        self.stream = (out / "scenario_events.jsonl").open("w", encoding="utf-8")

    def setup(self, traffic_manager: carla.TrafficManager, behavior_agent, record_video: bool = False,
              scripted_events: bool = True) -> None:
        if record_video:
            self.camera = UrbanCamera(self.ego, self.out)
        if scripted_events:
            self.lead = UrbanLead(self.ego, traffic_manager, self.route.plan)
            self.crossings = UrbanCrossings(self.ego.get_world(), self.route.plan, self.stations)
            self.crossings.ego = self.ego
            self.signals = UrbanSignals(self.ego, self.route.plan, self.stations)
        for component in (self.lead, self.crossings, self.signals):
            if component is None:
                continue
            component.log = self.record
            component.reason = lambda: behavior_agent.reason
            component.hazard_id = lambda: behavior_agent.hazard_actor_id
        self.record("scenario_ready", 0.0, route_m=self.route.length_m,
                    left_turns=self.route.left_turns, right_turns=self.route.right_turns,
                    crossing_actors=len(self.crossings.crossings) if self.crossings is not None else 0,
                    signal_station=self.signals.station if self.signals is not None else None,
                    scripted_events=scripted_events)

    def record(self, name: str, t: float, **fields) -> None:
        entry = dict(event=name, t=round(t, 3), progress_m=round(self.progress, 2), **fields)
        self.events.append(entry)
        self.stream.write(json.dumps(entry) + "\n")
        self.stream.flush()
        print("[scenario] " + json.dumps(entry), flush=True)
        if self.camera is not None:
            self.camera.snapshot(f"{len(self.events):03d}_{name}")

    def on_tick(self, t: float) -> None:
        location = self.ego.get_location()
        lo, hi = max(0, self.index - 5), min(len(self.xy), self.index + 35)
        distances = np.sum((self.xy[lo:hi] - (location.x, location.y))**2, axis=1)
        self.index = lo + int(np.argmin(distances))
        self.progress = max(self.progress, float(self.stations[self.index]))
        for index in self.turn_indices:
            if index not in self.turns and self.index > self.turn_ends[index] + 2:
                self.turns.append(index)
                self.record("junction_turn_traversed", t, direction=str(self.route.plan[index][1]))
        for component in (self.lead, self.crossings, self.signals):
            if component is not None:
                component.tick(t, self.progress)
        if self.camera is not None:
            self.camera.caption = f"t={t:6.1f}s  route={self.progress:.0f}/{self.route.length_m:.0f}m  speed={self.ego.get_velocity().length()*3.6:.1f}km/h"
        if round(t * 20) % 200 == 0:
            self.record("progress", t, speed_mps=self.ego.get_velocity().length(),
                        lead_phase=self.lead.phase if self.lead is not None else None,
                        lead_speed_mps=self.lead.actor.get_velocity().length() if self.lead is not None else None,
                        lead_gap_m=self.lead.actor.get_location().distance(self.ego.get_location()) if self.lead is not None else None)

    def save(self, traffic_metadata: dict) -> None:
        observed = {entry["event"] for entry in self.events}
        checks = dict(left_turn=any(self.route.plan[i][1] == RoadOption.LEFT for i in self.turns),
                      right_turn=any(self.route.plan[i][1] == RoadOption.RIGHT for i in self.turns),
                      crossing_started="pedestrian_crossing_started" in observed,
                      crossing_wait="ego_waited_for_crossing" in observed,
                      red_light_wait="ego_waited_at_red" in observed,
                      lead_vehicle_wait="ego_stopped_behind_lead" in observed,
                      lead_resumed="lead_resumed" in observed)
        document = dict(map=self.map_name, route_m=self.route.length_m,
                        ego_actor_id=self.ego.id,
                        route_progress_m=self.progress, route_spawn=self.route.spawn_index,
                        planned_left_turns=self.route.left_turns, planned_right_turns=self.route.right_turns,
                        planned_signals=self.route.signal_count, traffic=traffic_metadata,
                        checks=checks, all_events_observed=all(checks.values()),
                        video_frames=self.camera.frames if self.camera is not None else 0,
                        controlled_environment_events=self.lead is not None, events=self.events)
        with (self.out / "scenario.json").open("w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)

    def close(self) -> None:
        for component in (self.camera, self.lead, self.crossings, self.signals):
            if component is not None:
                component.close()
        self.stream.close()
