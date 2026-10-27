from __future__ import annotations

from typing import Callable

import carla
import numpy as np


class UrbanSignals:
    def __init__(self, ego: carla.Vehicle, plan: list, stations: np.ndarray) -> None:
        self.ego = ego
        self.log: Callable = lambda *args: None
        self.reason: Callable[[], str] = lambda: ""
        self.hazard_id: Callable = lambda: None
        self.light = None
        self.station = 0.0
        self.group = []
        self.original = []
        self.phase = "waiting"
        self.started = 0.0
        self.wait_started = None
        candidates = []
        for light in ego.get_world().get_actors().filter("traffic.traffic_light"):
            for stop in light.get_stop_waypoints():
                indices = [i for i, (wp, _) in enumerate(plan)
                           if wp.road_id == stop.road_id and wp.lane_id == stop.lane_id
                           and abs(wp.s - stop.s) < 4.0]
                for index in indices:
                    if 300.0 < stations[index] < 1000.0:
                        candidates.append((float(stations[index]), light))
        if candidates:
            self.station, self.light = min(candidates, key=lambda item: item[0])
            self.group = self.light.get_group_traffic_lights()
            self.original = [(light, light.get_state(), light.is_frozen()) for light in self.group]

    def tick(self, t: float, progress: float) -> None:
        if self.light is None:
            return
        gap = self.station - progress
        if self.phase == "waiting" and 0.0 < gap < 45.0:
            for light in self.group:
                light.freeze(True)
                light.set_state(carla.TrafficLightState.Red)
            self.phase = "red"
            self.started = t
            self.log("signal_red", t, light_id=self.light.id, station=self.station)
        if self.phase == "red":
            if (self.reason() == "red_light" and self.hazard_id() == self.light.id and 0.0 < gap < 20.0
                    and self.ego.get_velocity().length() < 0.4 and self.wait_started is None):
                self.wait_started = t
                self.log("ego_waited_at_red", t, light_id=self.light.id, gap_m=gap)
            waited = self.wait_started is not None and t - self.wait_started > 7.0
            if waited or t - self.started > 35.0:
                self.light.set_state(carla.TrafficLightState.Green)
                self.phase = "green"
                self.started = t
                self.log("signal_green", t, ego_wait_observed=self.wait_started is not None)
        if self.phase == "green" and t - self.started > 15.0:
            self.close()
            self.phase = "finished"
            self.log("signal_cycle_restored", t)

    def close(self) -> None:
        for light, state, frozen in self.original:
            light.set_state(state)
            light.freeze(frozen)
