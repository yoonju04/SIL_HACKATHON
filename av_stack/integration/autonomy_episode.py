"""Behavior-mode rollout and reference-versioned evidence logging."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Protocol

import carla

from integration.behavior_bridge import BehaviorBridge
from integration.behavior_control import BehaviorSpeedControl, update_reference
from integration.carla_io import follow_camera, read_state, to_carla_steer, vehicle_geometry
from integration.episode import EpisodeLog, LaneMonitor, StepRow, build_controller, step_controller
from integration.speed import speed_reference
from integration.fault_injection import FaultConfig
from integration.supervised_lateral import SupervisedLateral
from integration.unprotected_lateral import UnprotectedMPC


class EpisodeObserver(Protocol):
    def on_tick(self, simulation_time: float) -> None: ...


@dataclass(frozen=True)
class EpisodeSettings:
    controller: str
    duration: float
    dt: float
    out: Path
    follow: bool = False
    plot: bool = False
    observer: EpisodeObserver | None = None
    fault: FaultConfig = FaultConfig()


def run_episode(vehicle: carla.Vehicle, bridge: BehaviorBridge, settings: EpisodeSettings) -> dict:
    world = vehicle.get_world()
    cfg = settings
    cfg.out.mkdir(parents=True, exist_ok=True)
    geom = vehicle_geometry(vehicle)
    traj = bridge.trajectory
    supervised = SupervisedLateral(geom, cfg.dt, traj, cfg.fault) if cfg.controller == 'supervised' else None
    unprotected = UnprotectedMPC(geom, cfg.dt, traj, cfg.fault) if cfg.controller == 'mpc_unprotected' else None
    managed = supervised if supervised is not None else unprotected
    ctrl = managed if managed is not None else build_controller(cfg.controller, geom, cfg.dt, traj)
    speed_control = BehaviorSpeedControl(cfg.dt)
    lane = LaneMonitor(traj.lane_x, traj.lane_y)
    log = EpisodeLog()
    revision = bridge.revision
    traj.save(str(cfg.out / f"trajectory_{revision:04d}.json"))
    reference_key = None
    completed = False
    travelled = 0.0
    previous_position = None
    last_reason = None
    t = 0.0
    last_supervisor_state = 'mpc'
    with (cfg.out / "behavior.jsonl").open("w", encoding="utf-8") as behavior_log, \
            (cfg.out / "supervisor.jsonl").open("w", encoding="utf-8") as supervisor_log:
        while t < cfg.duration:
            if cfg.observer is not None:
                cfg.observer.on_tick(t)
            x, y, yaw, vx, vy, yaw_rate = read_state(vehicle)
            decision = bridge.step()
            if decision.route_revision != revision:
                traj = bridge.trajectory
                revision = decision.route_revision
                lane = LaneMonitor(traj.lane_x, traj.lane_y)
                traj.save(str(cfg.out / f"trajectory_{revision:04d}.json"))
            key = (revision, decision.speed_mps)
            if key != reference_key:
                if managed is not None:
                    managed.set_reference(traj, decision.speed_mps)
                else:
                    update_reference(ctrl, traj, decision.speed_mps)
                reference_key = key
            res = (managed.step(t, x, y, yaw, vx, vy, yaw_rate) if managed is not None
                   else step_controller(cfg.controller, ctrl, t, x, y, yaw, vx, vy, yaw_rate))
            goal_distance = math.hypot(x - traj.x[-1], y - traj.y[-1])
            arrival_zone = res.s >= traj.length - 3.0 and goal_distance <= 3.0
            stop = decision.emergency_stop or arrival_zone or (supervised is not None and supervised.stop_required)
            v_ref = 0.0 if stop else min(speed_reference(traj, res.s, vx), decision.speed_mps)
            pedals = speed_control.command(vx, v_ref, stop)
            steer = res.steer if res.steer is not None else (log.rows[-1].steer_rad if log.rows else 0.0)
            vehicle.apply_control(carla.VehicleControl(
                throttle=pedals.throttle, brake=pedals.brake,
                steer=to_carla_steer(steer, geom["max_steer_rad"])))
            reason = "destination" if arrival_zone else decision.reason
            if supervised is not None and supervised.stop_required:
                reason = 'supervisor_safe_stop'
            if reason != last_reason:
                print(f"[behavior] t={t:.2f}s reason={reason} target={v_ref:.2f}m/s stop={stop}")
                last_reason = reason
            record = asdict(decision)
            record.update(t=t, reason=reason, stop=stop, target_speed_mps=v_ref,
                          throttle=pedals.throttle, brake=pedals.brake,
                          steer_source=managed.active if managed is not None else cfg.controller, steer_rad=steer,
                          goal_distance_m=goal_distance)
            if supervised is not None:
                supervision = supervised.record()
                record.update(supervision)
                if supervised.active != last_supervisor_state:
                    supervisor_log.write(json.dumps(dict(t=t, **supervision), allow_nan=False) + '\n')
                    supervisor_log.flush()
                    last_supervisor_state = supervised.active
            if unprotected is not None:
                record.update(unprotected.record())
            behavior_log.write(json.dumps(record) + "\n")
            log.add(StepRow(t=t, x=x, y=y, yaw=yaw, vx=vx, vy=vy, yaw_rate=yaw_rate,
                            s=res.s, v_ref=v_ref, e_y=res.e_y, e_psi=res.e_psi,
                            e_y_lane=lane.lateral_offset(x, y), kappa=res.kappa,
                            steer_rad=steer, ok=res.ok, status=res.status,
                            solve_ms=res.solve_ms, qp_iterations=res.qp_iterations))
            if previous_position is not None:
                travelled += math.hypot(x - previous_position[0], y - previous_position[1])
            previous_position = (x, y)
            if arrival_zone and abs(vx) < 0.3:
                completed = True
                break
            if cfg.follow:
                follow_camera(world.get_spectator(), vehicle, carla)
            world.tick()
            t = round(t + cfg.dt, 6)

    summary = log.summary(cfg.dt)
    summary.update(route_completed=completed, distance_m=travelled, route_revisions=revision,
                   completion_rule="within 3 m of reference end and |vx| < 0.3 m/s",
                   termination="destination" if completed else "time_limit")
    if supervised is not None:
        summary['supervisor'] = supervised.record()
    if unprotected is not None:
        summary['unprotected'] = unprotected.record()
    meta = {"simulator": "carla", "town": world.get_map().name, "controller": cfg.controller,
            "agent": "CARLA BehaviorAgent 0.9.15", "perception": "simulator_ground_truth",
            "behavior": bridge.config.behavior, "dt": cfg.dt, "vehicle_params": geom,
            "planner": traj.meta, "lateral_owner": cfg.controller, "longitudinal_owner": "av_stack PI",
            "fault_injection": asdict(cfg.fault)}
    log.save(str(cfg.out), traj, meta, summary)
    if cfg.plot:
        log.plot(str(cfg.out), traj, f"BehaviorAgent + {cfg.controller.upper()} (final reference)")
    print(json.dumps(summary, indent=2))
    return summary
