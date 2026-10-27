"""Run one published Bench2Drive route on an already running CARLA server."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
GARAGE = ROOT / "third_party" / "carla_garage"
ROUTES = GARAGE / "Bench2Drive" / "leaderboard" / "data" / "bench2drive220.xml"
NAMED_ROUTE_FILES = {"bench2drive": ROUTES, "town10_long": ROOT / "scenarios" / "town10_long.xml",
                     "town04_highway": ROOT / "scenarios" / "town04_highway.xml"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List the original 220 route IDs without connecting to CARLA")
    parser.add_argument("--route-id", help="One ID from --list; this is a subset evaluation")
    parser.add_argument("--routes", default="bench2drive",
                        help="bench2drive (220 short routes), town10_long (Town10, 343 m, signals), town04_highway (2.2 km, no signals) or a route XML path")
    parser.add_argument("--max-speed", type=float, default=40.0, help="PDM-Lite speed cap [km/h]")
    parser.add_argument("--no-comfort", action="store_true", help="keep PDM-Lite's original bang-bang pedals")
    parser.add_argument("--no-view", action="store_true", help="do not open the chase-camera window")
    parser.add_argument("--green-lights", action="store_true", help="switch the light ahead of the ego to green (its junction's others red)")
    parser.add_argument("--monitor", type=float, default=0.0, metavar="SECONDS",
                        help="also print a supervisor status line every SECONDS (default: only takeover banners)")
    parser.add_argument("--controller", choices=("mpc", "lqr", "supervised", "mpc_unprotected", "original"), default="mpc",
                        help="original = unchanged TF++ PID baseline (comparison only, no av_stack control)")
    parser.add_argument("--planner", choices=("transfuserpp", "tcp", "pdm"), default="transfuserpp",
                        help="pdm = PDM-Lite privileged expert (route/speed/hazards) + av_stack steering")
    parser.add_argument("--mpc-extension", choices=("linear", "quadratic"), default="linear",
                        help="how the MPC reference is extended back past the vehicle (LQR always quadratic)")
    parser.add_argument("--speed-policy", choices=("follow", "model", "guarded"), default="follow",
                        help="follow = TF++ speed + LiDAR gap keeping; model = TF++ speed as is; guarded = + curvature cap")
    parser.add_argument("--longitudinal", choices=("tfpp", "pi"), default="tfpp",
                        help="tfpp = TF++ throttle/brake law (default); pi = old av_stack speed PI")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--record-front", action="store_true", help="Save sensor video and frame IDs for failure diagnosis")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/bench2drive_subset")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8050)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout", type=float, default=120.0, help="Wall-clock RPC/watchdog timeout, not route duration")
    parser.add_argument("--device", choices=("cuda", "cuda:0"), default="cuda")
    parser.add_argument("--mpc-fault", choices=("none", "solver_fail", "nan", "stale", "exception", "delay", "bias", "drift",
                                                "intermittent_bias", "saturate", "output_delay"), default="none")
    parser.add_argument("--fault-magnitude", type=float, default=0.05,
                        help="bias [rad], drift rate [rad/s], saturate side (sign) or output_delay ticks")
    parser.add_argument("--fault-on", type=float, default=0.2, help="intermittent_bias: seconds with the bias on")
    parser.add_argument("--fault-off", type=float, default=0.2, help="intermittent_bias: seconds with the bias off")
    parser.add_argument("--fault-start", type=float, default=20.0)
    parser.add_argument("--fault-duration", type=float, default=5.0)
    parser.add_argument("--stop-after-fault", type=float, help="end the route this many seconds after fault start (demo videos)")
    args = parser.parse_args()
    args.routes = Path(NAMED_ROUTE_FILES.get(args.routes, args.routes)).resolve()
    if not args.routes.is_file():
        parser.error(f"Route file not found: {args.routes}")
    routes = ET.parse(args.routes).getroot().findall("route")
    if args.list:
        print("Published Bench2Drive 0.0.3 routes; each selected route is a subset evaluation.")
        for route in routes:
            scenarios = ", ".join(s.get("type", "") for s in route.findall("./scenarios/scenario"))
            print(f"{route.get('id'):>5}  {route.get('town'):<8}  {scenarios}")
        return 0
    if args.route_id is None and len(routes) == 1:
        args.route_id = routes[0].get("id")
    selected = next((route for route in routes if route.get("id") == args.route_id), None)
    if selected is None:
        parser.error("Provide a valid --route-id from --list")
    if args.model is None and args.planner != "pdm":
        args.model = ROOT / ("models/tcp_b2d.ckpt" if args.planner == "tcp" else "models/transfuserpp/pretrained_models/all_towns")
    if args.planner == "pdm":
        if args.controller == "original":
            parser.error("--controller original is only available for --planner transfuserpp")
    elif args.planner == "tcp":
        if not args.model.is_file():
            parser.error("TCP --model must be a released checkpoint file")
    elif not args.model.is_dir() or not (args.model / "config.json").is_file():
        parser.error("TF++ --model must contain config.json")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Timeout must be finite and positive")
    from integration.fault_injection import FaultConfig
    try:
        FaultConfig(args.mpc_fault, args.fault_start, args.fault_duration, args.fault_magnitude, args.fault_on, args.fault_off)
    except ValueError as error:
        parser.error(str(error))
    if args.mpc_fault != "none" and args.controller not in ("supervised", "mpc_unprotected"):
        parser.error("Fault injection requires supervised or mpc_unprotected")
    if args.controller == "original" and args.planner != "transfuserpp":
        parser.error("--controller original is only available for --planner transfuserpp")
    if args.planner == "transfuserpp" and not list(args.model.glob("*.pth")):
        parser.error("--model must contain released .pth weights")
    args.out = args.out.resolve()
    if (args.out / "results.json").exists():
        parser.error("Output already contains results.json; choose a fresh --out directory")
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(model=None if args.model is None else str(args.model.resolve()), planner=args.planner, record_front=args.record_front, controller=args.controller, longitudinal=args.longitudinal, mpc_extension=args.mpc_extension, speed_policy=args.speed_policy, mpc_fault=args.mpc_fault,
                  fault_start=args.fault_start, fault_duration=args.fault_duration, fault_magnitude=args.fault_magnitude, fault_on_s=args.fault_on, fault_off_s=args.fault_off, stop_after_fault=args.stop_after_fault,
                  max_speed_kmh=args.max_speed, comfort=not args.no_comfort, monitor_period=args.monitor, view=not args.no_view, green_lights=args.green_lights,
                  out=str(args.out), device=args.device, dt=0.05)
    agent_config = args.out / "agent_config.json"
    agent_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = dict(benchmark="Bench2Drive 0.0.3", evaluation_scope="single published route subset",
                    full_benchmark_score=False, route_id=args.route_id, town=selected.get("town"),
                    source_commit="48e6a6dbcb133220333bd65e242fbc2b492bb15b",
                    routes_file=str(args.routes), routes_sha256=hashlib.sha256(args.routes.read_bytes()).hexdigest(),
                    scenario_types=[s.get("type") for s in selected.findall("./scenarios/scenario")],
                    controller=args.controller, planner=args.planner, seed=args.seed)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    from integration.tfpp_benchmark import evaluate
    return evaluate(args, agent_config)


if __name__ == "__main__":
    raise SystemExit(main())
