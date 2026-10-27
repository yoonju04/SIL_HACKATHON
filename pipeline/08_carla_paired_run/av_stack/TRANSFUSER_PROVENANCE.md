# TransFuser++ sensor prediction integration

Upstream: https://github.com/autonomousvision/carla_garage/tree/48e6a6dbcb133220333bd65e242fbc2b492bb15b
Branch: leaderboard_2. Vendored commit: 48e6a6dbcb133220333bd65e242fbc2b492bb15b.
The repository includes Bench2Drive 0.0.3 and its matching scenario_runner.
Published routes: Bench2Drive/leaderboard/data/bench2drive220.xml (220 routes).
Route XML and scenario implementations are not rewritten by this integration.

Model archive: https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip
Use models/transfuserpp/pretrained_models/all_towns, seed 0 model_0030_0.pth.
The archive is cached in work/pretrained_models.zip. The all_towns model has seen these towns;
this experiment is not a held-out-town generalization claim.
Upstream code is MIT; model weights are CC BY 4.0. Retain upstream license and attribution.
Paper: TransFuser++ / CARLA Garage, https://github.com/autonomousvision/carla_garage.
Our sensor preprocessing/forward adapter derives from upstream team_code/sensor_agent.py.

Local upstream patch: team_code/transfuser.py image encoder initialization uses pretrained=False.
The released complete state_dict is subsequently loaded strictly; this avoids an unnecessary
ImageNet download and does not change the released checkpoint parameters.
No other upstream control policy is reused for actuation: original run_step is bypassed,
PID instances are deleted from loaded networks, and control_pid/control_pid_direct raise.
Stop-sign controller, forced creep and upstream throttle/steering are not called.

Ownership: RGB + LiDAR + GNSS + IMU + speedometer + navigation route -> pretrained
predicted spatial checkpoints and speed -> our spline/speed constraints -> own MPC/LQR/PI.
The model predicts desired speed; our PI converts it into throttle/brake.
The benchmark evaluator reads fixed ego geometry and flattens its steering curve for our
steering calibration. The agent receives those intrinsics, not a simulator actor handle.
Scenario/evaluator ground truth remains permitted for spawning actors and scoring only.
The inherited get_metric_info helper is not called by this benchmark evaluation path.

This is an adapted controller experiment, not the original published TF++ policy/score.
One selected route is a benchmark subset, not the full 220-route aggregate result.
The supervisor monitors MPC computation/output/tracking; it does not supervise neural-network
perception correctness, blocked inference, the simulator process or all driving hazards.
The learned path defines the tracking error; official criteria must judge road departure/collisions.

Runtime: isolated .venv-tfpp (Python 3.10, PyTorch 2.5.0 CUDA 12.1, CARLA 0.9.15).
requirements-tfpp-lock.txt records installed packages. Original conda CARLA environment is unchanged.

Spectator-only patch: Bench2Drive scenario_manager.py reuses integration.carla_io.follow_camera (8m behind, 4m up, pitch -15 degrees) instead of the fixed 70m overhead view. Model sensors and evaluation criteria are unchanged.

Post-collision revision: near-ego reference extension now fits a local quadratic in arc length
instead of a straight tangent. MPC uses 20 x 0.1s preview with measured-speed horizon sampling.
The reference no longer resets a spawn acceleration profile every neural prediction.
A separate raw-LiDAR front-corridor stopping request caps the planner target speed and requests
full stopping through our own speed controller. It is a limited sensor safety aid, not a complete
object predictor or collision guarantee. Original TF++ controllers remain blocked.
Official route ActorFlow sink deletion and background traffic clearing are unchanged.
