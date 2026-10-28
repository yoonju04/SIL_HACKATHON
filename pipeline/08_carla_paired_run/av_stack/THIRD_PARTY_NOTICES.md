# Third-party notices

## CARLA 0.9.15 agents (MIT): vendored source

- Repository: https://github.com/carla-simulator/carla
- Commit: `d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18`
- Files and original license: `third_party/carla_0915/`.
- Local modification: `PATCHES.md` records the exact-zero-angle hazard fix.
- Put on the import path by `integration/tfpp_bootstrap.py` for the CARLA Garage / PDM-Lite agents.
  (The earlier BehaviorAgent bridge that replaced the upstream PID endpoint was removed.)

## CARLA Garage / TransFuser++ and Bench2Drive (MIT code, CC BY 4.0 weights): fetched, not vendored

- Repository: https://github.com/autonomousvision/carla_garage (branch leaderboard_2)
- Commit: `48e6a6dbcb133220333bd65e242fbc2b492bb15b` (includes Bench2Drive 0.0.3 + scenario_runner)
- Not committed here (size). `tools/fetch_third_party.ps1` clones the pinned commit and applies
  `third_party/patches/carla_garage.patch` (two local changes: image encoder `pretrained=False`,
  follow-camera spectator). Weights: CC BY 4.0, fetched with `-WithModels`.
- Details: `TRANSFUSER_PROVENANCE.md`.

## Bench2DriveZoo TCP (CC BY-NC-ND 4.0): optional, never redistributed

- Repository: https://github.com/Thinklab-SJTU/Bench2DriveZoo (branch tcp/admlp)
- Commit: `8a08b07883f10b7d83f6bf5dd475bda91a91c50a`
- NonCommercial-NoDerivatives: the source is neither committed nor modified. It is fetched only
  with `tools/fetch_third_party.ps1 -WithTCP`; `integration/tcp_predictor.py` imports it unchanged.
- Details: `TCP_PROVENANCE.md`.

## PythonRobotics (MIT): ported code

- **Repository:** https://github.com/AtsushiSakai/PythonRobotics (commit cdd0cc8)
- **Ported files**
  - `planner/cubic_spline.py`, from `PathPlanning/CubicSpline/cubic_spline_planner.py`
  - `planner/quintic.py`, from `PathPlanning/QuinticPolynomialsPlanner/quintic_polynomials_planner.py`

```
The MIT License (MIT)

Copyright (c) 2016 - now Atsushi Sakai and other contributors:
https://github.com/AtsushiSakai/PythonRobotics/contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Autoware Universe (Apache-2.0): algorithm re-implemented in Python

- **Repository:** https://github.com/autowarefoundation/autoware_universe (commit 3b3a5ad)
- **Source package:** `control/autoware_mpc_lateral_controller`
- **What was taken**
  - the model equations, discretisation, cost structure, constraints and default weights. These were re-implemented in Python in `mpc_controller/`.
  - the Butterworth filter coefficients, used in `mpc_controller/filters.py`.
- **Licence:** Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0
- **Changes:** the changes from the original are listed in the module docstrings.

## Apollo (Apache-2.0): algorithm re-implemented in Python

- **Repository:** https://github.com/ApolloAuto/apollo (commit d53aa3d)
- **Source package:** `modules/control/controllers/lat_based_lqr_controller`
- **What was taken:** the model matrices, discretisation, iterative Riccati solution, feed-forward structure and default weights. These were re-implemented in Python in `lqr_controller/`.
- **Licence:** Apache License 2.0
- **Changes:** the changes from the original are listed in the module docstrings.

## Papers

- **R. Rajamani**, *Vehicle Dynamics and Control*, 2nd ed., Springer, 2012, chapters 2 and 3.
- **P. Falcone, F. Borrelli, J. Asgari, H. E. Tseng, D. Hrovat**, "Predictive Active Steering Control for Autonomous Vehicle Systems", *IEEE TCST* 15(3), 2007.
- **M. Werling, J. Ziegler, S. Kammel, S. Thrun**, "Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame", *ICRA* 2010.
- **B. Stellato et al.**, "OSQP: an operator splitting solver for quadratic programs", *Math. Prog. Comp.* 12, 2020.
- **J. M. Snider**, "Automatic Steering Methods for Autonomous Automobile Path Tracking", CMU-RI-TR-09-08, 2009.
- **B. Paden et al.**, "A Survey of Motion Planning and Control Techniques for Self-driving Urban Vehicles", *IEEE T-IV* 1(1), 2016.
