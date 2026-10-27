# Optional Bench2Drive TCP trajectory predictor

Source: https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/tcp/admlp

Pinned source commit: `8a08b07883f10b7d83f6bf5dd475bda91a91c50a`.
Preserved source license: `third_party/bench2drive_tcp/LICENSE`, CC BY-NC-ND 4.0.
The repository source license and the Hugging Face model-card license are different; do not assume the source is MIT or Apache licensed.

Checkpoint: https://huggingface.co/rethinklab/Bench2DriveZoo/blob/140aea58c02185e787c3df39d7ad79e910967b8b/tcp_b2d.ckpt

Local path: `models/tcp_b2d.ckpt` (319450037 bytes).
SHA256: `e6573ff1f8ea910b9a53eddfb68f69cac469bf5bfa253a516578f6126110b4fe`.
The model card declares Apache-2.0 for the model repository.

The upstream source is not modified (NoDerivatives). `integration/tcp_predictor.py` replaces `TCP.resnet.resnet34` at import time so ResNet34 skips the unrelated ImageNet download; all 288 checkpoint entries are loaded strictly afterwards.

Not redistributed: `third_party/bench2drive_tcp/` and `models/tcp_b2d.ckpt` are git-ignored. `tools/fetch_third_party.ps1 -WithTCP` clones the pinned commit locally.

The adapter reproduces the official three-camera JPEG/crop/resize/normalization and trajectory branch. It never calls the model's full control forward, PID, or action policy. Four predicted waypoints are 0.5 seconds apart; desired speed is mean adjacent-point distance divided by 0.5, matching the original trajectory policy's speed calculation. Auxiliary predicted observed speed is not used as a target. Our MPC/LQR and longitudinal PI remain actuator owners. An additional LiDAR supplies our existing front obstacle speed cap, not the TCP network. Benchmark route metadata provides navigation; scenario events and actor ground truth are not model perception inputs.

This is a comparison option, not a demonstrated fix. The official `TCP/TCP-only-traj.json` record for route 26956 reports completion 100%, one vehicle collision, driving score 60 with the original trajectory controller. That is not a result for our MPC adapter. The installed adapter's synthetic sensor preprocessing plus GPU forward measured about 81-88 ms per prediction on this RTX 3090; this does not establish real-time CARLA throughput or driving quality.

`--planner transfuserpp` remains the default. `--planner tcp` selects this alternative. Both support `--record-front`, writing sensor video and frame IDs without changing inference inputs. No live CARLA evaluation was launched by Codex for this change because the user chose to execute driving themselves.
